"""Teacher demonstrations for phase D behaviour cloning.

Each sample pairs a client-visible observation (``observe``) with the MPPI
teacher's action for that 10 Hz control step. Only the approach segment is
kept: live episodes above HANDOVER_AGL, where the mod's rule-based flare takes
over. The observation must stay reproducible in the mod from data a client
has (aircraft state, NavInfo runway, NavTelemetry rings and wind), because
phase E rebuilds it in C#; ``OBSERVATION_NAMES`` documents every component.

Data is written in shards. A shard is one seeded batch of episodes flown to the
end, saved as ``shard-NNNNN.npz`` plus a ``.json`` manifest (written last, with
the payload's SHA-256). Finished shards are skipped, so an interrupted run
resumes at the first missing shard and reproduces the same samples.
"""

from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .environment import GATE_DISTANCES, GLIDE_SLOPE, EnvState, runway_coordinates
from .mppi import MPPIConfig, TeacherConfig, TeacherState, _hold, _initialize, plan, ring_tolerance, summarize
from .physics import RAD, default_tuning, wrap180

HANDOVER_AGL = 10.0
OBSERVATION_NAMES = (
    "along/4000", "lateral/100", "height_above_ring_path/50",
    "next_ring_distance/1000", "next_ring_lateral_units", "next_ring_vertical_units", "has_next_ring",
    "(speed-47)/10", "gamma/5", "sin(track-runway)", "cos(track-runway)", "wrap180(heading-runway)/30",
    "pitch/10", "roll/30", "pitch_rate/10", "roll_rate/20", "yaw_rate/10",
    "min(agl,200)/100", "vertical_speed/5", "flap_pos/2", "alpha/10",
    "wind_along_runway/10", "wind_cross_runway/10", "stall_warning",
    "last_pitch", "last_roll", "last_yaw", "last_throttle",
    "latency/0.6",
)
OBSERVATION_SIZE = len(OBSERVATION_NAMES)
STATE_SIZE = OBSERVATION_SIZE - 5          # the delayed part: everything before last action and latency
MAX_LATENCY = 0.6                          # seconds; clients see the state this late at most (interp + rtt/2)
SCHEMA = "mayday-jax-teacher-demos-v1"


def observe(env: EnvState, last_action, latency=0.0):
    """Client-visible observation (heading-invariant, runway frame). See OBSERVATION_NAMES.

    ``latency`` is the client's display delay in seconds (0 for the host); a delayed observation
    is built by combining an older ``observe`` state part with the current last action and latency.
    """
    plane = env.aircraft
    to_go = -env.along
    distances = jnp.asarray(GATE_DISTANCES, dtype=jnp.float32)
    ahead = distances < to_go                              # rings still in front
    has_next = jnp.any(ahead)
    next_distance = jnp.max(jnp.where(ahead, distances, -1.0))
    next_distance = jnp.where(has_next, next_distance, 0.0)
    half_width, half_height = ring_tolerance(next_distance)
    height = plane.py - env.runway_elev
    ring_lateral = jnp.where(has_next, env.lateral / half_width, 0.0)
    ring_vertical = jnp.where(has_next, (height - next_distance * GLIDE_SLOPE) / half_height, 0.0)
    track = (plane.track - env.runway_heading) * RAD
    wind_along, wind_cross = runway_coordinates(plane.wx, plane.wz, env.runway_heading)
    vertical_speed = plane.speed * jnp.sin(plane.gamma * RAD)
    values = (
        env.along / 4000.0, env.lateral / 100.0, (height - jnp.maximum(to_go, 0.0) * GLIDE_SLOPE) / 50.0,
        jnp.where(has_next, (to_go - next_distance) / 1000.0, 0.0), ring_lateral, ring_vertical,
        has_next.astype(jnp.float32),
        (plane.speed - 47.0) / 10.0, plane.gamma / 5.0, jnp.sin(track), jnp.cos(track),
        wrap180(plane.heading - env.runway_heading) / 30.0,
        plane.pitch / 10.0, plane.roll / 30.0, plane.prate / 10.0, plane.rrate / 20.0, plane.yrate / 10.0,
        jnp.minimum(plane.agl, 200.0) / 100.0, vertical_speed / 5.0, plane.flap_pos / 2.0, plane.alpha / 10.0,
        wind_along / 10.0, wind_cross / 10.0, plane.stall_warn.astype(jnp.float32),
    )
    return jnp.concatenate((jnp.stack([jnp.asarray(v, jnp.float32) for v in values]), last_action.astype(jnp.float32),
                            jnp.reshape(jnp.asarray(latency, jnp.float32) / MAX_LATENCY, (1,))))


@partial(jax.jit, static_argnames=("cfg", "steps"))
def _record_chunk(state: TeacherState, tuning, dt, cfg: MPPIConfig, steps: int):
    batch_plan = jax.vmap(plan, in_axes=(0, 0, 0, 0, None, None, None))
    batch_hold = jax.vmap(_hold, in_axes=(0, 0, None, None, None))
    batch_observe = jax.vmap(observe)

    def body(current: TeacherState, _):
        key, sub = jax.random.split(current.key)
        keys = jax.random.split(sub, current.nominal.shape[0])
        nominal, _, _ = batch_plan(current.environments, current.nominal, current.last_action, keys, tuning, dt, cfg)
        action = nominal[:, 0]
        envs = current.environments
        observation = batch_observe(envs, current.last_action)
        valid = ~envs.done & (envs.aircraft.agl >= HANDOVER_AGL)
        following = batch_hold(envs, action, tuning, dt, cfg.hold)
        shifted = jnp.concatenate((nominal[:, 1:], nominal[:, -1:]), axis=1)
        return TeacherState(following, shifted, action, key, current.control_steps + 1), (observation, action, valid)

    return jax.lax.scan(body, state, None, length=steps)


def collect_shard(config: TeacherConfig, tuning, device, max_control_steps=1800, chunk_steps=100):
    """Fly one seeded batch with the teacher and return (arrays, summary)."""
    cfg = config.mppi
    tuning = jax.device_put(tuning, device)
    dt = jax.device_put(jnp.float32(config.dt), device)
    state = _initialize(jax.device_put(jax.random.PRNGKey(config.seed), device),
                        jax.device_put(jnp.float32(config.difficulty), device), tuning, config.batch_size, cfg)
    observations, actions, episodes, steps = [], [], [], []
    step = 0
    while step < max_control_steps and not bool(jnp.all(state.environments.done)):
        n = min(chunk_steps, max_control_steps - step)
        state, (obs, act, valid) = _record_chunk(state, tuning, dt, cfg, n)
        obs, act, valid = np.asarray(obs), np.asarray(act), np.asarray(valid)   # (n, batch, ...)
        t_index, b_index = np.nonzero(valid)
        observations.append(obs[t_index, b_index])
        actions.append(act[t_index, b_index])
        episodes.append(b_index.astype(np.int32))
        steps.append((t_index + step).astype(np.int32))
        step += n
    arrays = {
        "observations": np.concatenate(observations).astype(np.float32),
        "actions": np.concatenate(actions).astype(np.float32),
        "episode": np.concatenate(episodes),
        "control_step": np.concatenate(steps),
        "gate_states": np.asarray(state.environments.gate_states, dtype=np.int8),
        "termination": np.asarray(state.environments.termination, dtype=np.int8),
        "touchdown_sink": np.asarray(state.environments.touchdown_vertical_speed, dtype=np.float32),
    }
    return arrays, {"control_steps": step, **summarize(state.environments)}


def _shard_paths(out_dir, index):
    base = Path(out_dir) / f"shard-{index:05d}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def shard_complete(out_dir, index):
    npz, manifest = _shard_paths(out_dir, index)
    if not (npz.exists() and manifest.exists()):
        return False
    try:
        meta = json.loads(manifest.read_text())
        return hashlib.sha256(npz.read_bytes()).hexdigest() == meta["sha256"]
    except (OSError, ValueError, KeyError):
        return False


def generate(out_dir, shards, base: TeacherConfig, difficulties=(0.25, 0.5, 0.75, 1.0), device=None, *,
             tuning=None, max_control_steps=1800, progress=print, collect=None, schema=SCHEMA, extra_meta=None):
    """Write ``shards`` shards; shard i uses seed base.seed + i and difficulty difficulties[i % len].

    ``collect(config, tuning, device, max_control_steps) -> (arrays, summary)`` defaults to the
    teacher-only ``collect_shard``; DAgger passes its own collector and schema.
    """
    collect = collect_shard if collect is None else collect
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tuning = default_tuning() if tuning is None else tuning
    device = device if device is not None else jax.devices()[0]
    tuning_json = {k: np.asarray(v).tolist() for k, v in tuning._asdict().items()}
    written = 0
    for index in range(shards):
        if shard_complete(out, index):
            continue
        config = replace(base, seed=base.seed + index, difficulty=float(difficulties[index % len(difficulties)]))
        started = time.time()
        arrays, summary = collect(config, tuning, device, max_control_steps)
        npz, manifest = _shard_paths(out, index)
        tmp = npz.with_name(npz.name + ".tmp")
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp, npz)
        meta = {
            "schema": schema, "shard": index, "seed": config.seed, "difficulty": config.difficulty,
            "batch_size": config.batch_size, "dt": config.dt, "handover_agl": HANDOVER_AGL,
            "observation_names": list(OBSERVATION_NAMES), "mppi": asdict(config.mppi), "tuning": tuning_json,
            "samples": int(arrays["actions"].shape[0]), "seconds": round(time.time() - started, 1),
            "summary": summary, "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(), **(extra_meta or {}),
        }
        manifest.write_text(json.dumps(meta, indent=1))
        written += 1
        if progress is not None:
            progress(f"shard {index}: difficulty {config.difficulty} · {meta['samples']:,} samples · "
                     f"rings {summary['ring_pass_rate']:.1%} · {meta['seconds']}s")
    return {"shards": shards, "written": written}


def load(out_dir):
    """Concatenate every complete shard: observations, actions, and a global episode id."""
    obs, act, ep = [], [], []
    offset = 0
    for manifest in sorted(Path(out_dir).glob("shard-*.json")):
        index = int(manifest.stem.split("-")[1])
        if not shard_complete(out_dir, index):
            continue
        meta = json.loads(manifest.read_text())
        with np.load(manifest.with_suffix(".npz")) as data:
            o = data["observations"]
            if o.shape[1] == OBSERVATION_SIZE - 1:      # v1 shards predate the latency input: host, latency 0
                o = np.concatenate((o, np.zeros((o.shape[0], 1), o.dtype)), axis=1)
            obs.append(o); act.append(data["actions"]); ep.append(data["episode"] + offset)
        offset += meta["batch_size"]
    if not obs:
        return None
    return {"observations": np.concatenate(obs), "actions": np.concatenate(act), "episode": np.concatenate(ep)}
