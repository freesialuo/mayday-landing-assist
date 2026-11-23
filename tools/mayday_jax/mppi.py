# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Phase C: privileged MPPI teacher for the approach environment.

The teacher plans with the exact environment model, including each episode's
disturbance parameters (wind, turbulence phases, competing seat pulses). It is
an upper bound and a source of demonstrations; the phase D student only sees
what a client could observe.

Planning runs at 10 Hz by default: every knot holds one (pitch, roll, yaw,
throttle) action for ``hold`` physics frames. Each control step samples noisy
action sequences around a warm-started nominal plan, scores them with
``_sequence_cost``, and averages them with weights exp(-cost / lambda), where
lambda adapts to the spread of the sampled costs.
"""

from dataclasses import asdict, dataclass, field
from functools import partial
import math
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoints import load_checkpoint, save_checkpoint
from .environment import (
    GATE_DISTANCES, GLIDE_SLOPE, LEFT_AREA, OFF_RUNWAY_TOUCHDOWN, RUNWAY_TOUCHDOWN, TIMEOUT,
    EnvState, advance, reset,
)
from .physics import RAD, default_tuning, wrap180
from .rollout import _demo_actions

ACTION_LOW = (-1.0, -1.0, -1.0, 0.0)
ACTION_HIGH = (1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class MPPIConfig:
    samples: int = 128
    horizon: int = 30              # knots
    hold: int = 6                  # physics frames per knot (6 x 1/60 s = 0.1 s)
    noise_std: tuple = (0.30, 0.30, 0.10, 0.15)
    temperature: float = 0.25      # fraction of the (mean - min) cost spread
    target_speed: float = 47.0
    path_weight: float = 1.0
    edge_weight: float = 10.0      # extra cost beyond `edge` ring units
    edge: float = 0.6
    align_weight: float = 0.2
    speed_weight: float = 0.3
    roll_weight: float = 0.2
    smooth_weight: float = 0.05
    flare_weight: float = 0.5
    idle_weight: float = 2.0       # throttle cost below 8 m AGL: pull to idle in the flare
    float_weight: float = 2.0      # airborne beyond the aim point: ((along - aim) / 200)^2
    terminal_weight: float = 5.0
    miss_cost: float = 300.0
    pass_reward: float = 50.0
    land_reward: float = 100.0
    sink_cost: float = 50.0
    crash_cost: float = 500.0


@dataclass(frozen=True)
class TeacherConfig:
    batch_size: int = 64
    seed: int = 1
    difficulty: float = 0.5
    dt: float = 1 / 60
    chunk_steps: int = 50          # control steps per jitted chunk
    mppi: MPPIConfig = field(default_factory=MPPIConfig)


class TeacherState(NamedTuple):
    environments: EnvState
    nominal: jax.Array             # (batch, horizon, 4)
    last_action: jax.Array         # (batch, 4)
    key: jax.Array
    control_steps: jax.Array


def ring_tolerance(distance):
    """Half width and half height of a ring at ``distance`` metres before the threshold (GateGeometry)."""
    half_width = 20.0 * (2.2 + 1.8 * jnp.clip((distance - 600.0) / 3400.0, 0.0, 1.0))
    return half_width, 0.55 * half_width


LAST_RING = min(GATE_DISTANCES)
AIM_PAST_THRESHOLD = 200.0     # touchdown aim point, metres past the threshold
RUNWAY_LATERAL_SCALE = 15.0    # lateral scale over the runway (paved half width is 30 m)


def path_error(env: EnvState):
    """Lateral and vertical offset from the reference path, in units of the local tolerance.

    Up to the last ring the reference is the ring path itself (3 degrees to the
    threshold) scaled by the ring half size. Inside the last ring it blends to a
    3-degree path aimed AIM_PAST_THRESHOLD beyond the threshold, and the lateral
    scale tightens toward the runway, so a touchdown on the path lands on paved runway.
    """
    distance = jnp.maximum(-env.along, 0.0)
    half_width, half_height = ring_tolerance(distance)
    blend = jnp.clip((LAST_RING - distance) / LAST_RING, 0.0, 1.0)
    ring_height = distance * GLIDE_SLOPE
    aim_height = jnp.maximum(AIM_PAST_THRESHOLD - env.along, 0.0) * GLIDE_SLOPE
    reference = ring_height + blend * (aim_height - ring_height)
    lateral_scale = half_width + blend * (RUNWAY_LATERAL_SCALE - half_width)
    height = env.aircraft.py - env.runway_elev - reference
    return env.lateral / lateral_scale, height / half_height


def _clip_actions(actions):
    return jnp.clip(actions, jnp.asarray(ACTION_LOW), jnp.asarray(ACTION_HIGH))


def _hold(env, action, tuning, dt, hold):
    return jax.lax.fori_loop(0, hold, lambda _, e: advance(e, action, tuning, dt), env)


def _path_cost(env, cfg):
    lat, ver = path_error(env)
    edge = jnp.maximum(jnp.abs(lat) - cfg.edge, 0.0) ** 2 + jnp.maximum(jnp.abs(ver) - cfg.edge, 0.0) ** 2
    return lat**2 + ver**2 + cfg.edge_weight * edge


def _knot_cost(before, after, action, previous_action, cfg):
    live = ~before.done
    plane = after.aircraft
    align = (wrap180(plane.track - after.runway_heading) / 20.0) ** 2 * (after.along < 0.0)
    # Holding approach speed into the flare makes the aircraft float in ground effect,
    # so the speed term fades out between 20 m and 5 m AGL.
    speed = ((plane.speed - cfg.target_speed) / 5.0) ** 2 * jnp.clip((plane.agl - 5.0) / 15.0, 0.0, 1.0)
    airborne = ~plane.on_ground
    idle = jnp.where(airborne & (plane.agl < 8.0), action[3] ** 2, 0.0)
    floating = jnp.where(airborne, jnp.maximum(after.along - AIM_PAST_THRESHOLD, 0.0) / 200.0, 0.0) ** 2
    attitude = (plane.roll / 25.0) ** 2
    envelope = 5.0 * plane.stall_warn + 20.0 * plane.stalling + 5.0 * plane.overspeed
    scale = jnp.asarray((0.5, 0.5, 0.5, 0.3))
    smooth = jnp.sum(((action - previous_action) / scale) ** 2)
    sink = plane.speed * jnp.sin(plane.gamma * RAD)
    flare = jnp.where(~plane.on_ground & (plane.agl < 15.0), (sink + 1.0) ** 2, 0.0)
    running = (
        cfg.path_weight * _path_cost(after, cfg) + cfg.align_weight * align + cfg.speed_weight * speed
        + cfg.roll_weight * attitude + envelope + cfg.smooth_weight * smooth + cfg.flare_weight * flare
        + cfg.idle_weight * idle + cfg.float_weight * floating
    )

    judged = after.gate_states != before.gate_states
    gates = (cfg.miss_cost * jnp.count_nonzero(judged & (after.gate_states == 2))
             - cfg.pass_reward * jnp.count_nonzero(judged & (after.gate_states == 1)))
    ended = after.done & live
    landing = jnp.where(
        after.termination == RUNWAY_TOUCHDOWN,
        -cfg.land_reward + cfg.sink_cost * jnp.maximum(after.touchdown_vertical_speed - 1.5, 0.0) ** 2,
        0.0,
    )
    crash = jnp.where((after.termination == OFF_RUNWAY_TOUCHDOWN) | (after.termination == LEFT_AREA),
                      cfg.crash_cost, 0.0)
    return jnp.where(live, running + gates, 0.0) + jnp.where(ended, landing + crash, 0.0)


def _sequence_cost(env, actions, previous_action, tuning, dt, cfg):
    def knot(carry, action):
        current, last = carry
        following = _hold(current, action, tuning, dt, cfg.hold)
        return (following, action), _knot_cost(current, following, action, last, cfg)

    (final, _), costs = jax.lax.scan(knot, (env, previous_action), actions)
    terminal = jnp.where(final.done, 0.0, cfg.terminal_weight * _path_cost(final, cfg))
    return jnp.sum(costs) + terminal


def plan(env, nominal, previous_action, key, tuning, dt, cfg: MPPIConfig):
    """One MPPI update for one environment. Returns the new nominal plan and (best, weighted) cost."""
    std = jnp.asarray(cfg.noise_std, dtype=nominal.dtype)
    noise = jax.random.normal(key, (cfg.samples,) + nominal.shape, dtype=nominal.dtype) * std
    noise = noise.at[0].set(0.0)   # always evaluate the unperturbed nominal
    actions = _clip_actions(nominal[None] + noise)
    costs = jax.vmap(_sequence_cost, in_axes=(None, 0, None, None, None, None))(
        env, actions, previous_action, tuning, dt, cfg)
    best = jnp.min(costs)
    spread = jnp.maximum(jnp.mean(costs) - best, 1e-3)
    weights = jax.nn.softmax(-(costs - best) / (cfg.temperature * spread))
    updated = _clip_actions(jnp.tensordot(weights, actions, axes=1))
    return updated, best, jnp.sum(weights * costs)


@partial(jax.jit, static_argnames=("cfg",))
def _control_chunk(state: TeacherState, tuning, dt, steps, cfg: MPPIConfig):
    batch_plan = jax.vmap(plan, in_axes=(0, 0, 0, 0, None, None, None))
    batch_hold = jax.vmap(_hold, in_axes=(0, 0, None, None, None))

    def body(_, current: TeacherState):
        key, sub = jax.random.split(current.key)
        keys = jax.random.split(sub, current.nominal.shape[0])
        nominal, _, _ = batch_plan(current.environments, current.nominal, current.last_action, keys, tuning, dt, cfg)
        action = nominal[:, 0]
        envs = batch_hold(current.environments, action, tuning, dt, cfg.hold)
        shifted = jnp.concatenate((nominal[:, 1:], nominal[:, -1:]), axis=1)
        return TeacherState(envs, shifted, action, key, current.control_steps + 1)

    return jax.lax.fori_loop(0, steps, body, state)


_reset_batch = jax.vmap(reset, in_axes=(0, None, None))


@partial(jax.jit, static_argnames=("batch_size",))
def initial_environments(key, difficulty, tuning, batch_size):
    """Seeded scenarios shared by the teacher and the baseline evaluation."""
    key, reset_key = jax.random.split(key)
    return _reset_batch(jax.random.split(reset_key, batch_size), difficulty, tuning), key


@partial(jax.jit, static_argnames=("batch_size", "cfg"))
def _initialize(key, difficulty, tuning, batch_size, cfg: MPPIConfig):
    envs, key = initial_environments(key, difficulty, tuning, batch_size)
    first = _clip_actions(_demo_actions(envs))
    nominal = jnp.broadcast_to(first[:, None, :], (batch_size, cfg.horizon, 4))
    return TeacherState(envs, nominal, first, key, jnp.int32(0))


def summarize(envs: EnvState):
    """Ring, landing and termination statistics for a batch of (finished) episodes."""
    states = np.asarray(envs.gate_states)
    term = np.asarray(envs.termination)
    sink = np.asarray(envs.touchdown_vertical_speed)
    batch = states.shape[0]
    threaded = (states == 1).sum(axis=1)
    on_runway = term == RUNWAY_TOUCHDOWN
    return {
        "episodes": int(batch),
        "finished": int(np.asarray(envs.done).sum()),
        "ring_pass_rate": float(threaded.sum() / (batch * len(GATE_DISTANCES))),
        "all_rings_rate": float((threaded == len(GATE_DISTANCES)).mean()),
        "rings_missed": int((states == 2).sum()),
        "runway_touchdown_rate": float(on_runway.mean()),
        "off_runway_touchdowns": int((term == OFF_RUNWAY_TOUCHDOWN).sum()),
        "left_area": int((term == LEFT_AREA).sum()),
        "timeouts": int((term == TIMEOUT).sum()),
        "mean_touchdown_sink": float(sink[on_runway].mean()) if on_runway.any() else None,
        "max_touchdown_sink": float(sink[on_runway].max()) if on_runway.any() else None,
    }


def _identity(config: TeacherConfig, tuning):
    return {
        "schema": "mayday-jax-mppi-teacher-v1",
        "batch_size": config.batch_size,
        "seed": config.seed,
        "difficulty": config.difficulty,
        "dt": config.dt,
        "mppi": asdict(config.mppi),
        "tuning": {key: np.asarray(value).tolist() for key, value in tuning._asdict().items()},
    }


def run_teacher(config: TeacherConfig, checkpoint_dir, max_control_steps, checkpoint_every,
                device=None, progress=print, *, tuning=None):
    """Fly every seeded episode with the MPPI teacher until all end or ``max_control_steps``.

    Checkpoints (environments, nominal plans, last actions, RNG key) are saved every
    ``checkpoint_every`` control steps and at the end; a compatible checkpoint in
    ``checkpoint_dir`` is resumed automatically, reproducing the uninterrupted run.
    """
    if config.batch_size <= 0 or config.chunk_steps <= 0 or checkpoint_every <= 0 or max_control_steps < 0:
        raise ValueError("batch_size, chunk_steps and checkpoint_every must be positive")
    if not 0 <= config.difficulty <= 1 or not math.isfinite(config.dt) or config.dt <= 0:
        raise ValueError("difficulty must be in [0, 1] and dt positive")
    cfg = config.mppi
    directory = Path(checkpoint_dir) if checkpoint_dir is not None else None
    tuning = default_tuning() if tuning is None else tuning
    identity = _identity(config, tuning)
    device = device if device is not None else jax.devices()[0]
    tuning = jax.device_put(tuning, device)
    dt = jax.device_put(jnp.float32(config.dt), device)
    difficulty = jax.device_put(jnp.float32(config.difficulty), device)
    state = _initialize(jax.device_put(jax.random.PRNGKey(config.seed), device), difficulty, tuning,
                        config.batch_size, cfg)
    with jax.default_device(device):
        restored = load_checkpoint(directory, state, identity) if directory is not None else None
    step = 0 if restored is None else restored[0]
    if restored is not None:
        state = jax.device_put(restored[1], device)
    if progress is not None:
        progress(f"{device}: {config.batch_size} episodes, MPPI {cfg.samples} samples x {cfg.horizon} knots, "
                 f"control step {step} -> {max_control_steps}")
    if directory is not None and restored is None:
        save_checkpoint(directory, step, state, identity)

    while step < max_control_steps and not bool(jnp.all(state.environments.done)):
        next_save = (step // checkpoint_every + 1) * checkpoint_every
        end = min(step + config.chunk_steps, max_control_steps, next_save)
        state = _control_chunk(state, tuning, dt, jax.device_put(jnp.int32(end - step), device), cfg)
        step = end
        finished = int(jnp.count_nonzero(state.environments.done))
        if directory is not None and (step == next_save or step == max_control_steps or finished == config.batch_size):
            save_checkpoint(directory, step, state, identity)
        if progress is not None:
            progress(f"control step {step}: {finished}/{config.batch_size} episodes finished")

    jax.block_until_ready(state)
    return {"control_steps": step, "restored_step": None if restored is None else restored[0],
            **summarize(state.environments)}, state


@jax.jit
def _demo_chunk(envs, tuning, dt, steps):
    batch_advance = jax.vmap(advance, in_axes=(0, 0, None, None))
    return jax.lax.fori_loop(0, steps, lambda _, e: batch_advance(e, _demo_actions(e), tuning, dt), envs)


def run_baseline(config: TeacherConfig, device=None, *, tuning=None, max_seconds=180.0):
    """Fly the same seeded episodes with the phase B demo controller (per physics frame) for comparison."""
    tuning = default_tuning() if tuning is None else tuning
    device = device if device is not None else jax.devices()[0]
    tuning = jax.device_put(tuning, device)
    envs, _ = initial_environments(jax.device_put(jax.random.PRNGKey(config.seed), device),
                                   jax.device_put(jnp.float32(config.difficulty), device), tuning, config.batch_size)
    dt = jax.device_put(jnp.float32(config.dt), device)
    remaining = int(math.ceil(max_seconds / config.dt)) + 1
    while remaining > 0 and not bool(jnp.all(envs.done)):
        steps = min(remaining, 600)
        envs = _demo_chunk(envs, tuning, dt, jax.device_put(jnp.int32(steps), device))
        remaining -= steps
    return summarize(envs)
