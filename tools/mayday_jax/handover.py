"""Approach evaluation up to the flare handover, and DAgger data collection.

The network flies the approach only; below HANDOVER_AGL the mod's rule-based
flare takes over. ``evaluate_to_handover`` therefore stops each episode at its
first frame below HANDOVER_AGL and reports the ring verdicts plus the state the
flare logic would inherit (position, sink rate, speed, bank). Teacher and
student are measured the same way, and no stand-in landing controller is
involved.

``collect_dagger_shard`` lets the student fly while the privileged MPPI teacher
plans from every state the student reaches; the teacher's actions become the
labels. Shards use the teacher dataset format (``dataset.generate``).

Latency: a client sees the aircraft up to MAX_LATENCY late. Each student episode
gets a lag of 0..MAX_LAG control steps (host episodes 0). The student observes the
state part of ``observe`` from ``lag`` steps ago, its own last action now, and the
latency value; the teacher always labels the true current state.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .dataset import HANDOVER_AGL, MAX_LATENCY, STATE_SIZE, generate, observe
from .environment import GATE_DISTANCES
from .physics import RAD
from .mppi import TeacherConfig, TeacherState, _hold, _initialize, initial_environments, plan
from .policy import Normalizer, apply

DAGGER_SCHEMA = "mayday-jax-dagger-v2"
CONTROL_DT = 0.1                                   # 6 physics frames at 60 Hz
MAX_LAG = int(round(MAX_LATENCY / CONTROL_DT))      # 6 control steps


class Handover(NamedTuple):
    captured: jax.Array
    along: jax.Array
    lateral: jax.Array
    sink: jax.Array
    speed: jax.Array
    roll: jax.Array
    gamma: jax.Array


def _empty_handover(batch):
    zero = jnp.zeros((batch,), jnp.float32)
    return Handover(jnp.zeros((batch,), bool), zero, zero, zero, zero, zero, zero)


def _capture(handover: Handover, envs):
    plane = envs.aircraft
    newly = ~handover.captured & ~envs.done & (plane.agl < HANDOVER_AGL)
    pick = lambda new, old: jnp.where(newly, new, old)
    return Handover(
        handover.captured | newly, pick(envs.along, handover.along), pick(envs.lateral, handover.lateral),
        pick(-plane.speed * jnp.sin(plane.gamma * RAD), handover.sink), pick(plane.speed, handover.speed),
        pick(plane.roll, handover.roll), pick(plane.gamma, handover.gamma),
    )


def _finished(envs, handover):
    return handover.captured | envs.done


def sample_lags(key, batch, host_fraction=0.5):
    """Per-episode lag in control steps: 0 for host episodes, else uniform 1..MAX_LAG."""
    host_key, lag_key = jax.random.split(key)
    host = jax.random.uniform(host_key, (batch,)) < host_fraction
    lag = jax.random.randint(lag_key, (batch,), 1, MAX_LAG + 1)
    return jnp.where(host, 0, lag).astype(jnp.int32)


def _initial_buffer(envs, last):
    fresh = jax.vmap(observe)(envs, last)[:, :STATE_SIZE]
    return jnp.repeat(fresh[:, None, :], MAX_LAG + 1, axis=1)


def _delayed_observation(buffer, envs, last, lags):
    """Push the current state part into the buffer and assemble what a client with ``lags`` would see."""
    fresh = jax.vmap(observe)(envs, last)[:, :STATE_SIZE]
    buffer = jnp.concatenate((fresh[:, None, :], buffer[:, :-1]), axis=1)
    old = jnp.take_along_axis(buffer, lags[:, None, None], axis=1)[:, 0]
    latency = lags.astype(jnp.float32) * CONTROL_DT / MAX_LATENCY
    return jnp.concatenate((old, last, latency[:, None]), axis=1), buffer


@partial(jax.jit, static_argnames=("steps", "hold"))
def _student_chunk(envs, last, buffer, handover, lags, params, norm, tuning, dt, steps: int, hold: int):
    batch_hold = jax.vmap(_hold, in_axes=(0, 0, None, None, None))

    def body(carry, _):
        current, previous, history, captured = carry
        observation, history = _delayed_observation(history, current, previous, lags)
        action = apply(params, norm, observation)
        following = batch_hold(current, action, tuning, dt, hold)
        return (following, action, history, _capture(captured, following)), None

    (envs, last, buffer, handover), _ = jax.lax.scan(body, (envs, last, buffer, handover), None, length=steps)
    return envs, last, buffer, handover


@partial(jax.jit, static_argnames=("cfg", "steps"))
def _teacher_chunk(state: TeacherState, handover, tuning, dt, cfg, steps: int):
    batch_plan = jax.vmap(plan, in_axes=(0, 0, 0, 0, None, None, None))
    batch_hold = jax.vmap(_hold, in_axes=(0, 0, None, None, None))

    def body(carry, _):
        current, captured = carry
        key, sub = jax.random.split(current.key)
        keys = jax.random.split(sub, current.nominal.shape[0])
        nominal, _, _ = batch_plan(current.environments, current.nominal, current.last_action, keys, tuning, dt, cfg)
        action = nominal[:, 0]
        envs = batch_hold(current.environments, action, tuning, dt, cfg.hold)
        shifted = jnp.concatenate((nominal[:, 1:], nominal[:, -1:]), axis=1)
        nxt = TeacherState(envs, shifted, action, key, current.control_steps + 1)
        return (nxt, _capture(captured, envs)), None

    (state, handover), _ = jax.lax.scan(body, (state, handover), None, length=steps)
    return state, handover


def handover_summary(envs, handover: Handover):
    """Ring verdicts at the end of the approach plus statistics of the state handed to the flare logic."""
    states = np.asarray(envs.gate_states)
    batch = states.shape[0]
    threaded = (states == 1).sum(1)
    got = np.asarray(handover.captured)
    out = {
        "episodes": int(batch),
        "ring_pass_rate": float(threaded.sum() / (batch * len(GATE_DISTANCES))),
        "all_rings_rate": float((threaded == len(GATE_DISTANCES)).mean()),
        "rings_missed": int((states == 2).sum()),
        "rings_unjudged": int((states == 0).sum()),
        "handover_rate": float(got.mean()),
        "ended_before_handover": int((np.asarray(envs.done) & ~got).sum()),
    }
    if got.any():
        along, lateral = np.asarray(handover.along)[got], np.abs(np.asarray(handover.lateral)[got])
        sink, speed, roll = np.asarray(handover.sink)[got], np.asarray(handover.speed)[got], np.abs(np.asarray(handover.roll)[got])
        out.update({
            "along_p5_p50_p95": np.percentile(along, [5, 50, 95]).round(0).tolist(),
            "abs_lateral_p50_p95_max": [round(float(np.median(lateral)), 1), round(float(np.percentile(lateral, 95)), 1),
                                        round(float(lateral.max()), 1)],
            "sink_p50_p95_max": [round(float(np.median(sink)), 2), round(float(np.percentile(sink, 95)), 2),
                                 round(float(sink.max()), 2)],
            "speed_p5_p50_p95": np.percentile(speed, [5, 50, 95]).round(1).tolist(),
            "abs_roll_p95_max": [round(float(np.percentile(roll, 95)), 1), round(float(roll.max()), 1)],
            "lined_up_rate": float(((lateral < 15.0) & (along > -400.0) & (along < 600.0)).mean()),
        })
    return out


def evaluate_to_handover(config: TeacherConfig, device=None, *, tuning, params=None, norm=None, lag=None,
                         max_control_steps=1800, chunk=100):
    """Fly the seeded episodes of ``config`` until each one reaches the handover (or ends).

    With ``params``/``norm`` the student flies, seeing the state ``lag`` control steps late
    (an int for all episodes, or None for the host/client mix of ``sample_lags``);
    otherwise the privileged MPPI teacher (config.mppi) flies without delay.
    """
    device = device if device is not None else jax.devices()[0]
    tuning = jax.device_put(tuning, device)
    dt = jax.device_put(jnp.float32(config.dt), device)
    key = jax.device_put(jax.random.PRNGKey(config.seed), device)
    difficulty = jax.device_put(jnp.float32(config.difficulty), device)
    handover = jax.device_put(_empty_handover(config.batch_size), device)
    step = 0
    state = _initialize(key, difficulty, tuning, config.batch_size, config.mppi)
    if params is not None:
        params, norm = jax.device_put((params, Normalizer(*norm)), device)
        if lag is None:
            lags = sample_lags(jax.random.fold_in(key, 99), config.batch_size)
        else:
            lags = jnp.full((config.batch_size,), int(lag), jnp.int32)
        lags = jax.device_put(lags, device)
        envs, last = state.environments, state.last_action
        buffer = _initial_buffer(envs, last)
        while step < max_control_steps and not bool(jnp.all(_finished(envs, handover))):
            n = min(chunk, max_control_steps - step)
            envs, last, buffer, handover = _student_chunk(envs, last, buffer, handover, lags, params, norm, tuning, dt,
                                                          n, config.mppi.hold)
            step += n
    else:
        while step < max_control_steps and not bool(jnp.all(_finished(state.environments, handover))):
            n = min(chunk, max_control_steps - step)
            state, handover = _teacher_chunk(state, handover, tuning, dt, config.mppi, n)
            step += n
        envs = state.environments
    return {"control_steps": step, **handover_summary(envs, handover)}


@partial(jax.jit, static_argnames=("cfg", "steps"))
def _dagger_chunk(state: TeacherState, buffer, handover, lags, params, norm, tuning, dt, cfg, steps: int):
    batch_plan = jax.vmap(plan, in_axes=(0, 0, 0, 0, None, None, None))
    batch_hold = jax.vmap(_hold, in_axes=(0, 0, None, None, None))

    def body(carry, _):
        current, history, captured = carry
        key, sub = jax.random.split(current.key)
        keys = jax.random.split(sub, current.nominal.shape[0])
        envs = current.environments
        # the teacher plans from the true state the student brought us to; its first action is the label
        nominal, _, _ = batch_plan(envs, current.nominal, current.last_action, keys, tuning, dt, cfg)
        label = nominal[:, 0]
        observation, history = _delayed_observation(history, envs, current.last_action, lags)
        executed = apply(params, norm, observation)               # the student flies on what it can see
        valid = ~envs.done & (envs.aircraft.agl >= HANDOVER_AGL)
        following = batch_hold(envs, executed, tuning, dt, cfg.hold)
        shifted = jnp.concatenate((nominal[:, 1:], nominal[:, -1:]), axis=1)
        nxt = TeacherState(following, shifted, executed, key, current.control_steps + 1)
        return (nxt, history, _capture(captured, following)), (observation, label, valid)

    return jax.lax.scan(body, (state, buffer, handover), None, length=steps)


def collect_dagger_shard(config: TeacherConfig, tuning, device, max_control_steps=1800, chunk_steps=100, *,
                         params, norm):
    """Student-driven episodes labelled by the teacher, until every episode reaches the handover or ends."""
    cfg = config.mppi
    tuning = jax.device_put(tuning, device)
    dt = jax.device_put(jnp.float32(config.dt), device)
    params, norm = jax.device_put((params, Normalizer(*norm)), device)
    key = jax.device_put(jax.random.PRNGKey(config.seed), device)
    state = _initialize(key, jax.device_put(jnp.float32(config.difficulty), device), tuning, config.batch_size, cfg)
    lags = jax.device_put(sample_lags(jax.random.fold_in(key, 99), config.batch_size), device)
    buffer = _initial_buffer(state.environments, state.last_action)
    handover = jax.device_put(_empty_handover(config.batch_size), device)
    observations, labels, episodes, steps = [], [], [], []
    step = 0
    while step < max_control_steps and not bool(jnp.all(_finished(state.environments, handover))):
        n = min(chunk_steps, max_control_steps - step)
        (state, buffer, handover), (obs, act, valid) = _dagger_chunk(state, buffer, handover, lags, params, norm,
                                                                     tuning, dt, cfg, n)
        obs, act, valid = np.asarray(obs), np.asarray(act), np.asarray(valid)
        t_index, b_index = np.nonzero(valid)
        observations.append(obs[t_index, b_index])
        labels.append(act[t_index, b_index])
        episodes.append(b_index.astype(np.int32))
        steps.append((t_index + step).astype(np.int32))
        step += n
    arrays = {
        "observations": np.concatenate(observations).astype(np.float32),
        "actions": np.concatenate(labels).astype(np.float32),
        "episode": np.concatenate(episodes),
        "control_step": np.concatenate(steps),
        "lag_steps": np.asarray(lags, dtype=np.int8),
        "gate_states": np.asarray(state.environments.gate_states, dtype=np.int8),
        "termination": np.asarray(state.environments.termination, dtype=np.int8),
        "touchdown_sink": np.asarray(state.environments.touchdown_vertical_speed, dtype=np.float32),
    }
    summary = handover_summary(state.environments, handover)
    summary["client_fraction"] = float((np.asarray(lags) > 0).mean())
    return arrays, {"control_steps": step, **summary}


def generate_dagger(out_dir, shards, base: TeacherConfig, params, norm, policy_name, difficulties=(0.5, 0.75, 1.0, 1.0),
                    device=None, *, tuning, max_control_steps=1800, progress=print):
    """DAgger shards in the teacher dataset format (resumable); ``policy_name`` is recorded in each manifest."""
    collect = partial(collect_dagger_shard, params=params, norm=norm)
    return generate(out_dir, shards, base, difficulties, device, tuning=tuning, max_control_steps=max_control_steps,
                    progress=progress, collect=collect, schema=DAGGER_SCHEMA, extra_meta={"student_policy": policy_name})
