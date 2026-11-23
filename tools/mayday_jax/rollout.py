# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run and resume a vectorized batch of randomized approach episodes."""

from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoints import load_checkpoint, save_checkpoint
from .environment import EnvState, GLIDE_SLOPE, advance, reset
from .physics import default_tuning


@dataclass(frozen=True)
class RunConfig:
    batch_size: int = 2048
    seed: int = 1
    difficulty: float = 0.5
    chunk_steps: int = 128
    dt: float = 1 / 60


class RolloutState(NamedTuple):
    environments: EnvState
    key: jax.Array
    episodes_completed: jax.Array
    gates_passed: jax.Array
    gates_missed: jax.Array


_reset_batch = jax.vmap(reset, in_axes=(0, None, None))
_advance_batch = jax.vmap(advance, in_axes=(0, 0, None, None))


@partial(jax.jit, static_argnames=("batch_size",))
def _initialize(key, difficulty, tuning, batch_size):
    key, reset_key = jax.random.split(key)
    envs = _reset_batch(jax.random.split(reset_key, batch_size), difficulty, tuning)
    zero = jnp.int32(0)
    return RolloutState(envs, key, zero, zero, zero)


def _demo_actions(envs):
    """A simple approach controller for exercising rollouts before policy training."""
    plane = envs.aircraft
    heading_target = envs.runway_heading - jnp.degrees(jnp.arctan2(envs.lateral, 700.0))
    heading_error = (heading_target - plane.heading + 180.0) % 360.0 - 180.0
    bank_target = jnp.clip(heading_error * 0.9, -20.0, 20.0)
    roll = jnp.clip((bank_target - plane.roll) * 0.08, -1.0, 1.0)

    height_target = jnp.maximum(-envs.along * GLIDE_SLOPE + 2.8, 0.0)
    height_error = height_target - (plane.py - envs.runway_elev)
    gamma_target = jnp.clip(-3.0 + 0.065 * height_error, -8.0, 5.0)
    speed = jnp.maximum(plane.speed, 15.0)
    alpha_target = 14.0 * (26.0 / speed) ** 2
    pitch_target = gamma_target + alpha_target - 2.0
    pitch = jnp.clip((pitch_target - plane.pitch) * 0.22, -1.0, 1.0)

    drag = 0.0002318 * speed**2 * 1.35 + 950.6 / speed**2
    gravity = 9.81 * jnp.sin(jnp.radians(gamma_target))
    throttle = jnp.clip((drag + gravity) / 1.9 + 0.055 * (58.0 - speed), 0.0, 1.0)
    return jnp.stack((pitch, roll, jnp.zeros_like(roll), throttle), axis=-1)


@jax.jit
def _advance_chunk(batch, tuning, difficulty, dt, steps):
    def step_batch(_, current):
        previous = current.environments
        envs = _advance_batch(previous, _demo_actions(previous), tuning, dt)
        judged = envs.gate_states != previous.gate_states
        passes = jnp.count_nonzero(judged & (envs.gate_states == 1)).astype(jnp.int32)
        misses = jnp.count_nonzero(judged & (envs.gate_states == 2)).astype(jnp.int32)
        completed = jnp.count_nonzero(envs.done).astype(jnp.int32)

        def reset_finished(pair):
            landed, key = pair
            key, reset_key = jax.random.split(key)
            fresh = _reset_batch(jax.random.split(reset_key, landed.done.shape[0]), difficulty, tuning)

            def choose(old, new):
                mask = landed.done.reshape((landed.done.shape[0],) + (1,) * (old.ndim - 1))
                return jnp.where(mask, new, old)

            return jax.tree.map(choose, landed, fresh), key

        envs, key = jax.lax.cond(completed != 0, reset_finished, lambda pair: pair, (envs, current.key))
        return RolloutState(
            envs, key, current.episodes_completed + completed,
            current.gates_passed + passes, current.gates_missed + misses,
        )

    return jax.lax.fori_loop(0, steps, step_batch, batch)


def _identity(config, tuning):
    return {
        "schema": "mayday-jax-approach-rollout-v1",
        "batch_size": config.batch_size,
        "seed": config.seed,
        "difficulty": config.difficulty,
        "dt": config.dt,
        "tuning": {key: np.asarray(value).tolist() for key, value in tuning._asdict().items()},
    }


def run(config, checkpoint_dir, target_steps, checkpoint_every, device=None, progress=print, *, tuning=None):
    """Advance each environment to target_steps and save at each checkpoint boundary.

    A compatible checkpoint in checkpoint_dir is resumed automatically. If a chunk
    is interrupted, the previous complete checkpoint remains available. Checkpoint
    cadence and chunk size may change between sessions without changing trajectories.
    """
    if config.batch_size <= 0 or config.chunk_steps <= 0:
        raise ValueError("batch_size and chunk_steps must be positive")
    if not 0 <= config.seed < 2**32 or not 0 <= config.difficulty <= 1:
        raise ValueError("seed must be uint32 and difficulty must be in [0, 1]")
    if not math.isfinite(config.dt) or config.dt <= 0:
        raise ValueError("dt must be positive and finite")
    if target_steps < 0 or checkpoint_every <= 0:
        raise ValueError("target_steps must be nonnegative and checkpoint_every must be positive")

    directory = Path(checkpoint_dir) if checkpoint_dir is not None else None
    tuning = default_tuning() if tuning is None else tuning
    identity = _identity(config, tuning)
    device = device if device is not None else jax.devices()[0]
    tuning = jax.device_put(tuning, device)
    difficulty = jax.device_put(jnp.float32(config.difficulty), device)
    dt = jax.device_put(jnp.float32(config.dt), device)
    initial_key = jax.device_put(jax.random.PRNGKey(config.seed), device)
    state = _initialize(initial_key, difficulty, tuning, config.batch_size)
    with jax.default_device(device):
        restored = load_checkpoint(directory, state, identity) if directory is not None else None
    restored_step = None if restored is None else restored[0]
    step = 0 if restored is None else restored[0]
    if restored is not None:
        state = jax.device_put(restored[1], device)
    if target_steps < step:
        raise ValueError(f"target_steps ({target_steps}) is below checkpoint step {step}")
    if progress is not None:
        progress(f"{device}: {config.batch_size:,} environments, step {step:,} → {target_steps:,}")

    last_checkpoint = None
    if directory is not None and restored is None:
        last_checkpoint = save_checkpoint(directory, step, state, identity)
    while step < target_steps:
        next_save = ((step // checkpoint_every) + 1) * checkpoint_every if directory is not None else target_steps
        end = min(step + config.chunk_steps, target_steps, next_save)
        state = _advance_chunk(state, tuning, difficulty, dt, jax.device_put(jnp.int32(end - step), device))
        step = end
        if directory is not None and (step == next_save or step == target_steps):
            last_checkpoint = save_checkpoint(directory, step, state, identity)
            if progress is not None:
                progress(f"checkpoint {step:,}: {last_checkpoint}")

    jax.block_until_ready(state)
    return {
        "steps": step,
        "environment_steps": step * config.batch_size,
        "episodes_completed": int(state.episodes_completed),
        "gates_passed": int(state.gates_passed),
        "gates_missed": int(state.gates_missed),
        "restored_step": restored_step,
        "checkpoint": str(last_checkpoint) if last_checkpoint is not None else None,
    }
