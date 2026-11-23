# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Phase D: student policy, behaviour cloning, and closed-loop evaluation.

The student is a small MLP over the client-visible observation of
``dataset.observe`` (28 values) that outputs the 10 Hz action:
pitch, roll, yaw stick in [-1, 1] (tanh) and absolute throttle in [0, 1]
(sigmoid). Inputs are standardised with statistics from the training split,
stored alongside the weights. Plain JAX (no network library) keeps the
forward pass trivial to port to C# in phase E (``export_policy``).

Training checkpoints hold the parameters, optimiser state, RNG key and step,
through the same Drive checkpoint API as phases B and C, and resume bit for bit.
"""

from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .checkpoints import load_checkpoint, save_checkpoint
from .dataset import HANDOVER_AGL, OBSERVATION_NAMES, OBSERVATION_SIZE, observe, shard_complete
from .mppi import TeacherConfig, _clip_actions, _hold, initial_environments, summarize
from .rollout import _demo_actions

SCHEMA = "mayday-jax-bc-v1"
ACTION_NAMES = ("pitch", "roll", "yaw", "throttle")
ALONG_INDEX = OBSERVATION_NAMES.index("along/4000")
AGL_INDEX = OBSERVATION_NAMES.index("min(agl,200)/100")


@dataclass(frozen=True)
class BCConfig:
    hidden: tuple = (256, 256)
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    total_steps: int = 20000
    chunk_steps: int = 500
    seed: int = 0
    val_fraction: float = 0.05
    action_weights: tuple = (1.0, 1.0, 0.5, 1.0)
    # Drop samples further than this past the threshold (long-landing floats). The teacher's normal
    # final descent reaches the 10 m handover a few hundred metres past the threshold, so keep that.
    max_along: float = 1000.0


class Normalizer(NamedTuple):
    mean: jax.Array
    std: jax.Array


class TrainState(NamedTuple):
    params: tuple
    opt_state: tuple
    key: jax.Array
    step: jax.Array


def init_params(key, hidden=(256, 256), inputs=OBSERVATION_SIZE, outputs=4):
    sizes = (inputs, *hidden, outputs)
    layers = []
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, sub = jax.random.split(key)
        scale = math.sqrt(2.0 / fan_in) if i < len(sizes) - 2 else 0.01
        layers.append({"w": jax.random.normal(sub, (fan_in, fan_out), jnp.float32) * scale,
                       "b": jnp.zeros((fan_out,), jnp.float32)})
    return tuple(layers)


def apply(params, norm: Normalizer, obs):
    """Observation(s) -> action(s): ReLU MLP, tanh sticks, sigmoid throttle."""
    x = (obs - norm.mean) / norm.std
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    out = x @ params[-1]["w"] + params[-1]["b"]
    return jnp.concatenate((jnp.tanh(out[..., :3]), jax.nn.sigmoid(out[..., 3:4])), axis=-1)


def _optimizer(cfg: BCConfig):
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, cfg.learning_rate, cfg.warmup_steps, max(cfg.total_steps, cfg.warmup_steps + 1), cfg.learning_rate * 0.02)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=cfg.weight_decay))


def _complete_shards(directory):
    """(manifest, observations, actions, episode) for every complete shard, one at a time."""
    found = False
    for manifest in sorted(Path(directory).glob("shard-*.json")):
        if not shard_complete(directory, int(manifest.stem.split("-")[1])):
            continue
        found = True
        with np.load(manifest.with_suffix(".npz")) as data:
            obs, act, episode = data["observations"], data["actions"], data["episode"]
        if obs.shape[1] == OBSERVATION_SIZE - 1:            # v1 shards predate the latency input: host, latency 0
            obs = np.concatenate((obs, np.zeros((obs.shape[0], 1), obs.dtype)), axis=1)
        yield manifest, obs, act, episode
    if not found:
        raise FileNotFoundError(f"no complete shards in {directory}")


def prepare_data(dataset_dir, cfg: BCConfig):
    """Load shards from one directory or a list of them, drop samples past ``cfg.max_along``,
    split by episode, compute the normaliser.

    Streams shard by shard (filter, per-shard episode split, running mean / variance) and
    concatenates once at the end, so peak host memory stays near twice the kept data
    instead of several full copies of every dataset (which exhausted a standard Colab runtime).
    """
    dirs = [dataset_dir] if isinstance(dataset_dir, (str, Path)) else list(dataset_dir)
    rng = np.random.default_rng(cfg.seed)
    train_obs, train_act, val_obs, val_act = [], [], [], []
    total = np.zeros(OBSERVATION_SIZE, np.float64)
    total_sq = np.zeros(OBSERVATION_SIZE, np.float64)
    count = episodes = val_episodes = 0
    for directory in dirs:
        for _, obs, act, episode in _complete_shards(directory):
            keep = np.isfinite(obs).all(1) & np.isfinite(act).all(1) & (obs[:, ALONG_INDEX] * 4000.0 <= cfg.max_along)
            obs, act, episode = obs[keep], act[keep], episode[keep]
            ids = np.unique(episode)
            n_val = max(1, int(len(ids) * cfg.val_fraction)) if len(ids) > 1 else 0
            is_val = np.isin(episode, rng.choice(ids, n_val, replace=False)) if n_val else np.zeros(len(episode), bool)
            kept = obs[~is_val]
            total += kept.sum(0, dtype=np.float64)
            total_sq += np.square(kept, dtype=np.float64).sum(0)
            count += len(kept)
            train_obs.append(kept); train_act.append(act[~is_val])
            val_obs.append(obs[is_val]); val_act.append(act[is_val])
            episodes += len(ids); val_episodes += n_val
            del obs, act, episode, kept
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 0.0))
    std = np.where(std < 1e-3, 1.0, std)            # constant features pass through unscaled
    norm = Normalizer(mean.astype(np.float32), std.astype(np.float32))
    manifests = [m for directory in dirs for m in sorted(Path(directory).glob("shard-*.json"))]
    fingerprint = hashlib.sha256("".join(json.loads(m.read_text())["sha256"] for m in manifests).encode()).hexdigest()

    def join(parts):
        out = np.concatenate(parts)
        parts.clear()                               # release the per-shard copies right away
        return out

    return {
        "train": (join(train_obs), join(train_act)), "val": (join(val_obs), join(val_act)),
        "norm": norm, "fingerprint": fingerprint, "episodes": int(episodes), "val_episodes": int(val_episodes),
    }


def _loss(params, norm, obs, act, weights):
    return jnp.mean(jnp.sum(weights * (apply(params, norm, obs) - act) ** 2, axis=-1))


@partial(jax.jit, static_argnames=("cfg", "steps"))
def _train_chunk(state: TrainState, norm, obs, act, cfg: BCConfig, steps: int):
    optimizer = _optimizer(cfg)
    weights = jnp.asarray(cfg.action_weights, jnp.float32)

    def body(current: TrainState, _):
        key, sub = jax.random.split(current.key)
        index = jax.random.randint(sub, (cfg.batch_size,), 0, obs.shape[0])
        loss, grads = jax.value_and_grad(_loss)(current.params, norm, obs[index], act[index], weights)
        updates, opt_state = optimizer.update(grads, current.opt_state, current.params)
        return TrainState(optax.apply_updates(current.params, updates), opt_state, key, current.step + 1), loss

    return jax.lax.scan(body, state, None, length=steps)


@jax.jit
def _val_loss(params, norm, obs, act, weights):
    per_action = jnp.mean((apply(params, norm, obs) - act) ** 2, axis=0)
    return jnp.sum(weights * per_action), per_action


def _identity(cfg: BCConfig, fingerprint):
    # chunk_steps only batches work on the device; total_steps stays because it defines the LR schedule.
    config = {k: v for k, v in asdict(cfg).items() if k != "chunk_steps"}
    return {"schema": SCHEMA, "config": config, "dataset": fingerprint, "observations": list(OBSERVATION_NAMES)}


def train_bc(data, checkpoint_dir, cfg: BCConfig, device=None, checkpoint_every=2000, progress=print, *, stop_at=None):
    """Train (or resume) behaviour cloning on ``prepare_data`` output. Returns (metrics, params, norm).

    ``cfg.total_steps`` is the schedule length; ``stop_at`` ends this session early (resume later with the same cfg).
    """
    device = device if device is not None else jax.devices()[0]
    norm = jax.device_put(data["norm"], device)
    train_obs, train_act = (jax.device_put(x, device) for x in data["train"])
    val_obs, val_act = (jax.device_put(x, device) for x in data["val"])
    key = jax.device_put(jax.random.PRNGKey(cfg.seed), device)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, cfg.hidden)
    state = TrainState(params, _optimizer(cfg).init(params), key, jnp.int32(0))
    identity = _identity(cfg, data["fingerprint"])
    directory = Path(checkpoint_dir) if checkpoint_dir is not None else None
    with jax.default_device(device):
        restored = load_checkpoint(directory, state, identity) if directory is not None else None
    if restored is not None:
        state = jax.device_put(restored[1], device)
    elif directory is not None:
        save_checkpoint(directory, 0, state, identity)
    step = int(state.step)
    weights = jnp.asarray(cfg.action_weights, jnp.float32)
    history = []
    started = time.time()
    end = cfg.total_steps if stop_at is None else min(stop_at, cfg.total_steps)
    while step < end:
        next_save = (step // checkpoint_every + 1) * checkpoint_every
        n = min(cfg.chunk_steps, end - step, next_save - step)
        state, losses = _train_chunk(state, norm, train_obs, train_act, cfg, n)
        step += n
        val, per_action = _val_loss(state.params, norm, val_obs, val_act, weights)
        history.append({"step": step, "train": float(jnp.mean(losses)), "val": float(val),
                        **{f"val_{name}": float(v) for name, v in zip(ACTION_NAMES, per_action)}})
        if directory is not None and (step == next_save or step == end):
            save_checkpoint(directory, step, state, identity)
        if progress is not None:
            h = history[-1]
            progress(f"step {step:,}: train {h['train']:.5f} · val {h['val']:.5f} · {time.time() - started:.0f}s")
    metrics = {"steps": step, "restored_step": None if restored is None else restored[0],
               "history": history, "final_val": history[-1]["val"] if history else None}
    return metrics, state.params, data["norm"]


@partial(jax.jit, static_argnames=("steps", "hold"))
def _student_chunk(envs, last, params, norm, tuning, dt, steps: int, hold: int):
    batch_observe = jax.vmap(observe)
    batch_hold = jax.vmap(_hold, in_axes=(0, 0, None, None, None))

    def body(carry, _):
        current, previous = carry
        student = apply(params, norm, batch_observe(current, previous))
        handover = current.aircraft.agl < HANDOVER_AGL
        action = jnp.where(handover[:, None], _clip_actions(_demo_actions(current)), student)
        return (batch_hold(current, action, tuning, dt, hold), action), None

    (envs, last), _ = jax.lax.scan(body, (envs, last), None, length=steps)
    return envs, last


def evaluate_policy(params, norm, config: TeacherConfig, device=None, *, tuning, max_control_steps=1800, hold=6):
    """Fly the seeded episodes of ``config`` with the student; below HANDOVER_AGL a rule controller stands in for the flare."""
    device = device if device is not None else jax.devices()[0]
    tuning = jax.device_put(tuning, device)
    params, norm = jax.device_put((params, Normalizer(*norm)), device)
    envs, _ = initial_environments(jax.device_put(jax.random.PRNGKey(config.seed), device),
                                   jax.device_put(jnp.float32(config.difficulty), device), tuning, config.batch_size)
    last = _clip_actions(_demo_actions(envs))
    dt = jax.device_put(jnp.float32(config.dt), device)
    step = 0
    while step < max_control_steps and not bool(jnp.all(envs.done)):
        n = min(100, max_control_steps - step)
        envs, last = _student_chunk(envs, last, params, norm, tuning, dt, n, hold)
        step += n
    return {"control_steps": step, **summarize(envs)}


def export_policy(path, params, norm, cfg: BCConfig, extra=None):
    """Write weights + normaliser + I/O contract as .npz and .json (input for the phase E C# port)."""
    path = Path(path)
    arrays = {"obs_mean": np.asarray(norm.mean), "obs_std": np.asarray(norm.std)}
    for i, layer in enumerate(params):
        arrays[f"w{i}"] = np.asarray(layer["w"])
        arrays[f"b{i}"] = np.asarray(layer["b"])
    np.savez(path.with_suffix(".npz"), **arrays)
    meta = {"schema": SCHEMA, "observations": list(OBSERVATION_NAMES), "actions": list(ACTION_NAMES),
            "activation": "relu", "outputs": "tanh(pitch, roll, yaw), sigmoid(throttle)",
            "control_rate_hz": 10, "handover_agl": HANDOVER_AGL, "config": asdict(cfg), **(extra or {})}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return path.with_suffix(".npz")
