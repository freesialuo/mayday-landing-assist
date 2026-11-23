# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Behavior cloning and portable NumPy inference for the landing policies."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


SCHEMA = "mayday-landing-policy-v1"
ACTION_NAMES = ("pitch", "roll", "yaw", "throttle", "brake")
PHASE_NAMES = ("capture", "final", "flare", "rollout", "goaround")
ROLLOUT = PHASE_NAMES.index("rollout")
DYNAMICS_TARGETS = ("left_thrust", "right_thrust", "yaw_rate", "sink")

# Minimum useful spread in each input's native units. A binary transition is
# at most one normalized unit even if the rare class occurs in only one flight.
NORMALIZATION_FLOORS = {
    "along": 100.0, "cross": 30.0, "agl": 8.0, "speed": 3.0, "ground_speed": 3.0, "sink": 0.7,
    "pitch": 1.0, "roll": 3.0, "pitch_rate": 1.0, "roll_rate": 2.0, "yaw_rate": 0.5,
    "track_sin": 0.05, "track_cos": 0.05, "heading_sin": 0.05, "heading_cos": 0.05,
    "wind_along": 1.0, "wind_cross": 1.0,
    "runway_length": 200.0, "runway_half_width": 5.0, "next_gate_distance": 50.0,
    "gate_height_error": 5.0, "gate_half_width": 2.0, "gate_half_height": 2.0, "next_gate_index": 1.0,
    "gear": 0.1, "flap": 0.2, "stall_warning": 1.0, "on_ground": 1.0, "throttle": 0.05,
    "left_rpm": 0.1, "left_thrust": 0.1, "left_health": 0.05, "left_fire": 0.1,
    "left_repair": 0.1, "left_state": 1.0, "left_valid": 1.0,
    "right_rpm": 0.1, "right_thrust": 0.1, "right_health": 0.05, "right_fire": 0.1,
    "right_repair": 0.1, "right_state": 1.0, "right_valid": 1.0,
    "rain": 0.1, "visibility": 1000.0, "turbulence": 0.1, "ice": 0.1, "observation_age": 0.05,
    "other_stick": 0.05, "other_known": 1.0, "last_pitch": 0.05, "last_roll": 0.05,
    "last_yaw": 0.05, "last_throttle": 0.05, "last_brake": 1.0, "target_along": 100.0, "target_cross": 30.0,
}


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 1000
    batch_size: int = 128
    hidden_size: int = 48
    sequence_length: int = 24
    burn_in: int = 8
    learning_rate: float = 1e-3
    auxiliary_weight: float = 0.1
    action_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    # Hard-example sampling: anchor probability scales with
    # 1 + hard_weight * |label - label_center|^2 over the four flight axes, so
    # rare large corrections (gusts at flare, ring recovery) are not drowned by
    # the near-zero labels of routine flight. 0 keeps the phase-only sampler.
    hard_weight: float = 0.0
    label_center: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)


def _dense(layer, x):
    return x @ layer["w"] + layer["b"]


def _sigmoid(values, xp):
    return 1.0 / (1.0 + xp.exp(-xp.clip(values, -80.0, 80.0)))


def _encode(weights, observations, phases, xp):
    one_hot = (phases[..., None] == xp.arange(len(PHASE_NAMES))).astype(observations.dtype)
    return xp.tanh(_dense(weights["encoder"], xp.concatenate((observations, one_hot), axis=-1)))


def _gru(weights, encoded, carry, xp):
    xi_reset, xi_update, xi_new = xp.split(_dense(weights["gru_input"], encoded), 3, axis=-1)
    hi_reset, hi_update, hi_new = xp.split(_dense(weights["gru_state"], carry), 3, axis=-1)
    reset = _sigmoid(xi_reset + hi_reset, xp)
    update = _sigmoid(xi_update + hi_update, xp)
    candidate = xp.tanh(xi_new + reset * hi_new)
    return (1.0 - update) * candidate + update * carry


def _actions(weights, latent, phases, xp):
    air = _dense(weights["air"], latent)
    ground = _dense(weights["ground"], latent)
    logits = xp.where((phases == ROLLOUT)[..., None], ground, air)
    return xp.concatenate((xp.tanh(logits[..., :3]), _sigmoid(logits[..., 3:], xp)), axis=-1)


def _validate_phases(phases, length):
    phases = np.asarray(phases)
    if (phases.shape != (length,) or phases.dtype.kind not in "iu"
            or ((phases < 0) | (phases >= len(PHASE_NAMES))).any()):
        raise ValueError("phases must contain one valid integer phase per observation")
    return phases.astype(np.int32, copy=False)


def _float_array(values, shape, label):
    result = np.asarray(values, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite and have shape {shape}")
    return result


def _layout(architecture, features, hidden):
    shapes = {"encoder": (features + len(PHASE_NAMES), hidden)}
    if architecture == "mlp":
        shapes["hidden"] = (hidden, hidden)
        head_width = hidden
    else:
        shapes["gru_input"] = shapes["gru_state"] = (hidden, 3 * hidden)
        head_width = 2 * hidden
    shapes["air"] = shapes["ground"] = (head_width, len(ACTION_NAMES))
    return shapes


class Policy:
    """One recurrent session, or independent carries supplied by a batched caller."""

    mode = "direct"

    def __init__(self, payload):
        self.architecture = payload["architecture"]
        self.feature_names = tuple(payload["feature_names"])
        if (payload["schema"] != SCHEMA or self.architecture not in ("mlp", "gru")
                or tuple(payload["actions"]) != ACTION_NAMES or tuple(payload["phases"]) != PHASE_NAMES
                or not self.feature_names or not all(isinstance(name, str) and name for name in self.feature_names)
                or len(set(self.feature_names)) != len(self.feature_names)):
            raise ValueError("invalid landing policy schema")
        self.hidden_size = payload["hidden_size"]
        if type(self.hidden_size) is not int or not 1 <= self.hidden_size <= 256:
            raise ValueError("invalid landing policy hidden size")
        self.mean = _float_array(payload["mean"], (len(self.feature_names),), "mean")
        self.std = _float_array(payload["std"], self.mean.shape, "std")
        if np.any(self.std <= 0):
            raise ValueError("standard deviations must be positive")
        self.weights = {}
        for name, shape in _layout(self.architecture, len(self.feature_names), self.hidden_size).items():
            layer = payload["weights"][name]
            self.weights[name] = {"w": _float_array(layer["w"], shape, f"{name}.w"),
                                  "b": _float_array(layer["b"], (shape[1],), f"{name}.b")}
        self.reset()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def reset(self):
        self._carry = None

    def predict(self, observation, phase):
        observation = _float_array(observation, (len(self.feature_names),), "observation")
        actions, self._carry = self.predict_batch(observation[None, :], [phase], self._carry)
        return actions[0]

    def predict_batch(self, observations, phases, carry=None):
        """Return one action per row and a carry to reuse for the same batch rows."""
        observations = _float_array(observations, (len(observations), len(self.feature_names)), "observations")
        phases = _validate_phases(phases, len(observations))
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = (observations - self.mean) / self.std
        if not np.isfinite(normalized).all():
            raise ValueError("observations overflowed standardization")
        encoded = _encode(self.weights, normalized, phases, np)
        if self.architecture == "mlp":
            latent = np.tanh(_dense(self.weights["hidden"], encoded))
        else:
            shape = (len(observations), self.hidden_size)
            carry = np.zeros(shape, np.float32) if carry is None else _float_array(carry, shape, "carry")
            carry = _gru(self.weights, encoded, carry, np)
            latent = np.concatenate((encoded, carry), axis=-1)
        actions = _actions(self.weights, latent, phases, np)
        if not np.isfinite(actions).all():
            raise ValueError("landing policy produced a non-finite action")
        return actions, carry

    def predict_many(self, observations, phases, episode_ids):
        """Replay contiguous episodes without mutating the policy's session."""
        episode_ids = np.asarray(episode_ids)
        if episode_ids.shape != (len(observations),):
            raise ValueError("expected one episode ID per observation")
        if self.architecture == "mlp":
            return self.predict_batch(observations, phases)[0]
        actions = np.empty((len(observations), len(ACTION_NAMES)), np.float32)
        carry = None
        for index, (observation, phase, episode) in enumerate(zip(observations, phases, episode_ids, strict=True)):
            if index == 0 or episode != episode_ids[index - 1]:
                carry = None
            action, carry = self.predict_batch(np.asarray(observation)[None, :], [phase], carry)
            actions[index] = action[0]
        return actions


def save_policy(payload, path):
    """Write the self-contained deployable JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return path


def _episode_bounds(episode_ids):
    starts = np.concatenate(([0], np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1))
    ends = np.concatenate((starts[1:], [len(episode_ids)]))
    if len(np.unique(episode_ids[starts])) != len(starts):
        raise ValueError("episode rows must be contiguous")
    return np.repeat(starts, ends - starts), np.repeat(ends, ends - starts)


def _windows(anchors, bounds, config):
    starts, ends = bounds
    offsets = np.arange(-config.burn_in, config.sequence_length)
    indices = anchors[None, :] + offsets[:, None]
    first, last = starts[anchors][None, :], ends[anchors][None, :] - 1
    valid = (indices >= first) & (indices <= last)
    return (np.clip(indices, first, last), valid,
            valid & (offsets[:, None] >= 0), valid & (indices < last))


def _dynamics_data(observations, normalized, feature_names):
    required = (*DYNAMICS_TARGETS, "left_valid", "right_valid")
    missing = set(required).difference(feature_names)
    if missing:
        raise ValueError(f"GRU dynamics supervision needs features: {', '.join(sorted(missing))}")

    signals = normalized[:, [feature_names.index(name) for name in DYNAMICS_TARGETS]]
    reliable = np.ones(signals.shape, dtype=bool)
    reliable[:, :2] = observations[:, [feature_names.index("left_valid"), feature_names.index("right_valid")]] > 0.5
    return signals, reliable


def _normalization(observations, feature_names):
    mean = observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    spread = observations.std(axis=0, dtype=np.float64).astype(np.float32)
    floor = np.asarray([NORMALIZATION_FLOORS.get(name, 1e-3) for name in feature_names], dtype=np.float32)
    # If a signal never varied at useful resolution, predictions should be
    # invariant to its value. Otherwise the random input weights persist.
    ignored = spread <= floor * 1e-3
    return mean, np.maximum(spread, floor), ignored


def train(observations, actions, phases, episode_ids, *, architecture, feature_names, output_path=None,
          seed=0, config=TrainingConfig(), initial_payload=None, privileged=None):
    """Fit on contiguous labeled trajectories. Warm starts reuse weights and normalization.

    ``privileged`` (rows x k, twin-only) switches the GRU auxiliary head from
    one-step observation deltas to regressing these standardized targets from
    the recurrent state alone (asymmetric/RMA-style). The head is never deployed.
    """
    import jax
    import jax.numpy as jnp
    import optax

    feature_names = tuple(feature_names)
    if (architecture not in ("mlp", "gru") or not feature_names
            or not all(isinstance(name, str) and name for name in feature_names)
            or len(set(feature_names)) != len(feature_names)):
        raise ValueError("invalid architecture or feature names")
    if (not 1 <= config.hidden_size <= 256 or config.steps < 1 or config.batch_size < 1
            or config.sequence_length < 1 or config.burn_in < 0
            or not np.isfinite(config.learning_rate) or config.learning_rate <= 0
            or not np.isfinite(config.auxiliary_weight) or config.auxiliary_weight < 0):
        raise ValueError("invalid training configuration")
    action_weights = _float_array(config.action_weights, (len(ACTION_NAMES),), "action weights")
    if np.any(action_weights < 0) or not np.any(action_weights > 0):
        raise ValueError("action weights must be nonnegative with at least one positive value")
    count = len(observations)
    observations = _float_array(observations, (count, len(feature_names)), "observations")
    actions = _float_array(actions, (count, len(ACTION_NAMES)), "actions")
    phases = _validate_phases(phases, count)
    episode_ids = np.asarray(episode_ids)
    if count == 0 or episode_ids.shape != (count,) or episode_ids.dtype.kind not in "iuU":
        raise ValueError("training needs nonempty observations and one episode ID per row")
    if (np.any(np.abs(actions[:, :3]) > 1) or np.any(actions[:, 3:] < 0) or np.any(actions[:, 3:] > 1)):
        raise ValueError("training actions exceed their control ranges")
    bounds = _episode_bounds(episode_ids)
    rng = np.random.default_rng(seed)
    shapes = _layout(architecture, len(feature_names), config.hidden_size)
    use_dynamics = architecture == "gru" and config.auxiliary_weight > 0
    use_privileged = use_dynamics and privileged is not None
    if use_privileged:
        privileged = _float_array(privileged, (len(observations), np.shape(privileged)[1]), "privileged targets")
        privileged = (privileged - privileged.mean(0)) / np.maximum(privileged.std(0), 1e-3)
        shapes["dynamics"] = (config.hidden_size, privileged.shape[1])
    elif use_dynamics:
        shapes["dynamics"] = (config.hidden_size + len(ACTION_NAMES), len(DYNAMICS_TARGETS))
    params = {
        name: {"w": jnp.asarray(rng.normal(0, width**-0.5, size=(width, outputs)), dtype=jnp.float32),
               "b": jnp.zeros(outputs, dtype=jnp.float32)}
        for name, (width, outputs) in shapes.items()
    }
    if initial_payload is None:
        mean, std, ignored = _normalization(observations, feature_names)
    else:
        initial = Policy(initial_payload)
        if (initial.architecture != architecture or initial.hidden_size != config.hidden_size
                or initial.feature_names != feature_names):
            raise ValueError("warm-start policy does not match architecture, width and features")
        mean, std = initial.mean, initial.std
        ignored_names = initial_payload.get("training", {}).get("ignored_features", ())
        ignored = np.isin(feature_names, ignored_names)
        params.update(jax.tree.map(jnp.asarray, initial.weights))
    normalized = _float_array((observations - mean) / std, observations.shape, "standardized observations")
    if use_dynamics and not use_privileged:
        dynamics_signals, dynamics_reliable = _dynamics_data(observations, normalized, feature_names)
    normalized[:, ignored] = 0.0
    encoder_rows = np.concatenate((~ignored, np.ones(len(PHASE_NAMES), dtype=bool)))
    params["encoder"]["w"] = params["encoder"]["w"] * encoder_rows[:, None]
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(config.learning_rate))
    opt_state = optimizer.init(params)

    def mlp_loss(weights, obs, target, phase):
        encoded = _encode(weights, obs, phase, jnp)
        latent = jnp.tanh(_dense(weights["hidden"], encoded))
        behavior_loss = jnp.sum(action_weights * (_actions(weights, latent, phase, jnp) - target) ** 2, axis=-1).mean()
        return behavior_loss, (behavior_loss, jnp.float32(0.0))

    def gru_loss(weights, obs, target, phase, valid, supervise, delta, dynamics_mask):
        encoded = _encode(weights, obs, phase, jnp)

        def step(carry, inputs):
            current, mask = inputs
            stored = _gru(weights, current, carry, jnp)
            carry = jnp.where(mask[:, None], stored, carry)
            return carry, carry

        initial = jnp.zeros((obs.shape[1], config.hidden_size), dtype=obs.dtype)
        _, carry = jax.lax.scan(step, initial, (encoded, valid))
        latent = jnp.concatenate((encoded, carry), axis=-1)
        action_error = jnp.sum(action_weights * (_actions(weights, latent, phase, jnp) - target) ** 2, axis=-1)
        behavior_loss = jnp.sum(jnp.where(supervise, action_error, 0.0)) / jnp.maximum(1, supervise.sum())
        dynamics_loss = jnp.float32(0.0)
        if use_dynamics:
            features = carry if use_privileged else jnp.concatenate((carry, target), axis=-1)
            changes = _dense(weights["dynamics"], features)
            dynamics_loss = jnp.sum(jnp.where(dynamics_mask, (changes - delta) ** 2, 0.0)) / jnp.maximum(
                1, dynamics_mask.sum())
        return behavior_loss + config.auxiliary_weight * dynamics_loss, (behavior_loss, dynamics_loss)

    loss = mlp_loss if architecture == "mlp" else gru_loss

    @jax.jit
    def update(weights, state, batch):
        (_, components), gradients = jax.value_and_grad(loss, has_aux=True)(weights, *batch)
        updates, state = optimizer.update(gradients, state, weights)
        return optax.apply_updates(weights, updates), state, components

    # Reserve sampling capacity for brief phases such as flare and rollout.
    phase_counts = np.bincount(phases, minlength=len(PHASE_NAMES))
    probabilities = phase_counts[phases].astype(np.float64) ** -0.5
    if config.hard_weight > 0.0:
        offset = np.asarray(actions[:, :4], np.float64) - np.asarray(config.label_center[:4], np.float64)
        probabilities *= 1.0 + config.hard_weight * np.sum(offset ** 2, axis=1)
    cumulative = np.cumsum(probabilities)
    cumulative /= cumulative[-1]
    history = []
    for step in range(config.steps):
        anchors = np.searchsorted(cumulative, rng.random(config.batch_size), side="right")
        if architecture == "mlp":
            batch = (normalized[anchors], actions[anchors], phases[anchors])
        else:
            indices, valid, supervise, transition = _windows(anchors, bounds, config)
            delta, dynamics_mask = None, None
            if use_privileged:
                delta = privileged[indices]
                dynamics_mask = np.broadcast_to(supervise[..., None], delta.shape)
            elif use_dynamics:
                next_indices = np.minimum(indices + 1, bounds[1][indices] - 1)
                delta = dynamics_signals[next_indices] - dynamics_signals[indices]
                dynamics_mask = ((transition & supervise)[..., None]
                                 & dynamics_reliable[indices] & dynamics_reliable[next_indices])
            batch = (normalized[indices], actions[indices], phases[indices], valid, supervise, delta, dynamics_mask)
        params, opt_state, metrics = update(params, opt_state, batch)
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == config.steps:
            behavior_loss, dynamics_loss = map(float, metrics)
            history.append({"step": step + 1, "behavior_loss": behavior_loss, "dynamics_loss": dynamics_loss})

    deployable = {name: params[name] for name in _layout(architecture, len(feature_names), config.hidden_size)}
    payload = {
        "schema": SCHEMA, "architecture": architecture,
        "feature_names": list(feature_names), "actions": list(ACTION_NAMES), "phases": list(PHASE_NAMES),
        "hidden_size": config.hidden_size, "mean": mean.tolist(), "std": std.tolist(),
        "weights": jax.tree.map(lambda item: np.asarray(item).tolist(), deployable),
        "training": {"config": asdict(config), "seed": seed, "examples": count,
                     "episodes": int(np.count_nonzero(np.arange(count) == bounds[0])),
                     "ignored_features": [name for name, unused in zip(feature_names, ignored, strict=True) if unused],
                     "dynamics_targets": ("privileged" if use_privileged else list(DYNAMICS_TARGETS)) if use_dynamics else [],
                     "phase_examples": dict(zip(PHASE_NAMES, phase_counts.tolist(), strict=True)), "loss": history},
    }
    Policy(payload)
    if output_path is not None:
        save_policy(payload, output_path)
    return payload
