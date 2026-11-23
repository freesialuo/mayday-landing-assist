import json

import numpy as np
import pytest

from mayday_landing.features import NAMES
from mayday_landing.model import (
    ACTION_NAMES,
    DYNAMICS_TARGETS,
    PHASE_NAMES,
    SCHEMA,
    Policy,
    TrainingConfig,
    _dynamics_data,
    _episode_bounds,
    _normalization,
    _windows,
    train,
)
from mayday_landing.residual_model import (
    GuidedPolicy,
    apply_residual_actions,
    encode_residual_actions,
    unwrap_residual,
    wrap_prior,
    wrap_residual,
)


def parity_fixture(architecture):
    """Small deterministic policy also constructed by the C# parity test."""
    def dense(inputs, outputs, scale, salt):
        return {"w": [[scale * (((row + 3) * (col + 1) + salt) % 17 - 8)
                       for col in range(outputs)] for row in range(inputs)],
                "b": [scale * ((col % 5) - 2) for col in range(outputs)]}

    size = len(NAMES)
    head = 3 if architecture == "mlp" else 6
    weights = {"encoder": dense(size + len(PHASE_NAMES), 3, 0.0125, 3),
               "air": dense(head, 5, 0.09, 11), "ground": dense(head, 5, 0.065, 8)}
    if architecture == "mlp":
        weights["hidden"] = dense(3, 3, 0.07, 10)
    else:
        weights["gru_input"] = dense(3, 9, 0.08, 4)
        weights["gru_state"] = dense(3, 9, 0.08, 1)
    return {"schema": SCHEMA, "architecture": architecture, "feature_names": list(NAMES),
            "actions": list(ACTION_NAMES), "phases": list(PHASE_NAMES), "hidden_size": 3,
            "mean": [(index % 7 - 3) * 0.05 for index in range(size)],
            "std": [1.0 + (index % 5) * 0.125 for index in range(size)], "weights": weights}


def parity_sequence():
    observations = np.array([[(index * (step + 2) % 19 - 9) * 0.1 for index in range(len(NAMES))]
                             for step in range(5)], np.float32)
    return observations, np.array([0, 1, 2, 3, 4])


@pytest.mark.parametrize("architecture", ["mlp", "gru"])
def test_batch_and_replay_match_individual_control_and_reset(architecture):
    observations, phases = parity_sequence()
    policy = Policy(parity_fixture(architecture))
    expected = np.stack([policy.predict(obs, phase) for obs, phase in zip(observations, phases, strict=True)])
    assert np.all(np.abs(expected[:, :3]) <= 1)
    assert np.all((expected[:, 3:] >= 0) & (expected[:, 3:] <= 1))
    policy.reset()
    np.testing.assert_allclose(policy.predict(observations[0], phases[0]), expected[0], atol=1e-7)
    np.testing.assert_allclose(policy.predict_many(observations, phases, np.zeros(5)), expected, atol=1e-7)

    carry = None
    batched = []
    for index, phase in enumerate(phases):
        action, carry = policy.predict_batch([observations[index], observations[-1 - index]], [phase, phase], carry)
        batched.append(action)
    other_policy = Policy(parity_fixture(architecture))
    other = np.stack([other_policy.predict(obs, phase)
                      for obs, phase in zip(observations[::-1], phases, strict=True)])
    np.testing.assert_allclose(batched, np.stack((expected, other), axis=1), atol=1e-7)

    reset_rows = policy.predict_many(observations, phases, np.arange(len(phases)))
    assert (np.max(np.abs(reset_rows - expected)) > 1e-3) == (architecture == "gru")
    with pytest.raises(ValueError):
        policy.predict_batch(observations, np.full(5, 5))


def test_training_windows_preserve_episode_boundaries():
    episodes = np.array([2, 2, 7, 7, 7])
    indices, valid, supervise, transitions = _windows(
        np.array([1, 2]), _episode_bounds(episodes), TrainingConfig(burn_in=1, sequence_length=3))
    np.testing.assert_array_equal(indices, [[0, 2], [1, 2], [1, 3], [1, 4]])
    np.testing.assert_array_equal(valid, [[True, False], [True, True], [False, True], [False, True]])
    np.testing.assert_array_equal(supervise, [[False, False], [True, True], [False, True], [False, True]])
    np.testing.assert_array_equal(transitions, [[True, False], [False, True], [False, True], [False, False]])
    with pytest.raises(ValueError, match="contiguous"):
        _episode_bounds(np.array([0, 1, 0]))


def test_dynamics_labels_use_only_physical_signals_and_flag_unknown_thrust():
    names = ("left_valid", "sink", "right_thrust", "target_along", "yaw_rate", "left_thrust", "right_valid")
    raw = np.array([[0, 2, 3, 999, 5, 6, 1], [1, 4, 2, -999, -3, 9, 0]], dtype=np.float32)
    signals, reliable = _dynamics_data(raw, raw / 2, names)
    np.testing.assert_array_equal(signals, [[3, 1.5, 2.5, 1], [4.5, 1, -1.5, 2]])
    np.testing.assert_array_equal(reliable, [[False, True, True, True], [True, False, True, True]])


def test_normalization_uses_physical_scales_without_discarding_rare_binary_values():
    names = ("runway_length", "runway_half_width", "visibility", "agl", "left_rpm", "observation_age", "stall_warning", "custom")
    raw = np.tile([1500.0, 30.0, 8000.0, 120.0, 0.6, 0.1, 0.0, 73.0], (50, 1)).astype(np.float32)
    raw[-1] += [0.25, 0.01, 1.0, 0.25, 0.02, 0.01, 1.0, 10.0]
    _, std, ignored = _normalization(raw, names)
    np.testing.assert_allclose(std, [200, 5, 1000, 8, 0.1, 0.05, 1, 1.4], rtol=1e-6)
    np.testing.assert_array_equal(ignored, [True, True, True, False, False, False, False, False])


@pytest.mark.parametrize("architecture", ["mlp", "gru"])
def test_training_improves_fit_and_roundtrips_deployable_policy(architecture):
    rng = np.random.default_rng(91)
    feature_names = ("x", "y", *DYNAMICS_TARGETS, "left_valid", "right_valid", "runway_length")
    observations = rng.normal(size=(120, len(feature_names) - 1)).astype(np.float32)
    observations[:, -2:] = rng.integers(0, 2, size=(120, 2))
    observations = np.column_stack((observations, np.full(120, 1500.0, dtype=np.float32)))
    phases = np.tile(np.repeat([0, 1, 2, 3, 4], 4), 6)
    episode_ids = np.repeat(np.arange(6), 20)
    x, y = np.tanh(observations[:, :2]).T
    actions = np.stack((0.4 * x, 0.3 * y, -0.4 * x, 0.3 + 0.15 * y, 0.1 + 0.5 * (phases == 3)), axis=1)
    kwargs = {"architecture": architecture, "feature_names": feature_names, "seed": 15}
    initial = train(observations, actions, phases, episode_ids, **kwargs,
                    config=TrainingConfig(steps=1, hidden_size=8, batch_size=16, sequence_length=5, burn_in=2))
    shifted = observations.copy()
    shifted[:, -1] = np.linspace(930.0, 2400.0, len(shifted))
    trained = train(shifted, actions, phases, episode_ids, **kwargs, initial_payload=initial,
                    config=TrainingConfig(steps=75, hidden_size=8, batch_size=16, sequence_length=5, burn_in=2,
                                          learning_rate=0.01))
    expected = Policy(initial).predict_many(observations, phases, episode_ids)
    actual = Policy(json.loads(json.dumps(trained, allow_nan=False))).predict_many(observations, phases, episode_ids)
    assert np.mean((actual - actions) ** 2) < 0.35 * np.mean((expected - actions) ** 2)
    assert trained["mean"] == initial["mean"]
    assert trained["std"] == initial["std"]
    assert trained["training"]["ignored_features"] == ["runway_length"]
    np.testing.assert_array_equal(Policy(trained).predict_many(shifted, phases, episode_ids), actual)
    assert "dynamics" not in trained["weights"]
    assert trained["training"]["dynamics_targets"] == (list(DYNAMICS_TARGETS) if architecture == "gru" else [])
    assert (trained["training"]["loss"][-1]["dynamics_loss"] > 0) == (architecture == "gru")


def test_residual_label_projection_and_discrete_brake_agree_in_jax_and_numpy():
    import jax
    import jax.numpy as jnp

    prior = [0.99, -0.98, 0.99, 0.995, 1.0]
    actions = [[1, -1, -1, 1, 0], [1, -1, -1, 1, 0]]
    phases = [2, 3]
    labels = encode_residual_actions(actions, prior)
    np.testing.assert_allclose(labels[0], [0.1, -2 / 15, -1.0, 0.5 + 0.005 / 0.24, 0.0], atol=1e-6)
    actual = apply_residual_actions(labels, prior, phases)
    np.testing.assert_allclose(actual, [[1, -1, 0.87, 1, 0], [1, -1, 0.87, 1, 1]], atol=1e-6)
    jit_apply = jax.jit(lambda action, guidance, phase: apply_residual_actions(
        encode_residual_actions(action, guidance, xp=jnp), guidance, phase, xp=jnp))
    np.testing.assert_allclose(jit_apply(np.asarray(actions), np.asarray(prior), np.asarray(phases)), actual, atol=1e-6)


def test_guided_artifacts_replay_and_reject_mismatched_tuning():
    payload = parity_fixture("gru")
    speeds = [32, 29, 26]
    residual = wrap_residual(payload, stall_speeds=speeds)
    assert unwrap_residual(json.loads(json.dumps(residual))) == payload
    with pytest.raises(ValueError, match="stall speeds"):
        unwrap_residual(dict(residual, stall_speeds=[32, 0, 26]))
    with pytest.raises(ValueError, match="schema"):
        unwrap_residual(dict(residual, prior="unsupported"))

    prior = GuidedPolicy(wrap_prior(stall_speeds=speeds))
    guided = GuidedPolicy(residual)
    learner = Policy(payload)
    observations, phases = parity_sequence()
    for obs, phase in zip(observations, phases, strict=True):
        expected = apply_residual_actions(learner.predict(obs, phase), prior.predict(obs, phase), phase)
        np.testing.assert_allclose(guided.predict(obs, phase), expected, atol=1e-7)
    guided.reset()
    learner.reset()
    expected = apply_residual_actions(learner.predict(observations[-1], 4), prior.predict(observations[-1], 4), 4)
    np.testing.assert_allclose(guided.predict(observations[-1], 4), expected, atol=1e-7)


def test_gru_privileged_auxiliary_head_trains_and_is_not_deployed():
    import numpy as np
    from mayday_landing import model
    from mayday_landing.features import NAMES
    rng = np.random.default_rng(0)
    rows = 64
    obs = rng.normal(size=(rows, len(NAMES))).astype(np.float32)
    actions = np.clip(rng.normal(scale=0.2, size=(rows, 5)), -1, 1).astype(np.float32)
    actions[:, 3:] = 0.5
    payload = model.train(obs, actions, np.zeros(rows, np.int32), np.repeat(np.arange(4), 16),
                          architecture="gru", feature_names=NAMES, privileged=rng.normal(size=(rows, 10)),
                          config=model.TrainingConfig(steps=3, batch_size=4, hidden_size=8, sequence_length=4, burn_in=1))
    assert payload["training"]["dynamics_targets"] == "privileged"
    assert "dynamics" not in payload["weights"]
