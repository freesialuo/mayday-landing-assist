import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mayday_jax.physics import default_tuning
from mayday_landing.environment import Scenario, reset
from mayday_landing.experiment import (
    BRANCHES, FORK_SECONDS, Checkpoints, Labels, RollState, _assert_shared_prefix, _fork_permanent_loss,
    _initialize, _match, _pack, make_cases,
)
from mayday_landing.teacher import ACTION_LOW, PlannerConfig, causal_state, plan_batch


def test_unseen_event_schedules_do_not_change_teacher_actions():
    key = jax.random.PRNGKey(45)
    tuning = default_tuning()
    regular = Scenario()
    future_fault = regular._replace(failure_time=0.1, failure_side=1, permanent=True,
                                     repair_start=0.2, repair_interrupt=0.3, repair_resume=0.4,
                                     repair_duration=0.05, go_around_time=0.15, fire_intensity=0.65,
                                     engine_shutdown_time=0.12, extinguish_time=0.22, extinguish_duration=0.2)
    states = jax.tree.map(lambda *leaves: jnp.stack(leaves),
                          reset(key, tuning, regular), reset(key, tuning, future_fault))
    causal = jax.vmap(causal_state)(states)
    for leaf in jax.tree.leaves(causal):
        np.testing.assert_array_equal(leaf[0], leaf[1])
    actions = np.asarray(plan_batch(states, jnp.stack((key, key)), tuning, PlannerConfig(samples=3, horizon=3)))
    np.testing.assert_array_equal(actions[0], actions[1])
    assert np.all(actions >= np.asarray(ACTION_LOW) - 1e-7)
    assert np.all(actions <= 1.0)


def test_fork_requires_shared_world_and_keeps_histories():
    cases = make_cases([105, 108])
    world = _initialize(cases.keys, cases.scenario, default_tuning())
    memory = jnp.zeros((len(cases.names), 2), jnp.float32)
    state = RollState(world, memory, jnp.zeros_like(memory), memory, memory)
    _assert_shared_prefix(state, cases.names)
    broken = state._replace(memory=memory.at[2, 0].set(1.0))
    with pytest.raises(RuntimeError, match="diverged before"):
        _assert_shared_prefix(broken, cases.names)
    after = _fork_permanent_loss(world, cases.names)
    np.testing.assert_array_equal(after.observation_history, world.observation_history)
    np.testing.assert_array_equal(after.command_history, world.command_history)
    assert np.asarray(after.scenario.permanent).tolist() == (cases.names == "permanent").tolist()
    np.testing.assert_array_equal(np.asarray(cases.scenario.repair_start)[cases.names == "repaired"], FORK_SECONDS)
    assert tuple(cases.names[:len(BRANCHES)]) == BRANCHES


def test_native_weather_covers_each_delay_and_weather_pair():
    cases = make_cases(range(880000, 880024), packet_stress=True, native_weather=True)
    lag = np.asarray(cases.scenario.observation_lag)
    weather = np.asarray(cases.scenario.weather_kind)
    assert len(set(zip(lag.tolist(), weather.tolist(), strict=True))) == 24
    for offset in range(0, len(lag), len(BRANCHES)):
        assert len(set(weather[offset:offset + len(BRANCHES)])) == 1
    np.testing.assert_array_equal(np.asarray(cases.scenario.observation_jitter), lag >= 3)


def test_dataset_keeps_episode_runs_and_omits_absorbing_steps():
    observations = np.arange(12).reshape(3, 2, 2)
    actions = 2 * observations
    phases = np.array([[0, 1], [1, 2], [4, 3]])
    valid = np.array([[True, True], [True, True], [False, True]])
    labels = _pack([(observations[:2], actions[:2], phases[:2], valid[:2]),
                    (observations[2:], actions[2:], phases[2:], valid[2:])], batch_size=2, offset=7)
    np.testing.assert_array_equal(labels.episode_ids, [7, 7, 8, 8, 8])
    np.testing.assert_array_equal(labels.actions, 2 * labels.observations)
    np.testing.assert_array_equal(labels.phases, [0, 1, 1, 2, 3])


def test_outcome_comparison_is_by_seed_and_branch():
    left = [{"seed": 1, "branch": "repaired", "success": True},
            {"seed": 2, "branch": "repaired", "success": False}]
    right = [dict(left[1], success=True), dict(left[0])]
    assert _match(left, right) == {"pairs": 2, "left_only_success": 0, "right_only_success": 1,
                                   "both_success": 1, "both_failed": 0}
    with pytest.raises(ValueError, match="identical unique"):
        _match(left, right[:1])


def test_checkpoints_require_same_identity_and_valid_payload(tmp_path):
    identity = {"seed": 55, "model": {"sizes": (3, 5)}}
    arrays = Labels(np.arange(8), np.ones((8, 5)), np.zeros(8), np.ones(8))
    saves = Checkpoints(tmp_path, identity)
    saves.write("train-0000", [{"seed": 55}], arrays)
    other = Checkpoints(tmp_path, identity)
    rows, restored = other.read("train-0000", True)
    assert rows == [{"seed": 55}]
    for before, after in zip(arrays, restored, strict=True):
        np.testing.assert_array_equal(after, before)
    with pytest.raises(ValueError, match="source, device, tuning or run configuration differs"):
        Checkpoints(tmp_path, dict(identity, seed=56))
    (tmp_path / "train-0000.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        other.read("train-0000", True)
