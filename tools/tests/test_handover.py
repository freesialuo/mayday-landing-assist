import json

import jax
import jax.numpy as jnp
import numpy as np

from mayday_jax.dataset import OBSERVATION_NAMES, STATE_SIZE, generate, load, observe
from mayday_jax.handover import (
    DAGGER_SCHEMA, MAX_LAG, _delayed_observation, _initial_buffer, evaluate_to_handover, generate_dagger,
)
from mayday_jax.mppi import MPPIConfig, TeacherConfig, _hold, _initialize
from mayday_jax.physics import default_tuning
from mayday_jax.policy import BCConfig, prepare_data, train_bc

TINY = TeacherConfig(batch_size=2, seed=5, difficulty=0.5, mppi=MPPIConfig(samples=8, horizon=6, hold=6))
BC = BCConfig(hidden=(16, 16), batch_size=32, total_steps=20, chunk_steps=10, warmup_steps=5, val_fraction=0.25)
CPU = jax.devices("cpu")[0]


def test_delayed_observation_matches_history():
    tuning = default_tuning()
    state = _initialize(jax.random.PRNGKey(3), jnp.float32(0.5), tuning, 3, TINY.mppi)
    envs, last = state.environments, state.last_action
    lags = jnp.asarray([0, 2, MAX_LAG], jnp.int32)
    buffer = _initial_buffer(envs, last)
    fresh_history = []
    hold = jax.vmap(_hold, in_axes=(0, 0, None, None, None))
    for _ in range(MAX_LAG + 3):
        fresh_history.append(np.asarray(jax.vmap(observe)(envs, last)[:, :STATE_SIZE]))
        obs, buffer = _delayed_observation(buffer, envs, last, lags)
        for i, lag in enumerate((0, 2, MAX_LAG)):
            expected = fresh_history[max(len(fresh_history) - 1 - lag, 0)][i]
            np.testing.assert_array_equal(np.asarray(obs[i, :STATE_SIZE]), expected)
        np.testing.assert_allclose(np.asarray(obs[:, -1]), np.asarray(lags) * 0.1 / 0.6, rtol=1e-6)
        envs = hold(envs, last, tuning, jnp.float32(1 / 60), 6)


def test_jammed_gear_episodes_hold_flaps_one():
    state = _initialize(jax.random.PRNGKey(11), jnp.float32(1.0), default_tuning(), 200, TINY.mppi)
    jam = np.asarray(state.environments.disturbances.gear_jam)
    jammed = jam < 1.0
    assert 0.1 < jammed.mean() < 0.4                         # ~25 % at full difficulty
    assert (np.asarray(state.environments.flap_setting)[jammed] == 1).all()
    assert (np.asarray(state.environments.aircraft.gear_pos)[jammed] < 0.9).all()
    assert (np.asarray(state.environments.flap_setting)[~jammed] == 2).all()


def test_handover_eval_and_dagger_roundtrip(tmp_path):
    tuning = default_tuning()
    generate(tmp_path / "teacher", 2, TINY, device=CPU, tuning=tuning, max_control_steps=60, progress=None)
    data = prepare_data(tmp_path / "teacher", BC)
    _, params, norm = train_bc(data, None, BC, CPU, progress=None)

    for kwargs in ({}, {"params": params, "norm": norm}):          # teacher, then student
        report = evaluate_to_handover(TINY, CPU, tuning=tuning, max_control_steps=40, **kwargs)
        assert report["episodes"] == 2 and report["control_steps"] == 40
        assert 0.0 <= report["handover_rate"] <= 1.0

    first = generate_dagger(tmp_path / "dagger", 1, TINY, params, norm, "bc-test", device=CPU, tuning=tuning,
                            max_control_steps=60, progress=None)
    again = generate_dagger(tmp_path / "dagger", 1, TINY, params, norm, "bc-test", device=CPU, tuning=tuning,
                            max_control_steps=60, progress=None)
    assert first["written"] == 1 and again["written"] == 0
    meta = json.loads((tmp_path / "dagger" / "shard-00000.json").read_text())
    assert meta["schema"] == DAGGER_SCHEMA and meta["student_policy"] == "bc-test"
    dagger = load(tmp_path / "dagger")
    assert dagger["observations"].shape[1] == len(OBSERVATION_NAMES) and np.isfinite(dagger["actions"]).all()
    # observations carry the student's own previous action, labels are the teacher's: they differ
    last = dagger["observations"][1:, OBSERVATION_NAMES.index("last_pitch")]
    assert not np.allclose(last, dagger["actions"][:-1, 0])

    both = prepare_data([tmp_path / "teacher", tmp_path / "dagger"], BC)
    single = prepare_data(tmp_path / "teacher", BC)
    total = lambda d: d["train"][0].shape[0] + d["val"][0].shape[0]
    assert total(both) > total(single) and both["fingerprint"] != single["fingerprint"]
