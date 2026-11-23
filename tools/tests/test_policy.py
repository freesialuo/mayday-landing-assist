# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np

from mayday_jax.dataset import OBSERVATION_SIZE, generate
from mayday_jax.mppi import MPPIConfig, TeacherConfig
from mayday_jax.physics import default_tuning
from mayday_jax.policy import BCConfig, Normalizer, apply, evaluate_policy, export_policy, init_params, prepare_data, train_bc

TINY = TeacherConfig(batch_size=2, seed=5, difficulty=0.5, mppi=MPPIConfig(samples=8, horizon=6, hold=6))
BC = BCConfig(hidden=(16, 16), batch_size=32, total_steps=40, chunk_steps=10, warmup_steps=5, val_fraction=0.25)


def _dataset(tmp_path):
    cpu = jax.devices("cpu")[0]
    generate(tmp_path / "data", 2, TINY, device=cpu, tuning=default_tuning(), max_control_steps=60, progress=None)
    return prepare_data(tmp_path / "data", BC)


def test_apply_ranges():
    params = init_params(jax.random.PRNGKey(0), (8,))
    norm = Normalizer(jnp.zeros(OBSERVATION_SIZE), jnp.ones(OBSERVATION_SIZE))
    out = apply(params, norm, jax.random.normal(jax.random.PRNGKey(1), (50, OBSERVATION_SIZE)) * 10)
    assert out.shape == (50, 4)
    assert bool(jnp.all(jnp.abs(out[:, :3]) <= 1)) and bool(jnp.all((out[:, 3] >= 0) & (out[:, 3] <= 1)))


def test_bc_resume_is_bit_identical_and_exports(tmp_path):
    data = _dataset(tmp_path)
    assert data["train"][0].shape[1] == OBSERVATION_SIZE and data["val"][0].shape[0] > 0
    cpu = jax.devices("cpu")[0]
    first, _, _ = train_bc(data, tmp_path / "a", BC, cpu, checkpoint_every=20, progress=None, stop_at=20)
    resumed, params_a, norm = train_bc(data, tmp_path / "a", replace(BC, chunk_steps=7), cpu, checkpoint_every=20,
                                       progress=None)
    straight, params_b, _ = train_bc(data, tmp_path / "b", BC, cpu, checkpoint_every=15, progress=None)
    assert first["steps"] == 20 and resumed["restored_step"] == 20 and resumed["steps"] == straight["steps"] == 40
    for a, b in zip(jax.tree.leaves(params_a), jax.tree.leaves(params_b), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert np.isfinite(straight["final_val"])

    npz = export_policy(tmp_path / "policy", params_a, norm, BC)
    saved = np.load(npz)
    assert {"obs_mean", "obs_std", "w0", "b0", "w2", "b2"} <= set(saved.files)
    assert json.loads((tmp_path / "policy.json").read_text())["actions"] == ["pitch", "roll", "yaw", "throttle"]

    result = evaluate_policy(params_a, norm, replace(TINY, batch_size=2), cpu, tuning=default_tuning(), max_control_steps=50)
    assert result["control_steps"] == 50 and result["episodes"] == 2
