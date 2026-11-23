# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import jax
import jax.numpy as jnp
import numpy as np

from mayday_jax.dataset import OBSERVATION_NAMES, OBSERVATION_SIZE, generate, load, observe, shard_complete
from mayday_jax.mppi import MPPIConfig, TeacherConfig, _initialize
from mayday_jax.physics import default_tuning

TINY = TeacherConfig(batch_size=2, seed=5, difficulty=0.5, mppi=MPPIConfig(samples=8, horizon=6, hold=6))


def test_observation_is_finite_and_documented():
    state = _initialize(jax.random.PRNGKey(0), jnp.float32(1.0), default_tuning(), 4, TINY.mppi)
    obs = jax.vmap(observe)(state.environments, state.last_action)
    assert obs.shape == (4, OBSERVATION_SIZE) == (4, len(OBSERVATION_NAMES))
    assert bool(jnp.all(jnp.isfinite(obs)))
    has_next = obs[:, OBSERVATION_NAMES.index("has_next_ring")]
    assert bool(jnp.all(has_next == 1.0))                       # episodes start before the farthest ring
    # the farthest ring is next at the start: ring-relative errors stay within a few ring units
    assert bool(jnp.all(jnp.abs(obs[:, OBSERVATION_NAMES.index("next_ring_lateral_units")]) < 3.0))


def test_shards_resume_and_rebuild_corrupted(tmp_path):
    tuning = default_tuning()
    first = generate(tmp_path, 2, TINY, device=jax.devices("cpu")[0], tuning=tuning, max_control_steps=60, progress=None)
    assert first == {"shards": 2, "written": 2}
    assert shard_complete(tmp_path, 0) and shard_complete(tmp_path, 1)
    again = generate(tmp_path, 2, TINY, device=jax.devices("cpu")[0], tuning=tuning, max_control_steps=60, progress=None)
    assert again["written"] == 0

    data = load(tmp_path)
    assert data["observations"].shape[1] == OBSERVATION_SIZE
    assert data["actions"].shape == (data["observations"].shape[0], 4)
    assert np.isfinite(data["observations"]).all()
    assert set(np.unique(data["episode"])) <= {0, 1, 2, 3}
    agl = data["observations"][:, OBSERVATION_NAMES.index("min(agl,200)/100")] * 100.0
    assert (agl >= 10.0 - 1e-3).all()                           # approach segment only

    before = np.load(tmp_path / "shard-00001.npz")["actions"].copy()
    (tmp_path / "shard-00001.npz").write_bytes(b"truncated")
    assert not shard_complete(tmp_path, 1)
    rebuilt = generate(tmp_path, 2, TINY, device=jax.devices("cpu")[0], tuning=tuning, max_control_steps=60, progress=None)
    assert rebuilt["written"] == 1
    np.testing.assert_array_equal(np.load(tmp_path / "shard-00001.npz")["actions"], before)
