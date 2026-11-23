from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from mayday_jax.checkpoints import load_checkpoint
from mayday_jax.mppi import (
    MPPIConfig, TeacherConfig, _identity, _initialize, _sequence_cost, path_error, plan, ring_tolerance, run_teacher,
)
from mayday_jax.physics import default_tuning

SMALL = TeacherConfig(batch_size=2, seed=7, difficulty=0.5, chunk_steps=4,
                      mppi=MPPIConfig(samples=8, horizon=6, hold=6))


def test_ring_tolerance_matches_gate_geometry():
    hw, hh = ring_tolerance(jnp.asarray([600.0, 4000.0]))
    np.testing.assert_allclose(hw, [44.0, 80.0], rtol=1e-6)
    np.testing.assert_allclose(hh, [24.2, 44.0], rtol=1e-6)


def test_plan_shapes_and_never_worse_than_nominal():
    tuning = default_tuning()
    cfg = SMALL.mppi
    state = _initialize(jax.random.PRNGKey(0), jnp.float32(0.5), tuning, 2, cfg)
    env = jax.tree.map(lambda x: x[0], state.environments)
    nominal, last = state.nominal[0], state.last_action[0]
    dt = jnp.float32(1 / 60)
    updated, best, weighted = plan(env, nominal, last, jax.random.PRNGKey(1), tuning, dt, cfg)
    assert updated.shape == nominal.shape
    assert bool(jnp.all(jnp.isfinite(updated)))
    nominal_cost = _sequence_cost(env, nominal, last, tuning, dt, cfg)
    # sample 0 is the unperturbed nominal, so the best sampled cost can't exceed it
    assert float(best) <= float(nominal_cost) + 1e-3
    assert float(best) <= float(weighted) + 1e-3
    lat, ver = path_error(env)
    assert bool(jnp.isfinite(lat)) and bool(jnp.isfinite(ver))


def test_teacher_resume_is_bit_identical(tmp_path):
    tuning = default_tuning()
    interrupted, reference = tmp_path / "interrupted", tmp_path / "reference"
    first, _ = run_teacher(SMALL, interrupted, 10, 5, progress=None, tuning=tuning)
    resumed, _ = run_teacher(replace(SMALL, chunk_steps=3), interrupted, 20, 5, progress=None, tuning=tuning)
    straight, _ = run_teacher(SMALL, reference, 20, 7, progress=None, tuning=tuning)
    assert first["control_steps"] == 10 and resumed["restored_step"] == 10
    assert resumed["control_steps"] == straight["control_steps"] == 20

    template = _initialize(jax.random.PRNGKey(SMALL.seed), jnp.float32(SMALL.difficulty), tuning,
                           SMALL.batch_size, SMALL.mppi)
    identity = _identity(SMALL, tuning)
    step_a, state_a = load_checkpoint(interrupted, template, identity)
    step_b, state_b = load_checkpoint(reference, template, identity)
    assert step_a == step_b == 20
    for got, expected in zip(jax.tree.leaves(state_a), jax.tree.leaves(state_b), strict=True):
        np.testing.assert_array_equal(got, expected)
