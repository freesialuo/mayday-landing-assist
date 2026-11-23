# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mayday_jax.snapshot import Snapshot, sample_snapshot


def _snapshot(time, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
              heading=0.0, pitch=0.0, roll=0.0, yaw_rate=0.0, pitch_rate=0.0, roll_rate=0.0):
    return Snapshot(*(jnp.asarray(value, jnp.float32) for value in (
        time, position, velocity, heading, pitch, roll, yaw_rate, pitch_rate, roll_rate,
    )))


def test_hermite_position_uses_world_velocity_and_attitude_wraps():
    older = _snapshot(1.0, (0.0, 10.0, -4.0), (16.0, 0.0, 0.0), 350.0, 2.0, 170.0, 4.0, 6.0, 8.0)
    newer = _snapshot(1.5, (8.0, 14.0, 4.0), (0.0, -8.0, 8.0), 10.0, 10.0, -170.0, 8.0, 10.0, 12.0)

    sample = sample_snapshot(older, newer, 1.25)

    np.testing.assert_allclose(sample.snapshot.position, (5.0, 12.5, -0.5))
    np.testing.assert_allclose(sample.snapshot.velocity, (8.0, -4.0, 4.0))
    np.testing.assert_allclose(sample.snapshot[3:], (0.0, 6.0, 180.0, 6.0, 8.0, 10.0))
    assert sample.snapshot.time == 1.25
    assert not sample.extrapolating


@pytest.mark.parametrize(("gap", "expected"), [(5e-7, 8.0), (1e-4, 2.0), (0.75, 2.0)])
def test_position_falls_back_to_linear_for_tiny_and_long_gaps(gap, expected):
    older = _snapshot(0.0, velocity=(160.0, 0.0, 0.0))
    newer = _snapshot(gap, position=(8.0, 0.0, 0.0))

    sample = sample_snapshot(older, newer, gap * 0.25)

    assert sample.snapshot.position[0] == expected
    assert not sample.extrapolating


def test_newest_snapshot_extrapolates_with_a_time_cap_and_attitude_limits():
    current = _snapshot(0.0, (10.0, 10.0, 10.0), (10.0, -10.0, 0.0), 350.0, 88.0, 170.0,
                        50.0, 10.0, 50.0)
    before = sample_snapshot(current, current, -1.0)
    at_snapshot = sample_snapshot(current, current, 0.0)
    just_after = sample_snapshot(current, current, 0.005)
    flagged = sample_snapshot(current, current, 0.0051)
    capped = sample_snapshot(current, current, 0.75)

    np.testing.assert_array_equal(before.snapshot.position, current.position)
    assert before.snapshot.time == current.time
    assert not before.extrapolating
    np.testing.assert_array_equal(at_snapshot.snapshot.position, current.position)
    assert not at_snapshot.extrapolating
    np.testing.assert_allclose(just_after.snapshot.position, (10.05, 9.95, 10.0))
    assert not just_after.extrapolating
    assert flagged.extrapolating
    assert capped.extrapolating
    assert capped.snapshot.time == pytest.approx(0.3)
    np.testing.assert_allclose(capped.snapshot.position, (13.0, 7.0, 10.0))
    np.testing.assert_allclose(capped.snapshot[3:], (5.0, 89.0, -175.0, 50.0, 10.0, 50.0))
    np.testing.assert_array_equal(capped.snapshot.velocity, current.velocity)


def test_sampler_compiles_and_batches_queries_on_both_sides_of_history():
    older = _snapshot(1.0, position=(0.0, 0.0, 0.0), velocity=(10.0, 0.0, 0.0))
    newer = _snapshot(1.5, position=(5.0, 0.0, 0.0), velocity=(10.0, 0.0, 0.0))
    targets = jnp.asarray((0.5, 1.25, 1.5, 1.625), jnp.float32)

    samples = jax.jit(jax.vmap(sample_snapshot, in_axes=(None, None, 0)))(older, newer, targets)

    np.testing.assert_allclose(samples.snapshot.time, (1.0, 1.25, 1.5, 1.625))
    np.testing.assert_allclose(samples.snapshot.position[:, 0], (0.0, 2.5, 5.0, 6.25))
    np.testing.assert_array_equal(samples.extrapolating, (False, False, False, True))
