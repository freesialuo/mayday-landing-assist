"""Sample the kinematic fields of the game's client flight snapshots.

Pass snapshots from the same world origin and a target in server time. Buffer
management, clock estimation, packet timing and packed telemetry fields live
outside this sampler.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .physics import fmod360, wrap180


class Snapshot(NamedTuple):
    """Timestamp in seconds, world position and ground velocity as xyz vectors.

    Attitude and angular rates use degrees and degrees per second.
    """

    time: jax.Array
    position: jax.Array
    velocity: jax.Array
    heading: jax.Array
    pitch: jax.Array
    roll: jax.Array
    yaw_rate: jax.Array
    pitch_rate: jax.Array
    roll_rate: jax.Array


class SnapshotSample(NamedTuple):
    snapshot: Snapshot
    extrapolating: jax.Array


def _lerp(older, newer, fraction):
    return older + (newer - older) * fraction


def _interpolate(older, newer, target_time):
    gap = newer.time - older.time
    enough_time = gap > 1e-6
    fraction = (target_time - older.time) / jnp.where(enough_time, gap, 1.0)
    fraction = jnp.clip(jnp.where(enough_time, fraction, 1.0).astype(jnp.float32), 0.0, 1.0)
    span = jnp.asarray(gap, jnp.float32)

    # FlightSnapshot.Hermite uses endpoint velocities as world position tangents.
    squared = fraction * fraction
    cubed = squared * fraction
    hermite = (
        (2.0 * cubed - 3.0 * squared + 1.0) * older.position
        + (cubed - 2.0 * squared + fraction) * span * older.velocity
        + (-2.0 * cubed + 3.0 * squared) * newer.position
        + (cubed - squared) * span * newer.velocity
    )
    position = jnp.where((span > 1e-4) & (span <= 0.5), hermite,
                         _lerp(older.position, newer.position, fraction))
    return Snapshot(
        _lerp(older.time, newer.time, fraction),
        position,
        _lerp(older.velocity, newer.velocity, fraction),
        fmod360(older.heading + fraction * wrap180(newer.heading - older.heading)),
        _lerp(older.pitch, newer.pitch, fraction),
        wrap180(older.roll + fraction * wrap180(newer.roll - older.roll)),
        _lerp(older.yaw_rate, newer.yaw_rate, fraction),
        _lerp(older.pitch_rate, newer.pitch_rate, fraction),
        _lerp(older.roll_rate, newer.roll_rate, fraction),
    )


def _extrapolate(snapshot, elapsed):
    dt = jnp.clip(elapsed, 0.0, 0.3)
    return snapshot._replace(
        time=snapshot.time + dt,
        position=snapshot.position + dt * snapshot.velocity,
        heading=fmod360(snapshot.heading + dt * snapshot.yaw_rate),
        pitch=jnp.clip(snapshot.pitch + dt * snapshot.pitch_rate, -89.0, 89.0),
        roll=wrap180(snapshot.roll + dt * snapshot.roll_rate),
    )


def sample_snapshot(older: Snapshot, newer: Snapshot, target_time) -> SnapshotSample:
    """Sample two adjacent snapshots ordered by timestamp.

    A target before the available history returns the first snapshot. A target
    past the newest snapshot extrapolates at most 300 ms and is flagged after
    5 ms. For one available snapshot, pass it as both endpoints. With JAX's
    default timestamp precision, use a common nearby origin for all times.
    """
    before_history = target_time < older.time
    at_or_after_latest = target_time >= newer.time
    elapsed = jnp.asarray(target_time - newer.time, jnp.float32)
    interpolated = _interpolate(older, newer, target_time)
    extrapolated = _extrapolate(newer, elapsed)
    snapshot = jax.tree.map(
        lambda first, between, past: jnp.where(before_history, first, jnp.where(at_or_after_latest, past, between)),
        older, interpolated, extrapolated,
    )
    return SnapshotSample(snapshot, elapsed > 0.005)
