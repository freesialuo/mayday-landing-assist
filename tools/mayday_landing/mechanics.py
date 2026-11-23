# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Engine state and weather observations for the Mayday landing simulator."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


OFF, STARTING, RUNNING, DEAD = range(4)
START_SECONDS = 2.6
SWAP_WORK_SECONDS = 10.0


class EngineState(NamedTuple):
    """Arrays with the final axis in left, right order."""

    state: jax.Array
    spool: jax.Array
    rpm: jax.Array
    thrust: jax.Array
    health: jax.Array
    fire: jax.Array
    repair: jax.Array
    swap: jax.Array
    swap_progress: jax.Array
    starter_elapsed: jax.Array

    @property
    def running(self):
        return self.state == RUNNING

    @property
    def starting(self):
        return self.state == STARTING

    @property
    def repair_progress(self):
        return jnp.where(self.swap, self.swap_progress, self.repair)


class WeatherState(NamedTuple):
    """Sampled conditions at the aircraft, plus separate visibility observations."""

    wind: jax.Array
    turbulence: float = 0.0
    rain: float = 0.0
    icing: float = 0.0
    engine_ice: float = 0.0
    fog_at_plane: float = 0.0
    fog_at_camera: float = 0.0
    visibility_m: float = jnp.inf


def initial_engines(throttle=0.0):
    """Start the already-running pair at their throttle equilibrium."""
    throttle = jnp.clip(jnp.broadcast_to(jnp.asarray(throttle, dtype=jnp.float32), (2,)), 0.0, 1.0)
    zero = jnp.zeros_like(throttle)
    return EngineState(
        state=jnp.full((2,), RUNNING, dtype=jnp.int32),
        spool=throttle,
        rpm=_running_rpm(throttle, jnp.ones_like(throttle)),
        thrust=throttle,
        health=jnp.ones_like(throttle),
        fire=zero,
        repair=zero,
        swap=jnp.zeros((2,), dtype=jnp.bool_),
        swap_progress=zero,
        starter_elapsed=zero,
    )


def _move_towards(current, target, limit):
    return current + jnp.clip(target - current, -limit, limit)


def _running_rpm(spool, thrust_factor):
    return (0.22 + 0.78 * spool) * (0.82 + 0.18 * jnp.clip(thrust_factor / 1.15, 0.0, 1.0))


def step_engines(
    engines,
    throttle,
    dt,
    *,
    fire_start=False,
    extinguish=0.0,
    repair_work=0.0,
    swap_start=False,
    swap_work=0.0,
    cowl_access=False,
    restart=False,
    shutdown=False,
    icing=0.0,
    airborne=False,
    airspeed=0.0,
    restart_delay=START_SECONDS,
):
    """Advance the two engines for dt seconds.

    Per-engine inputs are scalars or broadcastable to the engine arrays.
    Extinguishing is a fire-intensity decrement. Repair work is a fraction of
    one serviceable wrench job and swap work is measured in seconds of work.
    Scenario code supplies eligible work and ignition actions. A swap request
    requires access through an open or missing cowl.
    """
    dt = jnp.maximum(dt, 0.0)
    throttle = jnp.clip(throttle, 0.0, 1.0)
    ice = jnp.clip(icing, 0.0, 1.0)
    shutdown = jnp.asarray(shutdown, dtype=jnp.bool_)
    swap_start = jnp.asarray(swap_start, dtype=jnp.bool_) & jnp.asarray(cowl_access, dtype=jnp.bool_)
    dead = (engines.state == DEAD) | (engines.health <= 0.0)
    mode = jnp.where(dead, DEAD, jnp.where(shutdown, OFF, engines.state))

    fire = jnp.where(fire_start & ~dead, jnp.maximum(engines.fire, 0.35), engines.fire)
    fire = jnp.maximum(0.0, fire - jnp.maximum(extinguish, 0.0))
    burning = fire > 0.0

    # Damage is applied at the end of the native engine tick. A burning engine
    # can still produce reduced thrust until it stops or reaches zero health.
    can_start = (mode == OFF) & ~shutdown & ~burning & ~engines.swap & ~swap_start
    starting = (mode == STARTING) | (restart & can_start)
    starter_elapsed = jnp.where(starting, engines.starter_elapsed + dt, 0.0)
    started = starting & (starter_elapsed >= restart_delay)
    mode = jnp.where(starting, STARTING, mode)
    mode = jnp.where(started, RUNNING, mode)

    running = mode == RUNNING
    spool_rate = jnp.where(throttle > engines.spool, 0.55, 0.8)
    spool = _move_towards(engines.spool, jnp.where(running, throttle, 0.0), dt * jnp.where(running, spool_rate, 1.0))
    spool = jnp.where(starting, 0.0, spool)
    thrust_factor = (0.45 + 0.55 * engines.health) * (1.0 - 0.35 * ice) * jnp.where(burning, 0.7, 1.0)
    thrust = jnp.where(running, spool * thrust_factor, 0.0)
    starter_rpm = 0.06 + 0.204 * jnp.clip(starter_elapsed / START_SECONDS, 0.0, 1.0)
    windmill_rpm = jnp.where(airborne, 0.12 * jnp.clip(airspeed / 70.0, 0.0, 1.3), 0.0)
    rpm_target = jnp.where(
        running,
        _running_rpm(spool, thrust_factor),
        jnp.where(starting, starter_rpm, windmill_rpm),
    )
    rpm_rate = jnp.where(rpm_target > engines.rpm, jnp.where(starting, 0.25, 0.5), jnp.where(running, 0.4, 0.25))
    rpm = _move_towards(engines.rpm, rpm_target, dt * rpm_rate)

    fire = jnp.where(burning, _move_towards(fire, 1.0, dt * 0.035), 0.0)
    health = jnp.maximum(0.0, engines.health - jnp.where(burning, 0.01 * (1.0 + fire) * dt, 0.0))
    destroyed = dead | (health <= 0.0)
    mode = jnp.where(destroyed, DEAD, mode)
    thrust = jnp.where(destroyed, 0.0, thrust)

    # Swap is a distinct service action. The caller grants it only when the
    # scenario allows replacement, and a completed swap leaves the engine Off.
    swap = engines.swap | (swap_start & ~running & ~starting & ~burning)
    swap_progress = jnp.where(
        swap, jnp.clip(engines.swap_progress + jnp.maximum(swap_work, 0.0) / SWAP_WORK_SECONDS, 0.0, 1.0), 0.0
    )
    swapped = swap & (swap_progress >= 1.0 - 1e-6)
    mode = jnp.where(swapped, OFF, mode)
    health = jnp.where(swapped, 1.0, health)

    can_repair = (mode == OFF) & ~burning & ~swap & (health < 1.0)
    work = jnp.where(can_repair, jnp.maximum(repair_work, 0.0), 0.0)
    repair = jnp.where((engines.repair >= 1.0) & (work > 0.0), 0.0, engines.repair)
    finished = (repair < 1.0) & (repair + work >= 1.0 - 1e-6)
    health = jnp.where(finished, jnp.minimum(1.0, health + 0.30), health)
    repair = jnp.where(swapped, 0.0, jnp.where(finished, 1.0, jnp.minimum(1.0, repair + work)))

    return EngineState(
        state=mode,
        spool=jnp.where(swapped, 0.0, spool),
        rpm=rpm,
        thrust=thrust,
        health=health,
        fire=fire,
        repair=repair,
        swap=swap & ~swapped,
        swap_progress=jnp.where(swapped, 0.0, swap_progress),
        starter_elapsed=jnp.where(starting & ~started & ~destroyed, starter_elapsed, 0.0),
    )


def weather_effects(weather, *, hazard_turbulence=0.0):
    """Combine measured weather turbulence with the current hazard strength."""
    return weather._replace(
        turbulence=jnp.clip(jnp.maximum(weather.turbulence, hazard_turbulence), 0.0, 1.0),
        engine_ice=jnp.clip(weather.engine_ice, 0.0, 1.0),
    )
