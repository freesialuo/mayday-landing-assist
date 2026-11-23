import jax
import jax.numpy as jnp
import numpy as np

from mayday_landing.mechanics import (
    DEAD,
    OFF,
    RUNNING,
    STARTING,
    WeatherState,
    initial_engines,
    step_engines,
    weather_effects,
)


advance = jax.jit(step_engines)


def test_spool_tracks_each_lever_and_gates_thrust():
    engines = initial_engines(jnp.array([0.2, 0.8]))
    engines = advance(engines, jnp.array([0.7, 0.3]), 0.2)
    np.testing.assert_allclose(engines.spool, [0.31, 0.64], atol=1e-6)
    np.testing.assert_allclose(engines.thrust, engines.spool, atol=1e-6)
    stopped = advance(engines, 1.0, 0.1, shutdown=jnp.array([True, False]), airborne=True, airspeed=70.0)
    np.testing.assert_array_equal(stopped.state, [OFF, RUNNING])
    np.testing.assert_allclose(stopped.spool, [0.21, 0.695], atol=1e-6)
    np.testing.assert_allclose(stopped.thrust, [0.0, 0.695], atol=1e-6)
    assert stopped.rpm[0] > 0.12
    windmilled = advance(stopped, 1.0, 5.0, airborne=True, airspeed=70.0)
    np.testing.assert_allclose(windmilled.rpm[0], 0.12, atol=1e-6)


def test_fire_derates_thrust_without_immediate_cutoff():
    left = jnp.array([True, False])
    engines = advance(initial_engines(0.8), 0.8, 1.0, fire_start=left)
    np.testing.assert_array_equal(engines.state, [RUNNING, RUNNING])
    np.testing.assert_allclose(engines.thrust, [0.56, 0.8], atol=1e-6)
    np.testing.assert_allclose(engines.fire, [0.385, 0.0], atol=1e-6)
    np.testing.assert_allclose(engines.health, [0.98615, 1.0], atol=1e-6)
    engines = advance(engines, 0.8, 1.0, extinguish=left)
    np.testing.assert_allclose(engines.fire, [0.0, 0.0], atol=1e-6)
    assert 0.56 < engines.thrust[0] < engines.thrust[1]


def test_repair_survives_scenario_pause_and_starter_has_no_thrust():
    engines = initial_engines(0.4)._replace(health=jnp.array([0.55, 1.0]))
    first = advance(engines, 0.4, 1.0, shutdown=jnp.array([True, False]), repair_work=0.45)
    paused = advance(first, 0.4, 5.0)
    np.testing.assert_allclose(paused.repair_progress, [0.45, 0.0], atol=1e-6)
    np.testing.assert_allclose(paused.health, [0.55, 1.0], atol=1e-6)
    repaired = advance(paused, 0.4, 1.0, repair_work=0.55)
    np.testing.assert_allclose(repaired.repair_progress, [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(repaired.health, [0.85, 1.0], atol=1e-6)
    starting = advance(repaired, 0.4, 1.0, restart=True)
    np.testing.assert_array_equal(starting.state, [STARTING, RUNNING])
    np.testing.assert_allclose(starting.thrust, [0.0, 0.4], atol=1e-6)
    assert starting.rpm[0] > 0.0
    running = advance(starting, 0.4, 1.6)
    np.testing.assert_array_equal(running.state, [RUNNING, RUNNING])
    np.testing.assert_allclose(running.thrust[0], 0.0, atol=1e-6)
    spooled = advance(running, 0.4, 0.1)
    np.testing.assert_allclose(spooled.thrust[0], 0.055 * (0.45 + 0.55 * 0.85), atol=1e-6)


def test_dead_requires_open_cowl_and_completed_swap_before_restart():
    engines = initial_engines(0.6)._replace(health=jnp.array([0.0, 1.0]))
    denied = advance(engines, 0.6, 1.0, repair_work=1.0, swap_start=True, restart=True)
    np.testing.assert_array_equal(denied.state, [DEAD, RUNNING])
    np.testing.assert_array_equal(denied.repair_progress, [0.0, 0.0])
    swapping = advance(denied, 0.6, 1.0, swap_start=True, cowl_access=True, swap_work=4.0)
    np.testing.assert_allclose(swapping.repair_progress, [0.4, 0.0], atol=1e-6)
    blocked = advance(swapping, 0.6, 1.0, restart=True)
    np.testing.assert_array_equal(blocked.state, [DEAD, RUNNING])
    swapped = advance(blocked, 0.6, 1.0, swap_work=6.0)
    np.testing.assert_array_equal(swapped.state, [OFF, RUNNING])
    np.testing.assert_allclose(swapped.health, [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(swapped.thrust, [0.0, 0.6], atol=1e-6)
    np.testing.assert_array_equal(advance(swapped, 0.6, 0.2, restart=True).state, [STARTING, RUNNING])


def test_weather_keeps_rain_and_fog_out_of_force_inputs():
    wind = jnp.array([3.0, -0.2, -4.0])
    clear = WeatherState(wind, turbulence=0.15)
    fog = clear._replace(rain=0.8, icing=1.0, fog_at_plane=0.7, fog_at_camera=0.9, visibility_m=120.0)
    clear_effects = jax.jit(weather_effects)(clear, hazard_turbulence=0.7)
    fog_effects = jax.jit(weather_effects)(fog, hazard_turbulence=0.7)
    np.testing.assert_array_equal(fog_effects.wind, clear_effects.wind)
    np.testing.assert_array_equal(fog_effects.turbulence, clear_effects.turbulence)
    np.testing.assert_array_equal(fog_effects.engine_ice, clear_effects.engine_ice)
    assert fog_effects.visibility_m == 120.0
    np.testing.assert_allclose(fog_effects.turbulence, 0.7, atol=1e-6)
    engines = initial_engines(0.6)
    iced = advance(engines, 0.6, 0.2, icing=jnp.array([1.0, 0.0]))
    np.testing.assert_allclose(iced.thrust, [0.39, 0.6], atol=1e-6)


def test_engine_state_shape_and_dtype_are_stable_through_scan():
    def simulate(engines, throttles):
        return jax.lax.scan(lambda e, t: (step_engines(e, t, 0.02), e.thrust), engines, throttles)

    final, thrust_path = jax.jit(simulate)(initial_engines(0.2), jnp.array([0.6, 0.6, 0.4]))
    assert final.swap.dtype == jnp.bool_
    assert final.state.dtype == jnp.int32
    assert thrust_path.shape == (3, 2)
    np.testing.assert_allclose(final.thrust, [0.233, 0.233], atol=1e-6)
