# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import jax
import jax.numpy as jnp
import numpy as np

from mayday_jax import environment as approach
from mayday_jax.physics import default_tuning
from mayday_landing.environment import Scenario, reset
from mayday_landing.teacher import blind_disturbances
from mayday_landing.weather import GUSTS, weather_at

TUNING = default_tuning()


def _state(elapsed):
    state = reset(jax.random.key(5), TUNING, Scenario(difficulty=1.0, weather_kind=GUSTS))
    weather = state.weather._replace(first_gust=jnp.float32(10.0), gust_period=jnp.float32(4.0))
    d = state.flight.disturbances._replace(
        competing_start=jnp.float32(5.0), competing_period=jnp.float32(3.0), competing_duration=jnp.float32(1.0),
        competing_axes=jnp.asarray((0.5, 0.0, 0.0), jnp.float32))
    flight = state.flight._replace(elapsed=jnp.float32(elapsed), disturbances=d)
    return state._replace(weather=weather, flight=flight)


def _gust(state, t):
    return float(weather_at(state.weather, t, 50.0, -500.0).gust_heave)


def _pilot(state, t):
    return np.asarray(approach._competing_input(state.flight.disturbances, t))


def test_blind_prediction_keeps_current_pulse_and_hides_later_ones():
    inside = _state(14.1)  # inside the 14.0-14.3 s gust and the odd (negative) co-pilot pulse 14-15 s
    blind = blind_disturbances(inside)
    assert _gust(inside, 14.2) == _gust(blind, 14.2) == -14.0
    assert _gust(inside, 18.1) == -14.0 and _gust(blind, 18.1) == 0.0
    np.testing.assert_allclose(_pilot(blind, 14.5), _pilot(inside, 14.5))
    assert np.any(_pilot(inside, 17.5) != 0.0) and not np.any(_pilot(blind, 17.5))

    between = blind_disturbances(_state(12.0))
    assert _gust(between, 14.1) == 0.0
    assert not np.any(_pilot(between, 12.5)) and not np.any(_pilot(between, 14.5))


def test_residual_teacher_single_pass_matches_and_annealing_runs():
    from mayday_landing.environment import observe, observe_phase
    from mayday_landing.guidance import guidance
    from mayday_landing.residual_teacher import residual_plan
    from mayday_landing.teacher import PlannerConfig
    state = reset(jax.random.key(2), TUNING, Scenario(difficulty=0.5))
    base = guidance(observe(state), observe_phase(state), TUNING)
    key = jax.random.key(11)
    one = residual_plan(state, base, key, TUNING, PlannerConfig(samples=5, horizon=3))
    three = residual_plan(state, base, key, TUNING, PlannerConfig(samples=5, horizon=3, anneal=3))
    assert one.shape == three.shape == (5,)
    assert np.all(np.isfinite(np.asarray(three)))
