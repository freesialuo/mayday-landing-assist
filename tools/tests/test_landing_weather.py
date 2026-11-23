import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mayday_landing.weather import (
    CLEAR,
    CLEAR_VISIBILITY,
    FOG,
    GUSTS,
    ICING,
    LOW_VIS_WINDY,
    RAIN,
    advance_airframe_ice,
    atmospheric_icing,
    fog_density,
    fog_visibility,
    sample_weather,
    weather_at,
)


def test_native_fog_severity_and_leg_floors():
    assert fog_visibility(8, 0.0) == 1500.0
    assert fog_visibility(8, 0.2) == 1500.0
    assert fog_visibility(8, 0.6) == 910.0
    np.testing.assert_array_equal(fog_visibility(jnp.array([1, 3, 4, 5, 6]), 1.0),
                                  [1200.0, 1200.0, 900.0, 900.0, 320.0])
    assert fog_density(1.0, 120.0, 4800.0) == 1.0
    assert fog_density(1.0, 155.0, 5400.0) == pytest.approx(0.25)
    assert fog_density(1.0, 190.0, 0.0) == 0.0
    assert fog_density(1.0, 0.0, 6000.0) == 0.0


def test_fog_changes_visibility_with_same_physics_inputs():
    key = jax.random.key(12)
    clear = sample_weather(key, 0.7, kind=CLEAR, leg=8)
    fog = sample_weather(key, 0.7, kind=FOG, leg=8)
    at = jax.jit(weather_at)
    clear_low = at(clear, 10.0, 5.0, -600.0)
    fog_low = at(fog, 10.0, 5.0, -600.0)
    assert clear_low.visibility == CLEAR_VISIBILITY
    assert 320.0 <= fog_low.visibility <= 1500.0
    assert fog_low.fog_at_plane == 1.0
    assert at(fog, 10.0, 250.0, -600.0).visibility == CLEAR_VISIBILITY
    for field in ("rain", "turbulence", "airframe_ice", "headwind", "crosswind", "gust_heave"):
        np.testing.assert_array_equal(getattr(clear_low, field), getattr(fog_low, field))


def test_rain_field_and_turbulence_fade_outside_the_cell():
    rainy = sample_weather(jax.random.key(13), 0.7, kind=RAIN)
    centre = weather_at(rainy, 15.0, 50.0, rainy.rain_centre)
    outside = weather_at(rainy, 15.0, 50.0, rainy.rain_centre + rainy.rain_radius + 100.0)
    np.testing.assert_allclose(centre.rain, 0.55 + 0.45 * rainy.severity, atol=1e-6)
    assert 0.03 < centre.turbulence <= 0.15
    assert outside.rain == 0.0
    assert outside.turbulence == pytest.approx(0.03)
    assert centre.visibility == outside.visibility == CLEAR_VISIBILITY
    assert centre.icing == centre.airframe_ice == 0.0


def test_wind_gust_timing_and_fog_compose():
    key = jax.random.key(16)
    windy = sample_weather(key, 1.0, kind=GUSTS)
    windy_fog = sample_weather(key, 1.0, kind=LOW_VIS_WINDY)
    assert abs(windy.crosswind) == 8.5
    assert 18.0 <= windy.gust_period * (0.7 + 0.6 * windy.severity) <= 32.0
    assert weather_at(windy, 0.0, 40.0, 0.0).crosswind == 0.0
    first = windy.first_gust
    assert weather_at(windy, first - 0.1, 40.0, 0.0).gust_heave == 0.0
    pulse = weather_at(windy, first + 0.1, 40.0, 0.0)
    assert pulse.gust_heave == -14.0
    assert pulse.gust_strength == pytest.approx(0.5 + 0.5 * float(windy.severity))
    assert 8.5 * 0.82 <= abs(pulse.crosswind) <= 8.5 * 1.18
    assert weather_at(windy, first + 0.4, 40.0, 0.0).gust_heave == 0.0
    both = weather_at(windy_fog, first + 0.1, 40.0, 0.0)
    assert both.crosswind == pulse.crosswind
    assert both.gust_heave == pulse.gust_heave
    assert both.visibility < pulse.visibility


def test_icing_only_accumulates_under_exposure_and_thaws_afterward():
    assert atmospheric_icing(750.0, 450.0, 0.2) == 0.0
    assert atmospheric_icing(400.0, 450.0, 0.8) == 0.0
    assert atmospheric_icing(750.0, 450.0, 0.8) == pytest.approx((0.35 + 0.65 * 0.8) / 2.0)
    assert advance_airframe_ice(0.2, 0.8, 5.0) == pytest.approx(0.248)
    assert advance_airframe_ice(0.2, 0.8, 5.0, airborne=False) == pytest.approx(0.18)
    assert advance_airframe_ice(0.2, 0.8, 5.0, deice=True) == pytest.approx(0.098)
    icing = sample_weather(jax.random.key(4), 0.8, kind=ICING)
    initial = weather_at(icing, 0.0, 100.0, -2000.0)
    exposure_end = weather_at(icing, icing.icing_duration, 80.0, -1000.0)
    thaw = weather_at(icing, icing.icing_duration + 5.0, 30.0, -200.0)
    assert initial.icing > 0.4
    assert 0.0 < initial.airframe_ice < exposure_end.airframe_ice
    assert exposure_end.icing == 0.0
    assert thaw.airframe_ice == pytest.approx(float(exposure_end.airframe_ice) - 0.02)


def test_seeded_profile_is_reproducible_under_jit_and_vmap():
    keys = jax.random.split(jax.random.key(18), 64)
    build = jax.jit(jax.vmap(sample_weather, in_axes=(0, None)))
    profiles = build(keys, 1.0)
    np.testing.assert_array_equal(profiles.kind, build(keys, 1.0).kind)
    np.testing.assert_array_equal(build(keys, 0.0).kind, np.zeros(64))
    assert set(map(int, profiles.kind)) == {CLEAR, RAIN, FOG, GUSTS, LOW_VIS_WINDY, ICING}
    sample = jax.jit(jax.vmap(weather_at, in_axes=(0, None, None, None)))(profiles, 12.0, 30.0, 0.0)
    assert sample.visibility.shape == (64,)
    assert np.isfinite(sample.turbulence).all()
