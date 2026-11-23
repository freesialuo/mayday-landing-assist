"""Opt-in, seeded weather profiles for the landing simulator."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


CLEAR, RAIN, FOG, GUSTS, LOW_VIS_WINDY, ICING = range(6)
WEATHER_NAMES = ("clear", "rain", "fog", "gusts", "low_vis_windy", "icing")
CLEAR_VISIBILITY = 8000.0


class WeatherScenario(NamedTuple):
    kind: jax.Array
    severity: jax.Array
    leg: jax.Array
    headwind: jax.Array
    crosswind: jax.Array
    wind_phase: jax.Array
    gust_period: jax.Array
    first_gust: jax.Array
    rain_centre: jax.Array
    rain_radius: jax.Array
    initial_ice: jax.Array
    icing_duration: jax.Array


class WeatherSample(NamedTuple):
    rain: jax.Array
    visibility: jax.Array
    fog_at_plane: jax.Array
    turbulence: jax.Array
    icing: jax.Array
    airframe_ice: jax.Array
    headwind: jax.Array
    crosswind: jax.Array
    gust_strength: jax.Array
    gust_heave: jax.Array


def _unit(value):
    return jnp.clip(value, 0.0, 1.0)


def _smooth(start, end, value):
    t = _unit((value - start) / (end - start))
    return t * t * (3.0 - 2.0 * t)


def fog_visibility(leg, severity):
    """Airport fog visibility in meters from the native FogVisibility curve."""
    value = 1500.0 - 1180.0 * _smooth(0.2, 1.0, severity)
    early_floor = jnp.where(leg <= 3, 1200.0, jnp.where(leg <= 5, 900.0, 320.0))
    return 10.0 * jnp.round(jnp.maximum(value, early_floor) / 10.0)


def fog_density(severity, agl, distance):
    """Horizontal and vertical fades for one fully established airport bank."""
    severity = _unit(severity)
    radius = 4500.0 + 1500.0 * severity
    top = 70.0 + 90.0 * severity
    return (1.0 - _smooth(0.8 * radius, radius, distance)) * (1.0 - _smooth(top - 40.0, top + 30.0, agl))


def atmospheric_icing(altitude, freezing_level, moisture):
    """The moisture and freezing-level branch of native SampleAt."""
    return jnp.where(
        moisture > 0.2,
        (0.35 + 0.65 * moisture) * _unit((altitude - freezing_level) / 600.0),
        0.0,
    )


def advance_airframe_ice(ice, atmospheric, dt, *, airborne=True, deice=False):
    """Apply native accumulation and passive or powered shedding rates."""
    collecting = (atmospheric > 0.0) & jnp.asarray(airborne, dtype=jnp.bool_)
    accumulation = jnp.where(collecting, 0.012 * _unit(atmospheric), 0.0)
    shedding = jnp.where(deice, 0.03, jnp.where(collecting, 0.0, 0.004))
    return _unit(ice + jnp.maximum(dt, 0.0) * (accumulation - shedding))


def sample_weather(key, difficulty=0.5, *, kind=None, leg=None):
    """Choose an approach-local profile. Weights and cell placement are training inputs."""
    difficulty = _unit(jnp.float32(difficulty))
    draw = jax.random.uniform(key, (12,), dtype=jnp.float32)
    if kind is None:
        chosen = 1 + (draw[1] * (len(WEATHER_NAMES) - 1)).astype(jnp.int32)
        kind = jnp.where(draw[0] < 0.8 * difficulty, chosen, CLEAR)
    else:
        kind = jnp.asarray(kind, dtype=jnp.int32)
    if leg is None:
        leg = 1 + (draw[2] * 8).astype(jnp.int32)
    else:
        leg = jnp.asarray(leg, dtype=jnp.int32)

    severity = 0.35 + 0.65 * draw[3]
    windy = (kind == GUSTS) | (kind == LOW_VIS_WINDY)
    direction = jnp.where(draw[4] < 0.5, -1.0, 1.0)
    gust_period = (18.0 + 14.0 * draw[6]) / (0.7 + 0.6 * severity)
    return WeatherScenario(
        kind=kind,
        severity=severity,
        leg=leg,
        headwind=(2.0 * draw[5] - 1.0) * 2.0 * difficulty,
        crosswind=jnp.where(windy, direction * (5.5 + 3.0 * difficulty), (2.0 * draw[4] - 1.0) * difficulty),
        wind_phase=2.0 * jnp.pi * draw[7],
        gust_period=gust_period,
        first_gust=(0.5 + 0.5 * draw[8]) * gust_period,
        rain_centre=-2000.0 + 1500.0 * draw[9],
        rain_radius=3000.0 + 2000.0 * draw[10],
        initial_ice=jnp.where(kind == ICING, 0.08 + 0.24 * draw[11], 0.0),
        icing_duration=jnp.where(kind == ICING, 30.0 + 60.0 * draw[8], 0.0),
    )


def weather_at(profile, elapsed, agl, along, cross=0.0):
    """Evaluate local weather along a runway-relative approach and rollout."""
    elapsed = jnp.maximum(elapsed, 0.0)
    kind, severity = profile.kind, profile.severity
    foggy = (kind == FOG) | (kind == LOW_VIS_WINDY)
    windy = (kind == GUSTS) | (kind == LOW_VIS_WINDY)
    rainy = (kind == RAIN) | (kind == ICING)
    fog = jnp.where(foggy, fog_density(severity, agl, jnp.hypot(along, cross)), 0.0)
    visibility = jnp.where(fog > 0.01, fog_visibility(profile.leg, severity), CLEAR_VISIBILITY)

    radial = jnp.hypot(along - profile.rain_centre, cross)
    cell = 1.0 - _smooth(0.3 * profile.rain_radius, profile.rain_radius, radial)
    rain = jnp.where(rainy, (0.55 + 0.45 * severity) * cell, 0.0)

    # Seeded sinusoid stands in for WNoise in this replayable profile.
    wave = jnp.sin(2.0 * jnp.pi * 0.35 * elapsed + profile.wind_phase)
    front_noise = 0.85 + 0.15 * wave
    rain_turbulence = jnp.where(rainy, (0.07 + 0.08 * severity) * cell * front_noise, 0.0)
    wind_turbulence = jnp.where(windy, (0.2 + 0.25 * severity) * front_noise, 0.0)
    turbulence = jnp.maximum(0.03, jnp.maximum(rain_turbulence, wind_turbulence))
    crosswind_gain = jnp.where(windy, _unit(elapsed / 6.0) * (1.0 + 0.18 * wave), 1.0)
    gust_time = elapsed - profile.first_gust
    gust = windy & (gust_time >= 0.0) & (jnp.mod(gust_time, profile.gust_period) < 0.3)

    # This is an injected icing incident on final, where the ordinary
    # freezing-level branch seldom produces icing near a low-elevation runway.
    # Use the incident exposure up to elapsed, then passive thawing.
    incident = 0.4 + 0.6 * severity
    icing = jnp.where((kind == ICING) & (elapsed < profile.icing_duration), incident, 0.0)
    exposure_time = jnp.minimum(elapsed, profile.icing_duration)
    iced = advance_airframe_ice(profile.initial_ice, jnp.where(kind == ICING, incident, 0.0), exposure_time)
    ice = advance_airframe_ice(iced, 0.0, elapsed - exposure_time)
    return WeatherSample(
        rain=rain,
        visibility=visibility,
        fog_at_plane=fog,
        turbulence=turbulence,
        icing=icing,
        airframe_ice=ice,
        headwind=profile.headwind,
        crosswind=profile.crosswind * crosswind_gain,
        gust_strength=jnp.where(gust, 0.5 + 0.5 * severity, 0.0),
        gust_heave=jnp.where(gust, -14.0, 0.0),
    )
