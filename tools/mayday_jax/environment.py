"""Seeded approach episodes for the Mayday Airways Demo flight twin.

ControlLawT.cs supplies the manual rate law and envelopes. GatesT.cs supplies
the gate dimensions, frame-based judging, and state codes. The recorded meta
confirms the farthest-first distances; the C# twin's nominal 3-degree slope is
used pending an in-game measurement of the ring center heights.

This training scenario starts on a mature, gear-down approach with flat
ground and a rectangular runway. Throttle is an absolute shared lever and
feeds normalized thrust directly. Engine spool, failures, network delay,
and takeover logic are outside this scenario. Smooth seeded sinusoids and
periodic short stick disturbances approximate weather and competing input.
Leaving the gate activation area ends an episode. This model is for the game.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .physics import DEG, G, RAD, Controls, Ground, State, Tuning, alpha_required, stall_speed, step

GATE_DISTANCES = (4000.0, 3000.0, 2000.0, 1200.0, 600.0)
GLIDE_SLOPE = 0.05240777928304121  # tan(3 degrees)
GATE_ACTIVATION_RADIUS = 8000.0
RUNWAY_LENGTH = 1500.0  # Nominal runway from Tests/Program.cs
RUNWAY_HALF_WIDTH = 30.0
MAX_EPISODE_SECONDS = 180.0

RUNNING = 0
RUNWAY_TOUCHDOWN = 1
OFF_RUNWAY_TOUCHDOWN = 2
TIMEOUT = 3
LEFT_AREA = 4


class Disturbances(NamedTuple):
    wind_x: jax.Array
    wind_z: jax.Array
    turbulence: jax.Array
    phases: jax.Array
    jolt_period: jax.Array
    jolt_heave: jax.Array
    competing_start: jax.Array
    competing_period: jax.Array
    competing_duration: jax.Array
    competing_axes: jax.Array
    gear_jam: jax.Array            # 1.0 = gear down and locked; < 1 = stuck at that extension


class EnvState(NamedTuple):
    aircraft: State
    runway_heading: jax.Array
    runway_elev: jax.Array
    along: jax.Array
    lateral: jax.Array
    elapsed: jax.Array
    trim_speed: jax.Array
    flap_setting: jax.Array
    flap_candidate: jax.Array
    flap_candidate_time: jax.Array
    disturbances: Disturbances
    gate_states: jax.Array
    gate_lateral_error: jax.Array
    gate_vertical_error: jax.Array
    done: jax.Array
    termination: jax.Array
    touchdown_vertical_speed: jax.Array
    touchdown_along: jax.Array
    touchdown_lateral: jax.Array


def runway_coordinates(px, pz, runway_heading):
    """Return along and runway-right offsets from the threshold at (0, 0)."""
    east = jnp.sin(runway_heading * RAD)
    north = jnp.cos(runway_heading * RAD)
    return px * east + pz * north, px * north - pz * east


def gate_geometry(runway_elev=0.0):
    """Return arrays of judge heights, half-widths, and half-heights, in metres."""
    distances = jnp.asarray(GATE_DISTANCES, dtype=jnp.float32)
    half_width = 20.0 * (2.2 + 1.8 * _unit((distances - 600.0) / 3400.0))
    # The visual center is 2.8 m above this judge height. GatesT.Update adds
    # that same offset to the rendered aircraft position before comparing.
    height = runway_elev + distances * GLIDE_SLOPE
    return height, half_width, 0.55 * half_width


def _unit(value):
    return jnp.clip(value, 0.0, 1.0)


def _trim_attitude(aircraft, trim_speed, tuning):
    reference = jnp.maximum(trim_speed, 1.3 * stall_speed(aircraft.flap_pos, tuning))
    error = aircraft.speed - reference
    bank_cos = jnp.maximum(0.3, jnp.cos(aircraft.roll * RAD))
    gain = jnp.where(error < 0.0, 1.2, 0.5)
    required = alpha_required(reference, aircraft.flap_pos, 0.0, tuning)
    return jnp.clip(gain * error + required / bank_cos - 2.0, -12.0, 15.0)


def _protections(aircraft, pitch_rate, roll_rate, tuning):
    alpha, pitch, roll = aircraft.alpha, aircraft.pitch, aircraft.roll
    pitch_rate = jnp.where(
        (pitch_rate > 0.0) & (alpha > 11.9),
        pitch_rate * _unit((13.3 - alpha) / 1.399999),
        pitch_rate,
    )
    pitch_rate = jnp.where(
        alpha > 13.02, jnp.minimum(pitch_rate, -(alpha - 13.02) * 4.0), pitch_rate
    )
    near_ground = (aircraft.gear_pos > 0.5) & (aircraft.agl < 3.0)
    lo = jnp.where(near_ground, 6.0, 16.0)
    hi = jnp.where(near_ground, 7.5, 20.0)
    pitch_rate = jnp.where(
        (pitch_rate > 0.0) & (pitch > lo), pitch_rate * _unit((hi - pitch) / (hi - lo)), pitch_rate
    )
    pitch_rate = jnp.where(
        pitch > hi, jnp.minimum(pitch_rate, -(pitch - hi) * 2.0), pitch_rate
    )
    pitch_rate = jnp.where(
        (pitch_rate < 0.0) & (pitch < -14.0),
        pitch_rate * _unit((pitch + 18.0) * 0.25),
        pitch_rate,
    )
    pitch_rate = jnp.where(
        pitch < -18.0, jnp.maximum(pitch_rate, (-18.0 - pitch) * 2.0), pitch_rate
    )
    excess_speed = aircraft.speed - 88.0
    pitch_rate = jnp.where(
        (excess_speed > 0.0) & (pitch_rate < 0.0),
        pitch_rate * _unit(1.0 - excess_speed * 0.25),
        pitch_rate,
    )
    pitch_rate += jnp.where(
        excess_speed > 0.0, jnp.minimum(8.0, excess_speed**2 * 0.08 + excess_speed * 0.3), 0.0
    )
    bank_height = jnp.where(aircraft.gear_pos > 0.5, _unit(aircraft.agl / 12.0), 1.0)
    wing_limit = jnp.maximum(4.0, tuning.wing_strike_roll - 1.5)
    bank_hi = (45.0 - wing_limit) * bank_height + wing_limit
    bank_lo = (33.0 - wing_limit * 0.6) * bank_height + wing_limit * 0.6
    abs_roll = jnp.abs(roll)
    roll_rate = jnp.where(
        (abs_roll > bank_lo) & (roll * roll_rate > 0.0),
        roll_rate * _unit((bank_hi - abs_roll) / (bank_hi - bank_lo)),
        roll_rate,
    )
    roll_rate -= jnp.maximum(0.0, abs_roll - bank_hi) * jnp.sign(roll) * 3.0
    return pitch_rate, roll_rate


def _limits(aircraft, pitch_rate, roll_rate):
    pitch, roll = aircraft.pitch, aircraft.roll
    abs_roll = jnp.abs(roll)
    roll_rate = jnp.where(
        (abs_roll > 45.0) & (roll * roll_rate > 0.0),
        roll_rate * _unit((70.0 - abs_roll) / 25.0),
        roll_rate,
    )
    pitch_rate = jnp.where(
        (pitch > 20.0) & (pitch_rate > 0.0),
        pitch_rate * _unit((30.0 - pitch) / 10.0),
        pitch_rate,
    )
    pitch_rate = jnp.where(
        (pitch < -22.0) & (pitch_rate < 0.0),
        pitch_rate * _unit((pitch + 35.0) / 13.0),
        pitch_rate,
    )
    stall_alpha = aircraft.alpha - 14.0
    stall_pitch_rate = jnp.minimum(
        pitch_rate - 10.0 * jnp.minimum(stall_alpha * 0.25, 1.0),
        -jnp.clip(stall_alpha * 2.5, 3.0, 28.0),
    )
    return jnp.where(aircraft.stalling, stall_pitch_rate, pitch_rate), roll_rate


def manual_rates(aircraft, pitch_in, roll_in, trim_speed, throttle, tuning, dt):
    """Port of the occupied cockpit's Manual law, returning q, p, and trim speed."""
    speed, pitch, roll = aircraft.speed, aircraft.pitch, aircraft.roll
    assist = tuning.assists_enabled
    pitch_active = jnp.abs(pitch_in) > 0.05
    speed_gain = _unit((speed - 12.0) / 28.0)
    low = jnp.where(aircraft.gear_pos <= 0.5, 0.0, 1.0 - _unit((aircraft.agl - 4.0) * 0.125))
    air_trim = jnp.where(
        assist & (throttle > 0.6) & (trim_speed < 50.0),
        jnp.minimum(50.0, trim_speed + 3.0 * dt),
        trim_speed,
    )
    air_trim = jnp.where(pitch_active, speed, air_trim)
    trim_pitch = _trim_attitude(aircraft, air_trim, tuning)
    trim_rate = jnp.clip(((jnp.minimum(pitch, 9.0) - trim_pitch) * low + trim_pitch - pitch) * 0.25, -3.0, 3.0)
    stick_rate = pitch_in * 10.0 * (speed_gain * 0.65 + 0.35) * (1.0 - 0.55 * low)
    pitch_rate = jnp.where(pitch_active, stick_rate, trim_rate)
    roll_hold = jnp.where(assist, jnp.clip(-roll * 0.6, -12.0, 12.0), jnp.clip(-roll * 0.45, -10.0, 10.0))
    roll_rate = jnp.where(
        jnp.abs(roll_in) > 0.05, (speed_gain * 0.7 + 0.3) * roll_in * 40.0, roll_hold
    )
    near_stall = (low > 0.0) & (pitch_rate > 0.0) & (aircraft.alpha > 11.9)
    flare_factor = (_unit((13.3 - aircraft.alpha) / 1.4) - 1.0) * low + 1.0
    pitch_rate = jnp.where(near_stall, pitch_rate * flare_factor, pitch_rate)
    protected_pitch, protected_roll = _protections(aircraft, pitch_rate, roll_rate, tuning)
    pitch_rate, roll_rate = _limits(
        aircraft,
        jnp.where(assist, protected_pitch, pitch_rate),
        jnp.where(assist, protected_roll, roll_rate),
    )

    rotation_gain = _unit((speed - 32.3) / 5.7)
    rotating = (pitch_in > 0.05) & (rotation_gain > 0.0)
    rotation_rate = pitch_in * 6.0 * rotation_gain
    rotation_rate *= jnp.where(assist & (pitch > 6.0), _unit((7.5 - pitch) / 1.5), 1.0)
    ground_pitch = jnp.where(
        rotating,
        rotation_rate,
        jnp.where((pitch_in < -0.05) | ((pitch > 0.01) & (speed < 20.0)), -6.0,
                  jnp.where(pitch > 0.01, -3.0, 0.0)),
    )
    return (
        jnp.where(aircraft.on_ground, ground_pitch, pitch_rate),
        jnp.where(aircraft.on_ground, -roll * 5.0, roll_rate),
        jnp.where(aircraft.on_ground, jnp.where(rotating, speed, trim_speed), air_trim),
    )


def _auto_flap(env, tuning, dt):
    # AutoFlapT's mature-airborne branch. The approach gear command stays down.
    aircraft = env.aircraft
    speed, setting = aircraft.speed, env.flap_setting
    locked = aircraft.gear_pos > 0.9
    want = jnp.where((setting == 0) & (speed < 56.0), 1, setting)
    want = jnp.where((setting == 2) & ~((speed <= 58.0) & locked), 1, want)
    want = jnp.where(locked & (speed < 52.0), 2, want)
    want = jnp.where(speed > 60.0, 0, want)
    want = jnp.where(aircraft.on_ground, jnp.where(speed < 20.0, 1, setting), want)
    changing = tuning.auto_flaps & (want != setting)
    hold = jnp.where(want == env.flap_candidate, env.flap_candidate_time, 0.0) + dt
    accepted = changing & (hold >= 1.5)
    pending = changing & ~accepted
    return jnp.where(accepted, want, setting), jnp.where(pending, want, -1), jnp.where(pending, hold, 0.0)


def _paved(along, lateral):
    return (along >= 0.0) & (along <= RUNWAY_LENGTH) & (jnp.abs(lateral) <= RUNWAY_HALF_WIDTH)


def _weather(disturbances, elapsed):
    # Sinusoids replace the game's native Perlin noise. Frequencies and
    # amplitude ceilings follow DisturbanceT, with wind ramped in over 6 s.
    frequencies = jnp.asarray((0.7, 0.9, 1.3, 0.35), dtype=jnp.float32)
    pitch, roll, heave, wind = jnp.sin(2.0 * jnp.pi * elapsed * frequencies + disturbances.phases)
    tau = disturbances.turbulence
    jolt = (tau > 0.45) & (elapsed >= disturbances.jolt_period) & (elapsed % disturbances.jolt_period < 0.25)
    heave = 5.0 * tau * heave + jnp.where(jolt, disturbances.jolt_heave, 0.0)
    wind_gain = _unit(elapsed / 6.0) * (1.0 + 0.18 * wind)
    return 5.0 * tau * pitch, 14.0 * tau * roll, heave, wind_gain


def _competing_input(disturbances, elapsed):
    since_start = elapsed - disturbances.competing_start
    pulse = jnp.floor(since_start / disturbances.competing_period)
    inside = (since_start >= 0.0) & (since_start % disturbances.competing_period < disturbances.competing_duration)
    direction = jnp.where(pulse % 2.0 == 0.0, 1.0, -1.0)
    return jnp.where(inside, disturbances.competing_axes * direction, 0.0)


def reset(key, difficulty, tuning: Tuning) -> EnvState:
    """Sample one approach. Difficulty is clipped to [0, 1]."""
    difficulty = _unit(jnp.asarray(difficulty, dtype=jnp.float32))
    scenario_key, weather_key, competing_key = jax.random.split(key, 3)
    yaw_u, elevation_u, distance_u, cross_u, height_u, speed_u, heading_u, gamma_u, bank_u = (
        jax.random.uniform(scenario_key, (9,), dtype=jnp.float32)
    )
    bands = 2.0 * jnp.asarray((cross_u, height_u, speed_u, heading_u, gamma_u, bank_u)) - 1.0
    runway_heading = yaw_u * 360.0
    runway_elev = elevation_u * 45.0
    along = -GATE_DISTANCES[0] - (300.0 + distance_u * 240.0)
    lateral = bands[0] * (12.0 + 143.0 * difficulty)
    altitude = -along * GLIDE_SLOPE + bands[1] * (6.0 + 32.0 * difficulty)
    speed = 47.0 + bands[2] * (3.0 + 3.0 * difficulty)
    heading = (runway_heading + bands[3] * (2.0 + 10.0 * difficulty)) % 360.0
    gamma = -3.0 + bands[4] * (0.2 + 0.8 * difficulty)
    roll = bands[5] * (3.0 + 9.0 * difficulty)
    east = jnp.sin(runway_heading * RAD)
    north = jnp.cos(runway_heading * RAD)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    false = jnp.asarray(False)
    # Jammed gear (up to 25 % of episodes at full difficulty): stuck between 35 % and 85 % extension.
    # The game's auto-flap schedule then holds flaps 1 because the gear is not locked.
    jam_key = jax.random.fold_in(scenario_key, 7)
    jam_u, jam_pos_u = jax.random.uniform(jam_key, (2,), dtype=jnp.float32)
    gear_jam = jnp.where(jam_u < 0.25 * difficulty, 0.35 + 0.5 * jam_pos_u, 1.0)
    flap = jnp.where(gear_jam < 1.0, 1.0, 2.0)
    required = alpha_required(speed, flap, 0.0, tuning)
    load = jnp.cos(gamma * RAD) / jnp.cos(roll * RAD)
    alpha = load * required
    aircraft = State(
        px=along * east + lateral * north,
        py=runway_elev + altitude,
        pz=along * north - lateral * east,
        speed=speed, gamma=gamma, track=heading, heading=heading,
        pitch=gamma + alpha - 2.0, roll=roll, prate=zero, rrate=zero, yrate=zero,
        on_ground=false, gear_pos=gear_jam, flap_pos=flap, steer_angle=zero,
        wx=zero, wy=zero, wz=zero, wind_factor=zero + 1.0,
        alpha=alpha, alpha_req=required, load_factor=load, ground_y=runway_elev,
        agl=altitude, ground_effect=zero, wow=zero, surface=jnp.asarray(0, dtype=jnp.int32),
        stalling=false, stall_warn=false, skidding=false, belly=false, overspeed=false,
        braking=false, drifting=false, track_lag=false, backward=false,
    )
    weather = jax.random.uniform(weather_key, (10,), dtype=jnp.float32)
    crosswind = (weather[0] * 2.0 - 1.0) * 8.0 * difficulty
    headwind = (weather[1] * 2.0 - 1.0) * 4.0 * difficulty
    tau = (0.3 + 0.7 * weather[2]) * difficulty
    competing = jax.random.uniform(competing_key, (6,), dtype=jnp.float32)
    disturbances = Disturbances(
        wind_x=crosswind * north - headwind * east,
        wind_z=-crosswind * east - headwind * north,
        turbulence=tau,
        phases=weather[3:7] * (2.0 * jnp.pi),
        jolt_period=(1.5 + 3.5 * weather[7]) / jnp.maximum(tau, 0.01),
        jolt_heave=(8.0 + 8.0 * weather[8]) * jnp.where(weather[9] < 0.6, -1.0, 1.0) * tau,
        competing_start=3.0 + 7.0 * competing[0],
        competing_period=7.0 + 9.0 * competing[1],
        # Copilot pulses up to 1.5 s: longer inputs are a takeover the rule layer yields to.
        competing_duration=0.2 + 1.3 * competing[2],
        competing_axes=(2.0 * competing[3:] - 1.0) * (0.9 * difficulty),
        gear_jam=gear_jam,
    )
    return EnvState(
        aircraft=aircraft, runway_heading=runway_heading, runway_elev=runway_elev,
        along=along, lateral=lateral, elapsed=zero, trim_speed=speed,
        flap_setting=flap.astype(jnp.int32),
        flap_candidate=jnp.asarray(-1, dtype=jnp.int32),
        flap_candidate_time=zero, disturbances=disturbances,
        gate_states=jnp.zeros(len(GATE_DISTANCES), dtype=jnp.int32),
        gate_lateral_error=jnp.zeros(len(GATE_DISTANCES), dtype=jnp.float32),
        gate_vertical_error=jnp.zeros(len(GATE_DISTANCES), dtype=jnp.float32),
        done=false, termination=jnp.asarray(RUNNING, dtype=jnp.int32),
        touchdown_vertical_speed=zero, touchdown_along=zero, touchdown_lateral=zero,
    )


def _update_gates(env, aircraft, along, lateral, outside):
    distances = jnp.asarray(GATE_DISTANCES, dtype=jnp.float32)
    height, half_width, half_height = gate_geometry(env.runway_elev)
    rearm = along < -(GATE_DISTANCES[0] + 300.0)
    gate_states = jnp.where(rearm, 0, env.gate_states)
    crossing = (~outside) & (gate_states == 0) & (env.along < -distances) & (along >= -distances)
    vertical_error = aircraft.py - height
    threaded = (jnp.abs(lateral) <= half_width) & (jnp.abs(vertical_error) <= half_height)
    return (
        jnp.where(crossing, jnp.where(threaded, 1, 2), gate_states),
        jnp.where(crossing, lateral, jnp.where(rearm, 0.0, env.gate_lateral_error)),
        jnp.where(crossing, vertical_error, jnp.where(rearm, 0.0, env.gate_vertical_error)),
    )


def _touchdown_sink(before, after, heave, dt):
    # FlightSim.Step clears Gamma at contact. Reconstruct its airborne update
    # from the returned load and roll rate to get StepResult.Vs at impact.
    bank = jnp.clip(before.roll + after.rrate * dt, -75.0, 75.0)
    speed = jnp.maximum(after.speed, 5.0)
    gamma_rate = ((after.load_factor * jnp.cos(bank * RAD) - jnp.cos(before.gamma * RAD))
                  * (G / speed) * DEG + (heave / speed) * DEG)
    gamma = jnp.clip(before.gamma + gamma_rate * dt, -85.0, 60.0)
    return jnp.maximum(0.0, -jnp.sin(gamma * RAD) * after.speed)


def _advance(env, action, tuning, dt):
    current = env.aircraft
    pitch, roll, yaw = jnp.clip(
        jnp.clip(action[:3], -1.0, 1.0) + _competing_input(env.disturbances, env.elapsed),
        -1.0, 1.0,
    )
    throttle = _unit(action[3])
    q, p, trim = manual_rates(current, pitch, roll, env.trim_speed, throttle, tuning, dt)
    flap, flap_candidate, hold = _auto_flap(env, tuning, dt)
    turb_pitch, turb_roll, heave, wind_gain = _weather(env.disturbances, env.elapsed)
    in_air = ~current.on_ground
    heave = jnp.where(in_air, heave, 0.0)
    controls = Controls(
        pitch_rate=q + jnp.where(in_air, turb_pitch, 0.0),
        roll_rate=p + jnp.where(in_air, turb_roll, 0.0),
        rudder=yaw, steer=jnp.clip(yaw + roll, -1.0, 1.0), brakes=0.0, park_brake=False,
        thr_l=throttle, thr_r=throttle, drag_mul=1.0, gear_down=True,
        gear_jammed=env.disturbances.gear_jam < 1.0, gear_jam_pos=env.disturbances.gear_jam,
        flap=flap, heave=heave, reverse=False,
    )
    current = current._replace(
        wx=env.disturbances.wind_x * wind_gain, wz=env.disturbances.wind_z * wind_gain
    )
    ground_before = Ground(env.runway_elev, jnp.where(_paved(env.along, env.lateral), 2, 0))
    # Contact height is identical everywhere in this scenario, so the surface
    # at the integrated position can be assigned after the physics step.
    aircraft = step(current, controls, dt, tuning, ground_before, ground_before)
    along, lateral = runway_coordinates(aircraft.px, aircraft.pz, env.runway_heading)
    paved = _paved(along, lateral)
    aircraft = aircraft._replace(surface=jnp.where(paved, 2, 0).astype(jnp.int32))
    elapsed = env.elapsed + dt
    touchdown = in_air & aircraft.on_ground
    landed_on_runway = paved & (aircraft.gear_pos > 0.9)
    outside = along**2 + lateral**2 > GATE_ACTIVATION_RADIUS**2
    reason = jnp.where(elapsed >= MAX_EPISODE_SECONDS, TIMEOUT, RUNNING)
    reason = jnp.where(outside, LEFT_AREA, reason)
    reason = jnp.where(touchdown, jnp.where(landed_on_runway, RUNWAY_TOUCHDOWN, OFF_RUNWAY_TOUCHDOWN), reason)
    gate_states, gate_lateral, gate_vertical = _update_gates(env, aircraft, along, lateral, outside)
    return env._replace(
        aircraft=aircraft, along=along, lateral=lateral, elapsed=elapsed, trim_speed=trim,
        flap_setting=flap, flap_candidate=flap_candidate, flap_candidate_time=hold,
        gate_states=gate_states, gate_lateral_error=gate_lateral, gate_vertical_error=gate_vertical,
        done=reason != RUNNING, termination=reason.astype(jnp.int32),
        touchdown_vertical_speed=jnp.where(touchdown, _touchdown_sink(current, aircraft, heave, dt), 0.0),
        touchdown_along=jnp.where(touchdown, along, 0.0),
        touchdown_lateral=jnp.where(touchdown, lateral, 0.0),
    )


def advance(env: EnvState, action, tuning: Tuning, dt) -> EnvState:
    """Advance one positive-dt physics frame; terminal episodes are absorbing.

    Action is (pitch, roll, yaw, absolute throttle). Stick axes use [-1, 1]
    and throttle uses [0, 1]. Gate errors are populated only after judging
    (state 1 or 2). Touchdown sink speed is positive down, in metres per second.
    """
    action = jnp.asarray(action, dtype=env.aircraft.speed.dtype)
    dt = jnp.asarray(dt, dtype=env.elapsed.dtype)
    return jax.lax.cond(env.done, lambda: env, lambda: _advance(env, action, tuning, dt))
