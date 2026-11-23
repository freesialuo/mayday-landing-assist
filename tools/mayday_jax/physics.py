# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Float32 JAX port of Twin/FlightSimT.cs and its TwinMath.cs tuning.

Angles and angular rates use degrees and degrees/second. Positions are in
metres, speed is in m/s and dt is in seconds. Each call advances one aircraft.
"""

from typing import NamedTuple

import jax.numpy as jnp


class State(NamedTuple):
    px: float = 0.0
    py: float = 0.0
    pz: float = 0.0
    speed: float = 0.0
    gamma: float = 0.0
    track: float = 0.0
    heading: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    prate: float = 0.0
    rrate: float = 0.0
    yrate: float = 0.0
    on_ground: bool = False
    gear_pos: float = 0.0
    flap_pos: float = 0.0
    steer_angle: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    wind_factor: float = 0.0
    alpha: float = 0.0
    alpha_req: float = 0.0
    load_factor: float = 0.0
    ground_y: float = 0.0
    agl: float = 0.0
    ground_effect: float = 0.0
    wow: float = 0.0
    surface: int = 0
    stalling: bool = False
    stall_warn: bool = False
    skidding: bool = False
    belly: bool = False
    overspeed: bool = False
    braking: bool = False
    drifting: bool = False
    track_lag: bool = False
    backward: bool = False


class Controls(NamedTuple):
    pitch_rate: float = 0.0
    roll_rate: float = 0.0
    rudder: float = 0.0
    steer: float = 0.0
    brakes: float = 0.0
    park_brake: bool = False
    thr_l: float = 0.0
    thr_r: float = 0.0
    drag_mul: float = 1.0
    gear_down: bool = False
    gear_jammed: bool = False
    gear_jam_pos: float = 0.0
    flap: int = 0
    heave: float = 0.0
    reverse: bool = False


class Ground(NamedTuple):
    height: float = 0.0
    surface: int = 2


class Tuning(NamedTuple):
    stall_speeds: tuple = (30.0, 28.0, 26.0)
    wheelbase: float = 6.0
    wing_strike_roll: float = 12.0
    assists_enabled: bool = True
    auto_flaps: bool = True
    handbrake_fun: bool = False
    reverse_thrust: float = 0.6
    reverse_fade_start: float = 8.0
    reverse_max_speed: float = 30.0


_DTYPES = {float: jnp.float32, tuple: jnp.float32, int: jnp.int32, bool: jnp.bool_}
RAD = 0.01745329
DEG = 57.29578
G = 9.81


def _as_arrays(record):
    """Normalize scalar callers even when the process has JAX x64 enabled."""
    cls = type(record)
    return cls(*(jnp.asarray(value, _DTYPES[kind])
                 for value, kind in zip(record, cls.__annotations__.values())))


def default_tuning():
    return _as_arrays(Tuning())


def move_towards(current, target, max_delta):
    delta = target - current
    return jnp.where(jnp.abs(delta) <= max_delta, target,
                     current + jnp.sign(delta) * max_delta)


def wrap180(angle):
    repeated = jnp.clip(angle - jnp.floor(angle / 360.0) * 360.0, 0.0, 360.0)
    return jnp.where(repeated > 180.0, repeated - 360.0, repeated)


def fmod360(angle):
    remainder = jnp.fmod(angle, 360.0)
    return jnp.where(remainder < 0.0, remainder + 360.0, remainder)


def stall_speed(flap, tuning):
    f = jnp.clip(flap, 0.0, 2.0)
    i = jnp.minimum(jnp.floor(f).astype(jnp.int32), 1)
    speeds = jnp.asarray(tuning.stall_speeds, jnp.float32)
    slope = speeds[i + 1] - speeds[i]
    offset = slope * jnp.clip(f - i, 0.0, 1.0)
    # The fraction is nonnegative, so this preserves the product's value and
    # forces it to round to float32 before the addition, as in the game's lerp.
    # A fused multiply-add can move alpha_required by three ULPs at rest.
    return jnp.copysign(offset, slope) + speeds[i]


def alpha_required(speed, flap, ground_effect, tuning):
    ratio = stall_speed(flap, tuning) / jnp.maximum(speed, 1.0)
    return (1.0 - ground_effect * 0.12) * ratio * 14.0 * ratio


def level_drag(speed, gear, flap, drag_mul):
    vc = jnp.maximum(speed, 15.0)
    return ((gear * 0.15 + 1.0 + flap * 0.1) * drag_mul * speed * 0.0002318 * speed
            + 950.6 / (vc * vc))


def step(state: State, controls: Controls, dt, tuning: Tuning,
         ground_before: Ground, ground_after: Ground) -> State:
    """Advance FlightSim.Step with terrain samples at the old and new position."""
    s, c, t = _as_arrays(state), _as_arrays(controls), _as_arrays(tuning)
    g0, g1 = _as_arrays(ground_before), _as_arrays(ground_after)
    dt = jnp.asarray(dt, jnp.float32)
    ground = s.on_ground
    v0 = s.speed
    gear_target = jnp.where(c.gear_jammed, jnp.clip(c.gear_jam_pos, 0.0, 1.0),
                            c.gear_down.astype(jnp.float32))
    gear = move_towards(s.gear_pos, gear_target, dt / jnp.where(c.gear_jammed, 1.6, 6.0))
    wheels = gear > 0.9
    flap = move_towards(s.flap_pos, jnp.clip(c.flap, 0, 2).astype(jnp.float32), dt / 3.0)

    agl0 = jnp.maximum(0.0, s.py - g0.height)
    proximity = 1.0 - agl0 / 12.5
    effect = jnp.where(agl0 < 12.5, proximity * proximity, 0.0)
    a_req = alpha_required(v0, flap, effect, t)
    a_ref = jnp.maximum(a_req, 0.05)
    alpha = s.pitch - s.gamma + 2.0
    load = jnp.where(alpha <= 14.0, alpha / a_ref,
                     jnp.maximum(0.45, 1.0 - (alpha - 14.0) * 0.06) * (14.0 / a_ref))
    load = jnp.clip(load, -1.0, 3.2)

    response = 1.0 - jnp.exp(-dt / 0.18)
    prate = (c.pitch_rate - s.prate) * response + s.prate
    rrate = (c.roll_rate - s.rrate) * response + s.rrate
    pitch = jnp.clip(prate * dt + s.pitch, jnp.where(ground, 0.0, -35.0),
                     jnp.where(ground, 10.0, 30.0))
    roll = jnp.clip(rrate * dt + s.roll, -75.0, 75.0)
    left, right = jnp.clip(c.thr_l, 0.0, 1.0), jnp.clip(c.thr_r, 0.0, 1.0)
    thrust = (left + right) * 0.95
    reverse = c.reverse & ground & wheels
    span = jnp.maximum(0.1, t.reverse_max_speed - t.reverse_fade_start)
    fade = 1.0 - jnp.clip((v0 - t.reverse_fade_start) / span, 0.0, 1.0)
    reverse_k = jnp.where(s.backward, fade * t.reverse_thrust, t.reverse_thrust)
    thrust = jnp.where(reverse, -thrust * reverse_k, thrust)
    backward = ground & jnp.where((v0 < 0.05) & (jnp.abs(thrust) > 0.0001),
                                  thrust < 0.0, s.backward)
    thrust = jnp.where(backward, -thrust, thrust)

    beta = jnp.where(ground, 0.0, wrap180(s.heading - s.track))
    b = beta * RAD
    drag = ((gear * 0.15 + 1.0 + flap * 0.1) * jnp.maximum(c.drag_mul, 1.0)
            * v0 * 0.0002318 * v0 * ((b + b) * b + 1.0))
    hand_fun = t.handbrake_fun & c.park_brake
    drag = jnp.where(~ground & hand_fun, drag * 1.6, drag)
    n_eff = jnp.where(ground, jnp.clip(load, 0.0, 1.0), load)
    vc = jnp.maximum(v0, 15.0)
    acc = (thrust - drag - (n_eff * 950.6 * n_eff) / (vc * vc) * (1.0 - effect * 0.5)
           - jnp.sin(s.gamma * RAD) * G)

    cr, sr = jnp.cos(roll * RAD), jnp.sin(roll * RAD)
    wow = jnp.where(ground, jnp.clip(1.0 - cr * load, 0.0, 1.0), 0.0)
    weight = jnp.maximum(wow, 0.05)
    paved = (g0.surface == 2) | (g0.surface == 3)
    mu = jnp.where(wheels, jnp.where(paved, 0.02, 0.06), 0.45)
    brake = jnp.where(c.park_brake, jnp.where(hand_fun, 6.5, 5.0),
                      jnp.where(c.brakes > 0.01, c.brakes * 3.6, 0.0))
    brake = jnp.where(wheels, brake, 0.0)
    braking = ground & (brake > 0.01) & (v0 > 0.5)
    friction = jnp.where(hand_fun & wheels, mu * G * weight + jnp.maximum(weight, 0.5) * brake,
                         (mu * G + brake) * weight)
    static_mu = jnp.where(wheels & (mu <= 0.03), 0.03, mu)
    stopped = ground & (v0 < 0.05) & (acc <= (static_mu * G + brake) * weight)
    acc = jnp.where(stopped, 0.0, jnp.where(ground, acc - friction, acc))
    spd = jnp.where(stopped, 0.0, v0)
    belly = jnp.where(ground, ~wheels & (spd > 1.0), s.belly)
    skidding = ground & wheels & c.park_brake & (spd > 3.0)
    v = jnp.maximum(0.0, acc * dt + spd)
    v5 = jnp.maximum(v, 5.0)
    f = jnp.clip((v - 12.0) / 28.0, 0.0, 1.0)
    asym = (left - right) * 3.5 * jnp.where(reverse, -t.reverse_thrust, 1.0)
    gv = G / v5

    # The ground branch also steers during the tick on which the aircraft lifts off.
    lift = (load * cr - 1.0) * gv * DEG
    liftoff = ground & (lift > 0.0) & (load * cr > 1.0) & ~backward
    cg = jnp.cos(s.gamma * RAD)
    air_gamma = jnp.clip(((load * cr - cg) * gv * DEG + (c.heave / v5) * DEG)
                         * dt + s.gamma, -85.0, 60.0)
    gamma = jnp.where(ground, jnp.where(liftoff, lift * dt, 0.0), air_gamma)
    turn = load * G * sr / (jnp.maximum(cg, 0.2) * v5) * DEG
    air_track = fmod360((beta * 0.3 + turn) * dt + s.track)
    air_yaw = ((f * 0.7 + 0.3) * c.rudder * 6.0 + turn + asym) - beta * 1.2
    roll = jnp.where(ground, roll, jnp.clip(beta * 1.5 * dt + roll, -75.0, 75.0))
    steer_max = jnp.where(pitch <= 2.0, jnp.clip((v - 3.0) / 22.0, 0.0, 1.0) * -51.0 + 55.0, 0.0)
    steer_cmd = jnp.clip(c.steer, -1.0, 1.0)
    steer = move_towards(s.steer_angle, jnp.where(ground, steer_cmd * steer_max, 0.0),
                         dt * jnp.where(ground, 90.0, 60.0))
    signed_speed = jnp.where(backward, -v, v)
    yaw_steer = jnp.clip(jnp.tan(steer * RAD) * signed_speed / t.wheelbase * DEG, -35.0, 35.0)
    rud_eff = jnp.clip(v / 5.0 + 0.3, 0.0, 1.0)
    ground_yaw = asym * 0.5 * rud_eff + c.rudder * 6.0 * f + yaw_steer
    ground_yaw = jnp.where(belly, ground_yaw * 0.2, ground_yaw)
    lat = jnp.abs(v * ground_yaw * RAD)
    skid_turn = ground & (lat > 4.4145)
    # Keep the inactive zero-lateral-acceleration path finite under autodiff.
    denom = jnp.where(skid_turn, lat, 1.0)
    ground_yaw = jnp.where(skid_turn, ground_yaw * (4.4145 / denom), ground_yaw)
    v = jnp.where(skid_turn, jnp.maximum(0.0, v - dt * 2.5), v)
    start_drift = ground & hand_fun & ~belly & ~backward & (v > 4.0) & (v < 22.0) & (jnp.abs(c.steer) > 0.3)
    drifting = s.drifting | start_drift
    skidding = skidding | skid_turn | start_drift
    drift_rate = jnp.clip((v - 4.0) / 6.0, 0.0, 1.0) * 28.0
    ground_yaw = jnp.where(start_drift, drift_rate * steer_cmd + ground_yaw * 0.5, ground_yaw)
    yaw = jnp.where(ground, ground_yaw, air_yaw)
    heading = fmod360(yaw * dt + s.heading)
    lag_track = ground & (s.track_lag | start_drift) & t.handbrake_fun & (v > 0.5) & ~backward
    max_delta = dt * 50.0
    delta = wrap180(heading - s.track)
    limited_delta = jnp.where(max_delta < jnp.abs(delta), jnp.sign(delta) * max_delta, delta)
    lagged_track = fmod360(jnp.where((delta <= -max_delta) | (max_delta <= delta),
                                     limited_delta + s.track, heading))
    wheel_track = jnp.where(backward, fmod360(heading + 180.0), heading)
    track = jnp.where(ground, jnp.where(lag_track, lagged_track, wheel_track), air_track)
    lag_track = lag_track & (drifting | (jnp.abs(wrap180(track - heading)) >= 0.05))

    # A liftoff tick already blends wind and cannot touch down on this tick.
    still_grounded = ground & ~liftoff
    wind = jnp.where(still_grounded, 0.0, move_towards(s.wind_factor, 1.0, dt / 2.5))
    cg = jnp.cos(gamma * RAD)
    vx = wind * s.wx + jnp.sin(track * RAD) * cg * v
    vz = wind * s.wz + jnp.cos(track * RAD) * cg * v
    vy = wind * s.wy + jnp.sin(gamma * RAD) * v
    px, py, pz = vx * dt + s.px, vy * dt + s.py, vz * dt + s.pz
    contact = g1.height - jnp.where(wheels, 0.0, 1.15)
    touchdown = ~ground & (py <= contact)
    left_ground = still_grounded & (g1.height < g0.height - 0.6)
    follow_surface = still_grounded & ~left_ground
    diff = contact - py
    follow_y = move_towards(py, contact, jnp.where(diff <= 0.0, 2.0, 0.5) * dt)
    py = jnp.where(touchdown, contact,
                   jnp.where(follow_surface, jnp.where(jnp.abs(diff) < 0.3, contact, follow_y), py))

    return State(
        px=px, py=py, pz=pz, speed=v, gamma=jnp.where(touchdown, 0.0, gamma),
        track=track, heading=heading, pitch=jnp.where(touchdown, jnp.clip(pitch, 0.0, 10.0), pitch),
        roll=roll, prate=prate, rrate=rrate, yrate=yaw, on_ground=follow_surface | touchdown,
        gear_pos=gear, flap_pos=flap, steer_angle=steer, wx=s.wx, wy=s.wy, wz=s.wz,
        wind_factor=jnp.where(touchdown, 0.0, wind), alpha=alpha, alpha_req=a_req,
        load_factor=load, ground_y=g1.height, agl=jnp.maximum(0.0, py - g1.height),
        ground_effect=effect, wow=wow, surface=g1.surface, stalling=~ground & (v0 > 1.0) & (alpha > 14.0),
        stall_warn=~ground & (v0 > 1.0) & (alpha > 11.9), skidding=skidding, belly=belly,
        overspeed=v0 > 90.0, braking=braking, drifting=drifting, track_lag=lag_track, backward=backward,
    )
