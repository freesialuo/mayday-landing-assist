# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Client-observable game-twin guidance for a bounded-residual experiment."""

import jax.numpy as jnp

from mayday_jax.environment import GATE_DISTANCES, GLIDE_SLOPE
from mayday_jax.physics import G, RAD, alpha_required, level_drag, wrap180
from .environment import AIM_POINT_OFFSET, FLARE, GO_AROUND, ROLLOUT
from .features import INDEX
from .mechanics import RUNNING


PRIOR_SCHEMA = "mayday-landing-observable-guidance-v2"


# Tunable constants, in order. ``gains`` scales each multiplicatively; the
# default (None) keeps the shipped values bit for bit.
GAIN_NAMES = (
    "final_lookahead", "flare_lookahead", "turn_gain", "beta_gain", "roll_gain", "roll_damping",
    "glide_gain", "flare_sink_gain", "flare_sink_min", "flare_height", "approach_speed", "flare_speed",
    "pitch_gain", "pitch_damping", "speed_gain", "idle_height", "beta_yaw", "flare_bank", "approach_bank",
    # Observation-age lead on bank and pitch (1.0 = 0.25/0.18 per second of age). Doubling it helped on
    # one seed set but did not replicate on fresh seeds (plan §9), and 4x was unstable at 600 ms.
    "roll_lead", "pitch_lead",
)


def guidance(observation, phase, tuning, gains=None):
    k = (lambda i: 1.0) if gains is None else (lambda i: gains[i])
    feature = lambda name: observation[INDEX[name]]
    age = jnp.clip(feature("observation_age"), 0.0, 0.75)
    along = feature("along") - AIM_POINT_OFFSET
    cross = feature("cross")
    height = feature("agl")
    ground_speed = feature("ground_speed")
    measured_speed = feature("speed")
    speed = jnp.maximum(measured_speed, 15.0)
    sink = feature("sink")
    east_wind, right_wind = feature("wind_along"), feature("wind_cross")
    track_sin, track_cos = feature("track_sin"), feature("track_cos")
    vx, vy = ground_speed * track_cos, ground_speed * track_sin
    track = jnp.arctan2(vy, vx) / RAD
    heading = jnp.arctan2(feature("heading_sin"), feature("heading_cos")) / RAD
    air_track = jnp.arctan2(vy - right_wind, vx - east_wind) / RAD
    beta = wrap180(heading - air_track)
    along += age * vx
    cross += age * vy
    height = jnp.maximum(0.0, height - age * sink)
    gear, flap = feature("gear"), feature("flap")
    pitch, bank = feature("pitch"), feature("roll")
    pitch_rate, roll_rate, yaw_rate = feature("pitch_rate"), feature("roll_rate"), feature("yaw_rate")
    yaw_prediction = heading + jnp.minimum(0.3, age) * yaw_rate
    is_ground = (feature("on_ground") > 0.5) | (phase == ROLLOUT)
    circuit = phase == GO_AROUND
    flare = ((phase == FLARE) | (height <= 10.5 * k(9))) & ~circuit

    to_ring = jnp.maximum(-GATE_DISTANCES[0] - along, 0.0)
    lookahead = jnp.where(along < -4000.0, jnp.clip(0.47 * to_ring, 190.0, 450.0),
                          jnp.where(flare, 155.0 * k(1), 260.0 * k(0)))
    path_track = jnp.clip(jnp.arctan2(-cross, lookahead) / RAD, -48.0, 48.0)
    waypoint_along = feature("target_along") - AIM_POINT_OFFSET
    waypoint_cross = feature("target_cross")
    circuit_track = jnp.arctan2(waypoint_cross - cross, waypoint_along - along) / RAD
    desired_track = jnp.where(circuit, circuit_track, path_track)
    turn = wrap180(desired_track - track)
    bound = jnp.where(circuit, 30.0, jnp.where(flare, 6.0 * k(17), 22.0 * k(18)))
    target_bank = jnp.clip(0.79 * k(2) * turn - 0.75 * k(3) * beta, -bound, bound)
    roll = (0.062 * k(4) / (1.0 + 1.6 * age)) * (target_bank - bank - (0.22 * k(5) + 0.25 * k(19) * age) * roll_rate)

    levels = jnp.stack((feature("left_health"), feature("right_health")))
    burning = jnp.stack((feature("left_fire"), feature("right_fire"))) > 0.0
    running = jnp.stack((feature("left_state"), feature("right_state"))) == RUNNING
    engines_available = jnp.maximum(0.25, jnp.sum(running * (0.45 + 0.55 * levels) *
                                                   jnp.where(burning, 0.7, 1.0)))
    glide_height = jnp.maximum(-along * GLIDE_SLOPE, 0.0)
    glide_vs = jnp.clip(-GLIDE_SLOPE * jnp.maximum(10.0, vx) + 0.092 * k(6) * (glide_height - height), -4.9, 2.8)
    flare_vs = -jnp.clip(0.18 * k(7) * height, 0.75 * k(8), 1.7)
    circuit_height = jnp.where((waypoint_along < -3800.0) & (jnp.abs(waypoint_cross) < 500.0),
                               jnp.maximum(240.0, glide_height), 235.0)
    circuit_vs = jnp.clip(0.09 * (circuit_height - height), -2.4,
                          jnp.where(engines_available > 1.5, 5.3, 1.7))
    # Speed priority: with iced or derated engines, trade climb for airspeed
    # instead of holding the climb at the stall warning (icing go-around timeouts).
    circuit_vs = jnp.minimum(circuit_vs, jnp.maximum(0.35 * (measured_speed - 44.0), 0.0))
    vs = jnp.where(circuit, circuit_vs, jnp.where(flare, flare_vs, glide_vs))
    gamma = jnp.arcsin(jnp.clip(-sink / speed, -0.65, 0.65)) / RAD
    desired_gamma = jnp.arcsin(jnp.clip(vs / speed, -0.25, 0.25)) / RAD
    minimal_bank_cos = jnp.maximum(jnp.cos(bank * RAD), 0.75)
    lift = jnp.clip(jnp.cos(gamma * RAD) + 0.085 * (desired_gamma - gamma), 0.44, 1.62) / minimal_bank_cos
    effect = jnp.clip(1.0 - height / 12.5, 0.0, 1.0) ** 2
    target_pitch = jnp.clip(gamma + alpha_required(speed, flap, effect, tuning) * lift - 2.0,
                            -10.0, jnp.where(flare, 6.8, 14.0))
    pitch_gain = 1.0 - 0.55 * jnp.clip((12.0 - height) / 8.0, 0.0, 1.0)
    pitch_command = (0.36 * k(12) / (1.0 + 1.9 * age)) * (
        target_pitch - pitch - (0.19 * k(13) + 0.18 * k(20) * age) * pitch_rate) / pitch_gain
    pitch_command = jnp.where(jnp.abs(pitch_command) >= 0.022,
                              jnp.copysign(jnp.maximum(jnp.abs(pitch_command), 0.055), pitch_command), 0.0)

    yaw = jnp.where(flare & (height < 3.0), -0.070 * yaw_prediction, -0.20 * k(16) * beta)
    yaw -= 0.58 * (feature("left_thrust") - feature("right_thrust"))
    ground_yaw = -jnp.clip(0.040 * (yaw_prediction + jnp.arctan2(cross, 38.0) / RAD)
                           + 0.026 * yaw_rate, -0.78, 0.78)
    drag = level_drag(speed, gear, flap, 1.0)
    wanted_speed = jnp.where(circuit, 52.0, jnp.where(flare, 43.0 * k(11), 47.0 * k(10)))
    thrust = 0.95 * engines_available * (1.0 - 0.35 * feature("ice"))
    throttle = (drag + G * jnp.sin(desired_gamma * RAD)) / thrust + 0.08 * k(14) * (wanted_speed - measured_speed)
    throttle = jnp.where(flare & (height < 5.5 * k(15)), 0.0, throttle)
    action = jnp.stack((jnp.where(is_ground, -0.42, pitch_command),
                        jnp.where(is_ground, 0.0, roll), jnp.where(is_ground, ground_yaw, yaw),
                        jnp.where(is_ground, 0.0, throttle), is_ground.astype(jnp.float32)))
    return jnp.clip(action, jnp.asarray((-1.0, -1.0, -1.0, 0.0, 0.0), jnp.float32), 1.0)
