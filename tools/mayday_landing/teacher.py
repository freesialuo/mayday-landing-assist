"""Privileged, causal MPPI for the Mayday game landing twin."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from mayday_jax.environment import GATE_DISTANCES, GLIDE_SLOPE, runway_coordinates
from mayday_jax.physics import RAD, G, level_drag, wrap180

from .environment import CRASH, FLARE, GO_AROUND, LEFT_AREA, LEGACY_WEATHER, OFF_RUNWAY, ROLLOUT, STOPPED, step


ACTION_LOW = (-1.0, -1.0, -1.0, 0.0, 0.0)


@dataclass(frozen=True)
class PlannerConfig:
    samples: int = 7
    horizon: int = 8
    temperature: float = 0.40
    noise: tuple = (0.12, 0.10, 0.13, 0.13, 0.12)
    # False removes future gust and co-pilot pulses from the prediction model
    # (see blind_disturbances); a pulse already in progress is kept.
    clairvoyant: bool = True
    # DIAL-MPC style annealing: iterations per step, each re-centred on the
    # previous weighted mean with the noise halved (1, 0.5, 0.25, ...).
    anneal: int = 1

    def __post_init__(self):
        if (self.samples < 3 or self.samples % 2 != 1 or self.horizon < 1 or self.temperature <= 0.0
                or self.anneal < 1):
            raise ValueError("MPPI needs an odd sample count >= 3, positive horizon and temperature")


def clip_action(action):
    return jnp.clip(action, jnp.asarray(ACTION_LOW, dtype=jnp.float32), 1.0)


def _guidance(state):
    """Targets in runway coordinates, shared by the feedback prior and MPPI cost."""
    aircraft = state.aircraft
    phase = state.phase
    circuit = phase == GO_AROUND
    flare = phase == FLARE
    speed = jnp.maximum(aircraft.speed, 15.0)
    wind_along, wind_cross = runway_coordinates(
        aircraft.wx * aircraft.wind_factor, aircraft.wz * aircraft.wind_factor, state.flight.runway_heading)

    dist_to_ring = jnp.maximum(-GATE_DISTANCES[0] - state.along, 0.0)
    intercept_lookahead = jnp.clip(dist_to_ring * 0.45, 190.0, 420.0)
    lookahead = jnp.where(state.along < -GATE_DISTANCES[0], intercept_lookahead,
                          jnp.where(flare, 140.0, 250.0))
    intercept = jnp.clip(jnp.arctan2(-state.lateral, lookahead) / RAD, -48.0, 48.0)
    target, waypoint_cross = state.go_around_waypoint
    waypoint_track = jnp.arctan2(waypoint_cross - state.lateral, target - state.along) / RAD
    target_track = jnp.where(circuit, waypoint_track, intercept)
    wind_perpendicular = wind_cross * jnp.cos(target_track * RAD) - wind_along * jnp.sin(target_track * RAD)
    target_air_track = wrap180(target_track - jnp.arcsin(jnp.clip(wind_perpendicular / speed, -0.3, 0.3)) / RAD)

    height = aircraft.py - state.flight.runway_elev
    glide_height = jnp.maximum(-state.along * GLIDE_SLOPE, 0.0)
    approach_vs = jnp.clip(-GLIDE_SLOPE * (speed + wind_along) + 0.095 * (glide_height - height), -5.0, 3.0)
    flare_vs = -jnp.clip(0.18 * height, 0.7, 1.8)
    recovering = state.go_around_stage >= 3
    circuit_height = jnp.where(recovering, jnp.maximum(240.0, glide_height), 235.0)
    twin_thrust = jnp.count_nonzero(state.engines.running) == 2
    circuit_vs = jnp.clip(0.09 * (circuit_height - height), -2.5, jnp.where(twin_thrust, 5.5, 1.8))
    target_vs = jnp.where(circuit, circuit_vs, jnp.where(flare, flare_vs, approach_vs))
    target_speed = jnp.where(circuit, 52.0, jnp.where(flare, 43.0, 47.0))
    return target_air_track, target_vs, target_speed, glide_height


def managed_action(state):
    """Deterministic hand-written comparison and feedback prior in the game twin."""
    aircraft = state.aircraft
    phase = state.phase
    circuit = phase == GO_AROUND
    flare = phase == FLARE
    track, vs, speed_target, _ = _guidance(state)
    speed = jnp.maximum(aircraft.speed, 15.0)
    track_rel = wrap180(aircraft.track - state.flight.runway_heading)
    heading_rel = wrap180(aircraft.heading - state.flight.runway_heading)
    beta = wrap180(aircraft.heading - aircraft.track)
    lag = jnp.asarray(state.scenario.command_lag, jnp.float32) * 0.1
    bank_bound = jnp.where(circuit, 30.0, jnp.where(flare, 6.0, 22.0))
    desired_bank = jnp.clip(0.84 * wrap180(track - track_rel) - 0.95 * beta, -bank_bound, bank_bound)
    roll = 0.084 * (desired_bank - aircraft.roll - (0.16 + lag) * aircraft.rrate)

    # Climb/descend at the requested vertical speed by commanding the lift
    # needed to close the flight-path-angle error. The manual law's tiny
    # command deadband is handled before issuing the stick command.
    gamma = jnp.arcsin(jnp.clip(vs / speed, -0.25, 0.25)) / RAD
    bank_cos = jnp.maximum(jnp.cos(aircraft.roll * RAD), 0.75)
    desired_load = jnp.clip(jnp.cos(aircraft.gamma * RAD) +
                            0.095 * (gamma - aircraft.gamma), 0.44, 1.62) / bank_cos
    desired_pitch = jnp.clip(aircraft.gamma + aircraft.alpha_req * desired_load - 2.0, -10.0,
                             jnp.where(flare, 6.8, 14.0))
    ground_pitch_scale = 1.0 - 0.55 * jnp.clip((12.0 - aircraft.agl) / 8.0, 0.0, 1.0)
    pitch = (0.45 / ground_pitch_scale) * (
        desired_pitch - aircraft.pitch - (0.13 + lag) * aircraft.prate)
    pitch = jnp.where(jnp.abs(pitch) > 0.020,
                      jnp.copysign(jnp.maximum(jnp.abs(pitch), 0.055), pitch), 0.0)

    ground_yaw = -jnp.clip(0.041 * (heading_rel + jnp.arctan2(state.lateral, 34.0) / RAD)
                           + (0.023 + 0.055 * lag) * aircraft.yrate, -0.78, 0.78)
    yaw = jnp.where(flare & (aircraft.agl < 3.0), -0.071 * heading_rel, -0.22 * beta)
    yaw -= 0.58 * (state.engine_thrust[0] - state.engine_thrust[1])

    drag = level_drag(speed, aircraft.gear_pos, aircraft.flap_pos, 1.0)
    thrust_fraction = (0.45 + 0.55 * state.engines.health) * jnp.where(state.engines.fire > 0.0, 0.7, 1.0)
    equivalent_engines = jnp.maximum(0.25, jnp.sum(state.engines.running * thrust_fraction))
    ice = jnp.where(state.scenario.weather_kind == LEGACY_WEATHER, state.scenario.ice, state.weather_sample.airframe_ice)
    available_thrust = equivalent_engines * 0.95 * (1.0 - 0.35 * ice)
    required = (drag + jnp.sin(gamma * RAD) * G) / available_thrust
    closing_speed = speed_target - aircraft.speed
    throttle = jnp.clip(required + 0.09 * closing_speed, 0.0, 1.0)
    throttle = jnp.where(flare & (aircraft.agl < 5.5), 0.0, throttle)
    ground = aircraft.on_ground | (phase == ROLLOUT)
    return clip_action(jnp.stack((jnp.where(ground, -0.42, pitch),
                                  jnp.where(ground, 0.0, roll),
                                  jnp.where(ground, ground_yaw, yaw),
                                  jnp.where(ground, 0.0, throttle),
                                  ground.astype(jnp.float32))))


def causal_state(state):
    """Remove event schedule entries whose transition has yet to be observed.

    Preserve active dynamics, including engines, in-progress repair and the
    actuator queue. Under the counterfactual model started repair may continue,
    but the oracle may not anticipate an unseen pause, restart or go-around.
    """
    schedule = state.scenario
    already = lambda time: jnp.where(time <= state.elapsed, time, jnp.inf)
    active_fault = state.fault_detected
    spraying = ((state.elapsed >= schedule.extinguish_time)
                & (state.elapsed < schedule.extinguish_time + schedule.extinguish_duration))
    return state._replace(scenario=schedule._replace(
        failure_time=jnp.where(active_fault, schedule.failure_time, jnp.inf),
        failure_side=jnp.where(active_fault, schedule.failure_side, -1),
        repair_start=already(schedule.repair_start),
        repair_interrupt=already(schedule.repair_interrupt),
        repair_resume=already(schedule.repair_resume),
        # Completion is another hidden boundary, even during active work.
        repair_duration=jnp.where(state.engine_restarted, schedule.repair_duration, jnp.inf),
        permanent=jnp.bool_(False),
        go_around_time=jnp.where(state.go_around_started, schedule.go_around_time, jnp.inf),
        fire_intensity=jnp.where(active_fault, schedule.fire_intensity, 0.0),
        engine_shutdown_time=jnp.where(state.engine_shutdown_commanded, schedule.engine_shutdown_time, jnp.inf),
        extinguish_time=jnp.where(spraying, state.elapsed, jnp.inf),
        extinguish_duration=jnp.where(spraying, jnp.inf, 0.0),
    ))


_NEVER = 1e9


def blind_disturbances(state):
    """Hide pulses that have not started yet; keep the current one to its end.

    The twin schedules gusts and co-pilot inputs periodically, so an oracle that
    rolls the true schedule forward knows the next pulse exactly. Smooth
    turbulence sinusoids and true (undelayed) state remain privileged.
    """
    weather = state.weather
    gust_time = state.elapsed - weather.first_gust
    in_gust = (gust_time >= 0.0) & (jnp.mod(gust_time, weather.gust_period) < 0.3)
    gust_start = state.elapsed - jnp.mod(gust_time, weather.gust_period)
    weather = weather._replace(first_gust=jnp.where(in_gust, gust_start, jnp.inf),
                               gust_period=jnp.full_like(weather.gust_period, _NEVER))
    d = state.flight.disturbances
    since = state.elapsed - d.competing_start
    in_pulse = (since >= 0.0) & (jnp.mod(since, d.competing_period) < d.competing_duration)
    pulse = jnp.floor(since / d.competing_period)
    pulse_start = d.competing_start + pulse * d.competing_period
    # Keep the sign of the current pulse: an odd pulse index flips direction.
    flip = jnp.where(jnp.mod(pulse, 2.0) == 1.0, -1.0, 1.0)
    d = d._replace(competing_start=jnp.where(in_pulse, pulse_start, jnp.inf),
                   competing_period=jnp.full_like(d.competing_period, _NEVER),
                   competing_axes=d.competing_axes * flip)
    return state._replace(weather=weather, flight=state.flight._replace(disturbances=d))


def _state_cost(current, following, action, residual):
    aircraft = following.aircraft
    track_target, vs_target, speed_target, glide_height = _guidance(following)
    phase = following.phase
    circuit = phase == GO_AROUND
    flare = phase == FLARE
    ground = aircraft.on_ground
    track = wrap180(aircraft.track - following.flight.runway_heading)
    vs = aircraft.speed * jnp.sin(aircraft.gamma * RAD)
    track_cost = (wrap180(track - track_target) / 20.0) ** 2
    speed_cost = ((aircraft.speed - speed_target) / 8.0) ** 2
    vertical_cost = ((vs - vs_target) / 2.5) ** 2
    glide_cost = ((aircraft.agl - glide_height) / 18.0) ** 2
    lateral_scale = jnp.clip(-following.along / 14.0, 13.0, 105.0)
    lateral_cost = (following.lateral / lateral_scale) ** 2
    ground_cost = (following.lateral / 11.0) ** 2 + (track / 11.0) ** 2 + (aircraft.speed / 25.0) ** 2
    bank_cost = (aircraft.roll / jnp.where(flare, 8.0, 32.0)) ** 2
    gate_changes = current.gate_states != following.gate_states
    missed = jnp.count_nonzero(gate_changes & (following.gate_states == 2))
    passed = jnp.count_nonzero(gate_changes & (following.gate_states == 1))
    hard_sink = jnp.maximum(following.flight.touchdown_vertical_speed - 1.75, 0.0)
    impact = (following.touchdown_count > current.touchdown_count) * (5.0 * hard_sink ** 2 - 4.0)
    catastrophe = ((following.termination == OFF_RUNWAY) | (following.termination == CRASH)
                   | (following.termination == LEFT_AREA))
    terminal = (~current.done & following.done) * (180.0 * catastrophe - 8.0 * (following.termination == STOPPED))
    air_cost = (0.42 * track_cost + 0.36 * speed_cost + 1.05 * vertical_cost + 0.19 * bank_cost
                + jnp.where(circuit, 0.0, lateral_cost + jnp.where(flare, 0.0, glide_cost))
                + 6.0 * aircraft.stall_warn + 14.0 * aircraft.stalling)
    regularization = 0.27 * jnp.sum(residual**2) + 0.12 * jnp.sum((action - current.command_history[0]) ** 2)
    # Ground braking is binary in the game. Brake during rollout and restrain
    # a candidate that tries to use throttle against the brake.
    ground_effort = (1.0 - action[4]) ** 2 + 4.0 * action[3] ** 2
    return jnp.where(current.done, 0.0, jnp.where(ground, ground_cost + ground_effort, air_cost)
                     + regularization + 60.0 * missed - 8.0 * passed + impact + terminal)


def _rollout_cost(state, residuals, tuning):
    def advance(current, residual):
        action = clip_action(managed_action(current) + residual)
        following = step(current, action, tuning)._replace(observation_history=current.observation_history)
        return following, _state_cost(current, following, action, residual)

    _, cost = jax.lax.scan(advance, state, residuals)
    return jnp.sum(cost)


def plan(state, key, tuning, config):
    """Execute the first control of an MPPI update about a feedback prior.

    The unperturbed control and antithetic residuals share one initial state.
    Smooth residuals retain actuator influence throughout the delayed horizon.
    """
    state_for_prediction = causal_state(state)
    if not config.clairvoyant:
        state_for_prediction = blind_disturbances(state_for_prediction)
    noise_shape = ((config.samples - 1) // 2, config.horizon, 5)
    noise = jax.random.normal(key, noise_shape, jnp.float32)
    noise = (0.65 * noise[:, :1, :] + 0.35 * noise) * jnp.asarray(config.noise, jnp.float32)
    residuals = jnp.concatenate((jnp.zeros((1, config.horizon, 5), jnp.float32), noise, -noise))
    costs = jax.vmap(_rollout_cost, in_axes=(None, 0, None))(state_for_prediction, residuals, tuning)
    minimum = jnp.min(costs)
    scale = jnp.maximum(0.8, (jnp.mean(costs) - minimum) * config.temperature)
    weight = jax.nn.softmax(-(costs - minimum) / scale)
    return clip_action(managed_action(state) + weight @ residuals[:, 0])


@partial(jax.jit, static_argnames=("config",))
def plan_batch(states, keys, tuning, config):
    return jax.vmap(plan, in_axes=(0, 0, None, None))(states, keys, tuning, config)
