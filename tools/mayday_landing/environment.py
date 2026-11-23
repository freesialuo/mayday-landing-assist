"""Game twin for a continuous intercept, approach, and runway stop.

The runway is flat and rectangular. Gate geometry uses the navigation aimpoint,
250 m past the physical threshold. Each 10 Hz command holds for six 60 Hz
physics frames. Separate seeded packet channels model delay, jitter and loss
in either direction. The clocks are ideal and there is no interpolation.
Legacy scenarios reuse reference approach disturbances. Opt-in profiles add
local rain, fog, wind and icing. Rain and fog alter telemetry; accumulated ice
derates thrust. The gear command stays down. A jammed gear follows the native
hand crank (+0.125 per crank, cleared at 1.0 once the leg reaches 0.999) when
the scenario schedules crew cranking.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from mayday_jax import environment as approach
from mayday_jax.physics import RAD, Controls, Ground, alpha_required, wrap180
from mayday_jax.physics import step as flight_step

from .features import NAMES
from .mechanics import OFF, RUNNING as RUNNING_ENGINE, EngineState, initial_engines, step_engines
from .weather import GUSTS, LOW_VIS_WINDY, WEATHER_NAMES, WeatherSample, WeatherScenario, sample_weather, weather_at


CAPTURE, FINAL, FLARE, ROLLOUT, GO_AROUND = range(5)
RUNNING, STOPPED, OFF_RUNWAY, TIMEOUT, LEFT_AREA, CRASH, HANDBACK = range(7)
PHYSICS_DT = 1.0 / 60.0
PHYSICS_PER_CONTROL = 6
CONTROL_DT = PHYSICS_DT * PHYSICS_PER_CONTROL
MAX_LAG_STEPS = 6
AIM_POINT_OFFSET = 250.0
RUNWAY_LENGTH = approach.RUNWAY_LENGTH
RUNWAY_HALF_WIDTH = approach.RUNWAY_HALF_WIDTH
STOP_SPEED = 0.1
STOP_FRAMES = 180
# Native PlaneFlight constants (GameAssembly RVA 0xB0C480 ServerCrankGear, FixedUpdate jam clear).
GEAR_CRANK_STEP = 0.125
GEAR_CLEAR_POSITION = 0.999
GEAR_DOWN_THRESHOLD = 0.9
GEAR_DECISION_ALONG = -1200.0
REVERSE_THROTTLE = 0.8
REVERSE_OFF_SPEED = 10.0
# Twin-only cap (the game has none). 480 s left 10-30 s of slack for a
# fault-free go-around circuit, so it was raised to 600 s.
MAX_EPISODE_SECONDS = 600.0
MAX_DISTANCE = 12_000.0
FAR_DISTANCE = 30_000.0
FAR_EPISODE_SECONDS = 900.0
DEFAULT_THROTTLE = 0.45
ACTION_SIZE = 5
LEGACY_WEATHER = -1
MIXED_WEATHER = -2

# Targets are in aimpoint-relative runway coordinates. The final segment
# restores the intercept before the first ring with the same flight state.
GO_AROUND_WAYPOINTS = (
    (1500.0, 0.0), (1500.0, 1500.0), (-4900.0, 1500.0),
    (-5100.0, 0.0), (-3750.0, 0.0),
)

OBSERVATION_NAMES = NAMES
OBSERVATION_SIZE = len(NAMES)
PACKET_SIZE = OBSERVATION_SIZE + 2
_OBSERVATION_AGE = NAMES.index("observation_age")
_LAST_ACTION = NAMES.index("last_pitch")


class Scenario(NamedTuple):
    difficulty: float = 0.5
    observation_lag: int = 0
    command_lag: int = 0
    failure_time: float = float("inf")
    failure_side: int = -1
    repair_start: float = float("inf")
    repair_interrupt: float = float("inf")
    repair_resume: float = float("inf")
    repair_duration: float = 8.0
    permanent: bool = False
    go_around_time: float = float("inf")
    rain: float = 0.0
    visibility: float = 30_000.0
    ice: float = 0.0
    other_known: bool = True
    observation_jitter: int = 0
    command_jitter: int = 0
    observation_loss: float = 0.0
    command_loss: float = 0.0
    fire_intensity: float = 0.0
    engine_shutdown_time: float = float("inf")
    extinguish_time: float = float("inf")
    extinguish_duration: float = 2.0
    weather_kind: int = LEGACY_WEATHER
    weather_leg: int = -1
    # Crew hand-crank schedule for a jammed gear: first crank time and seconds per crank.
    crank_start: float = float("inf")
    crank_interval: float = 0.6
    # Assist policy: go around once if the gear is unlocked at the decision ring.
    gear_go_around: bool = False
    # Keep intercept starts inside the mod's engagement window (|cross| <= 220 m).
    engageable_starts: bool = False
    # Native front gusts also kick bank by +/-12 deg and heading by +/-2 deg at onset
    # (PlaneFlight.OnHazard case 5 sets gustDir = +/-1; ApplyTurbulence applies it).
    gust_kick: bool = False
    # Assist rollout: reverse (CmdPilotAction 4) with REVERSE_THROTTLE above REVERSE_OFF_SPEED, both engines running.
    auto_reverse: bool = False
    # Far starts for an armed join: 9-20 km out, 0.5-4 km off the centreline, 300-900 m AGL,
    # heading +/-60 deg from the direct track to the capture area. Allows FAR_EPISODE_SECONDS.
    far_start: bool = False
    # Wider but still local starts: 4.85-9 km from the threshold (before ring 1), +/-600 m off the
    # centreline, +/-120 m off the glide path (>= 80 m AGL), 35-70 m/s, heading +/-45 deg, bank +/-25 deg.
    wide_start: bool = False
    # Crew gear-down command time (0 = already down at the start, inf = never). The assist never
    # moves the gear; with gear_go_around it goes around once and then hands back (HANDBACK).
    gear_down_time: float = 0.0


class Transport(NamedTuple):
    due: jax.Array
    delivered_sequence: jax.Array
    packet: jax.Array


class LandingState(NamedTuple):
    flight: approach.EnvState
    scenario: Scenario
    engines: EngineState
    weather: WeatherScenario
    weather_sample: WeatherSample
    phase: jax.Array
    applied_action: jax.Array
    command_history: jax.Array
    observation_history: jax.Array
    command_transport: Transport
    observation_transport: Transport
    packet_key: jax.Array
    sequence: jax.Array
    fault_detected: jax.Array
    repair_elapsed: jax.Array
    engine_restarted: jax.Array
    engine_shutdown_commanded: jax.Array
    go_around_started: jax.Array
    go_around_stage: jax.Array
    go_around_completed: jax.Array
    intercept_completed: jax.Array
    touchdown_count: jax.Array
    touchdown_roll: jax.Array
    touchdown_speed: jax.Array
    max_rollout_lateral: jax.Array
    stop_frames: jax.Array
    gear_jammed: jax.Array
    gear_jam_pos: jax.Array
    cranks: jax.Array

    @property
    def aircraft(self):
        return self.flight.aircraft

    @property
    def along(self):
        return self.flight.along

    @property
    def physical_along(self):
        return self.along + AIM_POINT_OFFSET

    @property
    def lateral(self):
        return self.flight.lateral

    @property
    def elapsed(self):
        return self.flight.elapsed

    @property
    def gate_states(self):
        return self.flight.gate_states

    @property
    def done(self):
        return self.flight.done

    @property
    def termination(self):
        return self.flight.termination

    @property
    def engine_thrust(self):
        return self.engines.thrust

    @property
    def stopped_seconds(self):
        return self.stop_frames * PHYSICS_DT

    @property
    def go_around_waypoint(self):
        return jnp.asarray(GO_AROUND_WAYPOINTS, dtype=jnp.float32)[self.go_around_stage]


def sample_scenario(key, difficulty=0.5, observation_lag=0, command_lag=0, *,
                    weather_kind=LEGACY_WEATHER, weather_leg=-1):
    """Sample normal, repairable and permanent faults in a seeded scenario.

    Go-arounds can be forced with ``Scenario._replace(go_around_time=...)``.
    Fault timing is in server seconds. Integrating the crew's active work time
    allows pauses without resetting repair progress.
    """
    difficulty = jnp.clip(jnp.float32(difficulty), 0.0, 1.0)
    draws = jax.random.uniform(key, (12,), dtype=jnp.float32)
    event = draws[0] < 0.4 * difficulty
    failure_time = jnp.where(event, 12.0 + 98.0 * draws[1], jnp.inf)
    side = jnp.where(event, (draws[2] > 0.5).astype(jnp.int32), -1)
    repair_start = failure_time + 3.0 + 10.0 * draws[3]
    interrupted = draws[4] < 0.5
    interrupt = jnp.where(interrupted, repair_start + 1.0 + draws[5], jnp.inf)
    resume = interrupt + 2.0 + 12.0 * draws[6]
    fire = event & (draws[8] < 0.35)
    contained = draws[9] > 0.35
    extinguish_at = jnp.where(fire & contained, failure_time + 3.0 + 7.0 * draws[10], jnp.inf)
    shut_at = jnp.where(fire & contained, failure_time + 1.0 + 2.0 * draws[11], jnp.inf)
    return Scenario(
        difficulty, observation_lag, command_lag, failure_time, side,
        repair_start, interrupt, resume, 6.0 + 4.0 * draws[7], jnp.where(fire, ~contained, draws[4] < 0.25),
        fire_intensity=jnp.where(fire, 0.35 + 0.45 * draws[10], 0.0),
        engine_shutdown_time=shut_at, extinguish_time=extinguish_at,
        weather_kind=weather_kind, weather_leg=weather_leg,
    )


def fire_scenario(base, *, failure_time, failure_side, extinguished, intensity=0.65):
    """Build paired fire cases sharing an ignition before crew action can differ."""
    return base._replace(
        failure_time=failure_time, failure_side=failure_side, fire_intensity=intensity,
        engine_shutdown_time=jnp.where(extinguished, failure_time + 3.0, jnp.inf),
        extinguish_time=jnp.where(extinguished, failure_time + 6.0, jnp.inf),
        repair_start=jnp.where(extinguished, failure_time + 8.0, jnp.inf),
        repair_interrupt=jnp.inf, repair_resume=jnp.inf, permanent=jnp.logical_not(extinguished),
    )


def _runway_surface(along, lateral):
    physical_along = along + AIM_POINT_OFFSET
    return (physical_along >= 0.0) & (physical_along <= RUNWAY_LENGTH) & (jnp.abs(lateral) <= RUNWAY_HALF_WIDTH)


def _intercept_reset(key, scenario, tuning):
    wind_key, intercept_key = jax.random.split(key)
    flight = approach.reset(wind_key, scenario.difficulty, tuning)
    ahead, cross, height, drift, bank = jax.random.uniform(intercept_key, (5,), dtype=jnp.float32)
    difficulty = jnp.clip(scenario.difficulty, 0.0, 1.0)
    along = -4700.0 - ahead * (400.0 + 1400.0 * difficulty)
    lateral_span = 230.0 + 520.0 * difficulty
    lateral_span = jnp.where(scenario.engageable_starts, jnp.minimum(lateral_span, 200.0), lateral_span)
    lateral = (2.0 * cross - 1.0) * lateral_span
    agl = -along * approach.GLIDE_SLOPE + (2.0 * height - 1.0) * (12.0 + 24.0 * difficulty)
    heading = (flight.runway_heading + jnp.arctan2(-lateral, -4000.0 - along) / RAD
               + (2.0 * drift - 1.0) * (6.0 + 12.0 * difficulty)) % 360.0
    roll = (2.0 * bank - 1.0) * (3.0 + 9.0 * difficulty)
    far_along = -9000.0 - 11000.0 * ahead
    far_lateral = jnp.where(cross < 0.5, -1.0, 1.0) * (500.0 + 3500.0 * jnp.abs(2.0 * cross - 1.0))
    far_agl = 300.0 + 600.0 * height
    far_heading = (flight.runway_heading + jnp.arctan2(-far_lateral, -6000.0 - far_along) / RAD
                   + (2.0 * drift - 1.0) * 60.0) % 360.0
    wide_key = jax.random.fold_in(intercept_key, 7)
    w_ahead, w_cross, w_height, w_heading, w_bank, w_speed = jax.random.uniform(wide_key, (6,), dtype=jnp.float32)
    wide_along = -4600.0 - 4150.0 * w_ahead
    wide_lateral = (2.0 * w_cross - 1.0) * 600.0
    wide_agl = jnp.maximum(80.0, -wide_along * approach.GLIDE_SLOPE + (2.0 * w_height - 1.0) * 120.0)
    wide_heading = (flight.runway_heading + (2.0 * w_heading - 1.0) * 45.0) % 360.0
    wide = scenario.wide_start
    along = jnp.where(wide, wide_along, along)
    lateral = jnp.where(wide, wide_lateral, lateral)
    agl = jnp.where(wide, wide_agl, agl)
    heading = jnp.where(wide, wide_heading, heading)
    roll = jnp.where(wide, (2.0 * w_bank - 1.0) * 25.0, roll)
    far = scenario.far_start
    along = jnp.where(far, far_along, along)
    lateral = jnp.where(far, far_lateral, lateral)
    agl = jnp.where(far, far_agl, agl)
    heading = jnp.where(far, far_heading, heading)
    east = jnp.sin(flight.runway_heading * RAD)
    north = jnp.cos(flight.runway_heading * RAD)
    aircraft = flight.aircraft
    aircraft = aircraft._replace(speed=jnp.where(wide, 35.0 + 35.0 * w_speed, aircraft.speed))
    load = jnp.cos(aircraft.gamma * RAD) / jnp.cos(roll * RAD)
    required = alpha_required(aircraft.speed, aircraft.flap_pos, 0.0, tuning)
    alpha = required * load
    aircraft = aircraft._replace(
        px=along * east + lateral * north, py=flight.runway_elev + agl,
        pz=along * north - lateral * east, agl=agl, heading=heading, track=heading,
        roll=roll, load_factor=load, alpha_req=required, alpha=alpha,
        pitch=aircraft.gamma + alpha - 2.0,
        gear_pos=jnp.where(scenario.gear_down_time > 0.0, 0.0, aircraft.gear_pos),
    )
    return flight._replace(aircraft=aircraft, along=along, lateral=lateral)


def _weather_profile(key, scenario):
    key = jax.random.fold_in(key, 87)
    sampled = sample_weather(key, scenario.difficulty)
    kind = jnp.clip(scenario.weather_kind, 0, len(WEATHER_NAMES) - 1)
    kind = jnp.where(scenario.weather_kind == MIXED_WEATHER, sampled.kind, kind)
    leg = jnp.where(scenario.weather_leg < 0, sampled.leg, jnp.clip(scenario.weather_leg, 1, 8))
    return sample_weather(key, scenario.difficulty, kind=kind, leg=leg)


def _weather_sample(scenario, profile, flight):
    def legacy():
        zero = jnp.float32(0.0)
        return WeatherSample(jnp.float32(scenario.rain), jnp.float32(scenario.visibility), zero,
                             flight.disturbances.turbulence, zero, jnp.float32(scenario.ice),
                             zero, zero, zero, zero)

    def local():
        return weather_at(profile, flight.elapsed, flight.aircraft.agl, flight.along + AIM_POINT_OFFSET, flight.lateral)

    return jax.lax.cond(scenario.weather_kind == LEGACY_WEATHER, legacy, local)


def _profile_wind(weather, runway_heading):
    east = jnp.sin(runway_heading * RAD)
    north = jnp.cos(runway_heading * RAD)
    return (-weather.headwind * east + weather.crosswind * north,
            -weather.headwind * north - weather.crosswind * east)


def reset(key, tuning, scenario=None):
    """Reset outside the first ring; all command and observation histories are local to this run."""
    if scenario is None:
        scenario_key, key = jax.random.split(key)
        scenario = sample_scenario(scenario_key)
    flight = _intercept_reset(key, scenario, tuning)
    weather = _weather_profile(key, scenario)
    sample = _weather_sample(scenario, weather, flight)
    wind_x, wind_z = _profile_wind(sample, flight.runway_heading)
    flight = flight._replace(aircraft=flight.aircraft._replace(wx=wind_x, wz=wind_z))
    action = jnp.asarray((0.0, 0.0, 0.0, DEFAULT_THROTTLE, 0.0), dtype=jnp.float32)
    integer = jnp.int32(0)
    real = jnp.float32(0.0)
    flag = jnp.bool_(False)
    history = jnp.zeros((MAX_LAG_STEPS + 1, PACKET_SIZE), dtype=jnp.float32)
    empty_due = jnp.full((MAX_LAG_STEPS + 1,), -1, dtype=jnp.int32)
    command = Transport(empty_due, integer, action)
    observation = Transport(empty_due, integer, history[0])
    state = LandingState(
        flight=flight, scenario=scenario, engines=initial_engines(DEFAULT_THROTTLE), phase=jnp.int32(CAPTURE),
        weather=weather, weather_sample=sample,
        applied_action=action, command_history=jnp.repeat(action[None, :], MAX_LAG_STEPS + 1, axis=0),
        observation_history=history, command_transport=command, observation_transport=observation,
        packet_key=jax.random.key_data(jax.random.fold_in(key, 41)), sequence=integer,
        fault_detected=flag, repair_elapsed=real, engine_restarted=flag, go_around_started=flag,
        engine_shutdown_commanded=flag,
        go_around_stage=integer, go_around_completed=flag, intercept_completed=flag,
        touchdown_count=integer, touchdown_roll=real, touchdown_speed=real, max_rollout_lateral=real,
        stop_frames=integer, gear_jammed=flight.disturbances.gear_jam < 1.0,
        gear_jam_pos=jnp.asarray(flight.disturbances.gear_jam, jnp.float32), cranks=integer,
    )
    packet = _observation_packet(state)
    return state._replace(observation_history=jnp.repeat(packet[None, :], MAX_LAG_STEPS + 1, axis=0),
                          observation_transport=observation._replace(packet=packet))


def _update_engines(state, throttle, dt, airframe_ice):
    scenario = state.scenario
    side = jnp.arange(2, dtype=jnp.int32) == scenario.failure_side
    fault = jnp.any(side) & (state.elapsed >= scenario.failure_time)
    just_failed = fault & ~state.fault_detected
    is_fire = scenario.fire_intensity > 0.0
    new_health = jnp.where(scenario.permanent, 0.0, 0.55)
    engines = state.engines._replace(
        health=jnp.where(just_failed & side & ~is_fire, new_health, state.engines.health),
        fire=jnp.where(just_failed & side & is_fire,
                       jnp.maximum(state.engines.fire, jnp.clip(scenario.fire_intensity, 0.0, 1.0)), state.engines.fire),
    )
    # Deliver the shutdown once, allowing a later crew restart. A direct failure
    # enters Off or Dead at once; a fire can burn while producing reduced thrust.
    fire_shutdown = fault & is_fire & (state.elapsed >= scenario.engine_shutdown_time)
    shutdown = side & ((just_failed & ~is_fire) | (fire_shutdown & ~state.engine_shutdown_commanded))
    spraying = fault & is_fire & (state.elapsed >= scenario.extinguish_time)
    spraying &= state.elapsed < scenario.extinguish_time + scenario.extinguish_duration
    repairable = fault & jnp.logical_not(scenario.permanent)
    work_active = repairable & (state.elapsed >= scenario.repair_start) & (
        (state.elapsed < scenario.repair_interrupt) | (state.elapsed >= scenario.repair_resume)
    )
    work_active &= jnp.any(side & (engines.state == OFF) & (engines.fire <= 0.0))
    duration = jnp.maximum(scenario.repair_duration, dt)
    work = jnp.minimum(state.repair_elapsed + jnp.where(work_active, dt, 0.0), duration)
    restart = repairable & (state.repair_elapsed >= duration - dt / 2.0)
    increment = jnp.where(work >= duration - dt / 2.0, jnp.maximum(1.0 - engines.repair, 0.0), dt / duration)
    engines = step_engines(
        engines, throttle, dt, shutdown=shutdown, restart=side & restart,
        extinguish=side * jnp.where(spraying, 0.8 * dt, 0.0),
        repair_work=side * jnp.where(work_active & ~restart, increment, 0.0),
        icing=airframe_ice, airborne=~state.aircraft.on_ground, airspeed=state.aircraft.speed,
    )
    restarted = state.engine_restarted | jnp.any(side & state.engines.starting & engines.running)
    return engines, fault, work, restarted, state.engine_shutdown_commanded | fire_shutdown


def _runway_motion(aircraft, runway_heading):
    air_track = (aircraft.track - runway_heading) * RAD
    heading = (aircraft.heading - runway_heading) * RAD
    wind_along, wind_cross = approach.runway_coordinates(aircraft.wx, aircraft.wz, runway_heading)
    wind_along *= aircraft.wind_factor
    wind_cross *= aircraft.wind_factor
    speed = aircraft.speed * jnp.cos(aircraft.gamma * RAD)
    velocity_along = speed * jnp.cos(air_track) + wind_along
    velocity_cross = speed * jnp.sin(air_track) + wind_cross
    ground_speed = jnp.hypot(velocity_along, velocity_cross)
    track = jnp.where(ground_speed > 3.0, jnp.arctan2(velocity_cross, velocity_along), heading)
    sink = -(aircraft.speed * jnp.sin(aircraft.gamma * RAD) + aircraft.wy * aircraft.wind_factor)
    return track, heading, ground_speed, sink, velocity_along, velocity_cross


def _observation_packet(state):
    flight = state.flight
    aircraft = flight.aircraft
    track, heading, ground_speed, sink, velocity_along, velocity_cross = _runway_motion(aircraft, flight.runway_heading)
    # Snapshot wind uses signed world-axis quanta of 0.25 m/s, with no vertical
    # component. The bridge derives displayed airspeed from that packed wind.
    wind_xz = jnp.stack((aircraft.wx, aircraft.wz)) * aircraft.wind_factor
    wind_xz = jnp.clip(jnp.round(wind_xz * 4.0), -127.0, 127.0) / 4.0
    wind_along, wind_cross = approach.runway_coordinates(*wind_xz, flight.runway_heading)
    indicated_speed = jnp.sqrt((velocity_along - wind_along)**2 + (velocity_cross - wind_cross)**2 + sink**2)
    distances = jnp.asarray(approach.GATE_DISTANCES, dtype=jnp.float32)
    gate_index = jnp.count_nonzero(flight.along >= -distances)
    has_gate = gate_index < len(approach.GATE_DISTANCES)
    nearest = jnp.minimum(gate_index, len(approach.GATE_DISTANCES) - 1)
    heights, widths, half_heights = approach.gate_geometry(flight.runway_elev)
    engine = state.engines
    target = jnp.where(state.phase == GO_AROUND,
                       state.go_around_waypoint + jnp.asarray((AIM_POINT_OFFSET, 0.0)), 0.0)
    opposing = jnp.max(jnp.abs(approach._competing_input(flight.disturbances, flight.elapsed)))
    values = (
        state.physical_along, flight.lateral, aircraft.agl, indicated_speed, ground_speed, sink,
        aircraft.pitch, aircraft.roll, aircraft.prate, aircraft.rrate, aircraft.yrate,
        jnp.sin(track), jnp.cos(track), jnp.sin(heading), jnp.cos(heading), wind_along, wind_cross,
        RUNWAY_LENGTH, RUNWAY_HALF_WIDTH, jnp.where(has_gate, -distances[nearest] - flight.along, 0.0),
        jnp.where(has_gate, aircraft.py - heights[nearest], 0.0), jnp.where(has_gate, widths[nearest], 0.0),
        jnp.where(has_gate, half_heights[nearest], 0.0), gate_index, aircraft.gear_pos,
        aircraft.flap_pos, aircraft.stall_warn, aircraft.on_ground, state.applied_action[3],
        engine.rpm[0], engine.thrust[0], engine.health[0], engine.fire[0], engine.repair_progress[0], engine.state[0], 1.0,
        engine.rpm[1], engine.thrust[1], engine.health[1], engine.fire[1], engine.repair_progress[1], engine.state[1], 1.0,
        state.weather_sample.rain, state.weather_sample.visibility, state.weather_sample.turbulence,
        state.weather_sample.airframe_ice, 0.0,
        jnp.where(state.scenario.other_known, opposing, 0.0), state.scenario.other_known,
        *state.applied_action, target[0], target[1], state.elapsed, state.phase,
    )
    return jnp.stack([jnp.asarray(value, dtype=jnp.float32) for value in values])


def observe(state):
    """Return only the client's delayed snapshot, its age, and the last locally sent action."""
    packet = state.observation_transport.packet
    observation = packet[:OBSERVATION_SIZE].at[_OBSERVATION_AGE].set(state.elapsed - packet[-2])
    return observation.at[_LAST_ACTION:_LAST_ACTION + ACTION_SIZE].set(state.command_history[0])


def observe_phase(state):
    """Return the action mode observable through the same delayed snapshot."""
    return state.observation_transport.packet[-1].astype(jnp.int32)


def _receive(key, sequence, transport, history, base_lag, jitter, loss):
    """Advance a sampled link by one decision; discard stale and reordered packets."""
    draw = jax.random.uniform(key, (2,), dtype=jnp.float32)
    jitter = jnp.clip(jitter, 0, MAX_LAG_STEPS)
    offset = jnp.floor(draw[0] * (2 * jitter + 1)).astype(jnp.int32) - jitter
    delay = jnp.clip(base_lag + offset, 0, MAX_LAG_STEPS)
    due = jnp.where(draw[1] < jnp.clip(loss, 0.0, 1.0), -1, sequence + delay)
    queue = jnp.concatenate((due[None], transport.due[:-1]))
    candidates = sequence - jnp.arange(MAX_LAG_STEPS + 1, dtype=jnp.int32)
    ready = (queue >= 0) & (queue <= sequence) & (candidates > transport.delivered_sequence)
    latest = jnp.argmax(ready)
    deliver = jnp.any(ready)
    return Transport(queue, jnp.where(deliver, candidates[latest], transport.delivered_sequence),
                     jnp.where(deliver, history[latest], transport.packet))


def _phase_update(state, aircraft, along, lateral):
    gear_unsafe = (state.scenario.gear_go_around
                   & (state.aircraft.gear_pos < GEAR_DOWN_THRESHOLD) & (state.along >= GEAR_DECISION_ALONG)
                   # A single engine cannot climb out with the extra gear drag.
                   & jnp.all(state.engines.state == RUNNING_ENGINE))
    started = state.go_around_started | (
        ((state.elapsed >= state.scenario.go_around_time) | gear_unsafe) & ~state.aircraft.on_ground
        & (state.touchdown_count == 0)
    )
    active = started & ~state.go_around_completed
    stage = state.go_around_stage
    ground_track, _, _, sink, _, _ = _runway_motion(aircraft, state.flight.runway_heading)
    heading_error = jnp.abs(wrap180(ground_track / RAD))
    advances = jnp.stack((
        (along >= 1200.0) & (aircraft.agl >= 110.0) & (aircraft.gamma >= 0.0),
        lateral >= 1200.0,
        along <= -4550.0,
        (along <= -4650.0) & (jnp.abs(lateral) <= 460.0),
    ))
    stage = jnp.minimum(stage + (active & (stage < 4) & advances[jnp.minimum(stage, 3)]), 4).astype(jnp.int32)
    capture_window = (
        (along >= -6750.0) & (along <= -4100.0) & (jnp.abs(lateral) <= 220.0)
        & (aircraft.agl >= 65.0) & (aircraft.agl <= 450.0)
        & (aircraft.speed >= 37.0) & (aircraft.speed <= 65.0)
        & (jnp.cos(heading_error * RAD) >= 0.82) & (jnp.abs(aircraft.roll) <= 18.0)
        & ~aircraft.stall_warn & (jnp.abs(sink) <= 5.0)
    )
    completed = state.go_around_completed | (
        active & (stage == 4) & capture_window & ~aircraft.on_ground
    )
    intercepted = state.intercept_completed | (
        (jnp.abs(lateral) < 130.0) & (heading_error < 24.0)
        & (jnp.abs(aircraft.agl + along * approach.GLIDE_SLOPE) < 35.0)
    )
    phase = jnp.where(intercepted | (along > -4000.0), FINAL, CAPTURE)
    flare = (aircraft.agl <= 10.0) | (state.phase == FLARE) | (state.touchdown_count > 0)
    phase = jnp.where(flare, FLARE, phase)
    phase = jnp.where(aircraft.on_ground, ROLLOUT, phase)
    phase = jnp.where(started & ~completed & ~aircraft.on_ground, GO_AROUND, phase)
    return phase.astype(jnp.int32), started, stage, completed, intercepted


def _physics_frame(state, action, tuning):
    flight = state.flight
    current = flight.aircraft
    dt = jnp.float32(PHYSICS_DT)
    pitch, roll, yaw = jnp.clip(action[:3] + approach._competing_input(flight.disturbances, flight.elapsed), -1.0, 1.0)
    reverse_on = (state.scenario.auto_reverse & current.on_ground & (state.touchdown_count > 0)
                  & (current.speed > REVERSE_OFF_SPEED) & jnp.all(state.engines.state == RUNNING_ENGINE))
    throttle = jnp.where(reverse_on, REVERSE_THROTTLE, action[3])
    local_weather = _weather_sample(state.scenario, state.weather, flight)
    engines, fault, work, restarted, shutdown_commanded = _update_engines(state, throttle, dt, local_weather.airframe_ice)
    gear_jammed, gear_jam_pos, cranks = _crank_gear(state)
    pitch_rate, roll_rate, trim = approach.manual_rates(current, pitch, roll, flight.trim_speed, throttle, tuning, dt)
    flap, candidate, hold = approach._auto_flap(flight, tuning, dt)
    legacy_weather = state.scenario.weather_kind == LEGACY_WEATHER
    disturbances = flight.disturbances._replace(
        turbulence=local_weather.turbulence,
        jolt_heave=jnp.where(legacy_weather, flight.disturbances.jolt_heave, 0.0),
    )
    gust_pitch, gust_roll, heave, wind_gain = approach._weather(disturbances, flight.elapsed)
    airborne = ~current.on_ground
    heave = jnp.where(airborne, heave + local_weather.gust_heave, 0.0)
    controls = Controls(
        pitch_rate=pitch_rate + jnp.where(airborne, gust_pitch, 0.0),
        roll_rate=roll_rate + jnp.where(airborne, gust_roll, 0.0),
        rudder=yaw, steer=jnp.clip(yaw + roll, -1.0, 1.0), brakes=action[4],
        thr_l=engines.thrust[0], thr_r=engines.thrust[1], gear_down=flight.elapsed >= state.scenario.gear_down_time,
        gear_jammed=gear_jammed, gear_jam_pos=gear_jam_pos,
        flap=flap, heave=heave, reverse=reverse_on,
    )
    weather_wx, weather_wz = _profile_wind(local_weather, flight.runway_heading)
    current = current._replace(
        wx=jnp.where(legacy_weather, flight.disturbances.wind_x * wind_gain, weather_wx),
        wz=jnp.where(legacy_weather, flight.disturbances.wind_z * wind_gain, weather_wz),
    )
    kick = jnp.where(airborne & ~legacy_weather, _gust_kick(state, dt), 0.0)
    current = current._replace(
        roll=jnp.where(kick != 0, jnp.clip(current.roll + 12.0 * kick, -75.0, 75.0), current.roll),
        heading=jnp.where(kick != 0, (current.heading + 2.0 * kick) % 360.0, current.heading),
    )
    initial_ground = Ground(flight.runway_elev, jnp.where(_runway_surface(flight.along, flight.lateral), 2, 0))
    aircraft = flight_step(current, controls, dt, tuning, initial_ground, initial_ground)
    along, lateral = approach.runway_coordinates(aircraft.px, aircraft.pz, flight.runway_heading)
    paved = _runway_surface(along, lateral)
    aircraft = aircraft._replace(surface=jnp.where(paved, 2, 0).astype(jnp.int32))
    contact = airborne & aircraft.on_ground
    first_contact = contact & (state.touchdown_count == 0)
    phase, started, stage, completed, intercepted = _phase_update(state, aircraft, along, lateral)
    elapsed = flight.elapsed + dt
    gate_outside = along**2 + lateral**2 > approach.GATE_ACTIVATION_RADIUS**2
    gates, gate_cross, gate_vert = approach._update_gates(flight, aircraft, along, lateral, gate_outside)
    gs = jnp.hypot(aircraft.px - current.px, aircraft.pz - current.pz) / dt
    braked_stop = aircraft.on_ground & paved & (gs < STOP_SPEED) & (action[4] > 0.5)
    stop_frames = jnp.where(braked_stop, state.stop_frames + 1, 0)
    sink = approach._touchdown_sink(current, aircraft, heave, dt)
    bad_contact = contact & ((aircraft.gear_pos <= 0.9) | (jnp.abs(aircraft.roll) > tuning.wing_strike_roll))
    # Flat-world degradation proxy. Native damage and landing scores remain
    # separate acceptance checks against game recordings.
    far = state.scenario.far_start
    outside = along**2 + lateral**2 > jnp.where(far, FAR_DISTANCE, MAX_DISTANCE)**2
    reason = jnp.where(elapsed >= jnp.where(far, FAR_EPISODE_SECONDS, MAX_EPISODE_SECONDS), TIMEOUT, RUNNING)
    reason = jnp.where(outside, LEFT_AREA, reason)
    reason = jnp.where(aircraft.on_ground & ~paved, OFF_RUNWAY, reason)
    reason = jnp.where(bad_contact | aircraft.belly, CRASH, reason)
    # Second arrival at the decision point still unlocked (or no climb for a go-around): the assist
    # releases control to the crew instead of landing; scored as a hand-back, not a crash.
    unlocked_at_decision = (state.scenario.gear_go_around & ~aircraft.on_ground & (state.touchdown_count == 0)
                            & (aircraft.gear_pos < GEAR_DOWN_THRESHOLD) & (along >= GEAR_DECISION_ALONG)
                            & (state.go_around_completed | ~jnp.all(state.engines.state == RUNNING_ENGINE)))
    reason = jnp.where(unlocked_at_decision & (reason == RUNNING), HANDBACK, reason)
    reason = jnp.where((stop_frames >= STOP_FRAMES) & (reason == RUNNING), STOPPED, reason).astype(jnp.int32)
    flight = flight._replace(
        aircraft=aircraft, along=along, lateral=lateral, elapsed=elapsed, trim_speed=trim,
        flap_setting=flap, flap_candidate=candidate, flap_candidate_time=hold,
        gate_states=gates, gate_lateral_error=gate_cross, gate_vertical_error=gate_vert,
        done=reason != RUNNING, termination=reason,
        touchdown_vertical_speed=jnp.where(first_contact, sink, flight.touchdown_vertical_speed),
        touchdown_along=jnp.where(first_contact, along + AIM_POINT_OFFSET, flight.touchdown_along),
        touchdown_lateral=jnp.where(first_contact, lateral, flight.touchdown_lateral),
    )
    return state._replace(
        flight=flight, engines=engines, weather_sample=local_weather, phase=phase, fault_detected=fault, repair_elapsed=work,
        engine_restarted=restarted, engine_shutdown_commanded=shutdown_commanded,
        go_around_started=started, go_around_stage=stage,
        go_around_completed=completed, intercept_completed=intercepted,
        touchdown_count=state.touchdown_count + contact.astype(jnp.int32),
        touchdown_roll=jnp.where(first_contact, aircraft.roll, state.touchdown_roll),
        touchdown_speed=jnp.where(first_contact, gs, state.touchdown_speed),
        max_rollout_lateral=jnp.maximum(state.max_rollout_lateral, jnp.where(aircraft.on_ground, jnp.abs(lateral), 0.0)),
        stop_frames=stop_frames,
        gear_jammed=gear_jammed & ~((gear_jam_pos >= 1.0) & (aircraft.gear_pos >= GEAR_CLEAR_POSITION)),
        gear_jam_pos=jnp.where(gear_jammed, gear_jam_pos, 1.0), cranks=cranks,
    )


def _gust_kick(state, dt):
    """+1/-1 on the first physics frame of a front gust when enabled, else 0."""
    weather = state.weather
    windy = (weather.kind == GUSTS) | (weather.kind == LOW_VIS_WINDY)
    gust_time = state.elapsed - weather.first_gust
    onset = (state.scenario.gust_kick & windy & (gust_time >= 0.0)
             & (jnp.mod(gust_time, weather.gust_period) < dt))
    # Seeded stand-in for the native 50/50 Random draw, one per gust.
    index = jnp.floor(gust_time / weather.gust_period)
    draw = jnp.mod(jnp.sin(index * 12.9898 + weather.wind_phase * 78.233) * 43758.5453, 1.0)
    return jnp.where(onset, jnp.where(draw >= 0.5, 1.0, -1.0), 0.0)


def _crank_gear(state):
    """Apply scheduled crank uses; each adds 0.125 to the jam stop while jammed and not crashed."""
    scenario = state.scenario
    due = jnp.where(state.elapsed >= scenario.crank_start,
                    jnp.floor((state.elapsed - scenario.crank_start) / jnp.maximum(scenario.crank_interval, 1e-3)) + 1.0,
                    0.0).astype(jnp.int32)
    use = state.gear_jammed & (due > state.cranks)
    position = jnp.where(use, jnp.minimum(state.gear_jam_pos + GEAR_CRANK_STEP, 1.0), state.gear_jam_pos)
    return state.gear_jammed, position, jnp.where(use, state.cranks + 1, state.cranks)


def step(state, action, tuning):
    """Apply one decision with separate server actuation and observation delays."""
    def advance(live):
        clean = jnp.clip(jnp.asarray(action, dtype=jnp.float32),
                         jnp.asarray((-1.0, -1.0, -1.0, 0.0, 0.0)), 1.0)
        scenario = live.scenario
        sequence = live.sequence + 1
        packet_key = jax.random.fold_in(live.packet_key, sequence)
        commands = jnp.concatenate((clean[None, :], live.command_history[:-1]), axis=0)
        received = _receive(jax.random.fold_in(packet_key, 0), sequence, live.command_transport,
                            commands, scenario.command_lag, scenario.command_jitter, scenario.command_loss)
        applied = received.packet
        applied = applied.at[3].set(jnp.round(applied[3] * 255.0) / 255.0)
        applied = applied.at[4].set((applied[4] >= 0.5).astype(jnp.float32))
        live = live._replace(command_history=commands, applied_action=applied,
                             sequence=sequence, command_transport=received)
        already_reentered = live.go_around_completed

        def frame(_, current):
            return jax.lax.cond(current.done, lambda entry: entry,
                                lambda entry: _physics_frame(entry, applied, tuning), current)

        live = jax.lax.fori_loop(0, PHYSICS_PER_CONTROL, frame, live)
        live = live._replace(phase=jnp.where(~already_reentered & live.go_around_completed, CAPTURE, live.phase))
        packets = jnp.concatenate((_observation_packet(live)[None, :], live.observation_history[:-1]), axis=0)
        observed = _receive(jax.random.fold_in(packet_key, 1), sequence, live.observation_transport,
                            packets, scenario.observation_lag, scenario.observation_jitter, scenario.observation_loss)
        return live._replace(observation_history=packets, observation_transport=observed)

    return jax.lax.cond(state.done, lambda entry: entry, advance, state)
