# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import jax
import jax.numpy as jnp
import numpy as np

from mayday_jax.physics import default_tuning
from mayday_landing.environment import (
    AIM_POINT_OFFSET,
    CAPTURE,
    GO_AROUND,
    OFF_RUNWAY,
    RUNNING,
    STOPPED,
    Scenario,
    fire_scenario,
    observe,
    observe_phase,
    reset,
    step,
)
from mayday_landing.features import INDEX, NAMES
from mayday_landing.weather import CLEAR, CLEAR_VISIBILITY, FOG, GUSTS, ICING


TUNING = default_tuning()
NEUTRAL = jnp.array([0.0, 0.0, 0.0, 0.45, 0.0], dtype=jnp.float32)
PARK = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
TICK = jax.jit(step)


def _run(state, action, steps):
    return jax.lax.fori_loop(0, steps, lambda _, live: step(live, action, TUNING), state)


RUN = jax.jit(_run)


def _location(state, along, lateral, agl, speed, *, track=0.0, on_ground=False):
    plane = state.aircraft._replace(
        px=jnp.float32(lateral), py=jnp.float32(agl), pz=jnp.float32(along),
        speed=jnp.float32(speed), gamma=jnp.float32(0.0), pitch=jnp.float32(0.0),
        roll=jnp.float32(0.0), track=jnp.float32(track), heading=jnp.float32(track),
        agl=jnp.float32(agl), ground_y=jnp.float32(0.0), gear_pos=jnp.float32(1.0),
        on_ground=jnp.bool_(on_ground),
    )
    flight = state.flight._replace(
        aircraft=plane, along=jnp.float32(along), lateral=jnp.float32(lateral),
        runway_heading=jnp.float32(0.0), runway_elev=jnp.float32(0.0),
    )
    return state._replace(flight=flight)


def test_intercept_positions_and_schema_are_batched():
    keys = jax.random.split(jax.random.key(8), 6)
    states = jax.jit(jax.vmap(lambda key: reset(key, TUNING, Scenario(difficulty=1.0))))(keys)
    observations = jax.jit(jax.vmap(observe))(states)
    assert observations.shape == (6, len(NAMES))
    assert observations.dtype == jnp.float32
    assert np.isfinite(observations).all()
    assert (states.along < -4700.0).all()
    assert (np.abs(states.lateral) > 130.0).any()
    np.testing.assert_allclose(observations[:, INDEX["along"]], states.along + AIM_POINT_OFFSET)
    np.testing.assert_allclose(observations[:, INDEX["next_gate_distance"]], -4000.0 - states.along)
    assert (states.phase == CAPTURE).all()


def test_ground_track_observation_uses_effective_wind_and_world_velocity():
    state = reset(jax.random.key(9), TUNING, Scenario(difficulty=0.0))
    state = _location(state, -5000.0, 0.0, 260.0, 46.0)
    plane = state.aircraft._replace(wind_factor=jnp.float32(0.4))
    weather = state.flight.disturbances._replace(wind_x=jnp.float32(8.0), wind_z=jnp.float32(-3.0))
    flight = state.flight._replace(aircraft=plane, disturbances=weather, elapsed=jnp.float32(8.0))
    state = TICK(state._replace(flight=flight), NEUTRAL, TUNING)
    observation = observe(state)
    plane = state.aircraft
    track = np.deg2rad(float(plane.track))
    air_speed = float(plane.speed) * np.cos(np.deg2rad(float(plane.gamma)))
    cross_wind = float(plane.wx * plane.wind_factor)
    along_wind = float(plane.wz * plane.wind_factor)
    along_velocity = air_speed * np.cos(track) + along_wind
    cross_velocity = air_speed * np.sin(track) + cross_wind
    ground_speed = np.hypot(along_velocity, cross_velocity)
    assert abs(cross_velocity / ground_speed - np.sin(track)) > 0.03
    np.testing.assert_allclose(observation[INDEX["ground_speed"]], ground_speed, atol=1e-5)
    np.testing.assert_allclose(observation[INDEX["track_sin"]], cross_velocity / ground_speed, atol=1e-6)
    np.testing.assert_allclose(observation[INDEX["track_cos"]], along_velocity / ground_speed, atol=1e-6)
    packed_cross = np.round(cross_wind * 4.0) / 4.0
    packed_along = np.round(along_wind * 4.0) / 4.0
    np.testing.assert_allclose(observation[INDEX["wind_cross"]], packed_cross, atol=1e-6)
    indicated_speed = np.linalg.norm((along_velocity - packed_along, cross_velocity - packed_cross,
                                      float(observation[INDEX["sink"]])))
    np.testing.assert_allclose(observation[INDEX["speed"]], indicated_speed, atol=1e-5)


def test_profile_wind_uses_runway_axes_and_applies_its_gust_envelope_once():
    state = reset(jax.random.key(8), TUNING, Scenario(difficulty=1.0, weather_kind=GUSTS))
    state = _location(state, -4950.0, 0.0, 260.0, 48.0)
    profile = state.weather._replace(headwind=jnp.float32(4.0), crosswind=jnp.float32(8.0),
                                     wind_phase=jnp.float32(0.5))
    flight = state.flight._replace(runway_heading=jnp.float32(90.0), elapsed=jnp.float32(8.0))
    state = TICK(state._replace(weather=profile, flight=flight), NEUTRAL, TUNING)
    assert float(state.weather_sample.crosswind) > 5.0
    assert float(state.aircraft.wx) < 0.0 and float(state.aircraft.wz) < 0.0
    np.testing.assert_allclose(state.aircraft.wx, -state.weather_sample.headwind, atol=1e-5)
    np.testing.assert_allclose(state.aircraft.wz, -state.weather_sample.crosswind, atol=1e-5)


def test_native_visibility_and_icing_profile_reach_the_physics_and_sensor():
    key = jax.random.key(5)
    clear = reset(key, TUNING, Scenario(weather_kind=CLEAR))
    assert float(observe(clear)[INDEX["visibility"]]) == CLEAR_VISIBILITY
    fog = reset(key, TUNING, Scenario(weather_kind=FOG, weather_leg=99))
    assert int(fog.weather.leg) == 8
    assert float(observe(fog)[INDEX["visibility"]]) == CLEAR_VISIBILITY
    fog = _location(fog, -1300.0, 0.0, 60.0, 46.0)
    fog = TICK(fog, NEUTRAL, TUNING)
    assert 320.0 <= float(observe(fog)[INDEX["visibility"]]) <= 1500.0

    iced = reset(key, TUNING, Scenario(weather_kind=999))
    assert int(iced.weather.kind) == ICING
    icing = iced.weather._replace(initial_ice=jnp.float32(0.94), icing_duration=jnp.float32(110.0))
    icy_flight = iced.flight._replace(elapsed=jnp.float32(100.0))
    iced = TICK(iced._replace(weather=icing, flight=icy_flight), NEUTRAL, TUNING)
    assert float(observe(iced)[INDEX["ice"]]) == 1.0
    assert (iced.engine_thrust < iced.applied_action[3] * 0.7).all()


def test_weather_profile_has_a_shared_paired_prefix():
    key = jax.random.key(42)
    scenario = Scenario(weather_kind=GUSTS, failure_time=0.0, failure_side=0,
                        observation_lag=2, repair_start=10.0)
    later = scenario._replace(repair_start=20.0)
    active = RUN(reset(key, TUNING, scenario), NEUTRAL, 50)
    other = RUN(reset(key, TUNING, later), NEUTRAL, 50)
    np.testing.assert_array_equal(observe(active), observe(other))
    np.testing.assert_array_equal(active.weather, other.weather)
    np.testing.assert_array_equal(active.weather_sample, other.weather_sample)


def test_observation_delay_and_actuation_are_independent():
    scenario = Scenario(observation_lag=2, command_lag=1)
    one = reset(jax.random.key(0), TUNING, scenario)
    start_along = observe(one)[INDEX["along"]]
    one = TICK(one, PARK, TUNING)
    assert float(one.applied_action[4]) == 0.0
    assert float(observe(one)[INDEX["last_brake"]]) == 1.0
    assert float(observe(one)[INDEX["along"]]) == float(start_along)
    two = TICK(one, NEUTRAL, TUNING)
    assert float(two.applied_action[4]) == 1.0
    assert float(observe(two)[INDEX["last_brake"]]) == 0.0
    assert float(observe(two)[INDEX["along"]]) == float(start_along)
    three = TICK(two, NEUTRAL, TUNING)
    assert float(three.applied_action[4]) == 0.0
    assert float(observe(three)[INDEX["along"]]) == float(one.physical_along)
    np.testing.assert_allclose(observe(three)[INDEX["observation_age"]], 0.2, atol=1e-6)


def test_packet_loss_and_jitter_are_seeded_and_never_roll_back_either_link():
    scenario = Scenario(observation_lag=2, command_lag=2,
                        observation_jitter=2, command_jitter=2,
                        observation_loss=0.4, command_loss=0.25)
    key = jax.random.key(42)
    world = reset(key, TUNING, scenario)
    repeated = reset(key, TUNING, scenario)
    sent_at = [float(world.elapsed)]
    sent_along = [float(world.physical_along)]
    sent_pitch = [float(world.applied_action[0])]
    received_sequence = (0, 0)
    skipped_packet = False
    frozen_snapshot = False
    different_links = False
    for tick in range(1, 31):
        action = NEUTRAL.at[0].set(tick / 100.0)
        world = TICK(world, action, TUNING)
        repeated = TICK(repeated, action, TUNING)
        observation = observe(world)
        np.testing.assert_array_equal(observation, observe(repeated))
        np.testing.assert_array_equal(world.applied_action, repeated.applied_action)
        sent_at.append(float(world.elapsed))
        sent_along.append(float(world.physical_along))
        sent_pitch.append(float(action[0]))
        obs_number = int(world.observation_transport.delivered_sequence)
        cmd_number = int(world.command_transport.delivered_sequence)
        assert received_sequence[0] <= obs_number <= tick
        assert received_sequence[1] <= cmd_number <= tick
        assert float(observation[INDEX["along"]]) == sent_along[obs_number]
        np.testing.assert_allclose(observation[INDEX["observation_age"]], sent_at[tick] - sent_at[obs_number], atol=1e-6)
        assert float(world.applied_action[0]) == sent_pitch[cmd_number]
        assert float(observation[INDEX["last_pitch"]]) == sent_pitch[tick]
        skipped_packet |= obs_number > received_sequence[0] + 1 or cmd_number > received_sequence[1] + 1
        frozen_snapshot |= tick > 5 and obs_number == received_sequence[0]
        different_links |= obs_number != cmd_number
        received_sequence = (obs_number, cmd_number)
    assert skipped_packet and frozen_snapshot and different_links

    loss = reset(key, TUNING, Scenario(observation_loss=1.0, command_loss=1.0, go_around_time=0.0))
    initial_observation = observe(loss)
    lost = RUN(loss, PARK, 10)
    assert int(lost.phase) == GO_AROUND
    assert int(observe_phase(lost)) == CAPTURE
    assert float(observe(lost)[INDEX["along"]]) == float(initial_observation[INDEX["along"]])
    np.testing.assert_allclose(observe(lost)[INDEX["observation_age"]], lost.elapsed, atol=1e-6)
    assert float(lost.applied_action[4]) == 0.0


def test_interrupted_repair_preserves_progress_and_requires_restart_spool():
    scenario = Scenario(difficulty=0.0, failure_time=0.0, failure_side=0,
                        repair_start=0.2, repair_interrupt=0.4, repair_resume=0.7,
                        repair_duration=0.4)
    start = reset(jax.random.key(1), TUNING, scenario)
    disabled = TICK(start, NEUTRAL, TUNING)
    assert float(disabled.engine_thrust[0]) == 0.0 < float(disabled.engine_thrust[1])
    paused = RUN(disabled, NEUTRAL, 4)
    still_paused = RUN(paused, NEUTRAL, 2)
    assert 0.3 < float(paused.engines.repair_progress[0]) < 0.7
    np.testing.assert_allclose(paused.engines.repair_progress, still_paused.engines.repair_progress)
    starting = RUN(still_paused, NEUTRAL, 5)
    assert starting.engines.starting[0]
    assert float(starting.engine_thrust[0]) == 0.0
    running = RUN(starting, NEUTRAL, 35)
    assert running.engine_restarted
    assert 0.0 < float(running.engine_thrust[0]) < float(running.engine_thrust[1])


def test_permanent_fault_and_unseen_future_do_not_expose_the_schedule():
    scenario = Scenario(difficulty=0.0, failure_time=0.5, failure_side=1,
                        repair_start=0.8, repair_duration=0.1, observation_lag=2)
    start = reset(jax.random.key(1), TUNING, scenario)
    other = reset(jax.random.key(1), TUNING, scenario._replace(permanent=True))
    for _ in range(5):
        np.testing.assert_array_equal(observe(start), observe(other))
        start, other = TICK(start, NEUTRAL, TUNING), TICK(other, NEUTRAL, TUNING)
    np.testing.assert_array_equal(observe(start), observe(other))
    destroyed = RUN(other, NEUTRAL, 45)
    assert int(destroyed.engines.state[1]) == 3
    assert float(destroyed.engine_thrust[1]) == 0.0
    assert not destroyed.engine_restarted


def test_fire_ignition_is_shared_until_crew_action_and_repair_recovers_only_contained_case():
    base = Scenario(difficulty=0.0, repair_duration=0.4, observation_lag=1, rain=0.5, visibility=6500.0)
    contained = fire_scenario(base, failure_time=0.0, failure_side=1, extinguished=True, intensity=0.84)
    uncontained = fire_scenario(base, failure_time=0.0, failure_side=1, extinguished=False, intensity=0.84)
    key = jax.random.key(3)
    contained = RUN(reset(key, TUNING, contained), NEUTRAL, 29)
    uncontained = RUN(reset(key, TUNING, uncontained), NEUTRAL, 29)
    np.testing.assert_array_equal(observe(contained), observe(uncontained))
    snapshot = observe(contained)
    assert float(snapshot[INDEX["right_fire"]]) >= 0.84
    assert float(snapshot[INDEX["left_fire"]]) == 0.0
    assert float(snapshot[INDEX["rain"]]) == 0.5
    assert float(snapshot[INDEX["visibility"]]) == 6500.0
    assert 0.0 < float(contained.engine_thrust[1]) < float(contained.engine_thrust[0])

    contained = RUN(contained, NEUTRAL, 100)
    assert float(contained.engines.fire[1]) == 0.0
    assert float(contained.engines.repair_progress[1]) == 1.0
    assert contained.engine_restarted and contained.engines.running[1]
    assert float(contained.engine_thrust[1]) > 0.0

    uncontained = RUN(uncontained, NEUTRAL, 590)
    assert int(uncontained.engines.state[1]) == 3
    assert float(uncontained.engine_thrust[1]) == 0.0
    assert not uncontained.engine_restarted


def test_runway_contact_rolls_before_three_braked_seconds_and_exit_fails():
    start = reset(jax.random.key(2), TUNING, Scenario(difficulty=0.0))
    approach = _location(start, -150.0, 0.0, 0.05, 28.0)
    approach = approach._replace(flight=approach.flight._replace(
        aircraft=approach.aircraft._replace(gamma=jnp.float32(-3.0))))
    contact = TICK(approach, PARK, TUNING)
    assert int(contact.touchdown_count) == 1
    assert int(contact.termination) == RUNNING
    assert 100.0 < float(contact.flight.touchdown_along) < 105.0
    rolling = RUN(contact, PARK, 130)
    assert float(rolling.physical_along) > float(contact.physical_along)
    assert int(rolling.termination) == STOPPED
    assert int(rolling.stop_frames) == 180
    np.testing.assert_allclose(rolling.stopped_seconds, 3.0, atol=1e-6)
    np.testing.assert_array_equal(observe(TICK(rolling, NEUTRAL, TUNING)), observe(rolling))

    stopped = _location(start, 0.0, 0.0, 0.0, 0.0, on_ground=True)
    assert int(RUN(stopped, NEUTRAL, 31).termination) == RUNNING
    almost = RUN(stopped, PARK, 29)
    assert int(almost.termination) == RUNNING
    assert int(TICK(almost, PARK, TUNING).termination) == STOPPED

    edge = _location(start, 1249.5, 0.0, 0.0, 22.0, on_ground=True)
    assert int(TICK(edge, PARK, TUNING).termination) == OFF_RUNWAY


def test_go_around_rearms_gates_and_phase_is_not_visible_before_packet_arrival():
    scenario = Scenario(difficulty=0.0, go_around_time=0.0, observation_lag=2)
    start = reset(jax.random.key(9), TUNING, scenario)
    one = TICK(start, NEUTRAL, TUNING)
    assert int(one.phase) == GO_AROUND
    assert int(observe_phase(one)) == CAPTURE
    three = RUN(one, NEUTRAL, 2)
    assert int(observe_phase(three)) == GO_AROUND
    assert float(observe(three)[INDEX["target_along"]]) == 1750.0

    joining = _location(three, -5000.0, 0.0, 255.0, 47.0)
    joining = joining._replace(
        go_around_stage=jnp.int32(4),
        flight=joining.flight._replace(gate_states=jnp.ones(5, dtype=jnp.int32)),
    )
    resumed = TICK(joining, NEUTRAL, TUNING)
    assert resumed.go_around_completed
    assert int(resumed.go_around_stage) == 4
    assert int(resumed.phase) != GO_AROUND
    assert float(resumed.elapsed) > float(joining.elapsed)
    np.testing.assert_array_equal(resumed.gate_states, np.zeros(5, dtype=np.int32))


def _jammed(state, position):
    plane = state.aircraft._replace(gear_pos=jnp.float32(position))
    return state._replace(flight=state.flight._replace(aircraft=plane),
                          gear_jammed=jnp.bool_(True), gear_jam_pos=jnp.float32(position))


def test_gear_crank_follows_native_steps_and_clears_only_at_full_lock():
    scenario = Scenario(difficulty=0.0, crank_start=0.0, crank_interval=0.5)
    start = _jammed(reset(jax.random.key(3), TUNING, scenario), 0.5)
    # 0.5 s per crank: four cranks bring the stop to 1.0 within 1.6 s.
    early = RUN(start, NEUTRAL, 16)  # cranks at 0, 0.5, 1.0 and 1.5 s; the leg trails at 1/1.6 per s
    assert bool(early.gear_jammed)
    assert int(early.cranks) == 4
    assert float(early.gear_jam_pos) == 1.0
    assert float(early.aircraft.gear_pos) < 0.999
    later = RUN(early, NEUTRAL, 30)
    assert not bool(later.gear_jammed)
    assert int(later.cranks) == 4
    assert float(later.aircraft.gear_pos) >= 0.999


def test_uncranked_jam_holds_and_gear_go_around_triggers_at_decision_ring():
    held = _jammed(reset(jax.random.key(4), TUNING, Scenario(difficulty=0.0)), 0.6)
    held = RUN(held, NEUTRAL, 20)
    assert bool(held.gear_jammed) and abs(float(held.aircraft.gear_pos) - 0.6) < 1e-4
    far = _jammed(_location(reset(jax.random.key(4), TUNING, Scenario(difficulty=0.0, gear_go_around=True)),
                            -2000.0, 0.0, 105.0, 47.0), 0.6)
    assert not bool(TICK(far, NEUTRAL, TUNING).go_around_started)
    near = _jammed(_location(far, -1100.0, 0.0, 58.0, 47.0), 0.6)
    assert int(TICK(near, NEUTRAL, TUNING).phase) == GO_AROUND
    locked = _location(far, -1100.0, 0.0, 58.0, 47.0)._replace(gear_jammed=jnp.bool_(False))
    assert not bool(TICK(locked, NEUTRAL, TUNING).go_around_started)


def test_engageable_starts_stay_inside_the_director_window():
    from mayday_landing.experiment import engageable_start, make_cases, _initialize
    seeds = np.arange(940000, 940064)
    wide = _initialize(*make_cases(seeds, True, True, True)[1::-1], TUNING)
    narrow = _initialize(*make_cases(seeds, True, True, True, engageable_starts=True)[1::-1], TUNING)
    assert np.max(np.abs(np.asarray(narrow.lateral))) <= 200.0 < np.max(np.abs(np.asarray(wide.lateral)))
    assert engageable_start(narrow).mean() > 0.95 > engageable_start(wide).mean()


def test_native_gust_kick_banks_twelve_degrees_at_onset_only_when_enabled():
    for enabled in (False, True):
        start = reset(jax.random.key(8), TUNING, Scenario(difficulty=0.5, weather_kind=GUSTS, gust_kick=enabled))
        start = _location(start, -3000.0, 0.0, 160.0, 47.0)
        start = start._replace(weather=start.weather._replace(first_gust=jnp.float32(0.05), gust_period=jnp.float32(30.0)))
        after = TICK(start, NEUTRAL, TUNING)
        change = abs(float(after.aircraft.roll) - float(start.aircraft.roll))
        assert (change > 8.0) if enabled else (change < 3.0)


def test_auto_reverse_shortens_a_braked_rollout_and_stops_forward():
    distances = []
    for reverse in (False, True):
        state = reset(jax.random.key(6), TUNING, Scenario(difficulty=0.0, auto_reverse=reverse))
        state = _location(state, 300.0 - AIM_POINT_OFFSET, 0.0, 0.0, 40.0, on_ground=True)
        state = state._replace(touchdown_count=jnp.int32(1))
        start = float(state.physical_along)
        end = RUN(state, PARK * jnp.array([1, 1, 1, 1, 1.0]), 200)
        assert float(end.aircraft.speed) < 0.2 and not bool(end.aircraft.backward)
        distances.append(float(end.physical_along) - start)
    assert distances[1] < distances[0] - 10.0


def test_late_gear_goes_around_once_then_hands_back_without_moving_the_gear():
    from mayday_landing.environment import HANDBACK
    never = Scenario(difficulty=0.0, gear_down_time=float("inf"), gear_go_around=True)
    start = reset(jax.random.key(4), TUNING, never)
    assert float(start.aircraft.gear_pos) == 0.0
    near = _location(start, -1100.0, 0.0, 58.0, 47.0)._replace(
        flight=_location(start, -1100.0, 0.0, 58.0, 47.0).flight)
    near = near._replace(flight=near.flight._replace(aircraft=near.aircraft._replace(gear_pos=jnp.float32(0.0))))
    assert int(TICK(near, NEUTRAL, TUNING).phase) == GO_AROUND
    second = near._replace(go_around_started=jnp.bool_(True), go_around_completed=jnp.bool_(True))
    assert int(TICK(second, NEUTRAL, TUNING).termination) == HANDBACK
    lowered = reset(jax.random.key(4), TUNING, never._replace(gear_down_time=0.5))
    assert float(RUN(lowered, NEUTRAL, 80).aircraft.gear_pos) > 0.9
