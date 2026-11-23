# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Train and evaluate two landing policies on disjoint, paired game-twin scenarios."""

import argparse
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mayday_jax import environment as approach_env
from mayday_jax.environment import GATE_DISTANCES
from mayday_jax.physics import RAD, default_tuning
from mayday_jax.replay import load_tuning

from . import model
from .environment import (
    AIM_POINT_OFFSET, FAR_EPISODE_SECONDS, _runway_motion, CONTROL_DT, LEGACY_WEATHER, MAX_EPISODE_SECONDS, RUNWAY_HALF_WIDTH, RUNWAY_LENGTH, STOPPED,
    Scenario, fire_scenario, observe, observe_phase, reset, step,
)
from .features import INDEX, NAMES
from .guidance import PRIOR_SCHEMA, guidance
from .residual_model import (
    RESIDUAL_LIMITS, apply_residual_actions, encode_residual_actions, unwrap_residual, wrap_prior, wrap_residual,
)
from .residual_teacher import residual_plan
from .teacher import PlannerConfig, managed_action, plan
from .weather import WEATHER_NAMES


BRANCHES = ("healthy", "repaired", "interrupted", "abandoned", "permanent", "go_around",
            "fire_contained", "fire_uncontained")
REPAIR_BRANCHES = ("repaired", "interrupted", "abandoned", "permanent")
TERMINATIONS = ("unfinished", "stopped", "off_runway", "timeout", "left_area", "crash", "handback")
FORK_STEP = 400
FORK_SECONDS = FORK_STEP * CONTROL_DT
MAX_SINK = 3.0
TRAIN_SEED_START = 410_000
HOLDOUT_SEED_START = 840_000
SUCCESSOR_HOLDOUT_SEED_START = 860_000
CONFIRMATION_SEED_START = 880_000


@dataclass(frozen=True)
class RunConfig:
    # Episodes end as soon as every case is done, so the far-start cap costs nothing otherwise.
    max_steps: int = round(FAR_EPISODE_SECONDS / CONTROL_DT) + 5
    chunk_steps: int = 100
    batch_seeds: int = 8


class RollState(NamedTuple):
    world: tuple
    memory: jax.Array
    handover: jax.Array
    fault: jax.Array
    recovery: jax.Array


class Cases(NamedTuple):
    scenario: Scenario
    keys: jax.Array
    seeds: np.ndarray
    names: np.ndarray


class Labels(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    phases: np.ndarray
    episode_ids: np.ndarray
    # Twin-only targets for an asymmetric auxiliary loss (never a policy input).
    privileged: np.ndarray | None = None


PRIVILEGED_NAMES = ("true_wind_along", "true_wind_cross", "gust_heave", "copilot_pitch", "copilot_roll",
                    "copilot_yaw", "delay_along", "delay_cross", "delay_agl", "delay_sink")


def privileged_targets(state, observation):
    """True disturbances and the gap between the true state and the delayed snapshot."""
    plane = state.aircraft
    wind_along, wind_cross = approach_env.runway_coordinates(
        plane.wx * plane.wind_factor, plane.wz * plane.wind_factor, state.flight.runway_heading)
    sink = _runway_motion(plane, state.flight.runway_heading)[3]
    copilot = approach_env._competing_input(state.flight.disturbances, state.elapsed)
    feature = lambda name: observation[INDEX[name]]
    return jnp.stack((wind_along, wind_cross, state.weather_sample.gust_heave, *copilot,
                      state.physical_along - feature("along"), state.lateral - feature("cross"),
                      plane.agl - feature("agl"), sink - feature("sink"))).astype(jnp.float32)


def make_cases(seeds, packet_stress=False, native_weather=False, gear_crank=False, gear_go_around=False,
               engageable_starts=False, gust_kick=False, auto_reverse=False, far_start=False, wide_start=False,
               gear_late=False):
    """Pair four fault continuations at one physical state per seed.

    ``gear_crank`` schedules a crew hand crank for a jammed gear from a separate
    seeded stream, so the other scenario draws keep their original values.
    """
    seeds = np.asarray(seeds, dtype=np.uint32)
    if seeds.ndim != 1 or not seeds.size or len(set(seeds.tolist())) != seeds.size:
        raise ValueError("expected a nonempty vector of unique scenario seeds")
    scenarios = []
    for seed in seeds.tolist():
        rng = np.random.default_rng(seed)
        difficulty = rng.uniform(0.2, 1.0)
        observation_lag = (0, 1, 3, 6)[seed % 4]
        command_lag = int(rng.integers(0 if observation_lag <= 1 else 1, min(observation_lag, 3) + 1))
        stress = packet_stress and observation_lag >= 3
        base = Scenario(difficulty=difficulty, observation_lag=observation_lag,
                        command_lag=command_lag, rain=rng.uniform(0.0, 0.7 * difficulty),
                        visibility=rng.uniform(6000.0, 28_000.0), ice=rng.uniform(0.0, 0.25 * difficulty),
                        other_known=observation_lag == 0,
                        observation_jitter=int(stress), command_jitter=int(stress),
                        observation_loss=0.01 if stress else 0.0, command_loss=0.01 if stress else 0.0)
        if native_weather:
            # One copy of every weather/delay combination in each 24 seeds.
            base = base._replace(weather_kind=(seed // 4) % len(WEATHER_NAMES))
        if gear_crank:
            crew = np.random.default_rng((int(seed), 0x6A4D))
            base = base._replace(crank_start=float(crew.uniform(5.0, 60.0)),
                                 crank_interval=float(crew.uniform(0.5, 1.2)))
        if gear_go_around:
            base = base._replace(gear_go_around=True)
        if engageable_starts:
            base = base._replace(engageable_starts=True)
        if gust_kick:
            base = base._replace(gust_kick=True)
        if auto_reverse:
            base = base._replace(auto_reverse=True)
        if far_start:
            base = base._replace(far_start=True)
        if wide_start:
            base = base._replace(wide_start=True)
        if gear_late:
            # Crew lowers the gear late (25%), never (15%), or it is already down (60%); the assist
            # goes around once at the decision ring and then hands back.
            crew = np.random.default_rng((int(seed), 0x6EA2))
            draw = crew.uniform()
            down = 0.0 if draw < 0.6 else (float(crew.uniform(10.0, 120.0)) if draw < 0.85 else float("inf"))
            base = base._replace(gear_down_time=down, gear_go_around=True)
        failed = base._replace(failure_time=rng.uniform(18.0, 32.0), failure_side=int(seed % 2),
                               repair_duration=rng.uniform(6.0, 10.0))
        repair = failed._replace(repair_start=FORK_SECONDS)
        interrupted = repair._replace(repair_interrupt=FORK_SECONDS + 2.5,
                                      repair_resume=FORK_SECONDS + rng.uniform(9.0, 20.0))
        abandoned = interrupted._replace(repair_resume=float("inf"))
        go_around = base._replace(go_around_time=rng.uniform(83.0, 96.0))
        # The permanent branch shares the repairable outage up to the fork.
        # Its existing fault is declared unrecoverable at the fork itself.
        fire = {"failure_time": failed.failure_time, "failure_side": failed.failure_side,
                "intensity": rng.uniform(0.40, 0.68)}
        scenarios.extend((base, repair, interrupted, abandoned, failed, go_around,
                          fire_scenario(base, extinguished=True, **fire),
                          fire_scenario(base, extinguished=False, **fire)))
    vector = Scenario(*(jnp.asarray(column) for column in zip(*scenarios, strict=True)))
    every_seed = np.repeat(seeds, len(BRANCHES))
    return Cases(vector, jax.vmap(jax.random.PRNGKey)(jnp.asarray(every_seed)), every_seed,
                 np.tile(BRANCHES, len(seeds)))


@jax.jit
def _initialize(keys, scenarios, tuning):
    return jax.vmap(reset, in_axes=(0, None, 0))(keys, tuning, scenarios)


def _fork_permanent_loss(world, names):
    permanent = jnp.asarray(names == "permanent")
    side = jnp.arange(2)[None, :] == world.scenario.failure_side[:, None]
    engines = world.engines._replace(health=jnp.where(permanent[:, None] & side, 0.0, world.engines.health))
    return world._replace(engines=engines, scenario=world.scenario._replace(permanent=permanent))


def _assert_shared_prefix(state, names):
    """Check world, sensor and actuator queues, and any student history."""
    pairs = tuple(np.flatnonzero(names == name) for name in REPAIR_BRANCHES)
    for leaf in jax.tree.leaves((state.world._replace(scenario=None), state.memory)):
        leaf = np.asarray(leaf)
        if any(not np.array_equal(leaf[pairs[0]], leaf[other]) for other in pairs[1:]):
            raise RuntimeError("repair branches diverged before their common fork")


def _student_action(actor, network, observations, phases, memory):
    weights, mean, std = network
    encoded = model._encode(weights, (observations - mean) / std, phases, jnp)
    if actor == "gru":
        memory = model._gru(weights, encoded, memory, jnp)
        latent = jnp.concatenate((encoded, memory), axis=-1)
    else:
        latent = jnp.tanh(model._dense(weights["hidden"], encoded))
    return model._actions(weights, latent, phases, jnp), memory


def _handover(previous, world):
    plane = world.aircraft
    at_flare = ~previous[:, 0].astype(bool) & ~world.done & (plane.agl <= 10.0) & ~plane.on_ground
    # Exclude the airborne leg of an intentionally triggered approach abort.
    at_flare &= ~world.go_around_started | world.go_around_completed
    current = jnp.stack((jnp.ones_like(plane.agl), world.along, world.lateral,
                         -plane.speed * jnp.sin(plane.gamma * RAD), plane.roll), axis=-1)
    return jnp.where(at_flare[:, None], current, previous)


def _transient(previous, event, following):
    """Time of first event, lateral position, and ten-second peak yaw and drift."""
    elapsed, origin, yaw, drift = previous.T
    start = event & (elapsed < 0.0)
    elapsed = jnp.where(start, following.elapsed, elapsed)
    origin = jnp.where(start, following.lateral, origin)
    active = (elapsed >= 0.0) & (following.elapsed - elapsed <= 10.0)
    yaw = jnp.where(active, jnp.maximum(yaw, jnp.abs(following.aircraft.yrate)), yaw)
    drift = jnp.where(active, jnp.maximum(drift, jnp.abs(following.lateral - origin)), drift)
    return jnp.stack((elapsed, origin, yaw, drift), axis=-1)


@partial(jax.jit, static_argnames=("actor", "record", "steps", "planner", "residual"))
def _advance(state, keys, network, tuning, start, beta, *, actor, record, steps, planner, residual=False):
    batch_observe = jax.vmap(observe)
    batch_phase = jax.vmap(observe_phase)
    batch_step = jax.vmap(step, in_axes=(0, 0, None))
    batch_teacher = jax.vmap(plan, in_axes=(0, 0, None, None))
    batch_residual_teacher = jax.vmap(residual_plan, in_axes=(0, 0, 0, None, None))
    batch_prior = jax.vmap(guidance, in_axes=(0, 0, None))
    batch_managed = jax.vmap(managed_action)
    fold_keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))

    def advance(carry, index):
        world, memory, handover, fault, recovery = carry
        observations, phases = batch_observe(world), batch_phase(world)
        if residual or actor == "prior":
            prior = batch_prior(observations, phases, tuning)
        if record or actor == "teacher":
            step_keys = fold_keys(keys, index)
            labels = (batch_residual_teacher(world, prior, step_keys, tuning, planner) if residual
                      else batch_teacher(world, step_keys, tuning, planner))
        if actor == "teacher":
            actions = labels
        elif actor == "managed":
            actions = batch_managed(world)
        elif actor == "prior":
            actions = prior
        else:
            actions, memory = _student_action(actor, network, observations, phases, memory)
            if residual:
                actions = apply_residual_actions(actions, prior, phases, xp=jnp)
            if record:
                mix = jax.vmap(jax.random.uniform)(fold_keys(step_keys, 1))
                actions = jnp.where((mix < beta)[:, None], labels, actions)
        following = batch_step(world, actions, tuning)
        failed_side = jnp.maximum(following.scenario.failure_side, 0)
        recovered_thrust = jnp.take_along_axis(following.engine_thrust, failed_side[:, None], axis=1)[:, 0]
        restored = following.engine_restarted & (recovered_thrust >= 0.12)
        if record and residual:
            labels = encode_residual_actions(labels, prior, xp=jnp)
        privileged = jax.vmap(privileged_targets)(world, observations) if record else None
        output = (observations, labels, phases, ~world.done, privileged) if record else None
        return RollState(following, memory, _handover(handover, following),
                         _transient(fault, following.fault_detected, following),
                         _transient(recovery, restored, following)), output

    return jax.lax.scan(advance, state, start + jnp.arange(steps))


def _network(payload):
    if payload is None:
        return None
    policy = model.Policy(payload)
    if policy.feature_names != NAMES:
        raise ValueError("policy features do not match experiment observations")
    return jax.tree.map(jnp.asarray, (policy.weights, policy.mean, policy.std))


def _pack(chunks, batch_size, offset):
    columns = [np.concatenate(column, axis=0).swapaxes(0, 1) for column in zip(*chunks, strict=True)]
    obs, actions, phases, valid = columns[:4]
    privileged = columns[4][valid] if len(columns) > 4 else None
    episode_ids = np.broadcast_to(np.arange(offset, offset + batch_size, dtype=np.int32)[:, None], valid.shape)
    return Labels(obs[valid], actions[valid], phases[valid], episode_ids[valid], privileged)


def _join(parts):
    result = []
    next_id = 0
    for rows in parts:
        result.append(rows._replace(episode_ids=rows.episode_ids + next_id))
        next_id += int(rows.episode_ids[-1]) + 1
    return _concat_labels(result)


def _concat_labels(parts):
    columns = zip(*parts, strict=True)
    return Labels(*(None if any(item is None for item in column) else np.concatenate(column) for column in columns))


def engageable_start(world):
    """Director CaptureWindow at reset (LandingDirector.cs): the mod only engages inside it."""
    w = jax.device_get(world)
    plane = w.flight.aircraft
    along = np.asarray(w.flight.along) + AIM_POINT_OFFSET
    track = np.radians(np.asarray(plane.track) - np.asarray(w.flight.runway_heading))
    return ((along >= -6500.0) & (along <= -1050.0) & (np.abs(np.asarray(w.flight.lateral)) <= 220.0)
            & (np.asarray(plane.agl) >= 65.0) & (np.asarray(plane.agl) <= 450.0)
            & (np.asarray(plane.speed) >= 37.0) & (np.asarray(plane.speed) <= 65.0)
            & (np.cos(track) >= 0.82) & (np.abs(np.asarray(plane.roll)) <= 18.0))


def episode_rows(state, cases, engageable=None):
    world = jax.device_get(state.world)
    handover = np.asarray(state.handover)
    faults = np.asarray(state.fault)
    restores = np.asarray(state.recovery)
    records = []
    for i, (seed, branch) in enumerate(zip(cases.seeds, cases.names, strict=True)):
        touchdown = int(world.touchdown_count[i])
        sink = float(world.flight.touchdown_vertical_speed[i])
        along = float(world.flight.touchdown_along[i])
        cross = float(world.flight.touchdown_lateral[i])
        paved_contact = touchdown > 0 and 0.0 <= along <= RUNWAY_LENGTH and abs(cross) <= RUNWAY_HALF_WIDTH
        rings = world.gate_states[i].tolist()
        parked = world.termination[i] == STOPPED
        reacquired = bool(world.go_around_completed[i])
        success = (all(ring == 1 for ring in rings) and paved_contact and 0.0 <= sink <= MAX_SINK and parked
                   and (branch != "go_around" or reacquired))
        weather_kind = int(world.scenario.weather_kind[i])
        records.append({
            "seed": int(seed), "branch": str(branch), "difficulty": float(world.scenario.difficulty[i]),
            "observation_lag_steps": int(world.scenario.observation_lag[i]),
            "command_lag_steps": int(world.scenario.command_lag[i]), "ring_verdicts": rings,
            "observation_jitter_steps": int(world.scenario.observation_jitter[i]),
            "command_jitter_steps": int(world.scenario.command_jitter[i]),
            "observation_packet_loss": float(world.scenario.observation_loss[i]),
            "command_packet_loss": float(world.scenario.command_loss[i]),
            "weather_family": "legacy" if weather_kind == LEGACY_WEATHER else WEATHER_NAMES[world.weather.kind[i]],
            "weather_leg": None if weather_kind == LEGACY_WEATHER else int(world.weather.leg[i]),
            "termination": TERMINATIONS[world.termination[i]], "elapsed_seconds": float(world.elapsed[i]),
            "fault_detected": bool(world.fault_detected[i]), "engine_restarted": bool(world.engine_restarted[i]),
            "engine_shutdown_commanded": bool(world.engine_shutdown_commanded[i]),
            "fire_intensity_final": np.asarray(world.engines.fire[i]).tolist(),
            "engine_health_final": np.asarray(world.engines.health[i]).tolist(),
            "repair_work_seconds": float(world.repair_elapsed[i]),
            "fault_detected_seconds": float(faults[i, 0]) if faults[i, 0] >= 0.0 else None,
            "fault_10s_peak_yaw_dps": float(faults[i, 2]) if faults[i, 0] >= 0.0 else None,
            "fault_10s_peak_lateral_change_m": float(faults[i, 3]) if faults[i, 0] >= 0.0 else None,
            "thrust_restored_seconds": float(restores[i, 0]) if restores[i, 0] >= 0.0 else None,
            "fault_to_restored_thrust_seconds": float(restores[i, 0] - faults[i, 0]) if restores[i, 0] >= 0.0 else None,
            "restored_10s_peak_yaw_dps": float(restores[i, 2]) if restores[i, 0] >= 0.0 else None,
            "restored_10s_peak_lateral_change_m": float(restores[i, 3]) if restores[i, 0] >= 0.0 else None,
            "go_around_started": bool(world.go_around_started[i]), "go_around_reacquired": reacquired,
            "go_around_stage": int(world.go_around_stage[i]),
            "touchdown_count": touchdown, "paved_touchdown": paved_contact,
            "touchdown_sink_mps": sink if touchdown else None,
            "touchdown_cross_m": cross if touchdown else None,
            "touchdown_along_m": along if touchdown else None,
            "touchdown_roll_deg": float(world.touchdown_roll[i]) if touchdown else None,
            "gear_extension_final": float(world.aircraft.gear_pos[i]),
            "gear_initially_jammed": bool(world.flight.disturbances.gear_jam[i] < 1.0),
            "gear_down_time": (float(world.scenario.gear_down_time[i])
                               if np.isfinite(world.scenario.gear_down_time[i]) else None),
            "gear_jammed_final": bool(world.gear_jammed[i]), "gear_cranks": int(world.cranks[i]),
            "rollout_max_cross_m": float(world.max_rollout_lateral[i]) if touchdown else None,
            "rollout_distance_m": float(world.flight.along[i] + AIM_POINT_OFFSET - along) if touchdown and parked else None,
            "stop_along_m": float(world.flight.along[i] + AIM_POINT_OFFSET) if parked else None,
            "stop_seconds": float(world.stopped_seconds[i]), "parked": bool(parked),
            "handover_10m": handover[i, 1:].tolist() if handover[i, 0] else None,
            "success": bool(success),
            "engageable_start": None if engageable is None else bool(engageable[i]),
        })
    return records


def fly(seeds, actor, tuning, planner, config, *, payload=None, collect=False, beta=0.0, progress=None,
        packet_stress=False, native_weather=False, checkpoints=None, stage=None, residual=False,
        gear_crank=False, gear_go_around=False, engageable_starts=False, gust_kick=False, auto_reverse=False,
        far_start=False, wide_start=False, gear_late=False):
    if actor in ("mlp", "gru") and (payload is None or payload["architecture"] != actor):
        raise ValueError("student rollout needs matching policy weights")
    if collect and actor in ("managed", "prior"):
        raise ValueError("use the expert or a student for labeled trajectories")
    network = _network(payload)
    width = payload["hidden_size"] if actor == "gru" else 0
    seed_list = list(seeds)
    episodes, datasets = [], []
    offset = 0
    for begin in range(0, len(seed_list), config.batch_seeds):
        cases = make_cases(seed_list[begin:begin + config.batch_seeds], packet_stress, native_weather,
                           gear_crank, gear_go_around, engageable_starts, gust_kick, auto_reverse, far_start,
                           wide_start, gear_late)
        shard = f"{stage}-{begin // config.batch_seeds:04d}"
        loaded = checkpoints.read(shard, collect) if checkpoints is not None else None
        if loaded is not None:
            records, rows = loaded
            episodes.extend(records)
            if collect:
                datasets.append(rows)
            offset += len(cases.seeds)
            if progress is not None:
                progress(f"{actor}: resumed {shard} ({offset} episodes)")
            continue
        world = _initialize(cases.keys, cases.scenario, tuning)
        engageable = engageable_start(world)
        size = len(cases.seeds)
        started = perf_counter()
        transient = jnp.zeros((size, 4), jnp.float32).at[:, 0].set(-1.0)
        state = RollState(world, jnp.zeros((size, width), jnp.float32), jnp.zeros((size, 5), jnp.float32),
                          transient, transient)
        chunks = []
        start = 0
        while start < config.max_steps and not bool(jnp.all(state.world.done)):
            if start == FORK_STEP:
                _assert_shared_prefix(state, cases.names)
                state = state._replace(world=_fork_permanent_loss(state.world, cases.names))
            end = min(start + config.chunk_steps, config.max_steps)
            if start < FORK_STEP:
                end = min(end, FORK_STEP)
            state, data = _advance(state, cases.keys, network, tuning, jnp.int32(start), jnp.float32(beta),
                                   actor=actor, record=collect, steps=end - start, planner=planner, residual=residual)
            if collect:
                chunks.append(jax.device_get(data))
            start = end
        records = episode_rows(state, cases, engageable)
        episodes.extend(records)
        rows = None
        if collect:
            rows = _pack(chunks, size, offset)
            datasets.append(rows)
        if checkpoints is not None:
            checkpoints.write(shard, records, rows, elapsed_seconds=perf_counter() - started)
        offset += size
        if progress is not None:
            progress(f"{actor}: {offset} episodes · {sum(row['parked'] for row in episodes)} parked")
    labels = _concat_labels(datasets) if collect else None
    return episodes, labels


def summarize(rows):
    count = len(rows)
    if not count:
        return {}
    landings = [row["touchdown_sink_mps"] for row in rows if row["paved_touchdown"]]
    crosses = [row["rollout_max_cross_m"] for row in rows if row["paved_touchdown"]]
    handovers = np.asarray([row["handover_10m"] for row in rows if row["handover_10m"] is not None])
    report = {
        "episodes": count, "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / count,
        "ring_pass_rate": sum(row["ring_verdicts"].count(1) for row in rows) / (count * len(GATE_DISTANCES)),
        "all_rings": sum(all(j == 1 for j in row["ring_verdicts"]) for row in rows),
        "paved_touchdowns": len(landings), "parked": sum(row["parked"] for row in rows),
        "faults": sum(row["fault_detected"] for row in rows),
        "restarts": sum(row["engine_restarted"] for row in rows),
        "go_around_started": sum(row["go_around_started"] for row in rows),
        "go_around_reacquired": sum(row["go_around_reacquired"] for row in rows),
        "gear_below_lock_at_end": sum(row["gear_extension_final"] <= 0.9 for row in rows),
        "terminations": {name: sum(row["termination"] == name for row in rows) for name in TERMINATIONS},
        "touchdown_sink_p95_mps": float(np.percentile(landings, 95)) if landings else None,
        "touchdown_sink_max_mps": max(landings, default=None),
        "mean_twin_flight_seconds": float(np.mean([row["elapsed_seconds"] for row in rows])),
        "rollout_cross_p95_m": float(np.percentile(crosses, 95)) if crosses else None,
        "handover_10m_count": len(handovers),
    }
    if handovers.size:
        along, cross, sink, roll = handovers.T
        report["handover_10m_sink_p95_mps"] = float(np.percentile(sink, 95))
        report["handover_10m_abs_roll_max_deg"] = float(np.abs(roll).max())
        report["handover_10m_lined_up_rate"] = float(((np.abs(cross) < 15.0) & (along > -400.0) & (along < 600.0)).mean())
    for field in ("fault_10s_peak_yaw_dps", "fault_10s_peak_lateral_change_m",
                  "fault_to_restored_thrust_seconds", "restored_10s_peak_yaw_dps",
                  "restored_10s_peak_lateral_change_m"):
        samples = [row[field] for row in rows if row[field] is not None]
        report[field] = {"observed": len(samples), "p95": float(np.percentile(samples, 95)) if samples else None}
    return report


def _match(left, right):
    keys = lambda rows: {(row["seed"], row["branch"]): row["success"] for row in rows}
    a, b = keys(left), keys(right)
    if a.keys() != b.keys() or len(a) != len(left) or len(b) != len(right):
        raise ValueError("paired comparison needs identical unique seed and branch keys")
    return {
        "pairs": len(a),
        "left_only_success": sum(a[key] and not b[key] for key in a),
        "right_only_success": sum(b[key] and not a[key] for key in a),
        "both_success": sum(a[key] and b[key] for key in a),
        "both_failed": sum(not a[key] and not b[key] for key in a),
    }


def _paired_performance(left, right):
    """Left-minus-right continuous outcomes with explicit common sample counts."""
    lookup = {(row["seed"], row["branch"]): row for row in right}
    pairs = [(row, lookup[(row["seed"], row["branch"])]) for row in left]
    metrics = {}
    for field in ("touchdown_sink_mps", "rollout_max_cross_m", "fault_to_restored_thrust_seconds",
                  "fault_10s_peak_yaw_dps", "fault_10s_peak_lateral_change_m", "restored_10s_peak_yaw_dps",
                  "restored_10s_peak_lateral_change_m"):
        comparable = pairs if field.startswith(("fault_", "restored_")) else [
            (a, b) for a, b in pairs if a["paved_touchdown"] and b["paved_touchdown"]]
        delta = [a[field] - b[field] for a, b in comparable if a[field] is not None and b[field] is not None]
        metrics[field] = {"pairs": len(delta), "median_delta": float(np.median(delta)) if delta else None,
                          "p95_delta": float(np.percentile(delta, 95)) if delta else None}
    return metrics


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_file(path, write):
    pending = path.with_name(path.name + ".pending")
    with pending.open("wb") as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _write_report(path, report):
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_file(path, lambda stream: stream.write(raw))


def _source_identity():
    here = Path(__file__).resolve().parent
    packages = {here.name: here, "mayday_jax": here.parent / "mayday_jax"}
    return {f"{package}/{path.name}": _digest(path)
            for package, folder in packages.items()
            for path in sorted(folder.iterdir()) if path.suffix == ".py" or path.name == "feature_names.txt"}


class Checkpoints:
    """One source, configuration and seed identity, with manifest-last commits."""

    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "identity.json"
        # JSON normalizes tuples before the equality check on resumed runs.
        self.identity = json.loads(json.dumps(identity, allow_nan=False))
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != self.identity:
                raise ValueError("checkpoint source, device, tuning or run configuration differs")
        elif any(self.directory.iterdir()):
            raise ValueError("checkpoint directory contains files without an experiment identity")
        else:
            _write_report(path, self.identity)

    def read(self, stage, has_labels=False):
        manifest = self.directory / f"{stage}.json"
        if not manifest.is_file():
            return None
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if not has_labels:
            return saved["data"], None
        path = manifest.with_suffix(".npz")
        if not path.is_file() or _digest(path) != saved["labels_sha256"]:
            raise ValueError(f"checksum mismatch for checkpoint {stage}")
        with np.load(path, allow_pickle=False) as archive:
            labels = Labels(*(archive[name] if name in archive.files else None for name in Labels._fields))
        return saved["data"], labels

    def write(self, stage, data, labels=None, elapsed_seconds=None):
        manifest = self.directory / f"{stage}.json"
        document = {"data": data}
        if elapsed_seconds is not None:
            document["elapsed_seconds"] = round(elapsed_seconds, 4)
        if labels is not None:
            path = manifest.with_suffix(".npz")
            _atomic_file(path, lambda stream: np.savez_compressed(
                stream, **{name: value for name, value in labels._asdict().items() if value is not None}))
            document["labels_sha256"] = _digest(path)
        _write_report(manifest, document)

    def timings(self):
        durations = {}
        for path in self.directory.glob("*.json"):
            if path.name == "identity.json":
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if "elapsed_seconds" in data:
                durations[path.stem] = data["elapsed_seconds"]
        return durations


def train_and_evaluate(args, progress=print):
    if args.train_seeds >= HOLDOUT_SEED_START - TRAIN_SEED_START:
        raise ValueError("training seed range overlaps held-out range")
    if args.holdout_seeds < 1 or args.train_seeds < 1 or args.batch_seeds < 1 or args.dagger_rounds < 0:
        raise ValueError("positive seed counts/batch and nonnegative DAgger rounds required")
    train_seeds = list(range(TRAIN_SEED_START, TRAIN_SEED_START + args.train_seeds))
    evaluation_only = args.evaluate_model is not None
    first_holdout = args.eval_seed_start
    if first_holdout is None:
        if evaluation_only:
            first_holdout = CONFIRMATION_SEED_START
        else:
            first_holdout = HOLDOUT_SEED_START if args.controller == "direct" else SUCCESSOR_HOLDOUT_SEED_START
    if first_holdout < 0 or first_holdout + args.holdout_seeds > 2**32:
        raise ValueError("held-out seeds must be uint32")
    holdout_seeds = list(range(first_holdout, first_holdout + args.holdout_seeds))
    if set(train_seeds).intersection(holdout_seeds):
        raise ValueError("training seeds overlap held-out seeds")
    residual = args.controller == "residual"
    prior_only = args.controller == "prior"
    tuning = default_tuning() if args.meta is None else load_tuning(args.meta)
    payloads = {}
    if evaluation_only:
        if args.baseline_only:
            raise ValueError("fixed saved model evaluation requires a held-out split")
        candidate = json.loads(args.evaluate_model.read_text(encoding="utf-8"))
        if (prior_only or residual) and not np.array_equal(
                np.asarray(candidate.get("stall_speeds"), dtype=np.float32),
                np.asarray(tuning.stall_speeds, dtype=np.float32)):
            raise ValueError("saved guidance artifact stall speeds do not match evaluation tuning")
        if prior_only:
            expected = wrap_prior(stall_speeds=tuning.stall_speeds)
            if any(candidate.get(field) != value for field, value in expected.items()):
                raise ValueError("saved prior artifact does not match the observable guidance schema")
        else:
            candidate = unwrap_residual(candidate) if residual else candidate
            policy = model.Policy(candidate)
            payloads[policy.architecture] = candidate
    planner = PlannerConfig(samples=args.samples, horizon=args.horizon, clairvoyant=not args.blind_teacher,
                            anneal=args.anneal)
    config = RunConfig(max_steps=round(args.episode_seconds / CONTROL_DT), batch_seeds=args.batch_seeds)
    if config.max_steps <= FORK_STEP:
        raise ValueError("episode limit must allow the repair fork")
    args.out.mkdir(parents=True, exist_ok=True)
    actual_device = str(jnp.zeros(()).device)
    metadata = {
        "schema": "mayday-landing-experiment-v1", "acceptance": "offline_only",
        "train_seeds": [] if prior_only or args.baseline_only or evaluation_only else train_seeds,
        "holdout_seeds": holdout_seeds,
        "planner": asdict(planner), "run": asdict(config), "training": [],
        "criteria": {"all_five_rings": True, "paved_touchdown": True, "max_touchdown_sink_mps": MAX_SINK,
                     "stopped_with_brakes_seconds": 3.0, "go_around_requires_reacquisition": True},
        "device": actual_device, "repair_fork_seconds": FORK_SECONDS, "evaluations": {},
        "controller": args.controller,
        "prior": PRIOR_SCHEMA if residual or prior_only else None,
        "residual_limits": RESIDUAL_LIMITS if residual else None,
        "tuning": {name: np.asarray(value).tolist() for name, value in tuning._asdict().items()},
        "tuning_source": None if args.meta is None else {
            "name": args.meta.name, "sha256": hashlib.sha256(args.meta.read_bytes()).hexdigest()},
        "comparison": "The managed twin controller approximates the existing rule-based approach and rollout.",
        "packet_stress": args.packet_stress,
        "native_weather": args.native_weather,
        "gear_crank": args.gear_crank,
        "blind_teacher": args.blind_teacher,
        "privileged_aux": args.privileged_aux,
        "gear_go_around": args.gear_go_around,
        "engageable_starts": args.engageable_starts,
        "gust_kick": args.gust_kick,
        "auto_reverse": args.auto_reverse,
        "far_starts": args.far_starts,
        "wide_starts": args.wide_starts,
        "gear_late": args.gear_late,
        "teacher_anneal": args.anneal,
        "restored_thrust_fraction": 0.12,
        "transient_window_seconds": 10.0,
    }
    if args.baseline_only:
        metadata["diagnostic_seeds"] = train_seeds
    if evaluation_only:
        metadata["fixed_evaluation_artifact"] = {
            "name": args.evaluate_model.name, "sha256": _digest(args.evaluate_model)}
    checkpoint_identity = {
        "source": _source_identity(), "jax": jax.__version__, "numpy": np.__version__,
        "experiment": metadata,
        "arguments": {name: value for name, value in vars(args).items()
                      if name not in ("out", "checkpoint_dir", "meta", "evaluate_model")},
    }
    checkpoints = Checkpoints(args.checkpoint_dir or args.out / "checkpoints", checkpoint_identity)
    emit = lambda line: progress(line, flush=True) if progress is print else progress(line)
    evaluate = lambda seeds, actor, **opts: fly(seeds, actor, tuning, planner, config, progress=emit,
                                              packet_stress=args.packet_stress, checkpoints=checkpoints,
                                              residual=residual and actor != "managed",
                                              native_weather=args.native_weather, gear_crank=args.gear_crank,
                                              gear_go_around=args.gear_go_around,
                                              engageable_starts=args.engageable_starts,
                                              gust_kick=args.gust_kick, auto_reverse=args.auto_reverse,
                                              far_start=args.far_starts, wide_start=args.wide_starts,
                                              gear_late=args.gear_late, **opts)
    if args.baseline_only or prior_only or evaluation_only:
        eval_seeds = train_seeds if args.baseline_only else holdout_seeds
    else:
        emit("Collecting causal MPPI demonstrations")
        first, demonstrations = evaluate(train_seeds, "teacher", collect=True, stage="demonstrations")
        training_parts = [demonstrations]
        metadata["demonstration_metrics"] = {name: summarize([r for r in first if r["branch"] == name]) for name in BRANCHES}
        metadata["demonstration_samples"] = int(len(demonstrations.actions))
        emit(f"MPPI demonstrations: {len(demonstrations.actions):,} training rows")
        for index in range(args.dagger_rounds + 1):
            dataset = _join(training_parts)
            settings = model.TrainingConfig(steps=args.train_steps if index == 0 else args.dagger_steps,
                                            hidden_size=args.hidden_size, batch_size=args.train_batch,
                                            action_weights=(3.0, 2.0, 1.5, 1.0, 0.0 if residual else 1.0),
                                            hard_weight=args.hard_weight,
                                            label_center=(0.0, 0.0, 0.0, 0.5 if residual else 0.0, 0.0))
            training = {"round": index, "training_rows": len(dataset.actions), "models": {}}
            for architecture in ("mlp", "gru"):
                emit(f"Training {architecture} round {index} with {len(dataset.actions):,} rows")
                stage = f"round-{index:02d}-{architecture}"
                saved = checkpoints.read(stage)
                if saved is None:
                    started = perf_counter()
                    privileged = dataset.privileged if args.privileged_aux and architecture == "gru" else None
                    payloads[architecture] = model.train(dataset.observations, dataset.actions, dataset.phases,
                                                           dataset.episode_ids, privileged=privileged,
                                                           architecture=architecture, feature_names=NAMES,
                                                           seed=541 + index, config=settings,
                                                           initial_payload=payloads.get(architecture))
                    checkpoints.write(stage, payloads[architecture], elapsed_seconds=perf_counter() - started)
                else:
                    payloads[architecture], _ = saved
                    model.Policy(payloads[architecture])
                    emit(f"Resumed {stage}")
                training["models"][architecture] = payloads[architecture]["training"]
            metadata["training"].append(training)
            if index < args.dagger_rounds:
                beta = 0.15 if index == 0 else 0.03
                training["teacher_action_probability_next_round"] = beta
                for architecture, payload in payloads.items():
                    emit(f"DAgger round {index + 1}: relabeling {architecture} trajectories")
                    _, collected = evaluate(train_seeds, architecture, payload=payload, collect=True, beta=beta,
                                            stage=f"dagger-{index + 1:02d}-{architecture}")
                    training_parts.append(collected)
        for architecture, payload in payloads.items():
            payload["qualification"] = "offline_experimental"
            path = args.out / f"{architecture}.json"
            _write_report(path, wrap_residual(payload, stall_speeds=tuning.stall_speeds) if residual else payload)
            metadata.setdefault("policy_sha256", {})[path.name] = _digest(path)
        eval_seeds = holdout_seeds
    if (residual or prior_only) and not args.baseline_only:
        path = args.out / "prior.json"
        prior_payload = wrap_prior(stall_speeds=tuning.stall_speeds)
        prior_payload["qualification"] = "offline_observable_baseline"
        _write_report(path, prior_payload)
        metadata.setdefault("policy_sha256", {})[path.name] = _digest(path)
    baselines = ("managed", "prior") if residual or prior_only else ("managed",)
    actors = baselines + (() if args.no_teacher_eval or prior_only or evaluation_only else ("teacher",)) + tuple(payloads)
    for actor in actors:
        emit(f"Evaluating {actor} on {len(eval_seeds)} {'training diagnostic' if args.baseline_only else 'held-out'} seeds")
        started = perf_counter()
        rows, _ = evaluate(eval_seeds, actor, payload=payloads.get(actor), stage=f"evaluation-{actor}")
        metadata["evaluations"][actor] = {
            "split": "training_diagnostic" if args.baseline_only else "holdout",
            "duration_seconds": round(perf_counter() - started, 2), "overall": summarize(rows),
            "branches": {name: summarize([r for r in rows if r["branch"] == name]) for name in BRANCHES},
            "branches_gear_locked": {name: summarize([r for r in rows if r["branch"] == name
                and r["gear_extension_final"] > 0.9]) for name in BRANCHES},
            "observation_lag": {str(lag): summarize([r for r in rows if r["observation_lag_steps"] == lag])
                                for lag in (0, 1, 3, 6)},
            "weather": {name: summarize([r for r in rows if r["weather_family"] == name])
                        for name in (WEATHER_NAMES if args.native_weather else ("legacy",))},
            "paired_repair": {name: _match([r for r in rows if r["branch"] == "repaired"],
                [dict(r, branch="repaired") for r in rows if r["branch"] == name])
                for name in REPAIR_BRANCHES if name != "repaired"},
            "episodes": rows,
        }
        _write_report(args.out / "report.json", metadata)
    comparisons = (("gru", "mlp"),) + tuple((name, base) for name in ("mlp", "gru") for base in baselines)
    if "prior" in baselines:
        comparisons += (("prior", "managed"),)
    results = metadata["evaluations"]
    metadata["paired_models"] = {
        f"{left}_vs_{right}": {
            name: _match([r for r in results[left]["episodes"] if r["branch"] == name],
                         [r for r in results[right]["episodes"] if r["branch"] == name])
            for name in BRANCHES}
        for left, right in comparisons if left in results and right in results
    }
    metadata["paired_performance"] = {
        f"{left}_vs_{right}": {
            name: _paired_performance([r for r in results[left]["episodes"] if r["branch"] == name],
                                      [r for r in results[right]["episodes"] if r["branch"] == name])
            for name in BRANCHES}
        for left, right in comparisons if left in results and right in results
    }
    metadata["stage_wall_seconds"] = checkpoints.timings()
    for actor, result in metadata["evaluations"].items():
        seconds = sum(value for name, value in metadata["stage_wall_seconds"].items()
                      if name.startswith(f"evaluation-{actor}-"))
        flights = sum(row["elapsed_seconds"] for row in result["episodes"])
        result["twin_flight_seconds_per_wall_second"] = flights / seconds if seconds > 0.0 else None
    _write_report(args.out / "report.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meta", type=Path, help="FrameRecorder metadata for the game's physics tuning")
    parser.add_argument("--checkpoint-dir", type=Path, help="resume directory, default <out>/checkpoints")
    parser.add_argument("--evaluate-model", type=Path, help="benchmark one frozen artifact without training")
    parser.add_argument("--eval-seed-start", type=int, help="first explicit held-out seed")
    parser.add_argument("--train-seeds", type=int, default=24)
    parser.add_argument("--holdout-seeds", type=int, default=16)
    parser.add_argument("--batch-seeds", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--dagger-steps", type=int, default=750)
    parser.add_argument("--dagger-rounds", type=int, default=1)
    parser.add_argument("--train-batch", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--episode-seconds", type=float, default=MAX_EPISODE_SECONDS + 0.5)
    parser.add_argument("--device", choices=("cpu", "gpu"))
    parser.add_argument("--controller", choices=("direct", "residual", "prior"), default="direct")
    parser.add_argument("--packet-stress", action="store_true", help="add jitter and loss to delayed cohorts")
    parser.add_argument("--native-weather", action="store_true", help="sample native weather and route-leg profiles")
    parser.add_argument("--privileged-aux", action="store_true",
                        help="GRU auxiliary head regresses true gusts, co-pilot input and the delay gap (RMA-style)")
    parser.add_argument("--engageable-starts", action="store_true",
                        help="sample intercepts inside the director's engagement window (|cross| <= 220 m)")
    parser.add_argument("--gear-late", action="store_true", help="crew lowers gear late or never; assist goes around then hands back")
    parser.add_argument("--wide-starts", action="store_true", help="odd speed/angle starts up to 9 km, 600 m off line")
    parser.add_argument("--far-starts", action="store_true", help="armed-join starts 9-20 km out, up to 4 km off line")
    parser.add_argument("--auto-reverse", action="store_true", help="assist uses reverse thrust during rollout")
    parser.add_argument("--gust-kick", action="store_true", help="native +/-12 deg bank kick at each front gust")
    parser.add_argument("--anneal", type=int, default=1, help="DIAL-MPC style MPPI iterations per control step")
    parser.add_argument("--hard-weight", type=float, default=0.0,
                        help="oversample large teacher corrections: weight 1 + w*|label - center|^2")
    parser.add_argument("--blind-teacher", action="store_true", help="MPPI predictions omit future gust/co-pilot pulses")
    parser.add_argument("--gear-crank", action="store_true", help="crew hand-cranks a jammed gear (native mechanism)")
    parser.add_argument("--gear-go-around", action="store_true", help="go around once if the gear is unlocked at ring 4")
    parser.add_argument("--baseline-only", action="store_true", help="evaluate on training seeds for tuning")
    parser.add_argument("--no-teacher-eval", action="store_true", help="skip held-out teacher evaluation")
    args = parser.parse_args()
    device = jax.devices(args.device)[0] if args.device else jax.devices()[0]
    with jax.default_device(device):
        train_and_evaluate(args)


if __name__ == "__main__":
    main()
