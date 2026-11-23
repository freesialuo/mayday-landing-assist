# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Compare the JAX twin with one-step snapshots from the game's FrameRecorder."""

import argparse
import csv
import json
from pathlib import Path

import jax
import numpy as np

from .physics import Controls, Ground, State, Tuning, step


_STATE_FIELDS = {
    "px": "px", "py": "py", "pz": "pz", "speed": "speed", "gamma": "gamma",
    "track": "track", "heading": "heading", "pitch": "pitch", "roll": "roll",
    "prate": "prate", "rrate": "rrate", "yrate": "yrate", "onGround": "on_ground",
    "gearPos": "gear_pos", "flapPos": "flap_pos", "steerAngle": "steer_angle",
    "wx": "wx", "wy": "wy", "wz": "wz", "windFactor": "wind_factor", "alpha": "alpha",
    "alphaReq": "alpha_req", "loadFactor": "load_factor", "groundY": "ground_y",
    "agl": "agl", "groundEffect": "ground_effect", "wow": "wow", "surface": "surface",
    "stalling": "stalling", "stallWarn": "stall_warn", "skidding": "skidding", "belly": "belly",
    "overspeed": "overspeed", "braking": "braking", "drifting": "drifting",
    "trackLag": "track_lag", "backward": "backward",
}
_CONTROL_FIELDS = {
    "c_pitchRate": "pitch_rate", "c_rollRate": "roll_rate", "c_rudder": "rudder",
    "c_steer": "steer", "c_brakes": "brakes", "c_parkBrake": "park_brake", "c_thrL": "thr_l",
    "c_thrR": "thr_r", "c_dragMul": "drag_mul", "c_gearDown": "gear_down",
    "c_gearJammed": "gear_jammed", "c_gearJamPos": "gear_jam_pos", "c_flap": "flap",
    "c_heave": "heave", "c_reverse": "reverse",
}

# The comparison fields and tolerances match Tests/TwinTests.cs.
_COMPARE_FIELDS = (
    "px", "py", "pz", "speed", "gamma", "track", "heading", "pitch", "roll", "prate",
    "rrate", "yrate", "gearPos", "flapPos", "steerAngle", "windFactor", "alpha", "alphaReq",
    "loadFactor", "agl", "groundEffect", "wow",
)
_COMPARE_FLAGS = ("onGround", "stalling")


def _abs_tolerance(key):
    return 1e-2 if key in ("px", "py", "pz") else 1e-3


# Relative tolerance on top of the absolute one; only alphaReq has one. At taxi speed
# alpha_req = 14 * (Vs / V)**2 reaches ~9,000, where one float32 ulp is already ~1e-3,
# so CPU/GPU rounding-order differences of a few ulp must not fail the replay.
_REL_TOLERANCE = {"alphaReq": 1e-6}
_REQUIRED = (
    "dt", *("pre_" + key for key in _STATE_FIELDS), *_CONTROL_FIELDS,
    *("post_" + key for key in (*_COMPARE_FIELDS, *_COMPARE_FLAGS, "groundY", "surface")),
)
_BOOL_COLUMNS = {
    "pre_onGround", "pre_stalling", "pre_stallWarn", "pre_skidding", "pre_belly",
    "pre_overspeed", "pre_braking", "pre_drifting", "pre_trackLag", "pre_backward",
    "c_parkBrake", "c_gearDown", "c_gearJammed", "c_reverse", "post_onGround", "post_stalling",
}
_INTEGER_LIMITS = {
    "pre_surface": (0, 255),
    "post_surface": (0, 255),
    "c_flap": (-(2**31), 2**31 - 1),
}

_step_batch = jax.jit(jax.vmap(step, in_axes=(0, 0, 0, None, 0, 0)))


def _read_columns(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if not header:
            raise ValueError(f"{path}: missing CSV header")
        positions = {name: index for index, name in enumerate(header)}
        if len(positions) != len(header) or "" in positions:
            raise ValueError(f"{path}: CSV header contains duplicate or unnamed columns")
        missing = [name for name in _REQUIRED if name not in positions]
        if missing:
            raise ValueError(f"{path}: missing CSV columns: {', '.join(missing)}")

        indices = [positions[name] for name in _REQUIRED]
        rows = []
        for row_number, row in enumerate(reader, start=1):
            if len(row) != len(header):
                raise ValueError(
                    f"{path}: data row {row_number} has {len(row)} columns, expected {len(header)}"
                )
            values = []
            for name, index in zip(_REQUIRED, indices):
                try:
                    values.append(float(row[index]))
                except ValueError as exc:
                    raise ValueError(
                        f"{path}: data row {row_number} has an invalid {name}: {row[index]!r}"
                    ) from exc
            rows.append(values)

    if not rows:
        raise ValueError(f"{path}: no recorded ticks")

    columns = {}
    for name, values in zip(_REQUIRED, np.asarray(rows, dtype=np.float64).T):
        with np.errstate(over="ignore"):
            single = values.astype(np.float32)
        invalid = ~np.isfinite(single)
        if name == "dt":
            invalid |= single <= 0
        elif name in _BOOL_COLUMNS:
            invalid |= (values != 0) & (values != 1)
        elif name in _INTEGER_LIMITS:
            lower, upper = _INTEGER_LIMITS[name]
            invalid |= (values < lower) | (values > upper) | (values != np.trunc(values))
        if np.any(invalid):
            index = int(np.argmax(invalid))
            raise ValueError(f"{path}: data row {index + 1} has an invalid {name}: {values[index]}")

        if name in _BOOL_COLUMNS:
            columns[name] = values.astype(np.bool_)
        elif name in _INTEGER_LIMITS:
            columns[name] = values.astype(np.int32)
        else:
            columns[name] = single
    return columns


def load_tuning(meta_path):
    """Read the physics tuning from a FrameRecorder metadata file."""
    path = Path(meta_path)
    with path.open(encoding="utf-8-sig") as stream:
        meta = json.load(stream)
    if not isinstance(meta, dict) or not isinstance(meta.get("assists"), dict):
        raise ValueError(f"{path}: missing assists tuning object")
    assists = meta["assists"]

    def number(value, name):
        if type(value) in (int, float):
            with np.errstate(over="ignore"):
                try:
                    single = np.float32(value)
                except OverflowError:
                    single = np.float32(np.inf)
            if np.isfinite(single):
                return single
        raise ValueError(f"{path}: tuning parameter {name} must be a finite float")

    def flag(name):
        value = assists.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"{path}: tuning parameter assists.{name} must be a boolean")
        return value

    speeds = meta.get("stallSpeeds")
    if not isinstance(speeds, list) or len(speeds) != 3:
        raise ValueError(f"{path}: stallSpeeds must contain three flap-notch speeds")
    return Tuning(
        stall_speeds=np.asarray([number(speed, f"stallSpeeds[{i}]") for i, speed in enumerate(speeds)]),
        wheelbase=number(meta.get("wheelbase"), "wheelbase"),
        wing_strike_roll=number(meta.get("wingStrikeRoll"), "wingStrikeRoll"),
        assists_enabled=flag("enabled"),
        auto_flaps=flag("autoFlaps"),
        handbrake_fun=flag("handbrakeFun"),
        reverse_thrust=number(assists.get("reverseThrust"), "assists.reverseThrust"),
        reverse_fade_start=number(assists.get("reverseFadeStart"), "assists.reverseFadeStart"),
        reverse_max_speed=number(assists.get("reverseMaxSpeed"), "assists.reverseMaxSpeed"),
    )


def replay(csv_path, device=None):
    """Replay every CSV tick independently and return its comparison report.

    The sibling ``.meta.json`` supplies the recording's tuning. ``device`` accepts
    a JAX device, a backend name such as ``"cpu"`` or ``"gpu"``, or ``None`` for
    JAX's default placement. The report includes the frame count, per-field
    errors, flag mismatch counts and an overall pass status. Invalid recordings
    raise an exception.
    """
    path = Path(csv_path)
    columns = _read_columns(path)
    tuning = load_tuning(path.with_suffix(".meta.json"))
    pre = State(**{name: columns["pre_" + key] for key, name in _STATE_FIELDS.items()})
    controls = Controls(**{name: columns[key] for key, name in _CONTROL_FIELDS.items()})
    ground_before = Ground(columns["pre_groundY"], columns["pre_surface"])
    ground_after = Ground(columns["post_groundY"], columns["post_surface"])

    if isinstance(device, str):
        device = jax.devices(device)[0]
    inputs = jax.device_put((pre, controls, columns["dt"], tuning, ground_before, ground_after), device)
    result = _step_batch(*inputs)
    used_device = str(next(iter(result.px.devices())))
    result = jax.device_get(result)

    fields = {}
    for key in _COMPARE_FIELDS:
        delta = getattr(result, _STATE_FIELDS[key]).astype(np.float64) - columns["post_" + key]
        if key in ("track", "heading"):
            finite = np.isfinite(delta)
            delta[finite] = (delta[finite] + 180.0) % 360.0 - 180.0
        error = np.where(np.isfinite(delta), np.abs(delta), np.inf)
        worst = int(np.argmax(error))
        relative = _REL_TOLERANCE.get(key, 0.0)
        frame_tolerance = np.maximum(_abs_tolerance(key), relative * np.abs(columns["post_" + key]))
        fields[key] = {
            "max_abs_error": float(error[worst]),
            "rms_error": float(np.sqrt(np.mean(error**2))),
            "tolerance": _abs_tolerance(key),
            "relative_tolerance": relative,
            # Worst |error| / per-frame tolerance max(absolute, relative * |recorded|); <= 1 passes.
            "max_error_ratio": float(np.max(error / frame_tolerance)),
            "worst_row": worst + 1 if error[worst] else None,
        }

    mismatches = {
        key: getattr(result, _STATE_FIELDS[key]) != columns["post_" + key] for key in _COMPARE_FLAGS
    }
    mismatch_frames = int(np.count_nonzero(np.logical_or.reduce(tuple(mismatches.values()))))
    return {
        "frames": len(columns["dt"]),
        "passed": mismatch_frames == 0 and all(
            field["max_error_ratio"] <= 1.0 for field in fields.values()
        ),
        "fields": fields,
        "flag_mismatches": {key: int(np.count_nonzero(value)) for key, value in mismatches.items()},
        "flag_mismatch_frames": mismatch_frames,
        "device": used_device,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), help="JAX backend (defaults to automatic)")
    args = parser.parse_args(argv)
    try:
        report = replay(args.csv_path, args.device)
    except (OSError, ValueError, csv.Error, RuntimeError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")

    print(f"Replayed {report['frames']} ticks from {args.csv_path.name} on {report['device']}")
    print(f"{'field':<14}{'max |err|':>14}{'rms':>14}{'tolerance':>14}{'err/tol':>10}  worst row")
    for key, field in report["fields"].items():
        mark = "  <-- check" if field["max_error_ratio"] > 1.0 else ""
        rel = f"  (max with {field['relative_tolerance']:g}*|v|)" if field["relative_tolerance"] else ""
        print(
            f"{key:<14}{field['max_abs_error']:>14.6g}{field['rms_error']:>14.6g}"
            f"{field['tolerance']:>14.6g}{field['max_error_ratio']:>10.3f}  {field['worst_row'] or '-'}{mark}{rel}"
        )
    print(f"Flag mismatches across {report['flag_mismatch_frames']} ticks")
    for key, count in report["flag_mismatches"].items():
        print(f"  {key} {count}")
    print("REPLAY MATCH" if report["passed"] else "REPLAY MISMATCH")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
