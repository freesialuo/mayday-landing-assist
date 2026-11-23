"""Compare Mayday landing JSONL recordings without mixing recording clocks."""

import argparse
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median


SCHEMA = "mayday-landing.trace/1"
ENGINE_FIELDS = ("fire", "repair", "rpm", "thrust", "health", "state")
WEATHER_FIELDS = ("wind_along", "wind_cross", "rain", "visibility", "turbulence", "ice")
FIELDS = tuple(f"{side}_{field}" for side in ("left", "right") for field in ENGINE_FIELDS) + WEATHER_FIELDS


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def timestamp(value):
    return float(value) if finite(value) and value >= 0 else None


@dataclass(frozen=True)
class Sample:
    index: int
    local: float | None
    source: float | None
    engine_source: float | None
    values: dict


@dataclass(frozen=True)
class Event:
    name: str
    previous: Sample
    current: Sample


def _observation(values, names, valid):
    if not valid:
        return {}
    if not isinstance(values, list) or len(values) != len(names):
        raise ValueError("valid row has a missing or incorrectly sized observation")
    observation = dict(zip(names, values, strict=True))
    out = {}
    for name in FIELDS:
        value = observation.get(name)
        if not finite(value):
            continue
        if name not in WEATHER_FIELDS:
            side = name.split("_", 1)[0]
            side_valid = observation.get(f"{side}_valid")
            if not finite(side_valid) or side_valid <= 0.5:
                continue
        if name == "visibility" and value <= 0:
            continue
        out[name] = value
    return out


def load_trace(path):
    """Load the trace/1 feature manifest and valid samples by role and local session."""
    sessions = defaultdict(list)
    row_counts = defaultdict(int)
    with Path(path).open(encoding="utf-8-sig") as stream:
        try:
            header = json.loads(stream.readline())
        except ValueError as error:
            raise ValueError(f"{path}: invalid trace header") from error
        if not isinstance(header, dict) or header.get("schema") != SCHEMA:
            raise ValueError(f"{path}: expected {SCHEMA}")
        names = header.get("features")
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise ValueError(f"{path}: invalid feature manifest")
        for line_number, line in enumerate(stream, start=2):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or type(row.get("host")) is not bool or type(row.get("session")) is not int:
                    raise ValueError("row requires a boolean host and integer session")
                key = ("host" if row["host"] else "client", row["session"])
                row_counts[key] += 1
                if row.get("valid") is not True:
                    continue
                sessions[key].append(Sample(
                    line_number,
                    timestamp(row.get("time")),
                    timestamp(row.get("sourceTime")),
                    timestamp(row.get("engineSourceTime")),
                    _observation(row.get("observation"), names, True),
                ))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return sessions, row_counts


def _range(samples, attr):
    times = [value for row in samples if (value := getattr(row, attr)) is not None]
    return [min(times), max(times)] if times else None


def summarize_trace(trace):
    """Summarize one file even when no paired capture or clocks are available."""
    sessions, counts = trace
    summary = []
    for (role, session), count in sorted(counts.items()):
        samples = sessions.get((role, session), [])
        fields = {}
        for name in FIELDS:
            values = [sample.values[name] for sample in samples if name in sample.values]
            if values:
                fields[name] = {"samples": len(values), "min": min(values), "max": max(values)}
        summary.append({
            "role": role,
            "session": session,
            "rows": count,
            "valid_rows": len(samples),
            "local_time_range_s": _range(samples, "local"),
            "motion_source_time_range_s": _range(samples, "source"),
            "engine_source_stamps": sum(sample.engine_source is not None for sample in samples),
            "signals": fields,
        })
    return summary


def _motion_sessions(trace, role, session):
    sessions, _ = trace
    for (sample_role, session_id), rows in sessions.items():
        if sample_role != role or (session is not None and session_id != session):
            continue
        stamped = [row for row in rows if row.source is not None]
        if len(stamped) >= 2 and all(a.source <= b.source for a, b in zip(stamped, stamped[1:])):
            yield session_id, stamped


def _paired_session(host, client, host_session, client_session):
    candidates = []
    for host_id, host_rows in _motion_sessions(host, "host", host_session):
        for client_id, client_rows in _motion_sessions(client, "client", client_session):
            start = max(host_rows[0].source, client_rows[0].source)
            end = min(host_rows[-1].source, client_rows[-1].source)
            if end > start:
                candidates.append((host_id, client_id, host_rows, client_rows, (start, end)))
    if len(candidates) != 1:
        reason = ("ambiguous overlapping sessions; select each recording's local session ID" if candidates
                  else "no monotonic motion-source overlap for the requested roles and sessions")
        return None, reason
    return candidates[0], None


def _aligned_samples(host_rows, client_rows, overlap, tolerance):
    start, end = overlap
    host = [row for row in host_rows if start <= row.source <= end]
    host_times = [row.source for row in host]
    for row in client_rows:
        if not start <= row.source <= end:
            continue
        at = bisect_left(host_times, row.source)
        neighbors = host[max(0, at - 1):at + 1]
        if neighbors:
            closest = min(neighbors, key=lambda sample: abs(sample.source - row.source))
            if abs(closest.source - row.source) <= tolerance:
                yield closest, row


def _signal_comparison(pairs):
    comparison = {}
    for name in FIELDS:
        deltas = [client.values[name] - host.values[name] for host, client in pairs
                  if name in host.values and name in client.values]
        if not deltas:
            comparison[name] = {"status": "unavailable", "samples": 0}
        else:
            comparison[name] = {
                "status": "compared",
                "samples": len(deltas),
                "mean_client_minus_host": sum(deltas) / len(deltas),
                "mean_absolute_difference": sum(map(abs, deltas)) / len(deltas),
                "max_absolute_difference": max(map(abs, deltas)),
            }
    return comparison


def _events(rows, overlap, max_edge_gap):
    events = []
    for before, after in zip(rows, rows[1:]):
        if (after.index != before.index + 1 or before.local is None or after.local is None
                or not 0 < after.local - before.local <= max_edge_gap
                or not overlap[0] <= before.source <= after.source <= overlap[1]
                or after.source - before.source > max_edge_gap):
            continue
        for side in ("left", "right"):
            for field, threshold in (("fire", 0.01), ("repair", 0.01), ("repair", 0.99), ("state", None)):
                key = f"{side}_{field}"
                previous, current = before.values.get(key), after.values.get(key)
                if previous is None or current is None:
                    continue
                if threshold is None:
                    if previous != current:
                        events.append(Event(f"{side}_state_{previous:g}_to_{current:g}", before, after))
                elif (previous >= threshold) != (current >= threshold):
                    if field == "repair" and threshold == 0.99 and current < threshold:
                        continue
                    suffix = ("complete" if threshold == 0.99 else "start") if current >= threshold else "end"
                    events.append(Event(f"{key}_{suffix}", before, after))
    return events


def _event_delays(host_rows, client_rows, overlap, offset, event_window, max_edge_gap):
    if offset is None:
        return {"status": "unavailable", "reason": "no independent calibration of the recorder-local clocks"}
    host_events = _events(host_rows, overlap, max_edge_gap)
    client_events = _events(client_rows, overlap, max_edge_gap)

    def matching(event, candidates):
        return [index for index, other in enumerate(candidates)
                if event.name == other.name and abs(event.current.source - other.current.source) <= event_window]

    delays = []
    inconsistent = 0
    for index, host in enumerate(host_events):
        candidates = matching(host, client_events)
        if len(candidates) != 1:
            continue
        client = client_events[candidates[0]]
        if matching(client, host_events) != [index]:
            continue
        minimum = client.previous.local + offset - host.current.local
        maximum = client.current.local + offset - host.previous.local
        if maximum < 0:
            inconsistent += 1
            continue
        delays.append({
            "event": host.name,
            "host_motion_source_s": host.current.source,
            "client_motion_source_s": client.current.source,
            "first_observation_delay_s": client.current.local + offset - host.current.local,
            "sampling_bounds_s": [minimum, maximum],
        })
    result = {
        "status": "estimated" if delays else "unavailable",
        "host_edges": len(host_events),
        "client_edges": len(client_events),
        "inconsistent_pairs": inconsistent,
        "pairs": delays,
    }
    if delays:
        result["median_s"] = median(item["first_observation_delay_s"] for item in delays)
    else:
        result["reason"] = "no uniquely paired valid edges consistent with the supplied clock offset"
    return result


def compare_traces(host, client, *, same_flight=False, host_session=None, client_session=None,
                   motion_tolerance=0.08, client_clock_offset=None, event_window=1.0, max_edge_gap=0.5):
    """Align source-stamped observations and optionally estimate sampled event delay."""
    for name, value in (("motion tolerance", motion_tolerance), ("event window", event_window), ("edge gap", max_edge_gap)):
        if not finite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if client_clock_offset is not None and not finite(client_clock_offset):
        raise ValueError("client clock offset must be finite")

    result = {"host_recording": summarize_trace(host), "client_recording": summarize_trace(client)}
    if not same_flight:
        selection, reason = None, "confirm same-flight provenance; trace/1 session IDs are recorder-local"
    else:
        selection, reason = _paired_session(host, client, host_session, client_session)
    if selection is None:
        result["motion_alignment"] = {"status": "unavailable", "reason": reason}
        result["event_delay"] = {"status": "unavailable", "reason": "no shared motion alignment"}
        return result

    host_id, client_id, host_rows, client_rows, overlap = selection
    pairs = list(_aligned_samples(host_rows, client_rows, overlap, motion_tolerance))
    result["motion_alignment"] = {
        "status": "aligned" if pairs else "unavailable",
        "host_session": host_id,
        "client_session": client_id,
        "source_overlap_s": overlap,
        "tolerance_s": motion_tolerance,
        "paired_samples": len(pairs),
    }
    if not pairs:
        result["motion_alignment"]["reason"] = "no samples match within motion tolerance"
    result["signals"] = _signal_comparison(pairs)
    result["event_delay"] = (
        _event_delays(host_rows, client_rows, overlap, client_clock_offset, event_window, max_edge_gap)
        if pairs else {"status": "unavailable", "reason": "no shared motion alignment"}
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="one recording to summarize, or the host recording to pair")
    parser.add_argument("client", type=Path, nargs="?", help="client recording for a paired comparison")
    parser.add_argument("--same-flight", action="store_true", help="affirm both files were recorded in the same flight")
    parser.add_argument("--host-session", type=int, help="host file's local session ID")
    parser.add_argument("--client-session", type=int, help="client file's local session ID")
    parser.add_argument("--motion-tolerance", type=float, default=0.08, metavar="SECONDS")
    parser.add_argument("--client-clock-offset", type=float, metavar="SECONDS",
                        help="independently measured number to add to client local time to reach host local time")
    parser.add_argument("--event-window", type=float, default=1.0, metavar="SECONDS",
                        help="maximum source-time difference when pairing matching event edges")
    parser.add_argument("--max-edge-gap", type=float, default=0.5, metavar="SECONDS",
                        help="maximum interval between samples bounding an event edge")
    args = parser.parse_args(argv)

    try:
        primary = load_trace(args.recording)
        if args.client is None:
            result = {"recording": summarize_trace(primary)}
        else:
            result = compare_traces(
                primary, load_trace(args.client), same_flight=args.same_flight, host_session=args.host_session,
                client_session=args.client_session, motion_tolerance=args.motion_tolerance,
                client_clock_offset=args.client_clock_offset, event_window=args.event_window, max_edge_gap=args.max_edge_gap,
            )
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
