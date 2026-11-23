"""Summarize probe recordings (mayday-landing.probe/1) from in-game tests.

Fidelity: the measured bank kick after each native front gust (the twin uses ±12°),
native touchdowns and hazards, and the observation age the policy saw.
Model feedback: policy vs observable prior vs pilot, and windows around Shift+F6 marks.

usage: python -m mayday_landing.probe_report REC.jsonl [more.jsonl ...] [--json out.json]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

KICK_WINDOW = 1.0
MARK_WINDOW = 5.0
AXES = ("pitch", "roll", "yaw", "throttle")


def _rows(path):
    with open(path, encoding="utf-8") as stream:
        header = json.loads(stream.readline())
        if header.get("probe") is None:
            raise ValueError(f"{path} was recorded without the probe")
        names = header["features"]
        rows = [json.loads(line) for line in stream if line.strip()]
    return header, names, rows


def analyse(path):
    header, names, rows = _rows(path)
    index = {name: i for i, name in enumerate(names)}
    feature = lambda row, name: (row.get("observation") or [None] * len(names))[index[name]]
    times = np.array([row["time"] if row.get("time") is not None else np.nan for row in rows])
    rolls = np.array([feature(row, "roll") if feature(row, "roll") is not None else np.nan for row in rows])
    events = [(i, event) for i, row in enumerate(rows) for event in (row.get("probe") or {}).get("events") or []]

    kicks = []
    for i, event in events:
        if event["type"] != "gust" or event.get("onGround"):
            continue
        start = event.get("time") or times[i]
        window = (times >= start) & (times <= start + KICK_WINDOW)
        if not window.any():
            continue
        # The gust frame already carries the kick; the native step is the change from the previous row.
        before = rolls[max(i - 1, 0)]
        step = float(rolls[i] - before)
        deviation = rolls[window] - before
        peak = deviation[np.nanargmax(np.abs(deviation))]
        kicks.append({"time": start, "dir": event.get("dir"), "agl": event.get("agl"), "roll_before": float(before),
                      "step_deg": step, "peak_change_deg": float(peak),
                      "sign_matches_dir": bool(np.sign(step) == np.sign(event.get("dir") or 0))})

    touchdowns = [event | {"row": i} for i, event in events if event["type"] in ("touchdown", "contact")]
    if not touchdowns:
        # Older recordings: derive contacts from the frames (last airborne sink and bank).
        last = None
        for i, row in enumerate(rows):
            if not row.get("valid") or row.get("observation") is None:
                continue
            ground = feature(row, "on_ground") >= 0.5
            if last is not None and ground and feature(rows[last], "on_ground") < 0.5:
                sink, roll = feature(rows[last], "sink"), feature(rows[last], "roll")
                touchdowns.append({"type": "contact", "time": row["time"], "row": i, "sink": sink, "roll": roll,
                                   "gear": feature(rows[last], "gear"), "derived": True,
                                   "nativeCrash": sink > 8 or abs(roll) > 30,
                                   "nativeHardLanding": sink >= 2.5 or feature(rows[last], "gear") < 0.9,
                                   "wingStrike": abs(roll) > 9.06})
            last = i
    hazards = Counter(event.get("kind") for _, event in events if event["type"] == "hazard")
    ages = np.array([feature(row, "observation_age") for row in rows if row.get("valid")], dtype=float)

    gaps_prior, gaps_pilot = [], []
    for row in rows:
        probe = row.get("probe") or {}
        proposed, prior = row.get("proposed"), probe.get("prior")
        if proposed and prior and None not in proposed[:4] and None not in prior[:4]:
            gaps_prior.append(np.abs(np.subtract(proposed[:4], prior[:4])))
        pilot = row.get("pilot")
        if proposed and pilot and not row.get("authorized") and None not in proposed[:3] and None not in pilot[:3]:
            gaps_pilot.append(np.abs(np.subtract(proposed[:3], pilot[:3])))

    marks = []
    for i, event in events:
        if event["type"] != "mark":
            continue
        start = event.get("time") or times[i]
        window = [r for r, t in zip(rows, times) if abs(t - start) <= MARK_WINDOW]
        marks.append({"time": start, "rows": len(window),
                      "statuses": sorted({r["status"] for r in window}),
                      "authorized_fraction": float(np.mean([bool(r.get("authorized")) for r in window])) if window else None})

    mean = lambda values, axes: dict(zip(axes, np.mean(values, axis=0).round(4).tolist())) if values else None
    return {
        "file": str(path), "policySha256": header.get("policySha256"), "rows": len(rows),
        "seconds": float(np.nanmax(times) - np.nanmin(times)) if len(rows) else 0.0,
        "observation_age": {"p50": float(np.percentile(ages, 50)), "p95": float(np.percentile(ages, 95)),
                            "max": float(ages.max())} if ages.size else None,
        "gust_kicks": kicks,
        "gust_kick_summary": {"count": len(kicks),
                              "mean_abs_step_deg": float(np.mean([abs(k["step_deg"]) for k in kicks])) if kicks else None,
                              "sign_matches": sum(k["sign_matches_dir"] for k in kicks),
                              "twin_model_deg": 12.0},
        "touchdowns": touchdowns, "hazards": dict(hazards),
        "policy_minus_prior": mean(gaps_prior, AXES), "policy_minus_pilot": mean(gaps_pilot, AXES[:3]),
        "marks": marks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results = [analyse(path) for path in args.recordings]
    for r in results:
        k = r["gust_kick_summary"]
        print(f"== {r['file']}  {r['seconds']:.0f}s  age {r['observation_age']}")
        print(f"   gust kicks {k['count']} (mean |step| {k['mean_abs_step_deg']}, sign ok {k['sign_matches']}, twin {k['twin_model_deg']})"
              f"  touchdowns {[(round(t.get('vs', t.get('sink', 0)), 2), round(t['roll'], 1), t.get('nativeHardLanding')) for t in r['touchdowns']]}")
        print(f"   hazards {r['hazards']}  policy-prior {r['policy_minus_prior']}  policy-pilot {r['policy_minus_pilot']}"
              f"  marks {len(r['marks'])}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
