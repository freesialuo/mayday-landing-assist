"""Failure taxonomy and seed-bootstrap confidence intervals for experiment reports.

Branches of one seed share an initial state, so resampling is by seed and every
episode of a drawn seed comes along. Lag cohorts follow ``seed % 4``.

usage: python -m mayday_landing.taxonomy report.json [other.json ...] [--json out.json]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

SUPPORTED_LAGS = (0, 1, 3)
MAX_SINK = 3.0


def cause(row):
    """One primary label per episode, checked in causal order."""
    if row["success"]:
        return "success"
    term = row["termination"]
    if term == "crash":
        if row["gear_extension_final"] <= 0.9:
            return "crash_gear_unlocked"
        if row.get("touchdown_roll_deg") is not None and abs(row["touchdown_roll_deg"]) > 8.0:
            # Native OnTouchdown: bank above WingStrikeRoll (9.06 deg) up to 30 deg raises a
            # wing-strike damage hazard; only > 30 deg or > 8 m/s sink is a native crash.
            return "wing_strike_damage"
        return "crash_other"
    if term == "handback":
        return "handback_gear_unlocked"
    if term in ("off_runway", "timeout", "left_area"):
        return term
    if not all(v == 1 for v in row["ring_verdicts"]):
        return "ring_missed"
    if row["touchdown_sink_mps"] is not None and row["touchdown_sink_mps"] > MAX_SINK:
        return "hard_touchdown"
    if row["branch"] == "go_around" and not row["go_around_reacquired"]:
        return "go_around_not_reacquired"
    return "other_criteria"


def native_crash(row):
    """PlaneFlight.OnTouchdown crash rule: sink > 8 m/s or |bank| > 30 deg at touchdown."""
    return bool(row["touchdown_count"]) and ((row["touchdown_sink_mps"] or 0.0) > 8.0
                                            or abs(row["touchdown_roll_deg"] or 0.0) > 30.0)


def _by_seed(rows):
    seeds = sorted({r["seed"] for r in rows})
    index = {s: i for i, s in enumerate(seeds)}
    wins = np.zeros(len(seeds))
    counts = np.zeros(len(seeds))
    for r in rows:
        wins[index[r["seed"]]] += r["success"]
        counts[index[r["seed"]]] += 1
    return seeds, wins, counts


def bootstrap(rows, other=None, draws=4000, seed=0):
    """Return rate (or paired rate difference) with a 95% seed-bootstrap interval."""
    rng = np.random.default_rng(seed)
    seeds, wins, counts = _by_seed(rows)
    if other is not None:
        other_seeds, other_wins, other_counts = _by_seed(other)
        if other_seeds != seeds or not np.array_equal(counts, other_counts):
            raise ValueError("paired reports need identical seeds and branches")
        wins = wins - other_wins
    picks = rng.integers(0, len(seeds), (draws, len(seeds)))
    stats = wins[picks].sum(1) / counts[picks].sum(1)
    point = wins.sum() / counts.sum()
    low, high = np.percentile(stats, (2.5, 97.5))
    return {"value": float(point), "ci95": [float(low), float(high)], "seeds": len(seeds),
            "episodes": int(counts.sum()), "p_greater_than_zero": float((stats > 0).mean())}


def summarize(rows):
    supported = [r for r in rows if r["observation_lag_steps"] in SUPPORTED_LAGS]
    stress = [r for r in rows if r["observation_lag_steps"] not in SUPPORTED_LAGS]
    out = {"all": bootstrap(rows), "supported": bootstrap(supported) if supported else None,
           "stress_600ms": bootstrap(stress) if stress else None,
           "causes_supported": dict(Counter(cause(r) for r in supported).most_common()),
           "causes_stress": dict(Counter(cause(r) for r in stress).most_common()),
           "native_crashes_supported": sum(native_crash(r) for r in supported)}
    engageable = [r for r in supported if r.get("engageable_start")]
    if engageable:
        # Starts the mod would accept (LandingDirector CaptureWindow); others it refuses to engage.
        out["supported_engageable"] = bootstrap(engageable)
        out["causes_supported_engageable"] = dict(Counter(cause(r) for r in engageable).most_common())
    for key in ("weather_family", "branch"):
        groups = sorted({r[key] for r in supported})
        out["supported_by_" + key] = {
            g: {"rate": float(np.mean([r["success"] for r in supported if r[key] == g])),
                "causes": dict(Counter(cause(r) for r in supported if r[key] == g).most_common(3))}
            for g in groups}
    return out


def analyse(report):
    evaluations = report["evaluations"]
    result = {name: summarize(ev["episodes"]) for name, ev in evaluations.items()}
    if "prior" in evaluations:
        base = evaluations["prior"]["episodes"]
        for name, ev in evaluations.items():
            if name == "prior":
                continue
            pick = lambda rows: [r for r in rows if r["observation_lag_steps"] in SUPPORTED_LAGS]
            result[name]["minus_prior_supported"] = bootstrap(pick(ev["episodes"]), pick(base))
            result[name]["minus_prior_all"] = bootstrap(ev["episodes"], base)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    combined = {}
    for path in args.reports:
        combined[str(path)] = analyse(json.loads(path.read_text()))
        print(f"== {path}")
        for name, s in combined[str(path)].items():
            sup = s["supported"]
            line = f"  {name:8s} supported {sup['value']:.3f} [{sup['ci95'][0]:.3f}, {sup['ci95'][1]:.3f}]"
            if s["stress_600ms"]:
                line += f"  600ms {s['stress_600ms']['value']:.3f}"
            if "minus_prior_supported" in s:
                d = s["minus_prior_supported"]
                line += f"  Δprior {d['value']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] P>0={d['p_greater_than_zero']:.2f}"
            print(line)
            print("           causes", s["causes_supported"], "native crashes", s["native_crashes_supported"])
            if "supported_engageable" in s:
                e = s["supported_engageable"]
                print(f"           engageable {e['value']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}] n={e['episodes']}",
                      s["causes_supported_engageable"])
    if args.json:
        args.json.write_text(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
