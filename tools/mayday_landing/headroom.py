# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Can a privileged MPPI teacher solve the episodes the observable prior fails?

Reads an evaluation report, picks every seed with a supported-band (0-300 ms)
prior failure, and reflies those seeds (all branches, gear crank on) with the
prior and with MPPI teachers of increasing search budget. Writes headroom.json.

usage: python -m mayday_landing.headroom REPORT --out DIR --meta META [--planners 7x8,31x16,31x16b]  (suffix b = blind to future pulses)
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from mayday_jax.physics import default_tuning
from mayday_jax.replay import load_tuning

from .experiment import PlannerConfig, RunConfig, fly
from .taxonomy import SUPPORTED_LAGS, cause


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--controller", default="prior")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--planners", default="7x8,31x16")
    parser.add_argument("--batch-seeds", type=int, default=16)
    parser.add_argument("--gust-kick", action="store_true")
    parser.add_argument("--engageable-starts", action="store_true")
    parser.add_argument("--max-seeds", type=int, default=0, help="cap the failure seeds (0 = all)")
    args = parser.parse_args()
    tuning = load_tuning(args.meta) if args.meta else default_tuning()
    episodes = json.loads(args.report.read_text())["evaluations"][args.controller]["episodes"]
    failed = {(e["seed"], e["branch"]): cause(e) for e in episodes
              if e["observation_lag_steps"] in SUPPORTED_LAGS and not e["success"]}
    seeds = sorted({seed for seed, _ in failed})
    if args.max_seeds:
        seeds = seeds[:args.max_seeds]
        failed = {key: value for key, value in failed.items() if key[0] in set(seeds)}
    print(f"{len(failed)} failed episodes on {len(seeds)} seeds", Counter(failed.values()), flush=True)
    config = RunConfig(batch_seeds=args.batch_seeds)
    common = dict(packet_stress=True, native_weather=True, gear_crank=True, gust_kick=args.gust_kick,
                  engageable_starts=args.engageable_starts, progress=lambda line: print(line, flush=True))
    result = {"source": str(args.report), "failed": len(failed), "seeds": seeds,
              "failed_causes": dict(Counter(failed.values())), "runs": {}}
    runs = [("prior", "prior", PlannerConfig())]
    for spec in args.planners.split(","):
        # Suffixes: b = blind to future pulses, r = bounded residual teacher around the observable prior.
        blind, residual = "b" in spec, "r" in spec
        samples, horizon = (int(v) for v in spec.rstrip("br").split("x"))
        runs.append((f"teacher-{spec}", "teacher", PlannerConfig(samples=samples, horizon=horizon,
                                                                 clairvoyant=not blind), residual))
    for name, actor, planner, *flags in runs:
        began = perf_counter()
        rows, _ = fly(seeds, actor, tuning, planner, config, residual=bool(flags and flags[0]), **common)
        target = [r for r in rows if (r["seed"], r["branch"]) in failed]
        others = [r for r in rows if (r["seed"], r["branch"]) not in failed and r["observation_lag_steps"] in SUPPORTED_LAGS]
        result["runs"][name] = {
            "seconds": perf_counter() - began,
            "solved_previous_failures": sum(r["success"] for r in target), "previous_failures": len(target),
            "causes_on_previous_failures": dict(Counter(cause(r) for r in target)),
            "other_supported_success": float(np.mean([r["success"] for r in others])) if others else None,
            "episodes": rows,
        }
        print(name, "solved", result["runs"][name]["solved_previous_failures"], "/", len(target),
              result["runs"][name]["causes_on_previous_failures"], flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "headroom.json").write_text(json.dumps(result))
    print("HEADROOM DONE", flush=True)


if __name__ == "__main__":
    main()
