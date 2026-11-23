"""CMA-ES search over the observable prior guidance gains in the twin.

Each generation scores every candidate on the same fresh block of training
seeds (common random numbers), all branches, gear crank on. Gains are
multiplicative factors in log space, bounded to +/- ``--bound``.
State is saved after every generation, so a rerun with the same --out resumes.

usage: python -m mayday_landing.tune --out DIR --meta META [--generations 30]
"""
import argparse
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from mayday_jax import replay
from mayday_jax.physics import default_tuning

from .environment import observe, observe_phase, step
from .experiment import (FORK_STEP, engageable_start, TRAIN_SEED_START, RollState, RunConfig, _fork_permanent_loss,
                         _initialize, episode_rows, make_cases)
from .guidance import GAIN_NAMES, guidance
from .taxonomy import cause


@partial(jax.jit, static_argnames=("steps",))
def _advance(world, gains, tuning, *, steps):
    batch_observe = jax.vmap(observe)
    batch_phase = jax.vmap(observe_phase)
    batch_prior = jax.vmap(guidance, in_axes=(0, 0, None, 0))
    batch_step = jax.vmap(step, in_axes=(0, 0, None))

    def body(live, _):
        actions = batch_prior(batch_observe(live), batch_phase(live), tuning, gains)
        return batch_step(live, actions, tuning), None

    return jax.lax.scan(body, world, None, length=steps)[0]


def score(gain_rows, seeds, tuning, *, chunk=100, crank=True, go_around=False, engageable=False):
    """Fly every gain row on every (seed, branch); return episode records per row."""
    gain_rows = np.asarray(gain_rows, np.float32)
    base = make_cases(seeds, packet_stress=True, native_weather=True, gear_crank=crank, gear_go_around=go_around,
                      engageable_starts=engageable)
    n = len(base.seeds)
    repeat = lambda leaf: jnp.tile(leaf, (len(gain_rows),) + (1,) * (jnp.ndim(leaf) - 1))
    scenario = jax.tree.map(repeat, base.scenario)
    keys = repeat(base.keys)
    names = np.tile(base.names, len(gain_rows))
    world = _initialize(keys, scenario, tuning)
    engageable = engageable_start(world)
    gains = jnp.repeat(jnp.asarray(gain_rows), n, axis=0)
    max_steps = RunConfig().max_steps
    start = 0
    while start < max_steps and not bool(jnp.all(world.done)):
        if start == FORK_STEP:
            world = _fork_permanent_loss(world, names)
        end = min(start + chunk, max_steps, FORK_STEP if start < FORK_STEP else max_steps)
        world = _advance(world, gains, tuning, steps=end - start)
        start = end
    size = len(names)
    zeros = jnp.zeros((size, 5), jnp.float32)
    transient = jnp.zeros((size, 4), jnp.float32).at[:, 0].set(-1.0)
    cases = base._replace(seeds=np.tile(base.seeds, len(gain_rows)), names=names)
    rows = episode_rows(RollState(world, jnp.zeros((size, 0)), zeros, transient, transient), cases, engageable)
    return [rows[i * n:(i + 1) * n] for i in range(len(gain_rows))]


def fitness(rows):
    """Success rate plus a small shaped term so near-misses still rank."""
    shaped = []
    for r in rows:
        sink = r["touchdown_sink_mps"] or 0.0
        missed = sum(v != 1 for v in r["ring_verdicts"])
        shaped.append(-0.05 * max(sink - 2.0, 0.0) - 0.04 * missed - 0.2 * (r["termination"] == "crash"))
    return float(np.mean([r["success"] for r in rows]) + np.mean(shaped))


class CMA:
    """Minimal (mu/mu_w, lambda) CMA-ES maximizing fitness (Hansen 2016 defaults)."""

    def __init__(self, mean, sigma, popsize, seed):
        self.n = n = len(mean)
        self.lam = popsize
        self.mu = popsize // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.mean, self.sigma = np.asarray(mean, float), float(sigma)
        self.C, self.pc, self.ps = np.eye(n), np.zeros(n), np.zeros(n)
        self.rng = np.random.default_rng(seed)
        self.generation = 0

    def ask(self):
        values, vectors = np.linalg.eigh(self.C)
        self._B, self._D = vectors, np.sqrt(np.maximum(values, 1e-20))
        z = self.rng.standard_normal((self.lam, self.n))
        return self.mean + self.sigma * (z * self._D) @ self._B.T

    def tell(self, samples, scores):
        order = np.argsort(-np.asarray(scores))[:self.mu]
        chosen = samples[order]
        old = self.mean
        self.mean = self.w @ chosen
        y = (self.mean - old) / self.sigma
        invsqrt = self._B @ np.diag(1 / self._D) @ self._B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrt @ y
        self.generation += 1
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.generation)) < (1.4 + 2 / (self.n + 1)) * self.chin
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        steps = (chosen - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc)
                  + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * (steps.T * self.w) @ steps)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))

    def state(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in
                (("mean", self.mean), ("sigma", self.sigma), ("C", self.C), ("pc", self.pc), ("ps", self.ps),
                 ("generation", self.generation))} | {"rng": self.rng.bit_generator.state}

    def load(self, saved):
        for name in ("mean", "C", "pc", "ps"):
            setattr(self, name, np.asarray(saved[name], float))
        self.sigma, self.generation = float(saved["sigma"]), int(saved["generation"])
        self.rng.bit_generator.state = saved["rng"]


def evaluate_holdout(args, tuning, batch=64):
    """Fly default and tuned gains on the same fresh seeds; write an experiment-shaped report."""
    tuned = json.loads(args.evaluate.read_text())
    known = dict(zip(tuned["gain_names"], tuned["gains"]))
    if set(known) - set(GAIN_NAMES):
        raise ValueError("tuned gains use a different guidance parameterization")
    rows = np.stack((np.ones(len(GAIN_NAMES)), [known.get(name, 1.0) for name in GAIN_NAMES]))
    episodes = {"prior": [], "tuned": []}
    for begin in range(args.seed_start, args.seed_start + args.seeds, batch):
        seeds = np.arange(begin, min(begin + batch, args.seed_start + args.seeds))
        default, candidate = score(rows, seeds, tuning, engageable=args.engageable_starts)
        episodes["prior"] += default
        episodes["tuned"] += candidate
        print(f"evaluated {begin + len(seeds) - args.seed_start} seeds: default "
              f"{np.mean([e['success'] for e in episodes['prior']]):.4f} tuned "
              f"{np.mean([e['success'] for e in episodes['tuned']]):.4f}", flush=True)
    report = {"schema": "mayday-landing-tune-eval-v1", "tuned": tuned, "seed_start": args.seed_start,
              "seeds": args.seeds, "gear_crank": True,
              "evaluations": {name: {"episodes": eps} for name, eps in episodes.items()}}
    (args.out / "tune-eval.json").write_text(json.dumps(report))
    print("EVAL DONE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--seeds-per-generation", type=int, default=48)
    parser.add_argument("--sigma", type=float, default=0.12)
    parser.add_argument("--bound", type=float, default=0.45)
    parser.add_argument("--seed-start", type=int, default=TRAIN_SEED_START)
    parser.add_argument("--init", type=Path, help="tuned.json to warm-start from; unknown gains start at default")
    parser.add_argument("--freeze", default="", help="comma-separated gain names held at their default")
    parser.add_argument("--evaluate", type=Path, help="tuned.json: paired default-vs-tuned holdout evaluation")
    parser.add_argument("--seeds", type=int, default=512, help="holdout seeds for --evaluate")
    parser.add_argument("--engageable-starts", action="store_true")
    args = parser.parse_args()
    tuning = replay.load_tuning(args.meta) if args.meta else default_tuning()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.evaluate:
        evaluate_holdout(args, tuning)
        return
    frozen = np.array([name in args.freeze.split(",") for name in GAIN_NAMES])
    if args.freeze and not frozen.any():
        raise ValueError("no gain matched --freeze")
    state_path, log_path = args.out / "cma.json", args.out / "generations.jsonl"
    start = np.zeros(len(GAIN_NAMES))
    if args.init:
        previous = json.loads(args.init.read_text())
        known = dict(zip(previous["gain_names"], previous["log_gains"]))
        start = np.array([known.get(name, 0.0) for name in GAIN_NAMES])
    cma = CMA(start, args.sigma, args.population, seed=1729)
    if state_path.is_file():
        cma.load(json.loads(state_path.read_text()))
        print("resumed at generation", cma.generation, flush=True)
    while cma.generation < args.generations:
        began = perf_counter()
        g = cma.generation
        seeds = np.arange(args.seed_start + g * args.seeds_per_generation,
                          args.seed_start + (g + 1) * args.seeds_per_generation)
        samples = np.where(frozen, 0.0, np.clip(cma.ask(), -args.bound, args.bound))
        rows = np.vstack((start, np.clip(cma.mean, -args.bound, args.bound), samples))
        results = score(np.exp(rows), seeds, tuning)
        scores = [fitness(r) for r in results]
        success = [float(np.mean([e["success"] for e in r])) for r in results]
        cma.tell(samples, scores[2:])
        record = {"generation": g, "seconds": perf_counter() - began, "seeds": [int(seeds[0]), int(seeds[-1])],
                  "default": {"fitness": scores[0], "success": success[0]},
                  "mean": {"fitness": scores[1], "success": success[1], "log_gains": rows[1].tolist()},
                  "best_sample": {"fitness": max(scores[2:]), "success": success[2 + int(np.argmax(scores[2:]))]},
                  "sigma": cma.sigma,
                  "default_causes": {c: sum(cause(e) == c for e in results[0]) for c in {cause(e) for e in results[0]}},
                  "mean_causes": {c: sum(cause(e) == c for e in results[1]) for c in {cause(e) for e in results[1]}}}
        with log_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        state_path.write_text(json.dumps(cma.state()))
        print(f"gen {g}: default {success[0]:.3f} mean {success[1]:.3f} best {record['best_sample']['success']:.3f}"
              f" sigma {cma.sigma:.3f} ({record['seconds']:.0f}s)", flush=True)
    log_gains = np.where(frozen, 0.0, np.clip(cma.mean, -args.bound, args.bound))
    final = {"gain_names": list(GAIN_NAMES), "log_gains": log_gains.tolist(), "gains": np.exp(log_gains).tolist(),
             "generations": cma.generation, "frozen": [n for n, f in zip(GAIN_NAMES, frozen) if f]}
    (args.out / "tuned.json").write_text(json.dumps(final, indent=2))
    print("TUNE DONE", json.dumps(final["gains"]), flush=True)


if __name__ == "__main__":
    main()
