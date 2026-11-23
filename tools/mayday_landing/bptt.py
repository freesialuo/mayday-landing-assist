# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""P4: truncated BPTT fine-tuning of a residual MLP through the differentiable twin near the ground.

1. Fly the observable prior on training seeds and snapshot every episode the
   first time it descends through ``--start-agl`` on final (not in a circuit).
2. From batches of snapshots, unroll ``--horizon`` decisions with the residual
   MLP (deployed architecture, deployed residual limits) and descend a smooth
   flare-envelope loss by its pathwise gradient. Each unroll starts from a
   stop-gradient snapshot (truncated BPTT, cf. SHAC / rpg_flightning).
3. Write the fine-tuned policy as a residual artifact for --evaluate-model.

The loss weights the last metres heavily: sink above a height-dependent
ceiling, sink below a floor near the ground (no floating), bank beyond a
guard, lateral offset, and the first-contact sink and bank. Policies see only
the delayed observation; the gradient flows through the true physics.

usage: python -m mayday_landing.bptt --init mlp.json --out DIR --meta META [--steps 400]
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mayday_jax.physics import default_tuning
from mayday_jax.replay import load_tuning

from . import model
from .environment import FLARE, FINAL, _runway_motion, observe, observe_phase, step
from .experiment import (FORK_STEP, TRAIN_SEED_START, RunConfig, _fork_permanent_loss, _initialize,
                         engageable_start, make_cases)
from .guidance import guidance
from .residual_model import apply_residual_actions, unwrap_residual, wrap_residual


def _student(weights, mean, std, observation, phase):
    encoded = model._encode(weights, (observation - mean) / std, phase, jnp)
    latent = jnp.tanh(model._dense(weights["hidden"], encoded))
    return model._actions(weights, latent, phase, jnp)


def collect_starts(seeds, tuning, start_agl, chunk=100, gust_kick=False):
    """Prior rollouts; return one snapshot per episode that reached start_agl on final."""
    cases = make_cases(seeds, packet_stress=True, native_weather=True, gear_crank=True, engageable_starts=True,
                       gust_kick=gust_kick)
    world = _initialize(cases.keys, cases.scenario, tuning)
    taken = jnp.zeros(len(cases.seeds), bool)
    snapshot = world

    @jax.jit
    def advance(world, taken, snapshot):
        def body(carry, _):
            live, taken, snapshot = carry
            obs, phase = jax.vmap(observe)(live), jax.vmap(observe_phase)(live)
            live = jax.vmap(step, in_axes=(0, 0, None))(live, jax.vmap(guidance, in_axes=(0, 0, None))(obs, phase, tuning), tuning)
            ready = (~taken & ~live.done & (live.aircraft.agl <= start_agl) & ~live.aircraft.on_ground
                     & ((live.phase == FINAL) | (live.phase == FLARE)) & (live.touchdown_count == 0))
            snapshot = jax.tree.map(lambda new, old: jnp.where(ready.reshape(ready.shape + (1,) * (new.ndim - 1)), new, old),
                                    live, snapshot)
            return (live, taken | ready, snapshot), None
        return jax.lax.scan(body, (world, taken, snapshot), None, length=chunk)[0]

    start = 0
    while start < RunConfig().max_steps and not bool(jnp.all(taken | world.done)):
        if start == FORK_STEP:
            world = _fork_permanent_loss(world, cases.names)
        world, taken, snapshot = advance(world, taken, snapshot)
        start += chunk
    keep = np.asarray(taken)
    return jax.tree.map(lambda leaf: leaf[keep], snapshot), cases.seeds[keep], cases.names[keep]


def episode_loss(params, mean, std, start, tuning, horizon):
    """Smooth flare-envelope cost of one unroll from a stop-gradient snapshot."""
    start = jax.lax.stop_gradient(start)

    def body(carry, _):
        live, total = carry
        obs, phase = observe(live), observe_phase(live)
        prior = guidance(obs, phase, tuning)
        action = apply_residual_actions(_student(params, mean, std, obs, phase), prior, phase, xp=jnp)
        following = jax.checkpoint(step)(live, action, tuning)
        plane = following.aircraft
        sink = _runway_motion(plane, following.flight.runway_heading)[3]
        airborne = (~plane.on_ground).astype(jnp.float32)
        active = (~live.done).astype(jnp.float32)
        near = jax.nn.sigmoid((8.0 - plane.agl) / 1.5) * airborne
        ground_close = jax.nn.sigmoid((3.0 - plane.agl) / 0.6) * airborne
        ceiling = 1.6 + 0.35 * plane.agl
        cost = near * (jax.nn.relu(sink - ceiling) ** 2 + 0.5 * (jax.nn.relu(jnp.abs(plane.roll) - 4.0) / 4.0) ** 2
                       + 0.2 * (following.lateral / 12.0) ** 2)
        cost += ground_close * 2.0 * jax.nn.relu(0.9 - sink) ** 2          # no floating in ground effect
        contact = (following.touchdown_count > live.touchdown_count).astype(jnp.float32)
        cost += contact * (8.0 * jax.nn.relu(following.flight.touchdown_vertical_speed - 1.8) ** 2
                           + 4.0 * (jax.nn.relu(jnp.abs(plane.roll) - 3.0) / 3.0) ** 2)
        cost += 1e-3 * jnp.sum((action[:4] - prior[:4]) ** 2)
        return (following, total + active * cost), None

    (_, total), _ = jax.lax.scan(body, (start, jnp.float32(0.0)), None, length=horizon)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True, help="residual MLP artifact to fine-tune")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--train-seeds", type=int, default=512)
    parser.add_argument("--seed-start", type=int, default=TRAIN_SEED_START + 20_000)
    parser.add_argument("--start-agl", type=float, default=20.0)
    parser.add_argument("--horizon", type=int, default=70)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--anchor", type=float, default=1e-2, help="L2 pull toward the initial weights")
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--validation-seeds", type=int, default=32)
    parser.add_argument("--gust-kick", action="store_true")
    parser.add_argument("--zero-head", action="store_true", help="start from a zero residual (pure prior)")
    args = parser.parse_args()
    tuning = load_tuning(args.meta) if args.meta else default_tuning()
    args.out.mkdir(parents=True, exist_ok=True)
    artifact = json.loads(args.init.read_text())
    learner = unwrap_residual(artifact)
    policy = model.Policy(learner)
    if policy.architecture != "mlp":
        raise ValueError("BPTT fine-tuning supports the residual MLP")
    params = jax.tree.map(jnp.asarray, policy.weights)
    if args.zero_head:
        for head in ("air", "ground"):
            params[head] = jax.tree.map(jnp.zeros_like, params[head])
    initial = params
    mean, std = jnp.asarray(policy.mean), jnp.asarray(policy.std)

    began = perf_counter()
    seeds = np.arange(args.seed_start, args.seed_start + args.train_seeds)
    starts, start_seeds, start_names = collect_starts(seeds, tuning, args.start_agl, gust_kick=args.gust_kick)
    count = len(start_seeds)
    validation_seeds = np.arange(args.seed_start + 100_000, args.seed_start + 100_000 + args.validation_seeds)
    validation, _, _ = collect_starts(validation_seeds, tuning, args.start_agl, gust_kick=args.gust_kick)
    print(f"collected {count} flare starts from {len(seeds)} seeds in {perf_counter() - began:.0f}s", flush=True)

    def batch_loss(params, batch):
        losses = jax.vmap(episode_loss, in_axes=(None, None, None, 0, None, None))(
            params, mean, std, batch, tuning, args.horizon)
        anchor = sum(jnp.sum((a - b) ** 2) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(initial)))
        return losses.mean() + args.anchor * anchor, losses.mean()

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.learning_rate))
    state = optimizer.init(params)

    @jax.jit
    def update(params, state, batch):
        (_, loss), grads = jax.value_and_grad(batch_loss, has_aux=True)(params, batch)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss, optax.global_norm(grads)

    @jax.jit
    def validate(params):
        return jax.vmap(episode_loss, in_axes=(None, None, None, 0, None, None))(
            params, mean, std, validation, tuning, args.horizon).mean()

    best_loss, best_step, best = float(validate(params)), 0, params
    print(f"validation loss at start {best_loss:.4f}", flush=True)
    rng = np.random.default_rng(7)
    history = []
    for index in range(args.steps):
        pick = rng.choice(count, size=min(args.batch, count), replace=False)
        batch = jax.tree.map(lambda leaf: leaf[pick], starts)
        params, state, loss, norm = update(params, state, batch)
        if index == 0 or (index + 1) % min(20, args.validate_every) == 0 or (index + 1) % args.validate_every == 0:
            held_out = float(validate(params)) if (index + 1) % args.validate_every == 0 or index == 0 else None
            if held_out is not None and held_out < best_loss:
                best_loss, best_step, best = held_out, index + 1, params
            history.append({"step": index + 1, "loss": float(loss), "grad_norm": float(norm), "validation": held_out})
            print(f"step {index + 1}: loss {float(loss):.4f} grad {float(norm):.3g}"
                  + (f" val {held_out:.4f}" if held_out is not None else ""), flush=True)
    print(f"best validation {best_loss:.4f} at step {best_step}", flush=True)
    learner["weights"] = jax.tree.map(lambda item: np.asarray(item).tolist(), best)
    learner.setdefault("training", {})["bptt"] = {
        "init": args.init.name, "starts": count, "seed_range": [int(seeds[0]), int(seeds[-1])],
        "start_agl": args.start_agl, "horizon": args.horizon, "batch": args.batch, "steps": args.steps,
        "learning_rate": args.learning_rate, "anchor": args.anchor, "zero_head": args.zero_head, "loss": history,
        "best_validation": best_loss, "best_step": best_step}
    learner["qualification"] = "offline_experimental"
    out = wrap_residual(learner, stall_speeds=tuning.stall_speeds)
    (args.out / "mlp.json").write_text(json.dumps(out))
    print("BPTT DONE", args.out / "mlp.json", flush=True)


if __name__ == "__main__":
    main()
