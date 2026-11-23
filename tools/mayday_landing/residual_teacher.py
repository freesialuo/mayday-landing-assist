"""Bounded MPPI labels around the same observable prior the student will use."""

import jax
import jax.numpy as jnp

from .environment import observe, observe_phase, step
from .guidance import guidance
from .residual_model import RESIDUAL_LIMITS
from .teacher import _state_cost, blind_disturbances, causal_state, clip_action


def _rollout_cost(state, residuals, tuning):
    def predict(current, residual):
        base = guidance(observe(current), observe_phase(current), tuning)
        action = clip_action(base + residual)
        following = step(current, action, tuning)
        return following, _state_cost(current, following, action, residual)

    _, costs = jax.lax.scan(predict, state, residuals)
    return costs.sum()


def residual_plan(state, base, key, tuning, config):
    """One path integral update with the declared deployment control authority."""
    prediction = causal_state(state)
    # Preserve packets already sent. Score future decisions under nominal
    # transport delay since future jitter and loss are unavailable now.
    prediction = prediction._replace(scenario=prediction.scenario._replace(
        observation_jitter=0, command_jitter=0, observation_loss=0.0, command_loss=0.0))
    if not config.clairvoyant:
        prediction = blind_disturbances(prediction)
    limits = jnp.asarray(RESIDUAL_LIMITS, jnp.float32)
    shape = ((config.samples - 1) // 2, config.horizon, 5)
    mean = jnp.zeros((config.horizon, 5), jnp.float32)
    # Iteration 0 reproduces the single-pass teacher exactly; later passes
    # (anneal > 1) re-centre on the weighted mean with halved noise.
    for iteration in range(config.anneal):
        draw = jax.random.normal(jax.random.fold_in(key, iteration) if iteration else key, shape, jnp.float32)
        noise = jnp.clip(0.75 * draw[:, :1] + 0.25 * draw, -1.0, 1.0) * limits * (0.5 ** iteration)
        candidates = jnp.clip(mean + jnp.concatenate((jnp.zeros((1, config.horizon, 5), jnp.float32), noise, -noise)),
                              -limits, limits)
        costs = jax.vmap(_rollout_cost, in_axes=(None, 0, None))(prediction, candidates, tuning)
        best = jnp.min(costs)
        scale = jnp.maximum(0.4, (jnp.mean(costs) - best) * config.temperature)
        weight = jax.nn.softmax(-(costs - best) / scale)
        mean = jnp.tensordot(weight, candidates, axes=1)
    return clip_action(base + mean[0])
