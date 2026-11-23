# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Portable bounded corrections to observable landing guidance."""

import json
from pathlib import Path

import numpy as np

from .features import NAMES
from .model import ACTION_NAMES, PHASE_NAMES, ROLLOUT, Policy, _float_array, _validate_phases


SCHEMA = "mayday-landing-residual-v1"
PRIOR = "mayday-landing-observable-guidance-v1"
RESIDUAL_LIMITS = (0.10, 0.15, 0.12, 0.12, 0.0)


def encode_residual_actions(actions, priors, xp=np):
    """Encode the reachable action correction as labels for a direct learner."""
    actions, priors = xp.asarray(actions, dtype=xp.float32), xp.asarray(priors, dtype=xp.float32)
    limits = xp.asarray(RESIDUAL_LIMITS[:4], dtype=actions.dtype)
    residual = xp.clip((actions[..., :4] - priors[..., :4]) / limits, -1.0, 1.0)
    return xp.concatenate((residual[..., :3], 0.5 + 0.5 * residual[..., 3:4],
                           xp.zeros_like(residual[..., 3:4])), axis=-1)


def apply_residual_actions(predictions, priors, phases, xp=np):
    """Apply four bounded corrections and keep the prior's discrete ground brake."""
    predictions = xp.asarray(predictions, dtype=xp.float32)
    priors, phases = xp.asarray(priors, dtype=xp.float32), xp.asarray(phases)
    limits = xp.asarray(RESIDUAL_LIMITS[:4], dtype=predictions.dtype)
    residual = xp.concatenate((predictions[..., :3], 2.0 * predictions[..., 3:4] - 1.0), axis=-1)
    corrected = priors[..., :4] + xp.clip(residual, -1.0, 1.0) * limits
    brake = (phases == ROLLOUT)[..., None] & (priors[..., 4:5] >= 0.5)
    brake = xp.broadcast_to(brake, corrected[..., 3:4].shape).astype(predictions.dtype)
    return xp.concatenate((xp.clip(corrected[..., :3], -1.0, 1.0),
                           xp.clip(corrected[..., 3:4], 0.0, 1.0), brake), axis=-1)


def _stall_speeds(values):
    speeds = np.asarray(values)
    if speeds.shape != (3,) or speeds.dtype.kind not in "fiu":
        raise ValueError("residual policy needs three finite positive stall speeds")
    speeds = speeds.astype(np.float32)
    if not np.isfinite(speeds).all() or np.any(speeds <= 0):
        raise ValueError("residual policy needs three finite positive stall speeds")
    return speeds.tolist()


def wrap_prior(*, stall_speeds):
    """Export the observable rule-based comparator with explicit tuning."""
    return {"schema": PRIOR, "mode": "prior", "stall_speeds": _stall_speeds(stall_speeds),
            "feature_names": list(NAMES), "actions": list(ACTION_NAMES), "phases": list(PHASE_NAMES)}


def _check_learner(payload):
    learner = Policy(payload)
    if learner.feature_names != tuple(NAMES):
        raise ValueError("residual learner feature schema must match observable guidance")
    return learner


def wrap_residual(payload, *, stall_speeds):
    """Tie a trained learner to this prior and correction contract."""
    _check_learner(payload)
    return {
        "schema": SCHEMA,
        "mode": "residual",
        "prior": PRIOR,
        "residual_limits": list(RESIDUAL_LIMITS),
        "stall_speeds": _stall_speeds(stall_speeds),
        "learner": payload,
    }


def unwrap_residual(payload):
    """Validate the residual contract and return the nested direct learner."""
    if (payload.get("schema") != SCHEMA or payload.get("mode") != "residual" or payload.get("prior") != PRIOR
            or payload.get("residual_limits") != list(RESIDUAL_LIMITS)):
        raise ValueError("invalid landing residual policy schema")
    _stall_speeds(payload.get("stall_speeds"))
    learner = payload["learner"]
    _check_learner(learner)
    return learner


class GuidedPolicy:
    """Stateful replay for a tuned prior or a learned correction to that prior."""

    feature_names = tuple(NAMES)

    def __init__(self, payload):
        import jax
        import jax.numpy as jnp
        from mayday_jax.physics import default_tuning
        from .guidance import guidance

        self.mode = payload.get("mode")
        speeds = _stall_speeds(payload.get("stall_speeds"))
        if self.mode == "residual":
            self.learner = Policy(unwrap_residual(payload))
        elif (self.mode == "prior" and payload.get("schema") == PRIOR
              and payload.get("feature_names") == list(NAMES) and payload.get("actions") == list(ACTION_NAMES)
              and payload.get("phases") == list(PHASE_NAMES)):
            self.learner = None
        else:
            raise ValueError("invalid observable landing guidance schema")

        self.architecture = self.learner.architecture if self.learner is not None else "prior"
        tuning = default_tuning()._replace(stall_speeds=jnp.asarray(speeds, dtype=jnp.float32))
        self._prior = jax.jit(lambda observation, phase: guidance(observation, phase, tuning))

    def reset(self):
        if self.learner is not None:
            self.learner.reset()

    def predict(self, observation, phase):
        observation = _float_array(observation, (len(self.feature_names),), "observation")
        phase = _validate_phases([phase], 1)[0]
        prior = _float_array(self._prior(observation, phase), (len(ACTION_NAMES),), "guidance prior")
        if self.learner is None:
            return prior
        return apply_residual_actions(self.learner.predict(observation, phase), prior, phase)


def parse_policy(data):
    """Parse the three export formats supported by the game runtime."""
    payload = json.loads(data)
    if payload.get("schema") in (PRIOR, SCHEMA):
        return GuidedPolicy(payload)
    return Policy(payload)


def load_policy(path):
    return parse_policy(Path(path).read_bytes())
