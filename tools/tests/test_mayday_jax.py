# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from dataclasses import replace

import jax
import numpy as np
import pytest

from mayday_jax.checkpoints import CheckpointError, load_checkpoint
from mayday_jax.physics import default_tuning
from mayday_jax.rollout import RunConfig, _identity, _initialize, run


def test_resume_matches_uninterrupted_and_falls_back_from_corruption(tmp_path):
    config = RunConfig(batch_size=8, seed=42, chunk_steps=128, difficulty=0.25)
    interrupted = tmp_path / "interrupted"
    uninterrupted = tmp_path / "uninterrupted"
    tuning = default_tuning()

    first = run(config, interrupted, 3200, 1000, progress=None, tuning=tuning)
    second = run(replace(config, chunk_steps=201), interrupted, 7200, 1000, progress=None, tuning=tuning)
    reference = run(config, uninterrupted, 7200, 1700, progress=None, tuning=tuning)
    assert first["steps"] == 3200
    assert second["restored_step"] == 3200
    assert second["environment_steps"] == reference["environment_steps"] == 57600
    assert second["episodes_completed"] == reference["episodes_completed"] > 0

    template = _initialize(jax.random.PRNGKey(config.seed), config.difficulty, tuning, config.batch_size)
    identity = _identity(config, tuning)
    step, state = load_checkpoint(interrupted, template, identity)
    reference_step, reference_state = load_checkpoint(uninterrupted, template, identity)
    assert step == reference_step == 7200
    for got, expected in zip(jax.tree.leaves(state), jax.tree.leaves(reference_state), strict=True):
        np.testing.assert_array_equal(got, expected)

    newest_payload = next(interrupted.glob("checkpoint-00000000000000007200-*.npz"))
    newest_payload.write_bytes(b"partial drive write")
    recovered = run(config, interrupted, 7200, 1000, progress=None, tuning=tuning)
    assert recovered["restored_step"] == 7000
    _, recovered_state = load_checkpoint(interrupted, template, identity)
    for got, expected in zip(jax.tree.leaves(recovered_state), jax.tree.leaves(reference_state), strict=True):
        np.testing.assert_array_equal(got, expected)

    incompatible = replace(config, seed=config.seed + 1)
    with pytest.raises(CheckpointError):
        run(incompatible, interrupted, 7200, 1000, progress=None, tuning=tuning)
