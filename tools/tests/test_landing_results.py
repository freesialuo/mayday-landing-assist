# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import hashlib
import json
import sys
from zipfile import ZipFile

import pytest

from mayday_landing.results import main, restore, scorecard


def _archive(path, report, **models):
    with ZipFile(path, "w") as bundle:
        bundle.writestr("report.json", json.dumps(report))
        bundle.writestr("source.sha256", "fixture")
        bundle.writestr("arguments.json", "[]")
        for name, content in models.items():
            bundle.writestr(name, content)


def _report():
    episode = dict(success=True, ring_verdicts=[1] * 5, parked=True, termination="stopped",
                   branch="healthy", observation_lag_steps=3, gear_extension_final=1.0)
    return {"evaluations": {"mlp": {"split": "holdout", "episodes": [episode]}}}


def test_zip_can_be_scored_without_extraction(tmp_path, monkeypatch, capsys):
    report = _report()
    artifact = b'{"mode":"residual"}'
    report["policy_sha256"] = {"mlp.json": hashlib.sha256(artifact).hexdigest()}
    archive = tmp_path / "results.zip"
    _archive(archive, report, **{"mlp.json": artifact})
    monkeypatch.setattr(sys, "argv", ["results", str(archive)])
    main()
    assert json.loads(capsys.readouterr().out) == scorecard(report)
    assert list(tmp_path.iterdir()) == [archive]


def test_eval_zip_verifies_candidate_when_supplied(tmp_path):
    report = _report()
    artifact = b'{"mode":"residual"}'
    prior = b'{"mode":"prior"}'
    report["fixed_evaluation_artifact"] = {"name": "mlp.json", "sha256": hashlib.sha256(artifact).hexdigest()}
    report["policy_sha256"] = {"prior.json": hashlib.sha256(prior).hexdigest()}
    archive = tmp_path / "evaluation.zip"
    _archive(archive, report, **{"prior.json": prior})
    assert restore(archive) == report
    candidate = tmp_path / "original.json"
    candidate.write_bytes(artifact)
    assert restore(archive, tmp_path / "restored", fixed_artifact=candidate) == report
    candidate.write_bytes(b"wrong artifact")
    with pytest.raises(ValueError, match="Fixed evaluation artifact failed SHA-256"):
        restore(archive, fixed_artifact=candidate)

    _archive(archive, report, **{"prior.json": prior, "mlp.json": candidate.read_bytes()})
    with pytest.raises(ValueError, match="Fixed evaluation artifact failed SHA-256"):
        restore(archive)

    # A new export can use the same name as the fixed input while their bytes differ.
    report["fixed_evaluation_artifact"]["name"] = "prior.json"
    _archive(archive, report, **{"prior.json": prior})
    assert restore(archive) == report


def test_geared_score_is_unavailable_for_legacy_rows():
    report = _report()
    del report["evaluations"]["mlp"]["episodes"][0]["gear_extension_final"]
    scores = scorecard(report)["mlp"]
    assert scores["low_mid_attempts"]["success"] == 1
    assert scores["low_mid_gear_extended"] is None
