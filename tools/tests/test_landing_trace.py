# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import hashlib
import json
import sys

import pytest

from mayday_landing.features import ACTION_NAMES, INDEX, NAMES
from mayday_landing.replay_trace import audit, main
from mayday_landing.residual_model import wrap_prior


def test_replay_preserves_authority_violations_and_does_not_invent_prior_touchdown(tmp_path):
    obs = [0.0] * len(NAMES)
    base = dict(session=11, valid=True, seated=True, pilotOverride=False, otherOverride=False,
                authorized=False, ap=False, externalController=False, phase=3, status="manual", gates=341,
                sent=None, proposed=None, crashed=False)
    header = dict(schema="mayday-landing.trace/1", features=NAMES, actions=ACTION_NAMES)
    rows = []
    for seq, ground in enumerate((1, 0, 1)):
        obs[INDEX["on_ground"]] = ground
        rows.append(dict(base, seq=seq, time=seq/10, observation=list(obs)))
    rows[-1]["sent"] = [0.] * 5
    path = tmp_path / "flight.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (header, *rows)), encoding="utf8")
    report = audit(path, advice_only=True)
    assert report["sessions"][0]["observed_touchdowns"] == 1
    assert report["sessions"][0]["max_gates_passed"] == 5
    assert report["violations"] == [[2, "command without exclusive authority"]]


@pytest.mark.parametrize(("architecture", "replayed"), [("prior", 4), ("mlp", 4), ("gru", 1)])
def test_replay_requires_a_recorded_reset_only_for_recurrent_policy(tmp_path, architecture, replayed):
    class ProbePolicy:
        feature_names = NAMES

        def reset(self):
            pass

        def predict(self, obs, phase):
            return [0.0] * 5

    policy = ProbePolicy()
    policy.architecture = architecture
    header = dict(schema="mayday-landing.trace/1", features=NAMES, actions=ACTION_NAMES)
    base = dict(valid=True, phase=1, status="manual", sent=None, proposed=[0.0] * 5, crashed=False,
                observation=[0.0] * len(NAMES), predictedNow=True)
    rows = [dict(base, seq=seq, session=session, time=time, policyReset=reset)
            for seq, (session, time, reset) in enumerate([(11, 1.0, False), (11, 1.1, True),
                                                          (11, 1.0, False), (12, 2.0, False)])]
    path = tmp_path / "flight.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (header, *rows)), encoding="utf8")
    report = audit(path, policy)
    assert report["replayed_predictions"] == replayed
    assert report["skipped_unprimed_predictions"] == 4 - replayed
    assert report["violations"] == [[2, "local clock reversed"]]


def test_replay_matches_exact_candidate_bytes_and_allows_model_free_audits(tmp_path, monkeypatch, capsys):
    data = b"\xef\xbb\xbf" + json.dumps(wrap_prior(stall_speeds=(32, 29, 26))).encode() + b"\r\n"
    sha256 = hashlib.sha256(data).hexdigest()
    model = tmp_path / "prior.json"
    model.write_bytes(data)
    trace = tmp_path / "flight.jsonl"
    header = dict(schema="mayday-landing.trace/1", features=NAMES, actions=ACTION_NAMES, policySha256=sha256)
    row = dict(seq=0, session=11, time=0.0, valid=True, crashed=False, phase=3, status="manual", sent=None,
               observation=[0.0] * len(NAMES))
    trace.write_text("\n".join(json.dumps(item) for item in (header, row)), encoding="utf8")

    assert audit(trace)["policy_sha256"] == sha256
    monkeypatch.setattr(sys, "argv", ["replay_trace", str(trace), "--model", str(model)])
    with pytest.raises(SystemExit) as matched:
        main()
    assert matched.value.code == 0
    assert json.loads(capsys.readouterr().out)["policy_sha256"] == sha256

    model.write_bytes(data[:-2] + b"\n")
    with pytest.raises(ValueError, match="model SHA-256 differs from the trace"):
        main()
