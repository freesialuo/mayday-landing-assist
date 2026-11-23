import json

from mayday_landing.features import NAMES
from mayday_landing.probe_report import analyse


def _obs(roll, age=0.1):
    values = [0.0] * len(NAMES)
    values[NAMES.index("roll")] = roll
    values[NAMES.index("observation_age")] = age
    return values


def test_probe_report_measures_gust_kick_touchdown_and_feedback(tmp_path):
    path = tmp_path / "rec.jsonl"
    header = {"schema": "mayday-landing.trace/1", "features": list(NAMES), "probe": "mayday-landing.probe/1"}
    rows = []
    for i in range(40):
        t = 10.0 + 0.05 * i
        roll = 1.0 if i < 10 else -11.0  # native kick lands within one frame
        events = []
        if i == 10:
            events.append({"type": "gust", "time": 10.5, "dir": -1.0, "agl": 30.0, "onGround": False})
        if i == 30:
            events.append({"type": "touchdown", "time": t, "vs": 2.4, "roll": 2.0, "nativeCrash": False})
            events.append({"type": "hazard", "time": t, "kind": "Collision"})
        if i == 35:
            events.append({"type": "mark", "time": t})
        rows.append({"seq": i, "time": t, "valid": True, "authorized": False, "status": "Final approach.",
                     "observation": _obs(roll), "pilot": [0.1, 0.0, 0.0, 0.5, 0.0],
                     "proposed": [0.2, 0.1, 0.0, 0.5, 0.0], "sent": None,
                     "probe": {"prior": [0.1, 0.1, 0.0, 0.4, 0.0], "events": events or None}})
    path.write_text("\n".join(json.dumps(r) for r in [header, *rows]))
    result = analyse(path)
    kick = result["gust_kicks"][0]
    assert abs(kick["step_deg"] + 12.0) < 1e-6 and kick["sign_matches_dir"]
    assert result["gust_kick_summary"]["mean_abs_step_deg"] == 12.0
    assert result["touchdowns"][0]["vs"] == 2.4 and result["hazards"] == {"Collision": 1}
    assert abs(result["policy_minus_prior"]["throttle"] - 0.1) < 1e-6
    assert abs(result["policy_minus_pilot"]["pitch"] - 0.1) < 1e-6
    assert result["marks"][0]["rows"] > 0
