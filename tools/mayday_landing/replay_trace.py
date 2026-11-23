"""Audit an M1 flight recording and optionally replay actual C# inference ticks with Python."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from .features import ACTION_NAMES, INDEX, NAMES


def audit(path, policy=None, advice_only=False, *, policy_sha256=None):
    """Use recorded inference/reset ticks, rather than running a GRU on every rendered frame."""
    if policy is not None and tuple(policy.feature_names) != NAMES:
        raise ValueError("model and recorder feature orders differ")
    frames = predictions = skipped = writes = 0
    error_max = 0.0
    previous_session = previous_time = previous_status = None
    stateless = policy is not None and policy.architecture in ("mlp", "prior")
    primed = False
    sessions = []
    alerts = []
    events = []
    with Path(path).open(encoding="utf-8-sig") as file:
        header = json.loads(next(file, "{}"))
        if (header.get("schema") != "mayday-landing.trace/1" or tuple(header.get("features", ())) != NAMES
                or tuple(header.get("actions", ())) != ACTION_NAMES):
            raise ValueError("unsupported or reordered trace schema")
        recorded_sha256 = header.get("policySha256")
        if policy is not None and recorded_sha256 is not None:
            if policy_sha256 is None:
                raise ValueError("replay requires the original model file or its SHA-256")
            if not isinstance(recorded_sha256, str) or recorded_sha256.lower() != policy_sha256.lower():
                raise ValueError(f"model SHA-256 differs from the trace: expected {recorded_sha256}, got {policy_sha256}")
        for line in file:
            row = json.loads(line)
            seq = row["seq"]
            if seq != frames:
                raise ValueError(f"non-contiguous decision sequence at {seq}")
            frames += 1
            current = row["session"]
            now = row["time"]
            if now is None or not math.isfinite(now):
                raise ValueError(f"invalid local clock at {seq}")
            if previous_session != current:
                previous_session = current
                previous_time = previous_status = None
                primed = stateless
                sessions.append({"id": current, "first": now, "last": now, "max_gates_passed": 0,
                                 "observed_touchdowns": 0, "minimum_ground_speed": None, "crashed": False})
                was_ground = None
            session = sessions[-1]
            if previous_time is not None and now < previous_time:
                alerts.append([seq, "local clock reversed"])
                primed = stateless
            previous_time = now
            session["last"] = now
            session["crashed"] |= bool(row["crashed"])
            if previous_status != row["status"]:
                events.append([seq, now, row["phase"], row["status"]])
                previous_status = row["status"]
            obs = row["observation"]
            if len(obs) != len(NAMES):
                raise ValueError(f"invalid feature count at {seq}")
            if row["valid"]:
                mask = row.get("gates", 0)
                passed = sum((mask >> (2*i)) & 3 == 1 for i in range(5))
                session["max_gates_passed"] = max(session["max_gates_passed"], passed)
                ground = obs[INDEX["on_ground"]]
                if ground is not None:
                    ground = ground >= 0.5
                    if was_ground is False and ground:
                        session["observed_touchdowns"] += 1
                    was_ground = ground
                    speed = obs[INDEX["ground_speed"]]
                    if ground and speed is not None:
                        old = session["minimum_ground_speed"]
                        session["minimum_ground_speed"] = speed if old is None else min(old, speed)
            else:
                was_ground = None
            if row["sent"] is not None:
                writes += 1
                if (advice_only or not row["authorized"] or not row["seated"] or row["pilotOverride"]
                        or row["otherOverride"] or not row["valid"] or row["ap"] or row["externalController"]):
                    alerts.append([seq, "command without exclusive authority"])
            if row.get("policyReset") and policy is not None:
                policy.reset()
                primed = True
            if row.get("predictedNow") and policy is not None:
                if not primed:
                    skipped += 1
                    continue
                expected = row["proposed"]
                if expected is None or len(expected) != len(ACTION_NAMES):
                    alerts.append([seq, "missing prediction"])
                    continue
                values = policy.predict(obs, row["phase"])
                error_max = max(error_max, max(float(abs(a-b)) for a, b in zip(values, expected, strict=True)))
                predictions += 1
    if not frames:
        raise ValueError("trace contains no decisions")
    if error_max > 2e-5:
        alerts.append([None, "Python/C# predictions differ"])
    return {"frames": frames, "sessions": sessions, "sent_frames": writes, "policy_sha256": recorded_sha256,
            "replayed_predictions": predictions,
            "skipped_unprimed_predictions": skipped, "max_prediction_error": error_max, "violations": alerts,
            "status_changes": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--advice-only", action="store_true")
    args = parser.parse_args()
    policy = policy_sha256 = None
    if args.model:
        from .residual_model import parse_policy
        model_bytes = args.model.read_bytes()
        policy = parse_policy(model_bytes)
        policy_sha256 = hashlib.sha256(model_bytes).hexdigest()
    result = audit(args.trace, policy, args.advice_only, policy_sha256=policy_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result["violations"]))


if __name__ == "__main__":
    main()
