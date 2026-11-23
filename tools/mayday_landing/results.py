"""Verify a downloaded result bundle and score complete flights at the supported latencies."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def _verify_fixed_artifact(report, content):
    fixed = report.get("fixed_evaluation_artifact")
    if fixed is None:
        raise ValueError("Report does not identify a fixed evaluation artifact")
    if hashlib.sha256(content).hexdigest() != fixed["sha256"]:
        raise ValueError(f"Fixed evaluation artifact failed SHA-256 verification: {fixed['name']}")


def restore(bundle, directory=None, *, fixed_artifact=None):
    """Verify result contents and optionally extract them and check the evaluated input file."""
    permitted = {"mlp.json", "gru.json", "prior.json", "report.json", "source.sha256", "arguments.json",
                 "training.log", "hardware.jsonl", "handoff.json", "handoff.log"}
    with ZipFile(bundle) as archive:
        names = archive.namelist()
        if not {"report.json", "source.sha256", "arguments.json"} <= set(names) or len(names) != len(set(names)):
            raise ValueError("Incomplete or duplicate result members")
        if not set(names) <= permitted:
            raise ValueError("Unexpected result member")
        artifacts = {name: archive.read(name) for name in names}
    report = json.loads(artifacts["report.json"])
    exported = report.get("policy_sha256", {})
    for name, expected in exported.items():
        if name not in artifacts or hashlib.sha256(artifacts[name]).hexdigest() != expected:
            raise ValueError(f"Exported weights failed SHA-256 verification: {name}")

    fixed = report.get("fixed_evaluation_artifact")
    if fixed_artifact is not None:
        _verify_fixed_artifact(report, fixed_artifact.read_bytes())
    elif fixed is not None and fixed["name"] in artifacts and fixed["name"] not in exported:
        # An evaluation can independently export a baseline under the same name
        # as its original input, so those two records have separate checksums.
        _verify_fixed_artifact(report, artifacts[fixed["name"]])

    if directory is None:
        return report
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        path = directory / name
        if path.exists() and (not path.is_file() or path.read_bytes() != content):
            raise FileExistsError(f"Result would overwrite a different artifact: {path}")
    for name, content in artifacts.items():
        if not (directory / name).exists():
            (directory / name).write_bytes(content)
    return report


def _summary(rows):
    return {"n": len(rows), "success": sum(row["success"] for row in rows),
            "five_rings": sum(row["ring_verdicts"] == [1]*5 for row in rows),
            "stopped": sum(row["parked"] for row in rows),
            "termination": dict(Counter(row["termination"] for row in rows)),
            "go_around_reacquired": sum(row["go_around_reacquired"] for row in rows if row["branch"] == "go_around"),
            "go_around_attempts": sum(row["branch"] == "go_around" for row in rows)}


def scorecard(report):
    result = {}
    for name, evaluation in report.get("evaluations", {}).items():
        if evaluation["split"] != "holdout":
            raise ValueError("A scorecard requires held-out episodes")
        rows = evaluation["episodes"]
        low_mid = [row for row in rows if row["observation_lag_steps"] <= 3]
        locked = None
        if all(row.get("gear_extension_final") is not None for row in low_mid):
            locked = _summary([row for row in low_mid if row["gear_extension_final"] > 0.9])
        result[name] = {"all_attempts": _summary(rows), "low_mid_attempts": _summary(low_mid),
                        "low_mid_gear_extended": locked}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="training report or exported Colab ZIP")
    parser.add_argument("--extract", type=Path, help="verify and restore a ZIP into this directory")
    parser.add_argument("--fixed-artifact", type=Path, help="verify the original input of an evaluation-only run")
    args = parser.parse_args()
    if args.extract or args.source.suffix.lower() == ".zip":
        report = restore(args.source, args.extract, fixed_artifact=args.fixed_artifact)
    else:
        report = json.loads(args.source.read_bytes())
        if args.fixed_artifact is not None:
            _verify_fixed_artifact(report, args.fixed_artifact.read_bytes())
    print(json.dumps(scorecard(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
