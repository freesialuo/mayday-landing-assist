# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Build a training-only Colab bundle. No binaries, gameplay recordings or credentials."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build(output):
    tools_dir = Path(__file__).resolve().parents[1]
    output.parent.mkdir(parents=True, exist_ok=True)
    modules = {
        "mayday_jax": ("__init__", "environment", "physics", "replay"),
        "mayday_landing": ("__init__", "bptt", "environment", "experiment", "features", "guidance", "headroom", "mechanics",
                           "model", "residual_model", "residual_teacher", "taxonomy", "teacher", "tune", "weather"),
    }
    entries = [tools_dir / package / (name + ".py") for package, names in modules.items() for name in names]
    entries.append(tools_dir / "mayday_landing" / "feature_names.txt")
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in entries:
            archive.write(path, path.relative_to(tools_dir).as_posix())
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(build(parser.parse_args().output))
