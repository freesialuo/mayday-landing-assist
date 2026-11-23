# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Ordered raw game inputs shared with the embedded .NET schema."""
from importlib.resources import files

NAMES = tuple(files(__package__).joinpath("feature_names.txt").read_text(encoding="ascii").split())
INDEX = {name: index for index, name in enumerate(NAMES)}
ACTION_NAMES = ("pitch", "roll", "yaw", "throttle", "brake")
