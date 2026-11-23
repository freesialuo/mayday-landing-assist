# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Versioned PyTree checkpoints for local directories and mounted Google Drive."""

from __future__ import annotations

import hashlib
import io
import json
import math
import operator
from pathlib import Path
import re
import time
from typing import Any
from uuid import uuid4
from zipfile import BadZipFile

import jax
import numpy as np


_VERSION = 1
_FILE_NAME = re.compile(r"(checkpoint-([0-9]+)-([0-9]+)-[a-f0-9]{32})\.(?:json|npz)")


class CheckpointError(RuntimeError):
    """Checkpoint files exist, but none can restore the requested training state."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree_schema(definition: jax.tree_util.PyTreeDef) -> Any:
    node = definition.node_data()
    if node is None:
        return None
    node_type, metadata = node
    return [
        node_type.__module__,
        node_type.__qualname__,
        repr(metadata),
        [_tree_schema(child) for child in definition.children()],
    ]


def _leaf_storage(value: Any) -> tuple[str, Any]:
    if isinstance(value, jax.Array):
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            return f"key:{value.dtype}", jax.random.key_data(value)
        return "jax", value
    if isinstance(value, np.ndarray):
        return "numpy", value
    if isinstance(value, np.generic):
        return "numpy_scalar", value
    if type(value) in (bool, int, float, complex):
        return "python_scalar", np.asarray(value)
    raise TypeError(f"Unsupported checkpoint leaf type {type(value).__qualname__}")


def _flatten(tree: Any) -> tuple[list[Any], jax.tree_util.PyTreeDef, list[tuple[str, Any]], str]:
    leaves, definition = jax.tree_util.tree_flatten(tree)
    storage = [_leaf_storage(leaf) for leaf in leaves]
    schema = []
    for kind, value in storage:
        dtype = np.dtype(value.dtype)
        if dtype.hasobject or dtype.itemsize == 0:
            raise TypeError(f"Unsupported checkpoint leaf dtype {dtype}")
        schema.append([kind, value.shape, dtype.str, str(dtype)])
    return leaves, definition, storage, _sha256(_json_bytes([_tree_schema(definition), schema]))


def _write_new_file(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        if stream.write(data) != len(data):
            raise OSError(f"Incomplete checkpoint write to {path}")


def save_checkpoint(directory: str | Path, step: int, tree: Any, config: Any) -> Path:
    """Write a new generation and return its completion manifest path.

    ``config`` must be JSON-serializable and identify the run and state schema.
    ``tree`` may contain JAX arrays, NumPy arrays and scalars, or numeric Python
    scalars. Custom PyTree node metadata must have a stable representation.
    Each generation has its own filenames. Close the payload before writing the
    manifest so an interrupted write leaves earlier generations readable.
    """
    step = operator.index(step)
    if step < 0:
        raise ValueError("Checkpoint step must be nonnegative")
    config_hash = _sha256(_json_bytes(config))
    _, _, storage, schema_hash = _flatten(tree)
    # Byte views also preserve JAX dtypes that NumPy's NPY header stores as void.
    arrays = [np.ascontiguousarray(value).view(np.uint8).reshape(-1) for _, value in storage]
    with io.BytesIO() as buffer:
        np.savez(buffer, *arrays)
        payload = buffer.getvalue()

    manifest = _json_bytes({
        "version": _VERSION,
        "step": step,
        "config": config_hash,
        "schema": schema_hash,
        "length": len(payload),
        "sha256": _sha256(payload),
    })
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    generation = directory / f"checkpoint-{step:020d}-{time.time_ns():020d}-{uuid4().hex}"
    _write_new_file(generation.with_suffix(".npz"), payload)
    manifest_path = generation.with_suffix(".json")
    _write_new_file(manifest_path, manifest)
    return manifest_path


def _restore_leaf(encoded: np.ndarray, storage: tuple[str, Any], template: Any) -> Any:
    kind, expected = storage
    dtype = np.dtype(expected.dtype)
    size = math.prod(expected.shape)
    if (
        not isinstance(encoded, np.ndarray)
        or encoded.dtype != np.uint8
        or encoded.shape != (size * dtype.itemsize,)
    ):
        raise ValueError("Checkpoint array does not match its expected size or dtype")
    array = np.frombuffer(encoded, dtype=dtype, count=size).reshape(expected.shape)
    if kind.startswith("key:"):
        return jax.random.wrap_key_data(jax.device_put(array), impl=jax.random.key_impl(template))
    if kind == "jax":
        restored = jax.device_put(array)
        if restored.dtype != dtype:
            raise ValueError(f"Current JAX configuration cannot restore dtype {dtype}")
        return restored
    if kind == "numpy_scalar":
        return array[()]
    if kind == "python_scalar":
        return array.item()
    return array


def _read_generation(
    manifest_path: Path,
    step: int,
    config_hash: str,
    schema_hash: str,
    storage: list[tuple[str, Any]],
    leaves: list[Any],
) -> list[Any]:
    manifest = json.loads(manifest_path.read_bytes())
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != _VERSION
        or type(manifest.get("step")) is not int
        or manifest["step"] != step
    ):
        raise ValueError("Invalid checkpoint version or step")
    if manifest.get("config") != config_hash:
        raise ValueError("Training configuration differs")
    if manifest.get("schema") != schema_hash:
        raise ValueError("PyTree structure, shapes, or dtypes differ")

    payload = manifest_path.with_suffix(".npz").read_bytes()
    if manifest.get("length") != len(payload) or manifest.get("sha256") != _sha256(payload):
        raise ValueError("Payload size or SHA-256 does not match its manifest")

    names = [f"arr_{index}" for index in range(len(storage))]
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if archive.files != names:
            raise ValueError("Checkpoint archive has unexpected entries")
        return [_restore_leaf(archive[name], entry, leaf) for name, entry, leaf in zip(names, storage, leaves)]


def load_checkpoint(directory: str | Path, template: Any, config: Any) -> tuple[int, Any] | None:
    """Restore the highest compatible step using a matching PyTree template.

    JAX arrays are restored on the current default device. Fall back to older
    compatible generations if a newer payload or manifest is incomplete or corrupt.
    Return ``None`` for an unused directory and raise ``CheckpointError`` if
    checkpoint artifacts exist but none pass validation.
    """
    directory = Path(directory)
    generations = set()
    try:
        for path in directory.iterdir():
            match = _FILE_NAME.fullmatch(path.name)
            if match is not None:
                name, step, timestamp = match.groups()
                generations.add((int(step), int(timestamp), name))
    except FileNotFoundError:
        return None
    if not generations:
        return None

    config_hash = _sha256(_json_bytes(config))
    leaves, definition, storage, schema_hash = _flatten(template)
    errors = []
    for step, _, name in sorted(generations, reverse=True):
        manifest_path = directory / f"{name}.json"
        try:
            restored = _read_generation(manifest_path, step, config_hash, schema_hash, storage, leaves)
            return step, jax.tree_util.tree_unflatten(definition, restored)
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, EOFError, BadZipFile) as error:
            if len(errors) < 3:
                errors.append(f"{manifest_path.name}: {error}")

    details = "\n".join(errors)
    raise CheckpointError(f"No compatible, complete checkpoint in {directory}\n{details}")
