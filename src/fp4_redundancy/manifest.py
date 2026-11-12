"""Read and check explicit input manifests."""

from dataclasses import asdict
import json
from pathlib import Path

from .types import Component, ModelManifest, Target


def _keys(value: dict, required: set[str], optional=()) -> None:
    if not isinstance(value, dict):
        raise ValueError("Expected an object")
    missing = required - value.keys()
    unknown = value.keys() - required - set(optional)
    if missing or unknown:
        raise ValueError(f"Missing keys: {sorted(missing)}; unknown keys: {sorted(unknown)}")


def _integer(value: object, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_manifest(path: str | Path, *, check_files: bool = False) -> ModelManifest:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    _keys(raw, {"schema_version", "model", "targets"}, ("metadata",))
    if raw["schema_version"] != "1.0":
        raise ValueError("Unsupported schema_version")
    model = raw["model"]
    _keys(model, {"id", "revision", "coverage"})
    if not _text(model["id"]) or not _text(model["revision"]):
        raise ValueError("Model id and revision are required")
    if model["coverage"] not in ("subset", "complete"):
        raise ValueError("coverage must be subset or complete")
    if not isinstance(raw["targets"], list) or not raw["targets"]:
        raise ValueError("At least one explicit target is required")
    targets, target_ids, ranges, file_sizes = [], set(), {}, {}
    for item in raw["targets"]:
        _keys(item, {"id", "logical_shape", "components", "structure", "inference"})
        if not _text(item["id"]) or item["id"] in target_ids:
            raise ValueError("Target ids must be nonempty and unique")
        target_ids.add(item["id"])
        shape = item["logical_shape"]
        if not isinstance(shape, list) or not all(_integer(n, 1) for n in shape):
            raise ValueError("logical_shape must contain positive integers")
        if not isinstance(item["structure"], dict) or not isinstance(item["inference"], dict):
            raise ValueError("Structure and inference annotations must be objects")
        if not isinstance(item["components"], list) or not item["components"]:
            raise ValueError("Each target needs components")
        components, component_ids = [], set()
        for part in item["components"]:
            _keys(part, {"id", "role", "path", "offset", "length", "storage_format"})
            if any(not _text(part[k]) for k in ("id", "role", "path", "storage_format")):
                raise ValueError("Component identity, role, path and format are required")
            if part["id"] in component_ids:
                raise ValueError("Duplicate component id")
            component_ids.add(part["id"])
            if part["role"] not in ("packed_fp4", "scale", "other"):
                raise ValueError("Components are whole packed FP4, scales, or other bytes; no fields")
            if not _integer(part["offset"]) or not _integer(part["length"], 1):
                raise ValueError("Component range must use nonnegative offset and positive length")
            source = (path.parent / part["path"]).resolve()
            start, end = part["offset"], part["offset"] + part["length"]
            ranges.setdefault(source, []).append((start, end))
            if check_files:
                if source not in file_sizes:
                    if not source.is_file():
                        raise ValueError(f"Source file missing: {source}")
                    file_sizes[source] = source.stat().st_size
                if end > file_sizes[source]:
                    raise ValueError(f"Source range exceeds file: {source}")
            components.append(Component(**{**part, "path": str(source)}))
        targets.append(Target(item["id"], tuple(shape), tuple(components),
                              item["structure"], item["inference"]))
    for spans in ranges.values():
        previous_end = -1
        for start, end in sorted(spans):
            if start < previous_end:
                raise ValueError("Overlapping source ranges would double-count bytes")
            previous_end = end
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    return ModelManifest(model["id"], model["revision"], model["coverage"], tuple(targets), metadata)


def save_manifest(manifest, path):
    """Generated component paths are absolute, so the output location is irrelevant."""
    output = {"schema_version": "1.0",
              "model": {"id": manifest.model_id, "revision": manifest.revision, "coverage": manifest.coverage},
              "metadata": dict(manifest.metadata)}
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, ensure_ascii=False, separators=(",", ":"))[:-1] + ',"targets":[')
        for index, target in enumerate(manifest.targets):
            if index:
                stream.write(",")
            json.dump(asdict(target), stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("]}")
