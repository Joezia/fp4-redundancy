"""Header-only adapter for the supplied DeepSeek-V4.1-Flash export.

Packing/scales: pinned snapshot inference/convert.py, model.py and kernel.py.
Safetensors offsets are relative to the end of its 8-byte length + JSON header.
No checkpoint code is imported or executed; no tensor payload is read here.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from math import prod
from pathlib import Path
import re
import struct

from .types import Component, ModelManifest, Target

MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "2cba9e42aa026125f3ed06c6d98c1db82f7ca027"
DTYPE_BYTES = {"I8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F32": 4}
ADAPTER = "deepseek-v41-flash-v1"


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data):
    result = json.loads(data, object_pairs_hook=_object)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: str
    offset: int
    length: int


def read_header(path):
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors prefix: {path}")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(100_000_000, size - 8):
            raise ValueError(f"Invalid safetensors header length: {path}")
        raw = stream.read(length)
    if not raw.startswith(b"{"):
        raise ValueError("Safetensors header must begin with an object")
    header, tensors = _json(raw), []
    for name, item in header.items():
        if name == "__metadata__":
            if not isinstance(item, dict) or any(not isinstance(v, str) for v in item.values()):
                raise ValueError("Invalid safetensors metadata")
            continue
        dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported checkpoint dtype: {dtype} ({name})")
        if not isinstance(shape, list) or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError(f"Unsupported empty or invalid tensor shape: {name}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)):
            raise ValueError(f"Invalid tensor offsets: {name}")
        start, end = offsets
        if end - start != prod(shape) * DTYPE_BYTES[dtype] or end > size - 8 - length:
            raise ValueError(f"Tensor size/range mismatch: {name}")
        tensors.append(Tensor(name, dtype, tuple(shape), str(path.absolute()),
                              8 + length + start, end - start))
    cursor = 8 + length
    for tensor in sorted(tensors, key=lambda t: t.offset):
        if tensor.offset != cursor:
            raise ValueError(f"Overlapping or unindexed safetensors data: {path}")
        cursor += tensor.length
    if cursor != size:
        raise ValueError(f"Unindexed trailing safetensors data: {path}")
    return tensors, hashlib.sha256(raw).hexdigest(), 8 + length


def _structure(name):
    parts = name.split(".")
    domain = "mtp" if parts[0] == "mtp" else "vision" if parts[0] == "vision" else "text"
    match = re.match(r"(?:layers|mtp|vision\.blocks)\.(\d+)\.", name)
    expert = re.search(r"\.experts\.(\d+)\.", name)
    return {"module": name.rsplit(".", 1)[0], "parameter_role": parts[-1], "domain": domain,
            "layer": int(match[1]) if match else "unknown",
            "expert": int(expert[1]) if expert else "unknown"}


def load_checkpoint(snapshot, *, revision=REVISION):
    snapshot = Path(snapshot).absolute()
    if revision != REVISION or (re.fullmatch(r"[0-9a-f]{40}", snapshot.name) and snapshot.name != revision):
        raise ValueError("This adapter supports only the supplied fixed revision")
    config_raw = (snapshot / "config.json").read_bytes()
    config = _json(config_raw)
    quant = config.get("quantization_config", {})
    if (config.get("model_type") != "deepseek_v41" or quant.get("expert_dtype") != "fp4"
            or quant.get("quant_method") != "fp8"
            or quant.get("weight_block_size") != [32, 32] or quant.get("scale_fmt") != "ue8m0"):
        raise ValueError("Unsupported model/quantization configuration")
    index_raw = (snapshot / "model.safetensors.index.json").read_bytes()
    weight_map = _json(index_raw)["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Empty or invalid checkpoint weight_map")
    if set(weight_map.values()) != {p.name for p in snapshot.glob("*.safetensors")}:
        raise ValueError("Shard files and checkpoint index disagree")
    tensors, header_hashes, container_bytes = {}, {}, 0
    for shard in sorted(set(weight_map.values())):
        if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError("Invalid shard filename")
        items, digest, size = read_header(snapshot / shard)
        header_hashes[shard] = digest
        container_bytes += size
        for tensor in items:
            if tensor.name in tensors or weight_map.get(tensor.name) != shard:
                raise ValueError(f"Index/header disagreement: {tensor.name}")
            tensors[tensor.name] = tensor
    if set(tensors) != set(weight_map):
        raise ValueError("Index references missing tensors")
    scales = {name for name, t in tensors.items() if t.dtype == "F8_E8M0"}
    used_scales, targets = set(), []
    for name, weight in tensors.items():
        if name in scales:
            continue
        logical, block, scale = weight.shape, None, None
        fmt, role = weight.dtype, "other"
        if weight.dtype in ("I8", "F8_E4M3"):
            if len(weight.shape) != 2 or not name.endswith(".weight"):
                raise ValueError(f"Unsupported quantized tensor: {name}")
            scale_name = name.removesuffix(".weight") + ".scale"
            scale = tensors.get(scale_name)
            if scale is None or scale.dtype != "F8_E8M0" or scale_name in used_scales:
                raise ValueError(f"Missing/shared/invalid companion scale: {name}")
            rows, columns = weight.shape
            if weight.dtype == "I8":
                if not re.fullmatch(r"(?:layers|mtp)\.\d+\.ffn\.experts\.\d+\.w[123]\.weight", name):
                    raise ValueError(f"I8 is not a recognized packed FP4 expert: {name}")
                logical = (rows, columns * 2)
                fmt, role, block = "fp4_packed_low_first", "packed_fp4", (1, 32)
            elif re.fullmatch(r"layers\.\d+\.engram\.embed\.weight", name):
                block = (1, 32)
            else:
                block = (32, 32)
            expected = tuple((n + b - 1) // b for n, b in zip(logical, block))
            if scale.shape != expected or logical[1] % 32:
                raise ValueError(f"Scale layout mismatch for {name}: {scale.shape} vs {expected}")
            used_scales.add(scale_name)
        components = [Component("weight", role, weight.path, weight.offset, weight.length, fmt)]
        tensor_names, shapes = {"weight": name}, {"weight": list(weight.shape)}
        if scale is not None:
            components.append(Component("scale", "scale", scale.path, scale.offset, scale.length, scale.dtype))
            tensor_names["scale"], shapes["scale"] = scale.name, list(scale.shape)
        structure = {**_structure(name), "tensor_names": tensor_names, "storage_shapes": shapes,
                     "scale_block_shape": list(block) if block else None}
        if role == "packed_fp4":
            # Pinned inference/kernel.py defines FP4 = float4_e2m1fn; no model code is executed.
            structure["fp4_value_format"] = "E2M1FN"
        inference = {"consumer": "unknown", "access_unit": "unknown", "transfer_boundary": "unknown",
                     "layout": "safetensors C-order, little-endian",
                     "evidence": f"{MODEL_ID}@{revision}: inference/convert.py, model.py, kernel.py",
                     "reference_access_unit": "embedding_row" if ".engram.embed." in name else "unknown"}
        targets.append(Target(name, tuple(logical), tuple(components), structure, inference))
    if used_scales != scales:
        raise ValueError(f"Unpaired E8M0 scale(s): {sorted(scales - used_scales)[:3]}")
    counts, byte_counts = Counter(), Counter()
    for tensor in tensors.values():
        counts[tensor.dtype] += 1
        byte_counts[tensor.dtype] += tensor.length
    metadata = {"adapter": ADAPTER, "inventory_verified": True, "payload_hashed": False,
                "coverage_scope": "indexed_safetensors_tensor_payload",
                "config_sha256": hashlib.sha256(config_raw).hexdigest(),
                "index_sha256": hashlib.sha256(index_raw).hexdigest(), "header_sha256": header_hashes,
                "tensor_count": len(tensors), "shard_count": len(header_hashes),
                "dtype_counts": dict(counts), "dtype_bytes": dict(byte_counts),
                "tensor_bytes": sum(byte_counts.values()), "container_header_bytes": container_bytes}
    return ModelManifest(MODEL_ID, revision, "complete", tuple(targets), metadata)
