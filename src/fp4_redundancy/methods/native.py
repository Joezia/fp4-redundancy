"""Complete A plugins backed by pinned upstream C++ coders via buffer-only FFI."""

import base64
import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import struct
import zlib

from ..types import EncodedArtifact, MethodSpec, Support
from ._formats import FIELDS, support

ROOT = Path(__file__).resolve().parents[3]
FORMATS = {name: i for i, name in enumerate(("fp4_packed_low_first", "fp4_packed_high_first",
           "F8_E4M3", "F8_E5M2", "F8_E8M0", "BF16", "F32"))}
BACKEND = "cpp-sz3-yaecl-ryg-v1"
RICE_BACKEND = "cpp-flac-rice-v1"
PPM_BACKEND = "cpp-nayuki-ppm-yaecl-v1"
MAGIC = 0x31414E46


def _buffer_library(path, prefix):
    """Private shared buffer ABI, not a configurable codec-stage interface."""
    if not path.is_file():
        raise RuntimeError(f"Native library not built: {path.name}")
    lib = ctypes.CDLL(str(path))
    pointer = ctypes.POINTER(ctypes.c_uint8)
    for name in ("encode", "decode", "free", "error"):
        setattr(lib, "fp4_native_" + name, getattr(lib, prefix + "_" + name))
    lib.fp4_native_encode.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
                                    ctypes.c_size_t, ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_size_t)]
    lib.fp4_native_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_size_t)]
    lib.fp4_native_free.argtypes = [ctypes.c_void_p]
    lib.fp4_native_free.restype = None
    lib.fp4_native_error.restype = ctypes.c_char_p
    return lib


@lru_cache(maxsize=1)
def library():
    path = ROOT / "build/libfp4_entropy.so"
    if not path.is_file():
        raise RuntimeError("Native codecs not built; run python3 scripts/build_native.py")
    lib = ctypes.CDLL(str(path))
    pointer = ctypes.POINTER(ctypes.c_uint8)
    lib.fp4_native_encode.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
                                     ctypes.c_size_t, ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_size_t)]
    lib.fp4_native_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                     ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_size_t)]
    lib.fp4_native_free.argtypes = [ctypes.c_void_p]
    lib.fp4_native_free.restype = None
    lib.fp4_native_error.restype = ctypes.c_char_p
    return lib


def build_metadata():
    metadata = json.loads((ROOT / "build/native-build.json").read_text())
    fingerprint = hashlib.sha256()
    source = ROOT / "native"
    for path in sorted(p for p in source.rglob("*") if p.is_file() and not {"predictive", "shared_patterns", "math_structure", "codeword_vq", "grouped_units", "grouped_jpeg"}.intersection(p.relative_to(source).parts)):
        fingerprint.update(str(path.relative_to(source)).encode())
        fingerprint.update(path.read_bytes())
    if fingerprint.hexdigest() != metadata["source_sha256"]:
        raise RuntimeError("Native sources changed; rebuild explicitly")
    if hashlib.sha256((ROOT / "build/libfp4_entropy.so").read_bytes()).hexdigest() != metadata["library_sha256"]:
        raise RuntimeError("Native library differs from build metadata; rebuild explicitly")
    metadata["upstream"] = json.loads((ROOT / "native/third_party/sources.json").read_text())
    return metadata


def _call(function, *args, _library=None):
    lib = _library if _library is not None else library()
    output = ctypes.POINTER(ctypes.c_uint8)()
    length = ctypes.c_size_t()
    status = function(*args, ctypes.byref(output), ctypes.byref(length))
    if status:
        raise ValueError(lib.fp4_native_error().decode())
    try:
        return ctypes.string_at(output, length.value)
    finally:
        lib.fp4_native_free(output)


def _json(value):
    return json.dumps(value, separators=(",", ":")).encode()


class NativeA:
    _library = staticmethod(library)

    def _encode_frame(self, target, component, raw, split):
        lib = self._library()
        return _call(lib.fp4_native_encode, self.codec, FORMATS[component.storage_format],
                     split, raw, len(raw), _library=lib)

    def __init__(self, technique, exponent_split=False):
        if technique not in ("huffman", "arithmetic", "rans") and exponent_split:
            raise ValueError("This method retains complete codewords and has no exponent-split variant")
        library()  # Fail clearly; never fall back to Python or a different codec.
        self.codec = {"huffman": 0, "arithmetic": 1, "rans": 2, "golomb_rice": 3, "ppm": 4,
                      "ctw": 5, "paq": 6, "cabac": 7, "neural_entropy": 8, "cavlc": 9}[technique]
        self.backend = {"golomb_rice": RICE_BACKEND, "ppm": PPM_BACKEND,
                        "ctw": "cpp-kt-ctw-yaecl-v1", "paq": "cpp-zpaq-mix2-yaecl-v1",
                        "cabac": "cpp-openh264-cabacmodel-yaecl-v1", "neural_entropy": "cpp-fann-mlp-yaecl-v1",
                        "cavlc": "cpp-openh264-cavlc-v1"}.get(technique, BACKEND)
        self.exponent_split = exponent_split
        self.technique = technique
        method_id = technique + ("_fp8_bf16_exp" if exponent_split else "")
        source = {"huffman": "SZ3 canonical HuffmanEncoder<uint16_t>, 32-bit maximum code length",
                  "arithmetic": "YAECL arithmetic coding, 32-bit interval, static 24-bit integer CDF",
                  "rans": "ryg_rans byte rANS, 32-bit state, 12-16 bit static frequencies",
                  "golomb_rice": "Golomb-Rice (m=2^k), RFC 9639 section 9.2.7; adapted xiph/flac "
                                 "59fa3402cb19b6b7c85b916f138cfb03c7b27504 Rice bit I/O, unsigned codewords",
                  "ppm": "Nayuki Reference-arithmetic-coding ab6ee50afec04d235a4b82d17f407f0fd2b42e9a "
                         "order-3 PPM, escape=1, no exclusion, uniform order -1; existing YAECL 32-bit coder, 24-bit CDF",
                  "ctw": "m-ary KT/CTW; checked against cran/BCT 7256534c1617af656e8f3255676a34733225dcb3; integer posterior adaptation",
                  "paq": "ZPAQ MIX2 predictor/update and fixed squash/stretch tables, zpaq/zpaq 9ab539f644e364f0d92e2918b90ce2534c75653f",
                  "cabac": "OpenH264 1bca62311e447fd9c39fc93de2c4b96cd5ce971c CABAC probability/transition tables; A1 arithmetic backend",
                  "neural_entropy": "FANN 1ac003a62a5186a662ddab430fa479ab81e55cbd trained 8-8-1 autoregressive MLP; serialized Q12 weights and integer inference",
                  "cavlc": "OpenH264 1bca62311e447fd9c39fc93de2c4b96cd5ce971c CAVLC token/trailing-one/level/zero/run syntax; native VLC backend"}[technique]
        names = {"ppm": "Nayuki 风格 PPM＋算术编码", "ctw": "类 CTW 多符号定点模型＋算术编码",
                 "paq": "PAQ/ZPAQ 风格 Context Mixing＋算术编码", "cabac": "类 CABAC 概率模型＋算术编码",
                 "neural_entropy": "FANN 神经熵模型＋算术编码", "cavlc": "类 CAVLC 完整编码"}
        adaptations = {"ctw": "三阶 KT/上下文树混合；15-bit 整数后验权重，稀疏统计，确定性重置/重缩放。",
                       "paq": "复用 MIX2；仅保留两个整码字前缀专家，省略原归档/JIT/专用预测器；不是完整 PAQ/ZPAQ。",
                       "cabac": "复用概率和 MPS/LPS 更新；整码字前缀上下文，固定 range 桶 2；不使用原 CABAC 区间编码器。",
                       "neural_entropy": "每组件最多 2048 个训练样本、4 轮 FANN 训练；全部 81 个 Q12 权重计费；整数推理生成整码字 CDF。",
                       "cavlc": "原始整码字整数直接作非负 level，16 符号块，前块 nC，保留 CAVLC 全部五类编码步骤，扩展 level 前缀以覆盖 uint16；无视频变换/量化。"}
        display_name = names.get(technique, technique)
        self.spec = MethodSpec(method_id, display_name + " C/C++" + ("（FP8/BF16 指数分离）" if exponent_split else "（完整符号）"),
                               "A", "adapted" if exponent_split or self.codec >= 4 else "native", source,
                               ("FLAC Rice 位流循环适配为无符号码字；不使用音频预测或有符号折叠；"
                                "每流精确选 k，无原样回退。" if technique == "golomb_rice" else
                                "Nayuki 风格 PPM 无损适配：完整码字、稀疏上下文、65536 节点上限确定性重置、"
                                "2^20 计数重缩放；按符号数量结束；固定复用 A1 算术编码。" if technique == "ppm" else adaptations.get(technique, "")) +
                               ("FP8/BF16 指数与其余位分别编码" if exponent_split else ""),
                               self.backend,
                               "arithmetic" if self.codec >= 4 else "",
                               "complete_scheme" if technique == "cavlc" else "probability_model" if self.codec >= 4 else "")

    def check_support(self, target, parameters):
        if "unit_bits" in parameters:
            from .grouped_units import support as grouped_support
            return grouped_support(self, target, parameters)
        result = support(target, parameters)
        if not result.applicable:
            return result
        if any(c.length > 16 << 20 for c in target.components):
            return Support(False, "Native component limit is 16 MiB; use smaller chunks")
        if self.exponent_split and not any(c.storage_format in FIELDS for c in target.components):
            return Support(False, "Exponent separation applies only to FP8/BF16; use the whole-symbol method")
        return result

    def encode(self, target, components, parameters):
        if "unit_bits" in parameters:
            from .grouped_units import encode as grouped_encode
            return grouped_encode(self, target, components, parameters)
        supported = self.check_support(target, parameters)
        if not supported.applicable:
            raise ValueError(supported.reason)
        if set(components) != {c.id for c in target.components}:
            raise ValueError("Input component set differs from target")
        payload, parts = bytearray(), []
        for component in target.components:
            raw, fmt = components[component.id], component.storage_format
            if len(raw) != component.length:
                raise ValueError("Input length differs from target")
            split = self.exponent_split and fmt in FIELDS
            frame = self._encode_frame(target, component, raw, split)
            magic, size, codec, format_id, actual_split, count = struct.unpack_from("<IQIIII", frame)
            if (magic, size, codec, format_id, actual_split) != (MAGIC, len(raw), self.codec, FORMATS[fmt], int(split)):
                raise ValueError("Native encode returned an incompatible frame")
            offset, streams = 28, []
            for _ in range(count):
                n, model_size, data_size = struct.unpack_from("<QQQ", frame, offset)
                offset += 24
                model = frame[offset:offset + model_size]
                offset += model_size
                data = frame[offset:offset + data_size]
                offset += data_size
                record = {"count": n, "bytes": data_size,
                          "model": {"backend": self.backend, "table_b64": base64.b64encode(model).decode()}}
                if self.codec == 0:
                    record["model"]["payload_bits"] = struct.unpack_from("<Q", data)[0]
                elif self.codec == 1:
                    record["model"]["payload_bits"] = struct.unpack_from("<Q", model, len(model) - 8)[0]
                elif self.codec == 3:
                    record["model"]["rice_parameter"], record["model"]["payload_bits"] = struct.unpack("<BQ", model)
                elif self.codec == 4:
                    version, alphabet, order, contexts, rescale, precision, bits = struct.unpack("<IIIIIIQ", model)
                    record["model"].update(ppm_version=version, alphabet=alphabet, order=order,
                                          max_contexts=contexts, rescale_at=rescale, cdf_bits=precision,
                                          payload_bits=bits)
                elif self.codec >= 5:
                    record["model"]["payload_bits"] = struct.unpack_from("<Q", model, len(model) - 8)[0]
                payload.extend(data)
                streams.append(record)
            if offset != len(frame):
                raise ValueError("Native frame length mismatch")
            parts.append({"id": component.id, "role": component.role, "format": fmt,
                          "bytes": len(raw), "crc32": zlib.crc32(raw), "split": split, "streams": streams})
        header = {"version": 1, "method": self.spec.id, "backend": self.backend,
                  "target_id": target.id, "logical_shape": target.logical_shape, "components": parts}
        for key in ("chunk", "tensor_names", "storage_shapes", "scale_block_shape"):
            if key in target.structure:
                header[key] = target.structure[key]
        return EncodedArtifact(bytes(payload), _json(header))

    def decode(self, artifact):
        header = json.loads(artifact.side_information)
        if "unit_bits" in header:
            from .grouped_units import decode as grouped_decode
            return grouped_decode(self, artifact)
        if header["version"] != 1 or header["method"] != self.spec.id or header["backend"] != self.backend:
            raise ValueError("Native artifact backend/method/version mismatch")
        lib, output, offset = self._library(), {}, 0
        for part in header["components"]:
            fmt = part["format"]
            split = self.exponent_split and fmt in FIELDS
            if (part["id"] in output or part["split"] != split or len(part["streams"]) != (2 if split else 1)):
                raise ValueError("Invalid native component layout")
            frame = bytearray(struct.pack("<IQIIII", MAGIC, part["bytes"], self.codec,
                                          FORMATS[fmt], int(split), len(part["streams"])))
            for stream in part["streams"]:
                if stream["model"]["backend"] != self.backend:
                    raise ValueError("Native stream backend mismatch")
                model = base64.b64decode(stream["model"]["table_b64"], validate=True)
                end = offset + stream["bytes"]
                if stream["bytes"] < 0 or end > len(artifact.payload):
                    raise ValueError("Truncated native artifact")
                frame.extend(struct.pack("<QQQ", stream["count"], len(model), stream["bytes"]))
                frame.extend(model)
                frame.extend(artifact.payload[offset:end])
                offset = end
            raw_frame = bytes(frame)
            raw = _call(lib.fp4_native_decode, raw_frame, len(raw_frame), _library=lib)
            if len(raw) != part["bytes"] or zlib.crc32(raw) != part["crc32"]:
                raise ValueError("Native decode checksum/length mismatch")
            output[part["id"]] = raw
        if offset != len(artifact.payload) or not output:
            raise ValueError("Trailing native payload/empty components")
        return output
