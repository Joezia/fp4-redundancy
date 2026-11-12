"""Complete categorical medoid VQ + exact XOR correction + A1 arithmetic."""
from functools import lru_cache
import hashlib
import json
import os

from ..types import MethodSpec, Support
from ._formats import support
from .native import NativeA, ROOT, FORMATS, _call, _buffer_library


@lru_cache(maxsize=1)
def library():
    return _buffer_library(ROOT / "build/libfp4_codeword_vq.so", "fp4_vq")


def build_metadata():
    metadata = json.loads((ROOT / "build/codeword-vq-build.json").read_text())
    source, h = ROOT / "native/codeword_vq", hashlib.sha256()
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(source)).encode()); h.update(path.read_bytes())
    if h.hexdigest() != metadata["source_sha256"]:
        raise RuntimeError("Codeword VQ sources changed; run scripts/build_codeword_vq.py")
    for path, field in (("build/libfp4_codeword_vq.so", "library_sha256"), ("build/libfp4_entropy.so", "a1_library_sha256")):
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != metadata[field]:
            raise RuntimeError("Codeword VQ library/dependency changed; rebuild codeword VQ")
    for path, digest in metadata["dependencies_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Codeword VQ header changed; rebuild codeword VQ")
    metadata["upstream"] = json.loads((source/"sources.json").read_text())
    metadata["runtime_threads"] = int(os.environ.get("FP4_VQ_THREADS", "1"))
    return metadata


def options(parameters):
    defaults = {"vector_length": 8, "centers": 16, "correction_model": "auto", "objective": "rate"}
    if set(parameters) - defaults.keys():
        raise ValueError("Unknown codeword VQ parameter")
    value = {**defaults, **parameters}
    if type(value["vector_length"]) is not int or value["vector_length"] not in (2, 4, 8):
        raise ValueError("vector_length must be 2, 4 or 8")
    if type(value["centers"]) is not int or value["centers"] not in (4, 16, 64):
        raise ValueError("centers must be 4, 16 or 64")
    if value["correction_model"] not in ("shared", "conditioned", "auto") or value["objective"] not in ("hamming", "rate"):
        raise ValueError("Invalid correction_model or objective")
    return value


class NativeCodewordVQ(NativeA):
    _library = staticmethod(library)

    def __init__(self):
        library()
        self.codec, self.exponent_split, self.backend = 80, False, "cpp-codeword-vq-v1"
        self._options = options({})
        self.spec = MethodSpec("vq_codeword", "码字模式 VQ 无损适配 C/C++", "C", "adapted",
            "categorical medoid vector quantization; entropy-constrained assignment; existing YAECL A1 arithmetic",
            "实际观察的完整码字向量作 medoid，packed 码本；五轮 Hamming/预计索引＋XOR纠错成本训练，"
            "至多2048个分层确定性抖动抽取的原序训练向量；可选中心×位置条件纠错及计费后回退共享模型；非传统数值VQ。",
            self.backend, "arithmetic", "complete_scheme")

    def check_support(self, target, parameters):
        if "unit_bits" in parameters:
            from .grouped_units import support as grouped_support
            return grouped_support(self, target, parameters)
        base = support(target, {})
        if not base.applicable:
            return base
        try:
            options(parameters)
        except (ValueError, TypeError) as exc:
            return Support(False, str(exc))
        if any(c.length > 1 << 20 or c.storage_format == "BF16" for c in target.components):
            return Support(False, "Codeword VQ supports <=1 MiB components with complete 4/8-bit symbols; BF16 not implemented")
        return base

    def encode(self, target, components, parameters):
        self._options = options({k:v for k,v in parameters.items() if k != "unit_bits"})
        return super().encode(target, components, parameters)

    def _encode_frame(self, target, component, raw, split):
        o = self._options
        flags = o["vector_length"] | (o["centers"] << 8) | (("shared", "conditioned", "auto").index(o["correction_model"]) << 16)
        flags |= (o["objective"] == "rate") << 18
        lib = self._library()
        return _call(lib.fp4_native_encode, self.codec, FORMATS[component.storage_format], flags, raw, len(raw), _library=lib)
