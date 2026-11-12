"""Technique list and implemented method factories."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    name: str
    category: str


# A/B lossless directions are supplied by the user. Listing is not implementation.
CATALOG = (
    CatalogEntry("huffman", "Huffman Coding", "A"),
    CatalogEntry("arithmetic", "Arithmetic Coding", "A"),
    CatalogEntry("rans", "rANS", "A"),
    CatalogEntry("huffman_fp8_bf16_exp", "Huffman（FP8/BF16 指数分离）", "A"),
    CatalogEntry("arithmetic_fp8_bf16_exp", "算术编码（FP8/BF16 指数分离）", "A"),
    CatalogEntry("rans_fp8_bf16_exp", "rANS（FP8/BF16 指数分离）", "A"),
    CatalogEntry("tans", "tANS", "A"),
    CatalogEntry("golomb_rice", "Golomb–Rice Coding", "A"),
    CatalogEntry("ppm", "Nayuki 风格 PPM＋算术编码（完整符号）", "A"),
    CatalogEntry("ctw", "类 CTW 多符号定点模型＋算术编码", "A"),
    CatalogEntry("paq", "PAQ/ZPAQ 风格 Context Mixing＋算术编码", "A"),
    CatalogEntry("neural_entropy", "FANN 神经熵模型＋算术编码", "A"),
    CatalogEntry("cabac", "类 CABAC 概率模型＋算术编码", "A"),
    CatalogEntry("cavlc", "类 CAVLC 完整编码（原生 VLC）", "A"),
    CatalogEntry("delta", "Delta 码字适配＋算术编码", "B"),
    CatalogEntry("xor", "XOR 码字适配＋算术编码", "B"),
    CatalogEntry("dpcm_lossless", "二阶 DPCM 无损码字适配＋算术编码", "B"),
    CatalogEntry("adpcm_like_fp4", "IMA ADPCM 无损码字适配＋算术编码", "B"),
    CatalogEntry("jpeg_ls_like_fp4", "JPEG-LS 无损码字适配（CharLS）", "B"),
    CatalogEntry("png_filtering", "PNG Filtering 码字适配＋算术编码", "B"),
    CatalogEntry("fpc_like_fp4", "FPC 无损码字适配（专用编码）", "B"),
    CatalogEntry("sz_like_lossless", "SZ3 Lorenzo 无损码字适配（Huffman＋Zstd）", "B"),
    CatalogEntry("lpc_lossless", "LPC 无损码字适配＋算术编码", "B"),
    CatalogEntry("ltp_lossless", "单抽头 LTP 无损码字适配＋算术编码", "B"),
    CatalogEntry("motion_compensation", "整数位移 Motion Compensation 无损码字适配＋算术编码", "B"),
    CatalogEntry("intra_directional", "四方向 Intra Prediction 无损码字适配＋算术编码", "B"),
)

from .methods.shared_patterns import METHODS as C_METHODS
from .methods.math_structure import METHODS as D_METHODS
CATALOG += tuple(CatalogEntry(mid, record[2], "C") for mid, record in C_METHODS.items())
CATALOG += (CatalogEntry("inter_frame_reference", "Inter-frame Reference Coding（复用 motion_compensation）", "C"),)
CATALOG += tuple(CatalogEntry(mid, record[2], "D") for mid, record in D_METHODS.items())
CATALOG += (CatalogEntry("vq_codeword", "码字模式 VQ 无损适配", "C"),)


class Registry:
    """Only explicitly registered factories can execute. No research approval gate."""

    def __init__(self) -> None:
        self._factories = {}

    def register(self, factory) -> None:
        method_id = factory().spec.id
        if method_id in self._factories:
            raise ValueError(f"Duplicate method: {method_id}")
        self._factories[method_id] = factory

    def create(self, method_id: str):
        return self._factories[method_id]()

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def builtin_registry() -> Registry:
    from .methods.native import NativeA
    from .methods.predictive import NativeB, METHODS
    from .methods.shared_patterns import NativeC, InterFrameReference
    from .methods.math_structure import NativeD
    from .methods.codeword_vq import NativeCodewordVQ
    registry = Registry()
    for technique in ("huffman", "arithmetic", "rans", "golomb_rice", "ppm", "ctw", "paq", "cabac", "neural_entropy", "cavlc"):
        for split in ((False, True) if technique in ("huffman", "arithmetic", "rans") else (False,)):
            registry.register(lambda t=technique, s=split: NativeA(t, s))
    for technique in METHODS:
        registry.register(lambda t=technique: NativeB(t))
    for technique in C_METHODS:
        registry.register(lambda t=technique: NativeC(t))
    registry.register(InterFrameReference)
    for technique in D_METHODS:
        registry.register(lambda t=technique: NativeD(t))
    registry.register(NativeCodewordVQ)
    return registry


def catalog_output(registry: Registry | None = None) -> dict:
    implemented = builtin_registry().ids() if registry is None else registry.ids()
    return {"kind": "catalog_only",
            "methods": [{**asdict(entry), "implemented": entry.id in implemented}
                        for entry in CATALOG]}


def plan(manifest, categories: list[str]) -> dict:
    if set(categories) - set("ABCDEFG"):
        raise ValueError("Unknown categories")
    chosen = [entry for entry in CATALOG if entry.category in categories]
    return {"kind": "plan_only", "measurements": False,
            "model_id": manifest.model_id, "revision": manifest.revision,
            "coverage": manifest.coverage,
            "jobs": [{"target_id": target.id, "method_id": entry.id,
                      "original_bytes_declared": target.original_bytes,
                      "execution_status": "not_run"}
                     for target in manifest.targets for entry in chosen]}


def iter_chunk_plan(manifest, categories, chunk_bytes, target_id=None):
    """Stream metadata-only JSONL-sized records; never materialize all chunks."""
    from .chunks import iter_target_chunks

    if set(categories) - set("ABCDEFG"):
        raise ValueError("Unknown categories")
    methods = [entry.id for entry in CATALOG if entry.category in categories]
    if target_id is not None and not any(t.id == target_id for t in manifest.targets):
        raise ValueError(f"Unknown target: {target_id}")
    if not methods:
        return
    for target in manifest.targets:
        if target_id is not None and target.id != target_id:
            continue
        for chunk in iter_target_chunks(target, chunk_bytes):
            yield {"kind": "chunk_plan", "measurements": False, "execution_status": "not_run",
                   "model_id": manifest.model_id, "revision": manifest.revision,
                   "coverage": "subset" if target_id is not None else manifest.coverage,
                   "target_id": chunk.id, "logical_shape": chunk.logical_shape,
                   "original_bytes_declared": chunk.original_bytes,
                   "methods": methods, "chunk": chunk.structure["chunk"]}
