"""Complete C plugins: native matchers, dictionaries, grammars and trained codebooks."""
import ctypes
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import os

from ..types import MethodSpec, Support
from ._formats import support
from .native import NativeA, ROOT, FORMATS, _call
from .predictive import NativeB

# Fixed algorithms and wire variants, not independently selectable stages.
METHODS = {
    "rle": (30, "C1", "RLE 完整码字适配", "run-length coding", "发现相同码字的极大连续 run，保存长度/完整符号，LEB128 自含封装。"),
    "lz77": (31, "C1", "LZ77 完整码字适配", "Ziv/Lempel 1977 sliding-window triples",
             "哈希链完整核对，65535 符号窗口，最长 258，支持重叠复制；距离/长度/后继符号三元组，LEB128，末尾哨兵。"),
    "lz78": (32, "C1", "LZ78 完整码字适配", "Ziv/Lempel 1978 incremental dictionary",
             "实际构造动态 prefix/符号字典，输出索引/符号对，末尾剩余 phrase 明确终止；LEB128。"),
    "lzss": (33, "C1", "LZSS 完整码字适配", "Storer/Szymanski 1982; checked MichaelDipperstein/lzss",
             "实际匹配，4096 窗口、3..18 符号；每八项 flag byte，12-bit 距离和 4-bit 长度；宽符号 literal 用 LEB128，无预填充窗口。"),
    "lzw": (34, "C1", "LZW 完整码字适配", "Welch 1984; checked MichaelDipperstein/lzw",
            "完整原生字母表初始化，动态 phrase 字典，KwKwK；位宽从 w+1 增长至 20，字典满后冻结；独立块无 clear/EOI，码数及有效位数计费，非 GIF/TIFF 容器。"),
    "repair": (35, "C1", "Re-Pair 完整码字语法适配", "Larsson/Moffat 2000; checked Navarro Re-Pair 2010",
               "发现最高频非重叠 digram，建立规则并全局替换，直到无重复；链表/位置集合/优先队列；字典及根序列全部保存，非原作者二进制文件格式。"),
    "sequitur": (36, "C1", "Sequitur 完整码字语法适配", "craignm/sequitur b6afb6e6049cc7e54284f8f346604937b3c09296",
                 "复用作者 K=1 的 check/substitute/expand，在线维护 digram uniqueness/rule utility；缓冲区与 scoped cleanup，STL 索引；显式语法封装替代原 CLI 算术格式。"),
    "block_dedup": (37, "C1", "固定块去重（完整码字）", "fixed-block content-addressed deduplication",
                    "每 256 完整码字一块；哈希候选逐符号确认；首次字面块 packed 保存，后续引用已恢复字典；含短尾。"),
    "cdc_dedup": (38, "C1", "Gear CDC 去重完整码字适配", "Xia et al. FastCDC 2016; FastCDC-c gear table",
                  "Gear 滚动哈希、min/normal/max=64/256/1024 符号、双掩码归一化；宽符号 hash 适配；分块后完整内容确认、字面块和引用恢复，不是只分块。"),
    "vcdiff": (39, "C2", "Xdelta/VCDIFF 目标内参考适配", "Xdelta 3.1 Apache branch 7508fd2a823443b1f0173ca361620f21d62a7d37; RFC 3284",
               "保留上游匹配、地址缓存、ADD/RUN/COPY 与 VCDIFF 规则；从本组件前缀候选选择并 packed 保存付费参考，FP4 完整码字映射为字节，BF16 为 little-endian bytes。"),
    "rlz": (40, "C2", "RLZ 目标内参考适配", "Kuruppu/Puglisi/Zobel 2010; checked eax99/rlztools 495b3363f9aaa09ca5bf55164d8fa4ef7e3b29c3",
            "前缀参考候选按含参考成本选择；构建 suffix array，贪心最长参考匹配，offset/length 与 literal；自含参考，LEB128 对，非外部免费词典。"),
    "vq_lossless": (41, "C3", "VQ 原生数值训练＋精确纠错适配", "Lloyd vector quantization",
                    "连续八个完整码字一向量，前至多 512 向量实际 Lloyd 训练、最远点初始化，至多16中心，8轮；Q20 码本、索引和完整码字 XOR 纠错。"),
    "pq_lossless": (42, "C3", "PQ 原生数值训练＋精确纠错适配", "Jegou/Douze/Schmid 2011 product quantization",
                    "八维连续向量分两个四维子空间，分别训练码本/分配索引；Q20 码本与完整码字 XOR 纠错，不拆字段或重排。"),
    "opq_lossless": (43, "C3", "OPQ 原生数值训练＋精确纠错适配", "Ge/He/Ke/Sun 2013 non-parametric alternating OPQ",
                     "PQ 与 SVD orthogonal Procrustes 交替四轮，最终重新拟合；完整保存 Q20 旋转矩阵与码本，整数逆向近似及全部位模式 XOR 纠错；变换不作为外部重排实验。"),
    "rvq_lossless": (44, "C3", "RVQ 原生数值训练＋精确纠错适配", "residual vector quantization, two stages",
                     "真实训练第一级码本，再对剩余残差训练第二级；逐级分配、重建求和，保存两个码本/索引与精确纠错。"),
    "aq_lossless": (45, "C3", "AQ 原生数值训练＋精确纠错适配", "Babenko/Lempitsky CVPR 2014 additive quantization",
                    "两个全维码本求和；RVQ 初始化、beam=16 跨剩余码本搜索，四轮索引分配/联合 least-squares 更新；Q20 参数及索引/完整码字 XOR 纠错。"),
}


@lru_cache(maxsize=1)
def library():
    path = ROOT / "build/libfp4_shared_patterns.so"
    if not path.is_file():
        raise RuntimeError("C codecs not built; run python3 scripts/build_shared_patterns.py")
    lib = ctypes.CDLL(str(path))
    pointer = ctypes.POINTER(ctypes.c_uint8)
    lib.fp4_native_encode, lib.fp4_native_decode = lib.fp4_c_encode, lib.fp4_c_decode
    lib.fp4_native_free, lib.fp4_native_error = lib.fp4_c_free, lib.fp4_c_error
    lib.fp4_native_encode.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
                                    ctypes.c_size_t, ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_size_t)]
    lib.fp4_native_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_size_t)]
    lib.fp4_native_free.argtypes = [ctypes.c_void_p]
    lib.fp4_native_free.restype = None
    lib.fp4_native_error.restype = ctypes.c_char_p
    return lib


def build_metadata():
    metadata = json.loads((ROOT / "build/shared-pattern-build.json").read_text())
    source, h = ROOT / "native/shared_patterns", hashlib.sha256()
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(source)).encode()); h.update(path.read_bytes())
    if h.hexdigest() != metadata["source_sha256"]:
        raise RuntimeError("C sources changed; run python3 scripts/build_shared_patterns.py")
    for path, field in (("build/libfp4_shared_patterns.so", "library_sha256"),
                        ("build/libfp4_entropy.so", "a1_library_sha256"),
                        ("native/predictive/common.hpp", "common_header_sha256")):
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != metadata[field]:
            raise RuntimeError("C library/dependency differs from C build metadata; rebuild C")
    metadata["upstream"] = json.loads((source / "third_party/sources.json").read_text())
    metadata["runtime_threads"] = int(os.environ.get("FP4_C_THREADS", "1"))
    return metadata


class NativeC(NativeA):
    _library = staticmethod(library)

    def __init__(self, technique):
        self.codec, self.subcategory, name, source, adaptation = METHODS[technique]
        library()
        self.technique, self.exponent_split = technique, False
        self.backend = "cpp-c-" + technique + "-v1"
        self.spec = MethodSpec(technique, name + " C/C++", "C", "adapted", source,
                               adaptation, self.backend, "arithmetic", "complete_scheme")

    def check_support(self, target, parameters):
        if "unit_bits" in parameters:
            from .grouped_units import support as grouped_support
            return grouped_support(self, target, parameters)
        result = support(target, parameters)
        if not result.applicable:
            return result
        if any(c.length > 4 << 20 for c in target.components):
            return Support(False, "C component limit is 4 MiB; use smaller chunks")
        if self.subcategory == "C3" and any(c.role == "packed_fp4" for c in target.components):
            if target.structure.get("fp4_value_format") != "E2M1FN":
                return Support(False, "C3 numerical training requires explicit fp4_value_format=E2M1FN")
        return result

    def _encode_frame(self, target, component, raw, split):
        lib = self._library()
        return _call(lib.fp4_native_encode, self.codec, FORMATS[component.storage_format], 0, raw, len(raw), _library=lib)


class InterFrameReference(NativeB):
    """Catalog alias: exactly the existing B algorithm and native buffer API."""
    def __init__(self):
        super().__init__("motion_compensation")
        self.spec = replace(self.spec, id="inter_frame_reference", category="C",
                            name="Inter-frame Reference Coding（复用 B motion_compensation）",
                            adaptation=self.spec.adaptation + " C2 关联复用 motion_compensation，无新增算法。")
