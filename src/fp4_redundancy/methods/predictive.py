"""Fixed, complete B plugins. Native cores; shared buffer packaging only."""

import ctypes
from functools import lru_cache
import hashlib
import json

from ..types import MethodSpec, Support
from ._formats import FORMATS as WIDTHS, support
from .native import NativeA, ROOT, FORMATS, _call

# codec id, name, source variant, retained mechanisms and explicit changes
METHODS = {
    "delta": (10, "Delta 码字适配＋算术编码", "first differences",
              "前一码字作参考，模 2^w 精确差分；起始参考为零。"),
    "xor": (11, "XOR 码字适配＋算术编码", "successive-word XOR",
            "前一码字作参考，完整码字 XOR；起始参考为零。"),
    "dpcm_lossless": (12, "二阶 DPCM 无损码字适配＋算术编码", "RFC 9639 fixed order-2 prediction",
                      "固定二阶预测 2*x[-1]-x[-2]；精确模残差，无量化，零边界。"),
    "adpcm_like_fp4": (13, "IMA ADPCM 无损码字适配＋算术编码", "libsndfile 1.2.2 IMA/DVI ADPCM",
                       "保留 IMA 步长自适应、4-bit 决策、饱和和量化重建反馈；另存精确模补偿。"
                       "决策及补偿均算术编码并计费；不是 G.726，不将补偿反馈入原 IMA 状态。"),
    "jpeg_ls_like_fp4": (14, "JPEG-LS 无损码字适配（CharLS）", "CharLS 2.4.2 JPEG-LS, NEAR=0",
                         "完整码字作为 4/8/16-bit 灰度样本，保留上下文修正、run mode、自适应 Golomb 编码；"
                         "原序，按已有物理行布局，超格式尺寸分成独立连续条带；不是浮点数值图像。"),
    "png_filtering": (15, "PNG Filtering 码字适配＋算术编码", "W3C PNG third edition filters 0..4",
                      "保留 None/Sub/Up/Average/Paeth 及 tie 规则，逐行选择；从 byte 扩展为完整码字模 2^w，"
                      "保存行滤波号；只接滤波，不是完整 PNG/DEFLATE。"),
    "fpc_like_fp4": (16, "FPC 无损码字适配（专用编码）", "Burtscher/Ratanaworabhan FPC 1.1",
                     "保留 FCM/DFCM、较小 XOR 残差选择、selector 和专用零字节前缀编码；"
                     "4/8/16-bit 无符号码字，宽度适配 hash，4096 槽/表，安全缓冲区；不是原版 FP64 码流。"),
    "sz_like_lossless": (17, "SZ3 Lorenzo 无损码字适配（Huffman＋Zstd）",
                         "SZ3 db2eca3ea8ae526ee1328649a1caf1a691b45873 first-order Lorenzo; Zstd 1.5.7 level 3",
                         "保留所选 1D/2D Lorenzo 因果预测及上游 Huffman＋Lossless_zstd；"
                         "以整数码字的精确模残差替换误差有界量化，反馈完整恢复码字；"
                         "无浮点舍入/不可预测数据丢失；Huffman 表单列计费，不进入 Zstd；不含自动选择/回归/插值。"),
    "lpc_lossless": (18, "LPC 无损码字适配＋算术编码", "FLAC 59fa3402cb19b6b7c85b916f138cfb03c7b27504 LPC recursion",
                     "保留自相关与 Levinson-Durbin 拟合；每组件前 8192 码字拟合八阶，Q12 系数全存；"
                     "解码用整数预测及精确模残差，系数量化仅影响预测；非 FLAC 码流。"),
    "ltp_lossless": (19, "单抽头 LTP 无损码字适配＋算术编码", "RFC 6716 long-term/pitch prediction mechanism",
                     "每 256 码字选择历史 lag 和 Q12 gain，并完整保存；精确模残差；"
                     "lag 候选 1..128 及 256..4096 的 2 次幂，不是完整 Opus/SILK 五抽头方案。"),
    "motion_compensation": (20, "整数位移 Motion Compensation 无损码字适配＋算术编码",
                            "H.264 integer-sample block matching mechanism",
                            "连续行条带作离线参考帧，块匹配保存位移，引用本组件已解码前帧；"
                            "精确模残差；无跨目标参考、子像素插值、视频变换或量化，不声称时间相关性。"),
    "intra_directional": (21, "四方向 Intra Prediction 无损码字适配＋算术编码",
                         "H.264 spatial directional prediction mechanism",
                         "8x8 块四个整数方向，保存选择，预测仅引用已解码边界；"
                         "精确模残差，原序，不是完整 H.264 九模式/视频码流。"),
}


@lru_cache(maxsize=1)
def library():
    path = ROOT / "build/libfp4_predictive.so"
    if not path.is_file():
        raise RuntimeError("B codecs not built; run python3 scripts/build_predictive.py")
    lib = ctypes.CDLL(str(path))
    pointer = ctypes.POINTER(ctypes.c_uint8)
    lib.fp4_native_encode = lib.fp4_b_encode
    lib.fp4_native_decode = lib.fp4_b_decode
    lib.fp4_native_free = lib.fp4_b_free
    lib.fp4_native_error = lib.fp4_b_error
    lib.fp4_native_encode.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
                                    ctypes.c_size_t, ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_size_t)]
    lib.fp4_native_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_size_t)]
    lib.fp4_native_free.argtypes = [ctypes.c_void_p]
    lib.fp4_native_free.restype = None
    lib.fp4_native_error.restype = ctypes.c_char_p
    return lib


def build_metadata():
    metadata = json.loads((ROOT / "build/predictive-build.json").read_text())
    source = ROOT / "native/predictive"
    h = hashlib.sha256()
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(source)).encode())
        h.update(path.read_bytes())
    if h.hexdigest() != metadata["source_sha256"]:
        raise RuntimeError("B sources changed; run python3 scripts/build_predictive.py")
    for name, field in (("libfp4_predictive.so", "library_sha256"), ("libfp4_entropy.so", "a1_library_sha256")):
        if hashlib.sha256((ROOT / "build" / name).read_bytes()).hexdigest() != metadata[field]:
            raise RuntimeError("B/A1 library differs from B build metadata; rebuild B explicitly")
    metadata["upstream"] = json.loads((source / "third_party/sources.json").read_text())
    return metadata


def component_columns(target, component):
    """Physical rows if known and aligned; otherwise an explicit linear stream."""
    bits = WIDTHS[component.storage_format]
    count = component.length * 8 // bits
    shape = target.structure.get("storage_shapes", {}).get(component.id)
    if shape and len(shape) >= 2:
        # storage_shapes describe packed bytes for FP4 and native scalars otherwise.
        cols = shape[-1] * (2 if bits == 4 else 4 if component.storage_format == "F32" else 1)
        relative = target.structure.get("chunk", {}).get("ranges", {}).get(component.id, {}).get("offset", 0)
        row_bytes = cols * bits // 8
        if cols > 0 and row_bytes > 0 and relative % row_bytes == 0 and count % cols == 0:
            return cols
    return count


class NativeB(NativeA):
    _library = staticmethod(library)

    def __init__(self, technique):
        self.codec, name, source, adaptation = METHODS[technique]
        library()
        self.technique, self.exponent_split = technique, False
        self.backend = "cpp-b-" + technique + "-v1"
        self.spec = MethodSpec(technique, name + " C/C++", "B", "adapted", source,
                               adaptation, self.backend, "arithmetic", "complete_scheme")

    def check_support(self, target, parameters):
        if "unit_bits" in parameters:
            from .grouped_units import support as grouped_support
            return grouped_support(self, target, parameters)
        result = support(target, parameters)
        if result.applicable and any(c.length > 4 << 20 for c in target.components):
            return Support(False, "B component limit is 4 MiB; use smaller chunks")
        return result

    def _encode_frame(self, target, component, raw, split):
        lib = self._library()
        return _call(lib.fp4_native_encode, self.codec, FORMATS[component.storage_format],
                     component_columns(target, component), raw, len(raw), _library=lib)
