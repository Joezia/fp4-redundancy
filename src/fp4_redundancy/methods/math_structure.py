"""D complete plugins: exact transforms or numerical models with exact codeword correction."""
from functools import lru_cache
import hashlib
import json
import os

from ..types import MethodSpec, Support
from ._formats import support
from .native import NativeA, ROOT, FORMATS, _call, _buffer_library

METHODS = {
    "bwt_mtf": (60, "D1", "BWT＋MTF 完整码字可逆编码", "Burrows/Wheeler 1994 BWT; move-to-front",
                "64K 符号块实际循环排序、保存 primary、Fenwick MTF 正逆；A1 算术编码秩；完整恢复原序与 packed 字节。"),
    "dct_lossless": (61, "D1", "DCT-II 数值变换无损适配", "orthonormal DCT-II; FFTW REDFT10 definition",
                     "八维连续向量；保存 Q20 基矩阵与变换系数，用实际整数逆向路径生成完整码字 XOR 纠错；不称原生整数 DCT。"),
    "dst_lossless": (62, "D1", "DST-I 数值变换无损适配", "orthonormal DST-I; FFTW RODFT00 definition",
                     "八维正交 DST-I，保存 Q20 基矩阵/系数及完整码字 XOR 纠错。"),
    "mdct_lossless": (63, "D1", "MDCT 窗与重叠重建无损适配", "Princen/Bradley TDAC; checked Xiph MDCT/window specification",
                      "2N=16、hop=8、正弦窗；目标内首尾零扩展与 priming，保存窗/cosine 参数，完整 IMDCT/overlap-add 后精确纠错；非 Vorbis 码流。"),
    "cdf53": (64, "D1", "可逆 CDF 5/3 整数 lifting", "JPEG 2000 reversible 5/3; OpenJPEG dwt.c",
              "256 完整码字块，多级整数 predict/update，向负无穷取整、对称边界；int64 中间量及 int32 系数，越界明确失败；无纠错流。"),
    "pca_lossless": (65, "D1", "KLT/PCA 数值变换无损适配", "centered covariance eigendecomposition, Eigen SelfAdjointEigenSolver",
                     "前至多512个八维向量估计均值/完整八维基，全部保存 Q20；保存所有系数与实际逆向重建的码字纠错，不重做分解。"),
    "haar_lifting": (66, "D1", "Haar S-transform 可逆整数 lifting", "Calderbank/Daubechies/Sweldens/Yeo integer Haar S transform",
                     "256 码字块，多级 d=b-a、s=a+floor(d/2)，单独尾值直通；int64 运算/int32 存储；与 CDF 5/3 不同的明确方案。"),
    "bitplane": (67, "D1", "Bit-plane 内部变换与算术编码", "MSB-first bit-plane coding over complete native codewords",
                 "仅本插件将完整码字映射为 MSB→LSB 平面，各平面二元零阶 A1 算术编码，独立计费并逐码字拼回；不是全局字段预处理。"),
    "sparse_coding": (68, "D2", "OMP Sparse Coding 无损适配", "orthogonal matching pursuit, identity/DCT overcomplete dictionary",
                      "8×16 字典、最多3个非零系数；真实相关性选择与最小二乘系数求解；保存字典/位置/Q20系数/精确纠错。"),
    "ksvd_lossless": (69, "D2", "K-SVD 字典学习无损适配", "Aharon/Elad/Bruckstein IEEE TSP 2006 K-SVD",
                      "前至多512八维向量、16原子、稀疏度3、四轮 OMP 与受支持残差的逐原子 rank-1 SVD 更新；保存最终字典及全部必要数据与纠错，非聚类。"),
    "svd_lowrank": (70, "D3", "SVD rank-2 低秩块无损适配", "truncated SVD / Eckart-Young low-rank approximation",
                    "每连续64码字按C序8×8，保留rank2，保存两因子；用最终Q20因子重建及精确纠错；SVD与低秩近似共享此明确配置。"),
    "cp_lossless": (71, "D3", "CP-ALS 张量块无损适配", "Kolda/Bader 2009 CP decomposition / alternating least squares",
                    "连续64值C序4×4×4，CP rank2，20轮ALS、SVD最小二乘与列尺度规整；保存三个4×2因子和精确纠错。"),
    "tucker_lossless": (72, "D3", "Tucker HOSVD 张量块无损适配", "Kolda/Bader 2009 Tucker/HOSVD",
                        "C序4×4×4，multilinear ranks(2,2,2)，真实逐模展开/SVD与核张量投影；保存三因子、2×2×2核心及纠错。"),
    "tt_lossless": (73, "D3", "Tensor Train TT-SVD 无损适配", "Oseledets 2011 TT-SVD",
                    "C序4×4×4，顺序展开/截断SVD，TT ranks(1,2,2,1)；保存1×4×2、2×4×2、2×4×1核心与纠错。"),
    "kronecker_lossless": (74, "D3", "Kronecker 最近乘积块无损适配", "Van Loan/Pitsianis nearest Kronecker product via rearranged SVD",
                           "8×8块重排为4×16拟合矩阵，其rank-1 SVD得到2×2与4×4因子；保存因子和实际重建纠错。"),
    "circulant_lossless": (75, "D3", "Circulant 循环矩阵拟合无损适配", "least-squares projection onto circulant matrices",
                           "8×8块逐循环对角线均值拟合8参数，C[i,j]=c[(j-i) mod8]；保存参数与纠错，不要求源块恰好循环。"),
}
EXACT = frozenset(("bwt_mtf", "cdf53", "haar_lifting", "bitplane"))


@lru_cache(maxsize=1)
def library():
    path = ROOT / "build/libfp4_math_structure.so"
    if not path.is_file():
        raise RuntimeError("D codecs not built; run python3 scripts/build_math_structure.py")
    return _buffer_library(path, "fp4_d")


def build_metadata():
    metadata = json.loads((ROOT / "build/math-structure-build.json").read_text())
    source, h = ROOT / "native/math_structure", hashlib.sha256()
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(source)).encode()); h.update(path.read_bytes())
    if h.hexdigest() != metadata["source_sha256"]:
        raise RuntimeError("D sources changed; run python3 scripts/build_math_structure.py")
    for path, field in (("build/libfp4_math_structure.so", "library_sha256"), ("build/libfp4_entropy.so", "a1_library_sha256")):
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != metadata[field]:
            raise RuntimeError("D binary differs from build metadata; rebuild D")
    for path, digest in metadata["dependencies_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("D header dependency changed; rebuild D")
    metadata["upstream"] = json.loads((source / "sources.json").read_text())
    metadata["runtime_threads"] = int(os.environ.get("FP4_D_THREADS", "1"))
    return metadata


class NativeD(NativeA):
    _library = staticmethod(library)

    def __init__(self, technique):
        self.codec, self.subcategory, name, source, adaptation = METHODS[technique]
        library()
        self.technique, self.exponent_split = technique, False
        self.backend = "cpp-d-" + technique + "-v1"
        self.spec = MethodSpec(technique, name + " C/C++", "D", "native" if technique in EXACT else "adapted",
                               source, adaptation, self.backend, "arithmetic", "complete_scheme")

    def check_support(self, target, parameters):
        if "unit_bits" in parameters:
            from .grouped_units import support as grouped_support
            return grouped_support(self, target, parameters)
        result = support(target, parameters)
        if not result.applicable:
            return result
        if any(c.length > 1 << 20 for c in target.components):
            return Support(False, "D component limit is 1 MiB; use smaller chunks")
        if self.technique not in EXACT and any(c.role == "packed_fp4" for c in target.components):
            if target.structure.get("fp4_value_format") != "E2M1FN":
                return Support(False, "D numerical method requires explicit fp4_value_format=E2M1FN")
        return result

    def _encode_frame(self, target, component, raw, split):
        lib = self._library()
        return _call(lib.fp4_native_encode, self.codec, FORMATS[component.storage_format], 0, raw, len(raw), _library=lib)
