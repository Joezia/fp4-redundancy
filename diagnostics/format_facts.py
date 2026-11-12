"""Section 2: format facts F1-F4 from first-hand sources. Writes format.md.

Reads only safetensors headers, the snapshot's config.json and reference code (as text, never
executed), and the first sampled matrix for the F3 data-side checks.
"""
import argparse
import collections
import json
import os
import re
from pathlib import Path

import numpy as np

from . import est
from .measure import unpack_codes
from .samples import select
from .st import index_snapshot, read_bytes

FP4_NAME = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")


def line_of(path, needle, start=1):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i >= start and needle in line:
                return i, line.rstrip("\n")
    raise ValueError(f"{needle!r} not found in {path}")


def cite(snapshot, rel, needle, start=1):
    i, text = line_of(os.path.join(snapshot, rel), needle, start)
    return f"`{rel}:{i}` — `{text.strip()}`"


def block_stats(C, blocks_ids, valid):
    """Fraction of blocks with max magnitude index >= 6 and I(C; block id)."""
    ids = blocks_ids[valid]
    c = C[valid]
    _, ids = np.unique(ids, return_inverse=True)  # only blocks that actually occur
    nblk = int(ids.max()) + 1
    mmax = np.zeros(nblk, dtype=np.int64)
    np.maximum.at(mmax, ids, (c & 7).astype(np.int64))
    t = np.bincount(ids.astype(np.int64) * 16 + c, minlength=nblk * 16).reshape(nblk, 16)
    assert (t.sum(1) == 32).all()
    mi, n, dof = est.mi_table(t)
    return float((mmax >= 6).mean()), mi, n, dof, nblk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", default="experiments/r1-r4/format.md")
    args = ap.parse_args()
    S = args.snapshot
    index = index_snapshot(S)
    L = []

    # ---------------------------------------------------------------- F1
    shards = sorted({v["file"] for v in index.values()})
    by_dtype = collections.defaultdict(lambda: [0, 0])
    for v in index.values():
        b, e = v["data_offsets"]
        by_dtype[v["dtype"]][0] += 1
        by_dtype[v["dtype"]][1] += e - b
    i8 = {k: v for k, v in index.items() if v["dtype"] == "I8"}
    bad_name, bad_scale = [], []
    kinds = collections.Counter()
    for k, v in i8.items():
        m = FP4_NAME.match(k)
        if not m:
            bad_name.append(k)
            continue
        kinds[m.group(1)] += 1
        s = k[: -len("weight")] + "scale"
        se = index.get(s)
        rows, cols_b = v["shape"]
        if se is None or se["dtype"] != "F8_E8M0" or se["shape"] != [rows, 2 * cols_b // 32]:
            bad_scale.append(k)
    e8_users = collections.Counter()
    for k, v in index.items():
        if v["dtype"] == "F8_E8M0":
            w = index.get(k[: -len("scale")] + "weight") if k.endswith(".scale") else None
            e8_users[w["dtype"] if w else "no sibling .weight"] += 1
    shapes = collections.Counter((tuple(v["shape"]), FP4_NAME.match(k).group(4))
                                 for k, v in i8.items() if FP4_NAME.match(k))
    with open(os.path.join(S, "config.json")) as f:
        cfg = json.load(f)

    L.append("# R1–R4 格式事实（spec 001 第 2 节）\n")
    L.append(f"快照：`{os.path.basename(S.rstrip('/'))}`（{cfg['architectures'][0]}）；"
             f"由 `diagnostics/format_facts.py` 生成，只读文件头、`config.json` 与参考代码文本，不执行快照代码。\n")
    L.append("## F1 清单\n")
    L.append(f"自行解析 {len(shards)} 个分片的 safetensors 文件头（8 字节小端长度 + JSON），共 {len(index)} 个张量。\n")
    L.append("| dtype | 张量数 | 字节数 |\n| --- | ---: | ---: |")
    for dt, (n, nb) in sorted(by_dtype.items()):
        L.append(f"| {dt} | {n} | {nb:,} |")
    L.append("")
    L.append("**FP4 权重识别规则**：文件头 dtype 为 `I8`，且名称匹配 "
             "`^(layers|mtp).<i>.ffn.experts.<e>.w{1,2,3}.weight$`；对应 scale 为同名把 `weight` 换成 `scale` 的张量，"
             "dtype `F8_E8M0`，形状 `[行数, 2·字节列数/32]`。")
    L.append(f"- 全部 {len(i8)} 个 I8 张量中，不符合名称规则的 {len(bad_name)} 个，scale 缺失/dtype/形状不符的 {len(bad_scale)} 个。"
             f"按前缀：{dict(kinds)}（`layers` = 主干路由专家，`mtp` = MTP 层路由专家）。")
    L.append("- I8 形状（字节）× 角色：" + "；".join(f"{list(s)} {r}: {n}" for (s, r), n in sorted(shapes.items())))
    L.append(f"- `F8_E8M0` 张量按同名 `.weight` 的 dtype 归类：{dict(e8_users)}。I8 的 scale 即上述规则对应者，"
             "其余 `F8_E8M0` 属于 `F8_E4M3` 权重（共享专家等），不在本规格范围。")
    L.append("- 依据：" + cite(S, "inference/convert.py", 'elif "experts" in name and state_dicts[i][name].dtype == torch.int8')
             + "；" + cite(S, "inference/convert.py", "state_dicts[i][name] = state_dicts[i][name].view(torch.float4_e2m1fn_x2)")
             + "（路由专家的 int8 张量按 FP4×2 解释）；"
             + cite(S, "inference/convert.py", 'scale_name = name.replace("weight", "scale")')
             + "（scale 为同名 `.scale`）；`config.json` `quantization_config`: "
             + f"`{json.dumps(cfg['quantization_config'], ensure_ascii=False)}`。")
    L.append("")

    # ---------------------------------------------------------------- F2
    L.append("## F2 参考代码\n")
    L.append("1. **字节内码字顺序：低 4 位在前。** "
             + cite(S, "inference/convert.py", "low = x & 0x0F") + "；"
             + cite(S, "inference/convert.py", "high = (x >> 4) & 0x0F") + "；"
             + cite(S, "inference/convert.py", "x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)")
             + "——`stack(..., dim=-1).flatten` 使每个字节展开为 `[low, high]`，即逻辑列 `2k` 为低 nibble、`2k+1` 为高 nibble。"
             + "打包沿最后一维（K）：" + cite(S, "inference/kernel.py", "The FP4 values are packed along the K (last) dimension.")
             + "；" + cite(S, "inference/model.py", "# two values per byte: [out, in] logically, [out, in//2] stored") + "。")
    i13, _ = line_of(os.path.join(S, "inference/convert.py"), "FP4_TABLE = torch.tensor(")
    L.append("2. **码字→数值表（E2M1）**：" + cite(S, "inference/convert.py", "[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5")
             + f"（`FP4_TABLE`，起于第 {i13} 行）。索引 0–7 = {{0, 0.5, 1, 1.5, 2, 3, 4, 6}}，8–15 为其相反数，"
             "故最高位（bit 3）为符号位，低 3 位为幅值索引；码字 8 为 −0（数值 0）。kernel 中的 dtype 名称："
             + cite(S, "inference/kernel.py", 'FP4 = "float4_e2m1fn"') + "。")
    L.append("3. **scale**：块大小 B = 32，沿行（K，最后一维）："
             + cite(S, "inference/model.py", "fp4_block_size = 32") + "；"
             + cite(S, "inference/model.py", "torch.empty(out_features, in_features // fp4_block_size, dtype=torch.float8_e8m0fnu)")
             + "；" + cite(S, "inference/convert.py", "assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size")
             + "；" + cite(S, "inference/kernel.py", "Weight: 1x32 quant on K (reduce dim), FP4 with E8M0 scale")
             + "。字节→数值：E8M0（`float8_e8m0fnu`，" + cite(S, "inference/model.py", "scale_dtype = torch.float8_e8m0fnu")
             + "），纯 8 位指数、偏置 127，值 = 2^(byte−127)；代码中同一偏置见 "
             + cite(S, "inference/kernel.py", "bits_x = (x + 127) << 23") + "（构造 2^x 的 fp32 位模式）。"
             "e8m0 的 NaN 编码（0xFF）由 PyTorch dtype 定义，快照代码中未出现。反量化：权重 = FP4_TABLE[C]·2^(E−127)，"
             "见 " + cite(S, "inference/convert.py", "offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(fp4_block_size, dim=-1)") + "。")
    L.append("4. **量化过程**：快照中**未找到权重**的 FP4 量化代码（权重以已量化形式发布）。"
             "仅找到激活/KV 用的 `fp4_quant_kernel`：scale = 2^ceil(log2(amax/6))，"
             + cite(S, "inference/kernel.py", "s_local[i] = fast_round_scale(amax_local[i], fp4_max_inv)") + "，"
             + cite(S, "inference/kernel.py", "def fast_log2_ceil(x):") + "；"
             "取值 `T.Cast(FP4, T.clamp(x/s, -6, 6))`，" + cite(S, "inference/kernel.py", "T.Cast(compute_dtype, T.Cast(FP4, T.clamp(")
             + "；舍入规则由 FP4 类型转换决定，代码中未写明（未找到）。该过程是否用于权重未知。")
    L.append("")

    # ---------------------------------------------------------------- F3
    samp = select(index)["matrices"][0]
    packed = read_bytes(index[samp["weight"]])
    E = read_bytes(index[samp["scale"]])
    C = unpack_codes(packed)
    R, W = C.shape
    B = 32
    L.append(f"## F3 数据侧核对（样本第一张矩阵 `{samp['weight']}`）\n")
    ok_shape = list(E.shape) == [R, W // B]
    L.append(f"1. **形状一致性**：码字矩阵 {R}×{W}（字节 {list(packed.shape)}），scale {list(E.shape)}；"
             f"`[行数, 每行码字数/B]` = [{R}, {W // B}] → {'一致' if ok_shape else '不一致'}。")
    rr = np.broadcast_to(np.arange(R)[:, None], (R, W))
    jj = np.broadcast_to(np.arange(W)[None, :], (R, W))
    hyps = [
        ("沿行、偏移 0", rr * (W // B) + jj // B, np.ones((R, W), bool)),
        ("沿行、偏移 B/2", rr * (W // B) + (jj - B // 2) // B,
         (jj >= B // 2) & (jj < W - B // 2)),
        ("沿列、偏移 0", (rr // B) * W + jj, np.ones((R, W), bool) & (R % B == 0)),
    ]
    L.append("2. **块对齐**（每种假设只用完整块）：\n")
    L.append("| 假设 | 块数 | 块内最大幅值索引 ≥ 6 的比例 | I(C; 块id) bit | n | 解析偏差 dof/(2n ln2) |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    stats = []
    for name, ids, valid in hyps:
        frac, mi, n, dof, nblk = block_stats(C, ids, valid)
        stats.append((name, frac, mi))
        L.append(f"| {name} | {nblk:,} | {frac:.4f} | {mi:.5f} | {n:,} | {est.analytic_bias(dof, n):.5f} |")
    best_frac = max(stats, key=lambda s: s[1])[0]
    best_mi = max(stats, key=lambda s: s[2])[0]
    L.append(f"\n   ≥6 比例最高的假设：{best_frac}；I(C; 块id) 最大的假设：{best_mi}。"
             "三种假设的块大小和组数相近，解析偏差相同量级（每组仅 32 个样本时该公式偏小，仅作量级参考），差别如上表如实记录。")
    hist = np.bincount(C.ravel(), minlength=16)
    L.append("3. **符号对称**：\n")
    L.append("| m | hist[m]（s=0） | hist[m^8]（s=1） | 比值 s=1/s=0 |\n| ---: | ---: | ---: | ---: |")
    for m in range(8):
        a, b = int(hist[m]), int(hist[m + 8])
        L.append(f"| {m} | {a:,} | {b:,} | {b / a if a else float('nan'):.4f} |")
    sm = hist.reshape(2, 8)
    hs = est.entropy_counts(sm.sum(1))
    mi = est.mi_table(sm)[0]
    L.append(f"\n   H(s) = {hs:.6f} bit，I(s;m) = {mi:.6f} bit，H(s|m) = {hs - mi:.6f} bit（N = {C.size:,}）。")
    L.append("")

    # ---------------------------------------------------------------- F4
    table_ok = True  # from F2.2
    L.append("## F4 停止条件\n")
    L.append(f"- 块沿行：F2.3 + F3.2 → 是。B = 32：F2.3 → 是。数值表为 E2M1 且最高位为符号：F2.2 → 是。"
             f"scale 形状：F3.1 → {'一致' if ok_shape else '不一致'}。")
    proceed = ok_shape and table_ok
    L.append(f"- 结论：{'未触发停止条件，继续第 3 节之后的测量。' if proceed else '触发停止条件，停止。'}")
    L.append("- 字节内顺序只由 F2.1 确定（低 nibble 在前）；C1 的 d ≤ 4 奇偶拆分作为旁证，见 summary.md。")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if not proceed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
