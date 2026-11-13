"""Spec 003: aggregate the per-job pickles of r58_run into experiments/003-r5-r8/.

Usage: python -m diagnostics.r58_report --work <dir> [--out experiments/003-r5-r8]
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import r58

FIELDS = r58.FIELDS
EX_FIELDS = ["matrix", "layer", "expert", "role", "item", "kind", "row", "col", "length",
             "other_matrix", "other_row", "other_col", "all_zero", "content"]
ROW002 = 0.569  # 002 ledger row 4 saving, bit/weight (median of 36 matrices)
ABSV = "\\|v\\|"  # |v| escaped for Markdown table cells
PLOT_MATS = ["L0.E164.w1", "L13.E135.w2", "L39.E183.w3"]


def fnum(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.8g}"
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fnum(r.get(k, "")) for k in fields})


def load(work):
    import pickle
    mats, pairs = [], []
    for p in sorted(Path(work).glob("m*.pkl")):
        with open(p, "rb") as f:
            mats.append(pickle.load(f))
    for p in sorted(Path(work).glob("p*.pkl")):
        with open(p, "rb") as f:
            pairs.append(pickle.load(f))
    return mats, pairs


# ----------------------------------------------------------------------------- pooled R6.1

def pooled_r61(mats):
    rows, ex = [], []
    meta = {"matrix": "pooled", "layer": "", "expert": "", "role": "", "matrix_b": "",
            "pair_group": ""}
    keys = np.concatenate([m["pool"]["block_keys"] for m in mats])
    nt = np.concatenate([m["pool"]["block_nontrivial"] for m in mats])
    owner = np.concatenate([np.full(len(m["pool"]["block_keys"]), i) for i, m in enumerate(mats)])
    local = np.concatenate([np.arange(len(m["pool"]["block_keys"])) for m in mats])
    d_all = r58.distinct_rows(keys)[0]
    knt, ont, lnt = keys[nt], owner[nt], local[nt]
    d_nt, same, o = r58.distinct_rows(knt)
    rep_all, rep_nt = len(keys) - d_all, len(knt) - d_nt
    # cross-matrix non-trivial repeats: runs that span more than one matrix
    so = ont[o]
    start = np.r_[True, ~same]
    run = np.cumsum(start) - 1
    first_owner = so[start][run]
    cross = np.r_[False, same] & (so != first_owner)
    within_sum = sum(m["vals"]["r61"]["nontrivial_repeats"] for m in mats)
    for var, val in (("block_repeats", rep_all), ("nontrivial_block_repeats", rep_nt),
                     ("nontrivial_block_repeats_minus_within_matrix", rep_nt - within_sum),
                     ("nontrivial_block_repeat_windows_owned_by_other_matrix", int(cross.sum()))):
        rows.append(dict(meta, item="R6.1", variable=var, group_or_context="block",
                         subset="pooled 36 matrices", n=len(keys), value=val))
    for var, key in (("row_repeats", "row_hashes"), ("scale_row_repeats", "scale_row_hashes")):
        allh = [h for m in mats for h in m["pool"][key]]
        rows.append(dict(meta, item="R6.1", variable=var, group_or_context=var.split("_repeats")[0],
                         subset="pooled 36 matrices", n=len(allh), value=len(allh) - len(set(allh))))
    for var, key in (("matrix_repeats", "matrix_hash"), ("scale_matrix_repeats", "scale_hash")):
        allh = [m["pool"][key] for m in mats]
        rows.append(dict(meta, item="R6.1", variable=var, group_or_context=var.split("_repeats")[0],
                         subset="pooled 36 matrices", n=len(allh), value=len(allh) - len(set(allh))))
    ci = np.nonzero(cross)[0][:r58.MAX_EXAMPLES]
    sl, sown = lnt[o], so
    for i in ci:
        fo, fl = first_owner[i], sl[start][run][i]
        ma, mb = mats[fo]["meta"], mats[sown[i]]["meta"]
        nb = mats[sown[i]]["pool"]["block_keys"].shape[0] // len(mats[sown[i]]["pool"]["row_hashes"])
        rb, bb = divmod(int(sl[i]), nb)
        ra, ba = divmod(int(fl), nb)
        k = knt[o][i]
        content = "".join("%x" % ((int(k[t // 16]) >> (4 * (t % 16))) & 15) for t in range(32))
        ex.append(dict(mb, item="R6.1", kind="block (cross-matrix)", row=rb, col=bb * 32, length=32,
                       other_matrix=ma["matrix"], other_row=ra, other_col=ba * 32, all_zero=False,
                       content=content))
    return rows, ex, {"rep_all": rep_all, "rep_nt": rep_nt, "cross": int(cross.sum())}


# ----------------------------------------------------------------------------- helpers

def st(x):
    x = np.array([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if x.size == 0:
        return None
    return float(np.median(x)), float(x.min()), float(x.max())


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if v != 0 and (abs(v) < 10 ** (-nd) or abs(v) >= 1e5):
        return f"{v:.2e}"
    return f"{v:.{nd}f}"


def mmm(x, nd=4):
    s = st(x)
    return "— | — | —" if s is None else " | ".join(fmt(t, nd) for t in s)


def by_group(metas, xs, nd=4):
    layers = sorted({m["layer"] for m in metas})
    roles = ["w1", "w2", "w3"]
    cells = []
    for l in layers:
        s = st([x for m, x in zip(metas, xs) if m["layer"] == l])
        cells.append(fmt(s[0], nd) if s else "—")
    for ro in roles:
        s = st([x for m, x in zip(metas, xs) if m["role"] == ro])
        cells.append(fmt(s[0], nd) if s else "—")
    return cells, layers


def group_table(metas, items, nd=4):
    _, layers = by_group(metas, [0] * len(metas))
    out = ["| 量 | " + " | ".join(f"L{l}" for l in layers) + " | w1 | w2 | w3 |",
           "| --- |" + " ---: |" * (len(layers) + 3)]
    for label, xs in items:
        cells, _ = by_group(metas, xs, nd)
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return out


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


# ----------------------------------------------------------------------------- plots

def plots(mats, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = [m for name in PLOT_MATS for m in mats if m["meta"]["matrix"] == name] or mats[:3]
    doms = sorted({s["domain"] for m in pick for s in m["spectra"]}, key=["v", "w"].index)
    fig, axes = plt.subplots(len(doms), len(pick), figsize=(5 * len(pick), 3.6 * len(doms)),
                             squeeze=False)
    for j, m in enumerate(pick):
        for i, dom in enumerate(doms):
            ax = axes[i, j]
            for s in m["spectra"]:
                if s["domain"] != dom or s["variant"] == "S0":
                    continue
                style = dict(color="C0", lw=1.2, label="real") if s["variant"] == "real" else \
                    dict(color="C3", lw=0.8, alpha=0.7, label=f"S4 rep{s['rep']}")
                ax.plot(np.arange(1, len(s["sigma_norm"]) + 1), s["sigma_norm"], **style)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(f"{m['meta']['matrix']} ({dom} domain)")
            ax.set_xlabel("layer index")
            ax.set_ylabel("normalised singular value")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "spectrum.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, dom in zip(axes, ("v", "absv")):
        for name, color in (("real", "C0"), ("S0E", "C2"), ("S2", "C3")):
            curves = []
            for m in mats:
                r72 = m["vals"]["r72r82"]["r72"]
                if name == "real":
                    curves.append([r72["real"][dom][K]["ratio"] for K in r58.KM_K])
                else:
                    curves.append([np.mean([r72[f"{name}_{i}"][dom][K]["ratio"] for i in (1, 2)])
                                   for K in r58.KM_K])
            c = np.array(curves)
            ax.plot(r58.KM_K, np.median(c, axis=0), color=color, marker="o", ms=3,
                    label=f"{name} (median of {len(c)})")
            ax.fill_between(r58.KM_K, c.min(axis=0), c.max(axis=0), color=color, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K")
        ax.set_ylabel("distortion ratio (evaluation half)")
        ax.set_title(f"k-means, {'|v|' if dom == 'absv' else 'v'} domain, main scale group")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "kmeans_curve.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, 34) - 0.5
    for name, color in (("real", "C0"), ("S0E", "C2"), ("S2", "C3")):
        b = np.concatenate([m["vals"]["r71"]["best"][name][m["vals"]["r71"]["ncand"] > 0]
                            for m in mats])
        ax.hist(b, bins=bins, histtype="step", color=color, lw=1.4, label=f"{name} (n={len(b)})")
    ax.set_xlabel("best = max equal positions with a same-scale block (of 32)")
    ax.set_ylabel("queries")
    ax.set_title(f"R7.1 nearest block, {len(mats)} matrices x {r58.N_QUERY} queries")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "nn_match.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- summary

def summary(mats, pairs, prow, pool, out, extra):
    L = []
    w = L.append
    mm = [m["meta"] for m in mats]
    V = [m["vals"] for m in mats]
    doms = sorted(V[0]["r5"].keys(), key=["v", "w"].index)
    w("# 003 R5–R8 规律检测：结果摘要\n")
    w(f"样本：001 的 {len(mats)} 张矩阵与 {len(pairs)} 对配对（P1 同层同角色不同专家 "
      f"{sum(p['meta']['group'] == 'P1' for p in pairs)} 对，P2 同专家 w1/w3 "
      f"{sum(p['meta']['group'] == 'P2' for p in pairs)} 对，P3 相邻抽样层对照 "
      f"{sum(p['meta']['group'] == 'P3' for p in pairs)} 对）。全部测量在合并零后的码字 C' 上。"
      "“严格”= 留出条件熵或互信息（bit/权重，可与 002 的账相加）；“近似”= 高斯换算，只作上限参考。"
      "表中三列为中位数 | 最小 | 最大。\n")

    conclusions(mats, pairs, V, doms, w)

    # ---------------- R6
    w("## R6 精确重复\n")
    r61 = [v["r61"] for v in V]
    w("| 量（每张矩阵） | 中位数 | 最小 | 最大 |")
    w("| --- | ---: | ---: | ---: |")
    for lab, key in (("块重复数", "block_repeats"), ("非平凡块重复数（>2 种码字）", "nontrivial_repeats"),
                     ("全零块", "all_zero_blocks"), ("≤2 种码字的块", "le2_blocks"),
                     ("块重复数 S0", "S0"), ("块重复数 S2", "S2"),
                     ("非平凡块重复数 S2", "S2_nt"), ("行重复数", "row_repeats"),
                     ("scale 行重复数", "scale_row_repeats")):
        w(f"| {lab} | {mmm([x[key] for x in r61], 0)} |")
    w("")
    w(f"{len(mats)} 张合并：块重复 {pool['rep_all']}，非平凡块重复 {pool['rep_nt']}，其中跨矩阵 {pool['cross']}；"
      + "，".join(f"{r['variable']} = {r['value']}" for r in prow if r["variable"] in
                  ("row_repeats", "scale_row_repeats", "matrix_repeats", "scale_matrix_repeats"))
      + "。\n")
    r62 = [v["r62"] for v in V]
    w("R6.2 行内不对齐窗口 `dup(k)/窗口数`：\n")
    w("| k | 真实 中位数 | S2 中位数 | S0 中位数 | 真实 − S2 个数：中位数 | 最小 | 最大 |")
    w("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for k in r58.KS_R62:
        real = [g["dup"][k] / g["nwin"][k] for g in r62]
        s2 = [g["dup_S2"][k] / g["nwin"][k] for g in r62]
        s0 = [g["dup_S0"][k] / g["nwin"][k] for g in r62] if k in r62[0]["dup_S0"] else None
        net = [g["dup"][k] - g["dup_S2"][k] for g in r62]
        w(f"| {k} | {fmt(st(real)[0], 6)} | {fmt(st(s2)[0], 6)} | "
          f"{fmt(st(s0)[0], 6) if s0 else '—'} | {mmm(net, 0)} |")
    ks = [g["Kstar"] for g in r62]
    ks2 = [g["Kstar_S2"] for g in r62]
    w("")
    w(f"K*（dup(k) > 0 的最大 k）：真实 {mmm([k or 0 for k in ks], 0)}；S2 {mmm([k or 0 for k in ks2], 0)}"
      "（0 表示 k = 8 起已无重复）。近似覆盖率上限 `(dup(16) − S2)/N · H(C'|E)`："
      f"{mmm([g['approx'] for g in r62], 6)} bit/权重。")
    net16 = [g["dup"][16] - g["dup_S2"][16] for g in r62]
    n_big = sum(1 for g in r62 if any(g["dup"][k] > g["dup_S2"][k] for k in r58.KS_R62 if k >= 16))
    w(f"k ≥ 16 有净重复（真实 > S2）的矩阵：{n_big}/{len(r62)}。\n")
    r63 = [(p["meta"]["group"], p["vals"]["r63"]) for p in pairs]
    w("R6.3 跨矩阵窗口（B 的窗口也出现在 A 中的个数）：\n")
    w("| k | 组 | 真实 中位数 | 最大 | S2(B) 中位数 | 最大 |")
    w("| ---: | --- | ---: | ---: | ---: | ---: |")
    for k in r58.KS_R63:
        for g in ("P1", "P2", "P3"):
            a = [x[k][0] for gg, x in r63 if gg == g]
            b = [x[k][1] for gg, x in r63 if gg == g]
            if a:
                w(f"| {k} | {g} | {fmt(float(np.median(a)), 1)} | {max(a)} | "
                  f"{fmt(float(np.median(b)), 1)} | {max(b)} |")
    w("")
    w(f"例子见 `r6_examples.csv`（{extra['n_ex6']} 条）。\n")

    # ---------------- RX
    w("## RX 跨矩阵同位置\n")
    w("| 量 | 基线 | 组 | 真实 中位数 | 净值 中位数 | 最小 | 最大 | 显著对数 |")
    w("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    rxrows = [r for p in pairs for r in p["rows"] if r["item"] in ("RX.1", "RX.2", "RX.3")]
    for item, var in (("RX.1", "I(C'_A;C'_B)"), ("RX.1", "I(m_A;m_B)"), ("RX.1", "I(s_A;s_B)"),
                      ("RX.2", "corr_v"), ("RX.2", "corr_absv"), ("RX.2", "corr_w"),
                      ("RX.3", "I(E_A;E_B)")):
        bases = sorted({r["baseline"] for r in rxrows if r["variable"] == var})
        for b in bases:
            for g in ("P1", "P2", "P3"):
                rs = [r for r in rxrows if r["variable"] == var and r["baseline"] == b
                      and r["pair_group"] == g]
                if not rs:
                    continue
                nsig = sum(1 for r in rs if r["significant"] is True)
                w(f"| {var} | {b} | {g} | {fmt(st([r['value'] for r in rs])[0], 6)} | "
                  f"{mmm([r['net'] for r in rs], 6)} | {nsig}/{len(rs)} |")
    w("")
    w("`I(C'_A; C'_B)` 相对 S3 的净值是严格数（已知 A 时编码 B 可省的 bit/权重）。"
      "RX.2 显著阈值 5/√N；RX.3 单位为 bit/scale。\n")

    # ---------------- R5
    w("## R5 低秩\n")
    w("| 量 | 域 | 中位数 | 最小 | 最大 |")
    w("| --- | --- | ---: | ---: | ---: |")
    for dom in doms:
        r5 = [v["r5"][dom] for v in V]
        w(f"| 超出层数（σ > 1.02 × S4 最大σ） | {dom} | {mmm([x['exceed'] for x in r5], 0)} |")
        w(f"| 超出能量（份额 − S4 同层数份额） | {dom} | {mmm([x['exceed_energy'] for x in r5], 5)} |")
        for k in (1, 4, 16, 64):
            w(f"| 前 {k} 层能量份额（真实） | {dom} | "
              f"{mmm([_share(m, dom, 'real', k) for m in mats], 5)} |")
            w(f"| 前 {k} 层能量份额（S4 rep1） | {dom} | "
              f"{mmm([_share(m, dom, 'S4', k) for m in mats], 5)} |")
        w(f"| 留出 R² 最大值 | {dom} | {mmm([max(x['r2'].values()) for x in r5], 5)} |")
        w(f"| 留出 R² 最大值（S4） | {dom} | {mmm([max(x['r2_S4'].values()) for x in r5], 5)} |")
        ran = [x["r53"] for x in r5 if x["r53"] is not None]
        w(f"| R5.3 执行的矩阵数 | {dom} | {len(ran)} | | |")
        if ran:
            w(f"| R5.3 留出收益（严格） | {dom} | {mmm([x['gain'] for x in ran], 5)} |")
            w(f"| R5.3 净值 = 收益 − 因子成本（严格） | {dom} | {mmm([x['net'] for x in ran], 5)} |")
            w(f"| R5.3 所用 k | {dom} | {mmm([x['k'] for x in ran], 0)} |")
            w(f"| 近似 −½log₂(1−R²) | {dom} | {mmm([x['approx'] for x in ran], 5)} |")
    w("")
    for dom in doms:
        w(f"留出 R²(k)（{dom} 域，中位数）：" + "，".join(
            f"k={k}: {fmt(st([v['r5'][dom]['r2'][k] for v in V])[0], 5)}"
            f"（S4 {fmt(st([v['r5'][dom]['r2_S4'][k] for v in V])[0], 5)}）" for k in r58.R52_K) + "。\n")

    # ---------------- R7
    w("## R7 近似重复\n")
    r71 = [v["r71"] for v in V]
    w("R7.1 同 scale 块的最近邻（每张矩阵 500 个查询块）：\n")
    w("| 量 | 真实 | S0E | S2 |")
    w("| --- | --- | --- | --- |")
    for lab, key in (("best 中位数", "median"), ("best ≥ 18 的比例", "frac18"),
                     ("best > S0E 99% + 2 的比例", "frac_gt"), ("best ≥ m*（回本线）的比例", "frac_m")):
        w(f"| {lab} | " + " | ".join(mmm([x[n][key] for x in r71], 4).replace(" | ", " / ")
                                     for n in ("real", "S0E", "S2")) + " |")
    w("")
    w("（每格为 36 张矩阵的 中位数 / 最小 / 最大。）\n")
    w("R7.2 主 scale 组块的 k-means（失真比：评估集到最近原型 ÷ 到拟合均值；净值 = 留出收益 − log₂K/32）：\n")
    w("| 域 | K | 失真比 真实 | S0E | S2 | “类”净值 真实 | S2 | “类, p”净值 真实 | S2 |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for dom in ("v", "absv"):
        for K in r58.KM_K:
            def g(name, var):
                xs = []
                for v in V:
                    r72 = v["r72r82"]["r72"]
                    if name == "real":
                        xs.append(r72["real"][dom][K].get(var))
                    else:
                        xs.append(np.mean([r72[f"{name}_{i}"][dom][K][var] for i in (1, 2)])
                                  if var in r72[f"{name}_1"][dom][K] else None)
                s = st(xs)
                return fmt(s[0], 4) if s else "—"
            w(f"| {ABSV if dom == 'absv' else 'v'} | {K} | {g('real', 'ratio')} | {g('S0E', 'ratio')} | "
              f"{g('S2', 'ratio')} | {g('real', 'net_cls')} | {g('S2', 'net_cls')} | "
              f"{g('real', 'net_clsp')} | {g('S2', 'net_clsp')} |")
    w("")
    w("R7.3 行/列与最相似行/列的最大绝对相关：\n")
    w("| 范围 | 域 | 轴 | maxcorr 中位数 | 99% | 最大 | S4 最大 | 超过 1.2×S4 最大的个数 | \\|corr\\| > 0.5 的对数 |")
    w("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for scope, items in [("矩阵内", [(m["meta"], m["vals"]["r73"]) for m in mats])] + \
            [(f"配对 {g}", [(p["meta"], p["vals"]["r73"]) for p in pairs if p["meta"]["group"] == g])
             for g in ("P1", "P2", "P3")]:
        for dom in doms:
            for axis in ("row", "col"):
                xs = [x[dom][axis] for _, x in items]
                if not xs:
                    continue
                w(f"| {scope} | {dom} | {axis} | {fmt(np.median([np.median(x['best']) for x in xs]))} | "
                  f"{fmt(np.median([np.percentile(x['best'], 99) for x in xs]))} | "
                  f"{fmt(max(float(x['best'].max()) for x in xs))} | "
                  f"{fmt(max(x['s4max'] for x in xs))} | {sum(x['n_exceed'] for x in xs)} | "
                  f"{sum(x['count'] for x in xs)} |")
    w("")
    w("（maxcorr 中位数、99% 两列为各单位该统计量的中位数；最大、S4 最大取全部单位的最大值；最后两列为全部单位之和。例子见 `r7_rows.csv`。）\n")

    # ---------------- R8
    w("## R8 稀疏\n")
    w("| 量 | 域 | 中位数 | 最小 | 最大 |")
    w("| --- | --- | ---: | ---: | ---: |")
    for dom in doms:
        r81 = [v["r5"][dom]["r81"] for v in V]
        for side in ("left", "right"):
            w(f"| 峰度中位数（{side}） | {dom} | {mmm([x[f'{side}_median'] for x in r81], 3)} |")
            w(f"| 峰度最大（{side}） | {dom} | {mmm([x[f'{side}_max'] for x in r81], 3)} |")
            w(f"| S4 峰度最大（{side}） | {dom} | {mmm([x[f'{side}_s4_max'] for x in r81], 3)} |")
            w(f"| 超标层数，规格规则 2×S4 最大（{side}） | {dom} | "
              f"{mmm([x[f'{side}_exceed_spec'] for x in r81], 0)} |")
            w(f"| 超标层数，调整规则 2×S4 中位数（{side}） | {dom} | "
              f"{mmm([x[f'{side}_exceed'] for x in r81], 0)} |")
    r82 = [v["r72r82"]["r82"] for v in V]
    for s in r58.DICT_S:
        real = [x["real"]["dict"][s] for x in r82]
        s0e = [np.mean([x[f"S0E_{i}"]["dict"][s] for i in (1, 2)]) for x in r82]
        s2 = [np.mean([x[f"S2_{i}"]["dict"][s] for i in (1, 2)]) for x in r82]
        pca = [x["real"]["pca"][s] for x in r82]
        w(f"| R²_dict({s}) 真实 | v | {mmm(real)} |")
        w(f"| R²_dict({s}) − S0E | v | {mmm([a - b for a, b in zip(real, s0e)])} |")
        w(f"| R²_dict({s}) S2 − S0E | v | {mmm([a - b for a, b in zip(s2, s0e)])} |")
        w(f"| R²_pca({s}) 真实 | v | {mmm(pca)} |")
    nsig = defaultdict(int)
    for m in mats:
        for r in m["rows"]:
            if r["item"] == "R8.2" and r["variable"] == "R2_dict" and r["baseline"] == "S0E" \
                    and r["significant"] is True:
                nsig[r["k"]] += 1
    w("")
    w("R8.2 显著（真实 − S0E > 0.01 且 > 5 × |S0E 两次之差|）的矩阵数：" +
      "，".join(f"s={s}: {nsig.get(s, 0)}/{len(mats)}" for s in r58.DICT_S) + "。\n")

    # ---------------- consistency
    w("## 按层、按角色的中位数\n")
    items = []
    for dom in doms:
        items.append((f"R5 超出层数（{dom}）", [v["r5"][dom]["exceed"] for v in V]))
        items.append((f"R5.2 最大 R²（{dom}）", [max(v["r5"][dom]["r2"].values()) for v in V]))
        items.append((f"R8.1 左峰度最大（{dom}）", [v["r5"][dom]["r81"]["left_max"] for v in V]))
        items.append((f"R8.1 右峰度最大（{dom}）", [v["r5"][dom]["r81"]["right_max"] for v in V]))
    items.append(("R7.1 best 中位数 真实 − S0E", [v["r71"]["real"]["median"] - v["r71"]["S0E"]["median"]
                                               for v in V]))
    items.append(("R7.2 v K=64 “类, p”净值", [v["r72r82"]["r72"]["real"]["v"][64]["net_clsp"] for v in V]))
    items.append(("R7.2 \\|v\\| K=64 “类”净值", [v["r72r82"]["r72"]["real"]["absv"][64]["net_cls"] for v in V]))
    items.append(("R8.2 R²_dict(2) − S0E", [x["real"]["dict"][2] - np.mean([x[f"S0E_{i}"]["dict"][2]
                                                                          for i in (1, 2)]) for x in r82]))
    items.append(("R6.2 净重复 dup(16)", net16))
    for dom in doms:
        for axis, lab in (("row", "行"), ("col", "列")):
            items.append((f"R7.3 矩阵内超过 1.2×S4 最大的{lab}占比（{dom}）",
                          [v["r73"][dom][axis]["n_exceed"] / len(v["r73"][dom][axis]["best"]) for v in V]))
    L.extend(group_table(mm, items))
    w("")
    pm = [p["meta"] for p in pairs]
    pa = [{"layer": m["layer"], "role": m["role"]} for m in
          [next(x["meta"] for x in mats if x["meta"]["matrix"] == p["a"]) for p in pm]]
    rx3 = {(p["meta"]["a"], p["meta"]["b"]): r for p in pairs for r in p["rows"]
           if r["variable"] == "I(C'_A;C'_B)" and r["baseline"] == "S3"}
    w("RX.1 `I(C'_A;C'_B)` 相对 S3 的净值，按 A 的层、角色分组的中位数（全部 75 对）：\n")
    L.extend(group_table(pa, [("净值（bit/权重）", [rx3[(p["a"], p["b"])]["net"] for p in pm])], 6))
    w("")

    # ---------------- increments
    w(f"## 相对 002 第 4 行（节省 {ROW002} bit/权重）的增量\n")
    w("只列严格净值中位数为正、且在至少一半单位上显著（或按该项定义为正）的项。\n")
    inc = []
    for g in ("P1", "P2", "P3"):
        rs = [r for p in pairs for r in p["rows"] if r["pair_group"] == g
              and r["variable"] == "I(C'_A;C'_B)" and r["baseline"] == "S3"]
        nets = [r["net"] for r in rs]
        nsig = sum(1 for r in rs if r["significant"] is True)
        if rs and st(nets)[0] > 0 and nsig * 2 >= len(rs):
            inc.append((f"RX.1 已知 A 编码 B：I(C'_A;C'_B) − S3（{g}，{nsig}/{len(rs)} 对显著）", nets))
    for dom in doms:
        ran = [v["r5"][dom]["r53"] for v in V if v["r5"][dom]["r53"] is not None]
        nets = [x["net"] for x in ran]
        if ran and st(nets)[0] > 0:
            inc.append((f"R5.3 低秩预测档位（{dom} 域，{sum(n > 0 for n in nets)}/{len(V)} 张为正）", nets))
    for dom in ("v", "absv"):
        for K in r58.KM_K:
            if K > r58.KM_STRICT_MAX:
                continue
            for var, lab in (("net_cls", "类"), ("net_clsp", "类, p")):
                nets = [v["r72r82"]["r72"]["real"][dom][K][var] for v in V]
                if st(nets)[0] > 0:
                    inc.append((f"R7.2 k-means “{lab}”（{ABSV if dom == 'absv' else 'v'} 域，K={K}，"
                                f"{sum(n > 0 for n in nets)}/{len(V)} 张为正；仅主 scale 组）", nets))
    if inc:
        w("| 项 | 中位数 | 最小 | 最大 |")
        w("| --- | ---: | ---: | ---: |")
        for lab, xs in inc:
            w(f"| {lab} | {mmm(xs, 5)} |")
    else:
        w("无。")
    w("")
    w("## 说明\n")
    w("- R8.1 同时给出规格规则（峰度 > 2 × S4 前 64 层最大峰度）与调整规则（> 2 × S4 前 64 层峰度中位数），原因见规格 003 完成状态。")
    w("- R7.2 / R8.2 只在主 scale 组的抽样块上计算；R7.2 净值不含主组以外的块。")
    w("- 中间数据（奇异向量、原型、字典、块键）只存在于运行目录，汇总后删除。")
    w(f"- 运行：{extra['seconds_note']}。")
    (out / "summary.md").write_text("\n".join(L) + "\n")


def conclusions(mats, pairs, V, doms, w):
    """Section 8 item 1: per regularity, whether it exists, how much, where, and consistency."""
    w("## 结论\n")
    r61 = [v["r61"] for v in V]
    r62 = [v["r62"] for v in V]
    nbig = sum(1 for g in r62 if any(g["dup"][k] > g["dup_S2"][k] for k in r58.KS_R62 if k >= 16))
    kmax = max((g["Kstar"] or 0) for g in r62)
    w(f"- **R6 精确重复：不存在。** {len(V)} 张矩阵内及合并后，对齐的块、行、scale 行、整张矩阵重复数均为 "
      f"{sum(x['block_repeats'] + x['row_repeats'] + x['scale_row_repeats'] for x in r61)}；"
      f"行内不对齐窗口的最长重复 K* 最大为 {kmax}，k ≥ 16 有净重复（真实 > S2）的矩阵 {nbig}/{len(V)}；"
      "跨矩阵窗口在 k ≥ 16 时为 0，k = 12 与 S2 基线相当。")
    rx = {g: [r for p in pairs for r in p["rows"] if r["pair_group"] == g
              and r["variable"] == "I(C'_A;C'_B)" and r["baseline"] == "S3"] for g in ("P1", "P2", "P3")}
    cab = {g: [r for p in pairs for r in p["rows"] if r["pair_group"] == g
               and r["variable"] == "corr_absv" and r["baseline"] == "S3"] for g in ("P1", "P2", "P3")}
    p2ex = sorted([e for p in pairs if p["meta"]["group"] == "P2" for e in p["examples73"]
                   if e["axis"] == "row"], key=lambda e: -abs(e["corr"]))[:r58.MAX_EXAMPLES]
    same = sum(e["index_a"] == e["index_b"] for e in p2ex)
    pos = sum(e["corr"] > 0 for e in p2ex)
    sigs = {g: sum(r["significant"] is True for r in rx[g]) for g in rx}
    w(f"- **RX 跨矩阵同位置：只在 P2（同一专家的 w1 与 w3）存在。** 相对 S3 的严格净值 "
      f"`I(C'_A;C'_B)` 中位数 {fmt(st([r['net'] for r in rx['P2']])[0], 5)} bit/权重"
      f"（范围 {fmt(st([r['net'] for r in rx['P2']])[1], 6)}–{fmt(st([r['net'] for r in rx['P2']])[2], 5)}，"
      f"{sigs['P2']}/{len(rx['P2'])} 对显著），几乎全部来自幅值 m（`I(m_A;m_B)` 与之相当，符号 s 的互信息 ≈ 0）；"
      f"|v| 同位置相关中位数 {fmt(st([r['value'] for r in cab['P2']])[0])}。"
      f"P1（同层不同专家）{sigs['P1']}/{len(rx['P1'])} 对、对照 P3 {sigs['P3']}/{len(rx['P3'])} 对显著。"
      f"按行看关系强得多：R7.3 中 P2 的 w3 行与 w1 行的最大相关 99% 分位约 0.5，最强的 {len(p2ex)} 对里 "
      f"{same} 对是同一行号（同一中间神经元的 gate/up），其中 {pos} 对为正相关、其余为负——符号随行变化，"
      "所以同位置带符号相关的整体平均接近 0。")
    for dom in doms:
        r5 = [v["r5"][dom] for v in V]
        ran = [x["r53"] for x in r5 if x["r53"] is not None]
        nets = [x["net"] for x in ran]
        w(f"- **R5 低秩（{dom} 域）：存在弱的低秩成分，但不回本。** 超出 S4 的层数中位数 "
          f"{fmt(st([x['exceed'] for x in r5])[0], 0)}（0–{max(x['exceed'] for x in r5)}），超出能量中位数 "
          f"{fmt(st([x['exceed_energy'] for x in r5])[0], 4)}；留出 R² 最大值中位数 "
          f"{fmt(st([max(x['r2'].values()) for x in r5])[0], 4)}（最大 {fmt(st([max(x['r2'].values()) for x in r5])[2], 4)}，"
          f"S4 全部 ≤ 0）。R5.3 在 {len(ran)}/{len(V)} 张上执行，严格收益中位数 "
          f"{fmt(st([x['gain'] for x in ran])[0], 4) if ran else '—'}，扣除因子成本后净值全部为负"
          f"（最大 {fmt(max(nets), 4) if nets else '—'}）。" if all(n <= 0 for n in nets) else
          f"- **R5 低秩（{dom} 域）**：R5.3 净值 {mmm(nets, 4)}。")
    r71 = [v["r71"] for v in V]
    r73w = {dom: sum(v["r73"][dom]["row"]["n_exceed"] for v in V) for dom in doms}
    nrows = sum(len(v["r73"][doms[0]]["row"]["best"]) for v in V)
    wex = sorted([e for m in mats for e in m["examples73"]], key=lambda e: -abs(e["corr"]))[:r58.MAX_EXAMPLES]
    mats_w = sorted({e["matrix"] for e in wex})
    w(f"- **R7 近似重复：块级不存在；行级存在相似行/列。** R7.1 best 中位数真实 / S0E / S2 = "
      f"{fmt(st([x['real']['median'] for x in r71])[0], 0)} / {fmt(st([x['S0E']['median'] for x in r71])[0], 0)} / "
      f"{fmt(st([x['S2']['median'] for x in r71])[0], 0)}，best ≥ 18 的查询 0；best ≥ 回本线 m* 的比例 真实 / S0E / S2 = "
      f"{fmt(st([x['real']['frac_m'] for x in r71])[0], 3)} / {fmt(st([x['S0E']['frac_m'] for x in r71])[0], 3)} / "
      f"{fmt(st([x['S2']['frac_m'] for x in r71])[0], 3)}——真实与 S2 相同、略高于 S0E，差别来自块的组成（R2），"
      "不是块内形状。R7.2 各 K 的失真比与 S0E、S2 "
      "相同到第三位小数，严格净值全部为负且与 S2 相同（属 R2 的块组成，不是原型形状）。"
      f"R7.3 矩阵内超过 1.2×S4 最大值的行：{', '.join(f'{d} 域 {r73w[d]}' for d in doms)}（共 {nrows} 行）；"
      f"矩阵内最强的 {len(wex)} 对出自 {', '.join(mats_w)}，同一组行号在同层不同专家中重复出现。")
    r81 = {dom: [v["r5"][dom]["r81"] for v in V] for dom in doms}
    r82 = [v["r72r82"]["r82"] for v in V]
    d2 = [x["real"]["dict"][2] - np.mean([x[f"S0E_{i}"]["dict"][2] for i in (1, 2)]) for x in r82]
    w(f"- **R8 稀疏：块级字典不存在；奇异向量有少数尖峰层。** R8.2 真实 − S0E 的 R²_dict(2) 中位数 "
      f"{fmt(st(d2)[0], 5)}（最大 {fmt(st(d2)[2], 4)}），0/{len(V)} 张显著。R8.1 按规格规则（2×S4 最大）"
      + "；".join(f"{d} 域左/右超标层数中位数 {fmt(st([x['left_exceed_spec'] for x in r81[d]])[0], 0)}/"
                 f"{fmt(st([x['right_exceed_spec'] for x in r81[d]])[0], 0)}（最大 "
                 f"{max(x['left_exceed_spec'] for x in r81[d])}/{max(x['right_exceed_spec'] for x in r81[d])}）"
                 for d in doms)
      + "。w 域的 S4 本身也有峰度上千的局部化向量（幅值集中在少数行），所以 w 域的调整规则计数里有一部分是幅值分布而非稀疏系数。")
    lay = {}
    for r in rx["P2"]:
        L_ = int(r["matrix"].split(".")[0][1:])
        lay.setdefault(L_, []).append(r)
    p2l = "，".join(f"L{l} 中位数 {fmt(float(np.median([r['net'] for r in rs])), 5)}"
                   f"（{sum(r['significant'] is True for r in rs)}/{len(rs)} 显著）"
                   for l, rs in sorted(lay.items()))
    w(f"- **各层、各角色**：见文末一致性表。P2 的严格净值随层变化很大：{p2l}。"
      "R5 的留出 R² 与 R8.1 峰度在 L39 最高；R7.3 矩阵内的相似行集中在 w1/w3（行 = 中间神经元），"
      "相似列集中在 w2。R6、R7.1、R7.2、R8.2 在各层、各角色一致地为无。\n")


def _share(m, dom, variant, k):
    for s in m["spectra"]:
        if s["domain"] == dom and s["variant"] == variant and s["rep"] in (0, 1):
            x = s["sigma_norm"]
            return float((x[:k] ** 2).sum())
    return None


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="experiments/003-r5-r8")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mats, pairs = load(args.work)
    by_item = defaultdict(list)
    for res in mats + pairs:
        for r in res["rows"]:
            by_item[r["item"]].append(r)
    prow, pex, pool = pooled_r61(mats)
    write_csv(out / "r6_repeats.csv", FIELDS, by_item["R6.1"] + prow + by_item["R6.2"] + by_item["R6.3"])
    ex61 = [e for m in mats for e in m["examples6"] if e["item"] == "R6.1"]
    ex62 = [e for m in mats for e in m["examples6"] if e["item"] == "R6.2"]
    ex62.sort(key=lambda e: -e["length"])
    ex6 = (ex61 + pex)[:r58.MAX_EXAMPLES] + ex62[:r58.MAX_EXAMPLES]
    write_csv(out / "r6_examples.csv", EX_FIELDS, ex6)
    write_csv(out / "rx_pairs.csv", FIELDS, by_item["RX.1"] + by_item["RX.2"] + by_item["RX.3"])
    write_csv(out / "r5_spectrum.csv", FIELDS, by_item["R5.1"])
    write_csv(out / "r5_holdout.csv", FIELDS, by_item["R5.2"] + by_item["R5.3"])
    with open(out / "r5_singular_values.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["matrix", "layer", "expert", "role", "domain", "variant", "rep", "n",
                     "sigma_norm (space separated, descending)"])
        for m in mats:
            for s in m["spectra"]:
                mt = m["meta"]
                wr.writerow([mt["matrix"], mt["layer"], mt["expert"], mt["role"], s["domain"],
                             s["variant"], s["rep"], len(s["sigma_norm"]),
                             " ".join(f"{x:.5g}" for x in s["sigma_norm"])])
    write_csv(out / "r7_blocks.csv", FIELDS, by_item["R7.1"] + by_item["R7.2"])
    ex73w = sorted([e for m in mats for e in m["examples73"]], key=lambda e: -abs(e["corr"]))
    ex73p = sorted([e for p in pairs for e in p["examples73"]], key=lambda e: -abs(e["corr"]))
    exrows = []
    for scope, exs in (("within matrix", ex73w), ("pair", ex73p)):
        for e in exs[:r58.MAX_EXAMPLES]:
            exrows.append({k: e.get(k, "") for k in FIELDS} | {
                "item": "R7.3 example", "variable": "corr", "domain": e["domain"],
                "group_or_context": e["axis"], "subset": scope, "value": e["corr"],
                "note": f"{e['axis']} {e['index_b']} of {e.get('matrix_b') or e['matrix']} vs "
                        f"{e['axis']} {e['index_a']} of {e['matrix']}"})
    write_csv(out / "r7_rows.csv", FIELDS, by_item["R7.3"] + exrows)
    write_csv(out / "r8_sparse.csv", FIELDS, by_item["R8.1"] + by_item["R8.2"])
    secs = [r["seconds"] for r in mats + pairs]
    extra = {"n_ex6": len(ex6), "seconds_note":
             f"{len(mats)} 个矩阵任务、{len(pairs)} 个配对任务；单任务 {min(secs):.0f}–{max(secs):.0f} s，"
             f"合计 {sum(secs) / 3600:.1f} CPU·小时（每进程 1 个 BLAS 线程）"}
    summary(mats, pairs, prow, pool, out, extra)
    plots(mats, out)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
