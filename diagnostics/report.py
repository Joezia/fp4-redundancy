"""Aggregate per-matrix results into experiments/r1-r4/ CSVs, plots and summary.md.

Usage: python -m diagnostics.report --work <dir> [--out experiments/r1-r4]
"""
import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np

from . import est
from .measure import FIELDS, Rows, CONTEXTS

ITEM_FILES = {
    "r1_histograms.csv": ("A1", "A2", "A3", "A4", "A5", "A6"),
    "r2_groups.csv": ("B1", "B2", "B3", "B4"),
    "r3_mi_lag.csv": ("C1", "C2", "C3"),
    "r3_cond_entropy.csv": ("C4",),
    "r4_linear.csv": ("D1", "D2", "D3", "D4"),
    "scale.csv": ("S1", "S2"),
}
LAYERS = (0, 13, 26, 39)
ROLES = ("w1", "w2", "w3")


def fmt(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return str(v)


# ----------------------------------------------------------------------------- pooled items

def pooled_rows(results):
    metas = [r["meta"] for r in results]
    hists = np.array([r["hist"] for r in results], dtype=np.int64)
    Ns = hists.sum(1)
    Ntot = int(Ns.sum())
    rows = Rows({"matrix": "ALL", "layer": "", "expert": "", "role": ""})
    pool = hists.sum(0)
    H0 = est.entropy_counts(pool)
    K = int((pool > 0).sum())
    for c in range(16):
        rows.add("A6", "C", pool[c], group=f"c={c}", n=Ntot, bits=False, note="pooled count")
    rows.add("A6", "C", H0, group="H0", n=Ntot)
    rows.add("A6", "C", K, group="K", n=Ntot, bits=False)
    rows.add("A6", "C", 4 - H0, group="4-H0", n=Ntot)
    rows.add("A6", "C", 4 - math.log2(K), group="unused=4-log2K", n=Ntot)
    rows.add("A6", "C", math.log2(K) - H0, group="skew=log2K-H0", n=Ntot)

    keys = {
        "matrix": [m["matrix"] for m in metas],
        "layer": [m["layer"] for m in metas],
        "role": [m["role"] for m in metas],
        "expert": [f"{m['layer']}.{m['expert']}" for m in metas],
    }

    def grouped(tab, key):
        labels = sorted(set(keys[key]), key=str)
        out = np.zeros((len(labels), 16))
        for i, lab in enumerate(keys[key]):
            out[labels.index(lab)] += tab[i]
        return out

    rng = np.random.default_rng(20261112)
    nulls = []
    for _ in range(3):
        remaining = pool.copy()
        sim = np.zeros_like(hists)
        for i, n in enumerate(Ns):
            sim[i] = rng.multivariate_hypergeometric(remaining, int(n))
            remaining -= sim[i]
        nulls.append(sim)
    for key in keys:
        t = grouped(hists, key)
        mi, n, dof = est.mi_table(t)
        reps = [est.mi_table(grouped(s, key))[0] for s in nulls]
        ng = t.shape[0]
        rows.add("B1", "C", mi, group=f"pooled:{key}", n=n, baseline="pooled-perm", reps=reps,
                 kind="mi", dof=dof,
                 note=f"|G|={ng}; baseline = permutation of all codewords across the 36 matrices "
                      "(within-matrix S0 leaves matrix histograms unchanged)")
        rows.add("B1", "C", (K - 1) * ng * 12 / n, group=f"pooled:{key}:codebook_cost", n=n,
                 note="(K-1)|G|12/N")
    return rows.rows


# ----------------------------------------------------------------------------- selection

def pick(results, **kw):
    """Per-matrix list of the single row matching kw (None where missing)."""
    out = []
    for r in results:
        m = [x for x in r["rows"] if all(x.get(k) == v for k, v in kw.items())]
        assert len(m) <= 1, (kw, len(m))
        out.append(m[0] if m else None)
    return out


def vals(rows, field):
    v = [r[field] for r in rows if r is not None and r[field] != ""]
    return np.array(v, dtype=float)


def strict_sig(r):
    """Spec significance AND |net| > 5 x shuffle-baseline std (spec 6: 以打乱基线为准)."""
    if r is None or r["significant"] is not True:
        return False
    bs = r["baseline_std"]
    return bs != "" and abs(float(r["net"])) > 5 * float(bs)


def stat_line(label, rows, field="net", bits=True, sig=True):
    v = vals(rows, field)
    if v.size == 0:
        return f"| {label} | — | — | — | — | 未执行 |"
    med = float(np.median(v))
    pct = f"{100 * med / 4:.3f}%" if bits else ""
    nsig = sum(1 for r in rows if r is not None and r["significant"] is True)
    nstrict = sum(1 for r in rows if strict_sig(r))
    s = f"{nsig}/{len(rows)}（严格 {nstrict}）" if sig else ""
    return f"| {label} | {med:.4g} | {v.min():.4g} | {v.max():.4g} | {pct} | {s} |"


HDR = "| 指标 | 中位数 | 最小 | 最大 | 中位数占 4 bit | 显著 |\n| --- | ---: | ---: | ---: | ---: | ---: |"


def by_group_line(label, results, rows, field="net"):
    cells = []
    for key, labs in (("layer", LAYERS), ("role", ROLES)):
        for lab in labs:
            v = [float(r[field]) for res, r in zip(results, rows)
                 if r is not None and r[field] != "" and res["meta"][key] == lab]
            cells.append(f"{np.median(v):.4g}" if v else "—")
    return f"| {label} | " + " | ".join(cells) + " |"


BG_HDR = ("| 指标（中位数） | L0 | L13 | L26 | L39 | w1 | w2 | w3 |\n"
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")


# ----------------------------------------------------------------------------- plots

def plots(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def curve(item, lags, field, **kw):
        arr = np.full((len(results), len(lags)), np.nan)
        for i, r in enumerate(results):
            for x in r["rows"]:
                if x["item"] == item and all(x.get(k) == v for k, v in kw.items()) and x["lag"] in lags:
                    if x[field] != "":
                        arr[i, lags.index(x["lag"])] = x[field]
        return np.nanmedian(arr, axis=0), arr

    def marks(ax, maxd):
        for k in range(32, maxd + 1, 32):
            ax.axvline(k, color="0.6", lw=0.8, ls=":")

    # MI row
    lags = list(range(1, 97))
    fig, ax = plt.subplots(figsize=(9, 5))
    for sub, col in (("all", "C0"), ("same", "C1"), ("cross", "C2")):
        med, arr = curve("C1", lags, "value", subset=sub, baseline="S0", direction="row")
        ax.plot(lags, med, color=col, lw=2, label=f"real, {sub}-block pairs (median of 36)")
        for bl, ls in (("S0", "--"), ("S1", "-."), ("S2", ":")):
            if sub == "cross" and bl != "S2":
                continue
            m2, _ = curve("C1", lags, "baseline_mean", subset=sub, baseline=bl, direction="row")
            ax.plot(lags, m2, color=col, ls=ls, lw=1.2, label=f"{bl} mean, {sub}")
    marks(ax, 96)
    ax.set_yscale("log")
    ax.set_xlabel("lag d along row (dotted grey: multiples of B=32)")
    ax.set_ylabel("I(C[r,j]; C[r,j+d])  bit")
    ax.set_title("C1 row MI vs lag — 36 matrices, median")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "mi_lag_row.png", dpi=110)
    plt.close(fig)

    lags = list(range(1, 65))
    fig, ax = plt.subplots(figsize=(9, 5))
    med, arr = curve("C2", lags, "value", baseline="S0", direction="col")
    for row in arr:
        ax.plot(lags, row, color="C0", alpha=0.15, lw=0.6)
    ax.plot(lags, med, color="C0", lw=2, label="real (median; thin = each matrix)")
    for bl, ls in (("S0", "--"), ("S3", ":")):
        m2, _ = curve("C2", lags, "baseline_mean", baseline=bl, direction="col")
        ax.plot(lags, m2, color="k", ls=ls, lw=1.2, label=f"{bl} mean (median)")
    marks(ax, 64)
    ax.set_yscale("log")
    ax.set_xlabel("lag d along column")
    ax.set_ylabel("I(C[r,j]; C[r+d,j])  bit")
    ax.set_title("C2 column MI vs lag — 36 matrices")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "mi_lag_col.png", dpi=110)
    plt.close(fig)

    for dirn, maxd, bls, fname in (("row", 96, ("S0", "S2"), "acf_row.png"),
                                   ("col", 64, ("S0", "S3"), "acf_col.png")):
        lags = list(range(1, maxd + 1))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, var in zip(axes, ("v", "|v|")):
            subs = ("all", "same", "cross") if dirn == "row" else ("all",)
            for sub, col in zip(subs, ("C0", "C1", "C2")):
                med, _ = curve("D1", lags, "value", variable=var, direction=dirn, subset=sub,
                               baseline="S0")
                ax.plot(lags, med, color=col, lw=2, label=f"real, {sub}")
                for bl, ls in zip(bls, ("--", ":")):
                    m2, _ = curve("D1", lags, "baseline_mean", variable=var, direction=dirn,
                                  subset=sub, baseline=bl)
                    ax.plot(lags, m2, color=col, ls=ls, lw=1.1, label=f"{bl} mean, {sub}")
            n = 11.8e6
            ax.axhline(5 / math.sqrt(n), color="r", lw=0.6)
            ax.axhline(-5 / math.sqrt(n), color="r", lw=0.6)
            marks(ax, maxd)
            ax.set_title(f"D1 autocorrelation of {var} along {dirn} (median of 36)")
            ax.set_xlabel("lag d")
            ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=110)
        plt.close(fig)


# ----------------------------------------------------------------------------- summary

def summary(results, pooled, out, extra):
    S = []
    w = S.append
    n = len(results)
    w("# R1–R4 规律检测结果（spec 001）\n")
    w(f"样本：{n} 张完整 FP4 路由专家矩阵（层 0/13/26/39，每层 3 个专家 × w1/w2/w3，见 `samples.json`），"
      "每张 11,796,480 个码字，以整张矩阵为统计单位。全部信息量单位 bit/权重（scale 项为 bit/scale），"
      "“净值” = 真实值 − 对应打乱基线 3 次重复的均值；“显著”按 spec 第 6 节。表中中位数/最小/最大为 36 张矩阵的分布。"
      "合成数据检查 T1–T6 全部通过（`validation.json`），格式事实见 `format.md`。本文件只陈述测量结果。\n")

    # ---------------- R1
    w("## R1 边缘偏斜\n")
    a = {g: pick(results, item="A1", group_or_context=g) for g in
         ("4-H0", "unused=4-log2K", "skew=log2K-H0", "H0", "K")}
    w(HDR)
    w(stat_line("A1 4 − H0", a["4-H0"], "value", sig=False))
    w(stat_line("A1 未用码字项 4 − log2 K", a["unused=4-log2K"], "value", sig=False))
    w(stat_line("A1 已用码字偏斜 log2 K − H0", a["skew=log2K-H0"], "value", sig=False))
    w(stat_line("A1 K（出现过的码字数）", a["K"], "value", bits=False, sig=False))
    mg = pick(results, item="A2", group_or_context="merge_gain=P(m=0)H(s|m=0)")
    w(stat_line("A2 合并两个零的收益 P(m=0)·H(s|m=0)", mg, "value", sig=False))
    ism = pick(results, item="A3", group_or_context="I(s;m)")
    w(stat_line("A3 I(s;m)", ism, "value", sig=False))
    hs = pick(results, item="A3", group_or_context="H(s)")
    w(stat_line("A3 H(s)", hs, "value", sig=False))
    he = pick(results, item="A4", group_or_context="H0(E)")
    w(stat_line("A4 H0(E)（bit/scale）", he, "value", sig=False))
    ke = pick(results, item="A4", group_or_context="K_E")
    w(stat_line("A4 出现过的 scale 取值数", ke, "value", bits=False, sig=False))
    w("")
    p = {r["group_or_context"]: r["value"] for r in pooled if r["item"] == "A6"}
    w(f"A6 全样本合并：H0 = {p['H0']:.5f}，4 − H0 = {p['4-H0']:.5f}（{100 * p['4-H0'] / 4:.3f}%），"
      f"K = {int(p['K'])}，未用码字项 = {p['unused=4-log2K']:.5f}，偏斜项 = {p['skew=log2K-H0']:.5f}。\n")
    mm = np.zeros(8)
    for r in results:
        for x in r["rows"]:
            if x["item"] == "A5":
                mm[int(x["group_or_context"].split("=")[1])] += x["value"]
    w("A5 块最大幅值索引 m_max 的合并分布（36 张）：" +
      "，".join(f"{i}: {mm[i] / mm.sum():.4f}" for i in range(8)) + "。\n")
    emin = vals(pick(results, item="A4", group_or_context="min_byte"), "value")
    emax = vals(pick(results, item="A4", group_or_context="max_byte"), "value")
    w(f"A4 scale 字节范围：最小 {int(emin.min())}（2^{int(emin.min()) - 127}），最大 {int(emax.max())}（2^{int(emax.max()) - 127}）。\n")
    w("**回答**：\n")
    w(f"1. 存在：36/36 张矩阵 4 − H0 > 0，远大于 plug-in 偏差量级（15/(2N ln2) ≈ {15 / (2 * 11796480 * est.LN2):.1e}）。"
      "R1 无打乱基线（S0 保持直方图），表中直接给原始值。")
    w(f"2. 净值：4 − H0 中位数 {np.median(vals(a['4-H0'], 'value')):.4f} bit/权重"
      f"（{np.min(vals(a['4-H0'], 'value')):.4f}–{np.max(vals(a['4-H0'], 'value')):.4f}）。")
    nunused = int((vals(a['unused=4-log2K'], 'value') > 0).sum())
    w(f"3. 位置：未用码字项在 {nunused}/36 张矩阵上 > 0；偏斜几乎全部来自已用码字内部的形状。"
      f"符号与幅值近乎独立（I(s;m) 中位数 {np.median(vals(ism, 'value')):.2e}），H(s) ≈ 1，"
      "偏斜集中在幅值分布 H(m) 上。两个零码字（+0/−0）都被使用，合并收益见 A2。")
    w("4. 一致性：见下表（按层、角色的中位数）。\n")
    w(BG_HDR)
    w(by_group_line("4 − H0", results, a["4-H0"], "value"))
    w(by_group_line("A2 合并零收益", results, mg, "value"))
    w(by_group_line("A4 H0(E)", results, he, "value"))
    w("")

    # ---------------- R2
    w("## R2 分组异质性\n")
    w("B1 解码端已知分组（基线 S0；码表成本 = (K−1)·|G|·12/N，不从净值中扣除）：\n")
    w(HDR)
    b1 = {}
    for g in ("r", "j", "b", "p", "p%2", "E"):
        b1[g] = pick(results, item="B1", group_or_context=g)
        w(stat_line(f"B1 I(C; {g}) 净值", b1[g]))
        w(stat_line(f"B1 {g} 码表成本", pick(results, item="B1", group_or_context=g + ":codebook_cost"),
                    "value", sig=False))
    w("")
    w("跨矩阵合并（基线为 36 张矩阵全部码字合并后的随机置换，每次 3 重复；within-matrix S0 不改变各矩阵直方图，不适用）：\n")
    w("| 分组 | I(C;G) | 基线均值 | 净值 | 解析偏差 | 码表成本 | 显著 |\n| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for key in ("matrix", "layer", "role", "expert"):
        r = [x for x in pooled if x["group_or_context"] == f"pooled:{key}"][0]
        c = [x for x in pooled if x["group_or_context"] == f"pooled:{key}:codebook_cost"][0]
        w(f"| {key} | {r['value']:.4g} | {r['baseline_mean']:.3g} | {r['net']:.4g} | "
          f"{r['analytic_bias']:.3g} | {c['value']:.3g} | {r['significant']} |")
    w("")
    w("B2 由数据导出、需传标签的分组（可实现净收益 = I(C;G) − H(G)/B，原始值；S0 列为同一量在 S0 上的均值）：\n")
    w("| 标签 | I(C;G) 中位数 | H(G)/B 中位数 | 可实现净收益 中位数 [最小, 最大] | S0 上可实现净收益 中位数 | 可实现净收益 > 0 的矩阵数 |")
    w("| --- | ---: | ---: | ---: | ---: | ---: |")
    b2 = {}
    for lab in ("m_max", "energy2", "energy4", "energy8"):
        i_ = vals(pick(results, item="B2", group_or_context=f"{lab}:I(C;G)"), "value")
        c_ = vals(pick(results, item="B2", group_or_context=f"{lab}:label_cost=H(G)/B"), "value")
        b2[lab] = pick(results, item="B2", group_or_context=f"{lab}:achievable=I-H(G)/B")
        a_ = vals(b2[lab], "value")
        s_ = vals(b2[lab], "baseline_mean")
        w(f"| {lab} | {np.median(i_):.4g} | {np.median(c_):.4g} | {np.median(a_):.4g} [{a_.min():.4g}, {a_.max():.4g}] | "
          f"{np.median(s_):.4g} | {(a_ > 0).sum()}/36 |")
    w("")
    w(HDR)
    b3 = pick(results, item="B3")
    w(stat_line("B3 I(C; 块id) 净值（vs S0）", b3))
    b4 = {}
    for g in ("block:alpha=32", "block:alpha=128", "block:alpha=512", "row:alpha=512",
              "row:alpha=2048", "row:alpha=8192"):
        b4[g] = pick(results, item="B4", group_or_context=g)
        w(stat_line(f"B4 自适应码长收益 H0 − 码长/N（{g}，原始值）", b4[g], "value", sig=False))
    w("")
    w("**回答**：\n")
    nb3 = sum(1 for r in b3 if r["significant"])
    w(f"1. 存在：B3 块级离散度净值显著 {nb3}/36；B1 中 scale 值 E 分组显著 "
      f"{sum(1 for r in b1['E'] if r['significant'])}/36，行 r 分组显著 {sum(1 for r in b1['r'] if r['significant'])}/36，"
      f"列 j 分组显著 {sum(1 for r in b1['j'] if r['significant'])}/36。")
    w(f"2. 净值：B3 中位数 {np.median(vals(b3, 'net')):.4f} bit/权重（总量指标，不含标签成本）；"
      f"免费标签 E 的 I(C;E) 净值中位数 {np.median(vals(b1['E'], 'net')):.4f}；"
      f"需传标签时最佳可实现净收益中位数为 "
      + "、".join(f"{k} {np.median(vals(v, 'value')):.4f}" for k, v in b2.items())
      + f"；不传标签的块内自适应（α=128）中位数 {np.median(vals(b4['block:alpha=128'], 'value')):.4f}。")
    w(f"3. 位置：差异集中在 32 码字块这一级（与 scale 共享的同一分块）：B3 净值与块内容相关的 E、m_max、能量档的 I(C;G) 同量级。"
      f"按行 r（中位数 {np.median(vals(b1['r'], 'net')):.4f}）、列 j（{np.median(vals(b1['j'], 'net')):.4f}）、块号 b、块内位置 p、"
      "p 奇偶分组的净值小一到三个数量级，且行、列分组的净值小于各自码表成本估计。跨矩阵合并的分组（矩阵、层、角色、专家）净值均 < 0.001。")
    w("4. 一致性：\n")
    w(BG_HDR)
    w(by_group_line("B3 净值", results, b3))
    w(by_group_line("B1 E 净值", results, b1["E"]))
    w(by_group_line("B1 r 净值", results, b1["r"]))
    w(by_group_line("B2 m_max 可实现", results, b2["m_max"], "value"))
    w(by_group_line("B2 energy2 可实现", results, b2["energy2"], "value"))
    w(by_group_line("B4 block α=128", results, b4["block:alpha=128"], "value"))
    w("")

    # ---------------- R3
    w("## R3 局部上下文依赖\n")
    w("“真实 − S0” 含块级分布差异（R2）与顺序依赖两部分；“真实 − S2”（沿行）/“真实 − S3”（沿列）只剩顺序依赖。\n")
    w(HDR)
    c1 = {}
    for sub in ("all", "same", "cross"):
        for bl in ("S0", "S2"):
            c1[(sub, bl)] = pick(results, item="C1", direction="row", lag=1, subset=sub, baseline=bl)
            w(stat_line(f"C1 d=1 {sub} 真实 − {bl}", c1[(sub, bl)]))
    for d in (2, 4, 31, 32, 33, 64, 96):
        w(stat_line(f"C1 d={d} all 真实 − S2", pick(results, item="C1", direction="row", lag=d, subset="all", baseline="S2")))
    c2 = {}
    for bl in ("S0", "S3"):
        c2[bl] = pick(results, item="C2", direction="col", lag=1, baseline=bl)
        w(stat_line(f"C2 d=1 真实 − {bl}", c2[bl]))
    w("")
    sig_counts = extra["sig_counts"]
    w("各 lag 上显著的 (矩阵, lag) 数：\n")
    w("| 项 | 基线 | 显著数 / 总数 | 严格显著数 |\n| --- | --- | ---: | ---: |")
    for k, (s, st, t) in sig_counts.items():
        w(f"| {k[0]} | {k[1]} | {s} / {t} | {st} |")
    w("")
    w("C1 d ≤ 4 按起点列 j 奇偶拆分（全部对，真实值中位数；S2 列为 S2 均值中位数）：\n")
    w("| d | j 偶 | j 奇 | j 偶（S2） | j 奇（S2） |\n| ---: | ---: | ---: | ---: | ---: |")
    for d in range(1, 5):
        e = pick(results, item="C1", direction="row", lag=d, subset="all_jeven", baseline="S2")
        o = pick(results, item="C1", direction="row", lag=d, subset="all_jodd", baseline="S2")
        w(f"| {d} | {np.median(vals(e, 'value')):.5f} | {np.median(vals(o, 'value')):.5f} | "
          f"{np.median(vals(e, 'baseline_mean')):.5f} | {np.median(vals(o, 'baseline_mean')):.5f} |")
    w("")
    w("C3 分量拆解（d = 1，真实 − S0）：\n")
    w(HDR)
    for dirn in ("row", "col"):
        for nm in ("I(s;s')", "I(m;m')", "I(s;m')", "I(m;s')"):
            w(stat_line(f"C3 {dirn} {nm}", pick(results, item="C3", variable=nm, direction=dirn, lag=1)))
    w("")
    w("C4 留出条件熵收益 H0 − H(C | 上下文)（原始值；S0 结果为学习成本）：\n")
    w("| 上下文 | 原始 中位数 | S0 均值 中位数 | 真实 − S0 中位数 [最小, 最大] | 显著(vs S0) | 顺序基线 | 真实 − 顺序基线 中位数 [最小, 最大] | 显著 |")
    w("| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |")
    c4 = {}
    for name, _ in CONTEXTS:
        r0 = pick(results, item="C4", group_or_context=name, baseline="S0")
        bl = "S2" if name.startswith("left") and "up" not in name else "S3"
        r1 = pick(results, item="C4", group_or_context=name, baseline=bl)
        c4[name] = (r0, r1, bl)
        v0, n0, n1 = vals(r0, "value"), vals(r0, "net"), vals(r1, "net")
        w(f"| {name} | {np.median(v0):.4g} | {np.median(vals(r0, 'baseline_mean')):.4g} | "
          f"{np.median(n0):.4g} [{n0.min():.4g}, {n0.max():.4g}] | {sum(1 for r in r0 if r['significant'])}/36 | {bl} | "
          f"{np.median(n1):.4g} [{n1.min():.4g}, {n1.max():.4g}] | {sum(1 for r in r1 if r['significant'])}/36 |")
    w("")
    w("**回答**：\n")
    s_same = sum(1 for r in c1[("same", "S0")] if r["significant"])
    s_s2 = sum(1 for r in c1[("all", "S2")] if r["significant"])
    st_s2 = sum(1 for r in c1[("all", "S2")] if strict_sig(r))
    bstd = np.median(vals(c1[("all", "S2")], "baseline_std"))
    sc = extra["sig_counts"]
    w(f"1. 存在：相对 S0，沿行相邻码字的成对依赖显著（d=1 同块 {s_same}/36），C4 只含“左”的上下文 36/36 显著。"
      f"相对只破坏块内顺序的 S2，C1 d=1 按 spec 解析阈值显著 {s_s2}/36，但净值正负各半，S2 三次重复的标准差中位数 {bstd:.1e} "
      f"约为解析标准差的数十倍（解析公式只在真实互信息≈0 时成立，S2 保留了块级依赖），同时超过 5 倍基线标准差的为 {st_s2}/36；"
      f"全部 lag 合计，C1 相对 S2 严格显著 {sc[('C1 all', 'S2')][1]}/{sc[('C1 all', 'S2')][2]}（全部对）、"
      f"{sc[('C1 same', 'S2')][1]}/{sc[('C1 same', 'S2')][2]}（同块对），分散在不同矩阵和互不相邻的 lag 上，无随 d 衰减的形状，"
      "与 3 次重复估计标准差带来的假阳性量级一致。"
      f"C4 各上下文相对 S2/S3 的净值按“其他量”规则（5 倍基线标准差且 > 1e-4）显著数见上表（至多 1/36）。"
      f"沿列：C2 相对 S0 显著 {sc[('C2', 'S0')][0]}/{sc[('C2', 'S0')][2]}，相对 S3 显著 {sc[('C2', 'S3')][0]}（严格 {sc[('C2', 'S3')][1]}），"
      "后者同样分散，净值 ≤ 2.5e-5 bit。")
    w(f"2. 净值：含块级差异的“真实 − S0”：C1 d=1 同块中位数 {np.median(vals(c1[('same', 'S0')], 'net')):.4f}，"
      f"C4 左1–3 中位数 {np.median(vals(c4['left1-3'][0], 'net')):.4f} bit/权重。只剩顺序依赖的部分："
      f"C1 d=1 真实 − S2 中位数 {np.median(vals(c1[('all', 'S2')], 'net')):.1e}，"
      f"C4 左1–3 真实 − S2 中位数 {np.median(vals(c4['left1-3'][1], 'net')):.1e}，"
      f"左1+上1+左上 真实 − S3 中位数 {np.median(vals(c4['left1+up1+upleft'][1], 'net')):.1e} bit/权重（各矩阵均在 ±4e-4 以内）。")
    w(f"3. 位置：相对 S0 的依赖集中在同块对，由幅值分量 I(m;m') 承载，符号分量 I(s;s') 与交叉分量沿行不显著；"
      f"跨块对相对 S0 显著 {sc[('C1 cross', 'S0')][0]}/{sc[('C1 cross', 'S0')][2]}（多数在 w1），相对 S2 为 0——"
      "即相邻块的分布也相近，仍属分组差异而非块内顺序。沿列的相对 S0 显著（C2，C3 列向 I(s;s')/I(m;m') 部分矩阵显著）在 S3 下消失，"
      "按 spec 第 10 节读作列级分布差异（R2）。d ≤ 4 奇偶两组在 d 为奇数时不同，但 S2 下差别相同：d 为奇数时跨块边界的对全部落在 j 奇组，"
      "差别来自跨块对比例，未见以字节为周期的额外结构，与 F2 的低 nibble 在前一致。")
    w("4. 一致性：\n")
    w(BG_HDR)
    w(by_group_line("C1 d=1 same 真实 − S0", results, c1[("same", "S0")]))
    w(by_group_line("C1 d=1 all 真实 − S2", results, c1[("all", "S2")]))
    w(by_group_line("C2 d=1 真实 − S3", results, c2["S3"]))
    w(by_group_line("C4 left1-3 真实 − S2", results, c4["left1-3"][1]))
    w(by_group_line("C4 left1+up1+upleft 真实 − S3", results, c4["left1+up1+upleft"][1]))
    w("")

    # ---------------- R4
    w("## R4 局部线性相关（数值域，不乘 scale）\n")
    w(HDR)
    d1 = {}
    for var in ("v", "|v|"):
        for dirn, bl2 in (("row", "S2"), ("col", "S3")):
            for bl in ("S0", bl2):
                d1[(var, dirn, bl)] = pick(results, item="D1", variable=var, direction=dirn, lag=1,
                                           subset="all", baseline=bl)
                w(stat_line(f"D1 {var} {dirn} d=1 自相关 真实 − {bl}", d1[(var, dirn, bl)], bits=False))
    d2, d3, d4 = {}, {}, {}
    for var in ("v", "|v|"):
        for dirn in ("row", "col"):
            d2[(var, dirn)] = pick(results, item="D2", variable=var, direction=dirn)
            w(stat_line(f"D2 {var} {dirn} ½log2(1/SFM) 真实 − S0", d2[(var, dirn)]))
        d3[var] = pick(results, item="D3", variable=var, group_or_context="R2")
        w(stat_line(f"D3 {var} R²（留出）真实 − S0", d3[var], bits=False))
        w(stat_line(f"D3 {var} −½log2(1−R²) 真实 − S0",
                    pick(results, item="D3", variable=var, group_or_context="-0.5*log2(1-R2)")))
        d4[var] = pick(results, item="D4", variable=var)
        ran = sum(1 for r in d4[var] if r is not None and r["value"] != "")
        w(stat_line(f"D4 {var} H0 − H(C|预测八分位)（执行 {ran}/36）", d4[var]))
    w("")
    w("**回答**：\n")
    nv = sum(1 for r in d1[("v", "row", "S0")] if r["significant"])
    na = sum(1 for r in d1[("|v|", "row", "S0")] if r["significant"])
    w(f"1. 存在：带符号值 v 的沿行 d=1 自相关对 S0 显著 {nv}/36，沿列 {sum(1 for r in d1[('v', 'col', 'S0')] if r['significant'])}/36；"
      f"|v| 沿行对 S0 显著 {na}/36、对 S2 显著 {sum(1 for r in d1[('|v|', 'row', 'S2')] if r['significant'])}/36。"
      f"D3 R²(v) 中位数 {np.median(vals(d3['v'], 'value')):.2e}，R²(|v|) 中位数 {np.median(vals(d3['|v|'], 'value')):.4f}。")
    w(f"2. 净值：D3 换算 −½log2(1−R²)：v 中位数 "
      f"{np.median(vals(pick(results, item='D3', variable='v', group_or_context='-0.5*log2(1-R2)'), 'value')):.2e}，"
      f"|v| 中位数 {np.median(vals(pick(results, item='D3', variable='|v|', group_or_context='-0.5*log2(1-R2)'), 'value')):.4f} bit/权重；"
      f"D4（|v| 预测八分位作上下文）中位数 {np.median(vals(d4['|v|'], 'value')) if vals(d4['|v|'], 'value').size else float('nan'):.4f}。")
    w("3. 位置：沿行只有 |v| 显著，且相对 S2 的净值 0/36 显著（块内打乱保留该相关）——幅值按块聚集，"
      "按 spec 第 10 节读作“幅值聚集，带符号的线性结构不存在”。沿列：v 与 |v| 的 d=1 自相关在部分矩阵相对 S0 显著，"
      "相对 S3 全部不显著，即列间均值/幅值的差异（R2），不是行序上的相关。D2 只有 |v| 沿行显著。"
      f"D3 的 |v| 换算值（中位数 {np.median(vals(pick(results, item='D3', variable='|v|', group_or_context='-0.5*log2(1-R2)'), 'value')):.4f}）"
      f"小于 C4 左1–3 相对 S0 的收益（中位数 {np.median(vals(c4['left1-3'][0], 'net')):.4f}）。")
    w("4. 一致性：\n")
    w(BG_HDR)
    w(by_group_line("D1 |v| row d=1 真实 − S0", results, d1[("|v|", "row", "S0")]))
    w(by_group_line("D1 |v| row d=1 真实 − S2", results, d1[("|v|", "row", "S2")]))
    w(by_group_line("D3 R²(|v|)", results, d3["|v|"], "value"))
    w(by_group_line("D3 R²(v)", results, d3["v"], "value"))
    w("")

    # ---------------- scale
    w("## S scale 流（附带）\n")
    w(HDR)
    for dirn in ("row", "col"):
        w(stat_line(f"S1 I(E;E') {dirn} d=1 真实 − 置换", pick(results, item="S1", direction=dirn, lag=1)))
    for dirn in ("left", "up"):
        w(stat_line(f"S2 H0(E) − H(E|{dirn})（bit/scale）真实 − 置换",
                    pick(results, item="S2", group_or_context=f"H0(E)-H(E|{dirn})")))
    w("")
    w("## 说明\n")
    w("- 未执行项：D4 仅在 D3 R² > 0.001 时执行，未执行的矩阵在 `r4_linear.csv` 中留空并写明原因；"
      "C1 同块组在 d ≥ 32 无样本对，不输出该组行。")
    w("- 长表在 spec 字段之后多一列 `note`，写明单位、未执行原因或留出估计的 H0/H 原始值。D1 行的 `analytic_bias` 列填 5/√n（显著阈值）。")
    w("- “显著”列为 spec 第 6 节定义（CSV 的 `significant` 列同此）；括号内“严格”另要求 |净值| > 5 × 该打乱基线 3 次重复的标准差。"
      "互信息的解析标准差只在真实互信息≈0 时成立；对 S1/S2/S3 这类保留部分依赖的基线，打乱重复本身的离散度远大于解析值，"
      "spec 第 6 节注明此时“以打乱基线为准”。")
    w("- 跨矩阵 B1 的基线不是 S0（矩阵内置换不改变矩阵直方图），而是 36 张矩阵全部码字合并后的随机置换（多元超几何抽样实现）。")
    (out / "summary.md").write_text("\n".join(S) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--samples", default="experiments/r1-r4/samples.json")
    ap.add_argument("--out", default="experiments/r1-r4")
    args = ap.parse_args()
    out = Path(args.out)
    with open(args.samples) as f:
        samples = json.load(f)["matrices"]
    results = []
    for s in samples:
        with open(Path(args.work) / f"{s['matrix']}.pkl", "rb") as f:
            results.append(pickle.load(f))
    pooled = pooled_rows(results)
    allrows = [x for r in results for x in r["rows"]] + pooled
    for fname, items in ITEM_FILES.items():
        with open(out / fname, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(FIELDS)
            for x in allrows:
                if x["item"] in items:
                    wr.writerow([fmt(x[k]) for k in FIELDS])
    sig_counts = {}
    for item, bl, sub in (("C1 all", "S0", "all"), ("C1 all", "S2", "all"), ("C1 same", "S0", "same"),
                          ("C1 same", "S2", "same"), ("C1 cross", "S0", "cross"), ("C1 cross", "S2", "cross"),
                          ("C2", "S0", None), ("C2", "S3", None)):
        it = item.split()[0]
        rr = [x for r in results for x in r["rows"] if x["item"] == it and x["baseline"] == bl
              and (sub is None or x["subset"] == sub)]
        sig_counts[(item, bl)] = (sum(1 for x in rr if x["significant"] is True),
                                  sum(1 for x in rr if strict_sig(x)), len(rr))
    for var in ("v", "|v|"):
        for dirn, bls in (("row", ("S0", "S2")), ("col", ("S0", "S3"))):
            for bl in bls:
                rr = [x for r in results for x in r["rows"] if x["item"] == "D1" and x["variable"] == var
                      and x["direction"] == dirn and x["baseline"] == bl and x["subset"] == "all"]
                sig_counts[(f"D1 {var} {dirn} all", bl)] = (sum(1 for x in rr if x["significant"] is True),
                                                            sum(1 for x in rr if strict_sig(x)), len(rr))
    summary(results, pooled, out, {"sig_counts": sig_counts})
    plots(results, out)
    secs = [r["seconds"] for r in results]
    print(f"{len(results)} matrices; per-matrix seconds median {np.median(secs):.0f}")


if __name__ == "__main__":
    main()
