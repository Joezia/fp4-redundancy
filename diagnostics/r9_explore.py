"""Spec 005 section 5: autonomous exploration of B and C variants.

Stages (pickles in --work):
  synth  section-5.1 step 2: independent base (net must be <= 0.001) and one planted structure
         per variant (recovery criterion stated next to the construction).
  dev    development set: the 12 probe matrices (B and C).
  conf   confirmation set, run once per frozen variant: B = full-layer matrices other than the
         probe matrices; C = the second original expert of every layer (12 matrices).
Usage: python -m diagnostics.r9_explore --stage synth --variants VB1 VC1 [--snapshot ...] --work <dir>
"""
import argparse
import json
import math
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from . import r9, r9_vb  # noqa: E402
from .gaps_run import load  # noqa: E402
from .ledger import merge_zero  # noqa: E402
from .r9_check import P_N, base, const_E, mixture_blocks, tc4_data, theory_gain  # noqa: E402
from .r9_run import SAMPLES, names, seed_of  # noqa: E402
from .st import index_snapshot  # noqa: E402

R, W, B = 1024, 2048, 32


# ----------------------------------------------------------------------------- C variants

def vc_select(Cp, E, seed, dims, output="code", batch=None):
    """Train one network per embedding size; choose by validation CE + cost (no test data)."""
    runs = {}
    for d in dims:
        cfg = {"emb": d, "output": output}
        if batch:
            cfg["batch"] = batch
        runs[d] = r9.inr_fit(Cp, E, seed, cfg)
    score = {d: (r["best_val"] - r["base_val"]) + r["cost"] for d, r in runs.items()}
    pick = min(dims, key=lambda d: score[d])
    r = dict(runs[pick])
    r["selection"] = {"chosen": pick, "val_score": {str(d): s for d, s in score.items()},
                      "per_dim": {str(d): {k: runs[d][k] for k in ("gain", "cost", "net", "best_val")}
                                  for d in dims}}
    r["codelen"] = r["net_test"]
    return r


def vc_scalebins(Cp, E, seed, split=(0.7, 0.1, 0.2)):
    """VC3: P(C' | E, row bin, column bin). Each row (column) sends the quantile bin of its mean
    |v| - E-mean (log2(bins) bits each: only the bin index is needed by the decoder); the table is
    charged 16 bits per probability. Same 70/10/20 split and baseline as C2; the side (row, column,
    both) and the number of bins (2, 4, 8, 16) are chosen on the validation part (CE + cost)."""
    R, W = Cp.shape
    N = Cp.size
    rng = r9.rng_for(seed, 20)
    train, val, test = r9.inr_split(N, rng, split)
    vals, eb = r9.e_index(E)
    nE = len(vals)
    ew = np.repeat(eb.reshape(E.shape), B, axis=1).ravel()
    y = Cp.ravel().astype(np.int64)
    base = r9.baseline_table(y, ew, nE, train)
    bval, btest = r9.ce_bits(base, y, ew, val), r9.ce_bits(base, y, ew, test)
    X = r9.absv_resid(Cp, E)
    rmean, cmean = X.mean(axis=1), X.mean(axis=0)
    rows = np.repeat(np.arange(R), W)
    cols = np.tile(np.arange(W), R)
    nsym = len(np.unique(y))
    res = {}
    for side in ("row", "col", "both"):
        for nb in (2, 4, 8, 16):
            rb = r9.qbins(rmean, nb) if side in ("row", "both") else np.zeros(R, np.int64)
            cb = r9.qbins(cmean, nb) if side in ("col", "both") else np.zeros(W, np.int64)
            nr = nb if side in ("row", "both") else 1
            nc = nb if side in ("col", "both") else 1
            ctx = (ew * nr + rb[rows]) * nc + cb[cols]
            tab = r9.baseline_table(y, ctx, nE * nr * nc, train)
            used = np.unique(ctx)
            side_bits = (R * math.log2(nr) if nr > 1 else 0) + (W * math.log2(nc) if nc > 1 else 0)
            cost = (side_bits + len(used) * (nsym - 1) * r9.PARAM_BITS) / N
            res[(side, nb)] = {"val": r9.ce_bits(tab, y, ctx, val), "test": r9.ce_bits(tab, y, ctx, test),
                               "cost": cost}
    pick = min(res, key=lambda k: res[k]["val"] - bval + res[k]["cost"])
    r = res[pick]
    gain = btest - r["test"]
    return {"base_val": bval, "base_test": btest, "codelen": r["test"], "net_test": r["test"],
            "gain": gain, "cost": r["cost"], "net": gain - r["cost"],
            "selection": {"chosen": list(pick),
                          "val_score": {f"{s}_{n}": v["val"] - bval + v["cost"] for (s, n), v in res.items()},
                          "test_gain_all": {f"{s}_{n}": btest - v["test"] for (s, n), v in res.items()}}}


def vc_sumbins(Cp, E, seed, split=(0.7, 0.1, 0.2)):
    """VC4: one context from row and column together. Each row (column) sends its quantile level
    (nq levels, log2(nq) bits) of mean |v| - E-mean; the level centres (2 nq values) cost 16 bits
    each; context = (E, quantile bin of row centre + column centre, nb bins). Table 16 bits per
    probability. (nq, nb) chosen on the validation part (CE + cost); split and baseline as VC3."""
    R, W = Cp.shape
    N = Cp.size
    rng = r9.rng_for(seed, 20)
    train, val, test = r9.inr_split(N, rng, split)
    vals, eb = r9.e_index(E)
    nE = len(vals)
    ew = np.repeat(eb.reshape(E.shape), B, axis=1).ravel()
    y = Cp.ravel().astype(np.int64)
    base_t = r9.baseline_table(y, ew, nE, train)
    bval, btest = r9.ce_bits(base_t, y, ew, val), r9.ce_bits(base_t, y, ew, test)
    X = r9.absv_resid(Cp, E)
    rmean, cmean = X.mean(axis=1), X.mean(axis=0)
    nsym = len(np.unique(y))
    res = {}
    for nq in (4, 8, 16):
        rq, cq = r9.qbins(rmean, nq), r9.qbins(cmean, nq)
        rc = np.array([rmean[rq == i].mean() for i in range(nq)])[rq]
        cc = np.array([cmean[cq == i].mean() for i in range(nq)])[cq]
        S = (rc[:, None] + cc[None, :]).ravel()
        for nb in (2, 4, 8, 16):
            ctx = ew * nb + r9.qbins(S, nb)
            tab = r9.baseline_table(y, ctx, nE * nb, train)
            used = np.unique(ctx)
            side_bits = (R + W) * math.log2(nq) + 2 * nq * r9.PARAM_BITS
            cost = (side_bits + len(used) * (nsym - 1) * r9.PARAM_BITS) / N
            res[(nq, nb)] = {"val": r9.ce_bits(tab, y, ctx, val), "test": r9.ce_bits(tab, y, ctx, test),
                             "cost": cost}
    pick = min(res, key=lambda k: res[k]["val"] - bval + res[k]["cost"])
    r = res[pick]
    gain = btest - r["test"]
    return {"base_val": bval, "base_test": btest, "codelen": r["test"], "net_test": r["test"],
            "gain": gain, "cost": r["cost"], "net": gain - r["cost"],
            "selection": {"chosen": list(pick),
                          "val_score": {f"q{q}_b{n}": v["val"] - bval + v["cost"] for (q, n), v in res.items()},
                          "test_gain_all": {f"q{q}_b{n}": btest - v["test"] for (q, n), v in res.items()}}}


def vc_edev(Cp, E, seed, split=(0.7, 0.1, 0.2)):
    """VC5: P(C' | E, bin of E - mean E of the block's row / block column). The decoder already has
    every scale (scales are coded separately), so the context costs no side information; only the
    table is charged (16 bits per probability). Side (row, column, both) and bins (2, 4, 8) chosen
    on the validation part (CE + cost). Split and baseline as VC3."""
    R, W = Cp.shape
    N = Cp.size
    rng = r9.rng_for(seed, 20)
    train, val, test = r9.inr_split(N, rng, split)
    vals, eb = r9.e_index(E)
    nE = len(vals)
    ew = np.repeat(eb.reshape(E.shape), B, axis=1).ravel()
    y = Cp.ravel().astype(np.int64)
    base_t = r9.baseline_table(y, ew, nE, train)
    bval, btest = r9.ce_bits(base_t, y, ew, val), r9.ce_bits(base_t, y, ew, test)
    Ef = E.astype(np.float64)
    dr = Ef - Ef.mean(axis=1, keepdims=True)
    dc = Ef - Ef.mean(axis=0, keepdims=True)
    nsym = len(np.unique(y))
    res = {}
    for side in ("row", "col", "both"):
        for nb in (2, 4, 8):
            br = r9.qbins(dr.ravel(), nb) if side in ("row", "both") else 0
            bc = r9.qbins(dc.ravel(), nb) if side in ("col", "both") else 0
            nr = nb if side in ("row", "both") else 1
            nc = nb if side in ("col", "both") else 1
            cb = np.repeat(((eb * nr + br) * nc + bc).reshape(E.shape), B, axis=1).ravel()
            tab = r9.baseline_table(y, cb, nE * nr * nc, train)
            used = np.unique(cb)
            cost = len(used) * (nsym - 1) * r9.PARAM_BITS / N
            res[(side, nb)] = {"val": r9.ce_bits(tab, y, cb, val), "test": r9.ce_bits(tab, y, cb, test),
                               "cost": cost}
    pick = min(res, key=lambda k: res[k]["val"] - bval + res[k]["cost"])
    r = res[pick]
    gain = btest - r["test"]
    return {"base_val": bval, "base_test": btest, "codelen": r["test"], "net_test": r["test"],
            "gain": gain, "cost": r["cost"], "net": gain - r["cost"],
            "selection": {"chosen": list(pick),
                          "val_score": {f"{a}_{n}": v["val"] - bval + v["cost"] for (a, n), v in res.items()},
                          "test_gain_all": {f"{a}_{n}": btest - v["test"] for (a, n), v in res.items()}}}


def vc_combo(Cp, E, seed, split=(0.7, 0.1, 0.2)):
    """VC6: VC3 context (paid row/column bins of mean |v| - E-mean) and VC5 context (free bin of
    E - row mean E) together: P(C' | E, VC3 bins, VC5 row bin). Grid: VC3 side in (row, col, both)
    x bins (2, 4, 8) x VC5 row bins (1, 2, 4, 8); chosen on the validation part (CE + cost)."""
    R, W = Cp.shape
    N = Cp.size
    rng = r9.rng_for(seed, 20)
    train, val, test = r9.inr_split(N, rng, split)
    vals, eb = r9.e_index(E)
    nE = len(vals)
    ew = np.repeat(eb.reshape(E.shape), B, axis=1).ravel()
    y = Cp.ravel().astype(np.int64)
    base_t = r9.baseline_table(y, ew, nE, train)
    bval, btest = r9.ce_bits(base_t, y, ew, val), r9.ce_bits(base_t, y, ew, test)
    X = r9.absv_resid(Cp, E)
    rmean, cmean = X.mean(axis=1), X.mean(axis=0)
    rows = np.repeat(np.arange(R), W)
    cols = np.tile(np.arange(W), R)
    Ef = E.astype(np.float64)
    dr = (Ef - Ef.mean(axis=1, keepdims=True)).ravel()
    nsym = len(np.unique(y))
    res = {}
    for side in ("row", "col", "both"):
        for nb in (2, 4, 8):
            rb = r9.qbins(rmean, nb) if side in ("row", "both") else np.zeros(R, np.int64)
            cb = r9.qbins(cmean, nb) if side in ("col", "both") else np.zeros(W, np.int64)
            nr = nb if side in ("row", "both") else 1
            nc = nb if side in ("col", "both") else 1
            side_bits = (R * math.log2(nr) if nr > 1 else 0) + (W * math.log2(nc) if nc > 1 else 0)
            for ne in (1, 2, 4, 8):
                be = r9.qbins(dr, ne) if ne > 1 else np.zeros(dr.size, np.int64)
                bw = np.repeat(be.reshape(E.shape), B, axis=1).ravel()
                ctx = ((ew * ne + bw) * nr + rb[rows]) * nc + cb[cols]
                tab = r9.baseline_table(y, ctx, nE * ne * nr * nc, train)
                used = np.unique(ctx)
                cost = (side_bits + len(used) * (nsym - 1) * r9.PARAM_BITS) / N
                res[(side, nb, ne)] = {"val": r9.ce_bits(tab, y, ctx, val),
                                       "test": r9.ce_bits(tab, y, ctx, test), "cost": cost}
    pick = min(res, key=lambda k: res[k]["val"] - bval + res[k]["cost"])
    r = res[pick]
    gain = btest - r["test"]
    return {"base_val": bval, "base_test": btest, "codelen": r["test"], "net_test": r["test"],
            "gain": gain, "cost": r["cost"], "net": gain - r["cost"],
            "selection": {"chosen": list(pick),
                          "val_score": {f"{a}_{n}_e{e}": v["val"] - bval + v["cost"] for (a, n, e), v in res.items()},
                          "test_gain_all": {f"{a}_{n}_e{e}": btest - v["test"] for (a, n, e), v in res.items()}}}


C_VARIANTS = {
    "VC1": {"hyp": "原设计的嵌入成本 (行数+列数)·8·16/N ≈ 0.08 bit/权重，可能高于网络能拿到的收益；"
                   "若行列结构主要是幅值的低秩尺度（如 C1 所测），2–4 维就够，成本可降到 1/4–1/2。",
            "struct": "原设计网络，嵌入维数在 {2, 4, 16} 中按验证集 (交叉熵 + 成本) 选择；其余超参数同原设计",
            "run": lambda Cp, E, seed, batch=None: vc_select(Cp, E, seed, (2, 4, 16), batch=batch)},
    "VC2": {"hyp": "003-R5 显示带符号的值几乎没有行列结构，位置信息若存在主要在幅值上；只预测 8 档幅值、"
                   "符号按 1 bit 计，输出更小、更不易过拟合。",
            "struct": "幅值输出（8 个 logit）+ 每个非零码字 1 bit 符号；嵌入维数在 {2, 4, 8} 中按验证集选择",
            "run": lambda Cp, E, seed, batch=None: vc_select(Cp, E, seed, (2, 4, 8), output="mag",
                                                             batch=batch)},
    "VC3": {"hyp": "C1 的严格收益只在 k=1 时出现且很小（行或列整体的幅值尺度），网络与 16-bit 因子的成本都高于"
                   "这点收益；只传每行/每列幅值尺度的分位档编号（log2(档数) bit），用 (E, 行档, 列档) 的计数表预测。",
            "struct": "P(C'|E, 行档, 列档)，行/列档为该行/列 |v|−E 均值的平均数的分位档；侧（行、列、两者）与档数 "
                      "{2,4,8,16} 按验证集 (交叉熵 + 成本) 选择；表每个概率 16 bit；划分与基准同 C2",
            "run": lambda Cp, E, seed, batch=None: vc_scalebins(Cp, E, seed)},
    "VC4": {"hyp": "VC3 的“两者”选项用 档数² 个上下文，表成本随档数平方增长，开发集上常只能选 2–4 档；若行、列"
                   "尺度大致相加（C1 的秩 1 结构），把行尺度与列尺度合成一个量再分档，可用更细的档、更少的表项。",
            "struct": "每行/列发送其 |v|−E 均值的分位级 (nq 级, log2 nq bit)，级中心 2·nq 个按 16 bit；上下文 = "
                      "(E, 行中心+列中心 的分位档, nb 档)；(nq, nb) ∈ {4,8,16}×{2,4,8,16} 按验证集 (交叉熵+成本) "
                      "选择；表每个概率 16 bit；划分与基准同 VC3",
            "run": lambda Cp, E, seed, batch=None: vc_sumbins(Cp, E, seed)},
    "VC5": {"hyp": "C1b 显示 scale 矩阵本身有行列结构（R² 至多 0.156）；块的 E 相对所在行/块列平均 E 偏高时，"
                   "可能是块内有离群大值把 scale 抬高、其余码字偏小（反之亦然）。scale 已单独编码，解码端免费可得，"
                   "这一上下文不需要侧信息。",
            "struct": "P(C'|E, E−行平均E 的分位档, E−块列平均E 的分位档)；侧（行、列、两者）与档数 {2,4,8} 按验证集 "
                      "(交叉熵+成本) 选择；表每个概率 16 bit，无侧信息；划分与基准同 VC3",
            "run": lambda Cp, E, seed, batch=None: vc_edev(Cp, E, seed)},
    "VC6": {"hyp": "VC3（付费的 |v| 行/列档）与 VC5（免费的 E−行平均E 档）在开发集上收益最大的矩阵不同"
                   "（VC3：L0/L39 两侧；VC5：L39.E61.w1 行侧），两者可能互补，合并后净值高于各自单独。",
            "struct": "P(C'|E, E−行平均E 档, |v| 行/列档)；VC3 侧 ∈ {行,列,两者} × 档 {2,4,8}，VC5 行档 ∈ {1,2,4,8}，"
                      "按验证集 (交叉熵+成本) 联合选择；成本同 VC3（侧信息 + 表 16 bit/概率）",
            "run": lambda Cp, E, seed, batch=None: vc_combo(Cp, E, seed)},
}


# ----------------------------------------------------------------------------- synthetic checks

def planted_anchor(g):
    """Every block: one codeword of magnitude 6 (random position and sign), the other 31 i.i.d.
    from the independent-base distribution restricted to magnitudes < 6."""
    p = np.where(r9_vb.MAG < 7, P_N, 0.0)
    p = p / p.sum()
    nb = R * W // B
    blk = g.choice(16, size=(nb, B), p=p).astype(np.uint8)
    pos = g.integers(0, B, nb)
    blk[np.arange(nb), pos] = np.where(g.random(nb) < 0.5, 7, 15)
    return blk.reshape(R, W)


def anchor_theory(g, n=40_000):
    """Exact entropy of planted_anchor per weight vs. its K = 1 (i.i.d.) code length."""
    p = np.where(r9_vb.MAG < 7, P_N, 0.0)
    p = p / p.sum()
    # i.i.d. marginal of a codeword
    marg = (31 * p + np.where(r9_vb.MAG == 7, 0.5, 0.0)) / 32
    h_iid = -float((marg[marg > 0] * np.log2(marg[marg > 0])).sum())
    h_true = (math.log2(32) + 1 + 31 * -float((p[p > 0] * np.log2(p[p > 0])).sum())) / 32
    return h_iid - h_true


def planted_shared(g):
    """Three scale values (block E in {126, 127, 128} w.p. 0.5, 0.3, 0.2); four types shared by all
    E (independent-base shapes with sigma 1.0, 1.5, 2.0, 2.6); priors depend on E."""
    from .r9_check import merged_probs
    types = [merged_probs(s) for s in (1.0, 1.5, 2.0, 2.6)]
    pri = {126: [0.6, 0.2, 0.1, 0.1], 127: [0.1, 0.4, 0.4, 0.1], 128: [0.1, 0.1, 0.2, 0.6]}
    nb = R * W // B
    ev = g.choice([126, 127, 128], size=nb, p=[0.5, 0.3, 0.2])
    blk = np.empty((nb, B), dtype=np.uint8)
    th = 0.0
    for e, w in pri.items():
        idx = np.nonzero(ev == e)[0]
        blk[idx] = mixture_blocks(g, types, w, len(idx))
        th += len(idx) / nb * theory_gain(types, w, g)
    return blk.reshape(R, W), ev.reshape(R, W // B).astype(np.uint8), th


def planted_rowtype(g, sig=(1.3, 1.8)):
    """Each row draws a type (1/2 each; sigma 1.3 or 1.8 independent-base shape truncated to
    magnitudes < 6); every block of the row: one magnitude 6 (random position and sign) plus 31
    codewords i.i.d. from the row's type. Theory = K = 1 code length minus the exact entropy given
    the row type (the type itself is not charged: VB4 sends row bins as side information)."""
    from .r9_check import merged_probs
    ps = []
    for s_ in sig:
        p = np.where(r9_vb.MAG < 7, merged_probs(s_), 0.0)
        ps.append(p / p.sum())
    t = g.integers(0, 2, R)
    nb = W // B
    blk = np.empty((R, nb, B), dtype=np.uint8)
    for i in range(2):
        rows = np.nonzero(t == i)[0]
        blk[rows] = g.choice(16, size=(len(rows), nb, B), p=ps[i])
    pos = g.integers(0, B, (R, nb))
    ri, ci = np.meshgrid(np.arange(R), np.arange(nb), indexing="ij")
    blk[ri, ci, pos] = np.where(g.random((R, nb)) < 0.5, 7, 15)

    def H(p):
        p = p[p > 0]
        return -float((p * np.log2(p)).sum())
    frac = np.bincount(t, minlength=2) / R
    marg = sum(f * (31 * p + np.where(r9_vb.MAG == 7, 0.5, 0.0)) / 32 for f, p in zip(frac, ps))
    h_true = sum(f * (math.log2(32) + 1 + 31 * H(p)) / 32 for f, p in zip(frac, ps))
    info = (f"每行 1/2 概率取 σ={sig[0]} 或 {sig[1]} 的独立底形状（截断到幅值<6）；该行每块恰一个幅值 6"
            "（随机位置、符号），其余 31 个按行类型独立抽取")
    return blk.reshape(R, W), H(marg) - h_true, info


def planted_edev(g):
    """Row offset o_r in {0, 1} (1/2 each); block E = 126 + o_r + u, u in {0, 1} (1/2 each). Blocks
    with u = 1 (E above the row mean) draw codewords from sigma 1.2, others from sigma 2.2. E alone
    (126/127/128) leaves u ambiguous for E = 127; E - row mean identifies u."""
    from .r9_check import merged_probs
    o = g.integers(0, 2, R)
    u = g.integers(0, 2, (R, W // B))
    E = (126 + o[:, None] + u).astype(np.uint8)
    p1, p0 = merged_probs(1.2), merged_probs(2.2)
    blk = np.where(np.repeat(u, B, axis=1) == 1, g.choice(16, size=(R, W), p=p1),
                   g.choice(16, size=(R, W), p=p0)).astype(np.uint8)
    return blk, E


def planted_edev_theory():
    """H(C'|E) - H(C'|E, u) for planted_edev (exact)."""
    from .r9_check import merged_probs
    p1, p0 = merged_probs(1.2), merged_probs(2.2)

    def H(p):
        p = p[p > 0]
        return -float((p * np.log2(p)).sum())
    # E = 126: u = 0 (o = 0); E = 128: u = 1 (o = 1); E = 127: u = 0 or 1 equally (prob 1/2 of E)
    h_e = 0.25 * H(p0) + 0.25 * H(p1) + 0.5 * H((p0 + p1) / 2)
    return h_e - 0.5 * (H(p0) + H(p1))


def synth_variant(vid):
    g = np.random.default_rng([r9.SEED, 500, int(vid[2:]), 0 if vid[1] == "B" else 1])
    out = {"variant": vid}
    E0 = const_E()
    null = base(g)
    if vid.startswith("VB"):
        rn = r9_vb.run_variant(vid, null, E0, 1)
        out["null"] = {k: rn[k] for k in ("gain", "cost", "net")}
        out["null_pass"] = rn["net"] <= 0.001
        if vid == "VB1":
            Cp = planted_anchor(g)
            th = anchor_theory(g)
            rp = r9_vb.run_variant(vid, Cp, E0, 2)
            out["planted"] = {"construction": "每块恰一个幅值 6（随机位置、随机符号），其余 31 个按独立底"
                              "截断到幅值<6 后独立抽取", "gain": rp["gain"], "theory": th,
                              "criterion": "收益与理论值相差 < 3%"}
            out["planted_pass"] = abs(rp["gain"] - th) / th < 0.03
        elif vid == "VB2":
            Cp, E, th = planted_shared(g)
            rp = r9_vb.run_variant(vid, Cp, E, 2)
            out["planted"] = {"construction": "3 个 E (比例 .5/.3/.2)，4 个跨 E 共用的类型 (σ = 1.0/1.5/2.0/2.6 "
                              "的独立底形状)，先验随 E 变", "gain": rp["gain"], "theory": th,
                              "selection": rp["selection"], "criterion": "收益与理论值相差 < 3%"}
            out["planted_pass"] = abs(rp["gain"] - th) / th < 0.03
        elif vid == "VB4":
            Cp, th, info = planted_rowtype(g)
            rp = r9_vb.run_variant(vid, Cp, E0, 2)
            r1 = r9_vb.run_variant("VB1", Cp, E0, 2)
            out["planted"] = {"construction": info, "gain": rp["gain"], "theory": th,
                              "vb1_gain": r1["gain"], "criterion": "收益与理论值相差 < 3%，且高于 VB1"}
            out["planted_pass"] = abs(rp["gain"] - th) / th < 0.03 and rp["gain"] > r1["gain"]
        elif vid == "VB3":
            from .r9_check import merged_probs, t3_data
            Cp = t3_data(g)
            th = theory_gain([merged_probs(1.2), merged_probs(2.2)], [0.5, 0.5], g)
            rp = r9_vb.run_variant(vid, Cp, E0, 2)
            out["planted"] = {"construction": "001 的 T3（每块 1/2 概率 σ=1.2 或 2.2）", "gain": rp["gain"],
                              "theory": th, "selection": rp["selection"],
                              "criterion": "收益与理论值相差 < 2%"}
            out["planted_pass"] = abs(rp["gain"] - th) / th < 0.02
    elif vid == "VC5":
        run = C_VARIANTS[vid]["run"]
        rn = run(null, E0, 1)
        out["null"] = {k: rn[k] for k in ("gain", "cost", "net")}
        out["null_pass"] = rn["net"] <= 0.001
        Cp, E1 = planted_edev(g)
        rp = run(Cp, E1, 2)
        out["planted"] = {"construction": "E 有行偏移（每行 ±1，各 1/2）；E 高于行平均的块其码字取 σ=1.2 的独立底，"
                          "其余取 σ=2.2；两种 E 值下码字分布相同的部分由 E 本身无法区分",
                          "gain": rp["gain"], "selection": rp["selection"],
                          "criterion": "选中行侧，且收益 >= 0.8 × 已知上下文时的理论值"}
        th = planted_edev_theory()
        out["planted"]["theory"] = th
        out["planted_pass"] = rp["selection"]["chosen"][0] in ("row", "both") and rp["gain"] >= 0.8 * th
    else:
        run = C_VARIANTS[vid]["run"]
        rn = run(null, E0, 1, batch=16384)
        out["null"] = {k: rn[k] for k in ("gain", "cost", "net")}
        out["null_pass"] = rn["net"] <= 0.001
        Cp = tc4_data(g)
        rp = run(Cp, E0, 2, batch=16384)
        out["planted"] = {"construction": "第 7 节 TC4（行、列各一个对数正态尺度相乘）", "gain": rp["gain"],
                          "selection": rp["selection"], "criterion": "收益 >= 0.14（同 TC4）"}
        out["planted_pass"] = rp["gain"] >= 0.14
    out["pass"] = bool(out["null_pass"] and out["planted_pass"])
    return out


# ----------------------------------------------------------------------------- model runs

def run_matrix(vid, matrix, snapshot, sel):
    index = index_snapshot(snapshot)
    Cp, E = load(index, sel["tensors"][matrix])
    seed = seed_of(matrix)
    if vid.startswith("VB"):
        return r9_vb.run_variant(vid, Cp, E, seed)
    return C_VARIANTS[vid]["run"](Cp, E, seed)


def job(kind, vid, matrix, snapshot, sel, work, stage):
    t0 = time.time()
    name = f"{stage}_{vid}" + (f"_{matrix}" if matrix else "")
    if kind == "synth":
        res = synth_variant(vid)
    else:
        res = run_matrix(vid, matrix, snapshot, sel)
    res = {"variant": vid, "matrix": matrix, "stage": stage, "result": res,
           "seconds": time.time() - t0}
    out = Path(work) / f"{name}.pkl"
    with open(out.with_suffix(".tmp"), "wb") as f:
        pickle.dump(res, f)
    out.with_suffix(".tmp").rename(out)
    return name, res["seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("synth", "dev", "conf"))
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--snapshot")
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=60)
    args = ap.parse_args()
    with open(SAMPLES) as f:
        sel = json.load(f)
    nm = names(sel)
    work = str(Path(args.work).resolve())
    todo = []
    for vid in args.variants:
        if args.stage == "synth":
            todo.append(("synth", vid, None))
            continue
        sp = Path(work) / f"synth_{vid}.pkl"
        if not sp.exists() or not pickle.load(open(sp, "rb"))["result"]["pass"]:
            raise SystemExit(f"{vid}: synthetic check missing or failed; not run on model data")
        part = "b" if vid.startswith("VB") else "c"
        for m in nm[f"{part}_{args.stage}"]:
            todo.append(("model", vid, m))
    todo = [t for t in todo if not (Path(work) / (f"{args.stage}_{t[1]}" + (f"_{t[2]}" if t[2] else "")
                                                   + ".pkl")).exists()]
    print(f"{len(todo)} jobs", flush=True)
    t0 = time.time()
    nfail = 0
    with ProcessPoolExecutor(min(args.jobs, max(1, len(todo)))) as ex:
        futs = {ex.submit(job, k, v, m, args.snapshot, sel, work, args.stage): (v, m) for k, v, m in todo}
        for fut in as_completed(futs):
            try:
                n, sec = fut.result()
                print(f"done {n}: {sec:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
            except Exception:
                nfail += 1
                print(f"FAILED {futs[fut]}\n{traceback.format_exc()}", flush=True)
    print(f"{len(todo) - nfail} ok, {nfail} failed, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
