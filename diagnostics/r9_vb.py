"""Spec 005 section 5, part B variants: block generative models beyond the original mixture.

Every variant uses the B1 protocol: blocks of even rows fit / odd rows evaluate, then swap; gain is
measured against the same K = 1 code length (P(C'|E), +0.5 over the symbols of that E) as B1, and
every stored parameter costs 16 bits. Hyper-parameters that are selected (K) are selected inside
the fitting half (rows = 0 mod 4 vs 2 mod 4 of that half), never on the evaluation half.
"""
import math

import numpy as np

from . import r9

B = r9.B
LN2 = math.log(2.0)
MAG = np.arange(16) & 7
LOGC = np.array([math.lgamma(B + 1) - math.lgamma(k + 1) - math.lgamma(B - k + 1)
                 for k in range(B + 1)]) / LN2  # log2 C(32, k)


def setup(Cp, E):
    H = r9.block_hist(Cp)
    vals, ei = r9.e_index(E)
    nE = len(vals)
    A = r9.alphabet_per_e(H, ei, nE)
    rows = r9.block_rows(Cp)
    return {"H": H, "vals": vals, "ei": ei, "nE": nE, "A": A, "rows": rows,
            "key": r9.block_key(Cp), "N": Cp.size}


def baseline_bits(S, fit, ev):
    """K = 1 code length (bits, summed) of blocks `ev` with tables from blocks `fit`, per E."""
    bits = 0.0
    for k in range(S["nE"]):
        f = fit[S["ei"][fit] == k]
        e = ev[S["ei"][ev] == k]
        if not len(e):
            continue
        cnt = S["H"][f][:, S["A"][k]].sum(axis=0, dtype=np.float64) + 0.5
        lt = np.log2(cnt / cnt.sum())
        bits -= float((S["H"][e][:, S["A"][k]].astype(np.float64) @ lt).sum())
    return bits


def protocol(S, fit_eval, select=None):
    """Run `fit_eval(S, fit_idx, ev_idx, hp)` on both halves. If `select` is a list of
    hyper-parameter values, choose per half by fitting on rows 0 mod 4 (of the half) and
    evaluating on rows 2 mod 4 (of the half). Returns codelen, gain, cost, net (bits/weight)."""
    lens, base, costs, chosen = [], [], [], []
    for a in (0, 1):
        fit = np.nonzero(S["rows"] % 2 == a)[0]
        ev = np.nonzero(S["rows"] % 2 == 1 - a)[0]
        hp = None
        if select is not None:
            f0 = fit[(S["rows"][fit] // 2) % 2 == 0]
            f1 = fit[(S["rows"][fit] // 2) % 2 == 1]
            scores = {}
            for h in select:
                r = fit_eval(S, f0, f1, h)
                scores[h] = r["bits"] + r["cost_bits"]
            hp = min(select, key=lambda h: scores[h])
            chosen.append({"chosen": hp, "val_bits": {str(k): v for k, v in scores.items()}})
        r = fit_eval(S, fit, ev, hp)
        nw = len(ev) * B
        lens.append(r["bits"] / nw)
        base.append(baseline_bits(S, fit, ev) / nw)
        costs.append(r["cost_bits"] / S["N"])
    codelen, b1 = float(np.mean(lens)), float(np.mean(base))
    cost = float(max(costs))
    return {"codelen": codelen, "codelen1": b1, "gain": b1 - codelen, "cost": cost,
            "net": b1 - codelen - cost, "halves": lens, "selection": chosen}


# ----------------------------------------------------------------------------- VB1 max-anchored

def block_max(H):
    """Max magnitude index (0..7) and number of codewords at that magnitude, per block."""
    m = np.zeros(len(H), dtype=np.int64)
    for c in range(16):
        m = np.where(H[:, c] > 0, np.maximum(m, MAG[c]), m)
    n = np.zeros(len(H), dtype=np.int64)
    for c in range(16):
        n += np.where(MAG[c] == m, H[:, c], 0)
    return m, n


def fe_anchored(K):
    """VB1: P(block) = P(mx | E) P(n_mx | E, mx) / C(32, n_mx) * P(signs of maxima | E)
    * sum_z pi_{z|E,mx} prod theta_{z|E,mx}(rest). Rest = codewords of magnitude < mx."""

    def fe(S, fit, ev, hp):
        Kq = hp if hp is not None else K
        H = S["H"]
        mx, nmx = S["mx"], S["nmx"]
        bits = 0.0
        cost = 0
        for k in range(S["nE"]):
            fk = fit[S["ei"][fit] == k]
            ek = ev[S["ei"][ev] == k]
            mx_alpha = np.unique(mx[S["ei"] == k])
            cm = np.bincount(mx[fk], minlength=8)[mx_alpha] + 0.5
            lpm = dict(zip(mx_alpha.tolist(), np.log2(cm / cm.sum())))
            cost += len(mx_alpha)
            # sign of the maxima: P(negative) per E
            Hk = H[fk]
            neg = sum(float((Hk[:, c] * (mx[fk] == MAG[c])).sum()) for c in range(9, 16))
            pos = sum(float((Hk[:, c] * (mx[fk] == MAG[c])).sum()) for c in range(1, 8))
            pn = (neg + 0.5) / (neg + pos + 1.0)
            cost += 1
            for m in mx_alpha:
                fm = fk[mx[fk] == m]
                em = ek[mx[ek] == m]
                cn = np.bincount(nmx[fm], minlength=B + 1)[1:] + 0.5
                lpn = np.log2(cn / cn.sum())
                cost += B
                rest = (MAG < m) & S["A"][k]
                if m == 0 or not rest.any():
                    model = None
                else:
                    Kk = r9.k_eff(len(fm), Kq)
                    model = r9.em_fit(H[fm][:, rest], Kk, S["key"][fm], r9.rng_for(S["seed"], Kq, k, m))
                    cost += Kk * (int(rest.sum()) + 1)
                if not len(em):
                    continue
                he = H[em]
                b = -lpm[int(m)] - lpn[nmx[em] - 1] + LOGC[nmx[em]]
                if m > 0:
                    nneg = he[:, int(m) + 8].astype(np.float64)
                    npos = he[:, int(m)].astype(np.float64)
                    b = b - nneg * math.log2(pn) - npos * math.log2(1 - pn)
                if model is not None:
                    b = b + r9.mix_bits(he[:, rest], model)
                bits += float(b.sum())
        return {"bits": bits, "cost_bits": cost * r9.PARAM_BITS}

    return fe


# ----------------------------------------------------------------------------- VB2 shared types

def em_shared(Hs, ei, nE, K, key, rng, iters=r9.EM_ITERS, inits=(0, 1, 2)):
    """Mixture whose type distributions are shared by all E values; priors depend on E."""
    n = len(Hs)
    best = None
    Hd = Hs.astype(np.float64)  # cast once (numerically identical to casting every iteration)
    for kind in inits:
        sr, rh = r9._init_stats(Hs, K, kind, rng, key)
        # initial per-E priors from the same initial responsibilities (uniform within E)
        lpi = np.log(np.full((nE, K), 1.0 / K))
        lt = r9._mstep(sr, rh, n)[1]
        for it in range(iters + 1):
            L = Hd @ lt.T + lpi[ei]
            z = r9._lse(L)
            if it == iters:
                break
            r = np.exp(L - z[:, None])
            rh = r.T @ Hd
            th = rh + 0.5
            lt = np.log(th / th.sum(axis=1, keepdims=True))
            pe = np.stack([np.bincount(ei, weights=r[:, j], minlength=nE) for j in range(K)], 1)
            pe = np.maximum(pe / np.maximum(pe.sum(axis=1, keepdims=True), 1e-300), 1e-300)
            lpi = np.log(pe)
        ll = float(z.sum())
        if best is None or ll > best["ll"]:
            best = {"lpi": lpi, "lt": lt, "ll": ll, "init": kind}
    return best


def fe_shared(K):
    def fe(S, fit, ev, hp):
        Kq = hp if hp is not None else K
        A = S["A"].any(axis=0)
        Hf = S["H"][fit][:, A]
        Kk = int(max(1, min(Kq, len(fit) // r9.MIN_PER_CLASS)))
        m = em_shared(Hf, S["ei"][fit], S["nE"], Kk, S["key"][fit], r9.rng_for(S["seed"], Kq, 5))
        He = S["H"][ev][:, A].astype(np.float64)
        L = He @ m["lt"].T + m["lpi"][S["ei"][ev]]
        bits = float(-r9._lse(L).sum() / LN2)
        cost = Kk * (int(A.sum())) + S["nE"] * Kk
        return {"bits": bits, "cost_bits": cost * r9.PARAM_BITS}

    return fe


# ----------------------------------------------------------------------------- VB4 anchored + position context

def position_context(Cp, E, nb):
    """Per block: row bin (quantile bin of the row mean of |v| - E-mean) and block-column bin (same
    over the 32 columns of the block column), nb bins each; context = row bin * nb + column bin.
    Side information: (R + W/32) * log2(nb) bits."""
    R, W = Cp.shape
    X = r9.absv_resid(Cp, E)
    rb = r9.qbins(X.mean(axis=1), nb)
    cb = r9.qbins(X.reshape(R, W // B, B).mean(axis=(0, 2)), nb)
    ctx = (np.repeat(rb, W // B) * nb + np.tile(cb, R)).astype(np.int64)
    return ctx, (R + W // B) * math.log2(nb)


def fe_anchored_ctx(K):
    """VB4: VB1 with P(mx | E, ctx) and the rest-mixture priors pi_{z | E, mx, ctx} (types shared
    over ctx inside each (E, mx)); ctx = position_context (row bin x block-column bin)."""

    def fe(S, fit, ev, hp):
        H = S["H"]
        mx, nmx, ctx, nc = S["mx"], S["nmx"], S["ctx"], S["nctx"]
        bits = 0.0
        cost = 0
        for k in range(S["nE"]):
            fk = fit[S["ei"][fit] == k]
            ek = ev[S["ei"][ev] == k]
            mx_alpha = np.unique(mx[S["ei"] == k])
            na = len(mx_alpha)
            pos = {int(m): i for i, m in enumerate(mx_alpha)}
            cm = np.zeros((nc, na))
            np.add.at(cm, (ctx[fk], np.array([pos[int(m)] for m in mx[fk]], dtype=np.int64)), 1.0)
            lpm = np.log2((cm + 0.5) / (cm + 0.5).sum(axis=1, keepdims=True))
            cost += nc * na
            Hk = H[fk]
            neg = sum(float((Hk[:, c] * (mx[fk] == MAG[c])).sum()) for c in range(9, 16))
            psg = sum(float((Hk[:, c] * (mx[fk] == MAG[c])).sum()) for c in range(1, 8))
            pn = (neg + 0.5) / (neg + psg + 1.0)
            cost += 1
            for m in mx_alpha:
                fm = fk[mx[fk] == m]
                em = ek[mx[ek] == m]
                cn = np.bincount(nmx[fm], minlength=B + 1)[1:] + 0.5
                lpn = np.log2(cn / cn.sum())
                cost += B
                rest = (MAG < m) & S["A"][k]
                model = None
                if m > 0 and rest.any() and len(fm):
                    Kk = r9.k_eff(len(fm), K)
                    model = em_shared(H[fm][:, rest], ctx[fm], nc, Kk, S["key"][fm],
                                      r9.rng_for(S["seed"], K, k, m, 4))
                    cost += Kk * int(rest.sum()) + nc * Kk
                if not len(em):
                    continue
                he = H[em]
                b = -lpm[ctx[em], pos[int(m)]] - lpn[nmx[em] - 1] + LOGC[nmx[em]]
                if m > 0:
                    b = b - he[:, int(m) + 8] * math.log2(pn) - he[:, int(m)] * math.log2(1 - pn)
                if model is not None:
                    L = he[:, rest].astype(np.float64) @ model["lt"].T + model["lpi"][ctx[em]]
                    b = b - r9._lse(L) / LN2
                bits += float(b.sum())
        return {"bits": bits, "cost_bits": cost * r9.PARAM_BITS + S["side_bits"]}

    return fe


# ----------------------------------------------------------------------------- VB3 auto K

def fe_mixture(S, fit, ev, hp):
    """The original B1 mixture with K = hp, as a protocol function (for K selection)."""
    bits = 0.0
    cost = 0
    for k in range(S["nE"]):
        fk = fit[S["ei"][fit] == k]
        ek = ev[S["ei"][ev] == k]
        Kk = r9.k_eff(len(fk), hp)
        m = r9.em_fit(S["H"][fk][:, S["A"][k]], Kk, S["key"][fk], r9.rng_for(S["seed"], hp, 7, k))
        cost += Kk * 16
        if len(ek):
            bits += float(r9.mix_bits(S["H"][ek][:, S["A"][k]], m).sum())
    return {"bits": bits, "cost_bits": cost * r9.PARAM_BITS}


VARIANTS = {
    "VB1": {"hyp": "量化器按块 amax 定 scale，块内最大幅值及其个数是确定性的痕迹（002 第 5 行 m_max、"
                   "004-D 的 blkmax 分量）；独立同分布的混合无法表达\"恰好一个最大值\"。把最大幅值、"
                   "其个数和符号单独建模，其余码字在 (E, 最大幅值) 内按 K 类混合独立抽取。",
            "struct": "P(mx|E)·P(n_mx|E,mx)/C(32,n_mx)·P(符号|E)·Σ_z π_z Π θ_z(其余)，每个 (E,mx) 一个 "
                      "K=32 的混合（EM 150 轮、3 组初值，块数不足按 500·K 降 K）；全部表 +0.5 平滑",
            "make": lambda: (fe_anchored(32), None), "needs_max": True},
    "VB2": {"hyp": "块的类型只描述块内码字的相对形状，与 scale 无关（scale 已按 amax 归一化）；类型分布"
                   "跨 E 共用、只有先验随 E 变，可以用更大的 K 而不增加多少参数。",
            "struct": "θ_z 跨 E 共用，π_{z|E} 各 E 一组；K 在 {16, 32, 64, 128} 中按拟合半内部的验证选择",
            "make": lambda: (fe_shared(64), [16, 32, 64, 128]), "needs_max": False},
    "VB3": {"hyp": "原设计 K=32 是固定值；若收益随 K 仍在增长，更大的 K（由验证选择）可提高净值。",
            "struct": "原设计混合，K 在 {16, 32, 64, 128} 中按拟合半内部的验证选择（每个 E 仍按 500·K 降 K）",
            "make": lambda: (fe_mixture, [16, 32, 64, 128]), "needs_max": False},
    "VB4": {"hyp": "C 部分 VC3 显示行、列整体的幅值尺度带有少量信息（开发集净值至多约 0.004）；VB1 的块类型"
                   "先验和最大幅值分布若随块所在行档、列档变化，可在 VB1 之上再取得一部分。",
            "struct": "VB1，但 P(mx|E, 位置档) 与其余码字混合的先验 π_{z|E,mx,位置档} 随位置档变化（类型 θ_z "
                      "在同一 (E,mx) 内跨位置档共用）；位置档 = 行档(4) × 块列档(4)，档为该行/该块列 |v|−E 均值"
                      "的平均数的分位档，作为侧信息按 (R+W/32)·2 bit 计；K=32，EM 同 VB1",
            "make": lambda: (fe_anchored_ctx(32), None), "needs_max": True, "ctx_bins": 4},
}


def run_variant(vid, Cp, E, seed, override=None):
    v = VARIANTS[vid]
    fe, select = v["make"]() if override is None else override
    S = setup(Cp, E)
    S["seed"] = seed
    if v.get("needs_max"):
        S["mx"], S["nmx"] = block_max(S["H"])
    S["side_bits"] = 0.0
    if v.get("ctx_bins"):
        S["ctx"], S["side_bits"] = position_context(Cp, E, v["ctx_bins"])
        S["nctx"] = v["ctx_bins"] ** 2
    return protocol(S, fe, select)
