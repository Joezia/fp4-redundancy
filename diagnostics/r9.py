"""Spec 005: R9 traces and generative models of FP4 blocks and positions.

A  coarse-grid trace: n_odd histograms, grid level g, held-out H(C'|E) - H(C'|E,g) - label cost.
A4 candidate quantisation rules on synthetic weights.
B  latent-class (mixture of multinomials) model of each block, per scale value E, fitted by EM.
C1 low rank of |v| - E-mean (quadrant hold-out of 003-R5.2) and of the scale matrix (C1b).
C2 coordinate network (row/column embeddings + MLP) predicting P(C' | position), NumPy only.

Everything works on merged codewords C' (R, W) and block scales E (R, W/B). Bits throughout.
"""
import math

import numpy as np

from . import est
from .measure import E2M1
from .r58 import holdout_lowrank

B = 32
SEED = 20261115
LN2 = math.log(2.0)
ABSV = np.abs(E2M1)
ODD = np.zeros(16, dtype=bool)
ODD[[1, 3, 9, 11]] = True  # magnitudes 0.5 and 1.5
CP15 = np.array([c for c in range(16) if c != 8])
PARAM_BITS = 16


def rng_for(*parts):
    return np.random.default_rng([SEED, *[int(p) for p in parts]])


def e_index(E):
    """Distinct scale values (sorted), per-block index (flat, row-major blocks)."""
    vals, inv = np.unique(E.ravel(), return_inverse=True)
    return vals, inv.astype(np.int64)


def block_hist(Cp):
    """(nblocks, 16) codeword counts per block (blocks in row-major order, aligned with E.ravel())."""
    blk = Cp.reshape(-1, B)
    H = np.empty((blk.shape[0], 16), dtype=np.uint8)
    for c in range(16):
        H[:, c] = (blk == c).sum(axis=1)
    return H


def block_rows(Cp):
    R, W = Cp.shape
    return np.repeat(np.arange(R), W // B)


# ============================================================================ A

def grid_level(Cp):
    """Coarsest grid holding every magnitude of the block: 0 has 0.5/1.5, 1 has 1/3 (no odd),
    2 has 2/6 (all in {0,2,4,6}), 3 has 4 (all in {0,4}), 4 all zero."""
    m = (Cp & 7).reshape(-1, B)
    pres = np.stack([(m == k).any(axis=1) for k in range(8)], axis=1)
    g = np.full(m.shape[0], 4, dtype=np.int64)
    g[pres[:, 6]] = 3
    g[pres[:, 4] | pres[:, 7]] = 2
    g[pres[:, 2] | pres[:, 5]] = 1
    g[pres[:, 1] | pres[:, 3]] = 0
    return g


def n_odd(Cp):
    return ODD[Cp].reshape(-1, B).sum(axis=1)


def binom_pmf(n, q):
    k = np.arange(n + 1)
    lc = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in k])
    with np.errstate(divide="ignore"):
        lp = lc + k * np.log(q) + (n - k) * np.log1p(-q) if 0 < q < 1 else None
    if lp is None:
        out = np.zeros(n + 1)
        out[0 if q <= 0 else n] = 1.0
        return out
    return np.exp(lp)


def a_matrix(Cp, E, Cs):
    """A1-A3 for one matrix. Cs = S0E version of Cp (same E). Returns (per-E rows, matrix dict)."""
    R, W = Cp.shape
    vals, ei = e_index(E)
    nE = len(vals)
    no, nos = n_odd(Cp), n_odd(Cs)
    g, gs = grid_level(Cp), grid_level(Cs)
    rows = []
    for k, e in enumerate(vals):
        sel = ei == k
        nb = int(sel.sum())
        h = np.bincount(no[sel], minlength=B + 1) / nb
        hs = np.bincount(nos[sel], minlength=B + 1) / nb
        q = float(no[sel].sum()) / (nb * B)
        pb = binom_pmf(B, q)
        gd = np.bincount(g[sel], minlength=5) / nb
        gds = np.bincount(gs[sel], minlength=5) / nb
        rows.append({"E": int(e), "n_blocks": nb, "q_odd": q, "frac0_real": float(h[0]),
                     "frac0_binom": float(pb[0]), "frac0_S0E": float(hs[0]),
                     "tv_real_binom": float(0.5 * np.abs(h - pb).sum()),
                     "tv_S0E_binom": float(0.5 * np.abs(hs - pb).sum()),
                     "hist_real": h.tolist(), "hist_S0E": hs.tolist(), "hist_binom": pb.tolist(),
                     **{f"g{j}_real": float(gd[j]) for j in range(5)},
                     **{f"g{j}_S0E": float(gds[j]) for j in range(5)}})
    # A3: held-out (row parity) H(C'|E) - H(C'|E,g) - H(g|E)/B, all on the same halves.
    ctx = np.repeat(ei, B)
    gw = np.repeat(g, B)
    half = np.repeat(np.arange(R) % 2, W)
    h_e, _, n = est.holdout_cross_entropy(ctx, Cp.ravel(), half, nE)
    h_eg, _, _ = est.holdout_cross_entropy(ctx * 5 + gw, Cp.ravel(), half, nE * 5)
    hb = block_rows(Cp) % 2
    h_g, _, _ = est.holdout_cross_entropy(ei, g, hb, nE, k=5)
    gain = h_e - h_eg
    mat = {"h_E": h_e, "h_Eg": h_eg, "gain": gain, "label_bits_per_block": h_g,
           "label_cost": h_g / B, "net": gain - h_g / B, "n": n, "K_E": nE}
    return rows, mat


def rule_stats(Cp, E):
    """A4 statistics (also computed on model matrices for the real-data row)."""
    m = (Cp & 7).reshape(-1, B)
    mx = ABSV[m.max(axis=1)]  # m is monotone in magnitude
    vals, ei = e_index(E)
    cnt = np.bincount(ei, minlength=len(vals))
    top = np.argsort(-cnt, kind="stable")[:2]
    no = n_odd(Cp)
    q = float(no.sum()) / no.size / B
    frac0 = float((no == 0).mean())
    # binomial reference per E, weighted by block counts (same as A1)
    b0 = 0.0
    for k in range(len(vals)):
        sel = ei == k
        qk = float(no[sel].sum()) / (sel.sum() * B)
        b0 += sel.sum() * (1 - qk) ** B
    b0 /= len(no)
    mh = np.bincount(m.ravel(), minlength=8)
    order = "".join(sorted(["a", "b", "c"], key=lambda x: -mh[{"a": 1, "b": 2, "c": 3}[x]]))
    order = order.replace("a", "0.5>").replace("b", "1>").replace("c", "1.5>").rstrip(">")
    out = {"max3": float((mx == 3).mean()), "max4": float((mx == 4).mean()),
           "max6": float((mx == 6).mean()), "frac0": frac0, "frac0_binom": b0,
           "frac0_ratio": frac0 / b0 if b0 > 0 else float("inf"), "q_odd_all": q,
           "count_order": order, "share_0.5": float(mh[1] / mh.sum()),
           "share_1": float(mh[2] / mh.sum()), "share_1.5": float(mh[3] / mh.sum())}
    for j, k in enumerate(top):
        sel = ei == k
        out[f"top{j + 1}_E"] = int(vals[k])
        out[f"top{j + 1}_share"] = float(cnt[k] / cnt.sum())
        out[f"top{j + 1}_q_odd"] = float(no[sel].sum()) / (sel.sum() * B)
    return out


GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MID = (GRID[1:] + GRID[:-1]) / 2


def quant_rule(Wt, rule):
    """Synthetic quantiser. Wt (R, W) float -> (C' uint8, E uint8). Nearest rounding; ties go to the
    smaller magnitude (searchsorted on midpoints, same convention as synth.quantize)."""
    R, W = Wt.shape
    X = Wt.reshape(R, W // B, B)
    amax = np.abs(X).max(axis=2)
    amax = np.where(amax > 0, amax, 1.0)
    if rule == "F":
        e = np.floor(np.log2(amax)).astype(np.int64) - 2
    else:
        e = np.ceil(np.log2(amax / 6.0)).astype(np.int64)
    Y = np.abs(X) / np.exp2(e)[:, :, None]
    m = np.searchsorted(MID, np.minimum(Y, 6.0), side="left")
    if rule == "C+":
        while True:
            can = m.max(axis=2) <= 5  # every code <= 3 (index 5)
            can &= m.max(axis=2) > 0
            if not can.any():
                break
            dbl = np.array([0, 2, 4, 5, 6, 7, 7, 7])  # 0->0, .5->1, 1->2, 1.5->3, 2->4, 3->6
            m = np.where(can[:, :, None], dbl[m], m)
            e = np.where(can, e - 1, e)
    s = (X < 0) & (m > 0)
    C = (m + 8 * s).astype(np.uint8).reshape(R, W)
    Eo = np.clip(e + 127, 0, 255).astype(np.uint8)
    return C, Eo


def synth_weights(rng, dist, tau, shape=(1024, 2048)):
    R, W = shape
    if dist == "normal":
        x = rng.standard_normal(shape)
    elif dist == "laplace":
        x = rng.laplace(size=shape)
    elif dist == "t4":
        x = rng.standard_t(4, size=shape)
    else:
        raise ValueError(dist)
    if tau:
        s = np.exp(tau * rng.standard_normal((R, W // B)))
        x = x * np.repeat(s, B, axis=1)
    return x


# ============================================================================ B: mixture model

EM_ITERS = 150
EM_CHUNK = 1 << 18
MIN_PER_CLASS = 500


def _lse(L):
    mx = L.max(axis=1, keepdims=True)
    return mx[:, 0] + np.log(np.exp(L - mx).sum(axis=1))


def _estep(Hs, logpi, logth, want_r=True, chunk=EM_CHUNK, wts=None):
    """Returns total natural-log likelihood and (if want_r) sufficient stats (sum r, r^T H).
    wts: optional multiplicity of each row (identical block histograms merged).
    One exp per entry, in place (the log-sum-exp and the responsibilities share it)."""
    n, A = Hs.shape
    K = len(logpi)
    ll = 0.0
    sr = np.zeros(K)
    rh = np.zeros((K, A))
    lt = np.ascontiguousarray(logth.T)
    for i in range(0, n, chunk):
        h = Hs[i:i + chunk]
        if h.dtype != np.float64:
            h = h.astype(np.float64)
        L = h @ lt
        L += logpi
        mx = L.max(axis=1, keepdims=True)
        L -= mx
        np.exp(L, out=L)
        s = L.sum(axis=1, keepdims=True)
        z = mx[:, 0] + np.log(s[:, 0])
        w = None if wts is None else wts[i:i + chunk].astype(np.float64)
        ll += float(z.sum()) if w is None else float(z @ w)
        if want_r:
            if w is not None:
                s /= w[:, None]
            L /= s
            sr += L.sum(axis=0)
            rh += L.T @ h
    return ll, sr, rh


def _mstep(sr, rh, n, alpha=0.5):
    pi = np.maximum(sr / n, 1e-300)
    th = rh + alpha
    th = th / th.sum(axis=1, keepdims=True)
    return np.log(pi), np.log(th)


def _init_stats(Hs, K, kind, rng, key, chunk=EM_CHUNK, wts=None):
    """Sufficient statistics of an initial responsibility matrix.
    kind 0: K equal-size groups by quantile of `key` (block mean |v|), ties broken by order;
    kind 1: the same group sizes with blocks assigned at random; kind 2: Dirichlet(1) random."""
    n, A = Hs.shape
    w = np.ones(n) if wts is None else wts.astype(np.float64)
    if kind in (0, 1):
        # equal-weight groups by quantile of key (with multiplicities when rows are merged)
        order = np.argsort(key, kind="stable")
        cum = np.cumsum(w[order]) - w[order]
        grp = np.empty(n, dtype=np.int64)
        grp[order] = np.minimum((cum * K) // w.sum(), K - 1).astype(np.int64)
        if kind == 1:
            grp = grp[rng.permutation(n)]
        sr = np.bincount(grp, weights=w, minlength=K)
        rh = np.zeros((K, A))
        for a in range(A):
            rh[:, a] = np.bincount(grp, weights=w * Hs[:, a], minlength=K)
        return sr, rh
    sr = np.zeros(K)
    rh = np.zeros((K, A))
    for i in range(0, n, chunk):
        h = Hs[i:i + chunk].astype(np.float64)
        r = rng.dirichlet(np.ones(K), size=len(h)) * w[i:i + chunk, None]
        sr += r.sum(axis=0)
        rh += r.T @ h
    return sr, rh


def em_fit(Hs, K, key, rng, iters=EM_ITERS, inits=(0, 1, 2), wts=None):
    """Fit a K-class mixture of multinomials to block histograms Hs (n, A). Returns dict with
    log pi, log theta, training log-likelihood (nats) and the chosen initialisation.
    wts: multiplicities when identical histograms were merged (same fixed point as unmerged EM)."""
    n = len(Hs) if wts is None else float(wts.sum())
    if K == 1 or n == 0:
        sr = np.array([float(max(n, 1))])
        rh = (Hs.sum(axis=0, dtype=np.float64) if wts is None else
              wts.astype(np.float64) @ Hs.astype(np.float64))[None, :]
        lp, lt = _mstep(sr, rh, max(n, 1))
        ll = _estep(Hs, lp, lt, want_r=False, wts=wts)[0] if n else 0.0
        return {"logpi": lp, "logth": lt, "ll": ll, "init": 0, "lls": [ll]}
    best = None
    lls = []
    Hs = np.ascontiguousarray(Hs, dtype=np.float64)
    for kind in inits:
        sr, rh = _init_stats(Hs, K, kind, rng, key, wts=wts)
        lp, lt = _mstep(sr, rh, n)
        for _ in range(iters):
            ll, sr, rh = _estep(Hs, lp, lt, wts=wts)
            lp, lt = _mstep(sr, rh, n)
        ll = _estep(Hs, lp, lt, want_r=False, wts=wts)[0]
        lls.append(ll)
        if best is None or ll > best["ll"]:
            best = {"logpi": lp, "logth": lt, "ll": ll, "init": kind}
    best["lls"] = lls
    return best


def mix_bits(Hs, model):
    """Code length in bits of each block under the mixture (no labels transmitted)."""
    out = []
    lt = np.ascontiguousarray(model["logth"].T)
    for i in range(0, len(Hs), EM_CHUNK):
        h = Hs[i:i + EM_CHUNK].astype(np.float64)
        L = h @ lt
        L += model["logpi"]
        out.append(-_lse(L) / LN2)
    return np.concatenate(out) if out else np.zeros(0)


def k_eff(n, K):
    return int(max(1, min(K, n // MIN_PER_CLASS)))


def block_key(Cp):
    return ABSV[Cp].reshape(-1, B).mean(axis=1)


def alphabet_per_e(H, ei, nE):
    """Symbols occurring under each E value in the whole matrix (bool (nE, 16))."""
    A = np.zeros((nE, 16), dtype=bool)
    for k in range(nE):
        A[k] = H[ei == k].sum(axis=0) > 0
    return A


def b1_matrix(Cp, E, seed, Ks=(1, 2, 4, 8, 16, 32), iters=EM_ITERS):
    """B1: even block rows fit / odd evaluate and swap. Returns per-K codelen, gain, cost, net."""
    R, W = Cp.shape
    N = Cp.size
    H = block_hist(Cp)
    vals, ei = e_index(E)
    nE = len(vals)
    A = alphabet_per_e(H, ei, nE)
    half = block_rows(Cp) % 2
    key = block_key(Cp)
    res = {}
    for K in Ks:
        lens, keff = [], []
        for a in (0, 1):
            bits = 0.0
            ke = {}
            for k in range(nE):
                fit = np.nonzero((ei == k) & (half == a))[0]
                ev = np.nonzero((ei == k) & (half == 1 - a))[0]
                Kk = k_eff(len(fit), K)
                ke[int(vals[k])] = Kk
                Hs = H[fit][:, A[k]]
                m = em_fit(Hs, Kk, key[fit], rng_for(seed, K, a, k), iters=iters)
                if len(ev):
                    bits += float(mix_bits(H[ev][:, A[k]], m).sum())
            nev = int((half == 1 - a).sum()) * B
            lens.append(bits / nev)
            keff.append(ke)
        cost = max(sum(d.values()) for d in keff) * PARAM_BITS * 16 / N
        res[K] = {"codelen": float(np.mean(lens)), "codelen_halves": lens, "K_E": keff,
                  "cost": cost}
    for K in Ks:
        res[K]["gain"] = res[Ks[0]]["codelen"] - res[K]["codelen"]
        res[K]["net"] = res[K]["gain"] - res[K]["cost"]
    return res


def b2_classes(Cp, E, seed, K, iters=EM_ITERS):
    """B2: whole-matrix fit; per E and type: prior, odd share, zero share, mean |v|, P(max = 6)."""
    H = block_hist(Cp)
    vals, ei = e_index(E)
    A = alphabet_per_e(H, ei, len(vals))
    key = block_key(Cp)
    out = []
    for k, e in enumerate(vals):
        idx = np.nonzero(ei == k)[0]
        Kk = k_eff(len(idx), K)
        m = em_fit(H[idx][:, A[k]], Kk, key[idx], rng_for(seed, K, 9, k), iters=iters)
        th = np.zeros((Kk, 16))
        th[:, A[k]] = np.exp(m["logth"])
        pi = np.exp(m["logpi"])
        for z in np.argsort(th @ ABSV):
            p6 = th[z, 7] + th[z, 15]
            out.append({"E": int(e), "K": K, "K_E": Kk, "n_blocks": len(idx), "type": int(z),
                        "prior": float(pi[z]), "odd_share": float(th[z, ODD].sum()),
                        "zero_share": float(th[z, 0] + th[z, 8]),
                        "mean_absv": float(th[z] @ ABSV),
                        "p_max6": float(1 - (1 - p6) ** B),
                        "theta": th[z].tolist()})
    return out


def sequential_check(Hblk_codes, model, A):
    """TB5: per block, predict each codeword from the posterior over types given the earlier
    codewords; returns max relative error of prod(predictions) vs the mixture marginal."""
    lp, lt = model["logpi"], model["logth"]
    full = np.full((len(lp), 16), -np.inf)
    full[:, A] = lt
    errs = []
    for codes in Hblk_codes:
        post = lp.copy()
        logprod = 0.0
        for c in codes:
            pz = post - _lse(post[None])[0]
            logprod += _lse((pz + full[:, c])[None])[0]
            post = post + full[:, c]
        marg = _lse(np.array([lp + full[:, codes].sum(axis=1)]))[0]
        errs.append(abs(math.expm1(logprod - marg)))
    return float(max(errs))


# ============================================================================ C1

C1_K = (1, 2, 4, 8, 16, 32, 64)
C1B_K = (1, 2, 4, 8, 16)


def qbins(P, nb):
    edges = np.quantile(P, np.arange(1, nb) / nb)
    return np.searchsorted(edges, P, side="right")


def absv_resid(Cp, E):
    vals, ei = e_index(E)
    a = ABSV[Cp]
    mu = np.bincount(ei, weights=a.reshape(-1, B).sum(axis=1)) / (np.bincount(ei) * B)
    return a - np.repeat(mu[ei].reshape(E.shape), B, axis=1)


def c1_matrix(Cp, E, seed, Cs=None, ks=C1_K):
    """C1 on X = |v| - E-mean. Cs: S0E version (baseline R2). Strict gain if max R2 > 0.001."""
    R, W = Cp.shape
    N = Cp.size
    X = absv_resid(Cp, E)
    r2, predict = holdout_lowrank(X, rng_for(seed, 1), ks=ks)
    out = {"r2": r2}
    if Cs is not None:
        out["r2_S0E"] = holdout_lowrank(absv_resid(Cs, E), rng_for(seed, 1), ks=ks)[0]
    kb = max(ks, key=lambda k: r2[k])
    out["kbest"] = kb
    out["max_r2"] = r2[kb]
    if r2[kb] > 0.001:
        P = predict(kb)
        bins = qbins(P, 16).ravel()
        vals, ei = e_index(E)
        nE = len(vals)
        ctx = np.repeat(ei.reshape(E.shape), B, axis=1).ravel()
        half = np.repeat(np.arange(R) % 2, W)
        h_e, _, n = est.holdout_cross_entropy(ctx, Cp.ravel(), half, nE)
        h_eb, _, _ = est.holdout_cross_entropy(ctx * 16 + bins, Cp.ravel(), half, nE * 16)
        cost = kb * (R + W) * PARAM_BITS / N
        out.update(strict={"k": kb, "h_E": h_e, "h_Ebin": h_eb, "gain": h_e - h_eb,
                           "cost": cost, "net": h_e - h_eb - cost})
    else:
        out["strict"] = None
    return out


def c1b_matrix(E, seed, n_weights, ks=C1B_K):
    """C1b: low rank of the scale matrix (minus mean); baseline = whole-matrix permutation."""
    Rs, Ws = E.shape
    X = E.astype(np.float64) - E.mean()
    r2, predict = holdout_lowrank(X, rng_for(seed, 2), ks=ks)
    g = rng_for(seed, 3)
    Xp = g.permutation(X.ravel()).reshape(X.shape)
    r2p = holdout_lowrank(Xp, rng_for(seed, 2), ks=ks)[0]
    kb = max(ks, key=lambda k: r2[k])
    out = {"r2": r2, "r2_perm": r2p, "kbest": kb, "max_r2": r2[kb]}
    if r2[kb] > 0.001:
        bins = qbins(predict(kb), 8).ravel()
        vals, inv = np.unique(E.ravel(), return_inverse=True)
        half = np.repeat(np.arange(Rs) % 2, Ws)
        h_b, h0, n = est.holdout_cross_entropy(bins, inv, half, 8, k=len(vals))
        gain = h0 - h_b
        cost = kb * (Rs + Ws) * PARAM_BITS / n_weights
        out["strict"] = {"k": kb, "H_E": h0, "H_E_bin": h_b, "gain_per_scale": gain,
                         "gain": gain / B, "factor_cost": cost, "net": gain / B - cost}
    else:
        out["strict"] = None
    return out


# ============================================================================ C2: coordinate network

def top_svd(X, k):
    """Top-k singular triplets via the smaller Gram matrix."""
    R, W = X.shape
    if R <= W:
        G = X @ X.T
        w, U = np.linalg.eigh(G)
        idx = np.argsort(w)[::-1][:k]
        s = np.sqrt(np.maximum(w[idx], 0))
        U = U[:, idx]
        V = (X.T @ U) / np.where(s > 0, s, 1)
    else:
        G = X.T @ X
        w, V = np.linalg.eigh(G)
        idx = np.argsort(w)[::-1][:k]
        s = np.sqrt(np.maximum(w[idx], 0))
        V = V[:, idx]
        U = (X @ V) / np.where(s > 0, s, 1)
    return U, s, V


class Adam:
    def __init__(self, params, b1=0.9, b2=0.999, eps=1e-8):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.b1, self.b2, self.eps, self.t = b1, b2, eps, 0

    def step(self, params, grads, lr):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for k, g in grads.items():
            m, v = self.m[k], self.v[k]
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * (g * g)
            params[k] -= (lr / c1) * m / (np.sqrt(v / c2) + self.eps)


INR_DEFAULT = {"emb": 8, "hidden": 64, "epochs": 20, "batch": 65536, "lr": 0.002, "emb_decay": 0.01,
               "split": (0.7, 0.1, 0.2), "output": "code"}


def inr_split(N, rng, split):
    perm = rng.permutation(N)
    a = int(round(split[0] * N))
    b = a + int(round(split[1] * N))
    return np.sort(perm[:a]), np.sort(perm[a:b]), np.sort(perm[b:])


def baseline_table(y, e, nE, idx, k=16, alpha=0.5):
    t = np.bincount(e[idx] * k + y[idx], minlength=nE * k).reshape(nE, k) + alpha
    return np.log(t / t.sum(axis=1, keepdims=True))


def ce_bits(logp_tab, y, e, idx):
    return float(-logp_tab[e[idx], y[idx]].mean() / LN2)


def inr_init_emb(Cp, train, d):
    """Section 4 C2 embedding initialisation: SVD of (v - mean) and (|v| - mean) with non-training
    positions zeroed; d/2 components from each; U sqrt(s) and V sqrt(s); each table / its std."""
    R, W = Cp.shape
    mask = np.zeros(Cp.size, dtype=bool)
    mask[train] = True
    mask = mask.reshape(R, W)
    rows, cols = [], []
    h = d // 2
    for X in (E2M1[Cp], ABSV[Cp]):
        X = X - X[mask].mean()
        X = np.where(mask, X, 0.0)
        U, s, V = top_svd(X, h)
        rows.append(U * np.sqrt(s))
        cols.append(V * np.sqrt(s))
    re = np.concatenate(rows, axis=1)
    ce = np.concatenate(cols, axis=1)
    re = re / re.std()
    ce = ce / ce.std()
    return re.astype(np.float32), ce.astype(np.float32)


def inr_params(R, W, nE, d, hid, nout, rng, out_bias):
    din = 3 * d
    p = {"re": None, "ce": None,
         "W1": (rng.standard_normal((din, hid)) * math.sqrt(2.0 / (din + nE))).astype(np.float32),
         "W1e": (rng.standard_normal((nE, hid)) * math.sqrt(2.0 / (din + nE))).astype(np.float32),
         "b1": np.zeros(hid, np.float32),
         "W2": (rng.standard_normal((hid, hid)) * math.sqrt(2.0 / hid)).astype(np.float32),
         "b2": np.zeros(hid, np.float32),
         "W3": (rng.standard_normal((hid, nout)) * 0.01).astype(np.float32),
         "b3": out_bias.astype(np.float32)}
    return p


def n_net_params(d, nE, hid, nout):
    return (3 * d + nE) * hid + hid + hid * hid + hid + hid * nout + nout


def inr_forward(p, ri, ci, ei, keep=False):
    r = p["re"][ri]
    c = p["ce"][ci]
    x = np.concatenate([r, c, r * c], axis=1)
    a1 = x @ p["W1"] + p["W1e"][ei] + p["b1"]
    h1 = np.maximum(a1, 0)
    a2 = h1 @ p["W2"] + p["b2"]
    h2 = np.maximum(a2, 0)
    z = h2 @ p["W3"] + p["b3"]
    z = z - z.max(axis=1, keepdims=True)
    lz = np.log(np.exp(z).sum(axis=1, keepdims=True))
    logp = z - lz
    if keep:
        return logp, (r, c, x, h1, h2)
    return logp


def inr_grads(p, ri, ci, ei, y, R, W, nE):
    logp, (r, c, x, h1, h2) = inr_forward(p, ri, ci, ei, keep=True)
    n = len(y)
    P = np.exp(logp)
    loss = float(-logp[np.arange(n), y].mean())
    dz = P
    dz[np.arange(n), y] -= 1.0
    dz /= n
    g = {"W3": h2.T @ dz, "b3": dz.sum(axis=0)}
    dh2 = dz @ p["W3"].T
    dh2 *= h2 > 0
    g["W2"] = h1.T @ dh2
    g["b2"] = dh2.sum(axis=0)
    dh1 = dh2 @ p["W2"].T
    dh1 *= h1 > 0
    g["W1"] = x.T @ dh1
    g["b1"] = dh1.sum(axis=0)
    hid = dh1.shape[1]
    g["W1e"] = np.stack([np.bincount(ei, weights=dh1[:, j], minlength=nE) for j in range(hid)],
                        axis=1).astype(np.float32)
    dx = dh1 @ p["W1"].T
    d = r.shape[1]
    dr = dx[:, :d] + dx[:, 2 * d:] * c
    dc = dx[:, d:2 * d] + dx[:, 2 * d:] * r
    g["re"] = np.stack([np.bincount(ri, weights=dr[:, j], minlength=R) for j in range(d)],
                       axis=1).astype(np.float32)
    g["ce"] = np.stack([np.bincount(ci, weights=dc[:, j], minlength=W) for j in range(d)],
                       axis=1).astype(np.float32)
    return loss, g


def inr_ce(p, ri, ci, ei, y, chunk=1 << 18, sign_bits=None):
    """Cross-entropy in bits/position; sign_bits: extra bits per position (magnitude output)."""
    tot = 0.0
    for i in range(0, len(y), chunk):
        lp = inr_forward(p, ri[i:i + chunk], ci[i:i + chunk], ei[i:i + chunk])
        tot -= float(lp[np.arange(len(lp)), y[i:i + chunk]].sum())
    extra = 0.0 if sign_bits is None else float(sign_bits.mean())
    return tot / len(y) / LN2 + extra


def inr_fit(Cp, E, seed, cfg=None, log=None):
    """C2: train the coordinate network on 70% of positions, early-stop on 10%, test on 20%.
    Returns baseline / network test cross-entropy, gain, cost and net (bits/weight)."""
    cfg = {**INR_DEFAULT, **(cfg or {})}
    R, W = Cp.shape
    N = Cp.size
    rng = rng_for(seed, 20)
    train, val, test = inr_split(N, rng, cfg["split"])
    vals, eb = e_index(E)
    nE = len(vals)
    ew = np.repeat(eb.reshape(E.shape), B, axis=1).ravel()
    y = Cp.ravel().astype(np.int64)
    rows = np.repeat(np.arange(R), W)
    cols = np.tile(np.arange(W), R)
    base = baseline_table(y, ew, nE, train)
    base_val, base_test = ce_bits(base, y, ew, val), ce_bits(base, y, ew, test)
    d, hid = cfg["emb"], cfg["hidden"]
    if cfg["output"] == "mag":
        # magnitude only (8 classes); the sign of every non-zero codeword costs 1 bit
        nout = 8
        yt = y & 7
        sb = (yt > 0).astype(np.float64)
    else:
        nout = 16
        yt = y
        sb = None
    tr_counts = np.bincount(yt[train], minlength=nout) + 0.5
    out_bias = np.log(tr_counts / tr_counts.sum())
    p = inr_params(R, W, nE, d, hid, nout, rng, out_bias)
    p["re"], p["ce"] = inr_init_emb(Cp, train, d)
    opt = Adam(p)
    nb = math.ceil(len(train) / cfg["batch"])
    S = cfg["epochs"] * nb
    t = 0
    best = {"val": base_val, "epoch": -1, "params": None}
    hist = []
    for ep in range(cfg["epochs"]):
        order = train[rng.permutation(len(train))]
        losses = []
        for i in range(nb):
            sel = order[i * cfg["batch"]:(i + 1) * cfg["batch"]]
            lr = cfg["lr"] * 0.5 * (1 + math.cos(math.pi * t / S))
            loss, g = inr_grads(p, rows[sel], cols[sel], ew[sel], yt[sel], R, W, nE)
            opt.step(p, g, lr)
            p["re"] *= (1 - cfg["emb_decay"] * lr)
            p["ce"] *= (1 - cfg["emb_decay"] * lr)
            losses.append(loss)
            t += 1
        v = inr_ce(p, rows[val], cols[val], ew[val], yt[val], sign_bits=None if sb is None else sb[val])
        hist.append({"epoch": ep, "train_bits": float(np.mean(losses)) / LN2, "val_bits": v})
        if log:
            log(f"epoch {ep} train {hist[-1]['train_bits']:.5f} val {v:.5f} base {base_val:.5f}")
        if v < best["val"]:
            best = {"val": v, "epoch": ep, "params": {k: a.copy() for k, a in p.items()}}
    npar = n_net_params(d, nE, hid, nout)
    cost = ((R + W) * d + npar) * PARAM_BITS / N
    if best["params"] is None:
        net_test, gain = float("nan"), 0.0
    else:
        net_test = inr_ce(best["params"], rows[test], cols[test], ew[test], yt[test],
                          sign_bits=None if sb is None else sb[test])
        gain = base_test - net_test
    return {"base_val": base_val, "base_test": base_test, "net_test": net_test,
            "best_val": best["val"], "best_epoch": best["epoch"], "gain": gain, "cost": cost,
            "net": gain - cost, "n_params_net": npar, "n_emb": (R + W) * d, "history": hist,
            "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
            "K_E": nE}
