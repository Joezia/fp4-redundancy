"""Per-matrix R1-R4 measurements (spec sections 4-7).

`measure_matrix` is the single code path used for both synthetic validation (section 8) and model
matrices. Inputs are a codeword matrix C[r, j] in 0..15 (uint8) and a scale matrix E[r, j // B]
(uint8 E8M0 bytes). Nothing is written back; shuffles and derived quantities live only in memory.
"""
import math

import numpy as np

from . import est

# Codeword -> value table (no scale). Index = 4-bit codeword, bit 3 is the sign.
# Source: inference/convert.py FP4_TABLE (see experiments/r1-r4/format.md, F2).
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
ABS_E2M1 = np.abs(E2M1)

ROW_LAGS = 96
COL_LAGS = 64
COMP_LAGS = 8
PARITY_LAGS = 4
SCALE_LAGS = 8
REPS = 3
B4_BLOCK_ALPHAS = (32, 128, 512)
B4_ROW_ALPHAS = (512, 2048, 8192)
ENERGY_TIERS = (2, 4, 8)
D4_THRESHOLD = 0.001
D3_LEFT = 8

CONTEXTS = [
    ("left1", [(0, -1)]),
    ("left1-2", [(0, -1), (0, -2)]),
    ("left1-3", [(0, -1), (0, -2), (0, -3)]),
    ("up1", [(-1, 0)]),
    ("up1-2", [(-1, 0), (-2, 0)]),
    ("left1+up1", [(0, -1), (-1, 0)]),
    ("left1+up1+upleft", [(0, -1), (-1, 0), (-1, -1)]),
]
LEFT_ONLY = [name for name, offs in CONTEXTS if all(dr == 0 for dr, _ in offs)]
HAS_UP = [name for name, offs in CONTEXTS if any(dr != 0 for dr, _ in offs)]
SHUFFLE_ID = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "SE": 9}

FIELDS = ["matrix", "layer", "expert", "role", "item", "variable", "group_or_context",
          "direction", "lag", "subset", "n", "value", "analytic_bias", "baseline",
          "baseline_mean", "baseline_std", "net", "net_pct_of_4bit", "significant", "note"]


def unpack_codes(packed):
    """(R, W/2) uint8 bytes -> (R, W) codewords; low nibble is the earlier codeword (F2.1)."""
    rows, half = packed.shape
    out = np.empty((rows, 2 * half), dtype=np.uint8)
    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = packed >> 4
    return out


def shuffle(C, kind, rng, B):
    R, W = C.shape
    if kind == "S0":
        return rng.permutation(C.ravel()).reshape(R, W)
    if kind == "S1":
        return rng.permuted(C, axis=1)
    if kind == "S2":
        return rng.permuted(C.reshape(R, W // B, B), axis=2).reshape(R, W)
    if kind == "S3":
        return C[rng.permutation(R)]
    raise ValueError(kind)


# ----------------------------------------------------------------------------- records

class Rows:
    def __init__(self, meta):
        self.meta = meta
        self.rows = []

    def add(self, item, variable, value, *, group="", direction="", lag="", subset="", n=None,
            baseline="", reps=None, kind="other", dof=None, bits=True, note="", bias=None):
        r = {k: "" for k in FIELDS}
        r.update(self.meta)
        r.update(item=item, variable=variable, group_or_context=group, direction=direction,
                 lag=lag, subset=subset, n="" if n is None else int(n), note=note)
        r["value"] = _num(value)
        if kind == "mi" and dof is not None and n:
            r["analytic_bias"] = est.analytic_bias(dof, n)
        elif bias is not None:
            r["analytic_bias"] = bias
        r["_reps"] = None
        if reps is not None and baseline:
            reps = [float(x) for x in reps]
            r["_reps"] = reps
            bm = float(np.mean(reps))
            bs = float(np.std(reps, ddof=1)) if len(reps) > 1 else float("nan")
            r.update(baseline=baseline, baseline_mean=bm, baseline_std=bs)
            if value is not None and np.isfinite(value):
                net = float(value) - bm
                r["net"] = net
                if bits:
                    r["net_pct_of_4bit"] = 100.0 * net / 4.0
                if kind == "mi":
                    sig = net > 5.0 * est.analytic_sd(dof, n)
                elif kind == "acf":
                    sig = abs(net) > 5.0 / math.sqrt(n)
                else:
                    sig = (net > 5.0 * bs) and (net > 1e-4)
                r["significant"] = bool(sig)
        r["_dof"] = dof
        r["_bits"] = bits
        self.rows.append(r)
        return r


def _num(x):
    if x is None:
        return ""
    x = float(x)
    return x if np.isfinite(x) else ""


# ----------------------------------------------------------------------------- R2 groups

def _table(g, c, ng, k=16):
    key = g.astype(np.int64).ravel() * k + c.astype(np.int64).ravel()
    return np.bincount(key, minlength=ng * k).reshape(ng, k)


def b1_groups(X, Eidx, nE, B):
    """I(C; G) for groups known to the decoder (matrix-internal)."""
    R, W = X.shape
    jj = np.arange(W)
    groups = {
        "r": (np.broadcast_to(np.arange(R)[:, None], (R, W)), R),
        "j": (np.broadcast_to(jj[None, :], (R, W)), W),
        "b": (np.broadcast_to((jj // B)[None, :], (R, W)), W // B),
        "p": (np.broadcast_to((jj % B)[None, :], (R, W)), B),
        "p%2": (np.broadcast_to((jj % 2)[None, :], (R, W)), 2),
        "E": (np.repeat(Eidx, B, axis=1), nE),
    }
    out = {}
    for name, (g, ng) in groups.items():
        t = _table(g, X, ng)
        mi, n, dof = est.mi_table(t)
        out[name] = (mi, n, dof, int((t.sum(1) > 0).sum()), int((t.sum(0) > 0).sum()))
    return out


def block_labels(X, B):
    R, W = X.shape
    nb = W // B
    m_max = (X & 7).reshape(R, nb, B).max(axis=2)
    energy = ABS_E2M1[X].reshape(R, nb, B).mean(axis=2)
    labels = {"m_max": (m_max.astype(np.int64), 8)}
    for T in ENERGY_TIERS:
        edges = np.quantile(energy, [i / T for i in range(1, T)])
        labels[f"energy{T}"] = (np.searchsorted(edges, energy, side="right").astype(np.int64), T)
    return labels


def b2_derived(X, B):
    out = {}
    for name, (G, ng) in block_labels(X, B).items():
        t = _table(np.repeat(G, B, axis=1), X, ng)
        mi, n, dof = est.mi_table(t)
        hg = est.entropy_counts(np.bincount(G.ravel(), minlength=ng))
        out[name] = (mi, n, dof, hg / B, mi - hg / B)
    return out


def b3_block(X, B):
    R, W = X.shape
    nblk = R * (W // B)
    Xb = X.reshape(nblk, B).astype(np.int64)
    key = np.arange(nblk, dtype=np.int64)[:, None] * 16 + Xb
    t = np.bincount(key.ravel(), minlength=nblk * 16).reshape(nblk, 16)
    return est.mi_table(t)


def b4_adaptive(X, B, q, H0):
    """Sequential KT-like adaptive code length per block / per row; returns {(seg, alpha): gain}."""
    R, W = X.shape
    out = {}
    for seg, L, alphas in (("block", B, B4_BLOCK_ALPHAS), ("row", W, B4_ROW_ALPHAS)):
        Z = X.reshape(-1, L)
        prior = np.zeros(Z.shape, dtype=np.int32)
        for c in range(16):
            m = Z == c
            if not m.any():
                continue
            cs = np.cumsum(m, axis=1, dtype=np.int32)
            prior += (cs - 1) * m
        qz = q[Z]
        nseg = Z.shape[0]
        for a in alphas:
            num = np.log2(a * qz + prior).sum()
            den = nseg * np.log2(a + np.arange(L, dtype=np.float64)).sum()
            codelen = den - num
            out[(seg, a)] = H0 - codelen / Z.size
        del prior, qz
    return out


# ----------------------------------------------------------------------------- pair tables

def row_pair_tables(X, B, maxlag):
    """counts[d-1, flag, a*16+b] for pairs (C[r,j], C[r,j+d]); flag = same_block*2 + j%2."""
    R, W = X.shape
    Xi = X.astype(np.int32)
    jj = np.arange(W)
    out = np.zeros((maxlag, 4, 256), dtype=np.int64)
    for d in range(1, min(maxlag, W - 1) + 1):
        j = jj[: W - d]
        flag = ((((j % B) + d) < B).astype(np.int32) * 2 + (j % 2)) * 256
        key = Xi[:, : W - d] * 16
        key += Xi[:, d:]
        key += flag[None, :]
        out[d - 1] = np.bincount(key.ravel(), minlength=1024).reshape(4, 256)
    return out


def col_pair_tables(X, maxlag):
    R, W = X.shape
    Xi = X.astype(np.int32)
    out = np.zeros((maxlag, 256), dtype=np.int64)
    for d in range(1, min(maxlag, R - 1) + 1):
        key = Xi[: R - d] * 16
        key += Xi[d:]
        out[d - 1] = np.bincount(key.ravel(), minlength=256)
    return out


def row_subsets(t):
    """t: (4, 256) -> dict subset -> (16, 16) table."""
    return {
        "all": (t[0] + t[1] + t[2] + t[3]).reshape(16, 16),
        "same": (t[2] + t[3]).reshape(16, 16),
        "cross": (t[0] + t[1]).reshape(16, 16),
        "all_jeven": (t[0] + t[2]).reshape(16, 16),
        "all_jodd": (t[1] + t[3]).reshape(16, 16),
    }


_S = np.arange(16) >> 3
_M = np.arange(16) & 7


def components(t16):
    """Component MIs from a (16,16) codeword pair table."""
    out = {}
    for name, fa, fb, ka, kb in (("I(s;s')", _S, _S, 2, 2), ("I(m;m')", _M, _M, 8, 8),
                                 ("I(s;m')", _S, _M, 2, 8), ("I(m;s')", _M, _S, 8, 2)):
        t = np.zeros((ka, kb))
        np.add.at(t, (fa[:, None].repeat(16, 1), fb[None, :].repeat(16, 0)), t16)
        out[name] = est.mi_table(t)
    return out


def acf_from_table(t16, xtab, var):
    n = t16.sum()
    if n == 0 or var <= 0:
        return float("nan"), int(n)
    return float((t16 * np.outer(xtab, xtab)).sum() / n / var), int(n)


# ----------------------------------------------------------------------------- R3 contexts

def cond_entropy(X, names):
    R, W = X.shape
    out = {}
    offs = dict(CONTEXTS)
    for name in names:
        o = offs[name]
        r0 = max(-dr for dr, _ in o)
        c0 = max(-dc for _, dc in o)
        tgt = X[r0:, c0:]
        ctx = np.zeros(tgt.shape, dtype=np.int64)
        for k, (dr, dc) in enumerate(o):
            ctx += X[r0 + dr: R + dr, c0 + dc: W + dc].astype(np.int64) << (4 * k)
        half = np.broadcast_to((np.arange(r0, R) % 2)[:, None], tgt.shape)
        h, h0, n = est.holdout_cross_entropy(ctx, tgt, half, 16 ** len(o))
        out[name] = (h, h0, n)
    return out


# ----------------------------------------------------------------------------- R4 numeric

def spectral(Y, axis):
    F = np.fft.rfft(Y, axis=axis)
    P = (F.real ** 2 + F.imag ** 2).mean(axis=1 - axis)
    P = P[1:]
    if P.size == 0 or np.any(P <= 0):
        return float("nan")
    sf = math.exp(np.log(P).mean()) / P.mean()
    return 0.5 * math.log2(1.0 / sf)


def _features(Y, r0, r1):
    W = Y.shape[1]
    cur = Y[r0:r1]
    up = Y[r0 - 1:r1 - 1]
    f = [cur[:, D3_LEFT - k: W - 1 - k] for k in range(1, D3_LEFT + 1)]
    f += [up[:, D3_LEFT - 1: W - 2], up[:, D3_LEFT: W - 1], up[:, D3_LEFT + 1: W]]
    f.append(np.ones_like(cur[:, D3_LEFT: W - 1]))
    f.append(cur[:, D3_LEFT: W - 1])
    return np.stack(f, axis=-1)


def linear_prediction(Y, X, do_d4, chunk=96):
    """D3 (and optionally D4). Y: centred values (float64), X: codewords for D4 targets."""
    R, W = Y.shape
    nf = D3_LEFT + 3 + 1
    G = np.zeros((2, nf + 1, nf + 1))
    for r0 in range(1, R, chunk):
        r1 = min(R, r0 + chunk)
        A = _features(Y, r0, r1)
        par = np.arange(r0, r1) % 2
        for h in (0, 1):
            Ah = A[par == h].reshape(-1, nf + 1)
            G[h] += Ah.T @ Ah
    ws, r2s = [], []
    for a in (0, 1):
        w = np.linalg.lstsq(G[a][:nf, :nf], G[a][:nf, nf], rcond=None)[0]
        ws.append(w)
        Gt = G[1 - a]
        sse = Gt[nf, nf] - 2 * w @ Gt[:nf, nf] + w @ Gt[:nf, :nf] @ w
        n = Gt[nf - 1, nf - 1]
        sst = Gt[nf, nf] - Gt[nf - 1, nf] ** 2 / n
        r2s.append(1.0 - sse / sst if sst > 0 else float("nan"))
    r2 = float(np.mean(r2s))
    ntarget = int(G[0][nf - 1, nf - 1] + G[1][nf - 1, nf - 1])
    d4 = None
    if do_d4:
        preds = [np.empty((R - 1, W - 1 - D3_LEFT), dtype=np.float64) for _ in (0, 1)]
        for r0 in range(1, R, chunk):
            r1 = min(R, r0 + chunk)
            A = _features(Y, r0, r1)[..., :nf]
            for a in (0, 1):
                preds[a][r0 - 1:r1 - 1] = A @ ws[a]
        tgt = X[1:, D3_LEFT: W - 1].astype(np.int64)
        half = np.broadcast_to((np.arange(1, R) % 2)[:, None], tgt.shape)
        hs, h0s = [], []
        for a in (0, 1):
            edges = np.quantile(preds[a][half == a], [i / 8 for i in range(1, 8)])
            ctx = np.searchsorted(edges, preds[a], side="right")
            h, h0 = _ce_direction(ctx, tgt, half, a, 8)
            hs.append(h)
            h0s.append(h0)
        d4 = (float(np.mean(hs)), float(np.mean(h0s)), int(tgt.size))
    return r2, ntarget, d4


def _ce_direction(ctx, tgt, half, train_half, nctx, k=16, alpha=0.5):
    key = (half.astype(np.int64) * nctx + ctx) * k + tgt
    cnt = np.bincount(key.ravel(), minlength=2 * nctx * k).reshape(2, nctx, k).astype(float)
    train, test = cnt[train_half], cnt[1 - train_half]
    nt = test.sum()
    p = (train + alpha) / (train.sum(1, keepdims=True) + alpha * k)
    tr0 = train.sum(0)
    p0 = (tr0 + alpha) / (tr0.sum() + alpha * k)
    return float(-(test * np.log2(p)).sum() / nt), float(-(test.sum(0) * np.log2(p0)).sum() / nt)


# ----------------------------------------------------------------------------- per version

def version_stats(X, needs, B, Eidx, nE, q, H0, d4_vars=()):
    out = {}
    if "B1" in needs:
        out["B1"] = b1_groups(X, Eidx, nE, B)
    if "B2" in needs:
        out["B2"] = b2_derived(X, B)
    if "B3" in needs:
        out["B3"] = b3_block(X, B)
    if "B4" in needs:
        out["B4"] = b4_adaptive(X, B, q, H0)
    if "C1" in needs:
        out["C1"] = row_pair_tables(X, B, ROW_LAGS)
    if "C2" in needs:
        out["C2"] = col_pair_tables(X, COL_LAGS)
    if "C4" in needs:
        out["C4"] = cond_entropy(X, needs["C4"])
    if "D2" in needs or "D3" in needs:
        for var, tab in (("v", E2M1), ("|v|", ABS_E2M1)):
            Y = tab[X]
            Y = Y - Y.mean()
            if "D2" in needs:
                out[("D2", var)] = (spectral(Y, 1), spectral(Y, 0))
            if "D3" in needs:
                out[("D3", var)] = linear_prediction(Y, X, var in d4_vars)
            del Y
    return out


def measure_matrix(C, E, B, meta, seed):
    """Run every section-7 item on one matrix. Returns {'rows': [...], 'hist': ...}."""
    C = np.ascontiguousarray(C, dtype=np.uint8)
    E = np.ascontiguousarray(E, dtype=np.uint8)
    R, W = C.shape
    assert W % B == 0 and E.shape == (R, W // B), (C.shape, E.shape)
    N = C.size
    rows = Rows(meta)
    hist = np.bincount(C.ravel(), minlength=16).astype(np.int64)
    q = hist / N
    H0 = est.entropy_counts(hist)
    K = int((hist > 0).sum())

    # ---------------- A: R1 marginal
    for c in range(16):
        rows.add("A1", "C", hist[c], group=f"c={c}", n=N, bits=False, note="count")
    rows.add("A1", "C", H0, group="H0", n=N, note="bit/weight")
    rows.add("A1", "C", K, group="K", n=N, bits=False)
    rows.add("A1", "C", 4 - H0, group="4-H0", n=N)
    rows.add("A1", "C", 4 - math.log2(K), group="unused=4-log2K", n=N)
    rows.add("A1", "C", math.log2(K) - H0, group="skew=log2K-H0", n=N)
    z0, z8 = int(hist[0]), int(hist[8])
    rows.add("A2", "C", z0, group="count c=0 (+0)", n=N, bits=False)
    rows.add("A2", "C", z8, group="count c=8 (-0)", n=N, bits=False)
    pm0 = (z0 + z8) / N
    hs_m0 = est.entropy_counts([z0, z8]) if z0 + z8 > 0 else 0.0
    rows.add("A2", "C", pm0 * hs_m0, group="merge_gain=P(m=0)H(s|m=0)", n=N)
    sm = hist.reshape(2, 8)
    mi_sm, _, dof_sm = est.mi_table(sm)
    rows.add("A3", "s", est.entropy_counts(sm.sum(1)), group="H(s)", n=N)
    rows.add("A3", "m", est.entropy_counts(sm.sum(0)), group="H(m)", n=N)
    rows.add("A3", "s,m", mi_sm, group="I(s;m)", n=N, bias=est.analytic_bias(dof_sm, N))
    rows.add("A3", "s|m", est.entropy_counts(sm.sum(1)) - mi_sm, group="H(s|m)", n=N)
    ehist = np.bincount(E.ravel(), minlength=256)
    for byte in np.nonzero(ehist)[0]:
        rows.add("A4", "E", ehist[byte], group=f"E={byte}", n=E.size, bits=False,
                 note=f"count; value=2^{int(byte) - 127}")
    rows.add("A4", "E", est.entropy_counts(ehist), group="H0(E)", n=E.size,
             note="bit/scale; per weight divide by B")
    rows.add("A4", "E", int(np.nonzero(ehist)[0].min()), group="min_byte", n=E.size, bits=False)
    rows.add("A4", "E", int(np.nonzero(ehist)[0].max()), group="max_byte", n=E.size, bits=False)
    rows.add("A4", "E", int((ehist > 0).sum()), group="K_E", n=E.size, bits=False)

    evals, Eidx = np.unique(E, return_inverse=True)
    Eidx = Eidx.reshape(E.shape)
    nE = len(evals)

    # ---------------- versions
    def rng_for(kind, rep):
        return np.random.default_rng([seed, SHUFFLE_ID[kind], rep])

    real_needs = {"B1": 1, "B2": 1, "B3": 1, "B4": 1, "C1": 1, "C2": 1,
                  "C4": [n for n, _ in CONTEXTS], "D2": 1, "D3": 1}
    real = version_stats(C, real_needs, B, Eidx, nE, q, H0)
    d4_vars = tuple(v for v in ("v", "|v|") if real[("D3", v)][0] > D4_THRESHOLD)
    if d4_vars:
        extra = version_stats(C, {"D3": 1}, B, Eidx, nE, q, H0, d4_vars)
        for v in d4_vars:
            real[("D3", v)] = extra[("D3", v)]
    shuf = {"S0": [], "S1": [], "S2": [], "S3": []}
    needs = {
        "S0": dict(real_needs),
        "S1": {"C1": 1},
        "S2": {"C1": 1, "C4": LEFT_ONLY},
        "S3": {"C2": 1, "C4": HAS_UP},
    }
    for kind in ("S0", "S1", "S2", "S3"):
        for rep in range(REPS):
            Xs = shuffle(C, kind, rng_for(kind, rep), B)
            shuf[kind].append(version_stats(Xs, needs[kind], B, Eidx, nE, q, H0,
                                            d4_vars if kind == "S0" else ()))
            del Xs

    # A5 m_max histogram (real)
    mm = block_labels(C, B)["m_max"][0]
    mh = np.bincount(mm.ravel(), minlength=8)
    for m in range(8):
        rows.add("A5", "m_max", mh[m], group=f"m_max={m}", n=mm.size, bits=False, note="count")

    # ---------------- B1
    for g, (mi, n, dof, ng, kc) in real["B1"].items():
        reps = [s["B1"][g][0] for s in shuf["S0"]]
        rows.add("B1", "C", mi, group=g, n=n, baseline="S0", reps=reps, kind="mi", dof=dof,
                 note=f"|G|={ng}")
        rows.add("B1", "C", (kc - 1) * ng * 12 / N, group=g + ":codebook_cost", n=n,
                 note="(K-1)|G|12/N")

    # ---------------- B2
    for g, (mi, n, dof, cost, ach) in real["B2"].items():
        rows.add("B2", "C", mi, group=g + ":I(C;G)", n=n, baseline="S0",
                 reps=[s["B2"][g][0] for s in shuf["S0"]], kind="mi", dof=dof)
        rows.add("B2", "C", cost, group=g + ":label_cost=H(G)/B", n=n, baseline="S0",
                 reps=[s["B2"][g][3] for s in shuf["S0"]])
        rows.add("B2", "C", ach, group=g + ":achievable=I-H(G)/B", n=n, baseline="S0",
                 reps=[s["B2"][g][4] for s in shuf["S0"]],
                 note="S0 baseline_mean is the achievable value on S0")

    # ---------------- B3
    mi, n, dof = real["B3"]
    rows.add("B3", "C", mi, group="block_id", n=n, baseline="S0",
             reps=[s["B3"][0] for s in shuf["S0"]], kind="mi", dof=dof,
             note="analytic formula unreliable at B samples/group; read net vs S0")

    # ---------------- B4
    for (seg, a), gain in real["B4"].items():
        rows.add("B4", "C", gain, group=f"{seg}:alpha={a}", n=N, baseline="S0",
                 reps=[s["B4"][(seg, a)] for s in shuf["S0"]], note="H0 - codelen/N")

    # ---------------- C1 / C3 / D1 (row)
    mu_v = float((q * E2M1).sum())
    mu_a = float((q * ABS_E2M1).sum())
    xv, xa = E2M1 - mu_v, ABS_E2M1 - mu_a
    var_v, var_a = float((q * xv * xv).sum()), float((q * xa * xa).sum())
    rt = real["C1"]
    for d in range(1, min(ROW_LAGS, W - 1) + 1):
        subs = row_subsets(rt[d - 1])
        bsubs = {k: [row_subsets(s["C1"][d - 1]) for s in shuf[k]] for k in ("S0", "S1", "S2")}
        for sub, t in subs.items():
            if sub.startswith("all_j") and d > PARITY_LAGS:
                continue
            if t.sum() == 0:
                continue
            mi, n, dof = est.mi_table(t)
            for bl in ("S0", "S1", "S2"):
                reps = [est.mi_table(b[sub])[0] for b in bsubs[bl]]
                rows.add("C1", "C", mi, direction="row", lag=d, subset=sub, n=n, baseline=bl,
                         reps=reps, kind="mi", dof=dof)
            if sub.startswith("all_j"):
                continue
            for var, xt, vv in (("v", xv, var_v), ("|v|", xa, var_a)):
                r, n = acf_from_table(t, xt, vv)
                for bl in ("S0", "S2"):
                    reps = [acf_from_table(b[sub], xt, vv)[0] for b in bsubs[bl]]
                    rows.add("D1", var, r, direction="row", lag=d, subset=sub, n=n,
                             baseline=bl, reps=reps, kind="acf", bits=False,
                             bias=5.0 / math.sqrt(n), note="analytic_bias column = 5/sqrt(n)")
        if d <= COMP_LAGS:
            comp = components(subs["all"])
            bcomp = [components(b["all"]) for b in bsubs["S0"]]
            for name, (mi, n, dof) in comp.items():
                rows.add("C3", name, mi, direction="row", lag=d, subset="all", n=n,
                         baseline="S0", reps=[b[name][0] for b in bcomp], kind="mi", dof=dof)

    # ---------------- C2 / C3 / D1 (col)
    ct = real["C2"]
    for d in range(1, min(COL_LAGS, R - 1) + 1):
        t = ct[d - 1].reshape(16, 16)
        bt = {k: [s["C2"][d - 1].reshape(16, 16) for s in shuf[k]] for k in ("S0", "S3")}
        mi, n, dof = est.mi_table(t)
        for bl in ("S0", "S3"):
            rows.add("C2", "C", mi, direction="col", lag=d, subset="all", n=n, baseline=bl,
                     reps=[est.mi_table(b)[0] for b in bt[bl]], kind="mi", dof=dof)
        for var, xt, vv in (("v", xv, var_v), ("|v|", xa, var_a)):
            r, n = acf_from_table(t, xt, vv)
            for bl in ("S0", "S3"):
                rows.add("D1", var, r, direction="col", lag=d, subset="all", n=n, baseline=bl,
                         reps=[acf_from_table(b, xt, vv)[0] for b in bt[bl]], kind="acf",
                         bits=False, bias=5.0 / math.sqrt(n), note="analytic_bias column = 5/sqrt(n)")
        if d <= COMP_LAGS:
            comp = components(t)
            bcomp = [components(b) for b in bt["S0"]]
            for name, (mi, n, dof) in comp.items():
                rows.add("C3", name, mi, direction="col", lag=d, subset="all", n=n,
                         baseline="S0", reps=[b[name][0] for b in bcomp], kind="mi", dof=dof)

    # ---------------- C4
    for name, _ in CONTEXTS:
        h, h0, n = real["C4"][name]
        gain = h0 - h
        bls = ["S0"] + (["S2"] if name in LEFT_ONLY else ["S3"])
        for bl in bls:
            reps = [s["C4"][name][1] - s["C4"][name][0] for s in shuf[bl]]
            rows.add("C4", "C", gain, group=name, n=n, baseline=bl, reps=reps,
                     note=f"H0_holdout={h0:.6f}; H_holdout={h:.6f}")

    # ---------------- D2 / D3 / D4
    for var in ("v", "|v|"):
        for i, dirn in enumerate(("row", "col")):
            rows.add("D2", var, real[("D2", var)][i], group="0.5*log2(1/SFM)", direction=dirn,
                     n=N, baseline="S0", reps=[s[("D2", var)][i] for s in shuf["S0"]])
        r2, nt, d4 = real[("D3", var)]
        reps = [s[("D3", var)][0] for s in shuf["S0"]]
        rows.add("D3", var, r2, group="R2", n=nt, baseline="S0", reps=reps, bits=False)
        bits = -0.5 * math.log2(1 - r2) if r2 < 1 else float("nan")
        rows.add("D3", var, bits, group="-0.5*log2(1-R2)", n=nt, baseline="S0",
                 reps=[-0.5 * math.log2(1 - x) for x in reps])
        if d4 is not None:
            h, h0, n = d4
            reps = [s[("D3", var)][2][1] - s[("D3", var)][2][0] for s in shuf["S0"]]
            rows.add("D4", var, h0 - h, group="pred_octile", n=n, baseline="S0", reps=reps,
                     note=f"H0_holdout={h0:.6f}; H_holdout={h:.6f}")
        else:
            rows.add("D4", var, None, group="pred_octile", note=f"not run: D3 R2={r2:.3g} <= 0.001")

    # ---------------- S: scale stream
    scale_items(E, rows, rng_for)

    return {"rows": rows.rows, "hist": hist, "N": N}


def scale_items(E, rows, rng_for):
    R, nb = E.shape
    evals, Ei = np.unique(E, return_inverse=True)
    Ei = Ei.reshape(E.shape).astype(np.int64)
    k = len(evals)
    perms = [rng_for("SE", rep).permutation(Ei.ravel()).reshape(R, nb) for rep in range(REPS)]

    def pair_mi(A, d, axis):
        if axis == 1:
            a, b = A[:, :-d], A[:, d:]
        else:
            a, b = A[:-d], A[d:]
        return est.mi_table(_table(a, b, k, k))

    for dirn, axis, L in (("row", 1, nb), ("col", 0, R)):
        for d in range(1, min(SCALE_LAGS, L - 1) + 1):
            mi, n, dof = pair_mi(Ei, d, axis)
            rows.add("S1", "E", mi, direction=dirn, lag=d, n=n, baseline="perm(E)",
                     reps=[pair_mi(P, d, axis)[0] for P in perms], kind="mi", dof=dof)

    def ce(A, dirn):
        if dirn == "left":
            tgt, ctx = A[:, 1:], A[:, :-1]
            half = np.broadcast_to((np.arange(R) % 2)[:, None], tgt.shape)
        else:
            tgt, ctx = A[1:], A[:-1]
            half = np.broadcast_to((np.arange(1, R) % 2)[:, None], tgt.shape)
        if tgt.size == 0:
            return float("nan"), float("nan"), 0
        return est.holdout_cross_entropy(ctx, tgt, half, k, k=k)

    for dirn in ("left", "up"):
        h, h0, n = ce(Ei, dirn)
        reps = []
        for P in perms:
            hp, h0p, _ = ce(P, dirn)
            reps.append(h0p - hp)
        rows.add("S2", "E", h0 - h, group=f"H0(E)-H(E|{dirn})", n=n, baseline="perm(E)",
                 reps=reps, note=f"bit/scale; H0_holdout={h0:.6f}; H_holdout={h:.6f}")
