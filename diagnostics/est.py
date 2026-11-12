"""Plug-in estimators (spec section 6). All information quantities are in bits."""
import math

import numpy as np

LN2 = math.log(2.0)


def entropy_counts(counts):
    c = np.asarray(counts, dtype=np.float64).ravel()
    n = c.sum()
    if n <= 0:
        return float("nan")
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def mi_table(t):
    """Plug-in mutual information of a 2-D count table.

    Returns (mi, n, dof) with dof = (Kx-1)(Ky-1) over values that actually occur.
    """
    t = np.asarray(t, dtype=np.float64)
    n = t.sum()
    if n <= 0:
        return float("nan"), 0, 0
    px = t.sum(axis=1)
    py = t.sum(axis=0)
    nz = t > 0
    outer = np.outer(px, py)
    mi = float((t[nz] * np.log2(t[nz] * n / outer[nz])).sum() / n)
    kx = int((px > 0).sum())
    ky = int((py > 0).sum())
    return mi, int(n), (kx - 1) * (ky - 1)


def analytic_bias(dof, n):
    return dof / (2.0 * n * LN2) if n > 0 else float("nan")


def analytic_sd(dof, n):
    return math.sqrt(2.0 * dof) / (2.0 * n * LN2) if n > 0 else float("nan")


def mi_keys(g, c, ng, k=16):
    """I(C; G) where g in [0, ng) and c in [0, k) are aligned integer arrays."""
    key = g.astype(np.int64).ravel() * k + c.astype(np.int64).ravel()
    t = np.bincount(key, minlength=ng * k).reshape(ng, k)
    return mi_table(t)


def holdout_cross_entropy(ctx, tgt, half, nctx, k=16, alpha=0.5):
    """Held-out conditional entropy (spec section 6).

    Counts from one half (each cell + alpha), cross-entropy on the other half, then swap and
    average. Returns (H_ctx, H0, n) where H0 is the matching context-free held-out entropy on
    the same targets.
    """
    ctx = ctx.astype(np.int64).ravel()
    tgt = tgt.astype(np.int64).ravel()
    half = half.astype(np.int64).ravel()
    key = (half * nctx + ctx) * k + tgt
    cnt = np.bincount(key, minlength=2 * nctx * k).reshape(2, nctx, k).astype(np.float64)
    hs, h0s = [], []
    for a in (0, 1):
        train, test = cnt[a], cnt[1 - a]
        ntest = test.sum()
        p = (train + alpha) / (train.sum(axis=1, keepdims=True) + alpha * k)
        hs.append(float(-(test * np.log2(p)).sum() / ntest))
        tr0 = train.sum(axis=0)
        p0 = (tr0 + alpha) / (tr0.sum() + alpha * k)
        h0s.append(float(-(test.sum(axis=0) * np.log2(p0)).sum() / ntest))
    return float(np.mean(hs)), float(np.mean(h0s)), int(cnt.sum())
