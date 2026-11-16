"""Incremental saving per pattern (spec 008 Part 0).

A chain of models M0 ⊂ M1 ⊂ ... ⊂ Mk is evaluated on one item (an audio file, an image, an FP4
matrix). M0 is the raw fixed-length representation. Model Mk switches on the patterns of the first k
steps of an ordered pattern list; its code length L(Mk) is a sequential adaptive (prequential) code
length plus every side information actually coded. The increment of step k is
(L(Mk-1) - L(Mk)) / raw bits; the total saving L(M0) - L(Mlast) equals the sum of the increments.

Code lengths. Every probability is an adaptive count estimate that only uses symbols coded before:
a binary decision with counts (n0, n1) of earlier decisions in its context is coded with probability
(n_b + 1/2) / (n0 + n1 + 1) (Krichevsky-Trofimov), a K-ary symbol with (n_a + a) / (n + K a). The code
length of such a sequential code depends only on the final counts per context, so it is computed in
closed form from count tables (sum of log-gamma differences) without a per-symbol loop; it is exactly the
length a sequential arithmetic coder with the same estimator reaches (up to the coder's few bits).
Large alphabets are binarised: values below 2^10 by their binary tree (MSB first, one adaptive bit per
tree node and context), larger values by an Exp-Golomb code (adaptive bit-length class, then adaptive
mantissa bits). Every context is computed from data the decoder already has, or from side information
whose bits are counted. Parameters of online predictors are re-estimated from already-coded data only.

The domain chains (1-D audio, 2-D images, FP4 matrices) are below; each defines what every pattern
adds. The module is independent of the indicator framework: an item is an array (plus, for FP4, the
scale array and the partner matrix), produced by a media/array adapter.
"""
import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

# ----------------------------------------------------------------------------------------------
# closed-form adaptive code lengths
# ----------------------------------------------------------------------------------------------

_LG = {}


def _lgamma_table(alpha, n):
    """t[m] = sum_{i<m} ln(i + alpha), m = 0..n (cached, grown on demand)."""
    t = _LG.get(alpha)
    if t is None or len(t) <= n:
        m = max(n + 1, 1 << 16, 2 * (0 if t is None else len(t)))
        t = np.concatenate([[0.0], np.cumsum(np.log(np.arange(m, dtype=np.float64) + alpha))])
        _LG[alpha] = t
    return t


def _lgam(alpha, counts):
    counts = np.asarray(counts, dtype=np.int64)
    if counts.size == 0:
        return 0.0
    t = _lgamma_table(alpha, int(counts.max()))
    return float(t[counts].sum())


def kt_bits(keys, bits):
    """Adaptive (KT) code length in bits of binary decisions `bits` coded in contexts `keys`."""
    keys = np.asarray(keys, dtype=np.int64).ravel()
    bits = np.asarray(bits, dtype=np.int64).ravel()
    if keys.size == 0:
        return 0.0
    u, inv = np.unique(keys, return_inverse=True)
    n1 = np.bincount(inv, weights=bits, minlength=len(u)).astype(np.int64)
    n = np.bincount(inv, minlength=len(u)).astype(np.int64)
    n0 = n - n1
    lnp = _lgam(0.5, n0) + _lgam(0.5, n1) - _lgam(1.0, n)
    return -lnp / math.log(2)


def dirichlet_bits(sym, ctx, K, alpha=0.5):
    """Adaptive K-ary code length (add-alpha counts per context) of `sym` in contexts `ctx`."""
    sym = np.asarray(sym, dtype=np.int64).ravel()
    ctx = np.zeros_like(sym) if ctx is None else np.asarray(ctx, dtype=np.int64).ravel()
    if sym.size == 0:
        return 0.0
    u, inv = np.unique(ctx * K + sym, return_inverse=True)
    n_sym = np.bincount(inv)
    cu, cinv = np.unique(u // K, return_inverse=True)
    n_ctx = np.bincount(cinv, weights=n_sym).astype(np.int64)
    lnp = _lgam(alpha, n_sym) - _lgam(K * alpha, n_ctx)
    return -lnp / math.log(2)


def tree_bits(sym, ctx, nbits):
    """Adaptive binary-tree code (MSB first, one KT bit per tree node and context)."""
    sym = np.asarray(sym, dtype=np.int64).ravel()
    ctx = np.zeros_like(sym) if ctx is None else np.asarray(ctx, dtype=np.int64).ravel()
    keys, bits = [], []
    for d in range(nbits):
        node = (sym >> (nbits - d)) | (1 << d)
        keys.append(ctx * (1 << nbits) + node)
        bits.append((sym >> (nbits - 1 - d)) & 1)
    return kt_bits(np.concatenate(keys), np.concatenate(bits))


def bitlen(u):
    u = np.asarray(u, dtype=np.int64)
    out = np.zeros(u.shape, dtype=np.int64)
    v = u.copy()
    while np.any(v):
        nz = v > 0
        out[nz] += 1
        v >>= 1
    return out


def eg_bits(u, ctx, top=3, nclass=40):
    """Adaptive Exp-Golomb code of u >= 0: bit-length class k (tree code over 6 bits, in `ctx`),
    then the k-1 mantissa bits below the leading one; the first `top` mantissa bits are coded in
    context (k, position, prefix), the rest in context (k, position). The context `ctx` acts on the
    class, which carries the scale of the value."""
    u = np.asarray(u, dtype=np.int64).ravel()
    ctx = np.zeros_like(u) if ctx is None else np.asarray(ctx, dtype=np.int64).ravel()
    k = bitlen(u)
    L = tree_bits(k, ctx, 6)
    keys, bits = [], []
    kmax = int(k.max()) if k.size else 0
    for j in range(1, kmax):
        sel = k > j
        if not np.any(sel):
            continue
        uj, kj, cj = u[sel], k[sel], ctx[sel]
        b = (uj >> (kj - 1 - j)) & 1
        if j <= top:
            pref = (uj >> (kj - j)) & ((1 << j) - 1)
            key = (kj * 8 + j) * 16 + pref
            keys.append(key * 2 + 0)
        else:
            key = (kj * 64 + j)
            keys.append(key * 2 + 1 + (1 << 40))
        bits.append(b)
    if keys:
        L += kt_bits(np.concatenate(keys), np.concatenate(bits))
    return L


def zigzag(e):
    e = np.asarray(e, dtype=np.int64)
    return np.where(e >= 0, 2 * e, -2 * e - 1)


def wrap(e, nbits):
    """Residual modulo 2^nbits, mapped to the signed range [-2^(nbits-1), 2^(nbits-1))."""
    m = 1 << nbits
    return ((np.asarray(e, dtype=np.int64) + m // 2) % m) - m // 2


def code_values(u, ctx, nbits):
    """Code length of non-negative values u < 2^nbits in contexts ctx (tree if small, else Exp-Golomb)."""
    if nbits <= 10:
        return tree_bits(u, ctx, nbits)
    return eg_bits(u, ctx)


def compact(*parts):
    """Combine several small integer context features into one context id."""
    out = np.zeros(np.asarray(parts[0][0]).shape, dtype=np.int64)
    for p, n in parts:
        p = np.asarray(p, dtype=np.int64)
        out = out * int(n) + np.clip(p, 0, int(n) - 1)
    return out


def labels_bits(lab, K, shape=None):
    """Code length of a label sequence (or 2-D label grid): adaptive K-ary, context = previous label
    (left neighbour on a grid)."""
    lab = np.asarray(lab, dtype=np.int64)
    if lab.size == 0:
        return 0.0
    if lab.ndim == 1:
        prev = np.concatenate([[K], lab[:-1]])
    else:
        prev = np.full(lab.shape, K, dtype=np.int64)
        prev[:, 1:] = lab[:, :-1]
        prev[1:, 0] = lab[:-1, 0]
    return dirichlet_bits(lab.ravel(), prev.ravel(), K)


def level_labels(v, nlev, step=0.5):
    """Activity level of each group of values (rows of v): quantised log2 of the mean absolute
    deviation from the group mean, in steps of `step` octaves, clipped to nlev levels."""
    v = np.asarray(v, dtype=np.float64)
    mad = np.mean(np.abs(v - v.mean(axis=-1, keepdims=True)), axis=-1)
    lv = np.floor(np.log2(mad + 1.0) / step).astype(np.int64)
    return np.clip(lv, 0, nlev - 1)


def fixed_bits(n, nbits):
    return float(n) * nbits


# ----------------------------------------------------------------------------------------------
# chain driver
# ----------------------------------------------------------------------------------------------

SWITCHABLE = frozenset({"R2", "R3", "R4", "R5", "R6", "R7", "R8"})


def run_chain(codelen, order, raw_bits, assert_sum=True, switchable=SWITCHABLE):
    """Evaluate L(M0..Mk) for the ordered pattern list; codelen(frozenset) -> (bits, detail).

    Every switchable pattern has one on/off bit per item, coded at its step: the encoder switches the
    pattern on only if that lowers the code length, and the choice is kept for the later steps.
    `if_forced` keeps the increment the pattern would have had if always switched on."""
    active = []
    Ls, details, used, forced = [float(raw_bits)], [{}], {}, {}
    prev_code = None
    flags_bits = 0.0
    for p in order:
        L_on, det = codelen(frozenset(active + [p]))
        L_on = float(L_on)
        if p in switchable:
            flags_bits += 1.0
            on = prev_code is None or L_on < prev_code
        else:
            on = True
        forced[p] = (Ls[-1] - (L_on + flags_bits)) / raw_bits
        if on:
            active.append(p)
            prev_code = L_on
            details.append(det)
        else:
            details.append(details[-1])
        used[p] = on
        Ls.append(prev_code + flags_bits)
    inc = [(Ls[i] - Ls[i + 1]) / raw_bits for i in range(len(order))]
    total = (Ls[0] - Ls[-1]) / raw_bits
    if assert_sum:
        assert abs(sum(inc) - total) < 1e-9 * max(1.0, abs(total)), (sum(inc), total)
    return {"order": list(order), "raw_bits": float(raw_bits), "L": Ls, "increment": dict(zip(order, inc)),
            "total": total, "details": details, "used": used, "if_forced": forced}


def combine(results):
    """Sum several items' chains (same order) into one: increments relative to the summed raw bits."""
    order = results[0]["order"]
    raw = sum(r["raw_bits"] for r in results)
    Ls = [sum(r["L"][i] for r in results) for i in range(len(order) + 1)]
    inc = [(Ls[i] - Ls[i + 1]) / raw for i in range(len(order))]
    total = (Ls[0] - Ls[-1]) / raw
    assert abs(sum(inc) - total) < 1e-9 * max(1.0, abs(total))
    return {"order": order, "raw_bits": raw, "L": Ls, "increment": dict(zip(order, inc)), "total": total,
            "n_items": len(results)}


# ----------------------------------------------------------------------------------------------
# shared residual tools (1-D and 2-D)
# ----------------------------------------------------------------------------------------------

def lpc_predict(s, order, block=4096, window=32768, ridge=1e-3):
    """Causal linear prediction: for each block of `block` samples, coefficients are fitted by least
    squares (autocorrelation normal equations with a small ridge) on the last `window` already-coded
    samples, and the prediction is rounded to an integer. The first block is predicted by 0."""
    s = np.asarray(s, dtype=np.float64)
    n = len(s)
    pred = np.zeros(n, dtype=np.int64)
    for b0 in range(block, n, block):
        w = s[max(0, b0 - window):b0]
        if len(w) <= 2 * order:
            continue
        r = np.array([np.dot(w[k:], w[:len(w) - k]) for k in range(order + 1)])
        if r[0] <= 0:
            continue
        R = r[np.abs(np.arange(order)[:, None] - np.arange(order)[None, :])]
        R = R + ridge * r[0] * np.eye(order)
        a = np.linalg.solve(R, r[1:order + 1])
        b1 = min(n, b0 + block)
        seg = s[b0 - order:b1]
        p = np.convolve(seg, np.concatenate([[0.0], a]), mode="full")[order:order + (b1 - b0)]
        pred[b0:b1] = np.rint(p).astype(np.int64)
    return pred


def ltp(e, block, tmin, tmax, gain_levels):
    """Long-term (lag) prediction of residual e, chosen per block. Returns (new residual, side bits).
    Per block: one adaptive flag; if used, the lag (fixed length over [tmin, tmax]) and the gain index
    (fixed length over `gain_levels`). The encoder uses a block when the Gaussian estimate of the saving
    exceeds the side bits; the counted cost is the actual one."""
    e = np.asarray(e, dtype=np.int64)
    n = len(e)
    ef = e.astype(np.float64)
    out = e.copy()
    lag_bits = math.ceil(math.log2(tmax - tmin + 1))
    gbits = math.ceil(math.log2(len(gain_levels)))
    flags, side = [], 0.0
    c2 = np.concatenate([[0.0], np.cumsum(ef * ef)])
    for b0 in range(tmax, n, block):
        b1 = min(n, b0 + block)
        cur = ef[b0:b1]
        E0 = float(np.dot(cur, cur))
        if E0 <= 0:
            flags.append(0)
            continue
        # c[T] = sum_t cur[t] * e[t - T] for T in [tmin, tmax]
        hist = ef[b0 - tmax:b1 - tmin]
        m = len(cur) + len(hist)
        nf = 1 << (m - 1).bit_length()
        cc = np.fft.irfft(np.fft.rfft(hist, nf) * np.conj(np.fft.rfft(cur, nf)), nf)
        # index j in hist corresponds to e[b0 - tmax + j]; lag T = tmax - j
        Ts = np.arange(tmin, tmax + 1)
        js = tmax - Ts
        c = cc[js]
        En = c2[b1 - Ts] - c2[b0 - Ts]
        ok = En > 0
        red = np.where(ok, c * c / np.where(ok, En, 1), 0)
        k = int(np.argmax(red))
        T = int(Ts[k])
        beta = c[k] / En[k] if En[k] > 0 else 0.0
        gi = int(np.argmin(np.abs(np.asarray(gain_levels) - beta)))
        g = gain_levels[gi]
        lagged = ef[b0 - T:b1 - T]
        new = cur - np.rint(g * lagged)
        E1 = float(np.dot(new, new))
        est = 0.5 * len(cur) * math.log2(max(E0, 1.0) / max(E1, 1.0))
        if g != 0 and est > lag_bits + gbits + 1:
            out[b0:b1] = np.rint(cur - np.rint(g * lagged)).astype(np.int64)
            flags.append(1)
            side += lag_bits + gbits
        else:
            flags.append(0)
    side += kt_bits(np.zeros(len(flags)), np.array(flags))
    return out, side, {"ltp_blocks_used": int(np.sum(flags)), "ltp_blocks": len(flags)}


def lowrank(E2d, k, qbits=16):
    """Rank-k approximation of a 2-D residual array; unit-norm factors sent as float16 (qbits each)
    and one float32 scale per component.
    Returns (integer approximation, side bits)."""
    A = np.asarray(E2d, dtype=np.float64)
    # top-k right singular vectors from the smaller Gram matrix (same subspace as the SVD)
    if A.shape[0] >= A.shape[1]:
        w, V = np.linalg.eigh(A.T @ A)
        Vt = V[:, ::-1][:, :k].T
        US = A @ Vt.T
    else:
        w, U = np.linalg.eigh(A @ A.T)
        U = U[:, ::-1][:, :k]
        Vt = U.T @ A
        US = U
    # unit-norm factor columns in float16 plus one float32 scale per component (no float16 overflow)
    sc = np.linalg.norm(US, axis=0)
    sc = np.where(sc > 0, sc, 1.0).astype(np.float32).astype(np.float64)
    Uq = (US / sc).astype(np.float16).astype(np.float64) * sc
    Vq = Vt.astype(np.float16).astype(np.float64)
    approx = np.rint(Uq @ Vq).astype(np.int64)
    side = qbits * k * (A.shape[0] + A.shape[1]) + 32 * k
    return approx, float(side)


def dct_matrix(n):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    D = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    D[0] /= np.sqrt(2.0)
    return D


def sparse_transform(blocks, D, m, step=1.0):
    """Per block (rows of `blocks`, already in transform-ready vector form), keep the m largest
    transform coefficients, quantised with `step`; send flag + (index, value) per kept coefficient.
    Returns (integer approximation per block, side bits, used count)."""
    X = np.asarray(blocks, dtype=np.float64)
    nb, N = X.shape
    C = X @ D.T
    idx = np.argsort(-np.abs(C), axis=1)[:, :m]
    q = np.rint(np.take_along_axis(C, idx, axis=1) / step).astype(np.int64)
    Cs = np.zeros_like(C)
    np.put_along_axis(Cs, idx, q * step, axis=1)
    approx = np.rint(Cs @ D).astype(np.int64)
    ib = math.ceil(math.log2(N))
    vbits = (2 * bitlen(np.abs(q)) + 1 + (q != 0)).sum(axis=1).astype(np.float64)
    side = m * ib + vbits
    E0 = (X * X).sum(axis=1)
    R = X - approx
    E1 = (R * R).sum(axis=1)
    est = 0.5 * N * np.log2(np.maximum(E0, 1.0) / np.maximum(E1, 1.0))
    use = (est > side + 1) & (q != 0).any(axis=1)
    approx[~use] = 0
    bits = float(side[use].sum()) + kt_bits(np.zeros(nb), use.astype(np.int64))
    return approx, bits, int(use.sum())


def lzp(seq, k):
    """For each position t >= k: the position p < t whose preceding k values equal those before t
    (last such p), or -1. Hash of k values; a collision only lowers efficiency (decoder does the same)."""
    s = np.asarray(seq, dtype=np.int64)
    n = len(s)
    prevpos = np.full(n, -1, dtype=np.int64)
    if n <= k:
        return prevpos
    h = np.zeros(n - k + 1, dtype=np.uint64)
    P = np.uint64(1000003)
    with np.errstate(over="ignore"):
        for j in range(k):
            h = h * P + (s[j:n - k + 1 + j].astype(np.uint64) + np.uint64(1 << 20))
    # h[i] = hash of s[i:i+k]; context for position t = h[t-k]
    ctxh = h[:n - k]                       # for t = k..n-1
    t = np.arange(k, n)
    o = np.lexsort((t, ctxh))
    same = np.concatenate([[False], ctxh[o][1:] == ctxh[o][:-1]])
    prev_t = np.where(same, np.concatenate([[-1], t[o][:-1]]), -1)
    prevpos[t[o]] = prev_t
    return prevpos


def prev_class(u, shift, cap):
    """Bit-length class of the value `shift` positions earlier (0 when none), capped."""
    c = np.zeros(len(u), dtype=np.int64)
    if shift < len(u):
        c[shift:] = np.minimum(bitlen(u[:-shift]), cap)
    return c


# ----------------------------------------------------------------------------------------------
# 1-D chain: audio and speech (integer PCM)
# ----------------------------------------------------------------------------------------------

AUDIO_DEFAULT = {"frame": 1024, "levels": 16, "level_step": 0.5, "ctx_order": 2, "ctx_cap": 15,
                 "lpc_order": 16, "lpc_block": 4096, "lpc_window": 32768,
                 "rank_frame": 1024, "rank": 1, "lzp_k": 4, "ltp_block": 1024, "ltp_tmin": 20,
                 "ltp_tmax": 1023, "ltp_gains": [0.125 * i for i in range(-8, 9) if i != 0],
                 "sparse_block": 256, "sparse_m": 4, "sparse_step": 1.0}


def audio_channel_bits(s, nb, flags, P, info):
    """Code length of one integer channel s (nb-bit range) under the switched-on patterns."""
    n = len(s)
    if "R1" not in flags:
        return fixed_bits(n, nb)  # fixed-length code; other patterns cannot act without adaptive coding
    side = 0.0
    e = np.zeros(n, dtype=np.int64)
    pred = np.zeros(n, dtype=np.int64)
    if "R4" in flags:
        pred = lpc_predict(s, P["lpc_order"], P["lpc_block"], P["lpc_window"])
    e = wrap(s - pred, nb)
    if "R5" in flags:
        F = P["rank_frame"]
        m = n // F
        if m >= 2:
            ap, sb = lowrank(e[:m * F].reshape(m, F), P["rank"])
            e[:m * F] = wrap(e[:m * F] - ap.ravel(), nb)
            side += sb
    if "R7" in flags:
        e, sb, d = ltp(e, P["ltp_block"], P["ltp_tmin"], P["ltp_tmax"], P["ltp_gains"])
        e = wrap(e, nb)
        side += sb
        info.update(d)
    if "R8" in flags:
        N = P["sparse_block"]
        m = n // N
        if m >= 1:
            ap, sb, used = sparse_transform(e[:m * N].reshape(m, N), dct_matrix(N), P["sparse_m"], P["sparse_step"])
            e[:m * N] = wrap(e[:m * N] - ap.ravel(), nb)
            side += sb
            info["sparse_blocks_used"] = info.get("sparse_blocks_used", 0) + used
    u = zigzag(e)
    ctx_parts = []
    if "R2" in flags:
        F = P["frame"]
        nfr = -(-n // F)
        pad = np.zeros(nfr * F, dtype=np.int64)
        pad[:n] = e
        cnt = np.minimum(F, n - np.arange(nfr) * F)
        v = pad.reshape(nfr, F).astype(np.float64)
        mean = v.sum(axis=1) / cnt
        mad = np.abs(v - mean[:, None])
        mad[np.arange(F)[None, :] >= cnt[:, None]] = 0
        mad = mad.sum(axis=1) / cnt
        lab = np.clip(np.floor(np.log2(mad + 1.0) / P["level_step"]).astype(np.int64), 0, P["levels"] - 1)
        side += labels_bits(lab, P["levels"])
        ctx_parts.append((np.repeat(lab, F)[:n], P["levels"]))
    if "R3" in flags:
        for j in range(1, P["ctx_order"] + 1):
            ctx_parts.append((prev_class(u, j, P["ctx_cap"]), P["ctx_cap"] + 1))
    ctx = compact(*ctx_parts) if ctx_parts else np.zeros(n, dtype=np.int64)
    coded = np.ones(n, dtype=bool)
    if "R6" in flags:
        pp = lzp(s, P["lzp_k"])
        has = pp >= 0
        hit = np.zeros(n, dtype=bool)
        hit[has] = s[has] == s[pp[has]]
        prevhit = np.concatenate([[False], hit[:-1]])
        side += kt_bits(prevhit[has].astype(np.int64), hit[has].astype(np.int64))
        coded = ~hit
        info["lzp_hits"] = info.get("lzp_hits", 0) + int(hit.sum())
    return side + code_values(u[coded], ctx[coded], nb)


def audio_codelen(x, flags, P):
    """x: int array (channels, samples) of signed nb-bit PCM (nb = 16)."""
    x = np.asarray(x, dtype=np.int64)
    chans, nbs = [x[c] for c in range(x.shape[0])], [16] * x.shape[0]
    if "R10" in flags and x.shape[0] == 2:
        L, R = x[0], x[1]
        S = L - R
        M = (L + R) >> 1          # floor; L = M + ((S + 1) >> 1) ... exactly invertible with S
        chans, nbs = [M, S], [16, 17]
    info = {}
    tot = sum(audio_channel_bits(c, nb, flags, P, info) for c, nb in zip(chans, nbs))
    return tot, info


# ----------------------------------------------------------------------------------------------
# 2-D chain: images (8-bit RGB), planes coded one after another in raster order
# ----------------------------------------------------------------------------------------------

IMAGE_DEFAULT = {"block": 32, "levels": 16, "level_step": 0.5, "ctx_neighbors": 2, "ctx_cap": 7,
                 "raw_ctx_bits": 3, "lp_rows": 8, "lp_window_rows": 64, "rank": 1, "lzp_k": 6,
                 "bm_block": 16, "bm_range": 16, "bm_gains": [0.125 * i for i in range(-8, 9) if i != 0],
                 "sparse_block": 8, "sparse_m": 2, "sparse_step": 4.0}


def ycocg_r(img):
    R, G, B = (img[..., c].astype(np.int64) for c in range(3))
    Co = R - B
    t = B + (Co >> 1)
    Cg = G - t
    Y = t + (Cg >> 1)
    return [Y, Co, Cg], [8, 9, 9]


def _shift(a, dy, dx, fill=0):
    """b[i, j] = a[i + dy, j + dx] (fill outside)."""
    H, W = a.shape
    b = np.full_like(a, fill)
    ys, ye = max(0, -dy), min(H, H - dy)
    xs, xe = max(0, -dx), min(W, W - dx)
    if ys < ye and xs < xe:
        b[ys:ye, xs:xe] = a[ys + dy:ye + dy, xs + dx:xe + dx]
    return b


def image_lp(plane, prev_planes, P):
    """Causal linear prediction of a plane from its causal neighbours (W, N, NW, NE) and, for each earlier
    plane, the co-located value and its W and N neighbours. Coefficients (with a constant) are refitted
    every `lp_rows` rows by least squares on the last `lp_window_rows` already-coded rows; the first rows
    use the W neighbour (N in the first column)."""
    x = plane.astype(np.float64)
    H, W = x.shape
    feats = [_shift(x, 0, -1), _shift(x, -1, 0), _shift(x, -1, -1), _shift(x, -1, 1)]
    for q in prev_planes:
        qf = q.astype(np.float64)
        feats += [qf, _shift(qf, 0, -1), _shift(qf, -1, 0)]
    feats.append(np.ones_like(x))
    X = np.stack(feats, axis=-1)            # H, W, F
    F = X.shape[-1]
    # per-row normal equations
    XtX = np.einsum("hwf,hwg->hfg", X, X)
    Xty = np.einsum("hwf,hw->hf", X, x)
    cXtX = np.concatenate([np.zeros((1, F, F)), np.cumsum(XtX, axis=0)])
    cXty = np.concatenate([np.zeros((1, F)), np.cumsum(Xty, axis=0)])
    pred = np.zeros((H, W))
    first = np.where(np.arange(W)[None, :] > 0, feats[0], feats[1])
    step, win = P["lp_rows"], P["lp_window_rows"]
    for r0 in range(0, H, step):
        r1 = min(H, r0 + step)
        a0 = max(0, r0 - win)
        if r0 < 2:
            pred[r0:r1] = first[r0:r1]
            continue
        A = cXtX[r0] - cXtX[a0]
        b = cXty[r0] - cXty[a0]
        A = A + 1e-4 * np.diag(np.diag(A)) + 1e-9 * np.eye(F)   # ridge relative to each feature's own scale
        coef = np.linalg.solve(A, b)
        pred[r0:r1] = X[r0:r1] @ coef
    return np.rint(pred).astype(np.int64)


def block_match(e, P):
    """Intra block matching on residual plane e: per block, the causal offset (dy, dx) in the search range
    and a gain; e' = e - round(g * e[i + dy, j + dx]). Side: adaptive flag per block, then fixed-length
    offset and gain index."""
    H, W = e.shape
    B, S = P["bm_block"], P["bm_range"]
    gains = np.asarray(P["bm_gains"])
    offs = [(dy, dx) for dy in range(-S, 1) for dx in range(-S, S + 1) if dy < 0 or dx < 0]
    obits = math.ceil(math.log2(len(offs)))
    gbits = math.ceil(math.log2(len(gains)))
    nby, nbx = H // B, W // B
    ef = e.astype(np.float64)
    cur = ef[:nby * B, :nbx * B].reshape(nby, B, nbx, B)
    E0 = (cur * cur).sum(axis=(1, 3))
    best = np.zeros((nby, nbx))
    bidx = np.zeros((nby, nbx), dtype=np.int64)
    bc = np.zeros((nby, nbx))
    bE = np.ones((nby, nbx))
    for k, (dy, dx) in enumerate(offs):
        sh = _shift(ef, dy, dx)[:nby * B, :nbx * B].reshape(nby, B, nbx, B)
        c = (cur * sh).sum(axis=(1, 3))
        En = (sh * sh).sum(axis=(1, 3))
        red = np.where(En > 0, c * c / np.maximum(En, 1e-9), 0)
        upd = red > best
        best[upd], bidx[upd], bc[upd], bE[upd] = red[upd], k, c[upd], En[upd]
    beta = np.where(bE > 0, bc / bE, 0)
    gi = np.argmin(np.abs(gains[None, None, :] - beta[..., None]), axis=-1)
    g = gains[gi]
    out = e.copy()
    use = np.zeros((nby, nbx), dtype=bool)
    for by in range(nby):
        for bx in range(nbx):
            dy, dx = offs[bidx[by, bx]]
            ys, xs = by * B, bx * B
            blk = ef[ys:ys + B, xs:xs + B]
            ref = _shift(ef, dy, dx)[ys:ys + B, xs:xs + B]
            new = blk - np.rint(g[by, bx] * ref)
            E1 = float((new * new).sum())
            est = 0.5 * B * B * math.log2(max(E0[by, bx], 1.0) / max(E1, 1.0))
            if est > obits + gbits + 1:
                out[ys:ys + B, xs:xs + B] = new.astype(np.int64)
                use[by, bx] = True
    side = float(use.sum()) * (obits + gbits) + kt_bits(np.zeros(use.size), use.ravel().astype(np.int64))
    return out, side, int(use.sum())


def image_plane_bits(x, nb, signed, prev, flags, P, info):
    H, W = x.shape
    n = H * W
    if "R1" not in flags:
        return fixed_bits(n, nb)
    side = 0.0
    off = (1 << (nb - 1)) if signed else 0
    xs = x + off                              # values in [0, 2^nb)
    if "R4" in flags:
        pred = image_lp(xs, [q[0] + q[1] for q in prev], P)
    else:
        pred = np.full_like(xs, 1 << (nb - 1))   # fixed mid-level: coding x - 2^(nb-1) is a bijection of x
    e = wrap(xs - pred, nb)
    resid = True
    if "R5" in flags:
        ap, sb = lowrank(e, P["rank"])
        e = wrap(e - ap, nb) if resid else (e - ap) % (1 << nb)
        side += sb
    if "R7" in flags:
        e2, sb, used = block_match(e, P)
        e = wrap(e2, nb) if resid else e2 % (1 << nb)
        side += sb
        info["bm_blocks_used"] = info.get("bm_blocks_used", 0) + used
    if "R8" in flags:
        Bk = P["sparse_block"]
        nby, nbx = H // Bk, W // Bk
        D = dct_matrix(Bk)
        D2 = np.kron(D, D)
        blk = e[:nby * Bk, :nbx * Bk].reshape(nby, Bk, nbx, Bk).transpose(0, 2, 1, 3).reshape(-1, Bk * Bk)
        ap, sb, used = sparse_transform(blk, D2, P["sparse_m"], P["sparse_step"])
        ap = ap.reshape(nby, nbx, Bk, Bk).transpose(0, 2, 1, 3).reshape(nby * Bk, nbx * Bk)
        sub = e[:nby * Bk, :nbx * Bk] - ap
        e[:nby * Bk, :nbx * Bk] = wrap(sub, nb) if resid else sub % (1 << nb)
        side += sb
        info["sparse_blocks_used"] = info.get("sparse_blocks_used", 0) + used
    u = zigzag(e) if resid else e
    ctx_parts = []
    if "R2" in flags:
        Bk = P["block"]
        nby, nbx = -(-H // Bk), -(-W // Bk)
        v = np.zeros((nby * Bk, nbx * Bk))
        msk = np.zeros_like(v)
        v[:H, :W] = e
        msk[:H, :W] = 1
        vb = v.reshape(nby, Bk, nbx, Bk)
        mb = msk.reshape(nby, Bk, nbx, Bk)
        cnt = mb.sum(axis=(1, 3))
        mean = vb.sum(axis=(1, 3)) / cnt
        mad = (np.abs(vb - mean[:, None, :, None]) * mb).sum(axis=(1, 3)) / cnt
        lab = np.clip(np.floor(np.log2(mad + 1.0) / P["level_step"]).astype(np.int64), 0, P["levels"] - 1)
        side += labels_bits(lab, P["levels"], shape=lab.shape)
        full = np.repeat(np.repeat(lab, Bk, axis=0), Bk, axis=1)[:H, :W]
        ctx_parts.append((full, P["levels"]))
    if "R3" in flags:
        if "R4" in flags:      # neighbours' residual classes
            q = np.minimum(bitlen(u), P["ctx_cap"])
            nq = P["ctx_cap"] + 1
        else:                  # neighbours' quantised values
            q = xs >> (nb - P["raw_ctx_bits"])
            nq = 1 << P["raw_ctx_bits"]
        nbrs = [(0, -1), (-1, 0), (-1, -1), (-1, 1)][:P["ctx_neighbors"]]
        for dy, dx in nbrs:
            ctx_parts.append((_shift(q, dy, dx, fill=0), nq))
    ctx = compact(*ctx_parts) if ctx_parts else np.zeros((H, W), dtype=np.int64)
    coded = np.ones(n, dtype=bool)
    if "R6" in flags:
        s = xs.ravel()
        pp = lzp(s, P["lzp_k"])
        has = pp >= 0
        hit = np.zeros(n, dtype=bool)
        hit[has] = s[has] == s[pp[has]]
        prevhit = np.concatenate([[False], hit[:-1]])
        side += kt_bits(prevhit[has].astype(np.int64), hit[has].astype(np.int64))
        coded = ~hit
        info["lzp_hits"] = info.get("lzp_hits", 0) + int(hit.sum())
    return side + code_values(u.ravel()[coded], ctx.ravel()[coded], nb)


def image_codelen(img, flags, P):
    """img: uint8 array (H, W, 3)."""
    if "R10" in flags:
        planes, nbs = ycocg_r(img)
        signed = [False, True, True]
    else:
        planes, nbs = [img[..., c].astype(np.int64) for c in range(img.shape[-1])], [8] * img.shape[-1]
        signed = [False] * len(planes)
    info, tot, prev = {}, 0.0, []
    for x, nb, sg in zip(planes, nbs, signed):
        tot += image_plane_bits(x, nb, sg, prev, flags, P, info)
        prev.append((x, (1 << (nb - 1)) if sg else 0))
    return tot, info


# ----------------------------------------------------------------------------------------------
# hierarchical (back-off) adaptive code for K-ary symbols
# ----------------------------------------------------------------------------------------------

def _gammaln(x):
    import torch
    torch.set_num_threads(1)
    return torch.special.gammaln(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))).numpy()


def block_edges(n, nblocks=32, first=256):
    """Block boundaries: doubling from `first` up to n / nblocks, then equal steps of n / nblocks."""
    step = max(1, n // nblocks)
    e = [0]
    b = first
    while b < step and b < n:
        e.append(b)
        b *= 2
    x = e[-1] + step if len(e) > 1 else step
    while x < n:
        e.append(x)
        x += step
    return np.asarray(e, dtype=np.int64)


def fast_unique(key):
    """(unique values, inverse) for non-negative int keys; counting instead of sorting when the range is small."""
    mx = int(key.max()) + 1 if key.size else 1
    if mx <= max(4 * key.size, 1 << 22):
        present = np.bincount(key, minlength=mx) > 0
        u = np.flatnonzero(present)
        m = np.cumsum(present) - 1
        return u, m[key]
    return np.unique(key, return_inverse=True)


def hier_bits(sym, levels, K, beta, nblocks=32, alpha=0.5, first=256, cache=None, names=None):
    """Sequential adaptive code of K-ary `sym` (in coding order) with nested contexts.

    levels: list of context arrays, coarse to fine; each finer context refines the previous one.
    Level 0 is an add-alpha estimate. A finer level predicts (n_ctx,a + beta * q_parent(a)) / (n_ctx + beta),
    where q_parent is the parent level's prediction frozen at the start of each block (blocks double in
    size from `first` symbols, then are n / nblocks long). Freezing per block keeps the code causal and
    gives its length in closed form per block and context."""
    sym = np.asarray(sym, dtype=np.int64).ravel()
    n = len(sym)
    if n == 0:
        return 0.0
    if not levels:
        levels = [np.zeros(n, dtype=np.int64)]
    edges = block_edges(n, nblocks, first)
    blk = np.searchsorted(edges, np.arange(n), side="right") - 1
    prev_id = None
    prev_rows = None   # (row keys sorted, q per row, U)
    start = 0
    if cache is not None and names is not None:
        # reuse the longest cached prefix of levels (same names = same context arrays and symbols)
        for j in range(len(levels) - 1, 0, -1):
            hit = cache.get(tuple(names[:j]))
            if hit is not None:
                prev_id, prev_rows = hit
                start = j
                break
    for lv, ctx in enumerate(levels):
        if lv < start:
            continue
        ctx = np.asarray(ctx, dtype=np.int64).ravel()
        key = ctx if prev_id is None else prev_id * (int(ctx.max()) + 1) + ctx
        u, inv = fast_unique(key)
        idv = inv.astype(np.int64)
        U = len(u)
        par = None
        if prev_id is not None:
            par = np.zeros(U, dtype=np.int64)
            par[idv] = prev_id
        rk_all = blk * U + idv
        rkeys, rinv = fast_unique(rk_all)
        Rn = len(rkeys)
        M = np.bincount(rinv * K + sym, minlength=Rn * K).reshape(Rn, K).astype(np.float64)
        rb, rid = rkeys // U, rkeys % U
        order = np.lexsort((rb, rid))
        Mo = M[order]
        cum = np.cumsum(Mo, axis=0)
        excl = cum - Mo
        ido = rid[order]
        first_row = np.concatenate([[True], ido[1:] != ido[:-1]])
        g0 = np.maximum.accumulate(np.where(first_row, np.arange(Rn), 0))
        Co = excl - excl[g0]
        C = np.empty_like(Co)
        C[order] = Co
        if lv == 0:
            prior = np.full((Rn, K), alpha)
        else:
            pkeys, pq, pU = prev_rows
            pidx = np.searchsorted(pkeys, rb * pU + par[rid])
            prior = beta * pq[pidx]
        a = C + prior
        if lv == len(levels) - 1:
            mt = M.sum(axis=1)
            at = a.sum(axis=1)
            lnp = (_gammaln(at) - _gammaln(at + mt)).sum()
            sel = M > 0
            lnp += (_gammaln(a[sel] + M[sel]) - _gammaln(a[sel])).sum()
            return -lnp / math.log(2)
        prev_rows = (rkeys, a / a.sum(axis=1, keepdims=True), U)
        prev_id = idv
        if cache is not None and names is not None:
            cache[tuple(names[:lv + 1])] = (prev_id, prev_rows)
    raise AssertionError


# ----------------------------------------------------------------------------------------------
# FP4 chain: one weight matrix of 4-bit codes with one 8-bit scale per block of 32
# ----------------------------------------------------------------------------------------------

FP4_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
FP4_DEFAULT = {"block": 32, "beta": 64.0, "nblocks": 32, "ctx_order": 1, "lp_order": 4, "lp_rows": 64,
               "pred_tau": 0.25, "rank": 1, "lzp_k": 12, "dict_atoms": 16, "dict_iters": 8, "dict_sample": 20000}


def _row_lp(V, p, rows_per_fit):
    """Causal linear prediction of each value from the previous p values of its row; coefficients are
    refitted every `rows_per_fit` rows on all earlier rows (least squares)."""
    R, W = V.shape
    sh = [np.concatenate([np.zeros((R, k)), V[:, :-k]], axis=1) for k in range(1, p + 1)]
    XtX = np.zeros((R, p, p))
    Xty = np.zeros((R, p))
    for a in range(p):
        Xty[:, a] = (sh[a] * V).sum(axis=1)
        for b in range(a, p):
            XtX[:, a, b] = XtX[:, b, a] = (sh[a] * sh[b]).sum(axis=1)
    cA = np.concatenate([np.zeros((1, p, p)), np.cumsum(XtX, axis=0)])
    cb = np.concatenate([np.zeros((1, p)), np.cumsum(Xty, axis=0)])
    pred = np.zeros((R, W))
    for r0 in range(rows_per_fit, R, rows_per_fit):
        r1 = min(R, r0 + rows_per_fit)
        A = cA[r0] + 1e-6 * np.trace(cA[r0]) * np.eye(p)
        coef = np.linalg.solve(A, cb[r0])
        pred[r0:r1] = sum(coef[k] * sh[k][r0:r1] for k in range(p))
    return pred


def _tri(x, tau):
    return np.where(x < -tau, 0, np.where(x > tau, 2, 1)).astype(np.int64)


def _kmeans(X, K, iters, rng):
    Cn = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        d = (X * X).sum(1)[:, None] - 2 * X @ Cn.T + (Cn * Cn).sum(1)[None, :]
        lab = d.argmin(1)
        for k in range(K):
            s = lab == k
            if s.any():
                Cn[k] = X[s].mean(0)
    return Cn


def fp4_codelen(C, E, flags, P, order, partner=None, seed=0, cache=None):
    """C: codes (R, W) in 0..15 (bit 3 = sign); E: scale bytes (R, W / block); partner: codes of the
    partner matrix (same shape) or None. Patterns: R1c (code histogram), R1s (scale-stream histogram),
    R2..R8, R10. `order` fixes the order in which context levels are nested."""
    C = np.asarray(C, dtype=np.int64)
    R, W = C.shape
    n = R * W
    Bk = P["block"]
    info = {}
    bits = 0.0
    # scale stream
    bits += dirichlet_bits(E.ravel(), None, 256) if "R1s" in flags else 8.0 * E.size
    merge = "R10" in flags
    Cm = np.where(C == 8, 0, C) if merge else C
    sym = np.where(Cm > 8, Cm - 1, Cm) if merge else Cm      # 0..14 after merging, else 0..15
    K = 15 if merge else 16
    if "R1c" not in flags:
        return bits + n * math.log2(K), info
    V = FP4_VALUES[Cm]
    tau = P["pred_tau"] * float(np.sqrt(np.mean(V * V)) or 1.0)
    fc = cache if cache is not None else {}

    def feat(name, make):
        # every feature (and its side bits) is computed once per matrix and merge state
        key = ("feat", name, merge)
        if key not in fc:
            fc[key] = make()
        return fc[key]

    feats = {}
    side = 0.0

    def f_r2():
        su, sinv = np.unique(E, return_inverse=True)
        return np.repeat(sinv.reshape(E.shape), Bk, axis=1).ravel(), 0.0

    def f_r3():
        f = np.zeros((R, W), dtype=np.int64)
        for j in range(1, P["ctx_order"] + 1):
            prevc = np.full((R, W), 16, dtype=np.int64)
            prevc[:, j:] = Cm[:, :-j]
            f = f * 17 + prevc
        return f.ravel(), 0.0

    def f_r4():
        pv = _row_lp(V, P["lp_order"], P["lp_rows"])
        A = np.abs(V)
        pa = _row_lp(A - A.mean(), P["lp_order"], P["lp_rows"])
        return (_tri(pv, tau) * 3 + _tri(pa, tau)).ravel(), 0.0

    def f_r5():
        ap, sb = _lowrank_float(V, P["rank"])
        return _tri(ap, tau).ravel(), sb

    def f_r7():
        Pm = np.asarray(partner, dtype=np.int64)
        Pm = np.where(Pm == 8, 0, Pm) if merge else Pm
        Vp = FP4_VALUES[Pm]
        rs = np.sign((V * Vp).sum(axis=1))
        rs[rs == 0] = 1
        sb = kt_bits(np.zeros(R), (rs > 0).astype(np.int64))
        return ((np.sign(Vp) * rs[:, None] + 1).astype(np.int64) * 2 + (np.abs(Vp) >= 2.0)).ravel(), sb

    def f_r8():
        rng = np.random.default_rng(seed)
        X = V.reshape(R * (W // Bk), Bk)
        smp = X[rng.choice(len(X), min(P["dict_sample"], len(X)), replace=False)]
        D = _kmeans(smp, P["dict_atoms"], P["dict_iters"], rng).astype(np.float16).astype(np.float64)
        d = (X * X).sum(1)[:, None] - 2 * X @ D.T + (D * D).sum(1)[None, :]
        lab = d.argmin(1)
        sb = 16.0 * D.size + dirichlet_bits(lab, None, P["dict_atoms"])
        return _tri(D[lab].reshape(R, W), tau).ravel(), sb

    makers = {"R2": f_r2, "R3": f_r3, "R4": f_r4, "R5": f_r5, "R8": f_r8}
    if partner is not None:
        makers["R7"] = f_r7
    for p, mk in makers.items():
        if p in flags:
            feats[p], sb = feat(p, mk)
            side += sb
    levels = [np.zeros(n, dtype=np.int64)] + [feats[p] for p in order if p in feats]
    coded = np.ones(n, dtype=bool)
    if "R6" in flags:
        def f_r6():
            s6 = Cm.ravel()
            pp = lzp(s6, P["lzp_k"])
            has = pp >= 0
            hit = np.zeros(n, dtype=bool)
            hit[has] = s6[has] == s6[pp[has]]
            prevhit = np.concatenate([[False], hit[:-1]])
            return hit, kt_bits(prevhit[has].astype(np.int64), hit[has].astype(np.int64))
        hit, sb = feat("R6", f_r6)
        side += sb
        coded = ~hit
        info["lzp_hits"] = int(hit.sum())
    levels = [lv[coded] for lv in levels] if "R6" in flags else levels
    names = [("merge" if merge else "plain", "R6" in flags)] + [p for p in order if p in feats]
    bits += side + hier_bits(sym.ravel()[coded], levels, K, P["beta"], P["nblocks"], cache=cache, names=names)
    return bits, info


def _lowrank_float(V, k, qbits=16):
    A = np.asarray(V, dtype=np.float64)
    if A.shape[0] >= A.shape[1]:
        w, Vv = np.linalg.eigh(A.T @ A)
        Vt = Vv[:, ::-1][:, :k].T
        US = A @ Vt.T
    else:
        w, U = np.linalg.eigh(A @ A.T)
        US = U[:, ::-1][:, :k]
        Vt = US.T @ A
    sc = np.linalg.norm(US, axis=0)
    sc = np.where(sc > 0, sc, 1.0).astype(np.float32).astype(np.float64)
    Uq = (US / sc).astype(np.float16).astype(np.float64) * sc
    Vq = Vt.astype(np.float16).astype(np.float64)
    return Uq @ Vq, float(qbits * k * (A.shape[0] + A.shape[1]) + 32 * k)
