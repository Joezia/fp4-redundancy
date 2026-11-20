"""TX stage 1: local transforms (A1-A5, B1-B5, C1, C2, D1). Each method: fn(m: txcore.Mat, variant: dict) -> enc.

enc = {"streams": [txcore.Stream], "side": fixed side bits, "roundtrip": bool, "info": {...}}
Value-view variants carry dom = 'L' (block levels, in-block transforms) or 'X' (aligned values, cross-block).
Every method performs the inverse transform (decoder path) and checks it against the matrix.
"""
import math

import numpy as np

import txcore as T

S2 = math.sqrt(2.0)


def dom_arr(m, v):
    return m.L if v.get("dom", "L") == "L" else m.X


def check_val(m, v, A):
    return m.check_codes(L=A) if v.get("dom", "L") == "L" else m.check_codes(X=A)


def band_layout(G):
    """(R, g, N) -> (R, N * g) with column k * g + j = coefficient k of group j (subbands side by side)."""
    R, g, N = G.shape
    return np.ascontiguousarray(G.transpose(0, 2, 1).reshape(R, N * g))


def oriented(A, axis):
    return A if axis == "row" else np.ascontiguousarray(A.T)


def back(A, axis):
    return A if axis == "row" else np.ascontiguousarray(A.T)


# ------------------------------------------------------------------ A1 WHT (S-transform butterflies)

def wht_fwd(x):
    x = x.copy()
    N = x.shape[-1]
    h = 1
    while h < N:
        y = x.reshape(x.shape[:-1] + (N // (2 * h), 2, h))
        a, b = y[..., 0, :].copy(), y[..., 1, :].copy()
        d = a - b
        y[..., 0, :] = b + (d >> 1)
        y[..., 1, :] = d
        h *= 2
    return x


def wht_inv(c):
    x = c.copy()
    N = x.shape[-1]
    h = N // 2
    while h >= 1:
        y = x.reshape(x.shape[:-1] + (N // (2 * h), 2, h))
        s, d = y[..., 0, :].copy(), y[..., 1, :].copy()
        b = s - (d >> 1)
        y[..., 0, :] = d + b
        y[..., 1, :] = b
        h //= 2
    return x


def signs(N, seed=7):
    return np.where(np.random.default_rng([seed, N]).random(N) < 0.5, -1, 1).astype(np.int64)


def A1_wht(m, v):
    """v: dom, axis ('row' / 'col'), N, rsign (bool)."""
    A = oriented(dom_arr(m, v), v["axis"])
    R, W = A.shape
    N = v["N"]
    G = A.reshape(R, W // N, N)
    sg = signs(N) if v.get("rsign") else np.ones(N, dtype=np.int64)
    Cf = wht_fwd(G * sg)
    rec = back((wht_inv(Cf) * sg).reshape(R, W), v["axis"])
    info = T.coding_gain(Cf, np.broadcast_to(np.arange(N), Cf.shape))
    lay = back(band_layout(Cf), v["axis"])
    return {"streams": [T.Stream(lay, "int", "2d", name="coef")], "roundtrip": check_val(m, v, rec), "info": info}


# ------------------------------------------------------------------ A2 DCT / A3 DFT / A5 lapped: approximation + residual

def dct_mat(N):
    k = np.arange(N)[:, None]
    n = np.arange(N)[None, :]
    D = np.cos(math.pi * (2 * n + 1) * k / (2 * N)) * math.sqrt(2.0 / N)
    D[0] /= S2
    return D


def dct4_mat(M):
    k = np.arange(M)[:, None]
    n = np.arange(M)[None, :]
    return np.cos(math.pi * (2 * n + 1) * (2 * k + 1) / (4 * M)) * math.sqrt(2.0 / M)


def rdft_fwd(G):
    N = G.shape[-1]
    F = np.fft.rfft(G.astype(np.float64), axis=-1, norm="ortho")
    out = np.empty(G.shape, dtype=np.float64)
    h = N // 2
    out[..., 0] = F[..., 0].real
    out[..., 1:h] = S2 * F[..., 1:h].real
    out[..., h] = F[..., h].real
    out[..., h + 1:] = S2 * F[..., 1:h].imag
    return out


def rdft_inv(c):
    N = c.shape[-1]
    h = N // 2
    F = np.zeros(c.shape[:-1] + (h + 1,), dtype=np.complex128)
    F[..., 0] = c[..., 0]
    F[..., 1:h] = (c[..., 1:h] + 1j * c[..., h + 1:]) / S2
    F[..., h] = c[..., h]
    return np.fft.irfft(F, n=N, axis=-1, norm="ortho")


def approx_residual(m, v, fwd, inv):
    """generic orthonormal block transform along `axis` in groups of N: Q = round(fwd(x) / q),
    x_hat = round(inv(Q q)), residual = x - x_hat (exact). Streams: Q (subband layout), residual (2-D)."""
    A = oriented(dom_arr(m, v), v["axis"])
    R, W = A.shape
    N = W if v["N"] == "full" else v["N"]
    G = A.reshape(R, W // N, N).astype(np.float64)
    c = fwd(G)
    q = v["q"]
    Q = np.round(c / q).astype(np.int64)
    xh = np.round(inv(Q * q)).astype(np.int64)                 # decoder recomputes x_hat from Q
    res = A.reshape(R, W // N, N) - xh
    rec = back((inv(Q * q).round().astype(np.int64) + res).reshape(R, W), v["axis"])
    info = T.coding_gain(c, np.broadcast_to(np.arange(N), c.shape))
    info["residual_nonzero"] = float(np.count_nonzero(res) / res.size)
    return {"streams": [T.Stream(back(band_layout(Q), v["axis"]), "int", "2d", name="coef"),
                        T.Stream(back(res.reshape(R, W), v["axis"]), "int", "2d", name="residual")],
            "roundtrip": check_val(m, v, rec), "info": info}


def A2_dct(m, v):
    D = dct_mat(v["N"])
    return approx_residual(m, v, lambda G: G @ D.T, lambda C: C @ D)


def A3_dft(m, v):
    return approx_residual(m, v, rdft_fwd, rdft_inv)


def lapped_mats(M):
    J = np.eye(M)[::-1]
    V = J @ dct_mat(M).T @ dct4_mat(M) @ J
    I = np.eye(M)
    Wb = np.block([[I, J], [J, -I]]) / S2
    P = Wb @ np.block([[I, np.zeros((M, M))], [np.zeros((M, M)), V]]) @ Wb
    return P


def A5_lapped(m, v):
    """time-domain lapped transform (Tran et al. 2003): an orthogonal 2M-point prefilter across every internal
    32-block boundary of a row (V = J C2^T C4 J), then DCT-32 per block; approximation + residual."""
    M, N, q = v["M"], 32, v["q"]
    P = lapped_mats(M)
    A = m.X
    R, W = A.shape
    idx = (np.arange(1, W // N)[:, None] * N + np.arange(-M, M)[None, :])        # (boundaries, 2M)
    D = dct_mat(N)

    def fwd(Y):
        Y = Y.astype(np.float64).copy()
        Y[:, idx] = Y[:, idx] @ P.T
        return Y.reshape(R, W // N, N) @ D.T

    def inv(Cq):
        Y = (Cq @ D).reshape(R, W)
        Y[:, idx] = Y[:, idx] @ P                                               # P orthogonal: P^-1 = P^T
        return Y

    c = fwd(A)
    Q = np.round(c / q).astype(np.int64)
    xh = np.round(inv(Q * q)).astype(np.int64)
    res = A - xh
    rec = np.round(inv(Q * q)).astype(np.int64) + res
    info = T.coding_gain(c, np.broadcast_to(np.arange(N), c.shape))
    info["residual_nonzero"] = float(np.count_nonzero(res) / res.size)
    return {"streams": [T.Stream(band_layout(Q), "int", "2d", name="coef"), T.Stream(res, "int", "2d", name="residual")],
            "roundtrip": m.check_codes(X=rec), "info": info}


# ------------------------------------------------------------------ A4 DWT (integer lifting)

AL, BE, GA, DE = -1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971


def _nxt(a):
    return np.concatenate([a[..., 1:], a[..., -1:]], axis=-1)


def _prv(a):
    return np.concatenate([a[..., :1], a[..., :-1]], axis=-1)


def _r(x):
    return np.floor(x + 0.5).astype(np.int64)


def lift_fwd(x, w):
    e, o = x[..., 0::2].copy(), x[..., 1::2].copy()
    if w == "haar":
        d = o - e
        return e + (d >> 1), d
    if w == "53":
        d = o - ((e + _nxt(e)) >> 1)
        return e + ((_prv(d) + d + 2) >> 2), d
    d = o + _r(AL * (e + _nxt(e)))
    s = e + _r(BE * (_prv(d) + d))
    d = d + _r(GA * (s + _nxt(s)))
    s = s + _r(DE * (_prv(d) + d))
    return s, d


def lift_inv(s, d, w):
    if w == "haar":
        e = s - (d >> 1)
        o = d + e
    elif w == "53":
        e = s - ((_prv(d) + d + 2) >> 2)
        o = d + ((e + _nxt(e)) >> 1)
    else:
        s = s - _r(DE * (_prv(d) + d))
        d = d - _r(GA * (s + _nxt(s)))
        e = s - _r(BE * (_prv(d) + d))
        o = d - _r(AL * (e + _nxt(e)))
    x = np.empty(s.shape[:-1] + (s.shape[-1] + d.shape[-1],), dtype=np.int64)
    x[..., 0::2], x[..., 1::2] = e, o
    return x


def dwt1(x, w, lev):
    """multi-level along the last axis; output in Mallat order [s_L, d_L, ..., d_1] and band ids."""
    parts, bands = [], []
    s = x
    for l in range(1, lev + 1):
        s, d = lift_fwd(s, w)
        parts.append(d)
        bands.append(np.full(d.shape[-1], l))
    out = np.concatenate([s] + parts[::-1], axis=-1)
    band = np.concatenate([np.full(s.shape[-1], 0)] + bands[::-1])
    return out, band


def idwt1(c, w, lev):
    n = c.shape[-1]
    sizes = [n >> l for l in range(1, lev + 1)]                 # d_1 .. d_L sizes
    s = c[..., :sizes[-1]]
    pos = sizes[-1]
    for l in range(lev, 0, -1):
        d = c[..., pos:pos + sizes[l - 1]]
        pos += sizes[l - 1]
        s = lift_inv(s, d, w)
    return s


def dwt2(A, w, lev):
    out = A.copy()
    band = np.zeros(A.shape, dtype=np.int64)
    h, wd = A.shape
    for l in range(1, lev + 1):
        sub = out[:h, :wd]
        s, d = lift_fwd(sub, w)
        sub = np.concatenate([s, d], axis=1)
        s2, d2 = lift_fwd(sub.T, w)
        sub = np.concatenate([s2, d2], axis=1).T
        out[:h, :wd] = sub
        band[:h, :wd] = 0
        hh, ww = h // 2, wd // 2
        band[:hh, ww:wd] = 3 * l - 2
        band[hh:h, :ww] = 3 * l - 1
        band[hh:h, ww:wd] = 3 * l
        h, wd = hh, ww
    return out, band


def idwt2(Cf, w, lev):
    out = Cf.copy()
    H, W = Cf.shape
    for l in range(lev, 0, -1):
        h, wd = H >> (l - 1), W >> (l - 1)
        sub = out[:h, :wd].T
        sub = lift_inv(sub[:, :h // 2], sub[:, h // 2:], w).T
        sub = lift_inv(sub[:, :wd // 2], sub[:, wd // 2:], w)
        out[:h, :wd] = sub
    return out


def A4_dwt(m, v):
    """v: wavelet (haar / 53 / 97), dir ('inblock' rows in 32-blocks of L; 'row' / 'col' full lines of X; '2d' X), lev."""
    w, lev, dr = v["wavelet"], v["lev"], v["dir"]
    if dr == "2d":
        A = m.X
        Cf, band = dwt2(A, w, lev)
        rec = idwt2(Cf, w, lev)
        lay = Cf
        ok = m.check_codes(X=rec)
    elif dr == "inblock":
        A = m.L
        R, W = A.shape
        G = A.reshape(R, W // 32, 32)
        Cf, b = dwt1(G, w, lev)
        rec = idwt1(Cf, w, lev).reshape(R, W)
        band = np.broadcast_to(b, Cf.shape)
        lay = band_layout(Cf)
        ok = m.check_codes(L=rec)
    else:
        A = oriented(m.X, dr)
        Cf, b = dwt1(A, w, lev)
        rec = back(idwt1(Cf, w, lev), dr)
        band = np.broadcast_to(b, Cf.shape)
        lay = back(Cf, dr)
        ok = m.check_codes(X=rec)
    info = T.coding_gain(Cf, band)
    return {"streams": [T.Stream(lay, "int", "2d", name="coef")], "roundtrip": ok, "info": info}


# ------------------------------------------------------------------ B1 Lorenzo / column difference

def B1_lorenzo(m, v):
    A = m.L if v["kind"] == "row_inblock" else m.X
    Z = np.zeros((A.shape[0] + 1, A.shape[1] + 1), dtype=np.int64)
    Z[1:, 1:] = A
    if v["kind"] == "col":
        e = A - Z[:-1, 1:]
        rec = np.cumsum(e, axis=0)
    elif v["kind"] == "row_inblock":
        R, W = A.shape
        G = A.reshape(R, W // 32, 32)
        e = G.copy()
        e[..., 1:] = G[..., 1:] - G[..., :-1]
        rec = np.cumsum(e, axis=-1).reshape(R, W)
        e = e.reshape(R, W)
    else:                                   # 2-D Lorenzo: left + above - above-left
        e = A - (Z[1:, :-1] + Z[:-1, 1:] - Z[:-1, :-1])
        rec = np.cumsum(np.cumsum(e, axis=0), axis=1)
    ok = m.check_codes(L=rec) if v["kind"] == "row_inblock" else m.check_codes(X=rec)
    return {"streams": [T.Stream(e, "int", "2d", name="residual")], "roundtrip": ok, "info": {}}


# ------------------------------------------------------------------ B2 XOR (code view, 4-bit index with -0 merged)

def unmerge(sym):
    return np.where(sym >= 8, sym + 1, sym)


def B2_xor(m, v):
    c4 = unmerge(m.sym)
    if v["ref"] == "partner":
        if m.partner is None:
            return {"streams": [T.Stream(m.sym, "sym", "2d", 15, "sym")], "roundtrip": True,
                    "info": {"not_applicable": "w3 has no partner (decoded first)"}}
        ref = unmerge(m.partner)
    elif v["ref"] == "left":
        ref = np.zeros_like(c4)
        ref[:, 1:] = c4[:, :-1]
    else:
        ref = np.zeros_like(c4)
        ref[1:] = c4[:-1]
    x = c4 ^ ref
    # decoder: left / above references are decoded before the current code
    if v["ref"] == "left":
        rec = x.copy()
        for c in range(1, rec.shape[1]):
            rec[:, c] ^= rec[:, c - 1]
    elif v["ref"] == "up":
        rec = x.copy()
        for r in range(1, rec.shape[0]):
            rec[r] ^= rec[r - 1]
    else:
        rec = x ^ ref
    ok = m.check_codes(sym=T.merge(rec))
    return {"streams": [T.Stream(x, "sym", "2d", 16, "xor")], "roundtrip": ok, "info": {}}


# ------------------------------------------------------------------ B3 per-block regression

def B3_regress(m, v):
    """per 32-block polynomial (order 1 or 2, centred position) by least squares; coefficients quantised with
    step s * 16^-k (k = order of the term) and sent as integer streams (side); prediction rounded."""
    A = m.L
    R, W = A.shape
    G = A.reshape(R, W // 32, 32).astype(np.float64)
    t = np.arange(32) - 15.5
    order, s = v["order"], v["s"]
    V = np.stack([t ** k for k in range(order + 1)], axis=1)          # (32, order+1)
    beta = G @ np.linalg.pinv(V).T                                      # (R, g, order+1)
    steps = np.array([s * 16.0 ** (-k) for k in range(order + 1)])
    Qc = np.round(beta / steps).astype(np.int64)
    pred = np.round((Qc * steps) @ V.T).astype(np.int64)
    res = A.reshape(R, W // 32, 32) - pred
    rec = (np.round((Qc * steps) @ V.T).astype(np.int64) + res).reshape(R, W)
    streams = [T.Stream(res.reshape(R, W), "int", "2d", name="residual")]
    streams += [T.Stream(Qc[..., k], "int", "2d", name=f"coef{k}", side=True) for k in range(order + 1)]
    return {"streams": streams, "roundtrip": m.check_codes(L=rec), "info": {}}


# ------------------------------------------------------------------ B4 / B5 hierarchical interpolation (SZ3 style)

def _pred1(x, i, s, n, kind):
    """prediction of positions i (odd multiples of s) from the known grid at multiples of 2s, along the last axis."""
    l = x[..., i - s]
    has_r = i + s < n
    r = x[..., np.minimum(i + s, n - 1)]
    lin = np.where(has_r, (l + r) >> 1, l)
    if kind == "linear":
        return lin
    has_c = (i - 3 * s >= 0) & (i + 3 * s < n)
    ll = x[..., np.maximum(i - 3 * s, 0)]
    rr = x[..., np.minimum(i + 3 * s, n - 1)]
    cub = _r((-ll + 9 * l + 9 * r - rr) / 16.0)
    return np.where(has_c & has_r, cub, lin)


def interp1_fwd(x, S, kind):
    """x (..., n): anchors at multiples of S predicted from the previous anchor; then strides S/2 .. 1."""
    n = x.shape[-1]
    e = np.zeros_like(x)
    a = np.arange(0, n, S)
    e[..., a] = x[..., a] - np.concatenate([np.zeros(x.shape[:-1] + (1,), dtype=np.int64), x[..., a[:-1]]], axis=-1)
    s = S // 2
    while s >= 1:
        i = np.arange(s, n, 2 * s)
        e[..., i] = x[..., i] - _pred1(x, i, s, n, kind)
        s //= 2
    return e


def interp1_inv(e, S, kind):
    n = e.shape[-1]
    x = np.zeros_like(e)
    a = np.arange(0, n, S)
    x[..., a] = np.cumsum(e[..., a], axis=-1)
    s = S // 2
    while s >= 1:
        i = np.arange(s, n, 2 * s)
        x[..., i] = e[..., i] + _pred1(x, i, s, n, kind)
        s //= 2
    return x


def B4_interp(m, v):
    """v: kind (linear / cubic), dir (row / col on aligned values; inblock on levels, anchor = block start)."""
    kind, dr = v["kind"], v["dir"]
    if dr == "inblock":
        A = m.L
        R, W = A.shape
        G = A.reshape(R * (W // 32), 32)
        e = interp1_fwd(G, 32, kind)
        rec = interp1_inv(e, 32, kind).reshape(R, W)
        return {"streams": [T.Stream(e.reshape(R, W), "int", "2d", name="residual")],
                "roundtrip": m.check_codes(L=rec), "info": {}}
    A = oriented(m.X, dr)
    e = interp1_fwd(A, v.get("S", 64), kind)
    rec = back(interp1_inv(e, v.get("S", 64), kind), dr)
    return {"streams": [T.Stream(back(e, dr), "int", "2d", name="residual")], "roundtrip": m.check_codes(X=rec),
            "info": {}}


def interp2_fwd(X, S, kind, inverse=False):
    """2-D SZ3-style: anchors (multiples of S in both axes) by 2-D Lorenzo on the anchor grid; then for each stride
    s = S/2 .. 1: along axis 0 on columns that are multiples of 2s, then along axis 1 on rows multiple of s."""
    H, W = X.shape
    src = X.copy()
    out = np.zeros_like(X)
    ar, ac = np.arange(0, H, S), np.arange(0, W, S)
    if not inverse:
        g = X[np.ix_(ar, ac)]
        Z = np.zeros((g.shape[0] + 1, g.shape[1] + 1), dtype=np.int64)
        Z[1:, 1:] = g
        out[np.ix_(ar, ac)] = g - (Z[1:, :-1] + Z[:-1, 1:] - Z[:-1, :-1])
        cur = X
    else:
        cur = np.zeros_like(X)
        cur[np.ix_(ar, ac)] = np.cumsum(np.cumsum(src[np.ix_(ar, ac)], axis=0), axis=1)
    s = S // 2
    while s >= 1:
        rows = np.arange(s, H, 2 * s)
        cols = np.arange(0, W, 2 * s)
        sub = cur[:, cols].T                                      # (cols, H): along axis 0
        p = _pred1(sub, rows, s, H, kind).T                       # (rows, cols)
        if not inverse:
            out[np.ix_(rows, cols)] = cur[np.ix_(rows, cols)] - p
        else:
            cur[np.ix_(rows, cols)] = src[np.ix_(rows, cols)] + p
        rows2 = np.arange(0, H, s)
        cols2 = np.arange(s, W, 2 * s)
        sub = cur[rows2]
        p = _pred1(sub, cols2, s, W, kind)
        if not inverse:
            out[np.ix_(rows2, cols2)] = cur[np.ix_(rows2, cols2)] - p
        else:
            cur[np.ix_(rows2, cols2)] = src[np.ix_(rows2, cols2)] + p
        s //= 2
    return cur if inverse else out


def B5_multilevel(m, v):
    e = interp2_fwd(m.X, v.get("S", 64), v["kind"])
    rec = interp2_fwd(e, v.get("S", 64), v["kind"], inverse=True)
    return {"streams": [T.Stream(e, "int", "2d", name="residual")], "roundtrip": m.check_codes(X=rec), "info": {}}


# ------------------------------------------------------------------ C1 transposition / C2 space-filling curves (orderings)

def order_index(R, W, kind, unit):
    """permutation (flat indices into R x W, row-major) giving the traversal order."""
    if kind == "row":
        return np.arange(R * W)
    if kind == "col":
        return np.arange(R * W).reshape(R, W).T.ravel()
    if kind == "blockcol":                          # blocks column by column, codes inside a block in order
        g = W // 32
        idx = np.arange(R * W).reshape(R, g, 32)
        return idx.transpose(1, 0, 2).ravel()
    gh, gw = (R, W // 32) if unit == "block" else (R, W)
    yy, xx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    key = (zorder(yy, xx) if kind == "z" else hilbert(yy, xx, int(math.ceil(math.log2(max(gh, gw)))))).ravel()
    cells = np.argsort(key, kind="stable")
    if unit == "block":
        idx = np.arange(R * W).reshape(R, W // 32, 32).reshape(gh * gw, 32)
        return idx[cells].ravel()
    return cells


def zorder(y, x):
    k = np.zeros(y.shape, dtype=np.int64)
    for b in range(14):
        k |= ((x >> b) & 1) << (2 * b)
        k |= ((y >> b) & 1) << (2 * b + 1)
    return k


def hilbert(y, x, order):
    """Hilbert index on a 2^order grid (vectorised xy2d)."""
    x = x.astype(np.int64).copy()
    y = y.astype(np.int64).copy()
    d = np.zeros(x.shape, dtype=np.int64)
    s = 1 << (order - 1)
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        n = (1 << order) - 1
        sw = ry == 0
        fl = sw & (rx == 1)
        x = np.where(fl, n - x, x)
        y = np.where(fl, n - y, y)
        x, y = np.where(sw, y, x), np.where(sw, x, y)
        s >>= 1
    return d


def C_order(m, v):
    """C1 (transposition, block-column order) and C2 (z-order / Hilbert): reorder, then code as a 1-D sequence
    (C1 context = previous two values). v: kind, unit, data ('code' symbols / 'value' block levels)."""
    R, W = m.sym.shape if v["data"] == "code" else m.L.shape
    perm = order_index(R, W, v["kind"], v.get("unit", "code"))
    src = m.sym if v["data"] == "code" else m.L
    seq = src.ravel()[perm]
    inv = np.empty_like(seq)
    inv[perm] = seq
    if v["data"] == "code":
        ok = m.check_codes(sym=inv.reshape(R, W))
        st = T.Stream(seq, "sym", "seq", 15, "seq")
    else:
        ok = m.check_codes(L=inv.reshape(R, W))
        st = T.Stream(seq, "int", "seq", name="seq")
    return {"streams": [st], "roundtrip": ok, "info": {}}


# ------------------------------------------------------------------ D1 field / bit-plane splitting

def _field_ctx_stream(F, mask, K, name, none):
    """stream of field F at positions mask, C1 context = (left, above) values of the same field (none state where
    the neighbour has no value of this field). Implemented as a symbol stream with a custom context."""
    return FieldStream(F, mask, K, name, none)


class FieldStream(T.Stream):
    def __init__(self, F, mask, K, name, none):
        super().__init__(F[mask] if mask is not None else F.ravel(), "sym", "2d", K, name)
        self.F, self.mask, self.none = F, mask, none

    def bits(self, coder):
        if coder == "C0":
            return T.dirichlet(self.arr, None, self.K)
        G = np.where(self.mask, self.F, self.none) if self.mask is not None else self.F
        a, b = T.neighbours(G, "2d", self.none)
        sel = self.mask.ravel() if self.mask is not None else slice(None)
        return T.dirichlet(self.arr, a[sel] * (self.none + 1) + b[sel], self.K)


def D1_split(m, v):
    c4 = unmerge(m.sym)
    mag = c4 & 7
    nz = mag > 0
    sign = (c4 >> 3) & 1
    Z = (~nz).astype(np.int64)
    if v["kind"] == "scale_planes":
        E = m.E.astype(np.int64)
        st = [FieldStream((E >> b) & 1, None, 2, f"scale_bit{b}", 2) for b in range(8)]
        rec = sum(((E >> b) & 1) << b for b in range(8))
        return {"streams": st, "roundtrip": bool(np.array_equal(rec, E)), "info": {"replaces": "scale"}}
    streams = [FieldStream(Z, None, 2, "zero", 2), FieldStream(sign, nz, 2, "sign", 2)]
    if v["kind"] == "fields":
        streams.append(FieldStream(mag - 1, nz, 7, "mag", 7))
        rec_mag = np.where(nz, mag, 0)
    else:
        mm = mag - 1
        for b in (2, 1, 0):
            streams.append(FieldStream((mm >> b) & 1, nz, 2, f"mag_bit{b}", 2))
        rec_mag = np.where(nz, sum(((mm >> b) & 1) << b for b in range(3)) + 1, 0)
    rec = np.where(nz, rec_mag + 8 * sign, 0)
    return {"streams": streams, "roundtrip": m.check_codes(sym=T.merge(rec)), "info": {}}


# ------------------------------------------------------------------ variant grids (stage 1: confirm small, few parameters)

QS = [0.5, 1, 2, 4, 8]


def variants():
    V = {}
    V["A1_wht"] = [dict(dom="L", axis="row", N=N, rsign=rs) for N in (4, 8, 16, 32) for rs in (False, True)] + \
                  [dict(dom="X", axis="row", N=N, rsign=rs) for N in (64, 256) for rs in (False, True)] + \
                  [dict(dom="X", axis="col", N=N, rsign=rs) for N in (8, 32) for rs in (False, True)]
    V["A2_dct"] = [dict(dom=d, axis=a, N=N, q=q) for (d, a, N) in (("L", "row", 8), ("L", "row", 32), ("X", "row", 64),
                                                                   ("X", "row", 256), ("X", "col", 32)) for q in QS]
    V["A3_dft"] = [dict(dom=d, axis=a, N=N, q=q) for (d, a, N) in (("X", "row", "full"), ("X", "col", "full"), ("L", "row", 32))
                   for q in QS]
    V["A4_dwt"] = [dict(wavelet=w, dir=d, lev=l) for w in ("haar", "53", "97") for d in ("inblock", "row", "col", "2d")
                   for l in (1, 2, 3, 4, 5)]
    V["A5_lapped"] = [dict(M=M, q=q) for M in (4, 8) for q in (1, 2, 4, 8)]
    V["B1_lorenzo"] = [dict(kind=k) for k in ("col", "2d", "row_inblock")]
    V["B2_xor"] = [dict(ref=r) for r in ("left", "up", "partner")]
    V["B3_regress"] = [dict(order=o, s=s) for o in (1, 2) for s in (0.5, 1, 2, 4)]
    V["B4_interp"] = [dict(kind=k, dir=d) for k in ("linear", "cubic") for d in ("row", "col", "inblock")]
    V["B5_multilevel"] = [dict(kind=k) for k in ("linear", "cubic")]
    V["C_order"] = [dict(kind=k, unit="code", data=dd) for k in ("row", "col", "blockcol") for dd in ("code", "value")] + \
                   [dict(kind=k, unit=u, data=dd) for k in ("z", "hilbert") for u in ("block", "code") for dd in ("code", "value")]
    V["D1_split"] = [dict(kind=k) for k in ("fields", "planes", "scale_planes")]
    return V


VIEW = {"A1_wht": "value", "A2_dct": "value", "A3_dft": "value", "A4_dwt": "value", "A5_lapped": "value",
        "B1_lorenzo": "value", "B2_xor": "code", "B3_regress": "value", "B4_interp": "value", "B5_multilevel": "value",
        "C_order": "code", "D1_split": "code"}


def base_view(method, v):
    """the raw view the transform starts from (for the appearance)."""
    if method in ("B2_xor", "D1_split"):
        return "code"
    if method == "C_order":
        return "code_seq" if v["data"] == "code" else "level_seq"
    if method == "A4_dwt":
        return "level" if v["dir"] == "inblock" else "aligned"
    if method == "B1_lorenzo":
        return "level" if v["kind"] == "row_inblock" else "aligned"
    if method == "B4_interp":
        return "level" if v["dir"] == "inblock" else "aligned"
    if method in ("B3_regress",):
        return "level"
    if method in ("B5_multilevel", "A5_lapped"):
        return "aligned"
    return "level" if v.get("dom", "L") == "L" else "aligned"


def _g(*keys):
    return lambda v: tuple(v.get(k) for k in keys)


# groups for the per-method tables: the best variant (on train) of every group is also run on T2
GROUP_KEYS = {"A1_wht": ("dom", "axis"), "A2_dct": ("dom", "axis", "N"), "A3_dft": ("axis", "N"), "A4_dwt": ("wavelet", "dir"),
              "A5_lapped": ("M",), "B1_lorenzo": ("kind",), "B2_xor": ("ref",), "B3_regress": ("order",),
              "B4_interp": ("kind", "dir"), "B5_multilevel": ("kind",), "C_order": ("data", "kind", "unit"),
              "D1_split": ("kind",)}


def exclude_from_best(method, v):
    """D1 scale planes code only the scale stream: compared with the scale stream, not with the code variants."""
    return method == "D1_split" and v.get("kind") == "scale_planes"
