"""TX stage 2: global transforms (A6 KLT across experts, A7 ICA, A8 Tucker, C3 seriation, C4 BWT + MTF).

Per-matrix methods (main sample, txcore.eval_job): A7_ica (single matrix), C3_seriation, C4_bwt.
Group methods (cross-expert sample, group_job): A6_klt, A7_ica_shared, A8_tucker, C3_expert_order.
Cross-expert / cross-layer methods do not follow the repository's "code every target independently" rule:
they are measurements of how much information is shared, not deployable codecs (as 011 line C).
"""
import ctypes
import math
import os
import time

import numpy as np

import txcore as T
import tx_local as X

LIB = None


def lib():
    global LIB
    if LIB is None:
        LIB = ctypes.CDLL(str(T.DATA / "bin" / "libbwt.so"))
        i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        LIB.ibwt.argtypes = [i32p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, i32p]
        LIB.mtf.argtypes = [i32p, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, i32p]
    return LIB


# ------------------------------------------------------------------ helpers: approximation + residual, units

def q16(a):
    return np.asarray(a, dtype=np.float16).astype(np.float64)


def units(ms):
    """common real units for several matrices: aligned values times 2^(emin - smallest emin)."""
    e0 = min(int(m.E.min()) for m in ms)
    return [2.0 ** (int(m.E.min()) - e0) for m in ms]


def hidden_lines(m):
    """vectors along the hidden dimension (residual stream): rows of w1 / w3, columns of w2 (shape 2304 x 5120)."""
    return m.X if m.X.shape[1] == 5120 else np.ascontiguousarray(m.X.T)


def from_hidden(A, m):
    return A if m.X.shape[1] == 5120 else np.ascontiguousarray(A.T)


# ------------------------------------------------------------------ A6 KLT / PCA across experts (group)

def A6_klt(ms, v):
    """v: kind 'expert_axis' (KLT along the expert axis at every position; aligned = neurons matched to the first
    expert beforehand, permutation billed) or 'subspace' (shared top-k subspace of the hidden-dim lines), q, k."""
    n = len(ms)
    u = units(ms)
    side_each = [0.0] * n
    if v["kind"] == "expert_axis":
        Xs = [m.X for m in ms]
        if v.get("aligned"):
            Xs, side_each = align_group(ms)
        Y = np.stack([x.astype(np.float64).ravel() * s for x, s in zip(Xs, u)])          # (n, positions)
        Cv = Y @ Y.T / Y.shape[1]
        ev, U = np.linalg.eigh(Cv)
        U = q16(U[:, ::-1])
        Uinv = np.linalg.inv(U)
        Q = np.round((U.T @ Y) / v["q"]).astype(np.int64)
        del Y
        Yh = Uinv.T @ (Q * v["q"])
        for e, m in enumerate(ms):
            xh = np.round(Yh[e] / u[e]).astype(np.int64).reshape(m.X.shape)
            res = Xs[e] - xh
            rec = np.round((Uinv[:, e] @ (Q * v["q"])) / u[e]).astype(np.int64).reshape(m.X.shape) + res
            if v.get("aligned"):                   # back to this expert's own layout (keeps its 32-blocks intact)
                rec = unalign(rec, e, ms)
                res = unalign(res, e, ms)
            yield ({"streams": [T.Stream(Q[e].reshape(m.X.shape), "int", "2d", name=f"coef{e}"),
                                    T.Stream(res, "int", "2d", name="residual")],
                        "side": side_each[e] + 16.0 * n * n / n, "roundtrip": m.check_codes(X=rec),
                        "info": {"energy_share_first": float(ev[::-1][0] / ev.sum())}})
        return
    k, q = v["k"], v["q"]
    H = [hidden_lines(m).astype(np.float64) * s for m, s in zip(ms, u)]
    Cv = sum(h.T @ h for h in H)
    ev, V = np.linalg.eigh(Cv)
    Uk = q16(V[:, ::-1][:, :k])
    for e, m in enumerate(ms):
        Xh = hidden_lines(m)
        c = (Xh.astype(np.float64)) @ Uk
        Q = np.round(c / q).astype(np.int64)
        xh = np.round((Q * q) @ Uk.T).astype(np.int64)
        res = Xh - xh
        rec = from_hidden(np.round((Q * q) @ Uk.T).astype(np.int64) + res, m)
        yield ({"streams": [T.Stream(Q, "int", "2d", name="coef"), T.Stream(from_hidden(res, m), "int", "2d", name="residual")],
                    "side": 16.0 * Uk.size / n, "roundtrip": m.check_codes(X=rec),
                    "info": {"energy_share_k": float(ev[::-1][:k].sum() / ev.sum())}})


def neuron_vectors(group3):
    """per expert: (2304, 2 * 5120) w1 and w3 rows (dequantised scale-free aligned values) for neuron matching."""
    return [np.concatenate([g["w1"].X, g["w3"].X], axis=1).astype(np.float64) for g in group3]


_ALIGN = {}


def align_group(ms):
    """neurons of experts 2..n matched to expert 1 (010 T2 Hungarian matching on w1 / w3 rows); applied to rows of
    w1 / w3 and columns of w2. Cost: log2(2304!) bits per aligned expert (billed to this matrix's share: 1/3)."""
    import tx
    key = tuple(m.name for m in ms)
    if key not in _ALIGN:
        g3 = [{r: T.Mat.load(m.name.rsplit(".", 1)[0] + "." + r, with_partner=False) for r in ("w1", "w3")} for m in ms]
        nv = neuron_vectors(g3)
        perms = [np.arange(2304)]
        for e in range(1, len(ms)):
            cb, _, _ = tx.match_rows(nv[0], nv[e])
            perms.append(cb)                        # row cb[i] of expert e <-> row i of expert 0
        _ALIGN[key] = perms
    perms = _ALIGN[key]
    Xs = []
    for m, p in zip(ms, perms):
        Xs.append(m.X[p] if m.X.shape[0] == 2304 else m.X[:, p])
    cost = T.log2_fact(2304) / 3.0
    return Xs, [0.0] + [cost] * (len(ms) - 1)


def unalign(A, e, ms):
    p = _ALIGN[tuple(m.name for m in ms)][e]
    out = np.empty_like(A)
    if A.shape[0] == 2304:
        out[p] = A
    else:
        out[:, p] = A
    return out


# ------------------------------------------------------------------ A7 ICA (FastICA, symmetric, tanh)

def fastica(Z, iters=200, tol=1e-5, seed=0):
    """Z (samples, d), centred. Returns the unmixing matrix W (d, d): s = x @ W.T."""
    n, d = Z.shape
    C = Z.T @ Z / n
    ev, E = np.linalg.eigh(C)
    ev = np.maximum(ev, 1e-9 * ev.max())
    K = E / np.sqrt(ev)                                   # whitening: z_w = z @ K
    Zw = Z @ K
    rng = np.random.default_rng(seed)
    W, _ = np.linalg.qr(rng.normal(size=(d, d)))
    for it in range(iters):
        G = np.tanh(Zw @ W.T)
        Wn = (G.T @ Zw) / n - np.diag((1 - G * G).mean(0)) @ W
        s, u = np.linalg.eigh(Wn @ Wn.T)
        Wn = u @ np.diag(1.0 / np.sqrt(np.maximum(s, 1e-12))) @ u.T @ Wn
        conv = np.max(np.abs(np.abs(np.einsum("ij,ij->i", Wn, W)) - 1))
        W = Wn
        if conv < tol:
            break
    return W @ K.T, it + 1


def ica_vectors(m, unit):
    """unit 'block': 32-blocks of block levels L; 'seg': 256-segments of aligned rows X."""
    if unit == "block":
        return m.L.reshape(-1, 32), "L"
    return m.X.reshape(-1, 256), "X"


def ica_code(m, unit, Wu, q):
    V, dom = ica_vectors(m, unit)
    A = np.linalg.inv(Wu)
    S = V.astype(np.float64) @ Wu.T
    Q = np.round(S / q).astype(np.int64)
    vh = np.round((Q * q) @ A.T).astype(np.int64)
    res = V - vh
    rec = (np.round((Q * q) @ A.T).astype(np.int64) + res).reshape(m.X.shape)
    ok = m.check_codes(L=rec) if dom == "L" else m.check_codes(X=rec)
    R = m.X.shape[0]
    d = V.shape[1]
    lay = X.band_layout(Q.reshape(R, -1, d))
    info = T.coding_gain(S, np.broadcast_to(np.arange(d), S.shape))
    return [T.Stream(lay, "int", "2d", name="sources"), T.Stream(res.reshape(m.X.shape), "int", "2d", name="residual")], ok, info


def A7_ica(m, v):
    """single matrix: unmixing estimated on this matrix (200k vectors at most), sent as fp16 (side). The estimate
    does not depend on q, so it is cached per matrix and unit (data/010tx/ica/)."""
    cache = T.DATA / "ica" / f"{m.name}_{v['unit']}.npz" if m.name else None
    if cache is not None and cache.exists():
        z = np.load(cache)
        Wu, it = z["W"], int(z["it"])
    else:
        V, _ = ica_vectors(m, v["unit"])
        rng = np.random.default_rng(0)
        smp = V[rng.choice(len(V), min(len(V), 200000), replace=False)].astype(np.float64)
        Wu, it = fastica(smp - smp.mean(0))
        Wu = q16(Wu)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".{os.getpid()}.npz")
            np.savez(tmp, W=Wu, it=it)
            os.replace(tmp, cache)
    st, ok, info = ica_code(m, v["unit"], Wu, v["q"])
    info["ica_iters"] = it
    return {"streams": st, "side": 16.0 * Wu.size, "roundtrip": ok, "info": info}


_ICA_SHARED = {}


def A7_ica_shared(ms, v):
    key = (tuple(m.name for m in ms), v["unit"])
    if any(m.name is None for m in ms):            # synthetic matrices have no name: never cache
        _ICA_SHARED.pop(key, None)
    if key not in _ICA_SHARED:
        rng = np.random.default_rng(0)
        smp = []
        for m in ms:
            V, _ = ica_vectors(m, v["unit"])
            smp.append(V[rng.choice(len(V), min(len(V), 200000 // len(ms)), replace=False)])
        smp = np.concatenate(smp).astype(np.float64)
        Wu, it = fastica(smp - smp.mean(0))
        _ICA_SHARED[key] = (q16(Wu), it)
    Wu, it = _ICA_SHARED[key]
    for m in ms:
        st, ok, info = ica_code(m, v["unit"], Wu, v["q"])
        info["ica_iters"] = it
        yield ({"streams": st, "side": 16.0 * Wu.size / len(ms), "roundtrip": ok, "info": info})


# ------------------------------------------------------------------ A8 Tucker / HOSVD (group)

def A8_tucker(ms, v):
    """tensor (experts or layers) x rows x cols; truncated HOSVD with ranks (r0, r1, r2); factors and core sent as
    fp16 (side, shared by the group); approximation rounded, residual exact."""
    u = units(ms)
    Xs = [m.X.astype(np.float64) * s for m, s in zip(ms, u)]
    sc = float(np.float32(math.sqrt(np.mean([np.mean(x * x) for x in Xs]))))   # sent as one float32 (32 bit)
    Xs = [x / sc for x in Xs]
    u = [s / sc for s in u]                                                      # approx (normalised) / u -> own units
    r0, r1, r2 = v["ranks"]
    n = len(ms)
    G0 = np.array([[np.vdot(a, b) for b in Xs] for a in Xs])
    U0 = q16(np.linalg.eigh(G0)[1][:, ::-1][:, :min(r0, n)])
    C1 = sum(x @ x.T for x in Xs)
    U1 = q16(np.linalg.eigh(C1)[1][:, ::-1][:, :r1])
    C2 = sum(x.T @ x for x in Xs)
    U2 = q16(np.linalg.eigh(C2)[1][:, ::-1][:, :r2])
    Y = np.stack([U1.T @ x @ U2 for x in Xs])                        # (n, r1, r2)
    core = q16(np.einsum("ea,ebc->abc", U0, Y))
    side = (16.0 * (U0.size + U1.size + U2.size + core.size) + 32.0) / n
    for e, m in enumerate(ms):
        approx = U1 @ np.einsum("a,abc->bc", U0[e], core) @ U2.T
        xh = np.round(approx / u[e]).astype(np.int64)
        res = m.X - xh
        rec = np.round((U1 @ np.einsum("a,abc->bc", U0[e], core) @ U2.T) / u[e]).astype(np.int64) + res
        rel = float(((m.X - approx / u[e]) ** 2).sum() / max((m.X.astype(np.float64) ** 2).sum(), 1e-30))
        yield ({"streams": [T.Stream(res, "int", "2d", name="residual")], "side": side,
                    "roundtrip": m.check_codes(X=rec), "info": {"relative_error": rel}})


# ------------------------------------------------------------------ C3 seriation (rows of one matrix)

def row_corr_matrix(A):
    Z = A.astype(np.float64)
    Z = Z - Z.mean(1, keepdims=True)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-30)
    return Z @ Z.T


def order_spectral(Cm):
    S = np.abs(Cm)
    np.fill_diagonal(S, 0)
    L = np.diag(S.sum(1)) - S
    ev, V = np.linalg.eigh(L)
    return np.argsort(V[:, 1], kind="stable")


def order_tsp(Cm, passes=30, seed=0):
    """nearest-neighbour path on distance 1 - |corr|, improved by 2-opt (best move per start point, vectorised)."""
    D = 1.0 - np.abs(Cm)
    n = len(D)
    seen = np.zeros(n, dtype=bool)
    path = [0]
    seen[0] = True
    for _ in range(n - 1):
        d = np.where(seen, np.inf, D[path[-1]])
        j = int(np.argmin(d))
        path.append(j)
        seen[j] = True
    p = np.array(path)
    for _ in range(passes):
        improved = False
        for i in range(n - 2):
            a, b = p[i], p[i + 1]
            c, d = p[i + 2:], np.append(p[i + 3:], -1)
            dd = np.where(d >= 0, D[b, np.maximum(d, 0)], 0.0)
            gain = D[a, b] + np.where(d >= 0, D[c, np.maximum(d, 0)], 0.0) - D[a, c] - dd
            j = int(np.argmax(gain))
            if gain[j] > 1e-12:
                p[i + 1:i + 3 + j] = p[i + 1:i + 3 + j][::-1]
                improved = True
        if not improved:
            break
    return p


def order_hclust(A):
    from scipy.cluster.hierarchy import leaves_list, linkage
    return leaves_list(linkage(A.astype(np.float64), method="average", metric="correlation"))


def C3_seriation(m, v):
    """v: order (spectral / tsp / hclust / random / none), coding ('code' = code view C1 2-D, 'b1' = column
    difference of X, 'a4' = CDF 5/3 3 levels along columns of X). Permutation billed as log2(R!) bits."""
    R = m.X.shape[0]
    if v["order"] == "none":
        p = np.arange(R)
    elif v["order"] == "random":
        p = np.random.default_rng(0).permutation(R)
    elif v["order"] == "hclust":
        p = order_hclust(m.X)
    else:
        Cm = row_corr_matrix(m.X)
        p = order_spectral(Cm) if v["order"] == "spectral" else order_tsp(Cm)
    side = 0.0 if v["order"] == "none" else T.log2_fact(R)
    sub = T.Mat(L=None, C=m.C[p], E=m.E[p], name=m.name) if m.C is not None else None
    inv = np.empty_like(p)
    inv[p] = np.arange(R)
    if v["coding"] == "code":
        st = [T.Stream(sub.sym, "sym", "2d", 15, "sym")]
        ok = m.check_codes(sym=sub.sym[inv])
    elif v["coding"] == "b1":
        enc = X.B1_lorenzo(sub, {"kind": "col"})
        st, ok = enc["streams"], enc["roundtrip"] and m.check_codes(X=sub.X[inv])
    else:
        enc = X.A4_dwt(sub, {"wavelet": "53", "dir": "col", "lev": 3})
        st, ok = enc["streams"], enc["roundtrip"] and m.check_codes(X=sub.X[inv])
    return {"streams": st, "side": side, "roundtrip": ok, "info": {}}


# ------------------------------------------------------------------ C3 expert order (group): previous expert as context

class CrossStream(T.Stream):
    """symbols of one expert; C1 context = (left symbol, symbol at the same position of the previous expert)."""

    def __init__(self, sym, prev):
        super().__init__(sym, "sym", "2d", 15, "sym")
        self.prev = prev

    def bits(self, coder):
        if coder == "C0" or self.prev is None:
            return T.sym_bits(self.arr, 15, coder, "2d")
        a, _ = T.neighbours(self.arr, "2d", 15)
        return T.dirichlet(self.arr, a * 16 + self.prev.ravel(), 15)


def C3_expert_order(ms, v):
    """v: order ('index' / 'similar': greedy nearest-neighbour path over expert distance 1 - |corr| of the levels),
    aligned (neurons of every expert matched to expert 1; permutations billed). Every expert is coded in its own
    layout; with alignment, the context is the previous expert's code at the matched neuron."""
    n = len(ms)
    side = [0.0] * n
    perms = None
    if v["aligned"]:
        _, side = align_group(ms)
        perms = _ALIGN[tuple(m.name for m in ms)]

    def to_frame0(sym, e):                         # expert e's codes in expert 1's neuron order
        if perms is None:
            return sym
        p = perms[e]
        return sym[p] if sym.shape[0] == 2304 else sym[:, p]

    def from_frame0(A, e):
        if perms is None:
            return A
        return unalign(A, e, ms)

    F0 = [to_frame0(m.sym, e) for e, m in enumerate(ms)]
    order = list(range(n))
    if v["order"] == "similar":
        F = np.stack([T.levels(X.unmerge(s)).astype(np.float64).ravel() for s in F0])
        F = F - F.mean(1, keepdims=True)
        F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-30)
        D = 1 - np.abs(F @ F.T)
        order = [0]
        while len(order) < n:
            order.append(min((j for j in range(n) if j not in order), key=lambda j: D[order[-1], j]))
    out = [None] * n
    for i, e in enumerate(order):
        prev = from_frame0(F0[order[i - 1]], e) if i > 0 else None
        out[e] = {"streams": [CrossStream(ms[e].sym, prev)], "side": side[e] + T.log2_fact(n) / n,
                  "roundtrip": bool(np.array_equal(from_frame0(F0[e], e), ms[e].sym)), "info": {"position_in_order": i}}
    return out


# ------------------------------------------------------------------ C4 BWT + MTF

def bwt_rows(S):
    """cyclic BWT of every row of S (rows, n) by prefix doubling; returns last column and primary indices."""
    rows, n = S.shape
    rank = S.astype(np.int64).copy()
    rid = np.repeat(np.arange(rows), n).reshape(rows, n)
    k = 1
    while True:
        r2 = np.roll(rank, -k, axis=1)
        o = np.lexsort((r2.ravel(), rank.ravel(), rid.ravel()))
        a, b, c = rid.ravel()[o], rank.ravel()[o], r2.ravel()[o]
        new = np.concatenate([[True], (a[1:] != a[:-1]) | (b[1:] != b[:-1]) | (c[1:] != c[:-1])])
        nr = np.cumsum(new) - 1
        rank = np.empty(rows * n, dtype=np.int64)
        rank[o] = nr
        rank = rank.reshape(rows, n)
        distinct = new.sum()
        k *= 2
        if distinct == rows * n or k >= n:
            break
    o = o.reshape(rows, n) - (np.arange(rows) * n)[:, None]          # sorted rotation starts per row
    Lc = np.take_along_axis(S, (o - 1) % n, axis=1)
    prim = np.argmax(o == 0, axis=1)
    return Lc, prim


def ibwt_rows(Lc, prim, K):
    out = np.empty_like(Lc, dtype=np.int32)
    L32 = np.ascontiguousarray(Lc, dtype=np.int32)
    for r in range(Lc.shape[0]):
        row = np.empty(Lc.shape[1], dtype=np.int32)
        lib().ibwt(np.ascontiguousarray(L32[r]), Lc.shape[1], int(prim[r]), K, row)
        out[r] = row
    return out


def mtf_fwd(s, K):
    """vectorised MTF index: number of distinct symbols since the previous occurrence of the same symbol (initial
    list 0..K-1 as virtual occurrences at positions -1..-K)."""
    s = np.asarray(s, dtype=np.int64)
    n = len(s)
    last = np.empty((K, n), dtype=np.int64)
    pos = np.arange(n)
    for k in range(K):
        a = np.where(s == k, pos, -(k + 1))
        last[k] = np.maximum.accumulate(a)
    prev_self = np.empty(n, dtype=np.int64)
    prev_self[1:] = last[s[1:], np.arange(n - 1)]
    prev_self[0] = -(s[0] + 1)
    lastm1 = np.concatenate([(-(np.arange(K) + 1))[:, None], last[:, :-1]], axis=1)
    return (lastm1 > prev_self[None, :]).sum(0)


def mtf_inv(idx, K):
    out = np.empty(len(idx), dtype=np.int32)
    lib().mtf(np.ascontiguousarray(idx, dtype=np.int32), len(idx), K, 0, out)
    return out


def C4_bwt(m, v):
    """v: unit ('row' / 'matrix'), out ('bwt' / 'bwt_mtf'). Coded as a 1-D sequence; primary indices billed."""
    S = m.sym
    R, W = S.shape
    if v["unit"] == "row":
        Lc, prim = bwt_rows(S)
        side = R * math.log2(W)
        rec = ibwt_rows(Lc, prim, 15)
    else:
        Lc, prim = bwt_rows(S.reshape(1, -1))
        side = math.log2(S.size)
        rec = ibwt_rows(Lc, prim, 15).reshape(R, W)
    seq = Lc.ravel()
    if v["out"] == "bwt_mtf":
        mt = mtf_fwd(seq, 15)
        assert np.array_equal(mtf_inv(mt, 15), seq)
        seq = mt
    ok = m.check_codes(sym=np.asarray(rec, dtype=np.int64).reshape(R, W))
    return {"streams": [T.Stream(seq, "sym", "seq", 15, "seq")], "side": side, "roundtrip": ok, "info": {}}


# ------------------------------------------------------------------ variants

def variants():
    V = {}
    V["A7_ica"] = [dict(unit=u, q=q) for u in ("block", "seg") for q in (0.25, 0.5, 1, 2)]
    V["C3_seriation"] = [dict(order=o, coding=c) for o in ("none", "random", "spectral", "tsp", "hclust")
                         for c in ("code", "b1", "a4")]
    V["C4_bwt"] = [dict(unit=u, out=o) for u in ("row", "matrix") for o in ("bwt", "bwt_mtf")]
    return V


VIEW = {"A7_ica": "value", "C3_seriation": "mixed", "C4_bwt": "code"}
NEED_PARTNER = ()


def base_view(method, v):
    if method == "A7_ica":
        return "level" if v["unit"] == "block" else "aligned"
    if method == "C4_bwt":
        return "code_seq"
    if method == "C3_seriation":
        return "code" if v["coding"] == "code" else "aligned"
    return "aligned"


GROUP_VARIANTS = {
    "A6_klt": [dict(kind="expert_axis", aligned=a, q=q) for a in (False, True) for q in (1, 2, 4, 8)] +
              [dict(kind="subspace", k=k, q=q) for k in (8, 32, 128, 512) for q in (2, 4, 8)],
    "A7_ica_shared": [dict(unit=u, q=q) for u in ("block", "seg") for q in (0.25, 0.5, 1, 2)],
    "A8_tucker": [dict(ranks=r) for r in ((2, 32, 32), (4, 64, 64), (8, 128, 128), (8, 256, 256), (8, 512, 512))],
    "C3_expert_order": [dict(order=o, aligned=a) for o in ("index", "similar") for a in (False, True)],
}
GROUP_VIEW = {"A6_klt": "aligned", "A7_ica_shared": None, "A8_tucker": "aligned", "C3_expert_order": "code"}


# ------------------------------------------------------------------ group jobs (cross-expert sample)

def cross_sets():
    """{(layer, role): {'xsel': [8 names], 'xrep': [8 names]}} from txcore.cross_experts()."""
    ce = T.cross_experts(16)
    out = {}
    for L, es in ce.items():
        for r in T.ROLES:
            out[(L, r)] = {"xsel": [f"L{L}.E{e}.{r}" for e in es[:8]], "xrep": [f"L{L}.E{e}.{r}" for e in es[8:]]}
    return out


def layer_sets():
    """A8 layer mode: the same expert index in 4 adjacent layers (L .. L+3); 2 experts per base layer (indices < 128,
    so that the MTP layers 40-42 have them), chosen with a fixed seed."""
    out = {}
    for L in T.LAYERS:
        es = [int(e) for e in np.random.default_rng([20261119, 31, L]).permutation(128)[:2]]
        for e in es:
            for r in T.ROLES:
                out[(L, e, r)] = [f"L{L + d}.E{e}.{r}" for d in range(4)]
    return out


def group_job(job):
    """job: gkey, method, variants (list), names, split, nshare. Runs every variant on the group (shared work such as
    covariances is recomputed per variant only where needed) and returns one row per matrix and variant."""
    import tx_run
    t0 = time.time()
    ms = [T.Mat.load(n, with_partner=False) for n in job["names"]]
    fn = globals()[job["method"]]
    rows = []
    for v in job["variants"]:
        t1 = time.time()
        encs = fn(ms, v)
        for m, enc in zip(ms, encs):
            if job.get("report_only") and m.name not in job["report_only"]:
                continue
            e = T.evaluate(enc)
            e.update({"key": f"{job['method']}|{tx_run.vname(v)}|n{job['nshare']}|{m.name}", "gkey": job["gkey"],
                      "name": m.name, "method": job["method"], "variant": v, "vname": tx_run.vname(v),
                      "split": job["split"], "nshare": job["nshare"], "group": job["names"], "n": m.n,
                      "cpu_s": time.time() - t1})
            rows.append(e)
    return rows
