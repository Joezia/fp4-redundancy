"""Spec 004: A neuron correspondence, B shared row structure across experts, C2 round trip,
D context-mixing probe. C1 reuses `ledger.ledger_matrix`.

Every function here is the single code path for both the section-7 synthetic checks and the model
matrices. All measurements are on C' (merged zero); the caller merges before calling. Quantities
are in bits; per-weight values divide by the number of codewords of the target.
"""
import math
import struct
from itertools import permutations

import numpy as np

from . import est
from .measure import E2M1

B = 32
SEED = 20261114
ABSV = np.abs(E2M1)
FAMS = ("a", "g", "o")  # a = w1 rows, g = w3 rows, o = w2 columns
ROLE_OF = {"a": "w1", "g": "w3", "o": "w2"}
A1_PAIRS = (("a", "g"), ("a", "o"), ("g", "o"))
A1_SHIFTS = (1, 7, 100)
A1_Q = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
A3_CONFIGS = [("a", ("g",)), ("a", ("o",)), ("g", ("a",)), ("g", ("o",)), ("o", ("a",)),
              ("o", ("g",)), ("a", ("g", "o")), ("g", ("a", "o")), ("o", ("a", "g"))]
SLOPE_BITS = 16


def rng_for(*parts):
    return np.random.default_rng([SEED, *[int(p) for p in parts]])


def cfg_name(tgt, srcs):
    return f"{tgt}|{','.join(srcs)}"


# ----------------------------------------------------------------------------- families

def family(Cp, E, role):
    """Neuron-major view of one matrix: rows are neurons (w1/w3 rows, w2 columns), columns are the
    5120 positions. `half` is the 001 hold-out split = parity of the *stored matrix's* row."""
    Eb = np.repeat(E, B, axis=1)
    if role == "w2":
        c = np.ascontiguousarray(Cp.T)
        eb = np.ascontiguousarray(Eb.T)
        half = np.broadcast_to((np.arange(c.shape[1]) & 1)[None, :], c.shape)
    else:
        c = np.ascontiguousarray(Cp)
        eb = Eb
        half = np.broadcast_to((np.arange(c.shape[0]) & 1)[:, None], c.shape)
    sc = np.exp2(eb.astype(np.float64) - 127.0)
    uE, ei = np.unique(eb.ravel(), return_inverse=True)
    return {"c": c, "eb": eb, "sc": sc, "w": E2M1[c] * sc, "half": half, "role": role,
            "ei": ei.reshape(c.shape), "K": len(uE)}


def dom_values(f, dom):
    return f["w"] if dom == "w" else E2M1[f["c"]]


def center_unit(X):
    Z = X - X.mean(axis=1, keepdims=True)
    n = np.sqrt((Z * Z).sum(axis=1, keepdims=True))
    return Z / np.where(n > 0, n, 1.0)


# ----------------------------------------------------------------------------- A1, A2

def a1_pair(fx, fy, dom):
    Zx = center_unit(dom_values(fx, dom))
    Zy = center_unit(dom_values(fy, dom))
    rho = (Zx * Zy).sum(axis=1)
    base = np.concatenate([(Zx * np.roll(Zy, -d, axis=0)).sum(axis=1) for d in A1_SHIFTS])
    thr = float(np.quantile(np.abs(base), 0.999))
    sig = np.abs(rho) > thr
    out = {"q": {q: float(np.quantile(np.abs(rho), q)) for q in A1_Q}, "thr": thr,
           "base_q": {q: float(np.quantile(np.abs(base), q)) for q in A1_Q},
           "n": len(rho), "n_base": len(base), "frac_sig": float(sig.mean()),
           "sig_pos": float((rho[sig] > 0).mean()) if sig.any() else float("nan"),
           "sig_neg": float((rho[sig] < 0).mean()) if sig.any() else float("nan"),
           "all_pos": float((rho > 0).mean()), "mean_rho": float(rho.mean()), "rho": rho}
    # A2: is the most correlated source the same index? (both directions, significant rows only)
    Cm = np.abs(Zx @ Zy.T)
    idx = np.nonzero(sig)[0]
    if len(idx):
        out["a2_y_given_x"] = float((np.argmax(Cm[:, idx], axis=0) == idx).mean())  # target y
        out["a2_x_given_y"] = float((np.argmax(Cm[idx, :], axis=1) == idx).mean())  # target x
    else:
        out["a2_y_given_x"] = out["a2_x_given_y"] = float("nan")
    out["n_sig"] = int(len(idx))
    # adjusted A2 (section 10): significant rows with |rho| > 2 x threshold, i.e. clear of the
    # maximum null correlation over all other sources
    st = np.nonzero(np.abs(rho) > 2 * thr)[0]
    out["n_strong"] = int(len(st))
    if len(st):
        out["a2s_y_given_x"] = float((np.argmax(Cm[:, st], axis=0) == st).mean())
        out["a2s_x_given_y"] = float((np.argmax(Cm[st, :], axis=1) == st).mean())
    else:
        out["a2s_y_given_x"] = out["a2s_x_given_y"] = float("nan")
    off = Cm.copy()
    np.fill_diagonal(off, 0.0)
    out["null_max_median"] = float(np.median(off.max(axis=0)))
    return out


# ----------------------------------------------------------------------------- A3

def holdout_regression(t, S):
    """Per-neuron least squares (no intercept) on one parity half of the positions, prediction on
    the other half, then swap. t: (n, L); S: list of (n, L) sources. Returns the prediction."""
    n, L = t.shape
    p = len(S)
    par = np.arange(L) & 1
    pred = np.empty_like(t)
    for h in (0, 1):
        fc = np.nonzero(par == h)[0]
        ec = np.nonzero(par != h)[0]
        tf = t[:, fc]
        Sf = [s[:, fc] for s in S]
        G = np.empty((n, p, p))
        b = np.empty((n, p))
        for i in range(p):
            b[:, i] = (Sf[i] * tf).sum(axis=1)
            for j in range(i, p):
                G[:, i, j] = G[:, j, i] = (Sf[i] * Sf[j]).sum(axis=1)
        tr = np.trace(G, axis1=1, axis2=2)
        G += (1e-12 * tr + 1e-300)[:, None, None] * np.eye(p)
        beta = np.linalg.solve(G, b[:, :, None])[:, :, 0]
        pr = np.zeros((n, len(ec)))
        for i in range(p):
            pr += beta[:, i:i + 1] * S[i][:, ec]
        pred[:, ec] = pr
    return pred


def r2_rows(t, pred):
    sst = ((t - t.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    sse = ((t - pred) ** 2).sum(axis=1)
    r2 = np.where(sst > 0, 1.0 - sse / np.where(sst > 0, sst, 1.0), 0.0)
    return np.clip(r2, 0.0, 1.0 - 1e-12)


def pred_context(vhat, ei, K):
    """Section 3 A3.3 context: (E, 16-quantile bin of the prediction, 4-quantile bin of the
    neuron's prediction strength). vhat: (n, L) prediction in code-value units."""
    n, L = vhat.shape
    e16 = np.quantile(vhat, np.arange(1, 16) / 16)
    vb = np.searchsorted(e16, vhat, side="right")
    strength = np.sqrt((vhat * vhat).mean(axis=1))
    e4 = np.quantile(strength, (0.25, 0.5, 0.75))
    sb = np.searchsorted(e4, strength, side="right")
    ctx = (ei.astype(np.int64) * 16 + vb) * 4 + sb[:, None]
    return ctx, K * 64


def a3_config(T, srcs, shift=0):
    t = T["w"]
    n, L = t.shape
    S = [np.roll(s["w"], -shift, axis=0) if shift else s["w"] for s in srcs]
    pred = holdout_regression(t, S)
    r2 = r2_rows(t, pred)
    approx = float(np.mean(-0.5 * np.log2(1.0 - r2)))
    vhat = pred / T["sc"]
    ctx, nctx = pred_context(vhat, T["ei"], T["K"])
    h_ctx = est.holdout_cross_entropy(ctx, T["c"], T["half"], nctx)[0]
    h_e = est.holdout_cross_entropy(T["ei"], T["c"], T["half"], T["K"])[0]
    gain = h_e - h_ctx
    cost = len(S) * SLOPE_BITS * n / t.size
    return {"gain": gain, "approx": approx, "cost": cost, "net": gain - cost, "h_e": h_e,
            "h_ctx": h_ctx, "r2_median": float(np.median(r2)), "r2_mean": float(r2.mean()),
            "n": int(t.size)}


def a4_orders(nets):
    """nets: config name -> net. Returns {order: total} with the spec's (net2 + net3)/3, negatives
    recorded as 0 (no prediction used)."""
    out = {}
    for x, y, z in permutations(FAMS):
        n2 = nets[cfg_name(y, (x,))]
        n3 = nets[cfg_name(z, tuple(sorted((x, y), key=FAMS.index)))]
        out[f"{x}{y}{z}"] = (max(0.0, n2) + max(0.0, n3)) / 3.0
    return out


def a_expert(fams, configs=A3_CONFIGS, pairs=A1_PAIRS, doms=("w", "v")):
    """fams: dict a/g/o -> family(). Runs A1/A2 for the pairs and A3 (real and d=1) for the
    configurations; A4 when all nine configurations were run."""
    out = {"A1": {}, "A3": {}, "A3_base": {}}
    for x, y in pairs:
        for dom in doms:
            out["A1"][(x, y, dom)] = a1_pair(fams[x], fams[y], dom)
    for tgt, srcs in configs:
        name = cfg_name(tgt, srcs)
        out["A3"][name] = a3_config(fams[tgt], [fams[s] for s in srcs])
        out["A3_base"][name] = a3_config(fams[tgt], [fams[s] for s in srcs], shift=1)
    if len(out["A3"]) == len(A3_CONFIGS):
        nets = {k: v["net"] for k, v in out["A3"].items()}
        orders = a4_orders(nets)
        best = max(orders, key=orders.get)
        out["A4"] = {"orders": orders, "best": best, "total": orders[best]}
    return out


# ----------------------------------------------------------------------------- B

def omp_gram(DX, DD, xx, s):
    """Orthogonal matching pursuit through the Gram matrix. DX = X D^T (n, A), DD = D D^T (unit
    atoms), xx = ||x||^2. Returns sel (n, s), coef (n, s), residual energy after each step."""
    n, A = DX.shape
    ar = np.arange(n)
    sel = np.zeros((n, s), dtype=np.int64)
    corr = DX
    res = []
    coef = None
    for t in range(s):
        c = np.abs(corr)
        for u in range(t):
            c[ar, sel[:, u]] = -1.0
        sel[:, t] = np.argmax(c, axis=1)
        St = sel[:, :t + 1]
        G = DD[St[:, :, None], St[:, None, :]]
        b = DX[ar[:, None], St]
        coef = np.linalg.solve(G + 1e-10 * np.eye(t + 1), b[:, :, None])[:, :, 0]
        corr = DX - np.einsum("nt,nta->na", coef, DD[St])
        e = xx - 2.0 * (coef * b).sum(axis=1) + np.einsum("nt,ntu,nu->n", coef, G, coef)
        res.append(np.maximum(e, 0.0))
    return sel, coef, res


def _renorm(D, Xf, rng):
    nrm = np.sqrt((D * D).sum(axis=1))
    for i in np.nonzero(nrm <= 1e-12)[0]:
        while True:
            x = Xf[rng.integers(len(Xf))]
            if (x * x).sum() > 0:
                D[i] = x
                nrm[i] = math.sqrt(float((x * x).sum()))
                break
    return D / nrm[:, None]


def dict_learn(Xf, atoms, rng, iters=15, s=2):
    """003 R8.2 procedure with `atoms` atoms: init = random fit vectors, OMP with s atoms,
    least-squares dictionary update, renormalise; `iters` rounds."""
    n = len(Xf)
    D = _renorm(Xf[rng.choice(n, size=atoms, replace=False)].copy(), Xf, rng)
    xx = (Xf * Xf).sum(axis=1)
    ar = np.arange(n)[:, None]
    for _ in range(iters):
        sel, coef, _ = omp_gram(Xf @ D.T, D @ D.T, xx, s)
        Cm = np.zeros((n, atoms))
        Cm[ar, sel] = coef
        D = np.linalg.solve(Cm.T @ Cm + 1e-6 * np.eye(atoms), Cm.T @ Xf)
        D = _renorm(D, Xf, rng)
    return D


def b_stack(experts, role):
    """experts: list of (C', E). Returns centred unit-length w-domain vectors (all experts stacked),
    their means and lengths, and the per-expert neuron count."""
    W = np.concatenate([family(Cp, E, role)["w"] for Cp, E in experts])
    mu = W.mean(axis=1, keepdims=True)
    W -= mu
    nrm = np.sqrt((W * W).sum(axis=1, keepdims=True))
    W /= np.where(nrm > 0, nrm, 1.0)
    return W, mu[:, 0], nrm[:, 0]


def b_measure(X, nf, rng, atoms, ks, self_ks, ss):
    """B1 and B2 on one variant. X: stacked unit vectors; the first nf are the fit set."""
    Xf, Xe = X[:nf], X[nf:]
    L = X.shape[1]
    tot = float((Xe * Xe).sum())
    lam, V = np.linalg.eigh(Xf.T @ Xf)
    V = V[:, ::-1]
    kmax = min(max(ks), L)
    P = Xe @ V[:, :kmax]
    cum = np.cumsum((P * P).sum(axis=0))
    cap = {k: float(cum[k - 1] / tot) for k in ks if k <= L}
    perm = rng.permutation(len(Xe))
    h1, h2 = np.sort(perm[:len(Xe) // 2]), np.sort(perm[len(Xe) // 2:])
    ksm = max(self_ks)
    selfc = {k: [] for k in self_ks}
    for a, b in ((h1, h2), (h2, h1)):
        _, _, Vt = np.linalg.svd(Xe[a], full_matrices=False)
        Pb = Xe[b] @ Vt[:ksm].T
        cb = np.cumsum((Pb * Pb).sum(axis=0)) / float((Xe[b] * Xe[b]).sum())
        for k in self_ks:
            selfc[k].append(float(cb[min(k, len(cb)) - 1]))
    selfc = {k: float(np.mean(v)) for k, v in selfc.items()}
    D = dict_learn(Xf, atoms, rng)
    xx = (Xe * Xe).sum(axis=1)
    _, _, res = omp_gram(Xe @ D.T, D @ D.T, xx, max(ss))
    r2d = {s: 1.0 - float(res[s - 1].sum()) / tot for s in ss}
    return {"captured": cap, "captured_self": selfc, "r2_dict": r2d}, V[:, :kmax], D


def b3_predict(Xe, method, V=None, k=None, D=None):
    """Hold-out prediction of the eval vectors: coefficients from one parity half of the positions,
    prediction of the other half, swap."""
    n, L = Xe.shape
    par = np.arange(L) & 1
    pred = np.empty_like(Xe)
    for h in (0, 1):
        fc = np.nonzero(par == h)[0]
        ec = np.nonzero(par != h)[0]
        if method == "B1":
            Vk = V[:, :k]
            Vh = Vk[fc]
            beta = np.linalg.solve(Vh.T @ Vh + 1e-12 * np.eye(k), Vh.T @ Xe[:, fc].T)
            pred[:, ec] = (Vk[ec] @ beta).T
        else:
            Dh = D[:, fc]
            nh = np.sqrt((Dh * Dh).sum(axis=1))
            nh = np.where(nh > 0, nh, 1.0)
            Dhn = Dh / nh[:, None]
            Xh = Xe[:, fc]
            sel, coef, _ = omp_gram(Xh @ Dhn.T, Dhn @ Dhn.T, (Xh * Xh).sum(axis=1), 2)
            pr = np.zeros((n, len(ec)))
            for t in range(sel.shape[1]):
                pr += (coef[:, t] / nh[sel[:, t]])[:, None] * D[sel[:, t]][:, ec]
            pred[:, ec] = pr
    return pred


def b3_gain(pred, mu_e, nrm_e, eval_codes, n_per):
    """Prediction back to w (x length + mean), then / scale; context as A3 with E = scale byte;
    hold-out = the two eval experts (count on one, evaluate on the other, swap)."""
    c = np.concatenate([f[0] for f in eval_codes])
    eb = np.concatenate([f[1] for f in eval_codes])
    pw = pred * nrm_e[:, None] + mu_e[:, None]
    vhat = pw / np.exp2(eb.astype(np.float64) - 127.0)
    ctx, nctx = pred_context(vhat, eb, 256)
    half = np.repeat(np.arange(len(eval_codes)) % 2, n_per)[:, None] * np.ones((1, c.shape[1]),
                                                                              dtype=np.int64)
    h_ctx = est.holdout_cross_entropy(ctx, c, half, nctx)[0]
    h_e = est.holdout_cross_entropy(eb, c, half, 256)[0]
    return h_e, h_ctx


def b_layer(experts, s4_experts, role, seed, n_fit=6, atoms=512, ks=(4, 16, 64, 256, 1024),
            self_ks=(4, 16, 64, 256), ss=(1, 2, 4, 8), force_b3=False):
    """B1-B3 for one layer and one vector family. experts / s4_experts: lists of (C', E), the
    first n_fit are the fit experts, the rest the eval experts (the S4 list is the same experts
    after S4)."""
    Cp0 = experts[0][0]
    n_per = Cp0.shape[1] if role == "w2" else Cp0.shape[0]
    nf = n_fit * n_per
    out = {"n_per": n_per, "n_fit": nf, "n_eval": (len(experts) - n_fit) * n_per}
    X, _, _ = b_stack(s4_experts, role)
    out["S4"], _, _ = b_measure(X, nf, rng_for(seed, 1), atoms, ks, self_ks, ss)
    del X
    X, mu, nrm = b_stack(experts, role)
    L = X.shape[1]
    out["real"], V, D = b_measure(X, nf, rng_for(seed, 2), atoms, ks, self_ks, ss)
    out["L"] = L
    # significance
    dk = {}
    sig1 = {}
    for k in out["real"]["captured"]:
        ex = out["real"]["captured"][k] - k / L
        exs = out["S4"]["captured"][k] - k / L
        sig1[k] = bool(ex > 0.002 and ex > 5 * abs(exs))
        dk[k] = out["real"]["captured"][k] - out["S4"]["captured"][k]
    sig2 = {s: bool(out["real"]["r2_dict"][s] - out["S4"]["r2_dict"][s] > 0.005) for s in ss}
    out["sig_B1"], out["sig_B2"] = sig1, sig2
    run1 = any(sig1.values()) or force_b3
    run2 = sig2.get(2, False) or force_b3
    out["B3"] = {}
    if run1 or run2:
        Xe = X[nf:]
        mu_e, nrm_e = mu[nf:], nrm[nf:]
        ev = [family(Cp, E, role) for Cp, E in experts[n_fit:]]
        codes = [(f["c"], f["eb"]) for f in ev]
        del ev
        todo = []
        if run1:
            kk = [k for k in dk if k < L]
            kstar = max(kk, key=dk.get)
            todo.append(("B1", dict(V=V, k=kstar), kstar * SLOPE_BITS / L, kstar))
        if run2:
            ab = math.log2(atoms) if atoms != 512 else 9
            todo.append(("B2", dict(D=D), 2 * (ab + SLOPE_BITS) / L, ""))
        for name, kw, cost, kstar in todo:
            pred = b3_predict(Xe, name, **kw)
            r2 = 1.0 - float(((Xe - pred) ** 2).sum()) / float((Xe * Xe).sum())
            r2i = r2_rows(Xe, pred)
            h_e, h_ctx = b3_gain(pred, mu_e, nrm_e, codes, n_per)
            gain = h_e - h_ctx
            out["B3"][name] = {"gain": gain, "cost": cost, "net": gain - cost, "k": kstar,
                               "r2_holdout": r2,
                               "approx_pooled": -0.5 * math.log2(1.0 - max(0.0, r2)),
                               "approx": float(np.mean(-0.5 * np.log2(1.0 - r2i))),
                               "r2_vec_median": float(np.median(r2i)),
                               "h_e": h_e, "h_ctx": h_ctx, "forced": bool(force_b3)}
    return out


# ----------------------------------------------------------------------------- C2 round trip

A15 = np.array([c for c in range(16) if c != 8], dtype=np.uint8)
SYM = np.full(16, -1, dtype=np.int64)
SYM[A15] = np.arange(15)
PROB_BITS = 15
M = 1 << PROB_BITS
RANS_L = 1 << 23


def quant_freqs(counts):
    """Integer frequencies summing to 2^15; every occurring symbol gets at least 1."""
    counts = np.asarray(counts, dtype=np.float64)
    occ = counts > 0
    f = np.zeros(len(counts), dtype=np.int64)
    if not occ.any():
        return f
    f[occ] = np.maximum(1, np.round(counts[occ] / counts.sum() * M)).astype(np.int64)
    diff = M - int(f.sum())
    if diff > 0:
        f[int(np.argmax(f))] += diff
    for j in np.argsort(-f, kind="stable"):
        if diff >= 0:
            break
        take = min(-diff, int(f[j]) - 1)
        if take > 0:
            f[j] -= take
            diff += take
    assert f.sum() == M and np.all(f[occ] >= 1)
    return f


def rans_encode(fl, cl):
    out = bytearray()
    x = RANS_L
    for f, c in zip(reversed(fl), reversed(cl)):
        xmax = (RANS_L >> PROB_BITS << 8) * f
        while x >= xmax:
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // f) << PROB_BITS) + (x % f) + c
    out += x.to_bytes(4, "little")
    out.reverse()
    return bytes(out)


def c2_encode(Cp, E):
    """Static two-stream coder: scale stream under P(E), codeword stream under P(C'|E).
    File header: R, W (uint32), K (uint16), K scale bytes, K scale frequencies (uint16), K x 15
    codeword frequencies (uint16); probabilities are 16 bit each in the header."""
    R, W = Cp.shape
    assert W % B == 0 and E.shape == (R, W // B)
    sym = SYM[Cp.ravel()]
    assert (sym >= 0).all(), "C' must not contain codeword 8"
    uE, Ei = np.unique(E.ravel(), return_inverse=True)
    K = len(uE)
    fE = quant_freqs(np.bincount(Ei, minlength=K))
    Eiw = np.repeat(Ei, B)
    tab = np.bincount(Eiw * 15 + sym, minlength=K * 15).reshape(K, 15)
    fC = np.stack([quant_freqs(t) for t in tab])
    cE = np.concatenate([[0], np.cumsum(fE)[:-1]])
    cC = np.concatenate([np.zeros((K, 1), dtype=np.int64), np.cumsum(fC, axis=1)[:, :-1]], axis=1)
    header = (struct.pack("<IIH", R, W, K) + bytes(uE.astype(np.uint8).tolist())
              + fE.astype("<u2").tobytes() + fC.astype("<u2").tobytes())
    fl = np.concatenate([fE[Ei], fC[Eiw, sym]]).tolist()
    cl = np.concatenate([cE[Ei], cC[Eiw, sym]]).tolist()
    stream = rans_encode(fl, cl)
    return header, stream


def _lookup(freqs):
    lk = []
    for s, f in enumerate(freqs):
        lk.extend([s] * int(f))
    return lk


def c2_decode(header, stream):
    """Recover E and C' from the header and the stream only."""
    R, W, K = struct.unpack_from("<IIH", header, 0)
    o = 10
    uE = list(header[o:o + K])
    o += K
    fE = list(np.frombuffer(header, dtype="<u2", count=K, offset=o).astype(int))
    o += 2 * K
    fC = np.frombuffer(header, dtype="<u2", count=K * 15, offset=o).astype(int).reshape(K, 15)
    assert o + 2 * K * 15 == len(header)
    cE = [0] * K
    for i in range(1, K):
        cE[i] = cE[i - 1] + fE[i - 1]
    lkE = _lookup(fE)
    fCl = [list(map(int, r)) for r in fC]
    cCl = []
    for r in fCl:
        c = [0] * 15
        for i in range(1, 15):
            c[i] = c[i - 1] + r[i - 1]
        cCl.append(c)
    lkC = [_lookup(r) for r in fCl]
    mask = M - 1
    data = stream
    x = int.from_bytes(data[:4], "big")
    pos = 4
    nS = R * (W // B)
    ei = []
    for _ in range(nS):
        slot = x & mask
        s = lkE[slot]
        x = fE[s] * (x >> PROB_BITS) + slot - cE[s]
        while x < RANS_L:
            x = (x << 8) | data[pos]
            pos += 1
        ei.append(s)
    out = bytearray(R * W)
    t = 0
    for k in ei:
        f, c, lk = fCl[k], cCl[k], lkC[k]
        for _ in range(B):
            slot = x & mask
            s = lk[slot]
            x = f[s] * (x >> PROB_BITS) + slot - c[s]
            while x < RANS_L:
                x = (x << 8) | data[pos]
                pos += 1
            out[t] = s
            t += 1
    assert x == RANS_L and pos == len(data), "stream not fully consumed"
    Ed = np.array(uE, dtype=np.uint8)[np.array(ei, dtype=np.int64)].reshape(R, W // B)
    Cd = A15[np.frombuffer(bytes(out), dtype=np.uint8)].reshape(R, W)
    return Cd, Ed


def c2_roundtrip(C, E):
    """C: original codewords (with -0), E: scale bytes. Encodes C' = merge_zero(C)."""
    from .ledger import merge_zero
    Cp = merge_zero(C)
    header, stream = c2_encode(Cp, E)
    Cd, Ed = c2_decode(header, stream)
    scale = np.repeat(np.exp2(E.astype(np.float64) - 127.0), B, axis=1)
    scale_d = np.repeat(np.exp2(Ed.astype(np.float64) - 127.0), B, axis=1)
    N = C.size
    bits = 8 * (len(header) + len(stream))
    return {"E_equal": bool(np.array_equal(Ed, E)), "Cp_equal": bool(np.array_equal(Cd, Cp)),
            "values_equal": bool(np.array_equal(E2M1[Cd] * scale_d, E2M1[C] * scale)),
            "n_neg_zero": int((C == 8).sum()), "header_bytes": len(header),
            "stream_bytes": len(stream), "N": N, "bits_per_weight": bits / N,
            "header_bits_per_weight": 8 * len(header) / N}


# ----------------------------------------------------------------------------- D probe

D_LD = (2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64)
D_UD = (2, 3, 4, 8)
PBIN = np.array([0] + [1] * 3 + [2] * 4 + [3] * 8 + [4] * 16)
NEWTON_STEPS = 20
RIDGE = 1e-4
CHUNK = 20000


def _shift(X, dr, dc):
    """Value at (i - dr, j - dc); out of range -> 0."""
    R, W = X.shape
    out = np.zeros_like(X)
    if dr >= R or abs(dc) >= W:
        return out
    if dc >= 0:
        out[dr:, dc:] = X[:R - dr, :W - dc]
    else:
        out[dr:, :W + dc] = X[:R - dr, -dc:]
    return out


def d_components(Cp, E, cnt_rows):
    """Yields (name, context array (R, W) int64, number of contexts) one at a time."""
    R, W = Cp.shape
    nb = W // B
    _, Eb = np.unique(E, return_inverse=True)
    Eb = Eb.reshape(E.shape).astype(np.int64)
    K = int(Eb.max()) + 1
    e = np.repeat(Eb, B, axis=1)
    c = Cp

    def sh(dr, dc):
        return _shift(c, dr, dc).astype(np.int64)

    def bins8(x):
        edges = np.quantile(x[cnt_rows], np.arange(1, 8) / 8)
        return np.searchsorted(edges, x, side="right").astype(np.int64)

    yield "E", e, K
    L1, L2, L3 = sh(0, 1), sh(0, 2), sh(0, 3)
    yield "E,L1", e * 16 + L1, K * 16
    yield "E,L1,L2", (e * 16 + L1) * 16 + L2, K * 256
    yield "E,L1,L2,L3", ((e * 16 + L1) * 16 + L2) * 16 + L3, K * 4096
    yield "L1,L2,L3,L4", ((L1 * 16 + L2) * 16 + L3) * 16 + sh(0, 4), 65536
    U1 = sh(1, 0)
    yield "E,U1", e * 16 + U1, K * 16
    yield "E,U1,L1", (e * 16 + U1) * 16 + L1, K * 256
    yield "E,U1,L1,UL", ((e * 16 + U1) * 16 + L1) * 16 + sh(1, 1), K * 4096
    yield "E,U1,U2", (e * 16 + U1) * 16 + sh(2, 0), K * 256
    yield "E,U1,UR", (e * 16 + U1) * 16 + sh(1, -1), K * 256
    del L2, L3, U1
    for d in D_LD:
        yield f"E,L{d}", e * 16 + (L1 if d == 1 else sh(0, d)), K * 16
    for d in D_UD:
        yield f"E,U{d}", e * 16 + sh(d, 0), K * 16
    del L1
    p = np.broadcast_to(np.arange(W) % B, (R, W))
    yield "E,p", e * B + p, K * B
    pb = np.broadcast_to(PBIN[np.arange(W) % B], (R, W))
    av = ABSV[c].reshape(R, nb, B)
    prev = np.cumsum(av, axis=2) - av
    pp = np.arange(B, dtype=np.float64)
    pm = np.where(pp > 0, prev / np.where(pp > 0, pp, 1.0), 0.0).reshape(R, W)
    yield "E,pbin,blkmean", (e * 5 + pb) * 8 + bins8(pm), K * 40
    m = (c & 7).reshape(R, nb, B).astype(np.int64)
    cm = np.maximum.accumulate(m, axis=2)
    pmax = np.zeros_like(cm)
    pmax[:, :, 1:] = cm[:, :, :-1]
    yield "E,pbin,blkmax", (e * 5 + pb) * 8 + pmax.reshape(R, W), K * 40
    del m, cm, pmax, pm, prev
    Er = E.astype(np.int64)
    dl = np.full((R, nb), 5, dtype=np.int64)
    dl[:, 1:] = np.clip(Er[:, :-1] - Er[:, 1:], -2, 2) + 2
    du = np.full((R, nb), 5, dtype=np.int64)
    du[1:] = np.clip(Er[:-1] - Er[1:], -2, 2) + 2
    yield "E,dEleft,dEup", (e * 6 + np.repeat(dl, B, axis=1)) * 6 + np.repeat(du, B, axis=1), K * 36
    bm = av.mean(axis=2)
    lbm = np.zeros_like(bm)
    lbm[:, 1:] = bm[:, :-1]
    yield "E,leftblkmean", e * 8 + bins8(np.repeat(lbm, B, axis=1)), K * 8
    a2 = av.reshape(R, W)
    rp = np.cumsum(a2, axis=1) - a2
    jj = np.arange(W, dtype=np.float64)
    rm = np.where(jj > 0, rp / np.where(jj > 0, jj, 1.0), 0.0)
    yield "E,rowmean", e * 8 + bins8(rm), K * 8
    del rp, rm
    cs = np.zeros((R, W))
    for d in range(1, 9):
        cs += _shift(a2, d, 0)
    yield "E,colmean8", e * 8 + bins8(cs / 8.0), K * 8


def d_probe(Cp, E, seed, n_fit=300_000):
    """Section 6 context-mixing probe on one matrix C' (R, W) with block scales E."""
    R, W = Cp.shape
    rmod = np.arange(R) % 4
    cnt_rows = np.broadcast_to((rmod <= 1)[:, None], (R, W))
    cflat = Cp.ravel().astype(np.int64)
    cnt_idx = np.nonzero(np.repeat(rmod <= 1, W))[0]
    ev_idx = np.nonzero(np.repeat(rmod == 3, W))[0]
    pool = np.nonzero(np.repeat(rmod == 2, W))[0]
    rng = rng_for(seed, 40)
    fit_idx = np.sort(rng.choice(pool, size=min(n_fit, len(pool)), replace=False))
    _, Eb = np.unique(E, return_inverse=True)
    Eb = Eb.reshape(E.shape)
    K = int(Eb.max()) + 1
    ew = np.repeat(Eb, B, axis=1).ravel().astype(np.int64)
    nE = np.bincount(ew[cnt_idx] * 16 + cflat[cnt_idx], minlength=K * 16).reshape(K, 16)
    A = np.nonzero(nE.sum(axis=0) > 0)[0]
    nA = len(A)
    q = np.zeros((K, 16))
    q[:, A] = (nE[:, A] + 0.5) / (nE[:, A].sum(axis=1, keepdims=True) + 0.5 * nA)
    aidx = np.full(16, -1)
    aidx[A] = np.arange(nA)
    y_fit, y_ev = aidx[cflat[fit_idx]], aidx[cflat[ev_idx]]
    assert (y_ev >= 0).all() and (y_fit >= 0).all(), "symbol outside the counting alphabet"
    e_fit, e_ev = ew[fit_idx], ew[ev_idx]
    qA = q[:, A]
    comps = []
    for name, ctx, nctx in d_components(Cp, E, cnt_rows):
        ctx = ctx.ravel()
        t = np.bincount(ctx[cnt_idx] * 16 + cflat[cnt_idx], minlength=nctx * 16)
        t = t.reshape(nctx, 16)[:, A].astype(np.float64)
        comps.append({"name": name, "t": t, "tot": t.sum(axis=1), "nctx": nctx,
                      "ctx_fit": ctx[fit_idx].astype(np.int32), "ctx_ev": ctx[ev_idx].astype(np.int32),
                      "n_ctx_seen": int((t.sum(axis=1) > 0).sum())})
        del ctx
    Kc = len(comps)

    def logp(sel, which, e_sel):
        """(n, Kc, nA) natural-log probabilities of every component at positions `sel`."""
        out = np.empty((len(sel), Kc, nA))
        qq = 2.0 * qA[e_sel]
        for k, cp in enumerate(comps):
            cx = cp[which][sel]
            out[:, k, :] = np.log((cp["t"][cx] + qq) / (cp["tot"][cx] + 2.0)[:, None])
        return out

    def chunks(n):
        return [np.arange(i, min(n, i + CHUNK)) for i in range(0, n, CHUNK)]

    fit_chunks = chunks(len(fit_idx))

    def loss_grad(w, need_grad):
        tot = 0.0
        g = np.zeros(Kc)
        Hm = np.zeros((Kc, Kc))
        for sel in fit_chunks:
            Lk = logp(sel, "ctx_fit", e_fit[sel])
            s = np.einsum("nka,k->na", Lk, w)
            mx = s.max(axis=1, keepdims=True)
            lz = mx[:, 0] + np.log(np.exp(s - mx).sum(axis=1))
            ar = np.arange(len(sel))
            yy = y_fit[sel]
            tot -= float((s[ar, yy] - lz).sum())
            if need_grad:
                P = np.exp(s - lz[:, None])
                EL = np.einsum("na,nka->nk", P, Lk)
                g += (EL - Lk[ar, :, yy]).sum(axis=0)
                Lw = Lk * np.sqrt(P)[:, None, :]
                Lw = Lw.transpose(1, 0, 2).reshape(Kc, -1)
                Hm += Lw @ Lw.T - EL.T @ EL
        n = len(fit_idx)
        return tot / n, g / n, Hm / n

    w = np.zeros(Kc)
    w[0] = 1.0
    hist = []
    f0, g, Hm = loss_grad(w, True)
    hist.append(f0)
    for _ in range(NEWTON_STEPS):
        step = -np.linalg.solve(Hm + RIDGE * np.eye(Kc), g)
        t = 1.0
        accepted = False
        for _ in range(40):
            f1 = loss_grad(w + t * step, False)[0]
            if f1 < f0:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            break
        w = w + t * step
        f0, g, Hm = loss_grad(w, True)
        hist.append(f0)
    # evaluation
    ev_mix = 0.0
    ev_single = np.zeros(Kc)
    for sel in chunks(len(ev_idx)):
        Lk = logp(sel, "ctx_ev", e_ev[sel])
        ar = np.arange(len(sel))
        yy = y_ev[sel]
        ev_single -= Lk[ar, :, yy].sum(axis=0)
        s = np.einsum("nka,k->na", Lk, w)
        mx = s.max(axis=1, keepdims=True)
        lz = mx[:, 0] + np.log(np.exp(s - mx).sum(axis=1))
        ev_mix -= float((s[ar, yy] - lz).sum())
    n_ev = len(ev_idx)
    ln2 = math.log(2.0)
    single = ev_single / n_ev / ln2
    h_mix = ev_mix / n_ev / ln2
    h_e = float(single[0])
    return {"h_E": h_e, "h_mix": h_mix, "gain": h_e - h_mix,
            "components": [c["name"] for c in comps],
            "single_gain": {c["name"]: h_e - float(single[k]) for k, c in enumerate(comps)},
            "weights": {c["name"]: float(w[k]) for k, c in enumerate(comps)},
            "n_ctx_seen": {c["name"]: c["n_ctx_seen"] for c in comps},
            "fit_loss_bits": [x / ln2 for x in hist], "n_fit": len(fit_idx), "n_eval": n_ev,
            "n_count": len(cnt_idx), "K_E": K, "alphabet": A.tolist()}
