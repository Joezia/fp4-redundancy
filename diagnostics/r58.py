"""Spec 003: R5-R8 and RX measurements on merged-zero codewords C' (statistics only).

`measure_matrix` (R6.1, R6.2, R5, R7.1-R7.3, R8) and `measure_pair` (R6.3, RX, R7.3 across a pair)
are the single code paths used for both the section-7 synthetic checks and the model matrices.
Every function returns long-format rows (FIELDS) plus a `vals` dict of the scalars the checks and the
report need. Data-derived objects (clusters, dictionaries, nearest neighbours, singular vectors) are
recomputed on every shuffled copy and never written out.
"""
import hashlib
import math

import numpy as np

from . import est
from .ledger import cond_entropy, merge_zero
from .measure import E2M1

B = 32
KS_R62 = (8, 10, 12, 14, 16, 20, 24, 32, 48, 64)
KS_R63 = (12, 16, 24, 32)
R5_SHARE_K = (1, 2, 4, 8, 16, 32, 64, 128, 256)
R52_K = (1, 2, 4, 8, 16, 32, 64, 128)
KM_K = (2, 4, 8, 16, 32, 64, 128, 256)
KM_STRICT_MAX = 64
KM_ITERS = 20
DICT_ATOMS, DICT_ITERS, DICT_S = 64, 15, (1, 2, 3)
N_QUERY = 500
R72_SAMPLE = 100_000
REPS = 3
TOP_VEC = 64
MAX_EXAMPLES = 20
SEED = 20261113

FIELDS = ["matrix", "layer", "expert", "role", "matrix_b", "pair_group", "item", "variable",
          "domain", "group_or_context", "k", "subset", "n", "value", "analytic_bias", "baseline",
          "baseline_mean", "baseline_std", "net", "significant", "note"]

# item ids for deterministic per-item random streams
_ITEM = {"R6.1": 1, "R6.2": 2, "R6.3": 3, "RX": 4, "R5": 5, "R5.2": 6, "R7.1": 7, "R7.2": 8,
         "R7.3": 9, "R8.2": 10, "S4": 11, "sample": 12}


def item_rng(seed, item, *extra):
    return np.random.default_rng([SEED, int(seed), _ITEM[item], *[int(x) for x in extra]])


# ----------------------------------------------------------------------------- records

def _num(x):
    if x is None or x == "":
        return ""
    x = float(x)
    return x if np.isfinite(x) else ""


class Rows:
    def __init__(self, meta):
        self.meta = meta
        self.rows = []

    def add(self, item, variable, value, *, domain="", group="", k="", subset="", n=None,
            baseline="", reps=None, sig=None, bias=None, note="", net=None):
        r = dict.fromkeys(FIELDS, "")
        r.update(self.meta)
        r.update(item=item, variable=variable, domain=domain, group_or_context=group, k=k,
                 subset=subset, n="" if n is None else int(n), note=note)
        r["value"] = _num(value)
        if bias is not None:
            r["analytic_bias"] = _num(bias)
        if reps is not None:
            reps = [float(x) for x in reps]
            bm = float(np.mean(reps))
            bs = float(np.std(reps, ddof=1)) if len(reps) > 1 else float("nan")
            r.update(baseline=baseline, baseline_mean=bm, baseline_std=_num(bs))
            if value is not None and np.isfinite(value):
                r["net"] = float(value) - bm
        elif baseline:
            r["baseline"] = baseline
        if net is not None:
            r["net"] = _num(net)
        if sig is not None:
            r["significant"] = bool(sig)
        self.rows.append(r)
        return r


def mi_significant(net, dof, n, reps):
    """Section 4: net > 5 analytic sd AND > 5 replicate sd AND > 1e-4 bit."""
    bs = float(np.std(reps, ddof=1)) if len(reps) > 1 else float("nan")
    if not np.isfinite(bs):
        return None
    return bool(net > 5 * est.analytic_sd(dof, n) and net > 5 * bs and net > 1e-4)


# ----------------------------------------------------------------------------- data views

def shuffle(Cp, E, kind, rng):
    """Baselines of section 3 on C' (R, W) with block scales E (R, W/B). Returns (C', E)."""
    R, W = Cp.shape
    if kind == "S0":
        return rng.permutation(Cp.ravel()).reshape(R, W), E
    if kind == "S0E":
        out = Cp.copy()
        blocks = out.reshape(-1, B)
        flat = E.ravel()
        for e in np.unique(flat):
            idx = np.nonzero(flat == e)[0]
            blocks[idx] = rng.permutation(blocks[idx].ravel()).reshape(-1, B)
        return out, E
    if kind == "S2":
        return rng.permuted(Cp.reshape(R, W // B, B), axis=2).reshape(R, W), E
    if kind == "S3":
        p = rng.permutation(R)
        return Cp[p], E[p]
    if kind == "S4":
        m = Cp & 7
        s = rng.integers(0, 2, size=Cp.shape, dtype=np.uint8)
        return np.where(m > 0, m + 8 * s, 0).astype(np.uint8), E
    raise ValueError(kind)


def values(Cp, E, domain):
    v = E2M1[Cp]
    if domain == "v":
        return v
    if domain == "absv":
        return np.abs(v)
    if domain == "w":
        return v * np.repeat(np.exp2(E.astype(np.float64) - 127.0), B, axis=1)
    raise ValueError(domain)


def h_cond_e(Cp, E):
    _, Einv = np.unique(E.ravel(), return_inverse=True)
    return cond_entropy(np.repeat(Einv, B), int(Einv.max()) + 1, Cp.ravel())[0]


# ----------------------------------------------------------------------------- R6 exact repeats

GOLD = np.uint64(0x9E3779B97F4A7C15)
M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)


def pack16(Cp):
    """P[r, j] = codes j..j+15 of row r packed 4 bits each (code j lowest); past the row end = 0."""
    R, W = Cp.shape
    pad = np.zeros((R, W + 15), dtype=np.uint64)
    pad[:, :W] = Cp
    P = np.zeros((R, W), dtype=np.uint64)
    for i in range(16):
        P |= pad[:, i:i + W] << np.uint64(4 * i)
    return P


def window_words(P, k):
    """Exact keys of all in-row windows of length k: list of ceil(k/16) uint64 arrays, index r*n+j."""
    R, W = P.shape
    n = W - k + 1
    words = []
    for t in range(-(-k // 16)):
        x = P[:, 16 * t:16 * t + n]
        rem = k - 16 * t
        if rem < 16:
            x = x & np.uint64((1 << (4 * rem)) - 1)
        words.append(np.ascontiguousarray(x).ravel())
    return words, n


def mix(words):
    """64-bit hash of multi-word keys (splitmix-style); exactness is restored by verification."""
    with np.errstate(over="ignore"):
        h = np.zeros_like(words[0])
        for w in words:
            h = (h ^ w) * GOLD
            h ^= h >> np.uint64(31)
        h ^= h >> np.uint64(30)
        h *= M1
        h ^= h >> np.uint64(27)
        h *= M2
        h ^= h >> np.uint64(31)
    return h


def _runs(K):
    """K (n, w) sorted lexicographically -> boolean 'same as previous' (n-1,)."""
    return np.all(K[1:] == K[:-1], axis=1)


def dup_windows(words, want_pairs=False):
    """dup = windows - distinct windows (exact). Optionally (first, other) index pairs of repeats."""
    n = len(words[0])
    if len(words) == 1:
        cand = None
        key = words[0]
        o = np.argsort(key, kind="stable")
        ks = key[o]
        same = ks[1:] == ks[:-1]
        dup = int(same.sum())
        if not want_pairs:
            return dup, None
        order = o
    else:
        h = mix(words)
        o = np.argsort(h, kind="stable")
        hs = h[o]
        s = hs[1:] == hs[:-1]
        m = np.zeros(n, dtype=bool)
        m[1:] |= s
        m[:-1] |= s
        cand = o[m]
        if cand.size == 0:
            return 0, (np.zeros(0, np.int64), np.zeros(0, np.int64)) if want_pairs else None
        K = np.stack([w[cand] for w in words], axis=1)
        oc = np.lexsort(K.T[::-1])
        same = _runs(K[oc])
        dup = int(same.sum())
        if not want_pairs:
            return dup, None
        order = cand[oc]
    # pairs: each repeated window paired with the first window of its run
    first = np.empty(len(order), dtype=np.int64)
    start = np.r_[True, ~same]
    run_id = np.cumsum(start) - 1
    first_of_run = order[start]
    first = first_of_run[run_id]
    rep = np.r_[False, same]
    return dup, (first[rep], order[rep])


def block_keys(P, Cp):
    """Exact 2-word key of every aligned block (requires B == 32) and its number of distinct codes."""
    assert B == 32
    bk = np.stack([P[:, 0::B].ravel(), P[:, 16::B].ravel()], axis=1)
    s = np.sort(Cp.reshape(-1, B), axis=1)
    nd = 1 + (s[:, 1:] != s[:, :-1]).sum(axis=1)
    return bk, nd


def distinct_rows(K):
    if len(K) == 0:
        return 0, np.zeros(0, bool), np.zeros(0, np.int64)
    o = np.lexsort(K.T[::-1])
    same = _runs(K[o])
    return 1 + int((~same).sum()), same, o


def hexcode(x):
    return "".join("%x" % int(c) for c in x)


def r6_matrix(Cp, E, meta, rows, seed, H_E):
    vals = {}
    R, W = Cp.shape
    nb = W // B
    rng = item_rng(seed, "R6.1")
    P = pack16(Cp)
    bk, nd = block_keys(P, Cp)
    nblk = len(bk)
    d_all, same, o = distinct_rows(bk)
    rep_all = nblk - d_all
    zero = int(np.all(bk == 0, axis=1).sum())
    triv = nd <= 2
    nt = ~triv
    d_nt, same_nt, o_nt = distinct_rows(bk[nt])
    rep_nt = int(nt.sum()) - d_nt
    base = {}
    for kind in ("S0", "S2"):
        Xs, _ = shuffle(Cp, E, kind, rng)
        bks, nds = block_keys(pack16(Xs), Xs)
        base[kind] = nblk - distinct_rows(bks)[0]
        base[kind + "_nt"] = int((nds > 2).sum()) - distinct_rows(bks[nds > 2])[0]
    vals["r61"] = {"block_repeats": rep_all, "nontrivial_repeats": rep_nt, "all_zero_blocks": zero,
                   "le2_blocks": int(triv.sum()), "S0": base["S0"], "S2": base["S2"],
                   "S0_nt": base["S0_nt"], "S2_nt": base["S2_nt"]}
    for var, val, bkey in (("block_repeats", rep_all, ""), ("nontrivial_block_repeats", rep_nt, "_nt")):
        for kind in ("S0", "S2"):
            rows.add("R6.1", var, val, group="block", subset="matrix", n=nblk, baseline=kind,
                     reps=[base[kind + bkey]], note="1 replicate")
    rows.add("R6.1", "all_zero_blocks", zero, group="block", subset="matrix", n=nblk)
    rows.add("R6.1", "le2_distinct_code_blocks", int(triv.sum()), group="block", subset="matrix",
             n=nblk, note="includes all-zero blocks")
    # rows / scale rows
    row_h = [hashlib.sha1(Cp[r].tobytes()).hexdigest() for r in range(R)]
    srow_h = [hashlib.sha1(E[r].tobytes()).hexdigest() for r in range(R)]
    rows.add("R6.1", "row_repeats", R - len(set(row_h)), group="row", subset="matrix", n=R)
    rows.add("R6.1", "scale_row_repeats", R - len(set(srow_h)), group="scale_row", subset="matrix",
             n=R)
    vals["r61"]["row_repeats"] = R - len(set(row_h))
    vals["r61"]["scale_row_repeats"] = R - len(set(srow_h))
    # examples of non-trivial repeated blocks
    ex = []
    if rep_nt:
        nt_idx = np.nonzero(nt)[0][o_nt]
        start = np.r_[True, ~same_nt]
        run_first = nt_idx[start][np.cumsum(start) - 1]
        repm = np.r_[False, same_nt]
        for a, b in list(zip(run_first[repm], nt_idx[repm]))[:MAX_EXAMPLES]:
            ra, ba = divmod(int(a), nb)
            rb, bb = divmod(int(b), nb)
            ex.append(dict(meta, item="R6.1", kind="block", row=rb, col=bb * B, length=B,
                           other_row=ra, other_col=ba * B, all_zero=False,
                           content=hexcode(Cp[rb, bb * B:(bb + 1) * B])))
    pool = {"block_keys": bk, "block_nontrivial": nt, "row_hashes": row_h,
            "scale_row_hashes": srow_h,
            "matrix_hash": hashlib.sha1(Cp.tobytes() + str(Cp.shape).encode()).hexdigest(),
            "scale_hash": hashlib.sha1(E.tobytes() + str(E.shape).encode()).hexdigest()}

    # R6.2 in-row windows
    rng = item_rng(seed, "R6.2")
    X2, _ = shuffle(Cp, E, "S2", rng)
    X0, _ = shuffle(Cp, E, "S0", rng)
    P2, P0 = pack16(X2), pack16(X0)
    dup, dup2, dup0, nwin = {}, {}, {}, {}
    pairs_at = {}
    for k in KS_R62:
        words, n = window_words(P, k)
        nwin[k] = R * n
        dup[k], pr = dup_windows(words, want_pairs=k >= 16)
        if pr is not None:
            pairs_at[k] = (pr, n)
        dup2[k] = dup_windows(window_words(P2, k)[0])[0]
        if k <= 16:
            dup0[k] = dup_windows(window_words(P0, k)[0])[0]
        for kind, d in (("S2", dup2), ("S0", dup0)):
            if k in d:
                rows.add("R6.2", "dup_fraction", dup[k] / nwin[k], k=k, subset="matrix", n=nwin[k],
                         baseline=kind, reps=[d[k] / nwin[k]], note="1 replicate")
        if k > 16:
            rows.add("R6.2", "dup_fraction", dup[k] / nwin[k], k=k, subset="matrix", n=nwin[k],
                     baseline="S0", note="S0 only for k<=16")
        rows.add("R6.2", "dup_count", dup[k], k=k, subset="matrix", n=nwin[k], baseline="S2",
                 reps=[dup2[k]], note="1 replicate")
    kstar = max([k for k in KS_R62 if dup[k] > 0], default=None)
    kstar2 = max([k for k in KS_R62 if dup2[k] > 0], default=None)
    rows.add("R6.2", "K_star", kstar, subset="matrix",
             note="" if kstar else f"dup(k)=0 for all k>={KS_R62[0]}")
    rows.add("R6.2", "K_star", kstar2, subset="matrix", group="S2",
             note="S2 replicate" + ("" if kstar2 else f"; dup(k)=0 for all k>={KS_R62[0]}"))
    approx = (dup[16] - dup2[16]) / Cp.size * H_E
    rows.add("R6.2", "approx_coverage_bits", approx, k=16, subset="matrix", n=Cp.size,
             note="approximate upper bound: (dup16 - S2 dup16)/N * H(C'|E)")
    vals["r62"] = {"dup": dup, "dup_S2": dup2, "dup_S0": dup0, "Kstar": kstar, "Kstar_S2": kstar2,
                   "nwin": nwin, "approx": approx}
    # examples of net repeats at k >= 16 (largest k with real repeats)
    big = [k for k in KS_R62 if k >= 16 and dup[k] > 0 and dup[k] > dup2[k]]
    if big:
        k = max(big)
        (fa, fb), n = pairs_at[k]
        cand = []
        for a, b in zip(fa, fb):
            ra, ja = divmod(int(a), n)
            rb, jb = divmod(int(b), n)
            if ja > 0 and jb > 0 and Cp[ra, ja - 1] == Cp[rb, jb - 1]:
                continue  # not the start of a maximal match
            L = 0
            while ja + L < W and jb + L < W and Cp[ra, ja + L] == Cp[rb, jb + L]:
                L += 1
            seg = Cp[rb, jb:jb + L]
            cand.append(dict(meta, item="R6.2", kind=f"window k={k}", row=rb, col=jb, length=L,
                             other_row=ra, other_col=ja, all_zero=bool(np.all(seg == 0)),
                             content=hexcode(seg[:64]) + ("..." if L > 64 else "")))
            if len(cand) >= 200:
                break
        cand.sort(key=lambda d: -d["length"])
        ex.extend(cand[:MAX_EXAMPLES])
    return vals, ex, pool


def r63_pair(CpA, CpB, EB, rows, seed):
    rng = item_rng(seed, "R6.3")
    PA = pack16(CpA)
    XB, _ = shuffle(CpB, EB, "S2", rng)
    out = {}
    for k in KS_R63:
        wA, _ = window_words(PA, k)
        res = []
        for PB in (pack16(CpB), pack16(XB)):
            wB, _ = window_words(PB, k)
            res.append(_count_in(wA, wB))
        n = len(wB[0])
        out[k] = res
        rows.add("R6.3", "B_windows_in_A", res[0], k=k, n=n, baseline="S2(B)", reps=[res[1]],
                 note="1 replicate")
    return out


def _count_in(wA, wB):
    """Number of B windows whose exact content occurs among A windows."""
    if len(wA) == 1:
        return int(np.isin(wB[0], wA[0]).sum())
    hA, hB = mix(wA), mix(wB)
    ci = np.nonzero(np.isin(hB, hA))[0]
    if ci.size == 0:
        return 0
    oA = np.argsort(hA, kind="stable")
    hAs = hA[oA]
    lo = np.searchsorted(hAs, hB[ci], "left")
    hi = np.searchsorted(hAs, hB[ci], "right")
    eq = np.ones(ci.size, dtype=bool)
    for wa, wb in zip(wA, wB):
        eq &= wa[oA[lo]] == wb[ci]
    for i in np.nonzero(~eq & (hi - lo > 1))[0]:  # hash collision inside A: check the rest
        for j in range(lo[i] + 1, hi[i]):
            if all(wa[oA[j]] == wb[ci[i]] for wa, wb in zip(wA, wB)):
                eq[i] = True
                break
    return int(eq.sum())


# ----------------------------------------------------------------------------- RX same position

def _corr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum()) / d if d > 0 else float("nan")


def rx_pair(CpA, EA, CpB, EB, rows, seed):
    assert CpA.shape == CpB.shape and EA.shape == EB.shape
    N = CpA.size
    a = CpA.ravel().astype(np.int64)
    am, as_ = a & 7, a >> 3
    vA = E2M1[CpA].ravel()
    wA = values(CpA, EA, "w").ravel()

    def stats(Cb, Eb):
        b = Cb.ravel().astype(np.int64)
        out = {}
        out["I(C'_A;C'_B)"] = est.mi_table(np.bincount(a * 16 + b, minlength=256).reshape(16, 16))
        out["I(m_A;m_B)"] = est.mi_table(np.bincount(am * 8 + (b & 7), minlength=64).reshape(8, 8))
        out["I(s_A;s_B)"] = est.mi_table(np.bincount(as_ * 2 + (b >> 3), minlength=4).reshape(2, 2))
        vB = E2M1[Cb].ravel()
        out["corr_v"] = _corr(vA, vB)
        out["corr_absv"] = _corr(np.abs(vA), np.abs(vB))
        out["corr_w"] = _corr(wA, values(Cb, Eb, "w").ravel())
        return out

    real = stats(CpB, EB)
    base = {}
    for kind in ("S0", "S2", "S3"):
        rng = item_rng(seed, "RX", {"S0": 0, "S2": 2, "S3": 3}[kind])
        base[kind] = [stats(*shuffle(CpB, EB, kind, rng)) for _ in range(REPS)]
    vals = {"real": real, "base": base}
    for var in ("I(C'_A;C'_B)", "I(m_A;m_B)", "I(s_A;s_B)"):
        mi, n, dof = real[var]
        for kind in ("S0", "S2", "S3"):
            reps = [b[var][0] for b in base[kind]]
            net = mi - float(np.mean(reps))
            rows.add("RX.1", var, mi, n=n, bias=est.analytic_bias(dof, n), baseline=kind, reps=reps,
                     sig=mi_significant(net, dof, n, reps),
                     note="strict bit/weight vs S3" if (kind == "S3" and var == "I(C'_A;C'_B)") else "")
    for var in ("corr_v", "corr_absv", "corr_w"):
        for kind in ("S0", "S2", "S3"):
            reps = [b[var] for b in base[kind]]
            net = real[var] - float(np.mean(reps))
            rows.add("RX.2", var, real[var], n=N, baseline=kind, reps=reps,
                     sig=abs(net) > 5 / math.sqrt(N), note="threshold 5/sqrt(N)")
    # RX.3 scales at the same block position
    ea, eb = EA.ravel(), EB.ravel()
    _, ia = np.unique(ea, return_inverse=True)
    ub, ib = np.unique(eb, return_inverse=True)
    kb = len(ub)

    def emi(ibx):
        return est.mi_table(np.bincount(ia * kb + ibx, minlength=(ia.max() + 1) * kb)
                            .reshape(-1, kb))

    mi, n, dof = emi(ib)
    vals["scale_mi"] = mi
    for kind in ("whole", "rows"):
        rng = item_rng(seed, "RX", 10 + (kind == "rows"))
        reps = []
        for _ in range(REPS):
            if kind == "whole":
                x = rng.permutation(ib)
            else:
                x = ib.reshape(EB.shape)[rng.permutation(EB.shape[0])].ravel()
            reps.append(emi(x)[0])
        net = mi - float(np.mean(reps))
        rows.add("RX.3", "I(E_A;E_B)", mi, n=n, bias=est.analytic_bias(dof, n),
                 baseline="scale permutation (whole)" if kind == "whole" else "scale row-order permutation",
                 reps=reps, sig=mi_significant(net, dof, n, reps), note="bit per scale")
    return vals


# ----------------------------------------------------------------------------- R5 / R8.1

def _shares(s):
    e = s * s
    tot = e.sum()
    return e / tot


def _kurt(M, axis):
    x = M - M.mean(axis=axis, keepdims=True)
    m2 = (x * x).mean(axis=axis)
    m4 = (x ** 4).mean(axis=axis)
    return m4 / (m2 * m2)


def _svd(X, vectors):
    if vectors:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        k = min(TOP_VEC, len(s))
        return s, U[:, :k].copy(), Vt[:k].copy()
    return np.linalg.svd(X, compute_uv=False), None, None


def holdout_lowrank(X, rng, ks=R52_K):
    """Section 5.3 R5.2: quadrant hold-out low-rank prediction. Returns R2 per k (mean of the 4
    quadrant choices) and a function giving the assembled held-out prediction for one k."""
    R, W = X.shape
    pr, pc = rng.permutation(R), rng.permutation(W)
    R1, R2 = np.sort(pr[:R // 2]), np.sort(pr[R // 2:])
    C1, C2 = np.sort(pc[:W // 2]), np.sort(pc[W // 2:])
    kmax = min(max(ks), min(len(R1), len(C1)))
    quads = []
    r2 = {k: [] for k in ks}
    for Ra, Ro in ((R2, R1), (R1, R2)):
        for Cb, Co in ((C2, C1), (C1, C2)):
            A = X[np.ix_(Ro, Co)]
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            L = X[np.ix_(Ra, Co)] @ Vt[:kmax].T
            Rm = U[:, :kmax].T @ X[np.ix_(Ro, Cb)]
            Y = X[np.ix_(Ra, Cb)]
            yy = float((Y * Y).sum())
            for k in ks:
                kk = min(k, kmax)
                Yh = (L[:, :kk] / s[:kk]) @ Rm[:kk]
                r2[k].append(1.0 - float(((Y - Yh) ** 2).sum()) / yy)
            quads.append((Ra, Cb, L, s[:kmax], Rm))
    r2 = {k: float(np.mean(v)) for k, v in r2.items()}

    def predict(k):
        P = np.empty_like(X)
        for Ra, Cb, L, s, Rm in quads:
            P[np.ix_(Ra, Cb)] = (L[:, :k] / s[:k]) @ Rm[:k]
        return P

    return r2, predict


def r5_matrix(Cp, E, rows, seed, domains, H_E):
    """R5.1, R5.2, R5.3 and R8.1 for one matrix."""
    R, W = Cp.shape
    N = Cp.size
    rng4 = item_rng(seed, "S4")
    S4 = [shuffle(Cp, E, "S4", rng4)[0] for _ in range(2)]
    S0 = shuffle(Cp, E, "S0", item_rng(seed, "R5", 0))[0]
    vals, spectra = {}, []
    for dom in domains:
        Xr = values(Cp, E, dom)
        mu = Xr.mean()
        Xr = Xr - mu
        s_r, U_r, V_r = _svd(Xr, True)
        s4, k4l, k4r = [], [], []
        X4 = []
        for C4 in S4:
            X = values(C4, E, dom)
            X = X - X.mean()
            X4.append(X)
            s, U, V = _svd(X, True)
            s4.append(s)
            k4l.append(_kurt(U, 0))
            k4r.append(_kurt(V, 1))
        sp = [("real", 0, s_r)] + [("S4", i + 1, s) for i, s in enumerate(s4)]
        if dom == "v":
            X0 = values(S0, E, dom)
            sp.append(("S0", 1, _svd(X0 - X0.mean(), False)[0]))
        for var, rep, s in sp:
            spectra.append({"domain": dom, "variant": var, "rep": rep,
                            "sigma_norm": s / math.sqrt(float((s * s).sum()))})
        sh = {var + str(rep): np.cumsum(_shares(s)) for var, rep, s in sp}

        def eff(s):
            p = _shares(s)
            p = p[p > 0]
            return float(math.exp(-(p * np.log(p)).sum())), float((s * s).sum() / s[0] ** 2)

        for var, rep, s in sp:
            er, sr = eff(s)
            rows.add("R5.1", "effective_rank", er, domain=dom, group=var, subset=f"rep{rep}",
                     n=len(s))
            rows.add("R5.1", "stable_rank", sr, domain=dom, group=var, subset=f"rep{rep}", n=len(s))
            for k in R5_SHARE_K:
                if k <= len(s):
                    rows.add("R5.1", "energy_share_topk", sh[var + str(rep)][k - 1], domain=dom,
                             group=var, subset=f"rep{rep}", k=k)
        thr = 1.02 * float(np.mean([s[0] for s in s4]))
        n_ex = int((s_r > thr).sum())
        if n_ex:
            ex_e = float(sh["real0"][n_ex - 1] - np.mean([sh["S41"][n_ex - 1], sh["S42"][n_ex - 1]]))
        else:
            ex_e = 0.0
        rows.add("R5.1", "exceed_layers", n_ex, domain=dom, n=len(s_r),
                 note=f"sigma > 1.02 x mean S4 max sigma ({thr:.6g})")
        rows.add("R5.1", "exceed_energy", ex_e, domain=dom, k=n_ex,
                 note="share of the exceeding layers minus S4 share of as many layers")
        # R8.1 kurtosis
        # Two thresholds: the spec's 2 x S4 max, and 2 x S4 median (adjusted: S4 keeps every
        # position's magnitude, so rows/columns made heavy by a sparse factor also localise some S4
        # noise vectors and make the S4 max unbounded; see the completion section of spec 003).
        kl, kr = _kurt(U_r, 0), _kurt(V_r, 1)
        r81 = {}
        for side, kv, k4 in (("left", kl, k4l), ("right", kr, k4r)):
            s4all = np.concatenate(k4)
            t_max, t_med = 2 * float(s4all.max()), 2 * float(np.median(s4all))
            r81.update({f"{side}_exceed_spec": int((kv > t_max).sum()),
                        f"{side}_exceed": int((kv > t_med).sum()),
                        f"{side}_median": float(np.median(kv)), f"{side}_max": float(kv.max()),
                        f"{side}_thr_spec": t_max, f"{side}_thr": t_med,
                        f"{side}_s4_max": float(s4all.max()), f"{side}_s4_median": float(np.median(s4all))})
            rows.add("R8.1", f"kurtosis_{side}_median", float(np.median(kv)), domain=dom,
                     n=len(kv), baseline="S4", reps=[float(np.median(x)) for x in k4],
                     note="2 replicates (median per replicate)")
            rows.add("R8.1", f"kurtosis_{side}_max", float(kv.max()), domain=dom, n=len(kv),
                     baseline="S4", reps=[float(np.max(x)) for x in k4])
            rows.add("R8.1", f"kurtosis_{side}_exceed_layers_spec", int((kv > t_max).sum()),
                     domain=dom, n=len(kv),
                     note=f"spec rule: kurtosis > 2 x S4 max over top-{TOP_VEC} ({t_max:.4g})")
            rows.add("R8.1", f"kurtosis_{side}_exceed_layers", int((kv > t_med).sum()),
                     domain=dom, n=len(kv),
                     note=f"adjusted rule: kurtosis > 2 x S4 median over top-{TOP_VEC} ({t_med:.4g})")
        # R5.2 hold-out
        r2, predict = holdout_lowrank(Xr, item_rng(seed, "R5.2", 0))
        r2s4, _ = holdout_lowrank(X4[0], item_rng(seed, "R5.2", 0))
        for k in R52_K:
            approx = -0.5 * math.log2(1 - r2[k]) if 0 < r2[k] < 1 else float("nan")
            rows.add("R5.2", "holdout_R2", r2[k], domain=dom, k=k, n=N, baseline="S4",
                     reps=[r2s4[k]], note="1 replicate")
            rows.add("R5.2", "approx_bits", approx, domain=dom, k=k,
                     note="approximate: -1/2 log2(1-R2); blank when R2 <= 0")
            rows.add("R5.2", "factor_cost_bits", k * (R + W) * 16 / N, domain=dom, k=k)
        kbest = max(R52_K, key=lambda k: r2[k])
        r53 = None
        if r2[kbest] > 0.001:
            P = predict(kbest) + mu
            if dom == "w":
                P = P / np.repeat(np.exp2(E.astype(np.float64) - 127.0), B, axis=1)
            edges = np.quantile(P, np.arange(1, 8) / 8)
            bins = np.searchsorted(edges, P, side="right").ravel()
            _, Einv = np.unique(E.ravel(), return_inverse=True)
            nE = int(Einv.max()) + 1
            ctx = np.repeat(Einv, B)
            half = np.repeat(np.arange(R) % 2, W)
            h_e, h0, n = est.holdout_cross_entropy(ctx, Cp.ravel(), half, nE)
            h_eb, _, _ = est.holdout_cross_entropy(ctx * 8 + bins, Cp.ravel(), half, nE * 8)
            gain = h_e - h_eb
            cost = kbest * (R + W) * 16 / N
            approx = -0.5 * math.log2(1 - r2[kbest])
            r53 = {"k": kbest, "gain": gain, "cost": cost, "net": gain - cost, "approx": approx,
                   "ratio": gain / approx, "R2": r2[kbest]}
            rows.add("R5.3", "holdout_gain_bits", gain, domain=dom, k=kbest, n=n,
                     note="strict: holdout H(C'|E) - H(C'|E, octile of holdout prediction)")
            rows.add("R5.3", "net_bits", gain - cost, domain=dom, k=kbest, n=n,
                     note=f"strict: gain - factor cost {cost:.6g}")
            rows.add("R5.3", "gain_over_approx", gain / approx, domain=dom, k=kbest)
        else:
            rows.add("R5.3", "net_bits", None, domain=dom,
                     note=f"not run: max hold-out R2 = {r2[kbest]:.3g} <= 0.001 (k={kbest})")
        vals[dom] = {"exceed": n_ex, "exceed_energy": ex_e, "r2": r2, "r2_S4": r2s4,
                     "kbest": kbest, "r53": r53, "r81": r81, "s4_max": thr / 1.02,
                     "sigma1": float(s_r[0])}
    return vals, spectra


# ----------------------------------------------------------------------------- R7.1

def min_payback(n_cand, H_E):
    if n_cand <= 0:
        return None
    lc = math.log2(n_cand)
    for m in range(1, B + 1):
        if m * H_E > lc + math.log2(math.comb(B, m)):
            return m
    return None


def r71_matrix(Cp, E, rows, seed, H_E):
    rng = item_rng(seed, "R7.1")
    flat = E.ravel()
    nblk = flat.size
    queries = rng.choice(nblk, size=min(N_QUERY, nblk), replace=False)
    variants = {"real": Cp, "S0E": shuffle(Cp, E, "S0E", rng)[0], "S2": shuffle(Cp, E, "S2", rng)[0]}
    groups = {}
    for e in np.unique(flat[queries]):
        groups[e] = np.nonzero(flat == e)[0]
    ncand = np.array([len(groups[flat[q]]) - 1 for q in queries])
    mstar = [min_payback(n, H_E) for n in ncand]
    best = {}
    for name, X in variants.items():
        blocks = X.reshape(-1, B)
        out = np.full(len(queries), -1, dtype=np.int64)
        for e, gidx in groups.items():
            G = blocks[gidx]
            qi = np.nonzero(flat[queries] == e)[0]
            for c0 in range(0, len(qi), 8):
                sel = qi[c0:c0 + 8]
                q = queries[sel]
                eq = (G[None, :, :] == blocks[q][:, None, :]).sum(axis=2, dtype=np.int16)
                eq[np.arange(len(sel)), np.searchsorted(gidx, q)] = -1
                out[sel] = eq.max(axis=1)
        best[name] = out
    ok = ncand > 0
    p99_s0e = float(np.percentile(best["S0E"][ok], 99))
    vals = {"best": best, "ncand": ncand, "mstar": mstar}
    for name, b in best.items():
        bb = b[ok]
        for var, val in (("best_median", np.median(bb)), ("best_p90", np.percentile(bb, 90)),
                         ("best_p99", np.percentile(bb, 99)), ("best_max", bb.max())):
            rows.add("R7.1", var, float(val), group=name, n=len(bb))
        fr = float((bb > p99_s0e + 2).mean())
        fm = float(np.mean([bi >= m for bi, m in zip(bb, np.array(mstar, dtype=object)[ok])
                            if m is not None])) if any(m is not None for m in mstar) else float("nan")
        f18 = float((bb >= 18).mean())
        rows.add("R7.1", "frac_best_gt_S0E_p99+2", fr, group=name, n=len(bb),
                 note=f"S0E p99 = {p99_s0e:g}")
        rows.add("R7.1", "frac_best_ge_mstar", fm, group=name, n=len(bb),
                 note="m* = min m with m*H(C'|E) > log2(#candidates) + log2 C(32,m)")
        rows.add("R7.1", "frac_best_ge_18", f18, group=name, n=len(bb))
        h = np.bincount(bb, minlength=B + 1)
        for v in range(B + 1):
            if h[v]:
                rows.add("R7.1", "best_hist", int(h[v]), group=name, k=v, n=len(bb))
        vals[name] = {"median": float(np.median(bb)), "frac18": f18, "frac_gt": fr, "frac_m": fm}
    ms = [m for m in mstar if m is not None]
    rows.add("R7.1", "mstar_median", float(np.median(ms)) if ms else None, n=len(ms),
             note=f"H(C'|E) = {H_E:.6g}")
    return vals


# ----------------------------------------------------------------------------- R7.2 / R8.2

def main_sample(E, rng):
    flat = E.ravel()
    u, c = np.unique(flat, return_counts=True)
    main = u[np.argmax(c)]
    idx = np.nonzero(flat == main)[0]
    pick = rng.choice(idx, size=min(R72_SAMPLE, len(idx)), replace=False)
    return pick, int(main)


def _sqdist_min(X, Cn, chunk=8192):
    cc = (Cn * Cn).sum(axis=1)
    lab = np.empty(len(X), dtype=np.int64)
    dmin = np.empty(len(X))
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        d = cc[None, :] - 2.0 * (x @ Cn.T)
        lab[i:i + chunk] = np.argmin(d, axis=1)
        dmin[i:i + chunk] = d[np.arange(len(x)), lab[i:i + chunk]] + (x * x).sum(axis=1)
    return lab, np.maximum(dmin, 0.0)


def kmeans(X, K, rng, iters=KM_ITERS):
    n, d = X.shape
    Cn = X[rng.choice(n, size=K, replace=False)].copy()
    for _ in range(iters):
        lab, _ = _sqdist_min(X, Cn)
        cnt = np.bincount(lab, minlength=K)
        sums = np.stack([np.bincount(lab, weights=X[:, j], minlength=K) for j in range(d)], axis=1)
        nz = cnt > 0
        Cn[nz] = sums[nz] / cnt[nz, None]
        for kk in np.nonzero(~nz)[0]:
            Cn[kk] = X[rng.integers(n)]
    return Cn


def r72_variant(codes, rng_seed_parts):
    """k-means curves and strict gains for one variant. codes: (n, 32) C' of sampled blocks."""
    n = len(codes)
    nf = n // 2
    res = {}
    for dom in ("v", "absv"):
        X = E2M1[codes] if dom == "v" else np.abs(E2M1[codes])
        Xf, Xe = X[:nf], X[nf:]
        den = float(((Xe - Xf.mean(axis=0)) ** 2).sum())
        res[dom] = {}
        for K in KM_K:
            rng = np.random.default_rng([SEED, *rng_seed_parts, K, int(dom == "v")])
            Cn = kmeans(Xf, K, rng)
            lab_e, dm = _sqdist_min(Xe, Cn)
            out = {"ratio": float(dm.sum()) / den}
            if K <= KM_STRICT_MAX:
                tgt = codes[nf:]
                half = np.repeat(np.arange(len(tgt)) % 2, B)
                ctx = np.repeat(lab_e, B)
                h_c, h0, nn = est.holdout_cross_entropy(ctx, tgt.ravel(), half, K)
                pos = np.tile(np.arange(B), len(tgt))
                h_cp, _, _ = est.holdout_cross_entropy(ctx * B + pos, tgt.ravel(), half, K * B)
                cost = math.log2(K) / B
                out.update(gain_cls=h0 - h_c, gain_clsp=h0 - h_cp, net_cls=h0 - h_c - cost,
                           net_clsp=h0 - h_cp - cost, n=nn)
            res[dom][K] = out
    return res


def omp(X, D, s):
    n = len(X)
    sel = np.zeros((n, s), dtype=np.int64)
    resid = X.copy()
    coef = None
    ar = np.arange(n)
    for t in range(s):
        c = np.abs(resid @ D.T)
        for u in range(t):
            c[ar, sel[:, u]] = -1.0
        sel[:, t] = np.argmax(c, axis=1)
        Ds = D[sel[:, :t + 1]]
        G = Ds @ Ds.transpose(0, 2, 1) + 1e-10 * np.eye(t + 1)
        b = Ds @ X[:, :, None]
        coef = np.linalg.solve(G, b)[:, :, 0]
        resid = X - (coef[:, :, None] * Ds).sum(axis=1)
    return sel, coef, resid


def dict_learn(Xf, rng):
    n = len(Xf)
    D = Xf[rng.choice(n, size=DICT_ATOMS, replace=False)].copy()

    def renorm(D):
        nrm = np.sqrt((D * D).sum(axis=1))
        for i in np.nonzero(nrm <= 0)[0]:
            while True:
                x = Xf[rng.integers(n)]
                if (x * x).sum() > 0:
                    D[i] = x
                    nrm[i] = math.sqrt(float((x * x).sum()))
                    break
        return D / nrm[:, None]

    D = renorm(D)
    for _ in range(DICT_ITERS):
        sel, coef, _ = omp(Xf, D, 2)
        Cm = np.zeros((n, DICT_ATOMS))
        np.add.at(Cm, (np.arange(n)[:, None], sel), coef)
        D = np.linalg.solve(Cm.T @ Cm + 1e-6 * np.eye(DICT_ATOMS), Cm.T @ Xf)
        D = renorm(D)
    return D


def r82_variant(codes, rng_seed_parts):
    n = len(codes)
    nf = n // 2
    X = E2M1[codes]
    Xf, Xe = X[:nf], X[nf:]
    rng = np.random.default_rng([SEED, *rng_seed_parts])
    D = dict_learn(Xf, rng)
    ee = float((Xe * Xe).sum())
    out = {"dict": {}, "pca": {}}
    for s in DICT_S:
        _, _, resid = omp(Xe, D, s)
        out["dict"][s] = 1.0 - float((resid * resid).sum()) / ee
    lam, V = np.linalg.eigh(Xf.T @ Xf / nf)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    for s in DICT_S:
        Vs = V[:, :s]
        Xh = Xe @ Vs @ Vs.T
        out["pca"][s] = 1.0 - float(((Xe - Xh) ** 2).sum()) / ee
    out["eig"] = lam
    return out


def r72_r82_matrix(Cp, E, rows, seed):
    rng = item_rng(seed, "sample")
    pick, main = main_sample(E, rng)
    variants = {"real": Cp}
    for kind in ("S0E", "S2"):
        for rep in (1, 2):
            variants[f"{kind}_{rep}"] = shuffle(Cp, E, kind, item_rng(seed, "R7.2", rep, int(kind == "S2")))[0]
    r72, r82 = {}, {}
    for vi, (name, X) in enumerate(variants.items()):
        codes = X.reshape(-1, B)[pick]
        r72[name] = r72_variant(codes, (seed, _ITEM["R7.2"], vi))
        r82[name] = r82_variant(codes, (seed, _ITEM["R8.2"], vi))
    nb = len(pick)
    note = f"main scale E={main}, {nb} sampled blocks (half fit, half evaluate)"
    for dom in ("v", "absv"):
        for K in KM_K:
            for var in ("ratio", "net_cls", "net_clsp", "gain_cls", "gain_clsp"):
                if var not in r72["real"][dom][K]:
                    continue
                val = r72["real"][dom][K][var]
                for kind in ("S0E", "S2"):
                    reps = [r72[f"{kind}_{i}"][dom][K][var] for i in (1, 2)]
                    rows.add("R7.2", var, val, domain=dom, k=K, n=nb, baseline=kind, reps=reps,
                             note=("strict bit/weight; " if var.startswith("net") else "") + note)
    for s in DICT_S:
        for meth in ("dict", "pca"):
            val = r82["real"][meth][s]
            for kind in ("S0E", "S2"):
                reps = [r82[f"{kind}_{i}"][meth][s] for i in (1, 2)]
                sig = None
                if meth == "dict" and kind == "S0E":
                    diff = val - float(np.mean(reps))
                    sig = bool(diff > 0.01 and diff > 5 * abs(reps[0] - reps[1]))
                rows.add("R8.2", f"R2_{meth}", val, domain="v", k=s, n=nb, baseline=kind, reps=reps,
                         sig=sig, note=note)
    for i, lam in enumerate(r82["real"]["eig"]):
        rows.add("R8.2", "pca_eigenvalue", float(lam), domain="v", k=i + 1, n=nb // 2,
                 baseline="S0E", reps=[float(r82[f"S0E_{j}"]["eig"][i]) for j in (1, 2)],
                 note="eigenvalues of fit-half X^T X / n (not centred)")
    return {"r72": r72, "r82": r82, "main": main, "n": nb}


# ----------------------------------------------------------------------------- R7.3

def _normalize_rows(X):
    Z = X - X.mean(axis=1, keepdims=True)
    nrm = np.sqrt((Z * Z).sum(axis=1))
    Z = Z / np.where(nrm > 0, nrm, 1.0)[:, None]
    return Z.astype(np.float32), int((nrm <= 0).sum())


def _maxcorr(ZA, ZB, within, chunk=1024):
    """For every row of ZB: max |corr| over rows of ZA (excluding itself when within). Also the
    number of pairs with |corr| > 0.5 (i<j when within) and the strongest such pairs."""
    nB = len(ZB)
    best = np.zeros(nB)
    count = 0
    ex = []
    for i0 in range(0, nB, chunk):
        c = ZB[i0:i0 + chunk] @ ZA.T
        a = np.abs(c)
        if within:
            ii = np.arange(i0, min(i0 + chunk, nB))
            a[ii - i0, ii] = 0.0
            mask = a > 0.5
            mask &= np.arange(len(ZA))[None, :] > ii[:, None]
        else:
            mask = a > 0.5
        best[i0:i0 + chunk] = a.max(axis=1)
        nm = int(mask.sum())
        count += nm
        if nm:
            bi, aj = np.nonzero(mask)
            top = np.argsort(-a[bi, aj])[:MAX_EXAMPLES]
            ex.extend((int(i0 + bi[t]), int(aj[t]), float(c[bi[t], aj[t]])) for t in top)
    ex.sort(key=lambda t: -abs(t[2]))
    return best, count, ex[:MAX_EXAMPLES]


def r73(XA_by_dom, XB_by_dom, XB4_by_dom, rows, within, meta_ex):
    """XB4: S4 copy of B (or of the matrix itself when within)."""
    vals, exs = {}, []
    for dom in XB_by_dom:
        vals[dom] = {}
        for axis in ("row", "col"):
            tr = (lambda X: X) if axis == "row" else (lambda X: X.T)
            ZA, za0 = _normalize_rows(tr(XA_by_dom[dom]))
            ZB, zb0 = (ZA, za0) if within else _normalize_rows(tr(XB_by_dom[dom]))
            Z4, _ = _normalize_rows(tr(XB4_by_dom[dom]))
            best, cnt, ex = _maxcorr(ZA, ZB, within)
            ZA4 = Z4 if within else ZA
            best4, cnt4, _ = _maxcorr(ZA4, Z4, within)
            m4 = float(best4.max())
            nex = int((best > 1.2 * m4).sum())
            note = f"zero-variance {axis}s: {zb0}" if zb0 else ""
            for var, val, v4 in (("maxcorr_median", np.median(best), np.median(best4)),
                                 ("maxcorr_p99", np.percentile(best, 99), np.percentile(best4, 99)),
                                 ("maxcorr_max", best.max(), m4)):
                rows.add("R7.3", var, float(val), domain=dom, group=axis, n=len(best),
                         baseline="S4", reps=[float(v4)], note="1 replicate" + ("; " + note if note else ""))
            rows.add("R7.3", "n_exceed_1.2xS4max", nex, domain=dom, group=axis, n=len(best),
                     note=f"S4 max {m4:.4g}")
            rows.add("R7.3", "pairs_abs_corr_gt_0.5", cnt, domain=dom, group=axis, baseline="S4",
                     reps=[cnt4], note="1 replicate")
            vals[dom][axis] = {"best": best, "n_exceed": nex, "s4max": m4, "count": cnt}
            for i, j, c in ex:
                exs.append(dict(meta_ex, domain=dom, axis=axis, index_b=i, index_a=j, corr=c))
    return vals, exs


# ----------------------------------------------------------------------------- top level

def measure_matrix(C, E, meta, seed, domains=("v", "w")):
    """All per-matrix items. C: (R, W) uint8 codewords; E: (R, W/B) uint8 scale bytes."""
    Cp = merge_zero(C)
    rows = Rows(dict(meta, matrix_b="", pair_group=""))
    H_E = h_cond_e(Cp, E)
    vals = {"H_E": H_E}
    v6, ex6, pool = r6_matrix(Cp, E, meta, rows, seed, H_E)
    vals.update(v6)
    vals["r5"], spectra = r5_matrix(Cp, E, rows, seed, domains, H_E)
    vals["r71"] = r71_matrix(Cp, E, rows, seed, H_E)
    vals["r72r82"] = r72_r82_matrix(Cp, E, rows, seed)
    XS = {d: values(Cp, E, d) for d in domains}
    C4 = shuffle(Cp, E, "S4", item_rng(seed, "R7.3"))[0]
    X4 = {d: values(C4, E, d) for d in domains}
    vals["r73"], ex73 = r73(XS, XS, X4, rows, True, dict(meta, matrix_b="", pair_group=""))
    return {"rows": rows.rows, "vals": vals, "examples6": ex6, "examples73": ex73,
            "spectra": spectra, "pool": pool}


def measure_pair(CA, EA, CB, EB, meta_a, meta_b, group, seed, domains=("v", "w")):
    CpA, CpB = merge_zero(CA), merge_zero(CB)
    m = dict(meta_a, matrix_b=meta_b["matrix"], pair_group=group)
    rows = Rows(m)
    vals = {}
    vals["r63"] = r63_pair(CpA, CpB, EB, rows, seed)
    vals["rx"] = rx_pair(CpA, EA, CpB, EB, rows, seed)
    XA = {d: values(CpA, EA, d) for d in domains}
    XB = {d: values(CpB, EB, d) for d in domains}
    C4 = shuffle(CpB, EB, "S4", item_rng(seed, "R7.3"))[0]
    X4 = {d: values(C4, EB, d) for d in domains}
    vals["r73"], ex73 = r73(XA, XB, X4, rows, False, m)
    return {"rows": rows.rows, "vals": vals, "examples73": ex73}
