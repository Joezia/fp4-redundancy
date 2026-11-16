"""R6: a stretch of content recurs verbatim. For every window length k: the fraction of in-row
windows of length k that repeat an earlier window anywhere in the entry (windows - distinct windows,
counted exactly), on symbols with equivalent symbols merged. Baseline: shuffle inside units (whole
shuffle without units). Break-even: repeat length > log2(N) / H."""
import math

import numpy as np

from .. import baselines, breakeven
from ..dataset import entropy
from .common import (REPS, applicable, band, exceed_stats, med_baseline, mean_sd, nanlist,
                     not_applicable_summary, rng, verdict)

FIGURES = ("R6",)
GOLD = np.uint64(0x9E3779B97F4A7C15)
M1 = np.uint64(0xBF58476D1CE4E5B9)


def _packed(X, bits):
    """P[r, j] = symbols j .. j+S-1 of row r packed `bits` each (symbol j lowest), zero past the end."""
    R, W = X.shape
    S = 64 // bits
    pad = np.zeros((R, W + S), dtype=np.uint64)
    pad[:, :W] = X
    P = np.zeros((R, W), dtype=np.uint64)
    for i in range(S):
        P |= pad[:, i:i + W] << np.uint64(bits * i)
    return P, S


def _words(P, S, bits, k):
    R, W = P.shape
    n = W - k + 1
    out = []
    for t in range(-(-k // S)):
        x = P[:, S * t:S * t + n]
        rem = k - S * t
        if rem < S:
            x = x & np.uint64((1 << (bits * rem)) - 1)
        out.append(np.ascontiguousarray(x).ravel())
    return out


def distinct(words):
    """Exact number of distinct multi-word keys (hash, then verify equal-hash groups)."""
    if len(words) == 1:
        s = np.sort(words[0])
        return int(1 + np.count_nonzero(s[1:] != s[:-1])) if len(s) else 0
    with np.errstate(over="ignore"):
        h = np.zeros_like(words[0])
        for w in words:
            h = (h ^ w) * GOLD
            h ^= h >> np.uint64(31)
        h *= M1
        h ^= h >> np.uint64(29)
    o = np.argsort(h, kind="stable")
    hs = h[o]
    same = hs[1:] == hs[:-1]
    n_hash = int(1 + np.count_nonzero(~same))
    mark = np.zeros(len(h), dtype=bool)
    mark[1:] |= same
    mark[:-1] |= same
    cand = o[mark]
    if cand.size == 0:
        return n_hash
    hc = h[cand]
    Wc = np.stack([w[cand] for w in words], axis=1)
    n_groups_hash = len(np.unique(hc))
    n_exact = len(np.unique(np.column_stack([hc, Wc]), axis=0))
    return n_hash - n_groups_hash + n_exact


def dup_fraction(X, A, ks):
    bits = max(1, math.ceil(math.log2(A))) if A > 1 else 1
    P, S = _packed(X, bits)
    R, W = X.shape
    out, raw = [], []
    for k in ks:
        if k > W:
            out.append(float("nan"))
            raw.append(0)
            continue
        w = _words(P, S, bits, k)
        nwin = len(w[0])
        dup = nwin - distinct(w)
        out.append(dup / nwin)
        raw.append(int(dup))
    return out, raw


def compute(entry, cfg, seed):
    ks = [int(k) for k in cfg.get("ks", (8, 10, 12, 14, 16, 20, 24, 32, 48, 64))]
    X = entry.canonical()
    bname = baselines.pick(entry, "R6", cfg.get("baseline", "unit"))
    real, dup = dup_fraction(X, entry.A, ks)
    reps = []
    for r in range(REPS):
        s, _ = baselines.run(bname, entry, rng(seed, "R6", entry.name, r), symbols=X, values=None)
        reps.append(dup_fraction(s, entry.A, ks)[0])
    mu, sd = mean_sd(reps)
    H = entropy(np.bincount(X.ravel(), minlength=entry.A))
    kstar = max([k for k, d in zip(ks, dup) if d > 0], default=None)
    return {"R6": {"ks": ks, "frac": nanlist(real), "dup": dup, "base_mean": nanlist(mu),
                   "base_sd": nanlist(sd), "base_reps": [nanlist(x) for x in reps], "baseline": bname,
                   "H": H, "Kstar": kstar, "breakeven": breakeven.repeat_length(entry.N, H)}}


def aggregate(fig, items, cfg, seed):
    items = applicable(items)
    if not items:
        return not_applicable_summary(items)
    amp = cfg.get("min_amp", 1e-4)
    ks = items[0][1]["ks"]
    Rm, Mu, Sd = med_baseline(items, "frac", "base_mean", "base_sd")
    ok, _ = exceed_stats(Rm, Mu, Sd, amp, curve=True)
    be = float(np.median([r["breakeven"] for _, r in items]))
    passes = bool(np.any(ok & (np.asarray(ks) > be)))
    kst = [r["Kstar"] for _, r in items if r["Kstar"] is not None]
    return {"summary": {"ks": ks, "real": band([r["frac"] for _, r in items]),
                        "base": band([r["base_mean"] for _, r in items]), "breakeven": be,
                        "Kstar_median": float(np.median(kst)) if kst else None,
                        "baseline": items[0][1]["baseline"], "n_entries": len(items)},
            "verdict": verdict(bool(ok.any()), passes),
            "detail": {"exceed_k": [int(ks[i]) for i in np.flatnonzero(ok)], "min_amp": amp},
            "key": (f"K* = {np.median(kst):g}" if kst else "K* = none") + f", L* = {be:.1f}"}
