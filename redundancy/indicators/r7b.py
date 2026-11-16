"""R7b: segments resemble each other (groups). k-means on the `values` of sampled units (first half
fit, second half evaluate); distortion ratio = held-out squared distance to the nearest prototype /
held-out squared distance to the fit-half mean. Baseline: shuffle within `known` groups, whole
pipeline repeated. Break-even: ratio < 2^(-2 log2 K / L)."""
import numpy as np

from .. import baselines, breakeven
from .common import (REPS, applicable, band, exceed_stats, med_baseline, mean_sd, nanlist,
                     not_applicable_summary, rng, verdict)
from .sample import unit_sample, unit_values

FIGURES = ("R7b",)


def nearest(X, Cn, chunk=8192):
    cc = (Cn * Cn).sum(axis=1)
    lab = np.empty(len(X), dtype=np.int64)
    dmin = np.empty(len(X))
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        d = cc[None, :] - 2.0 * (x @ Cn.T)
        lab[i:i + chunk] = np.argmin(d, axis=1)
        dmin[i:i + chunk] = d[np.arange(len(x)), lab[i:i + chunk]] + (x * x).sum(axis=1)
    return lab, np.maximum(dmin, 0.0)


def kmeans(X, K, r, iters):
    n, d = X.shape
    Cn = X[r.choice(n, size=K, replace=False)].copy()
    for _ in range(iters):
        lab, _ = nearest(X, Cn)
        cnt = np.bincount(lab, minlength=K)
        sums = np.stack([np.bincount(lab, weights=X[:, j], minlength=K) for j in range(d)], axis=1)
        nz = cnt > 0
        Cn[nz] = sums[nz] / cnt[nz, None]
        for kk in np.flatnonzero(~nz):
            Cn[kk] = X[r.integers(n)]
    return Cn


def ratios(X, ks, seed, name, variant, iters):
    nf = len(X) // 2
    Xf, Xe = X[:nf], X[nf:]
    den = float(((Xe - Xf.mean(axis=0)) ** 2).sum())
    out = []
    for K in ks:
        if K > nf:
            out.append(float("nan"))
            continue
        Cn = kmeans(Xf, K, rng(seed, "R7b", name, variant, K), iters)
        out.append(float(nearest(Xe, Cn)[1].sum()) / den if den > 0 else float("nan"))
    return out


def compute(entry, cfg, seed):
    if entry.values is None or not entry.unit_len:
        return {"R7b": {"applicable": False, "reason": "no values" if entry.values is None else "no unit"}}
    ks = [int(k) for k in cfg.get("ks", (2, 4, 8, 16, 32, 64, 128, 256))]
    iters = cfg.get("iters", 20)
    L = entry.unit_len
    pick, label = unit_sample(entry, rng(seed, "R7b", entry.name, "sample"), cfg.get("sample", 100000))
    real = ratios(unit_values(entry.values, L, pick), ks, seed, entry.name, "real", iters)
    bname = baselines.pick(entry, "R7b", cfg.get("baseline", "known"))
    reps = []
    for i in range(REPS):
        _, v = baselines.run(bname, entry, rng(seed, "R7b", entry.name, "shuffle", i), symbols=entry.symbols)
        reps.append(ratios(unit_values(v, L, pick), ks, seed, entry.name, f"base{i}", iters))
    mu, sd = mean_sd(reps)
    return {"R7b": {"ks": ks, "ratio": nanlist(real), "base_mean": nanlist(mu), "base_sd": nanlist(sd),
                    "base_reps": [nanlist(x) for x in reps], "baseline": bname, "L": L,
                    "n_units": int(len(pick)), "known_label": label,
                    "breakeven": [breakeven.cluster_ratio(K, L) for K in ks]}}


def aggregate(fig, items, cfg, seed):
    items = applicable(items)
    if not items:
        return not_applicable_summary(items)
    amp = cfg.get("min_amp", 0.005)
    ks = items[0][1]["ks"]
    Rm, Mu, Sd = med_baseline(items, "ratio", "base_mean", "base_sd")
    ok, _ = exceed_stats(Rm, Mu, Sd, amp, sign=-1, curve=True)
    be = np.asarray(items[0][1]["breakeven"])
    passes = bool(np.any(ok & (Rm < be)))
    i64 = ks.index(64) if 64 in ks else 0
    return {"summary": {"ks": ks, "real": band([r["ratio"] for _, r in items]),
                        "base": band([r["base_mean"] for _, r in items]), "breakeven": be.tolist(),
                        "baseline": items[0][1]["baseline"], "n_entries": len(items)},
            "verdict": verdict(bool(ok.any()), passes),
            "detail": {"exceed_K": [int(ks[i]) for i in np.flatnonzero(ok)], "min_amp": amp},
            "key": f"ratio(K={ks[i64]}) = {Rm[i64]:.3f} (baseline {Mu[i64]:.3f}), "
                   f"break-even {be[i64]:.3f}"}
