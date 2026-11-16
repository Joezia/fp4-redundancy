"""R4: linear relation with nearby values. Autocorrelation of `values` (and of |values|) along rows,
d = 1..max_lag, with the global mean and variance. Baseline: the same after shuffling inside units
(whole shuffle without units). References: noise band ±5/sqrt(n); break-even |ρ| = 0.5 (differencing
starts to pay); upper bound ½·log2(1 / spectral flatness) of the row power spectrum."""
import math

import numpy as np

from .. import baselines, breakeven
from .common import (REPS, applicable, band, exceed_stats, med_baseline, mean_sd, nanlist,
                     not_applicable_summary, rng, verdict)

FIGURES = ("R4",)


def acf(V, D):
    x = V.astype(np.float64)
    y = x - x.mean()
    var = float((y * y).mean())
    R, W = y.shape
    if var <= 0:
        return [float("nan")] * D
    return [float((y[:, : W - d] * y[:, d:]).mean() / var) for d in range(1, D + 1)]


def flatness(V):
    y = V.astype(np.float64)
    y = y - y.mean()
    P = (np.abs(np.fft.rfft(y, axis=1)) ** 2).mean(axis=0)[1:]
    P = P[P > 0]
    if len(P) == 0:
        return float("nan")
    return float(np.exp(np.log(P).mean()) / P.mean())


def compute(entry, cfg, seed):
    if entry.values is None:
        return {"R4": {"applicable": False, "reason": "no values"}}
    R, W = entry.shape
    D = min(cfg.get("max_lag", 96), W - 1)
    bname = baselines.pick(entry, "R4", cfg.get("baseline", "unit"))
    V = entry.values
    real = {"v": acf(V, D), "abs": acf(np.abs(V), D)}
    base = {"v": [], "abs": []}
    for r in range(REPS):
        _, vs = baselines.run(bname, entry, rng(seed, "R4", entry.name, r))
        base["v"].append(acf(vs, D))
        base["abs"].append(acf(np.abs(vs), D))
    out = {"lags": list(range(1, D + 1)), "baseline": bname, "n_pairs_d1": R * (W - 1),
           "noise": 5.0 / math.sqrt(R * (W - 1)), "flatness": flatness(V)}
    out["bound"] = breakeven.flatness_bound(out["flatness"])
    for k in ("v", "abs"):
        mu, sd = mean_sd(base[k])
        out[k] = real[k]
        out[k + "_base_mean"] = nanlist(mu)
        out[k + "_base_sd"] = nanlist(sd)
    return {"R4": out}


def aggregate(fig, items, cfg, seed):
    items = applicable(items)
    if not items:
        return not_applicable_summary(items)
    amp = cfg.get("min_amp", 0.02)
    n = min(len(r["lags"]) for _, r in items)
    cut = [(nm, {k: (v[:n] if isinstance(v, list) else v) for k, v in r.items()}) for nm, r in items]
    exists, passes, det, curves = False, False, {}, {}
    for k in ("v", "abs"):
        Rm, Mu, Sd = med_baseline(cut, k, k + "_base_mean", k + "_base_sd")
        ok, _ = exceed_stats(Rm, Mu, Sd, amp, sign=0, curve=True)
        exists |= bool(ok.any())
        passes |= bool(np.any(ok & (np.abs(Rm) >= breakeven.AUTOCORR)))
        det[k] = [int(i + 1) for i in np.flatnonzero(ok)]
        curves[k] = {"real": band([r[k] for _, r in cut]), "base": band([r[k + "_base_mean"] for _, r in cut])}
    noise = float(np.median([r["noise"] for _, r in items]))
    bound = float(np.median([r["bound"] for _, r in items]))
    r1 = curves["v"]["real"]["median"][0]
    a1 = curves["abs"]["real"]["median"][0]
    return {"summary": {"lags": list(range(1, n + 1)), "curves": curves, "noise": noise,
                        "breakeven": breakeven.AUTOCORR, "bound": bound,
                        "flatness": float(np.median([r["flatness"] for _, r in items])),
                        "baseline": items[0][1]["baseline"], "n_entries": len(items)},
            "verdict": verdict(exists, passes),
            "detail": {"exceed_lags": det, "min_amp": amp},
            "key": f"ρ(1) = {r1:.3f}, |x| ρ(1) = {a1:.3f}, bound = {bound:.3g} bit"}
