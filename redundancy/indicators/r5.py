"""R5: rows are mixtures of a few patterns. Energy share of each singular value (squared, of the
mean-removed `values`), largest first. Baseline: the adapter's special baseline when given, else a
whole shuffle (a random spectrum). Break-even per layer: share > 1 - 2^(-2 (R+W) b / N)."""
import numpy as np

from .. import baselines, breakeven
from .common import (REPS, applicable, band, exceed_stats, med_baseline, mean_sd, nanlist,
                     not_applicable_summary, rng, verdict)

FIGURES = ("R5",)


def shares(V):
    X = V.astype(np.float64)
    X = X - X.mean()
    s = np.linalg.svd(X, compute_uv=False)
    e = s * s
    return e / e.sum()


def compute(entry, cfg, seed):
    if entry.values is None:
        return {"R5": {"applicable": False, "reason": "no values"}}
    R, W = entry.shape
    show = min(cfg.get("show", 64), min(R, W))
    bname = baselines.pick(entry, "R5", cfg.get("baseline", "whole"))
    real = shares(entry.values)
    reps = []
    for r in range(REPS):
        _, vs = baselines.run(bname, entry, rng(seed, "R5", entry.name, r))
        reps.append(shares(vs)[:show])
    mu, sd = mean_sd(reps)
    return {"R5": {"share": nanlist(real[:show]), "base_mean": nanlist(mu), "base_sd": nanlist(sd),
                   "base_reps": [nanlist(x) for x in reps], "n_layers": int(len(real)),
                   "baseline": bname, "breakeven": breakeven.layer_share(R, W, cfg.get("b", 16), entry.N)}}


def aggregate(fig, items, cfg, seed):
    items = applicable(items)
    if not items:
        return not_applicable_summary(items)
    amp = cfg.get("min_amp", 0.0005)
    n = min(len(r["share"]) for _, r in items)
    cut = [(nm, {k: (v[:n] if isinstance(v, list) else v) for k, v in r.items()}) for nm, r in items]
    Rm, Mu, Sd = med_baseline(cut, "share", "base_mean", "base_sd")
    ok, _ = exceed_stats(Rm, Mu, Sd, amp, curve=True)
    be = float(np.median([r["breakeven"] for _, r in items]))
    passes = bool(np.any(ok & (Rm > be)))
    return {"summary": {"layers": list(range(1, n + 1)), "real": band([r["share"] for _, r in cut]),
                        "base": band([r["base_mean"] for _, r in cut]), "breakeven": be,
                        "baseline": items[0][1]["baseline"], "n_entries": len(items),
                        "top": float(Rm[0]), "top_base": float(Mu[0])},
            "verdict": verdict(bool(ok.any()), passes),
            "detail": {"exceed_layers": [int(i + 1) for i in np.flatnonzero(ok)], "min_amp": amp},
            "key": f"top share = {Rm[0]:.4g} (baseline {Mu[0]:.4g}), break-even = {be:.3g}"}
