"""R3: any relation with nearby symbols. Plug-in MI of the pair (x[r, j], x[r, j + d]) for
d = 1..max_lag along rows, for all pairs and (when units exist) for pairs inside one unit.
Baseline: the configured shuffle (`baseline`; extra ones in `alt_baselines` are stored for
reference), the same curves after shuffling.
Break-even: MI > (A-1)·A·b / N."""
import numpy as np

from .. import baselines, breakeven
from ..dataset import mutual_information
from .common import (REPS, applicable, band, baseline_names, exceed_stats, med_baseline, mean_sd,
                     nanlist, not_applicable_summary, stream, verdict)

FIGURES = ("R3",)


def curves(X, A, L, D):
    R, W = X.shape
    Xi = X.astype(np.int32)
    jj = np.arange(W)
    mi_all, mi_same = [], []
    for d in range(1, D + 1):
        key = Xi[:, : W - d] * A + Xi[:, d:]
        if L:
            same = (((jj[: W - d] % L) + d) < L).astype(np.int32) * (A * A)
            key = key + same[None, :]
            t = np.bincount(key.ravel(), minlength=2 * A * A).reshape(2, A, A)
            mi_all.append(mutual_information(t[0] + t[1]))
            mi_same.append(mutual_information(t[1]) if t[1].sum() else float("nan"))
        else:
            t = np.bincount(key.ravel(), minlength=A * A).reshape(A, A)
            mi_all.append(mutual_information(t))
            mi_same.append(float("nan"))
    return mi_all, mi_same


def _baseline(entry, bname, seed, L, D):
    ball, bsame = [], []
    for r in range(REPS):
        s, _ = baselines.run(bname, entry, stream(seed, "R3", entry.name, bname, r, "unit"), values=None)
        a, sm = curves(s, entry.A, L, D)
        ball.append(a)
        bsame.append(sm)
    mu, sd = mean_sd(ball)
    mus, sds = mean_sd(bsame)
    return {"base_mean": nanlist(mu), "base_sd": nanlist(sd), "base_reps": ball,
            "base_same_unit_mean": nanlist(mus), "base_same_unit_sd": nanlist(sds)}


def compute(entry, cfg, seed):
    R, W = entry.shape
    D = min(cfg.get("max_lag", 96), W - 1)
    if D < 1:
        return {"R3": {"applicable": False, "reason": "rows too short"}}
    L = entry.unit_len
    bname, alts = baseline_names(entry, "R3", cfg, "unit")
    real_all, real_same = curves(entry.symbols, entry.A, L, D)
    out = {"lags": list(range(1, D + 1)), "mi": real_all, "mi_same_unit": nanlist(real_same),
           "baseline": bname, "unit_len": L, "breakeven": breakeven.pair_mi(entry.A, cfg.get("b", 16), entry.N)}
    out.update(_baseline(entry, bname, seed, L, D))
    if alts:
        out["alt"] = {b: _baseline(entry, b, seed, L, D) for b in alts}
    return {"R3": out}


def _med0(items, key):
    v = [r[key][0] for _, r in items if r[key][0] is not None]
    return float(np.median(v)) if v else None


def _judge(cut, items, mu_key, sd_key, amp):
    Rm, Mu, Sd = med_baseline(cut, "mi", mu_key, sd_key)
    ok, _ = exceed_stats(Rm, Mu, Sd, amp, curve=True)
    be = float(np.median([r["breakeven"] for _, r in items]))
    return Rm, Mu, ok, be, bool(np.any(ok & (Rm > be)))


def aggregate(fig, items, cfg, seed):
    items = applicable(items)
    if not items:
        return not_applicable_summary(items)
    amp = cfg.get("min_amp", 0.002)
    n = min(len(r["lags"]) for _, r in items)

    def cut_of(r, sub=None):
        d = dict(r)
        if sub is not None:
            d.update(r["alt"][sub])
        return {k: (v[:n] if isinstance(v, list) else v) for k, v in d.items() if k != "alt"}

    cut = [(nm, cut_of(r)) for nm, r in items]
    Rm, Mu, ok, be, passes = _judge(cut, items, "base_mean", "base_sd", amp)
    alt = {}
    for b in items[0][1].get("alt", {}):
        ca = [(nm, cut_of(r, b)) for nm, r in items]
        _, Ma, oka, _, pa = _judge(ca, items, "base_mean", "base_sd", amp)
        alt[b] = {"base": band([r["base_mean"] for _, r in ca]), "verdict": verdict(bool(oka.any()), pa),
                  "exceed_lags": [int(i + 1) for i in np.flatnonzero(oka)], "base_d1": float(Ma[0]),
                  "same_unit_d1_base": _med0([(nm, r) for nm, r in ca], "base_same_unit_mean")}
    return {"summary": {"lags": list(range(1, n + 1)), "real": band([r["mi"] for _, r in cut]),
                        "base": band([r["base_mean"] for _, r in cut]), "breakeven": be,
                        "baseline": items[0][1]["baseline"], "n_entries": len(items),
                        "unit_len": items[0][1].get("unit_len"), "alt": alt,
                        "real_d1": float(Rm[0]), "base_d1": float(Mu[0]),
                        "same_unit_d1": {"real": _med0(items, "mi_same_unit"),
                                         "base": _med0(cut, "base_same_unit_mean")}},
            "verdict": verdict(bool(ok.any()), passes),
            "detail": {"exceed_lags": [int(i + 1) for i in np.flatnonzero(ok)], "min_amp": amp},
            "key": f"MI(d=1) = {Rm[0]:.4g}, baseline = {Mu[0]:.4g}, break-even = {be:.3g}"}
