"""R8: every segment uses a few atoms of a dictionary. On the `values` of sampled units (first half
fit, second half evaluate): learn a dictionary of K unit-norm atoms with s atoms per segment
(alternating orthogonal matching pursuit and least-squares update, initialised with K random
fit segments), then the held-out explained share R2(K, s) with the same s. Baseline: shuffle within
`known` groups, whole pipeline repeated. Break-even: R2 > 1 - 2^(-2 s (log2 K + b) / L); the
judgement uses `b`, the figure also draws `b_alt`."""
import math

import numpy as np

from .. import baselines, breakeven
from .common import (REPS, applicable, band, exceed_stats, med_baseline, mean_sd, nanlist,
                     not_applicable_summary, rng, verdict)
from .sample import unit_sample, unit_values

FIGURES = ("R8",)


def omp(X, D, s):
    n = len(X)
    sel = np.zeros((n, s), dtype=np.int64)
    ar = np.arange(n)
    coef = None
    resid = X.copy()
    for t in range(s):
        c = np.abs(resid @ D.T)
        for u in range(t):
            c[ar, sel[:, u]] = -1.0
        sel[:, t] = np.argmax(c, axis=1)
        Ds = D[sel[:, :t + 1]]
        G = Ds @ Ds.transpose(0, 2, 1) + 1e-10 * np.eye(t + 1)
        coef = np.linalg.solve(G, Ds @ X[:, :, None])[:, :, 0]
        resid = X - (coef[:, :, None] * Ds).sum(axis=1)
    return sel, coef, resid


def _renorm(D, Xf, r):
    nrm = np.sqrt((D * D).sum(axis=1))
    for i in np.flatnonzero(nrm <= 0):
        while True:
            x = Xf[r.integers(len(Xf))]
            if (x * x).sum() > 0:
                D[i] = x
                nrm[i] = math.sqrt(float((x * x).sum()))
                break
    return D / nrm[:, None]


def learn(Xf, K, s, r, iters):
    D = _renorm(Xf[r.choice(len(Xf), size=K, replace=False)].copy(), Xf, r)
    for _ in range(iters):
        sel, coef, _ = omp(Xf, D, s)
        G = np.zeros((K, K))
        for a in range(s):
            for c in range(s):
                np.add.at(G, (sel[:, a], sel[:, c]), coef[:, a] * coef[:, c])
        CX = np.zeros((K, Xf.shape[1]))
        for a in range(s):
            np.add.at(CX, sel[:, a], coef[:, a, None] * Xf)
        D = _renorm(np.linalg.solve(G + 1e-6 * np.eye(K), CX), Xf, r)
    return D


def sweep(X, ks, ss, seed, name, variant, iters):
    nf = len(X) // 2
    Xf, Xe = X[:nf], X[nf:]
    ee = float((Xe * Xe).sum())
    out = {}
    for s in ss:
        for K in ks:
            if K > nf or s > X.shape[1]:
                out[f"{K},{s}"] = float("nan")
                continue
            D = learn(Xf, K, s, rng(seed, "R8", name, variant, K, s), iters)
            _, _, res = omp(Xe, D, s)
            out[f"{K},{s}"] = 1.0 - float((res * res).sum()) / ee if ee > 0 else float("nan")
    return out


def compute(entry, cfg, seed):
    only = cfg.get("entries")
    if only is not None and entry.name not in only:
        return {"R8": {"applicable": False, "reason": "entry not in the configured R8 sample"}}
    if entry.values is None or not entry.unit_len:
        return {"R8": {"applicable": False, "reason": "no values" if entry.values is None else "no unit"}}
    ks = [int(k) for k in cfg.get("ks", (16, 64, 256, 1024))]
    ss = [int(s) for s in cfg.get("ss", (1, 2, 4))]
    iters = cfg.get("iters", 15)
    L = entry.unit_len
    pick, label = unit_sample(entry, rng(seed, "R8", entry.name, "sample"), cfg.get("sample", 100000))
    real = sweep(unit_values(entry.values, L, pick), ks, ss, seed, entry.name, "real", iters)
    bname = baselines.pick(entry, "R8", cfg.get("baseline", "known"))
    reps = []
    for i in range(REPS):
        _, v = baselines.run(bname, entry, rng(seed, "R8", entry.name, "shuffle", i), symbols=entry.symbols)
        reps.append(sweep(unit_values(v, L, pick), ks, ss, seed, entry.name, f"base{i}", iters))
    keys = list(real)
    mu, sd = mean_sd([[rp[k] for k in keys] for rp in reps])
    b, b2 = cfg.get("b", 8), cfg.get("b_alt", 16)
    return {"R8": {"ks": ks, "ss": ss, "keys": keys, "r2": [real[k] for k in keys],
                   "base_mean": nanlist(mu), "base_sd": nanlist(sd), "baseline": bname, "L": L,
                   "n_units": int(len(pick)), "known_label": label, "b": b, "b_alt": b2,
                   "breakeven": [breakeven.dictionary(int(k.split(",")[0]), int(k.split(",")[1]), b, L) for k in keys],
                   "breakeven_alt": [breakeven.dictionary(int(k.split(",")[0]), int(k.split(",")[1]), b2, L)
                                     for k in keys]}}


def aggregate(fig, items, cfg, seed):
    items = applicable(items)
    if not items:
        return not_applicable_summary(items)
    amp = cfg.get("min_amp", 0.005)
    r0 = items[0][1]
    Rm, Mu, Sd = med_baseline(items, "r2", "base_mean", "base_sd")
    ok, _ = exceed_stats(Rm, Mu, Sd, amp, curve=True)
    be = np.asarray(r0["breakeven"])
    passes = bool(np.any(ok & (Rm > be)))
    keys = r0["keys"]
    i = keys.index("64,2") if "64,2" in keys else 0
    return {"summary": {"ks": r0["ks"], "ss": r0["ss"], "keys": keys,
                        "real": band([r["r2"] for _, r in items]), "base": band([r["base_mean"] for _, r in items]),
                        "breakeven": be.tolist(), "breakeven_alt": r0["breakeven_alt"], "b": r0["b"],
                        "b_alt": r0["b_alt"], "baseline": r0["baseline"], "n_entries": len(items)},
            "verdict": verdict(bool(ok.any()), passes),
            "detail": {"exceed": [keys[j] for j in np.flatnonzero(ok)], "min_amp": amp, "b": r0["b"]},
            "key": f"R2(K=64, s=2) = {Rm[i]:.3f} (baseline {Mu[i]:.3f}), break-even {be[i]:.3f} (b = {r0['b']})"}
