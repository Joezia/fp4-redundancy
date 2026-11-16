"""R2: probability distributions differ between groups.

For every grouping variable: plug-in I(symbol; label), baseline = the same after a whole shuffle of
the symbols (labels fixed), net = I - baseline mean. Groups of scope "entry" (one label per entry)
are measured pooled across entries, with a baseline that permutes the labels among entries.
Break-even: net > (A-1)(G-1) b / N, plus H(labels) / coverage for labels the decoder lacks."""
import numpy as np

from .. import breakeven
from ..dataset import entropy, mutual_information
from .common import (NONE, NA, REPS, applicable, exceed_stats, mean_sd, not_applicable_summary,
                     rng, verdict)

FIGURES = ("R2",)
TOP = 16


def _relabel(labels, shape):
    lab = np.ascontiguousarray(np.broadcast_to(labels, shape)).ravel()
    vals, inv = np.unique(lab, return_inverse=True)
    return vals, inv.astype(np.int64)


def compute(entry, cfg, seed):
    b = cfg.get("b", 16)
    A, N = entry.A, entry.N
    sym = entry.symbols.ravel().astype(np.int64)
    if not entry.groups:
        return {"R2": {"applicable": False, "reason": "no groups"}}
    shuf = [rng(seed, "R2", entry.name, r).permutation(sym) for r in range(REPS)]
    out = {"A": A, "N": N, "groups": {}, "counts": np.bincount(sym, minlength=A).tolist()}
    for name in sorted(entry.groups):
        g = entry.groups[name]
        if g.scope == "entry":
            out["groups"][name] = {"scope": "entry", "label": int(g.labels), "known": bool(g.known)}
            continue
        vals, inv = _relabel(g.labels, entry.shape)
        G = len(vals)
        key = inv * A
        t = np.bincount(key + sym, minlength=G * A).reshape(G, A)
        mi = mutual_information(t)
        reps = [mutual_information(np.bincount(key + s, minlength=G * A).reshape(G, A)) for s in shuf]
        lab_bits = 0.0
        if not g.known:
            lab_bits = entropy(t.sum(axis=1)) / (g.coverage or 1)
        top = np.argsort(-t.sum(axis=1), kind="stable")[:TOP]
        out["groups"][name] = {
            "scope": "symbol", "known": bool(g.known), "G": G, "mi": mi, "base": reps,
            "label_bits": lab_bits, "cost": breakeven.groups_mi(A, G, b, N, lab_bits),
            "top_labels": [int(vals[i]) for i in top], "top_counts": t[top].tolist()}
    return {"R2": out}


def aggregate(fig, items, cfg, seed):
    items = applicable(items)
    if not items:
        return not_applicable_summary(items)
    b, amp = cfg.get("b", 16), cfg.get("min_amp", 0.002)
    A = max(r["A"] for _, r in items)

    def padded(c):
        c = np.asarray(c, dtype=np.float64)
        return np.pad(c, [(0, 0)] * (c.ndim - 1) + [(0, A - c.shape[-1])])

    names = sorted(set.intersection(*[set(r["groups"]) for _, r in items]))
    rows = {}
    for name in names:
        gs = [r["groups"][name] for _, r in items]
        if gs[0]["scope"] == "entry":
            labs = np.array([g["label"] for g in gs])
            counts = np.asarray([padded(r["counts"]) for _, r in items])

            def pooled(lb):
                u, inv = np.unique(lb, return_inverse=True)
                t = np.zeros((len(u), A))
                np.add.at(t, inv, counts)
                return mutual_information(t), len(u)

            mi, G = pooled(labs)
            reps = [pooled(rng(seed, "R2", "pooled", name, r).permutation(labs))[0] for r in range(REPS)]
            mu, sd = mean_sd(reps)
            Ntot = float(sum(r["N"] for _, r in items))
            cost = breakeven.groups_mi(A, G, b, Ntot)
            rows[name] = {"scope": "entry", "known": gs[0]["known"], "G": G, "mi": mi,
                          "base_mean": float(mu), "base_sd": float(sd), "net": mi - float(mu),
                          "cost": cost, "pooled": True, "label_bits": 0.0}
            continue
        mi = np.array([g["mi"] for g in gs])
        mus = np.array([np.mean(g["base"]) for g in gs])
        sds = np.array([np.std(g["base"], ddof=1) for g in gs])
        rows[name] = {"scope": "symbol", "known": gs[0]["known"], "G": int(np.median([g["G"] for g in gs])),
                      "mi": float(np.median(mi)), "base_mean": float(np.median(mus)),
                      "base_sd": float(np.median(sds)), "net": float(np.median(mi - mus)),
                      "cost": float(np.median([g["cost"] for g in gs])),
                      "label_bits": float(np.median([g["label_bits"] for g in gs])), "pooled": False}
    for name, r in rows.items():
        ok, _ = exceed_stats(r["mi"], r["base_mean"], r["base_sd"], amp)
        r["exceeds"] = bool(ok[0])
        r["pays"] = bool(r["exceeds"] and r["net"] > r["cost"])
    if not rows:
        return {"summary": {}, "verdict": NA, "detail": {"reason": "no groups"}, "key": ""}
    best = max(rows, key=lambda k: rows[k]["net"])
    # distributions of the best group: pooled counts of the most frequent labels
    dist = {}
    overall = np.asarray([padded(r["counts"]) for _, r in items]).sum(axis=0)
    if rows[best]["scope"] == "symbol":
        for _, r in items:
            g = r["groups"][best]
            for lab, c in zip(g["top_labels"], g["top_counts"]):
                dist.setdefault(lab, np.zeros(A))
                dist[lab] += padded(c)
    else:
        for _, r in items:
            lab = r["groups"][best]["label"]
            dist.setdefault(lab, np.zeros(A))
            dist[lab] += padded(r["counts"])
    top = sorted(dist, key=lambda k: -dist[k].sum())[:6]
    exists = any(r["exceeds"] for r in rows.values())
    passes = any(r["pays"] for r in rows.values())
    rb = rows[best]
    return {"summary": {"groups": rows, "best": best, "alphabet_size": A,
                        "curves": {str(k): (dist[k] / dist[k].sum()).tolist() for k in top},
                        "overall": (overall / overall.sum()).tolist(), "n_entries": len(items)},
            "verdict": verdict(exists, passes),
            "detail": {"rule": "exceeds: MI - baseline mean > 5 sd and > min_amp; pays: net > cost",
                       "min_amp": amp, "b": b},
            "key": f"best = {best}: net MI = {rb['net']:.4g}, cost = {rb['cost']:.3g}"}
