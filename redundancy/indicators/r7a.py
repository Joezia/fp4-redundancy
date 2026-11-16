"""R7a: a segment resembles another segment (pairs). For every segment, the similarity to its most
similar other segment; a histogram over segments.

unit    (short segments; needs `unit`): similarity = number of equal positions (symbols with
        equivalent symbols merged); `queries` random units, candidates = all other units with the
        same `known` label (all units without `known`). Baseline: shuffle within `known` groups.
        Break-even m*: smallest m with m·H > log2(#candidates) + log2 C(L, m).
row     (long segments; needs `values`): |correlation| with the most correlated other row.
        Baseline: shuffle within `known` groups. Break-even: sqrt(1 - 2^(-2 (log2 #rows + b) / L)).
partner (needs `partner`): |correlation| of row i with partner row i (the partner's candidate for
        it). Baseline: partner rows randomly permuted. Break-even as for rows with one candidate.
Existence uses the mean similarity; the break-even test uses the share of segments beyond the line.
"""
import numpy as np

from .. import baselines, breakeven
from ..dataset import entropy
from .common import (REPS, applicable, exceed_stats, mean_sd, not_applicable_summary, rng, verdict)

FIGURES = ("R7a_unit", "R7a_row", "R7a_partner")
CORR_BINS = np.linspace(0.0, 1.0, 101)


def _best_equal(U, queries, groups_of_q, members):
    """For every query unit, the largest number of equal positions with another unit of its group."""
    out = np.full(len(queries), -1, dtype=np.int64)
    for g, gidx in members.items():
        qi = np.flatnonzero(groups_of_q == g)
        if len(qi) == 0 or len(gidx) < 2:
            continue
        G = U[gidx]
        for c0 in range(0, len(qi), 8):
            sel = qi[c0:c0 + 8]
            q = queries[sel]
            eq = (G[None, :, :] == U[q][:, None, :]).sum(axis=2, dtype=np.int16)
            eq[np.arange(len(sel)), np.searchsorted(gidx, q)] = -1
            out[sel] = eq.max(axis=1)
    return out


def unit_part(entry, cfg, seed):
    if not entry.unit_len:
        return {"applicable": False, "reason": "no unit"}
    L = entry.unit_len
    X = entry.canonical()
    U = X.reshape(-1, L)
    nu = len(U)
    lab = entry.known.ravel() if entry.known is not None else np.zeros(nu, dtype=np.int64)
    r = rng(seed, "R7a_unit", entry.name, "queries")
    queries = np.sort(r.choice(nu, size=min(cfg.get("queries", 500), nu), replace=False))
    members = {}
    for g in np.unique(lab[queries]):
        members[g] = np.flatnonzero(lab == g)
    gq = lab[queries]
    ncand = np.array([len(members[g]) - 1 for g in gq])
    H = entropy(np.bincount(X.ravel(), minlength=entry.A))
    mstar = np.array([breakeven.short_match(int(n), H, L) or (L + 1) for n in ncand])
    bname = baselines.pick(entry, "R7a_unit", cfg.get("baseline", "known"))
    real = _best_equal(U, queries, gq, members)
    reps = []
    for i in range(REPS):
        s, _ = baselines.run(bname, entry, rng(seed, "R7a_unit", entry.name, i), symbols=X, values=None)
        reps.append(_best_equal(s.reshape(-1, L), queries, gq, members))
    ok = ncand > 0

    def stats(b):
        b = b[ok]
        return {"mean": float(b.mean()), "median": float(np.median(b)),
                "share": float((b >= mstar[ok]).mean()),
                "hist": np.bincount(b, minlength=L + 1)[: L + 1].tolist()}

    return {"kind": "unit", "L": L, "H": H, "n_queries": int(ok.sum()), "baseline": bname,
            "mstar_median": float(np.median(mstar[ok])), "ncand_median": float(np.median(ncand)),
            "real": stats(real), "base": [stats(b) for b in reps]}


def _std_rows(V):
    X = V.astype(np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    n = np.sqrt((X * X).sum(axis=1, keepdims=True))
    return np.divide(X, n, out=np.zeros_like(X), where=n > 0)


def _max_corr(V):
    Z = _std_rows(V)
    C = np.abs(Z @ Z.T)
    np.fill_diagonal(C, -1.0)
    return C.max(axis=1)


def _corr_stats(c, line):
    return {"mean": float(c.mean()), "median": float(np.median(c)), "share": float((c > line).mean()),
            "hist": np.histogram(np.clip(c, 0, 1), bins=CORR_BINS)[0].tolist()}


def row_part(entry, cfg, seed):
    if entry.values is None:
        return {"applicable": False, "reason": "no values"}
    R, W = entry.shape
    if R < 2:
        return {"applicable": False, "reason": "fewer than 2 rows"}
    line = breakeven.long_corr(R - 1, cfg.get("b", 16), W)
    bname = baselines.pick(entry, "R7a_row", cfg.get("baseline", "known"))
    real = _max_corr(entry.values)
    reps = []
    for i in range(REPS):
        _, v = baselines.run(bname, entry, rng(seed, "R7a_row", entry.name, i), symbols=entry.symbols)
        reps.append(_max_corr(v))
    return {"kind": "row", "L": W, "line": line, "baseline": bname, "n": R,
            "real": _corr_stats(real, line), "base": [_corr_stats(b, line) for b in reps]}


def _aligned_corr(V, P):
    return np.abs((_std_rows(V) * _std_rows(P)).sum(axis=1))


def partner_part(entry, cfg, seed):
    if entry.values is None or entry.partner is None:
        return {"applicable": False, "reason": "no partner" if entry.partner is None else "no values"}
    R, W = entry.shape
    line = breakeven.long_corr(1, cfg.get("b", 16), W)
    real = _aligned_corr(entry.values, entry.partner)
    reps = [_aligned_corr(entry.values, baselines.misalign_partner(entry.partner,
                                                                   rng(seed, "R7a_partner", entry.name, i)))
            for i in range(REPS)]
    return {"kind": "partner", "L": W, "line": line, "baseline": "misalign", "n": R,
            "partner": entry.partner_name, "real": _corr_stats(real, line),
            "base": [_corr_stats(b, line) for b in reps]}


def compute(entry, cfg, seed):
    return {"R7a_unit": unit_part(entry, cfg, seed), "R7a_row": row_part(entry, cfg, seed),
            "R7a_partner": partner_part(entry, cfg, seed)}


def aggregate(fig, items, cfg, seed):
    items = applicable(items)
    if not items:
        return not_applicable_summary(items)
    kind = items[0][1]["kind"]
    amp = cfg.get("min_amp_unit", 0.25) if kind == "unit" else cfg.get("min_amp_corr", 0.005)
    amp_s = cfg.get("min_amp_share", 0.01)
    real_mean = np.median([r["real"]["mean"] for _, r in items])
    mu_sd = [mean_sd([b["mean"] for b in r["base"]]) for _, r in items]
    sh_sd = [mean_sd([b["share"] for b in r["base"]]) for _, r in items]
    mu, sd = np.median([m for m, _ in mu_sd]), np.median([s for _, s in mu_sd])
    ok, _ = exceed_stats(real_mean, mu, sd, amp)
    share = np.median([r["real"]["share"] for _, r in items])
    smu, ssd = np.median([m for m, _ in sh_sd]), np.median([s for _, s in sh_sd])
    ok_s, _ = exceed_stats(share, smu, ssd, amp_s)
    exists = bool(ok[0])

    def norm(h):
        h = np.asarray(h, dtype=np.float64)
        return h / h.sum() if h.sum() else h

    hr = np.median([norm(r["real"]["hist"]) for _, r in items], axis=0)
    hb = np.median([np.mean([norm(b["hist"]) for b in r["base"]], axis=0) for _, r in items], axis=0)
    real_med = float(np.median([r["real"]["median"] for _, r in items]))
    base_med = float(np.median([np.mean([b["median"] for b in r["base"]]) for _, r in items]))
    if kind == "unit":
        line = float(np.median([r["mstar_median"] for _, r in items]))
        x = list(range(len(hr)))
        key = f"median best = {real_med:g} (baseline {base_med:.3g}), m* = {line:g} of {items[0][1]['L']}"
    else:
        line = float(np.median([r["line"] for _, r in items]))
        x = [float(v) for v in 0.5 * (CORR_BINS[1:] + CORR_BINS[:-1])]
        what = "|ρ|" if kind == "partner" else "max|ρ|"
        key = f"median {what} = {real_med:.3f} (baseline {base_med:.3f}), ρ* = {line:.3f}"
    return {"summary": {"kind": kind, "x": x, "real": hr.tolist(), "base": hb.tolist(), "line": line,
                        "mean": float(real_mean), "base_mean": float(mu), "base_sd": float(sd),
                        "share": float(share), "base_share": float(smu), "median": real_med,
                        "base_median": base_med, "baseline": items[0][1]["baseline"],
                        "n_entries": len(items)},
            "verdict": verdict(exists, bool(exists and ok_s[0])),
            "detail": {"rule": "exists: mean similarity vs baseline; pays: share beyond the line vs baseline",
                       "min_amp": amp, "min_amp_share": amp_s},
            "key": key}
