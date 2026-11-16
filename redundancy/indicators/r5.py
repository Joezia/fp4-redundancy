"""R5: rows are mixtures of a few patterns. Energy share of each singular value (squared, of the
mean-removed `values`), largest first. Baseline: the adapter's special baseline when given, else a
whole shuffle (a random spectrum). Break-even per layer: share > 1 - 2^(-2 (R+W) b / N).

Optional strict gain (`gain` in the config, method of spec 003 R5.3): quadrant hold-out low-rank
prediction of the values with k layers; the prediction cut into `bins` quantile bins is a context;
gain = held-out H(symbol | known) - H(symbol | known, bin) on symbols with equivalent symbols merged
(halves = even / odd rows); factor cost = k (R + W) b / N; net = gain - cost, in bit per symbol."""
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


def holdout_prediction(X, r, k):
    """Quadrant hold-out: rows and columns split at random into halves; every quadrant is predicted
    with rank k from the three others (the opposite quadrant's SVD). Returns prediction and R2."""
    R, W = X.shape
    pr, pc = r.permutation(R), r.permutation(W)
    R1, R2 = np.sort(pr[:R // 2]), np.sort(pr[R // 2:])
    C1, C2 = np.sort(pc[:W // 2]), np.sort(pc[W // 2:])
    P = np.empty_like(X)
    r2 = []
    for Ra, Ro in ((R2, R1), (R1, R2)):
        for Cb, Co in ((C2, C1), (C1, C2)):
            U, s, Vt = np.linalg.svd(X[np.ix_(Ro, Co)], full_matrices=False)
            kk = min(k, len(s))
            L = X[np.ix_(Ra, Co)] @ Vt[:kk].T
            Rm = U[:, :kk].T @ X[np.ix_(Ro, Cb)]
            Yh = (L / s[:kk]) @ Rm
            Y = X[np.ix_(Ra, Cb)]
            P[np.ix_(Ra, Cb)] = Yh
            r2.append(1.0 - float(((Y - Yh) ** 2).sum()) / float((Y * Y).sum()))
    return P, float(np.mean(r2))


def holdout_cond_entropy(ctx, tgt, half, nctx, A, alpha=0.5):
    """Counts from one half (+alpha per cell), cross-entropy on the other, both ways averaged."""
    key = (half * nctx + ctx) * A + tgt
    cnt = np.bincount(key, minlength=2 * nctx * A).reshape(2, nctx, A).astype(np.float64)
    hs = []
    for a in (0, 1):
        train, test = cnt[a], cnt[1 - a]
        p = (train + alpha) / (train.sum(axis=1, keepdims=True) + alpha * A)
        hs.append(float(-(test * np.log2(p)).sum() / test.sum()))
    return float(np.mean(hs))


def strict_gain(entry, gcfg, b, seed):
    k, nb = int(gcfg.get("k", 4)), int(gcfg.get("bins", 16))
    R, W = entry.shape
    X = entry.values.astype(np.float64)
    X = X - X.mean()
    P, r2 = holdout_prediction(X, rng(seed, "R5", entry.name, "gain"), k)
    edges = np.quantile(P, np.arange(1, nb) / nb)
    bins = np.searchsorted(edges, P, side="right").ravel().astype(np.int64)
    kn = entry.known_per_symbol()
    if kn is None:
        ctx, nctx = np.zeros(entry.N, dtype=np.int64), 1
    else:
        _, inv = np.unique(kn.ravel(), return_inverse=True)
        ctx, nctx = inv.astype(np.int64), int(inv.max()) + 1
    tgt = entry.canonical().ravel().astype(np.int64)
    half = np.repeat(np.arange(R) % 2, W)
    h0 = holdout_cond_entropy(ctx, tgt, half, nctx, entry.A)
    h1 = holdout_cond_entropy(ctx * nb + bins, tgt, half, nctx * nb, entry.A)
    cost = k * (R + W) * b / entry.N
    return {"k": k, "bins": nb, "b": b, "holdout_R2": r2, "H_known": h0, "H_known_bin": h1,
            "gain": h0 - h1, "cost": cost, "net": h0 - h1 - cost}


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
    extra = {"gain": strict_gain(entry, cfg["gain"], cfg.get("b", 16), seed)} if cfg.get("gain") else {}
    return {"R5": {**extra, "share": nanlist(real[:show]), "base_mean": nanlist(mu), "base_sd": nanlist(sd),
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
    gains = [r["gain"] for _, r in items if "gain" in r]
    gsum = {}
    if gains:
        gsum = {q: {"median": float(np.median([g[q] for g in gains])), "min": float(np.min([g[q] for g in gains])),
                    "max": float(np.max([g[q] for g in gains]))} for q in ("gain", "cost", "net", "holdout_R2")}
        gsum.update({"k": gains[0]["k"], "bins": gains[0]["bins"], "b": gains[0]["b"], "n": len(gains)})
    return {"summary": {"gain": gsum, "layers": list(range(1, n + 1)), "real": band([r["share"] for _, r in cut]),
                        "base": band([r["base_mean"] for _, r in cut]), "breakeven": be,
                        "baseline": items[0][1]["baseline"], "n_entries": len(items),
                        "top": float(Rm[0]), "top_base": float(Mu[0])},
            "verdict": verdict(bool(ok.any()), passes),
            "detail": {"exceed_layers": [int(i + 1) for i in np.flatnonzero(ok)], "min_amp": amp},
            "key": f"top share = {Rm[0]:.4g} (baseline {Mu[0]:.4g}), break-even = {be:.3g}"}
