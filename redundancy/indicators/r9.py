"""R9: the generating process forbids or favours some situations. Every entry of `process_checks`
says what to count per unit and how to compute the distribution expected without the constraint:

kind "count_in_unit": count of symbols from `symbols` in each unit; expected: binomial(L, q) with q
    the share of those symbols, mixed over `known` labels when `condition_on_known` (q per label).
kind "max_rank_in_unit": the largest `rank[symbol]` in each unit; expected: units of independent
    symbols, P(max = k) = F(k)^L - F(k-1)^L with F the rank CDF, mixed over `known` labels likewise.
Two-level verdict: present when the total variation distance between observed and expected
exceeds `tv` (default 0.05)."""
import math

import numpy as np

from .common import NA, NONE, PRESENT, applicable, band

FIGURES = None  # one figure per check: "R9_<name>"


def _labels(entry, cond):
    nu = entry.N // entry.unit_len
    if cond and entry.known is not None:
        _, inv = np.unique(entry.known.ravel(), return_inverse=True)
        return inv
    return np.zeros(nu, dtype=np.int64)


def run_check(entry, chk):
    L = entry.unit_len
    U = entry.symbols.reshape(-1, L)
    lab = _labels(entry, chk.get("condition_on_known", True))
    nl = int(lab.max()) + 1
    w = np.bincount(lab, minlength=nl).astype(np.float64)
    if chk["kind"] == "count_in_unit":
        member = np.zeros(entry.A, dtype=bool)
        member[list(chk["symbols"])] = True
        cnt = member[U].sum(axis=1)
        obs = np.bincount(cnt, minlength=L + 1).astype(np.float64)
        q = np.bincount(lab, weights=cnt, minlength=nl) / (w * L)
        k = np.arange(L + 1)
        comb = np.array([math.comb(L, int(i)) for i in k], dtype=np.float64)
        exp = np.zeros(L + 1)
        for g in range(nl):
            if w[g]:
                exp += w[g] * comb * q[g] ** k * (1 - q[g]) ** (L - k)
        x = [int(i) for i in k]
    elif chk["kind"] == "max_rank_in_unit":
        rank = np.asarray(chk["rank"], dtype=np.int64)
        nr = int(rank.max()) + 1
        rk = rank[U]
        mx = rk.max(axis=1)
        obs = np.bincount(mx, minlength=nr).astype(np.float64)
        exp = np.zeros(nr)
        for g in range(nl):
            if not w[g]:
                continue
            pr = np.bincount(rk[lab == g].ravel(), minlength=nr).astype(np.float64)
            F = np.cumsum(pr / pr.sum())
            P = F ** L - np.concatenate([[0.0], F[:-1]]) ** L
            exp += w[g] * P
        x = list(range(nr))
    else:
        raise ValueError(chk["kind"])
    obs, exp = obs / obs.sum(), exp / exp.sum()
    return {"kind": chk["kind"], "name": chk["name"], "x": x, "x_labels": chk.get("x_labels"),
            "observed": obs.tolist(), "expected": exp.tolist(), "tv": float(0.5 * np.abs(obs - exp).sum()),
            "description": chk.get("description", "")}


def figures(entry):
    if not entry.process_checks or not entry.unit_len:
        return ["R9"]
    return ["R9_" + c["name"] for c in entry.process_checks]


def compute(entry, cfg, seed):
    if not entry.unit_len:
        return {"R9": {"applicable": False, "reason": "no unit"}}
    if not entry.process_checks:
        return {"R9": {"applicable": False, "reason": "no process_checks"}}
    return {"R9_" + c["name"]: run_check(entry, c) for c in entry.process_checks}


def aggregate(fig, items, cfg, seed):
    items = applicable(items)
    if not items:
        reasons = sorted({r.get("reason", "") for _, r in items})
        return {"summary": {}, "verdict": NA, "detail": {"reason": "; ".join(reasons)}, "key": ""}
    thr = cfg.get("tv", 0.05)
    n = min(len(r["x"]) for _, r in items)
    tv = float(np.median([r["tv"] for _, r in items]))
    r0 = items[0][1]
    first = {"observed": float(np.median([r["observed"][0] for _, r in items])),
             "expected": float(np.median([r["expected"][0] for _, r in items]))}
    return {"summary": {"kind": r0["kind"], "name": r0["name"], "x": r0["x"][:n],
                        "x_labels": r0["x_labels"][:n] if r0["x_labels"] else None,
                        "observed": band([r["observed"][:n] for _, r in items]),
                        "expected": band([r["expected"][:n] for _, r in items]), "tv": tv,
                        "first_bin": first, "description": r0["description"], "n_entries": len(items)},
            "verdict": PRESENT if tv > thr else NONE,
            "detail": {"rule": "total variation distance > tv", "tv_threshold": thr},
            "key": f"TV = {tv:.3f}; share at {r0['x'][0]}: {first['observed']:.3f} (expected {first['expected']:.3g})"}
