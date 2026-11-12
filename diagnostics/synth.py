"""Section 8: synthetic validation of the estimators (no model data).

Every dataset goes through measure.measure_matrix, the same code path as the model matrices.
Usage: python -m diagnostics.synth [--out experiments/r1-r4/validation.json]
"""
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import est
from .measure import measure_matrix

R_SYN, W_SYN, B = 1024, 2048, 32
EDGES = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
SEED = 20261112


def quantize(y):
    m = np.searchsorted(EDGES, np.abs(y))
    s = (y < 0).astype(np.uint8)
    return (8 * s + m).astype(np.uint8)


def code_probs(sigma):
    """P(C=c) for y ~ N(0, sigma^2) under `quantize` (searchsorted side='left')."""
    cdf = lambda x: 0.5 * (1.0 + math.erf(x / (sigma * math.sqrt(2.0))))
    lo = np.concatenate([[0.0], EDGES])
    hi = np.concatenate([EDGES, [np.inf]])
    pos = np.array([cdf(h) - cdf(l) for l, h in zip(lo, hi)])  # y in (l, h], y >= 0
    q = np.concatenate([pos, pos])  # symmetric; y = 0 has probability 0
    return q / q.sum()


def mi_joint(p):
    return est.mi_table(p * 1e12)[0]


# ----------------------------------------------------------------------------- datasets

def gen_T1(rng):
    vals = np.array([c for c in range(16) if c != 8], dtype=np.uint8)
    return vals[rng.integers(0, 15, size=(R_SYN, W_SYN))], {}


def gen_T2(rng):
    P = 0.7 * rng.dirichlet(np.full(16, 3.0), size=16) + 0.3 / 16
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi = pi / pi.sum()
    cP = np.cumsum(P, axis=1)
    cP[:, -1] = 1.0
    X = np.empty((R_SYN, W_SYN), dtype=np.uint8)
    X[:, 0] = np.searchsorted(np.cumsum(pi), rng.random(R_SYN), side="right").clip(0, 15)
    for j in range(1, W_SYN):
        u = rng.random(R_SYN)
        X[:, j] = (u[:, None] >= cP[X[:, j - 1]]).sum(axis=1)
    hpi = -(pi * np.log2(pi)).sum()
    hrows = -(P * np.log2(P)).sum(axis=1)
    theory = float(hpi - (pi * hrows).sum())
    return X, {"mi_d1": theory}


def gen_T3(rng):
    nb = W_SYN // B
    sig = np.where(rng.random((R_SYN, nb)) < 0.5, 1.2, 2.2)
    y = rng.standard_normal((R_SYN, nb, B)) * sig[:, :, None]
    q1, q2 = code_probs(1.2), code_probs(2.2)
    joint = 0.5 * np.outer(q1, q1) + 0.5 * np.outer(q2, q2)
    return quantize(y.reshape(R_SYN, W_SYN)), {"mi_same": float(mi_joint(joint))}


def gen_T4(rng):
    X = rng.integers(0, 16, size=(R_SYN, W_SYN)).astype(np.int64)
    use = rng.random((R_SYN, W_SYN)) < 0.3
    for j in range(2, W_SYN):
        X[:, j] = np.where(use[:, j], (X[:, j - 1] + X[:, j - 2]) % 16, X[:, j])
    p = 0.3 + 0.7 / 16
    h = -(p * math.log2(p) + 15 * (0.7 / 16) * math.log2(0.7 / 16))
    return X.astype(np.uint8), {"left12": 4 - h}


def gen_T5(rng):
    rho = 0.2
    x = np.empty((R_SYN, W_SYN))
    x[:, 0] = rng.standard_normal(R_SYN)
    e = rng.standard_normal((R_SYN, W_SYN)) * math.sqrt(1 - rho * rho)
    for j in range(1, W_SYN):
        x[:, j] = rho * x[:, j - 1] + e[:, j]
    return quantize(2.0 * x), {}


def gen_T6(rng):
    logs = 0.5 * np.cumsum(rng.normal(0.0, 0.05, size=(R_SYN, W_SYN)), axis=1)
    sig = np.exp(logs)
    # "除以均值": mean over the whole sigma array (each row has its own walk along j).
    sig = sig / sig.mean() * 1.5
    # Semi-analytic I(m;m') at d=1 (sigma_j ~ sigma_{j+1}): mixture over the sigma field's quantiles.
    mids = np.quantile(sig, (np.arange(400) + 0.5) / 400)
    joint = np.zeros((8, 8))
    for s in mids:
        q = code_probs(s)
        qm = q[:8] + q[8:]
        joint += np.outer(qm, qm) / len(mids)
    th = {"mi_mm_d1_semi_analytic": float(mi_joint(joint))}
    return quantize(sig * rng.standard_normal((R_SYN, W_SYN))), th


GENS = {"T1": gen_T1, "T2": gen_T2, "T3": gen_T3, "T4": gen_T4, "T5": gen_T5, "T6": gen_T6}


# ----------------------------------------------------------------------------- checks

def find(rows, **kw):
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def one(rows, **kw):
    m = find(rows, **kw)
    assert len(m) == 1, (kw, len(m))
    return m[0]


def chk(name, value, criterion, ok):
    return {"check": name, "value": value, "criterion": criterion, "pass": bool(ok)}


def rel(a, b):
    return abs(a - b) / abs(b)


def checks_T1(rows, th):
    out = []
    K = one(rows, item="A1", group_or_context="K")["value"]
    out.append(chk("A1 K", K, "== 15", K == 15))
    u = one(rows, item="A1", group_or_context="unused=4-log2K")["value"]
    out.append(chk("A1 unused-codeword term", u, "== 0.0931 (4-log2 15, to 4 d.p.)",
                   round(u, 4) == 0.0931))
    s = one(rows, item="A1", group_or_context="skew=log2K-H0")["value"]
    out.append(chk("A1 skew term", s, "< 1e-4", s < 1e-4))
    c1 = find(rows, item="C1", subset="all", baseline="S0", direction="row")
    scaled = [2 * r["n"] * est.LN2 * r["value"] for r in c1]
    m = float(np.mean(scaled))
    out.append(chk("C1 all: mean_d 2 n ln2 MI (d=1..96)", m, "196 +/- 5%",
                   196 * 0.95 <= m <= 196 * 1.05))
    for r in find(rows, item="B2"):
        if r["group_or_context"].endswith(":achievable=I-H(G)/B"):
            out.append(chk(f"B2 {r['group_or_context']}", r["value"], "<= 0", r["value"] <= 0))
    b3 = one(rows, item="B3")
    out.append(chk("B3 net", b3["net"], "|net| < 0.005", abs(b3["net"]) < 0.005))
    for r in find(rows, item="C4", baseline="S0"):
        out.append(chk(f"C4 {r['group_or_context']} raw", r["value"], "<= 0", r["value"] <= 0))
        out.append(chk(f"C4 {r['group_or_context']} raw - S0", r["net"], "|.| < 0.002",
                       abs(r["net"]) < 0.002))
    d1 = [r for r in find(rows, item="D1") if r["value"] != ""]
    nsig = sum(abs(r["value"]) > 5 / math.sqrt(r["n"]) for r in d1)
    out.append(chk("D1 significant count (|r| > 5/sqrt n, all rows)", nsig,
                   f"== 0 of {len(d1)}", nsig == 0))
    for var in ("v", "|v|"):
        r2 = one(rows, item="D3", variable=var, group_or_context="R2")["value"]
        out.append(chk(f"D3 R2 ({var})", r2, "< 1e-4", r2 < 1e-4))
    return out


def checks_T2(rows, th):
    t = th["mi_d1"]
    v = one(rows, item="C1", subset="all", baseline="S0", direction="row", lag=1)["value"]
    c4 = one(rows, item="C4", group_or_context="left1", baseline="S0")["value"]
    return [chk("theory I(d=1)", t, "from transition matrix", True),
            chk("C1 d=1 relative error", rel(v, t), "< 2%", rel(v, t) < 0.02),
            chk("C4 left1 relative error", rel(c4, t), "< 2%", rel(c4, t) < 0.02)]


def checks_T3(rows, th):
    t = th["mi_same"]
    out = [chk("theory same-block pair MI", t, "mixture of two quantized Gaussians", True)]
    s1 = one(rows, item="C1", subset="same", baseline="S0", lag=1)
    s16 = one(rows, item="C1", subset="same", baseline="S0", lag=16)
    out.append(chk("C1 same d=1 relative error", rel(s1["value"], t), "< 5%",
                   rel(s1["value"], t) < 0.05))
    out.append(chk("C1 same |d16-d1|/d1", rel(s16["value"], s1["value"]), "< 10%",
                   rel(s16["value"], s1["value"]) < 0.10))
    cross = find(rows, item="C1", subset="cross", baseline="S0")
    nsig = sum(bool(r["significant"]) for r in cross)
    out.append(chk("C1 cross-block significant count (vs S0, d=1..96)", nsig,
                   f"== 0 of {len(cross)}", nsig == 0))
    s2 = one(rows, item="C1", subset="same", baseline="S2", lag=1)
    ch = rel(s2["baseline_mean"], s2["value"])
    out.append(chk("C1 same d=1 change under S2", ch, "< 5%", ch < 0.05))
    s1b = one(rows, item="C1", subset="same", baseline="S1", lag=1)
    net_s1 = s1b["baseline_mean"] - s1["baseline_mean"]
    thr = 5 * est.analytic_sd(s1["_dof"], s1["n"])
    out.append(chk("C1 same d=1 under S1: S1 mean - S0 mean", net_s1,
                   f"<= 5 analytic sd ({thr:.3g}) (not significant)", net_s1 <= thr))
    e2 = one(rows, item="B2", group_or_context="energy2:achievable=I-H(G)/B")
    out.append(chk("B2 energy2 achievable (real)", e2["value"], "> 0.05", e2["value"] > 0.05))
    out.append(chk("B2 energy2 achievable (S0 mean)", e2["baseline_mean"], "<= 0",
                   e2["baseline_mean"] <= 0))
    b3 = one(rows, item="B3")
    out.append(chk("B3 net", b3["net"], "> 0.05", b3["net"] > 0.05))
    return out


def checks_T4(rows, th):
    out = []
    for d in (1, 2):
        r = one(rows, item="C1", subset="all", baseline="S0", direction="row", lag=d)
        out.append(chk(f"C1 d={d} significant (vs S0)", r["net"], "not significant",
                       not r["significant"]))
    l1 = one(rows, item="C4", group_or_context="left1", baseline="S0")["value"]
    out.append(chk("C4 left1 raw", l1, "|.| < 0.002", abs(l1) < 0.002))
    l12 = one(rows, item="C4", group_or_context="left1-2", baseline="S0")["value"]
    out.append(chk("C4 left1-2 relative error vs 0.5077", rel(l12, th["left12"]), "< 2%",
                   rel(l12, th["left12"]) < 0.02))
    return out


def checks_T5(rows, th):
    out = []
    r = one(rows, item="D1", variable="v", direction="row", subset="all", baseline="S0", lag=1)
    rho = r["value"]
    out.append(chk("D1 v d=1", rho, "in [0.18, 0.21]", 0.18 <= rho <= 0.21))
    thr = 5 / math.sqrt(r["n"])
    mx = max(abs(x) for x in r["_reps"])
    out.append(chk("D1 v d=1 under S0: max |r| over reps", mx, f"< 5/sqrt(n) = {thr:.3g}",
                   mx < thr))
    r2 = one(rows, item="D3", variable="v", group_or_context="R2")["value"]
    out.append(chk("D3 R2 vs D1(d=1)^2 relative error", rel(r2, rho * rho), "< 10%",
                   rel(r2, rho * rho) < 0.10))
    d2 = one(rows, item="D2", variable="v", direction="row")["value"]
    tgt = -0.5 * math.log2(1 - rho * rho)
    out.append(chk("D2 row vs -0.5 log2(1-D1^2) relative error", rel(d2, tgt), "< 10%",
                   rel(d2, tgt) < 0.10))
    c4 = one(rows, item="C4", group_or_context="left1", baseline="S0")["value"]
    d3b = -0.5 * math.log2(1 - r2)
    out.append(chk("C4 left1 vs D3 bits relative error", rel(c4, d3b), "< 15%",
                   rel(c4, d3b) < 0.15))
    return out


def checks_T6(rows, th):
    out = []
    r = one(rows, item="D1", variable="v", direction="row", subset="all", baseline="S0", lag=1)
    out.append(chk("D1 v d=1", r["value"], f"not significant (|r| <= {5 / math.sqrt(r['n']):.3g})",
                   abs(r["value"]) <= 5 / math.sqrt(r["n"])))
    a = one(rows, item="D1", variable="|v|", direction="row", subset="all", baseline="S0", lag=1)
    out.append(chk("D1 |v| d=1", a["value"], "> 0.2", a["value"] > 0.2))
    ss = find(rows, item="C3", variable="I(s;s')", direction="row")
    nsig = sum(bool(x["significant"]) for x in ss)
    out.append(chk("C3 I(s;s') row d=1..8 significant count", nsig, f"== 0 of {len(ss)}",
                   nsig == 0))
    mm = one(rows, item="C3", variable="I(m;m')", direction="row", lag=1)
    out.append(chk("C3 I(m;m') row d=1", mm["value"], "> 0.05", mm["value"] > 0.05))
    return out


CHECKS = {"T1": checks_T1, "T2": checks_T2, "T3": checks_T3, "T4": checks_T4,
          "T5": checks_T5, "T6": checks_T6}


def run_one(name):
    i = int(name[1:])
    rng = np.random.default_rng([SEED, i])
    X, th = GENS[name](rng)
    E = np.full((R_SYN, W_SYN // B), 127, dtype=np.uint8)
    res = measure_matrix(X, E, B, {"matrix": name, "layer": "", "expert": "", "role": "synthetic"},
                         seed=SEED + i)
    checks = CHECKS[name](res["rows"], th)
    return name, {"shape": [R_SYN, W_SYN], "B": B, "seed": [SEED, i], "theory": th,
                  "checks": checks, "pass": all(c["pass"] for c in checks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/r1-r4/validation.json")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    names = args.only or list(GENS)
    with ProcessPoolExecutor(len(names)) as ex:
        results = dict(ex.map(run_one, names))
    out = {"all_pass": all(r["pass"] for r in results.values()), "tests": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)
    for n, r in results.items():
        print(n, "PASS" if r["pass"] else "FAIL")
        for c in r["checks"]:
            v = c["value"]
            v = f"{v:.6g}" if isinstance(v, float) else v
            print(f"   [{'ok' if c['pass'] else 'XX'}] {c['check']}: {v}  ({c['criterion']})")
    print("ALL PASS" if out["all_pass"] else "SOME FAILED")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
