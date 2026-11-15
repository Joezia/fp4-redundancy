"""Spec 005 section 7: synthetic checks TA, TB1-TB5, TC1-TC4 (no model data).

Every construction goes through the same `r9` functions as the model matrices (`a_matrix`,
`b1_matrix`, `b2_classes`, `c1_matrix`, `inr_fit`). Shape 1024 x 2048, B = 32, constant scale 127,
quantiser and independent base as in spec 003 (1.68 * N(0, 1), merged zero).
Usage: python -m diagnostics.r9_check [--out experiments/005-r9/validation.json] [--jobs 10]
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from . import r9  # noqa: E402
from .ledger import merge_zero  # noqa: E402
from .r58 import R52_K, holdout_lowrank, shuffle  # noqa: E402
from .r58_check import _lowrank  # noqa: E402
from .synth import code_probs, gen_T3, quantize  # noqa: E402

R, W, B = 1024, 2048, 32
SEED = 20261115
TC_BATCH = 16384


def chk(name, value, criterion, ok):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    return {"check": name, "value": value, "criterion": criterion, "pass": bool(ok)}


def rng(i):
    return np.random.default_rng([SEED, i])


def const_E(shape=(R, W)):
    return np.full((shape[0], shape[1] // B), 127, dtype=np.uint8)


def merged_probs(sigma):
    p = code_probs(sigma).copy()
    p[0] += p[8]
    p[8] = 0.0
    return p / p.sum()


P_N = merged_probs(1.68)
P_E = np.where(r9.ODD, 0.0, P_N)
P_E = P_E / P_E.sum()


def base(g, shape=(R, W)):
    return merge_zero(quantize(1.68 * g.standard_normal(shape)))


def mixture_blocks(g, probs, w, nb):
    z = g.choice(len(w), size=nb, p=w)
    out = np.empty((nb, B), dtype=np.uint8)
    for k, p in enumerate(probs):
        idx = np.nonzero(z == k)[0]
        out[idx] = g.choice(16, size=(len(idx), B), p=p)
    return out


def theory_gain(probs, w, g, n=40_000):
    """Section 7: mean[-log2 prod pbar] - mean[-log2 sum_z w_z prod p_z] over n sampled blocks, / B."""
    blk = mixture_blocks(g, probs, w, n)
    pbar = sum(wk * p for wk, p in zip(w, probs))
    with np.errstate(divide="ignore"):
        lp = [np.log(p) for p in probs]
        lbar = np.log(pbar)
    a = -lbar[blk].sum(axis=1)
    comp = np.stack([math.log(wk) + l[blk].sum(axis=1) for wk, l in zip(w, lp)], axis=1)
    b = -r9._lse(comp)
    return float((a - b).mean() / LN2_ / B)


LN2_ = math.log(2.0)


def ta_data(g):
    return mixture_blocks(g, [P_N, P_E], [0.7, 0.3], R * W // B).reshape(R, W)


def t3_data(g):
    return merge_zero(gen_T3(g)[0])


# ----------------------------------------------------------------------------- tests

def t_TA():
    Cp = ta_data(rng(1))
    E = const_E()
    Cs = shuffle(Cp, E, "S0E", rng(2))[0]
    rows, mat = r9.a_matrix(Cp, E, Cs)
    r = rows[0]
    return [chk("A1 share of blocks with n_odd = 0 (real)", r["frac0_real"], "in [0.29, 0.31]",
                0.29 <= r["frac0_real"] <= 0.31),
            chk("A1 binomial share of n_odd = 0", r["frac0_binom"], "< 0.001", r["frac0_binom"] < 0.001),
            chk("A1 S0E share of n_odd = 0 (info)", r["frac0_S0E"], "info", True),
            chk("A3 net = gain - H(g|E)/B", {"net": mat["net"], "gain": mat["gain"],
                "label_cost": mat["label_cost"]}, "net in [0.13, 0.15]", 0.13 <= mat["net"] <= 0.15),
            chk("A2 g distribution real (info)", [r[f"g{j}_real"] for j in range(5)], "info", True)]


def t_TB1():
    Cp = base(rng(3))
    res = r9.b1_matrix(Cp, const_E(), 11)
    gains = {K: res[K]["gain"] for K in res}
    ok = all(-0.001 <= gains[K] <= 0.0005 for K in (2, 4, 8, 16))
    return [chk("B1 gain(K), K = 2..16 (32 info)", gains, "K=2..16 in [-0.001, 0.0005]", ok)]


def t_TB2():
    g = rng(4)
    Cp = t3_data(g)
    q1, q2 = merged_probs(1.2), merged_probs(2.2)
    th = theory_gain([q1, q2], [0.5, 0.5], rng(5))
    res = r9.b1_matrix(Cp, const_E(), 12)
    g2, g16 = res[2]["gain"], res[16]["gain"]
    out = [chk("B1 gain(2) vs theory", {"gain2": g2, "theory": th, "rel_diff": abs(g2 - th) / th},
               "relative difference < 2%", abs(g2 - th) / th < 0.02),
           chk("B1 gain(16) >= gain(2) - 0.001", {"gain16": g16, "gain2": g2,
               "all": {K: res[K]["gain"] for K in res}}, ">=", g16 >= g2 - 0.001)]
    # TB4: S2 of the same data
    Cs = shuffle(Cp, const_E(), "S2", rng(6))[0]
    res2 = r9.b1_matrix(Cs, const_E(), 12)
    d = max(abs(res2[K]["codelen"] - res[K]["codelen"]) for K in res)
    out_b4 = [chk("TB4 max |codelen(S2) - codelen(TB2)| over K", d, "< 1e-9", d < 1e-9)]
    # TB5: sequential posterior prediction vs mixture marginal (K = 2 model fitted on TB2)
    H = r9.block_hist(Cp)
    A = H.sum(axis=0) > 0
    m = r9.em_fit(H[:, A], 2, r9.block_key(Cp), r9.rng_for(12, 2, 0, 0))
    pick = rng(7).choice(len(H), 100, replace=False)
    err = r9.sequential_check(Cp.reshape(-1, B)[pick].astype(np.int64), m, A)
    out_b5 = [chk("TB5 max relative error, sequential product vs marginal (100 blocks)", err,
                  "< 1e-9", err < 1e-9)]
    return {"TB2": out, "TB4": out_b4, "TB5": out_b5}


def t_TB3():
    Cp = ta_data(rng(8))
    th = theory_gain([P_N, P_E], [0.7, 0.3], rng(9))
    res = r9.b1_matrix(Cp, const_E(), 13)
    g2 = res[2]["gain"]
    cls = r9.b2_classes(Cp, const_E(), 13, 2)
    odd = sorted(c["odd_share"] for c in cls)
    pri = sorted((c["prior"] for c in cls), reverse=True)
    return [chk("B1 gain(2) vs theory", {"gain2": g2, "theory": th, "rel_diff": abs(g2 - th) / th},
                "relative difference < 3%", abs(g2 - th) / th < 0.03),
            chk("B2 (K=2) min odd share of a type", odd, "< 0.005", odd[0] < 0.005),
            chk("B2 (K=2) priors vs (0.7, 0.3)", pri, "each within 0.02",
                abs(pri[0] - 0.7) < 0.02 and abs(pri[1] - 0.3) < 0.02)]


def _c(Cp, seed, batch=TC_BATCH, inr=True):
    E = const_E()
    Cs = shuffle(Cp, E, "S0E", rng(seed + 100))[0]
    c1 = r9.c1_matrix(Cp, E, seed, Cs=Cs)
    c2 = r9.inr_fit(Cp, E, seed, {"batch": batch}) if inr else None
    return c1, c2


def _c2_view(c2):
    return {k: c2[k] for k in ("gain", "cost", "net", "base_test", "net_test", "best_epoch")}


def t_TC1():
    Cp = base(rng(10))
    c1, c2 = _c(Cp, 21)
    return [chk("C1 max hold-out R2", {"max_r2": c1["max_r2"], "r2": c1["r2"],
                "r2_S0E": c1["r2_S0E"]}, "< 0.001", c1["max_r2"] < 0.001),
            chk("C2 gain", _c2_view(c2), "in [0, 0.001]", 0 <= c2["gain"] <= 0.001)]


def t_TC2():
    Cp = merge_zero(_lowrank(False, SEED + 40))
    c1, c2 = _c(Cp, 22)
    return [chk("C1 max hold-out R2", {"max_r2": c1["max_r2"], "r2": c1["r2"]}, "< 0.005",
                c1["max_r2"] < 0.005),
            chk("C2 gain", _c2_view(c2), ">= 0.05", c2["gain"] >= 0.05)]


def tc3_data(g):
    U = g.standard_normal((R, 4))
    V = g.standard_normal((W, 4))
    z = U @ V.T / 2
    s = np.exp(0.5 * z)
    s = s / math.sqrt((s * s).mean())
    return merge_zero(quantize(1.68 * s * g.standard_normal((R, W))))


def t_TC3():
    Cp = tc3_data(rng(11))
    E = const_E()
    v = r9.E2M1[Cp]
    r2v = holdout_lowrank(v - v.mean(), r9.rng_for(23, 5), ks=R52_K)[0]
    c1, c2 = _c(Cp, 23)
    st = c1["strict"] or {}
    return [chk("003-R5.2 max hold-out R2 of signed v", {"max": max(r2v.values()), "r2": r2v},
                "< 0.001", max(r2v.values()) < 0.001),
            chk("C1 max hold-out R2", {"max_r2": c1["max_r2"], "r2": c1["r2"]}, ">= 0.2",
                c1["max_r2"] >= 0.2),
            chk("C1 strict gain", st, ">= 0.15", st.get("gain", -1) >= 0.15),
            chk("C2 gain", _c2_view(c2), ">= 0.18", c2["gain"] >= 0.18)]


def tc4_data(g):
    a = g.standard_normal(R)
    b = g.standard_normal(W)
    s = np.exp(0.3 * a)[:, None] * np.exp(0.3 * b)[None, :]
    s = s / math.sqrt((s * s).mean())
    return merge_zero(quantize(1.68 * s * g.standard_normal((R, W))))


def t_TC4():
    Cp = tc4_data(rng(12))
    c1, c2 = _c(Cp, 24)
    st = c1["strict"] or {}
    return [chk("C1 strict gain", {**st, "r2": c1["r2"]}, ">= 0.14", st.get("gain", -1) >= 0.14),
            chk("C2 gain", _c2_view(c2), ">= 0.14", c2["gain"] >= 0.14)]


TESTS = {"TA": t_TA, "TB1": t_TB1, "TB2": t_TB2, "TB3": t_TB3, "TC1": t_TC1, "TC2": t_TC2,
         "TC3": t_TC3, "TC4": t_TC4}


def run(name):
    t0 = time.time()
    out = TESTS[name]()
    return name, out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/005-r9/validation.json")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    names = args.only or list(TESTS)
    tests = {}
    with ProcessPoolExecutor(min(args.jobs, len(names))) as ex:
        for name, out, sec in ex.map(run, names):
            if isinstance(out, dict):
                for k, v in out.items():
                    tests[k] = {"checks": v, "seconds": sec}
            else:
                tests[name] = {"checks": out, "seconds": sec}
            print(name, f"{sec:.0f}s", flush=True)
    for k in sorted(tests):
        for c in tests[k]["checks"]:
            print(f"{k:5s} {'PASS' if c['pass'] else 'FAIL'} {c['check']}: {json.dumps(c['value'], default=float)[:300]}")
    order = ["TA", "TB1", "TB2", "TB3", "TB4", "TB5", "TC1", "TC2", "TC3", "TC4"]
    tests = {k: tests[k] for k in order if k in tests}
    all_pass = all(c["pass"] for t in tests.values() for c in t["checks"]) and len(tests) == len(order)
    if args.only:
        print("partial run; validation.json not written; all_pass(partial) =", all_pass)
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"seed": SEED, "shape": [R, W], "B": B, "numpy_version": np.__version__,
                   "tests": tests, "all_pass": all_pass}, f, indent=1, default=float)
    print("all_pass", all_pass)


if __name__ == "__main__":
    main()
