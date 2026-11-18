"""Spec 010 (transform trials T1-T3) on DeepSeek FP4 expert matrices.

    PYTHONPATH=. python experiments/010-transforms/tx.py synth            # synthetic checks of every new method
    PYTHONPATH=. python experiments/010-transforms/tx.py base   --workers 48
    PYTHONPATH=. python experiments/010-transforms/tx.py t1     --workers 16
    PYTHONPATH=. python experiments/010-transforms/tx.py t2     --workers 24
    PYTHONPATH=. python experiments/010-transforms/tx.py t3     --workers 16

Sample: layers 0, 13, 26, 39; 4 experts per layer from numpy default_rng([20261119, 10, layer]).permutation(384).
All code lengths come from redundancy/increment.py (sequential adaptive, side information billed); the 008
parameters (results/fp4/increment_dev/params.json of 008) are used unchanged. The reference is the 008 chain
(default order): w1 coded with w3 as R7 partner, w3 and w2 without partner.
Results are appended to experiments/010-transforms/results/<stage>.jsonl (one line per task; reruns skip done tasks).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from redundancy import increment as I

E10 = Path("experiments/010-transforms")
RES = E10 / "results"
SAMPLES = "data/008/fp4/matrices.json"
LAYERS = [0, 13, 26, 39]
SEED = 20261119
P = json.load(open("experiments/008-scale-media/results/fp4/increment_dev/params.json"))
ORDER = ["R1c", "R1s", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R10"]     # 008 default order
COL_ORDER = ["R1c", "R1s", "R2", "R3", "R4", "R6", "R7", "R10"]           # column direction (no block patterns)
V16 = I.FP4_VALUES


def log2_fact(n):
    return math.lgamma(n + 1) / math.log(2)


def sample_experts():
    return {L: [int(e) for e in np.random.default_rng([SEED, 10, L]).permutation(384)[:4]] for L in LAYERS}


_AD = None


def read(name):
    global _AD
    if _AD is None:
        from redundancy.adapters.fp4 import FP4
        _AD = FP4({"samples": SAMPLES, "snapshot_env": "SNAPSHOT"})
    C, E = _AD._read(_AD.samples[name])
    return C.astype(np.int64), np.ascontiguousarray(E)


def deq(C, E, B=32):
    return (V16[C] * np.exp2(np.repeat(E.astype(np.float64), B, axis=1) - 127)).astype(np.float32)


def row_corr(A, B):
    a = A - A.mean(1, keepdims=True)
    b = B - B.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), 0.0)


def slim(res):
    return {k: res[k] for k in ("L", "increment", "total", "used", "if_forced", "raw_bits")}


def chain(C, E, partner=None, order=ORDER, params=None):
    cache = {}
    Pp = params or P
    return slim(I.run_chain(lambda fl: I.fp4_codelen(C, E, fl, Pp, order, partner=partner, cache=cache),
                            order, C.size * 4.25))


# ------------------------------------------------------------------ T1: neuron sign normalisation

def t1_flip(C1, E1, C3, E3, C2):
    r = row_corr(deq(C1, E1).astype(np.float64), deq(C3, E3).astype(np.float64))
    neg = r < 0
    C3n, C2n = C3.copy(), C2.copy()
    C3n[neg] ^= 8                      # sign bit of every code of w3 row i (scale unchanged)
    C2n[:, neg] ^= 8                   # and of w2 column i
    return C3n, C2n, neg


def swiglu(W1, W2, W3, x):
    a = W1 @ x
    h = (a / (np.float32(1) + np.exp(-a))) * (W3 @ x)
    return W2 @ h


def t1_invariance(C1, E1, C3, E3, C2, E2, C3n, C2n, seed=0, tokens=16):
    W1, W3, W2 = deq(C1, E1), deq(C3, E3), deq(C2, E2)
    W3n, W2n = deq(C3n, E3), deq(C2n, E2)
    x = np.random.default_rng(seed).standard_normal((W1.shape[1], tokens)).astype(np.float32)
    y0, y1 = swiglu(W1, W2, W3, x), swiglu(W1, W2n, W3n, x)
    return {"bitwise_identical": bool(y0.tobytes() == y1.tobytes()), "max_abs_diff": float(np.abs(y0 - y1).max()),
            "tokens": tokens, "dtype": "float32", "rows_flipped": None}


# ------------------------------------------------------------------ T2: cross-expert permutation alignment

def match_rows(VA, VB):
    """A row i <-> B row cb[i], maximising the summed correlation (Hungarian algorithm)."""
    from scipy.optimize import linear_sum_assignment
    Za = VA - VA.mean(1, keepdims=True)
    Zb = VB - VB.mean(1, keepdims=True)
    Za /= np.maximum(np.linalg.norm(Za, axis=1, keepdims=True), 1e-30)
    Zb /= np.maximum(np.linalg.norm(Zb, axis=1, keepdims=True), 1e-30)
    Cm = (Za.astype(np.float64) @ Zb.astype(np.float64).T)
    ra, cb = linear_sum_assignment(Cm, maximize=True)
    return cb, Cm[ra, cb], np.diag(Cm)


def aligned(A, cb):
    """partner array for B: row cb[i] of the result is row i of A."""
    out = np.empty_like(A)
    out[cb] = A
    return out


# ------------------------------------------------------------------ T3: row reordering + column-direction coding

def nn_order(F, metric):
    """Greedy nearest-neighbour tour over rows, starting at row 0."""
    n = len(F)
    if metric == "corr":
        Z = F - F.mean(1, keepdims=True)
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-30)
        S = Z @ Z.T
    else:   # L1 distance of scale vectors
        from scipy.spatial.distance import cdist
        S = -cdist(F, F, "cityblock")
    S = S.astype(np.float64)
    seen = np.zeros(n, dtype=bool)
    order = [0]
    seen[0] = True
    for _ in range(n - 1):
        s = S[order[-1]].copy()
        s[seen] = -np.inf
        j = int(np.argmax(s))
        order.append(j)
        seen[j] = True
    return np.array(order)


def scale_bits(E, ctx):
    """Scale stream (row-major order) with the scale above ('above') or to the left ('left') as context,
    hierarchical adaptive code as for the codes; 'none' = the 008 order-0 code."""
    if ctx == "none":
        return I.dirichlet_bits(E.ravel(), None, 256)
    c = np.full(E.shape, 256, dtype=np.int64)
    if ctx == "above":
        c[1:] = E[:-1]
    else:
        c[:, 1:] = E[:, :-1]
    return I.hier_bits(E.ravel().astype(np.int64), [np.zeros(E.size, dtype=np.int64), c.ravel()], 256,
                       P["beta"], P["nblocks"])


def col_codelen(C, E, partner=None, params=None, B=32):
    """Code length function for coding the matrix column by column (adjacent rows are neighbours).
    Uses fp4_codelen on the transpose with a per-position scale map (block = 1), then replaces its
    scale-stream term by the real stream (one byte per block, context = scale above). R7's partner is
    first sign-normalised per original row (the per-row sign is billed once, as in the 008 chain)."""
    Pp = dict(params or P)
    Pp["block"] = 1
    CT = np.ascontiguousarray(C.T)
    ST = np.ascontiguousarray(np.repeat(E, B, axis=1).T)
    pT, rs_bits = None, 0.0
    if partner is not None:
        rs = np.sign((V16[C] * V16[partner]).sum(axis=1))
        rs[rs == 0] = 1
        pn = partner.copy()
        pn[rs < 0] ^= 8
        pT = np.ascontiguousarray(pn.T)
        rs_bits = I.kt_bits(np.zeros(len(rs)), (rs > 0).astype(np.int64))
    wrong = {True: I.dirichlet_bits(ST.ravel(), None, 256), False: 8.0 * ST.size}
    true = {True: scale_bits(E, "above"), False: 8.0 * E.size}
    cache = {}

    def f(flags):
        b, info = I.fp4_codelen(CT, ST, flags, Pp, COL_ORDER, partner=pT, cache=cache)
        r1s = "R1s" in flags
        b = b - wrong[r1s] + true[r1s]
        if "R7" in flags and pT is not None:
            b += rs_bits
        return b, info
    return f


def col_chain(C, E, partner=None, params=None):
    return slim(I.run_chain(col_codelen(C, E, partner, params), COL_ORDER, C.size * 4.25))


# ------------------------------------------------------------------ tasks on the real sample

def task_base(name, partner_name):
    t0 = time.process_time()
    C, E = read(name)
    part = read(partner_name)[0] if partner_name else None
    return {"task": name, "partner": partner_name, "chain": chain(C, E, part), "cpu": time.process_time() - t0}


def task_t1(L, e):
    t0 = time.process_time()
    n = f"L{L}.E{e}."
    (C1, E1), (C2, E2), (C3, E3) = read(n + "w1"), read(n + "w2"), read(n + "w3")
    C3n, C2n, neg = t1_flip(C1, E1, C3, E3, C2)
    inv = t1_invariance(C1, E1, C3, E3, C2, E2, C3n, C2n, seed=L * 1000 + e)
    inv["rows_flipped"] = int(neg.sum())
    out = {"task": f"L{L}.E{e}", "invariance": inv, "rows": int(len(neg)),
           "w3_given_w1_before": chain(C3, E3, C1), "w3_given_w1_after": chain(C3n, E3, C1),
           "w1_given_w3_after": chain(C1, E1, C3n), "w3_after": chain(C3n, E3), "w2_after": chain(C2n, E2)}
    out["cpu"] = time.process_time() - t0
    return out


def task_t2(L, a, b):
    t0 = time.process_time()
    A1, EA1 = read(f"L{L}.E{a}.w1")
    A3, _ = read(f"L{L}.E{a}.w3")
    B1, EB1 = read(f"L{L}.E{b}.w1")
    B3, EB3 = read(f"L{L}.E{b}.w3")
    rng = np.random.default_rng([SEED, 2, L, a, b])
    out = {"task": f"L{L}.E{a}-E{b}", "perm_bits": log2_fact(B1.shape[0])}
    for kind in ("real", "shuffled"):
        if kind == "real":
            X1, X3 = B1, B3
        else:   # B's codes fully shuffled inside each matrix (scales stay)
            X1 = rng.permutation(B1.ravel()).reshape(B1.shape)
            X3 = rng.permutation(B3.ravel()).reshape(B3.shape)
        cb, mcorr, diag = match_rows(deq(A1, EA1), deq(X1, EB1))
        q = lambda v: {"median": float(np.median(v)), "p99": float(np.percentile(v, 99)),
                       "mean": float(np.mean(v)), "max": float(np.max(v))}
        rec = {"matched_corr": q(mcorr), "identity_corr": q(diag),
               "identity_fraction": float(np.mean(cb == np.arange(len(cb)))),
               "w1_given_A": chain(X1, EB1, aligned(A1, cb)), "w3_given_A": chain(X3, EB3, aligned(A3, cb))}
        if kind == "shuffled":
            rec["w1_008"] = chain(X1, EB1, X3)
            rec["w3_008"] = chain(X3, EB3)
        out[kind] = rec
    out["cpu"] = time.process_time() - t0
    return out


def task_t3(L, e):
    t0 = time.process_time()
    n = f"L{L}.E{e}."
    (C1, E1), (C3, E3) = read(n + "w1"), read(n + "w3")
    R = C1.shape[0]
    orders = {"original": np.arange(R),
              "nn_code_corr": nn_order(np.concatenate([V16[C1], V16[C3]], axis=1), "corr"),
              "nn_scale_dist": nn_order(np.concatenate([E1, E3], axis=1).astype(np.float64), "l1"),
              "random": np.random.default_rng([SEED, 3, L, e]).permutation(R)}
    out = {"task": f"L{L}.E{e}", "perm_bits": log2_fact(R)}
    for k, o in orders.items():
        c1, e1, c3, e3 = C1[o], E1[o], C3[o], E3[o]
        out[k] = {"w1": col_chain(c1, e1, c3), "w3": col_chain(c3, e3),
                  "scale_bits": {m: {"w1": scale_bits(e1, m), "w3": scale_bits(e3, m)} for m in ("none", "above", "left")}}
    out["cpu"] = time.process_time() - t0
    return out


def task_t3row(L, e):
    """T3 variant: rows reordered, codes still coded row by row (008 chain); the scale stream gets one more
    option, context = scale of the row above, used only if shorter (1 switch bit per matrix)."""
    t0 = time.process_time()
    n = f"L{L}.E{e}."
    (C1, E1), (C3, E3) = read(n + "w1"), read(n + "w3")
    R = C1.shape[0]
    orders = {"nn_code_corr": nn_order(np.concatenate([V16[C1], V16[C3]], axis=1), "corr"),
              "nn_scale_dist": nn_order(np.concatenate([E1, E3], axis=1).astype(np.float64), "l1"),
              "random": np.random.default_rng([SEED, 3, L, e]).permutation(R)}
    out = {"task": f"L{L}.E{e}", "perm_bits": log2_fact(R)}
    for k, o in orders.items():
        c1, e1, c3, e3 = C1[o], E1[o], C3[o], E3[o]
        rec = {"w1": chain(c1, e1, c3), "w3": chain(c3, e3)}
        for m, ee in (("w1", e1), ("w3", e3)):
            s0, sa = scale_bits(ee, "none"), scale_bits(ee, "above")
            rec[m + "_scale"] = {"order0": s0, "above": sa, "L_with_scale_switch": rec[m]["L"][-1] - s0 + min(s0, sa) + 1.0}
        out[k] = rec
    out["cpu"] = time.process_time() - t0
    return out


# ------------------------------------------------------------------ synthetic checks

def synth_codes(rng, shape, p_zero=0.12):
    mag = rng.choice(8, size=shape, p=np.array([p_zero] + [(1 - p_zero) / 7] * 7))
    return mag + 8 * rng.integers(0, 2, size=shape)


def synth():
    rng = np.random.default_rng(1)
    out = {}
    R, W = 256, 512
    E = np.full((R, W // 32), 120, dtype=np.uint8)
    # T1: w3 rows = +/- (w1 rows) with noise; flipping makes all row correlations positive, output invariant
    C1 = synth_codes(rng, (R, W))
    s = rng.integers(0, 2, size=R)
    C3 = np.where(rng.random((R, W)) < 0.8, C1 ^ (8 * s[:, None]), synth_codes(rng, (R, W)))
    C2 = synth_codes(rng, (W, R))
    E2 = np.full((W, R // 32), 120, dtype=np.uint8)
    C3n, C2n, neg = t1_flip(C1, E, C3, E, C2)
    inv = t1_invariance(C1, E, C3, E, C2, E2, C3n, C2n)
    before, after = chain(C3, E, C1), chain(C3n, E, C1)
    rs_bits = I.kt_bits(np.zeros(R), 1 - s)
    out["T1"] = {"rows_flipped": int(neg.sum()), "rows_with_negative_sign": int(s.sum()),
                 "flips_match_truth": bool(np.array_equal(neg, s == 1)), "invariance": inv,
                 "saving_bits": before["L"][-1] - after["L"][-1], "expected_about_bits": rs_bits,
                 "pass": bool(np.array_equal(neg, s == 1) and inv["bitwise_identical"]
                              and abs((before["L"][-1] - after["L"][-1]) - rs_bits) < 0.25 * rs_bits + 20)}
    # T2: B = A with rows permuted and 30% of codes redrawn -> the matching finds the permutation and the
    # saving exceeds the permutation cost; independent B -> matched correlations as on the shuffled baseline
    A = synth_codes(rng, (R, W))
    perm = rng.permutation(R)
    Bm = np.where(rng.random((R, W)) < 0.7, A[perm], synth_codes(rng, (R, W)))
    cb, mc, _ = match_rows(deq(A, E), deq(Bm, E))
    hit = float(np.mean(cb == np.argsort(perm)))
    gain = chain(Bm, E)["L"][-1] - chain(Bm, E, aligned(A, cb))["L"][-1]
    Bi = synth_codes(rng, (R, W))
    cbi, mci, _ = match_rows(deq(A, E), deq(Bi, E))
    Bs = rng.permutation(Bi.ravel()).reshape(R, W)
    cbs, mcs, _ = match_rows(deq(A, E), deq(Bs, E))
    gain_i = chain(Bi, E)["L"][-1] - chain(Bi, E, aligned(A, cbi))["L"][-1]
    out["T2"] = {"related": {"permutation_recovered": hit, "saving_bits": gain, "perm_bits": log2_fact(R),
                             "matched_corr_median": float(np.median(mc))},
                 "independent": {"matched_corr_median": float(np.median(mci)), "baseline_median": float(np.median(mcs)),
                                 "saving_bits": gain_i},
                 # on independent data the matching itself is chosen with B, so the coded B gains up to the
                 # information in the permutation; the net (gain - permutation bits) must stay negative
                 "pass": bool(hit > 0.99 and gain > log2_fact(R) and abs(np.median(mci) - np.median(mcs)) < 0.02
                              and gain_i < log2_fact(R))}
    # column direction: vertical copies -> large R3 in column order; scale copied from the row above
    Cv = synth_codes(rng, (R, W))
    for r in range(1, R):
        keep = rng.random(W) < 0.8
        Cv[r, keep] = Cv[r - 1, keep]
    Ev = np.empty((R, W // 32), dtype=np.uint8)
    Ev[0] = rng.integers(110, 130, size=W // 32)
    for r in range(1, R):
        Ev[r] = np.where(rng.random(W // 32) < 0.9, Ev[r - 1], rng.integers(110, 130, size=W // 32))
    colc, rowc = col_chain(Cv, Ev), chain(Cv, Ev)
    # scale stream check on a larger scale matrix (as many scales as a third of a real matrix):
    # 90% copied from the row above -> about 0.9 bit instead of log2(20) = 4.3 bit per scale
    Eb = np.empty((2048, 64), dtype=np.uint8)
    Eb[0] = rng.integers(110, 130, size=64)
    for r in range(1, 2048):
        Eb[r] = np.where(rng.random(64) < 0.9, Eb[r - 1], rng.integers(110, 130, size=64))
    sb = {m: scale_bits(Eb, m) / Eb.size for m in ("none", "above", "left")}
    out["column_direction"] = {"col_R3": colc["increment"]["R3"], "row_R3": rowc["increment"]["R3"],
                               "col_total": colc["total"], "row_total": rowc["total"], "scale_bits": sb,
                               # row-direction R3 is small but not 0: the slowly drifting rows make the
                               # finer adaptive level useful even without dependence along the row
                               "pass": bool(colc["increment"]["R3"] > 0.2 and colc["increment"]["R3"] > 10 * rowc["increment"]["R3"]
                                            and sb["above"] < 0.5 * sb["none"] and sb["left"] > 0.9 * sb["none"])}
    iid = synth_codes(rng, (R, W))
    ci = col_chain(iid, E)
    out["column_direction_iid"] = {"increments": ci["increment"],
                                   "pass": bool(all(abs(ci["increment"][p]) < 0.002 for p in ("R2", "R3", "R4", "R6", "R7")))}
    # T3: rows from 8 prototypes, shuffled -> nearest-neighbour order brings similar rows together
    protos = synth_codes(rng, (8, W))
    lab = rng.integers(0, 8, size=R)
    Cc = np.where(rng.random((R, W)) < 0.7, protos[lab], synth_codes(rng, (R, W)))
    Ec = (115 + lab[:, None] + np.zeros((1, W // 32), dtype=np.int64)).astype(np.uint8)
    o_nn = nn_order(V16[Cc], "corr")
    o_sc = nn_order(Ec.astype(np.float64), "l1")
    o_rand = rng.permutation(R)
    adj = lambda o: float(np.mean(lab[o][1:] == lab[o][:-1]))
    g_nn, g_rand = col_chain(Cc[o_nn], Ec[o_nn]), col_chain(Cc[o_rand], Ec[o_rand])
    out["T3"] = {"same_cluster_adjacent": {"nn_code_corr": adj(o_nn), "nn_scale_dist": adj(o_sc), "random": adj(o_rand)},
                 "col_total_nn": g_nn["total"], "col_total_random": g_rand["total"],
                 "R3_nn": g_nn["increment"]["R3"], "R3_random": g_rand["increment"]["R3"],
                 "pass": bool(adj(o_nn) > 0.9 and adj(o_sc) > 0.9 and adj(o_rand) < 0.3
                              and g_nn["total"] > g_rand["total"] + 0.05)}
    out["all_pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict) and "pass" in v)
    return out


# ------------------------------------------------------------------ driver

def done(path):
    d = set()
    if path.exists():
        for line in open(path):
            try:
                d.add(json.loads(line)["task"])
            except (json.JSONDecodeError, KeyError):
                pass
    return d


def pool(jobs, workers, path):
    have = done(path)
    jobs = [j for j in jobs if j[0] not in have]
    print(f"{len(jobs)} tasks to run, {workers} workers", flush=True)
    with ProcessPoolExecutor(workers) as ex, open(path, "a") as f:
        futs = {ex.submit(fn, *args): tid for tid, fn, args in jobs}
        for fu in as_completed(futs):
            r = fu.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print(time.strftime("%H:%M:%S"), "done", futs[fu], f"{r.get('cpu', 0):.0f} cpu-s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["synth", "base", "t1", "t2", "t3", "t3row"])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    ex = sample_experts()
    if a.stage == "synth":
        res = synth()
        json.dump(res, open(RES / "synthetic_checks.json", "w"), indent=1)
        print(json.dumps(res, indent=1))
        return
    json.dump({"layers": LAYERS, "seed": SEED, "experts": ex}, open(RES / "sample.json", "w"), indent=1)
    if a.stage == "base":
        jobs = []
        for L, es in ex.items():
            for e in es:
                for r in ("w1", "w2", "w3"):
                    n = f"L{L}.E{e}.{r}"
                    jobs.append((n, task_base, (n, f"L{L}.E{e}.w3" if r == "w1" else None)))
    elif a.stage == "t1":
        jobs = [(f"L{L}.E{e}", task_t1, (L, e)) for L, es in ex.items() for e in es]
    elif a.stage == "t2":
        jobs = [(f"L{L}.E{x}-E{y}", task_t2, (L, x, y)) for L, es in ex.items()
                for i, x in enumerate(es) for y in es[i + 1:]]
    elif a.stage == "t3":
        jobs = [(f"L{L}.E{e}", task_t3, (L, e)) for L, es in ex.items() for e in es]
    else:
        jobs = [(f"L{L}.E{e}", task_t3row, (L, e)) for L, es in ex.items() for e in es]
    pool(jobs, a.workers, RES / f"{a.stage}.jsonl")


if __name__ == "__main__":
    main()
