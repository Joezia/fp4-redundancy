"""Spec 015: k-gram code length of the FP4 codes (supplementary indicator for R3).

    PYTHONPATH=. python experiments/015-kgram/kgram.py run {deepseek|kimi} [--workers N]
    PYTHONPATH=. python experiments/015-kgram/kgram.py summarize
    PYTHONPATH=. python experiments/015-kgram/kgram.py selftest

Samples as in 013 (008: DeepSeek 516 matrices = plan B first 172 experts; 009: Kimi-K3 552 matrices = plan_552).
Codes C (16 raw codes, +0 and -0 kept apart) and block scales E are read with redundancy.adapters.fp4; nothing
else from the framework is used for the computation.

Code length: one adaptive count table per context, every symbol starting at 0.5 (KT), each code coded in turn
and its count incremented afterwards; every matrix starts from empty tables. The total of this sequential code is
order-independent within a context, so it is computed exactly in closed form:
    bits = -sum_ctx [ sum_s (lnG(n_s + 1/2) - lnG(1/2)) - (lnG(N + 8) - lnG(8)) ] / ln 2.
Context = the k codes immediately before the code in the same row (k = 0..4); at the start of a row the shorter
available context is used (positions before the row start are filled with the value 16, which no code takes, so a
shorter context is a context of its own). Optionally the block-scale byte E of the code is added to the context.
Data: (a) original; (b) the 32 codes of every block shuffled inside the block; (c) every row shuffled; the shuffles
use a fixed seed per matrix. The scale-context set is computed on (a) only.
Reported in bits per code (the scale bytes themselves are not coded here).
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
from scipy.special import gammaln  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = HERE / "results"
SEED = 20261124
KS = [0, 1, 2, 3, 4]
B = 32
MODELS = {
    "deepseek": dict(plan="experiments/008-scale-media/results/fp4/B/plan.json", first=172,
                     items="experiments/008-scale-media/results/fp4_increments_B.json",
                     samples="data/008/fp4/matrices.json", snapshot_env="SNAPSHOT"),
    "kimi": dict(plan="experiments/009-kimi-k3/results/plan_552.json", first=None, items=None,
                 samples="data/009/kimi/matrices.json", snapshot_env="KIMI_SNAPSHOT"),
}


def tasks_of(model):
    m = MODELS[model]
    plan = json.load(open(ROOT / m["plan"]))
    tasks = plan["tasks"][:m["first"]] if m["first"] else plan["tasks"]
    if m["items"]:
        ents = sorted(e for t in tasks for e in t["entries"])
        assert ents == sorted(json.load(open(ROOT / m["items"]))["items"]), "sample differs from 008"
    return tasks


def kt_bits_dense(key, sym, n_ctx, K=16):
    """Closed-form sequential KT code length; key in 0..n_ctx-1, sym in 0..K-1."""
    cnt = np.bincount(key * K + sym, minlength=n_ctx * K).reshape(n_ctx, K)
    tot = cnt.sum(1)
    used = tot > 0
    c = cnt[used]
    lnp = (gammaln(c + 0.5) - gammaln(0.5)).sum() - (gammaln(tot[used] + K * 0.5) - gammaln(K * 0.5)).sum()
    return float(-lnp / math.log(2))


def context_keys(C, k):
    """C (R, W) int codes 0..15 -> key of the previous k codes of the row (pad 16), in 0..17^k - 1."""
    R, W = C.shape
    key = np.zeros((R, W), dtype=np.int64)
    for j in range(1, k + 1):
        prev = np.full((R, W), 16, dtype=np.int64)
        prev[:, j:] = C[:, :-j]
        key = key * 17 + prev
    return key


def shuffle_blocks(C, rng):
    R, W = C.shape
    X = C.reshape(R, W // B, B)
    order = np.argsort(rng.random(X.shape), axis=2)
    return np.take_along_axis(X, order, axis=2).reshape(R, W)


def shuffle_rows(C, rng):
    order = np.argsort(rng.random(C.shape), axis=1)
    return np.take_along_axis(C, order, axis=1)


def matrix_bits(C, E):
    """All values for one matrix: dict variant -> {k: bits}."""
    R, W = C.shape
    assert W % B == 0
    seed = SEED
    data = {"a": C, "b": shuffle_blocks(C, np.random.default_rng([seed, 1])),
            "c": shuffle_rows(C, np.random.default_rng([seed, 2]))}
    out = {}
    for v, X in data.items():
        assert np.array_equal(np.sort(X, axis=1), np.sort(C, axis=1))      # same multiset per row
        s = X.ravel()
        out[v] = {k: kt_bits_dense(context_keys(X, k).ravel(), s, 17 ** k) for k in KS}
    Ew = np.repeat(E.astype(np.int64), B, axis=1)
    eu, einv = np.unique(Ew, return_inverse=True)
    einv = einv.reshape(R, W)
    s = C.ravel()
    out["a_scale"] = {k: kt_bits_dense((einv * 17 ** k + context_keys(C, k)).ravel(), s, len(eu) * 17 ** k)
                      for k in KS}
    return out


def expert_task(model, task):
    from redundancy.adapters.fp4 import FP4
    m = MODELS[model]
    ad = FP4({"samples": str(ROOT / m["samples"]), "snapshot_env": m["snapshot_env"]})
    res = []
    for name in task["entries"]:
        t0 = time.time()
        C, E = ad._read(ad.samples[name])
        C = C.astype(np.int64)
        bits = matrix_bits(C, E)
        res.append({"matrix": name, "layer": task["layer"], "expert": task["expert"], "shape": list(C.shape),
                    "n_codes": int(C.size), "raw_bits": float(4 * C.size), "bits": bits,
                    "bits_per_code": {v: {k: b / C.size for k, b in d.items()} for v, d in bits.items()},
                    "seconds": time.time() - t0})
    return res


def cmd_run(a):
    RES.mkdir(parents=True, exist_ok=True)
    path = RES / f"kgram_{a.model}_items.jsonl"
    done = {json.loads(l)["matrix"] for l in open(path)} if path.exists() else set()
    todo = [t for t in tasks_of(a.model) if not all(e in done for e in t["entries"])]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.model}: {len(todo)} experts to run", flush=True)
    t0 = time.time()
    with open(path, "a") as fo, ProcessPoolExecutor(a.workers) as ex:
        for i, rs in enumerate(ex.map(expert_task, [a.model] * len(todo), todo), 1):
            for r in rs:
                fo.write(json.dumps(r) + "\n")
            fo.flush()
            if i % 20 == 0:
                print(f"{i}/{len(todo)} {time.time() - t0:.0f} s", flush=True)
    print(f"RUN_DONE {a.model} wall {time.time() - t0:.0f} s", flush=True)


def cmd_summarize(a):
    for model in MODELS:
        want = [e for t in tasks_of(model) for e in t["entries"]]
        rows = {}
        for l in open(RES / f"kgram_{model}_items.jsonl"):
            r = json.loads(l)
            rows[r["matrix"]] = r
        assert all(e in rows for e in want), f"{model}: missing matrices"
        rows = [rows[e] for e in want]
        n = sum(r["n_codes"] for r in rows)
        table = {}
        for v in ("a", "b", "c", "a_scale"):
            table[v] = {}
            for k in KS:
                pooled = sum(r["bits"][v][str(k)] for r in rows) / n
                med = float(np.median([r["bits_per_code"][v][str(k)] for r in rows]))
                table[v][k] = {"pooled_bits_per_code": pooled, "median_bits_per_code": med,
                               "pooled_saving_pct_of_4bit": 100 * (1 - pooled / 4)}
        out = {"model": model, "n_matrices": len(rows), "n_codes": n, "seed": SEED,
               "unit": "bits per code (raw 4); pooled = sum of bits / sum of codes over all matrices "
                       "(= weighting by raw bits); median over matrices",
               "variants": {"a": "original", "b": "codes shuffled inside every 32-code block",
                            "c": "every row shuffled", "a_scale": "original, context = (block scale, previous k)"},
               "table": table, "seconds_total": float(sum(r["seconds"] for r in rows)),
               "per_matrix": {r["matrix"]: {"shape": r["shape"], "bits_per_code": r["bits_per_code"]} for r in rows}}
        json.dump(out, open(RES / f"kgram_{model}.json", "w"), indent=1)
        print(f"== {model}: {len(rows)} matrices, cpu {out['seconds_total']:.0f} s")
        print("k | (a) | (b) | (c) | +scale   [pooled; median]")
        for k in KS:
            print(k, " | ".join(f"{table[v][k]['pooled_bits_per_code']:.5f} ({table[v][k]['median_bits_per_code']:.5f})"
                                for v in ("a", "b", "c", "a_scale")))


def cmd_selftest(a):
    """Closed form == the sequential KT coder written out literally, on tiny synthetic codes."""
    rng = np.random.default_rng(0)
    C = rng.integers(0, 16, size=(5, 64))
    E = rng.integers(120, 124, size=(5, 2))
    for k in KS:
        key = context_keys(C, k)
        closed = kt_bits_dense(key.ravel(), C.ravel(), 17 ** k)
        tables, bits = {}, 0.0
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                ctx = tuple(C[i, max(0, j - k):j])
                t = tables.setdefault(ctx, np.full(16, 0.5))
                bits -= math.log2(t[C[i, j]] / t.sum())
                t[C[i, j]] += 1
        print(f"k={k}: closed {closed:.6f} sequential {bits:.6f}")
        assert abs(closed - bits) < 1e-6
        from redundancy.increment import dirichlet_bits
        assert abs(dirichlet_bits(C.ravel(), key.ravel(), 16) - closed) < 1e-6
    r = matrix_bits(C, E)
    assert all(abs(r[v][0] - r["a"][0]) < 1e-6 for v in ("b", "c")), "k = 0 must not depend on the order"
    print("selftest ok; k=0 equal for (a), (b), (c):", r["a"][0], r["b"][0], r["c"][0])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model", choices=list(MODELS))
    r.add_argument("--workers", type=int, default=32)
    r.add_argument("--limit", type=int, default=None)
    sub.add_parser("summarize")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
