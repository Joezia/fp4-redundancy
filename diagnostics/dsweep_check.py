"""Spec 006 section 4: synthetic checks T1, T2 of the dictionary sweep (no model data).

Shape 1024 x 2048, B = 32, constant scale 127 (so S0E is the whole-matrix permutation baseline);
all 65,536 blocks in random order, first half fit, second half evaluate (`dsweep.sweep_variant`).
Usage: python -m diagnostics.dsweep_check [--out experiments/006-dict-sweep/validation.json]
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

from . import dsweep  # noqa: E402
from .ledger import merge_zero  # noqa: E402
from .synth import quantize  # noqa: E402

R, W, B = 1024, 2048, 32
SEED = 20261115


def const_E():
    return np.full((R, W // B), 127, dtype=np.uint8)


def t1_data():
    """Independent base (003): quantize(1.68 N(0,1))."""
    rng = np.random.default_rng(SEED + 10)
    return merge_zero(quantize(1.68 * rng.standard_normal((R, W))))


def t2_data():
    """003's T6: 48 unit atoms, 2 per block, signal share 0.6."""
    rng = np.random.default_rng(SEED + 60)
    atoms = rng.standard_normal((48, B))
    atoms /= np.sqrt((atoms * atoms).sum(axis=1, keepdims=True))
    nbk = R * W // B
    a1 = rng.integers(48, size=nbk)
    a2 = (a1 + rng.integers(1, 48, size=nbk)) % 48
    c = rng.standard_normal((nbk, 2))
    sig = 4 * (c[:, :1] * atoms[a1] + c[:, 1:] * atoms[a2])
    y = 1.68 * (math.sqrt(0.6) * sig + math.sqrt(0.4) * rng.standard_normal((nbk, B)))
    return merge_zero(quantize(y).reshape(R, W))


DATA = {"T1": (t1_data, 1), "T2": (t2_data, 2)}


def _job(arg):
    name, variant = arg
    t0 = time.time()
    make, seed = DATA[name]
    res = dsweep.sweep_variant(make(), const_E(), seed, variant)
    return name, variant, res, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/006-dict-sweep/validation.json")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(args.jobs) as ex:
        for name, variant, r, sec in ex.map(_job, [(n, v) for n in DATA for v in ("real", "S0E")]):
            res[(name, variant)] = r
            print(f"{name} {variant}: {sec:.0f}s", flush=True)
    out = {"shape": [R, W], "B": B, "seed": SEED, "baseline": "whole-matrix permutation (S0E with a "
           "constant scale)", "tests": {}}
    for name in DATA:
        real, base = res[(name, "real")]["dict"], res[(name, "S0E")]["dict"]
        diff = {f"K={K},s={s}": real[(K, s)] - base[(K, s)] for (K, s) in real}
        vals = {f"K={K},s={s}": {"real": real[(K, s)], "baseline": base[(K, s)]} for (K, s) in real}
        if name == "T1":
            checks = [{"check": "|real - baseline| for all 12 (K, s)", "value": diff,
                       "criterion": "all < 0.005", "pass": all(abs(d) < 0.005 for d in diff.values())}]
        else:
            checks = [{"check": "real - baseline for all 12 (K, s)", "value": diff,
                       "criterion": "all > 0.05", "pass": all(d > 0.05 for d in diff.values())},
                      {"check": "real - baseline at (K = 64, s = 2)", "value": diff["K=64,s=2"],
                       "criterion": "> 0.2", "pass": diff["K=64,s=2"] > 0.2}]
        out["tests"][name] = {"checks": checks, "R2": vals,
                              "R2_pca": {"real": res[(name, "real")]["pca"],
                                         "baseline": res[(name, "S0E")]["pca"]}}
        for c in checks:
            print(name, "PASS" if c["pass"] else "FAIL", c["check"], json.dumps(c["value"])[:400])
    out["all_pass"] = all(c["pass"] for t in out["tests"].values() for c in t["checks"])
    out["seconds"] = time.time() - t0
    print("all_pass", out["all_pass"], f"{out['seconds']:.0f}s")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
