"""S3-b: SeedLM made lossless ("approximation + residual").

Side information: the S3-a seeds and coefficients of every block, charged at the paper's K + 4 + 4P bits per block.
The decoder rebuilds w_hat (Algorithm 2) and decodes the true codes (R10-merged, 15 symbols) in raster order with
an adaptive count model (the 008 method: Dirichlet 1/2 counts per context) whose context is
(the interval of w_hat relative to the FP4 grid of the weight's own block scale, the block scale byte E).
Interval: w_hat / 2^(E-127) is placed among the 15 grid points -6..6 (16 intervals, the two outer ones unbounded);
each inner interval is split into S equal sub-bins. S in {1, 2, 4, 8, 16} is chosen on 8 training matrices
(the first sampled expert of each 010 layer, w1 and w2), then used on the test matrices.
Total = side information + code bits + the 008 scale stream; net gain = reference total - total.
On test matrices the codes are really encoded and decoded (racoder.c) and compared bit for bit.

usage: python s3b_lossless.py SET C P K MODE      (SET = train8 | T1 | T2)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s3_run as R  # noqa: E402
import slm as S  # noqa: E402
import racoder  # noqa: E402
from txcore import dirichlet  # noqa: E402

OUT = os.path.join(HERE, "../results/s3")
GRID = np.array([-6, -4, -3, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
SUBS = (1, 2, 4, 8, 16)
REF = os.path.join(R.ROOT, "experiments/010-transforms/results/tx/reference.json")


def names(which):
    return R.names(which)


def contexts(u, E_w, sub):
    i = np.searchsorted(GRID, u, side="right")                 # 0..15
    lo = GRID[np.clip(i - 1, 0, 14)]
    hi = GRID[np.clip(i, 0, 14)]
    inner = (i > 0) & (i < 15)
    frac = np.where(inner, (u - lo) / np.where(inner, hi - lo, 1), 0)
    sb = np.where(inner, np.minimum((frac * sub).astype(np.int64), sub - 1), 0)
    b = i * sub + sb
    key = b * 256 + E_w.astype(np.int64)
    _, ctx = np.unique(key, return_inverse=True)
    return ctx.reshape(u.shape)


def run_matrix(name, C, P, K, mode, which, chosen=None):
    W, sym, scale = R.load(name)
    z = np.load(os.path.join(R.DATA, f"{name}_C{C}P{P}K{K}_{mode}.npz"))
    n = W.size
    What = S.reconstruct(z["seeds"].astype(np.int64), z["q"], z["e"], K, C, P).reshape(-1)[:n].reshape(W.shape)
    u = What.astype(np.float64) / scale
    E_w = np.round(np.log2(scale)).astype(np.int64) + 127
    n_blocks = len(z["seeds"])
    side = n_blocks * (K + 4 + 4 * P)
    res = dict(matrix=name, C=C, P=P, K=K, mode=mode, n=int(n), n_blocks=int(n_blocks), side_bits=int(side),
               ideal_bits={})
    for sb in SUBS:
        res["ideal_bits"][str(sb)] = float(dirichlet(sym.ravel(), contexts(u, E_w, sb).ravel(), 15))
    if chosen is not None:
        ctx = contexts(u, E_w, chosen).ravel()
        t0 = time.time()
        words, nctx = racoder.encode(sym.ravel(), ctx)
        t_enc = time.time() - t0
        t0 = time.time()
        dec = racoder.decode(words, ctx, n, nctx)
        t_dec = time.time() - t0
        ok = bool(np.array_equal(dec, sym.ravel()))
        ref = json.load(open(REF))[which]["per_matrix"][name]
        code_bits = 32 * len(words)
        total = side + code_bits + ref["scale_bits"]
        res.update(sub=chosen, code_bits_actual=int(code_bits), roundtrip_ok=ok, scale_bits=ref["scale_bits"],
                   total_bits=float(total), ref_total_bits=ref["total"], net_gain_bits=float(ref["total"] - total),
                   bpw_total=float(total / n), net_gain_bpw=float((ref["total"] - total) / n),
                   encode_s=t_enc, decode_s=t_dec)
    return res


def main():
    which, C, P, K, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    os.makedirs(OUT, exist_ok=True)
    chosen = None
    if which != "train8":
        tr = [json.loads(l) for l in open(os.path.join(OUT, "s3b_train8.jsonl"))]
        tr = [r for r in tr if (r["C"], r["P"], r["K"], r["mode"]) == (C, P, K, mode)]
        assert len(tr) == 8, f"need the 8 training matrices first ({len(tr)})"
        chosen = min(SUBS, key=lambda s: sum(r["ideal_bits"][str(s)] for r in tr))
    path = os.path.join(OUT, f"s3b_{which}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done.add((r["matrix"], r["C"], r["P"], r["K"], r["mode"]))
    for name in names(which):
        if (name, C, P, K, mode) in done:
            continue
        r = run_matrix(name, C, P, K, mode, which, chosen)
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
