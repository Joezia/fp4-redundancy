"""S3: SeedLM on FP4 matrices.

S3-a (lossy, as in the paper): the input is the dequantised FP4 matrix (code value x 2^(E-127)), flattened in
storage (row-major) order and cut into blocks of C elements (S-7). For each block the full seed search of
Algorithm 3 is run (GPU), then the matrix is rebuilt with Algorithm 2.
Reported per matrix: bits per element (Eq. 3), relative MSE ||W - W_hat||^2 / ||W||^2, and the fraction of
weights whose reconstruction, rounded to the nearest FP4 grid point of its own block scale, differs from the
original (R10-merged) code; also the fraction of blocks whose shared exponent sits at the floor e = -8.
The seeds, mantissas and exponents are saved (data/012/s3/) for S3-b.

usage: python s3_run.py SET C P K MODE        (SET = T1 | T2 | train8 | train)
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments/010-transforms"))
sys.path.insert(0, os.path.join(ROOT, "experiments/010-transforms/tx"))
sys.path.insert(0, os.path.join(ROOT, "experiments/011-limits-and-models"))
import slm as S  # noqa: E402
import seedlm_gpu as G  # noqa: E402

OUT = os.path.join(HERE, "../results/s3")
DATA = os.path.join(ROOT, "data/012/s3")
# E2M1 magnitudes; merged symbol -> signed value (txcore: 0..7 = +, 8..14 = -)
VAL = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def names(which):
    import common as K
    import tx
    if which == "T1":
        return K.T1
    if which == "T2":
        return K.T2
    ex = tx.sample_experts()
    if which == "train8":          # first sampled expert of each 010 layer, w1 and w2 (S3-b binning selection)
        return [f"L{L}.E{es[0]}.{r}" for L, es in ex.items() for r in ("w1", "w2")]
    return K.names({L: es[:2] for L, es in ex.items()})


def load(name):
    import tx
    from txcore import merge
    C, E = tx.read(name)
    sym = merge(C)
    scale = np.exp2(np.repeat(E.astype(np.float64), 32, axis=1) - 127)
    W = (VAL[sym] * scale).astype(np.float32)          # equals tx.deq(C, E) (-0 -> +0)
    return W, sym, scale


def nearest_code(x, scale):
    """nearest FP4 grid point of each weight's own block scale -> merged symbol (ties -> smaller |value|)."""
    u = np.abs(x / scale)
    mags = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])
    mid = (mags[1:] + mags[:-1]) / 2
    m = np.searchsorted(mid, u, side="left")            # u == midpoint -> lower magnitude
    m = np.minimum(m, 7)
    neg = (x < 0) & (m > 0)
    return np.where(neg, m + 7, m)


def run_matrix(name, C, P, K, mode, gcb):
    W, sym, scale = load(name)
    flat = W.reshape(-1)
    pad = (-len(flat)) % C
    blocks = np.concatenate([flat, np.zeros(pad, np.float32)]).reshape(-1, C)
    t0 = time.time()
    if K >= 20:
        seeds, q, e, err = G.search_lazy(blocks, K, C, P, mode)
    else:
        seeds, q, e, err = G.search(blocks, gcb, mode)
    torch.cuda.synchronize()
    t_search = time.time() - t0
    R = S.reconstruct(seeds, q, e, K, C, P).reshape(-1)[:len(flat)].reshape(W.shape).astype(np.float64)
    W64 = W.astype(np.float64)
    rel_mse = float(((W64 - R) ** 2).sum() / (W64 ** 2).sum())
    code_hat = nearest_code(R, scale)
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(os.path.join(DATA, f"{name}_C{C}P{P}K{K}_{mode}.npz"), seeds=seeds.astype(np.uint32),
                        q=q, e=e)
    return dict(matrix=name, C=C, P=P, K=K, mode=mode, n=int(W.size), n_blocks=int(len(blocks)), pad=int(pad),
                bits_per_element=S.bits_per_element(C, P, K), rel_mse=rel_mse,
                code_mismatch=float(np.mean(code_hat != sym)), floor_frac=float(np.mean(e == -8)),
                ceil_frac=float(np.mean(e == 7)), search_s=t_search)


def main():
    which, C, P, K, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"s3a_{which}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done.add((r["matrix"], r["C"], r["P"], r["K"], r["mode"]))
    gcb = G.GPUCodebook(S.Codebook(K, C, P, device="cuda")) if K < 20 else None
    for name in names(which):
        if (name, C, P, K, mode) in done:
            continue
        r = run_matrix(name, C, P, K, mode, gcb)
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
