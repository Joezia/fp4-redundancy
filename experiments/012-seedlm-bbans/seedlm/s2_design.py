"""S2 (part 1): the paper's own design-space experiment (Section 3.4, Eq. 4, Table 1).

For a bit budget M in {3, 4}, every positive integer (C, P, K) with M*C = K + 4 + 4P, K in Table 6
(2..24) and P < C, is evaluated on standard normal vectors w in R^C: per-element expected squared
error E||w - U(s*) t*||^2 / C after the full seed search. The paper reports only the chosen
configuration (Table 1: M=4 -> C=8, P=3, K=16; M=3 -> C=12, P=4, K=16), so the check is whether the
same configuration comes out best. Grid bound C <= 16 and the per-element normalisation are our
choices (see report). Both exponent rules are run.

usage: python s2_design.py [n_vectors]
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slm as S  # noqa: E402
import seedlm_gpu as G  # noqa: E402

OUT = os.path.join(HERE, "../results/s1")


def grid(M, cmax=16):
    out = []
    for C in range(2, cmax + 1):
        for P in range(1, C):
            K = M * C - 4 - 4 * P
            if K in S.TAPS:
                out.append((C, P, K))
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    path = os.path.join(OUT, "s2_design.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done.add((r["M"], r["C"], r["P"], r["K"], r["mode"], r["n"]))
    for M in (4, 3):
        for (C, P, K) in grid(M):
            for mode in ("range", "paper"):
                if (M, C, P, K, mode, n) in done:
                    continue
                rng = np.random.default_rng([12, C])        # same vectors for every (P, K, mode) at this C
                W = rng.standard_normal((n, C)).astype(np.float32)
                t0 = time.time()
                s, q, e, err = G.search_lazy(W, K, C, P, mode)
                torch.cuda.synchronize()
                r = dict(M=M, C=C, P=P, K=K, mode=mode, n=n, mse_per_elem=float(err.mean() / C),
                         rel_err=float(np.sqrt(err.sum() / (W.astype(np.float64) ** 2).sum())),
                         floor_frac=float(np.mean(e == -8)), sec=time.time() - t0)
                print(json.dumps(r), flush=True)
                with open(path, "a") as f:
                    f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
