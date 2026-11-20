"""Spec 010 P1b diagnostics (moved to 011): is the probe's gain the chain's learning cost or structure the chain does not see?

    PYTHONPATH=. python experiments/011-limits-and-models/p1b_diag.py      -> results/p1b/diag.json

For every test matrix:
  - chain and offline-probe code length per tenth of the coding order (a learning-cost effect is largest at
    the start, where the chain has seen little; structure the chain misses gives a gain in every tenth);
  - the static entropy of the codes given the chain's own finest context (counts of the whole matrix, no learning
    cost; not a code length, a lower bound for any code that uses only the chain's contexts). A probe below this
    bound uses information outside the chain's contexts.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1b  # noqa: E402


def static_entropy(sym, levels, K):
    key = np.zeros(len(sym), dtype=np.int64)
    for ctx in levels:
        ctx = np.asarray(ctx, dtype=np.int64)
        key = key * (int(ctx.max()) + 1) + ctx
        _, key = np.unique(key, return_inverse=True)
    M = np.bincount(key * K + sym, minlength=(key.max() + 1) * K).reshape(-1, K).astype(np.float64)
    tot = M.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(M > 0, M * np.log2(tot / M), 0.0).sum()
    return float(h), int((tot > 0).sum())


def one(name):
    import torch
    torch.set_num_threads(4)
    chain = p1b.base_chains()[name]
    C, E, partner, _ = p1b.load(name)
    sym, p, K, side, scale, levels = p1b.chain_probs(C, E, p1b.active(chain), partner)
    h_static, n_ctx = static_entropy(sym, levels, K)
    st = p1b.matrix_stream(name, chain)
    model = p1b.make_net(15)
    model.load_state_dict(torch.load(p1b.RES / "offline_model.pt"))
    model.eval()
    n = len(sym)
    chain_pos = -np.log2(p[np.arange(n), sym])
    probe_pos = np.empty(n)
    with torch.no_grad():
        for i in range(0, n, 65536):
            idx = np.arange(i, min(n, i + 65536))
            probe_pos[idx] = p1b.nll_bits(model, p1b.to_t(p1b.batch_at(st, idx)), None, K).numpy()
    edges = np.linspace(0, n, 11).astype(int)
    tenth = [{"chain": float(chain_pos[a:b].mean()), "probe": float(probe_pos[a:b].mean())}
             for a, b in zip(edges[:-1], edges[1:])]
    return {"task": name, "n": n, "chain_bits": float(chain_pos.sum()), "probe_bits": float(probe_pos.sum()),
            "static_entropy_bits": h_static, "chain_contexts": n_ctx, "tenths": tenth}


def main():
    _, te = p1b.split()
    with ProcessPoolExecutor(12, mp_context=p1b.SPAWN) as ex:
        rows = list(ex.map(one, te))
    n = sum(r["n"] for r in rows)
    agg = {"chain_bits_per_weight": sum(r["chain_bits"] for r in rows) / n,
           "probe_bits_per_weight": sum(r["probe_bits"] for r in rows) / n,
           "static_entropy_bits_per_weight": sum(r["static_entropy_bits"] for r in rows) / n,
           "gain_per_tenth": [sum((r["tenths"][k]["chain"] - r["tenths"][k]["probe"]) * r["n"] / 10 for r in rows) / (n / 10)
                              for k in range(10)]}
    agg["chain_learning_cost_bits_per_weight"] = agg["chain_bits_per_weight"] - agg["static_entropy_bits_per_weight"]
    agg["probe_below_static_bits_per_weight"] = agg["static_entropy_bits_per_weight"] - agg["probe_bits_per_weight"]
    json.dump({"aggregate": agg, "rows": rows}, open(p1b.RES / "diag.json", "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
