"""Spec 011 A3: the "coarse block" rule. A block is coarse if none of its codes has magnitude 0.5 or 1.5, i.e. it
uses only {0, 1, 2, 3, 4, 6} = 2 x {0, 0.5, 1, 1.5, 2, 3} and could be stored exactly with the scale doubled.
Counts coarse blocks per matrix and compares with what independent codes would give (binomial with the matrix's
odd rate among non-zero codes). Writes results/a/a3_odd.json.

    PYTHONPATH=. python experiments/011-limits-and-models/a3_odd.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bfeat as BF  # noqa: E402
import common as K  # noqa: E402


def one(name):
    M = BF.Matrices()
    S, E = M.codes(name)
    R, W = S.shape
    mag, _ = BF.mag_sign(S)
    lv = mag.reshape(R, W // 32, 32)
    odd = np.isin(lv, [1, 3]).sum(-1)
    nz = (lv > 0).sum(-1)
    q = odd.sum() / max(1, nz.sum())
    coarse = odd == 0
    eb = E.astype(np.int64)
    return {"task": name, "layer": M.reg[name]["layer"], "role": M.reg[name]["role"], "blocks": int(odd.size),
            "coarse_share": float(coarse.mean()), "expected_if_independent": float(np.mean((1 - q) ** nz)),
            "odd_rate_nonzero": float(q), "odd_var": float(odd.var()), "binomial_var": float(np.mean(nz * q * (1 - q))),
            "coarse_scale_minus_fine_scale_median": float(np.median(eb[coarse]) - np.median(eb[~coarse])) if coarse.any() and (~coarse).any() else None,
            "coarse_max_level_hist": np.bincount(lv[coarse].max(-1), minlength=8).tolist() if coarse.any() else None}


def main():
    sets = {"T1": K.T1, "T2": K.T2, "pool_round1": K.pool_round(1)}
    out = {}
    with ProcessPoolExecutor(48) as ex:
        for k, ns in sets.items():
            rows = list(ex.map(one, ns))
            nb = sum(r["blocks"] for r in rows)
            out[k] = {"matrices": len(rows), "coarse_share": sum(r["coarse_share"] * r["blocks"] for r in rows) / nb,
                      "expected_if_independent": sum(r["expected_if_independent"] * r["blocks"] for r in rows) / nb,
                      "by_role": {ro: float(np.mean([r["coarse_share"] for r in rows if r["role"] == ro])) for ro in ("w1", "w2", "w3")},
                      "min_matrix": min(r["coarse_share"] for r in rows), "max_matrix": max(r["coarse_share"] for r in rows),
                      "rows": rows}
            print(k, {a: b for a, b in out[k].items() if a != "rows"}, flush=True)
    by_layer = {}
    for r in out["pool_round1"]["rows"]:
        by_layer.setdefault(r["layer"], []).append(r["coarse_share"])
    out["pool_by_layer"] = {L: float(np.mean(v)) for L, v in sorted(by_layer.items())}
    json.dump(out, open(HERE / "results" / "a" / "a3_odd.json", "w"), indent=1)
    print("by layer", {L: round(v, 3) for L, v in out["pool_by_layer"].items()})


if __name__ == "__main__":
    main()
