"""Spec 014, supplement (user question): do the FP4 expert weights carry the trace of a SmoothQuant-like
activation-to-weight scale migration?

A migration W' = W diag(s) scales the input channels of a linear layer and folds 1/s into whatever produces its input:
  * w1, w3 read the same normalised hidden state x, so one factor per hidden dimension j multiplies column j of both
    (folded into the RMSNorm weight). It is shared by all experts of the layer if folded into that norm.
    Trace: the column-RMS profiles of w1 and w3 (hidden dimension) are strongly correlated with each other and
    with those of other experts of the same layer.
  * w2 reads h = silu(w1 x) * (w3 x); a factor s_i on column i of w2 can only be compensated by row i of w3 (1/s_i).
    Trace: the w2 column RMS and the w3 row RMS of the same neuron are anti-correlated.
For each expert of the full 008 / 009 samples this script computes, on log RMS (dequantized values):
  corr(w1 col, w3 col) over hidden dimensions; corr(w3 row, w2 col) and corr(w1 row, w2 col) over neurons;
and for every pair of experts in the same layer corr(w1 col of a, w1 col of b). A shuffled baseline (all elements
of each matrix permuted, an independent permutation per matrix, seeds 20261123 + 1 / 2 / 3) is computed for the same quantities.

    PYTHONPATH=. python experiments/014-heatmaps/smoothcheck.py [--workers N]
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

from redundancy.adapters.fp4 import E2M1, FP4  # noqa: E402

import fullsample as F  # noqa: E402

RES = F.RES
SEED = 20261123


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def logrms(x, axis):
    return np.log(np.sqrt((x * x).mean(axis=axis)) + 1e-30)


def expert_profiles(model, task):
    m = F.MODELS[model]
    ad = FP4({"samples": str(F.ROOT / m["samples"]), "snapshot_env": m["snapshot_env"]})
    prof = {}
    for name in task["entries"]:
        role = name.rsplit(".", 1)[1]
        C, E = ad._read(ad.samples[name])
        w = E2M1[C.astype(np.int64)] * np.exp2(np.repeat(E.astype(np.float64), 32, axis=1) - 127)
        # an independent permutation per matrix (seed + role), so the baseline shares nothing between w1, w3, w2
        ws = w.ravel()[np.random.default_rng(SEED + {'w1': 1, 'w3': 3, 'w2': 2}[role]).permutation(w.size)].reshape(w.shape)
        for tag, x in (("real", w), ("shuffled", ws)):
            # rows = output channels, columns = input channels (PyTorch layout)
            prof[(role, tag, "row")] = logrms(x, 1)
            prof[(role, tag, "col")] = logrms(x, 0)
    out = {"task": task["id"], "layer": task["layer"], "expert": task["expert"]}
    for tag in ("real", "shuffled"):
        out[tag] = {"w1col_w3col": corr(prof[("w1", tag, "col")], prof[("w3", tag, "col")]),
                    "w3row_w2col": corr(prof[("w3", tag, "row")], prof[("w2", tag, "col")]),
                    "w1row_w2col": corr(prof[("w1", tag, "row")], prof[("w2", tag, "col")]),
                    "w1row_w3row": corr(prof[("w1", tag, "row")], prof[("w3", tag, "row")])}
    out["w1col"] = {t: prof[("w1", t, "col")].tolist() for t in ("real", "shuffled")}
    return out


def q(v):
    v = np.asarray(v)
    return {"median": float(np.median(v)), "p10": float(np.quantile(v, .1)), "p90": float(np.quantile(v, .9)),
            "min": float(v.min()), "max": float(v.max()), "frac_negative": float(np.mean(v < 0)), "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    summary = {"seed": SEED, "models": {}}
    for model in F.MODELS:
        tasks = F.tasks_of(model)
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(expert_profiles, [model] * len(tasks), tasks))
        res = {}
        for tag in ("real", "shuffled"):
            res[tag] = {k: q([r[tag][k] for r in rows]) for k in rows[0][tag]}
            pairs = []
            by_layer = {}
            for r in rows:
                by_layer.setdefault(r["layer"], []).append(r)
            per_layer = {}
            for L, rs in by_layer.items():
                for x, y in combinations(rs, 2):
                    c = corr(x["w1col"][tag], y["w1col"][tag])
                    pairs.append(c)
                    per_layer.setdefault(str(L), []).append(round(c, 4))
            res[tag]["w1col_cross_expert_same_layer"] = q(pairs)
            res[tag]["w1col_cross_expert_by_layer"] = per_layer
        res["per_expert"] = [{k: r[k] for k in ("task", "layer", "expert", "real", "shuffled")} for r in rows]
        summary["models"][model] = res
        print(f"== {model}: {len(rows)} experts, {time.time() - t0:.0f} s", flush=True)
        for k in [k for k in res["real"] if not k.endswith("by_layer")]:
            print(f"  {k}: real median {res['real'][k]['median']:+.3f} (p10 {res['real'][k]['p10']:+.3f}, "
                  f"p90 {res['real'][k]['p90']:+.3f}, negative {res['real'][k]['frac_negative']:.2f}); "
                  f"shuffled median {res['shuffled'][k]['median']:+.3f} (p10 {res['shuffled'][k]['p10']:+.3f}, "
                  f"p90 {res['shuffled'][k]['p90']:+.3f})", flush=True)
    summary["seconds"] = time.time() - t0
    json.dump(summary, open(RES / "smoothcheck.json", "w"), indent=1)


if __name__ == "__main__":
    main()
