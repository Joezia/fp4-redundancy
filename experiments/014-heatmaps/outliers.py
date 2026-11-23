"""Spec 014, supplementary counts read from results/stats.json (no new data access): rows / columns whose RMS of the
dequantized values is below 0.5x or above 1.5x the median of that matrix, the same counts for the shuffled matrix,
and, per expert, how many low-RMS neurons coincide between w1 (rows), w3 (rows) and w2 (columns).
Adds the field "outliers" to every matrix and "neuron_overlap" per expert in results/stats.json."""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"
S = json.load(open(RES / "stats.json"))


def counts(a):
    a = np.asarray(a)
    m = np.median(a)
    return {"median": float(m), "n_below_0.5x": int((a < 0.5 * m).sum()), "n_above_1.5x": int((a > 1.5 * m).sum()),
            "idx_below_0.5x": np.nonzero(a < 0.5 * m)[0].tolist(), "idx_above_1.5x": np.nonzero(a > 1.5 * m)[0].tolist()}


by_expert = {}
for r in S["matrices"]:
    s = r["stats"]["deq"]
    r["outliers"] = {k: counts(s[k]) for k in ("row_rms", "col_rms", "row_rms_shuffled", "col_rms_shuffled")}
    by_expert.setdefault(r["expert_task"], {})[r["role"]] = r
S["neuron_overlap"] = {}
for task, d in by_expert.items():
    low = {"w1": set(d["w1"]["outliers"]["row_rms"]["idx_below_0.5x"]),
           "w3": set(d["w3"]["outliers"]["row_rms"]["idx_below_0.5x"]),
           "w2": set(d["w2"]["outliers"]["col_rms"]["idx_below_0.5x"])}
    S["neuron_overlap"][task] = {"low_rms_neurons": {k: len(v) for k, v in low.items()},
                                 "w1_and_w3": len(low["w1"] & low["w3"]), "w1_and_w2": len(low["w1"] & low["w2"]),
                                 "all_three": len(low["w1"] & low["w3"] & low["w2"])}
json.dump(S, open(RES / "stats.json", "w"))
for r in S["matrices"]:
    o = r["outliers"]
    print(r["matrix"], " ".join(f"{k}: <0.5x {o[k]['n_below_0.5x']} >1.5x {o[k]['n_above_1.5x']}" for k in o))
print(json.dumps(S["neuron_overlap"], indent=0))
