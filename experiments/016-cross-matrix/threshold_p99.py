"""Spec 016, B2: fraction of real neuron queries whose best partner (per category) exceeds the 99th percentile of the
shuffled-query control (less sensitive to single outlier control queries than the control maximum).

    PYTHONPATH=.:experiments/016-cross-matrix python experiments/016-cross-matrix/threshold_p99.py
"""
import json

import numpy as np

import common as K

out = {}
for m in ("deepseek", "kimi"):
    z = dict(np.load(K.RES / f"scan_{m}.npz", allow_pickle=True))
    ents = K.entries(m); n = int(z["vec_neuron_n_real"]); meta = z["vec_neuron_meta"][:n]
    cv = z["vec_neuron_cat_v_w"]
    lay = np.array([ents[int(i)]["layer"] for i in meta[:, 0]])
    nl = len(set(e["layer"] for e in ents))
    res = {}
    for k, c in ((1, "same_expert"), (2, "same_layer"), (3, "other_layer")):
        null = cv[k, n:]; real = cv[k, :n]
        p99 = np.quantile(null, .99); p90 = np.quantile(null, .9)
        res[c] = {"null_p90": float(p90), "null_p99": float(p99), "null_max": float(null.max()),
                  "frac_real_above_null_p99": float((real > p99).mean()), "frac_real_above_null_p90": float((real > p90).mean()),
                  "real_median": float(np.median(real)), "null_median": float(np.median(null))}
        thirds = {}
        for t, (lo, hi) in enumerate(((0, nl / 3), (nl / 3, 2 * nl / 3), (2 * nl / 3, 1e9))):
            mk = (lay - min(lay) >= lo) & (lay - min(lay) < hi)
            thirds[["浅", "中", "深"][t]] = float((real[mk] > p99).mean())
        res[c]["frac_above_p99_by_depth_third"] = thirds
    out[m] = res
json.dump(out, open(K.RES / "b2_threshold_p99.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False)[:600])
