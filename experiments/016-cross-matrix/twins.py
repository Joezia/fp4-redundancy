"""Spec 016, follow-up of B2: strict code-length value of the most similar neuron vector found in the whole model.

    PYTHONPATH=.:experiments/016-cross-matrix python experiments/016-cross-matrix/twins.py {deepseek|kimi} [--workers N]

For every real neuron query of the B2 scan and each partner category (best overall, same expert, same layer other
expert, other layer; partner = highest |cos w| found in that category over the whole model):
  * the partner's codes are sign-aligned (sign bit flipped when cos < 0);
  * conditional code length of the query codes given its own block scale and the partner code at the same hidden
    position, P(c | E, h), against P(c | E); both tables are fitted on the other half of the queries (2-fold
    cross-fitting, Dirichlet-1/2 smoothing), so no table cost and no in-sample fit;
  * net = gain - reference cost, reference cost = log2(number of vectors of this type in the model) + 1 sign bit
    + 1 flag bit.
Per query and pooled bits per weight are reported. Upper-bound caveats (partner may come later in decoding order,
the search itself used |cos w|) are written in the report.
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

import common as K  # noqa: E402

CATS = ["best", "same_expert", "same_layer", "other_layer"]


def vec(model, i, j):
    e = K.entries(model)[i]
    C, E = K.load(model, e)
    Cn, En = K.neuron_major(C, E, e["role"])
    return Cn[j].copy(), En[j].copy()


def pair_task(model, qi, qj, hits, given=None):
    cq, eq = vec(model, qi, qj) if given is None else given
    wq = K.E2M1[cq] * np.exp2(eq.astype(np.float64) - 127)
    out = {"cq": cq, "eq": eq}
    for cat, (hi, hj) in hits.items():
        if hi < 0:
            continue
        ch, eh = vec(model, hi, hj)
        wh = K.E2M1[ch] * np.exp2(eh.astype(np.float64) - 127)
        cs = float(wq @ wh / (np.linalg.norm(wq) * np.linalg.norm(wh) + 1e-300))
        if cs < 0:
            ch = ch ^ 8
        out[cat] = {"ch": ch, "cos": cs, "eq_frac": float((cq == ch).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(K.MODELS))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--null", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    src = K.RES / f"scan_{a.model}.npz"
    if not src.exists():
        src = K.DATA / f"scan_{a.model}_ckpt.npz"
    z = dict(np.load(src, allow_pickle=True))
    n = int(z["vec_neuron_n_real"])
    meta = z["vec_neuron_meta"][:n]
    off = n if a.null else 0           # null control: the shuffled queries (codes from the query file) and their hits
    ti = z["vec_neuron_top_i_w"][0, off:off + n]
    ci = z["vec_neuron_cat_i_w"][:, off:off + n]
    given = [None] * n
    if a.null:
        qf = np.load(K.DATA / f"queries_{a.model}.npz")
        given = list(zip(qf["neuron_Cs"][:n], qf["neuron_Es"][:n]))
    hits = []
    for q in range(n):
        h = {"best": divmod(int(ti[q]), 100000)}
        for k, c in ((1, "same_expert"), (2, "same_layer"), (3, "other_layer")):
            h[c] = divmod(int(ci[k, q]), 100000) if ci[k, q] >= 0 else (-1, -1)
        hits.append(h)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(pair_task, [a.model] * n, meta[:, 0].tolist(), meta[:, 1].tolist(), hits, given, chunksize=4))
    ents = K.entries(a.model)
    n_vec = sum(e["codeword_shape"][0 if e["role"] != "w2" else 1] for e in ents)
    ref = math.log2(n_vec) + 2
    L = len(rows[0]["cq"])
    fold = np.arange(n) % 2
    res = {"model": a.model, "source": src.name, "n_queries": n, "vector_length": L, "n_vectors_in_model": n_vec,
           "reference_cost_bits": ref, "cats": {}}
    for cat in CATS:
        idx = [q for q in range(n) if cat in rows[q]]
        gains = np.zeros(n) * np.nan
        for f in (0, 1):
            tr = [q for q in idx if fold[q] != f]
            te = [q for q in idx if fold[q] == f]
            t3 = np.full((256, 16, 16), 0.5)       # (scale, partner code, code)
            t2 = np.full((256, 16), 0.5)           # (scale, code)
            for q in tr:
                np.add.at(t3, (rows[q]["eq"], rows[q][cat]["ch"], rows[q]["cq"]), 1)
                np.add.at(t2, (rows[q]["eq"], rows[q]["cq"]), 1)
            p3 = t3 / t3.sum(axis=2, keepdims=True)
            p2 = t2 / t2.sum(axis=1, keepdims=True)
            for q in te:
                r = rows[q]
                l3 = -np.log2(p3[r["eq"], r[cat]["ch"], r["cq"]]).sum()
                l2 = -np.log2(p2[r["eq"], r["cq"]]).sum()
                gains[q] = l2 - l3
        g = gains[idx]
        cosv = np.array([abs(rows[q][cat]["cos"]) for q in idx])
        eqf = np.array([rows[q][cat]["eq_frac"] for q in idx])
        net = g - ref
        res["cats"][cat] = {
            "n": len(idx),
            "gain_bits_per_vector": {"median": float(np.median(g)), "mean": float(g.mean()), "p90": float(np.quantile(g, .9)), "max": float(g.max())},
            "net_bits_per_vector": {"median": float(np.median(net)), "mean": float(net.mean())},
            "frac_net_positive": float((net > 0).mean()),
            "pooled_gain_bits_per_weight": float(g.sum() / (len(idx) * L)),
            "pooled_net_positive_only_bits_per_weight": float(np.maximum(net, 0).sum() / (len(idx) * L)),
            "abs_cos": {"median": float(np.median(cosv)), "p90": float(np.quantile(cosv, .9)), "max": float(cosv.max())},
            "code_eq_frac_median": float(np.median(eqf)),
            "gain_vs_cos_bins": {f"{lo:.1f}-{hi:.1f}": [int(((cosv >= lo) & (cosv < hi)).sum()),
                                                       float(np.mean(g[(cosv >= lo) & (cosv < hi)])) if ((cosv >= lo) & (cosv < hi)).any() else None]
                                 for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .5), (.5, 1.01))},
        }
        print(cat, json.dumps(res["cats"][cat]), flush=True)
    res["seconds"] = time.time() - t0
    res["null_control"] = a.null
    json.dump(res, open(K.RES / f"twins_{a.model}{'_null' if a.null else ''}.json", "w"), indent=1)
    print("TWINS_DONE", a.model, f"{res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
