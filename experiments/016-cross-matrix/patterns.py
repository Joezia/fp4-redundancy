"""Spec 016, follow-up of B2: what the cross-matrix neuron similarity looks like (pattern description, no code lengths).

    PYTHONPATH=.:experiments/016-cross-matrix python experiments/016-cross-matrix/patterns.py {deepseek|kimi} [--workers N]

For every real neuron query of the B2 scan and its most similar neuron (highest |cos w| over the whole model) in each
category -- same expert (other matrix), same layer other expert, other layer -- this records: signed cosine (w and v),
code-equality rate against its chance level, sign agreement on positions where both are non-zero, correlation of
|v|, and where the partner is (layer distance, role pair, same neuron index, same expert index). "Above control"
means above the largest value reached by the shuffled queries in the same category (conservative threshold).
"""
import argparse
import collections
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

import common as K  # noqa: E402

CATS = {1: "same_expert", 2: "same_layer", 3: "other_layer"}


def vec(model, i, j):
    e = K.entries(model)[i]
    C, E = K.load(model, e)
    Cn, En = K.neuron_major(C, E, e["role"])
    return Cn[j].copy(), En[j].copy()


def describe(cq, eq, ch, eh):
    vq, vh = K.E2M1[cq].astype(np.float64), K.E2M1[ch].astype(np.float64)
    wq, wh = vq * np.exp2(eq - 127.0), vh * np.exp2(eh - 127.0)
    cw = float(wq @ wh / (np.linalg.norm(wq) * np.linalg.norm(wh) + 1e-300))
    cv = float(vq @ vh / (np.linalg.norm(vq) * np.linalg.norm(vh) + 1e-300))
    ch_al = ch ^ 8 if cw < 0 else ch                     # sign-aligned partner
    hq, hh = np.bincount(cq, minlength=16) / len(cq), np.bincount(ch_al, minlength=16) / len(ch)
    nz = (np.abs(vq) > 0) & (np.abs(vh) > 0)
    sq, sh = np.sign(vq[nz]), np.sign(vh[nz]) * (1 if cw >= 0 else -1)
    aq, ah = np.abs(vq), np.abs(vh)
    return {"cos_w": cw, "cos_v": cv, "eq_rate": float((cq == ch_al).mean()), "eq_chance": float(hq @ hh),
            "sign_agree": float((sq == sh).mean()) if nz.any() else None,
            "absv_corr": float(np.corrcoef(aq, ah)[0, 1])}


def task(model, qi, qj, partners):
    cq, eq = vec(model, qi, qj)
    out = {}
    for c, (hi, hj) in partners.items():
        if hi >= 0:
            ch, eh = vec(model, hi, hj)
            out[c] = describe(cq, eq.astype(np.float64), ch, eh.astype(np.float64))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(K.MODELS))
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    t0 = time.time()
    src = K.RES / f"scan_{a.model}.npz"
    z = dict(np.load(src, allow_pickle=True))
    ents = K.entries(a.model)
    n = int(z["vec_neuron_n_real"])
    meta = z["vec_neuron_meta"][:n]
    cv, ci = z["vec_neuron_cat_v_w"], z["vec_neuron_cat_i_w"]
    thr = {c: float(cv[k, n:].max()) for k, c in CATS.items()}
    partners = [{c: divmod(int(ci[k, q]), 100000) if ci[k, q] >= 0 else (-1, -1) for k, c in CATS.items()} for q in range(n)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(task, [a.model] * n, meta[:, 0].tolist(), meta[:, 1].tolist(), partners, chunksize=4))
    L = lambda i: ents[i]["layer"]
    res = {"model": a.model, "n_queries": n, "control_threshold_abs_cos_w": thr, "cats": {}}
    for k, c in CATS.items():
        above = [q for q in range(n) if cv[k, q] > thr[c] and c in rows[q]]
        d = {"n_above_control": len(above), "frac_above_control": len(above) / n}
        if above:
            R = [rows[q][c] for q in above]
            d["abs_cos_w"] = np.quantile([abs(r["cos_w"]) for r in R], [.5, .9, 1]).tolist()
            d["frac_negative_cos"] = float(np.mean([r["cos_w"] < 0 for r in R]))
            d["eq_rate_over_chance"] = np.quantile([r["eq_rate"] / r["eq_chance"] for r in R], [.5, .9, 1]).tolist()
            d["sign_agree"] = np.quantile([r["sign_agree"] for r in R if r["sign_agree"] is not None], [.5, .9, 1]).tolist()
            d["absv_corr"] = np.quantile([r["absv_corr"] for r in R], [.5, .9, 1]).tolist()
            hit = [partners[q][c] for q in above]
            src_ = [tuple(meta[q]) for q in above]
            d["layer_distance"] = dict(sorted(collections.Counter(abs(L(h[0]) - L(s[0])) for h, s in zip(hit, src_)).items()))
            d["role_pairs"] = dict(collections.Counter("-".join(sorted((ents[s[0]]["role"], ents[h[0]]["role"]))) for h, s in zip(hit, src_)))
            d["same_neuron_index"] = float(np.mean([h[1] == s[1] for h, s in zip(hit, src_)]))
            d["same_expert_index"] = float(np.mean([ents[h[0]]["expert"] == ents[s[0]]["expert"] for h, s in zip(hit, src_)]))
            d["by_query_layer"] = {}
            for Lq in sorted({L(s[0]) for s in src_}):
                m = [L(s[0]) == Lq for s in src_]
                d["by_query_layer"][str(Lq)] = int(sum(m))
        # all queries (including below control): median similarity and equality ratio, real vs control threshold
        Rall = [rows[q][c] for q in range(n) if c in rows[q]]
        d["all_queries"] = {"abs_cos_w_median": float(np.median([abs(r["cos_w"]) for r in Rall])),
                            "eq_rate_over_chance_median": float(np.median([r["eq_rate"] / r["eq_chance"] for r in Rall]))}
        res["cats"][c] = d
        print(c, json.dumps(d)[:1500], flush=True)
    # per query layer: fraction above control in each category
    lay = np.array([L(int(i)) for i in meta[:, 0]])
    res["frac_above_control_by_layer"] = {str(Lq): {c: float((cv[k, :n][lay == Lq] > thr[c]).mean()) for k, c in CATS.items()} | {"n": int((lay == Lq).sum())}
                                          for Lq in sorted(set(lay.tolist()))}
    role = np.array([ents[int(i)]["role"] for i in meta[:, 0]])
    res["frac_above_control_by_role"] = {r: {c: float((cv[k, :n][role == r] > thr[c]).mean()) for k, c in CATS.items()} for r in ("w1", "w3", "w2")}
    tv = z["vec_neuron_top_v_w"]
    res["top1_to_top4"] = {"real_quantiles_50_90_99": [np.quantile(tv[k, :n], [.5, .9, .99]).tolist() for k in range(4)],
                           "control_quantiles_50_90_99": [np.quantile(tv[k, n:], [.5, .9, .99]).tolist() for k in range(4)]}
    res["seconds"] = time.time() - t0
    json.dump(res, open(K.RES / f"patterns_{a.model}.json", "w"), indent=1)
    print(json.dumps(res["frac_above_control_by_role"]), json.dumps(res["frac_above_control_by_layer"])[:1500])
    print("PATTERNS_DONE", a.model, f"{res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
