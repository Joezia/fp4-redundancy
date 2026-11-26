"""Spec 016, follow-up of B2: are the cross-expert neuron similarities one-to-one "twins" or a few directions shared by
many neurons of nearby layers?

    PYTHONPATH=.:experiments/016-cross-matrix python experiments/016-cross-matrix/shared_dirs.py {deepseek|kimi} [--layers ...]

For each chosen layer L: the neuron vectors (w1 / w3 rows, w2 columns; dequantized w, each normalised to unit length) of
N_EXP random experts of layers L-1, L, L+1 give an uncentred second-moment matrix (hidden x hidden); its top
eigenvectors are the directions shared by many neurons there. For every B2 query of layer L whose same-layer or
other-layer partner is above the control threshold:
  * energy of the query in the top-k shared directions (k = 1, 4, 16, 64), against the shuffled copy of the query;
  * |cos| with its partner before and after removing the top-k shared directions from both vectors.
The experts of the query and of its partner are excluded from the direction estimate.
"""
import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "32")
import numpy as np  # noqa: E402

import common as K  # noqa: E402

N_EXP = 24
KS = [1, 4, 16, 64]


def neuron_vectors(model, i):
    e = K.entries(model)[i]
    C, E = K.load(model, e)
    Cn, En = K.neuron_major(C, E, e["role"])
    W = K.E2M1[Cn].astype(np.float32) * np.exp2(En.astype(np.float32) - 127)
    return W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(K.MODELS))
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(20261126)
    ents = K.entries(a.model)
    z = dict(np.load(K.RES / f"scan_{a.model}.npz", allow_pickle=True))
    q = np.load(K.DATA / f"queries_{a.model}.npz")
    n = int(z["vec_neuron_n_real"]); meta = z["vec_neuron_meta"][:n]
    cv, ci = z["vec_neuron_cat_v_w"], z["vec_neuron_cat_i_w"]
    thr = {k: float(cv[k, n:].max()) for k in (2, 3)}
    layers_all = sorted({e["layer"] for e in ents})
    by_le = {}
    for i, e in enumerate(ents):
        by_le.setdefault((e["layer"], e["expert"]), []).append(i)
    lay_q = np.array([ents[int(i)]["layer"] for i in meta[:, 0]])
    layers = a.layers or [layers_all[0], layers_all[len(layers_all) // 4], layers_all[len(layers_all) // 2],
                          layers_all[3 * len(layers_all) // 4], layers_all[-4]]
    res = {"model": a.model, "n_experts_per_layer_for_directions": N_EXP, "layers": {}}
    for L in layers:
        qs = [k for k in np.nonzero(lay_q == L)[0] if cv[2, k] > thr[2] or cv[3, k] > thr[3]]
        excl = set()
        for k in qs:
            excl.add((L, ents[int(meta[k, 0])]["expert"]))
            for c in (2, 3):
                hi = int(ci[c, k]) // 100000
                excl.add((ents[hi]["layer"], ents[hi]["expert"]))
        M = None
        nvec = 0
        for Lx in (L - 1, L, L + 1):
            if Lx not in layers_all:
                continue
            exs = [ex for (l2, ex) in by_le if l2 == Lx and (l2, ex) not in excl]
            for ex in rng.choice(exs, min(N_EXP, len(exs)), replace=False):
                for i in by_le[(Lx, int(ex))]:
                    V = neuron_vectors(a.model, i)
                    M = V.T @ V if M is None else M + V.T @ V
                    nvec += len(V)
        evals, evecs = np.linalg.eigh(M.astype(np.float64))
        U = evecs[:, ::-1][:, :max(KS)].astype(np.float32)
        top_share = (evals[::-1][:max(KS)] / evals.sum()).tolist()
        rows = []
        for k in qs:
            i, j = int(meta[k, 0]), int(meta[k, 1])
            x = neuron_vectors(a.model, i)[j]
            xs = (K.E2M1[q["neuron_Cs"][k]] * np.exp2(q["neuron_Es"][k].astype(np.float32) - 127)).astype(np.float32)
            xs /= np.linalg.norm(xs)
            r = {"query": ents[i]["matrix"] + f"[{j}]", "energy_topk": [float(((U[:, :kk].T @ x) ** 2).sum()) for kk in KS],
                 "energy_topk_shuffled": [float(((U[:, :kk].T @ xs) ** 2).sum()) for kk in KS], "partners": {}}
            for c, nm in ((2, "same_layer"), (3, "other_layer")):
                if cv[c, k] <= thr[c]:
                    continue
                hi, hj = divmod(int(ci[c, k]), 100000)
                y = neuron_vectors(a.model, hi)[hj]
                cos = [abs(float(x @ y))]
                for kk in KS:
                    P = U[:, :kk]
                    xr, yr = x - P @ (P.T @ x), y - P @ (P.T @ y)
                    cos.append(abs(float(xr @ yr / (np.linalg.norm(xr) * np.linalg.norm(yr)))))
                r["partners"][nm] = {"partner": ents[hi]["matrix"] + f"[{hj}]", "abs_cos_after_removing_topk": cos}
            rows.append(r)
        summ = {"n_queries": len(rows), "n_vectors_for_directions": nvec, "eigen_share_top": top_share[:8],
                "eigen_share_top_k_cumulative": {kk: float(sum(top_share[:kk])) for kk in KS},
                "query_energy_topk_median": {kk: float(np.median([r["energy_topk"][t] for r in rows])) for t, kk in enumerate(KS)} if rows else {},
                "shuffled_energy_topk_median": {kk: float(np.median([r["energy_topk_shuffled"][t] for r in rows])) for t, kk in enumerate(KS)} if rows else {}}
        for nm in ("same_layer", "other_layer"):
            cs = [r["partners"][nm]["abs_cos_after_removing_topk"] for r in rows if nm in r["partners"]]
            if cs:
                summ[f"{nm}_abs_cos_median_before_and_after_removing_k"] = dict(zip(["0"] + [str(kk) for kk in KS], np.median(cs, axis=0).tolist()))
        res["layers"][str(L)] = {"summary": summ, "rows": rows}
        print(L, json.dumps(summ), f"{time.time() - t0:.0f}s", flush=True)
    res["control_threshold_same_layer"], res["control_threshold_other_layer"] = thr[2], thr[3]
    res["seconds"] = time.time() - t0
    json.dump(res, open(K.RES / f"shared_dirs_{a.model}.json", "w"), indent=1)
    print("SHARED_DONE", a.model)


if __name__ == "__main__":
    main()
