"""Spec 014, supplement (user request): neuron / hidden-dimension magnitude statistics on the full 008 and 009 samples
(DeepSeek 516 matrices = plan B first 172 experts; Kimi-K3 552 matrices = plan_552), no figures.

    PYTHONPATH=. python experiments/014-heatmaps/fullsample.py run {deepseek|kimi} [--workers N]
    PYTHONPATH=. python experiments/014-heatmaps/fullsample.py summarize

Per matrix, for the code value v and the dequantized value w = v 2^(E-127):
  coefficient of variation (std / mean) of the row RMS and of the column RMS, real and after a fixed-seed shuffle
  of all elements (baseline); the neuron dimension is the rows of w1 / w3 and the columns of w2.
Per matrix also: number of neurons / hidden dimensions whose RMS (dequantized) is below 0.5x or above 1.5x the
median; per expert: how many low neurons coincide in w1, w3 and w2.
Codes and scales are read with redundancy.adapters.fp4 as in 008 / 009; nothing else is imported.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

from redundancy.adapters.fp4 import E2M1, FP4  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = HERE / "results"
SEED = 20261123
MODELS = {
    "deepseek": dict(plan="experiments/008-scale-media/results/fp4/B/plan.json", first=172,
                     items="experiments/008-scale-media/results/fp4_increments_B.json",
                     samples="data/008/fp4/matrices.json", snapshot_env="SNAPSHOT"),
    "kimi": dict(plan="experiments/009-kimi-k3/results/plan_552.json", first=None, items=None,
                 samples="data/009/kimi/matrices.json", snapshot_env="KIMI_SNAPSHOT"),
}


def tasks_of(model):
    m = MODELS[model]
    plan = json.load(open(ROOT / m["plan"]))
    tasks = plan["tasks"][:m["first"]] if m["first"] else plan["tasks"]
    if m["items"]:
        ents = sorted(e for t in tasks for e in t["entries"])
        assert ents == sorted(json.load(open(ROOT / m["items"]))["items"]), "sample differs from 008"
    return tasks


def cv(a):
    return float(a.std() / a.mean())


def rms(x):
    x2 = x * x
    return np.sqrt(x2.mean(axis=1)), np.sqrt(x2.mean(axis=0))


def expert_task(model, task):
    t0 = time.time()
    m = MODELS[model]
    ad = FP4({"samples": str(ROOT / m["samples"]), "snapshot_env": m["snapshot_env"]})
    out = {}
    low = {}
    for name in task["entries"]:
        C, E = ad._read(ad.samples[name])
        role = name.rsplit(".", 1)[1]
        v = E2M1[C.astype(np.int64)]
        w = v * np.exp2(np.repeat(E.astype(np.float64), 32, axis=1) - 127)
        R, W = v.shape
        perm = np.random.default_rng(SEED).permutation(v.size)
        rec = {"model": model, "matrix": name, "layer": task["layer"], "expert": task["expert"], "role": role,
               "shape": [R, W]}
        for vs, x in (("code", v), ("deq", w)):
            rr, cr = rms(x)
            xs = x.ravel()[perm].reshape(R, W)
            rrs, crs = rms(xs)
            neuron, hidden = (rr, cr) if role != "w2" else (cr, rr)
            neuron_s, hidden_s = (rrs, crs) if role != "w2" else (crs, rrs)
            rec[vs] = {"neuron_cv": cv(neuron), "neuron_cv_shuffled": cv(neuron_s),
                       "hidden_cv": cv(hidden), "hidden_cv_shuffled": cv(hidden_s)}
            if vs == "deq":
                for dim, a in (("neuron", neuron), ("hidden", hidden)):
                    med = np.median(a)
                    rec[vs][f"{dim}_below_0.5x"] = int((a < 0.5 * med).sum())
                    rec[vs][f"{dim}_above_1.5x"] = int((a > 1.5 * med).sum())
                    rec[vs][f"{dim}_min_ratio"] = float(a.min() / med)
                    rec[vs][f"{dim}_max_ratio"] = float(a.max() / med)
                med = np.median(neuron)
                low[role] = set(np.nonzero(neuron < 0.5 * med)[0].tolist())
        out[name] = rec
    ov = {"low_neurons": {r: len(s) for r, s in low.items()}}
    if all(r in low for r in ("w1", "w3", "w2")):
        ov.update(w1_and_w3=len(low["w1"] & low["w3"]), w1_and_w2=len(low["w1"] & low["w2"]),
                  w3_and_w2=len(low["w3"] & low["w2"]), all_three=len(low["w1"] & low["w3"] & low["w2"]))
    return {"task": task["id"], "layer": task["layer"], "expert": task["expert"], "matrices": out, "overlap": ov,
            "seconds": time.time() - t0}


def cmd_run(a):
    RES.mkdir(parents=True, exist_ok=True)
    path = RES / f"fullsample_{a.model}_items.jsonl"
    done = {json.loads(l)["task"] for l in open(path)} if path.exists() else set()
    todo = [t for t in tasks_of(a.model) if t["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.model}: {len(todo)} experts to run", flush=True)
    t0 = time.time()
    with open(path, "a") as fo, ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(expert_task, [a.model] * len(todo), todo), 1):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            if i % 20 == 0:
                print(f"{i}/{len(todo)} {time.time() - t0:.0f} s", flush=True)
    print(f"RUN_DONE {a.model} wall {time.time() - t0:.0f} s", flush=True)


def q(a):
    a = np.asarray(a)
    return {"median": float(np.median(a)), "p90": float(np.quantile(a, 0.9)), "max": float(a.max()),
            "min": float(a.min())}


def cmd_summarize(a):
    out = {"seed": SEED, "models": {}}
    for model in MODELS:
        rows = [json.loads(l) for l in open(RES / f"fullsample_{model}_items.jsonl")]
        want = {t["id"] for t in tasks_of(model)}
        assert {r["task"] for r in rows} == want, f"{model}: incomplete"
        mats = [m for r in rows for m in r["matrices"].values()]
        res = {"n_matrices": len(mats), "n_experts": len(rows), "by_role": {}, "by_layer": {}, "strong": [],
               "wall_note": "see results/fullsample.log"}
        for role in ("w1", "w3", "w2"):
            sel = [m for m in mats if m["role"] == role]
            ratio = [m["deq"]["neuron_cv"] / m["deq"]["neuron_cv_shuffled"] for m in sel]
            res["by_role"][role] = {
                "n": len(sel),
                "deq_neuron_cv": q([m["deq"]["neuron_cv"] for m in sel]),
                "deq_neuron_cv_over_shuffled": q(ratio),
                "code_neuron_cv": q([m["code"]["neuron_cv"] for m in sel]),
                "deq_hidden_cv_over_shuffled": q([m["deq"]["hidden_cv"] / m["deq"]["hidden_cv_shuffled"] for m in sel]),
                "n_with_low_neurons": int(sum(m["deq"]["neuron_below_0.5x"] > 0 for m in sel)),
                "n_with_10pct_low_neurons": int(sum(m["deq"]["neuron_below_0.5x"] >= 0.1 * m["shape"][0 if role != "w2" else 1]
                                                    for m in sel)),
                "frac_neurons_low": float(sum(m["deq"]["neuron_below_0.5x"] for m in sel)
                                          / sum(m["shape"][0 if role != "w2" else 1] for m in sel)),
                "ratio_counts": {k: int(sum(lo <= x < hi for x in ratio)) for k, (lo, hi) in
                                 {"<2": (0, 2), "2-5": (2, 5), "5-10": (5, 10), "10-20": (10, 20), ">=20": (20, 1e9)}.items()},
            }
        layers = sorted({m["layer"] for m in mats})
        for L in layers:
            sel = [m for m in mats if m["layer"] == L]
            exs = [r for r in rows if r["layer"] == L]
            res["by_layer"][str(L)] = {
                "experts": [r["expert"] for r in exs],
                "w2_deq_neuron_cv": [round(m["deq"]["neuron_cv"], 4) for m in sel if m["role"] == "w2"],
                "w2_code_neuron_cv": [round(m["code"]["neuron_cv"], 4) for m in sel if m["role"] == "w2"],
                "w1_deq_neuron_cv": [round(m["deq"]["neuron_cv"], 4) for m in sel if m["role"] == "w1"],
                "low_neurons_w2": [m["deq"]["neuron_below_0.5x"] for m in sel if m["role"] == "w2"],
                "low_neurons_all_three": [r["overlap"].get("all_three") for r in exs],
            }
        for m in sorted(mats, key=lambda m: -m["deq"]["neuron_cv"])[:30]:
            res["strong"].append({k: m[k] for k in ("matrix", "role", "shape")} | {
                "deq_neuron_cv": m["deq"]["neuron_cv"], "code_neuron_cv": m["code"]["neuron_cv"],
                "neuron_below_0.5x": m["deq"]["neuron_below_0.5x"]})
        res["seconds_total"] = float(sum(r["seconds"] for r in rows))
        out["models"][model] = res
    json.dump(out, open(RES / "fullsample_summary.json", "w"), indent=1)
    for model, res in out["models"].items():
        print(f"== {model}: {res['n_matrices']} matrices")
        for role, s in res["by_role"].items():
            print(f"  {role}: deq neuron CV median {s['deq_neuron_cv']['median']:.4f} p90 {s['deq_neuron_cv']['p90']:.4f} "
                  f"max {s['deq_neuron_cv']['max']:.4f}; ratio to shuffled median {s['deq_neuron_cv_over_shuffled']['median']:.1f} "
                  f"max {s['deq_neuron_cv_over_shuffled']['max']:.1f}; ratio bins {s['ratio_counts']}; code CV median "
                  f"{s['code_neuron_cv']['median']:.4f}; with low neurons {s['n_with_low_neurons']}, >=10% low "
                  f"{s['n_with_10pct_low_neurons']}, frac low {s['frac_neurons_low']:.4f}; hidden ratio median "
                  f"{s['deq_hidden_cv_over_shuffled']['median']:.2f}")
        print("  strongest:", [(s["matrix"], round(s["deq_neuron_cv"], 3), s["neuron_below_0.5x"]) for s in res["strong"][:12]])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model", choices=list(MODELS))
    r.add_argument("--workers", type=int, default=48)
    r.add_argument("--limit", type=int, default=None)
    sub.add_parser("summarize")
    a = ap.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
