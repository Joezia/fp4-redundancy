"""Spec 009, enlarged sample: summarise the chain model (default order) on 552 Kimi-K3 matrices.

    PYTHONPATH=. python experiments/009-kimi-k3/chain_552.py

Reads results/increment_552/items.jsonl plus the 24 matrices of results/increment_36/items.jsonl that belong to
plan_552 (rounds 1-2; their default-order chain is reused). Writes results/chain_552.json:
  - per matrix: increment of each pattern and the total, bit/weight (= 4.25 x fraction of raw bits);
  - over all matrices: the pooled value (sum of bits / sum of weights, as 008 reports), and the mean, median and
    10% / 90% quantiles of the per-matrix values;
  - by layer group (layers 1-10, 11-20, ..., 91-92): pooled increments and total;
  - the 008 DeepSeek 516-matrix values for comparison (results/fp4_increments_B.json of 008), with the same
    per-matrix statistics when 008's per-matrix rows are available.
"""
import json
from pathlib import Path

import numpy as np

from redundancy.increment_run import aggregate

E = Path("experiments/009-kimi-k3")
R = E / "results"
D8 = Path("experiments/008-scale-media/results")
RAW = 4.25


def rows_of(path):
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[r["item"]] = r
    return out


def per_matrix(rows):
    out = {}
    for r in rows:
        c = r["chains"]["default"]
        d = {k: RAW * v for k, v in c["increment"].items()}
        d["total"] = RAW * c["total"]
        d["used"] = c["used"]
        out[r["item"]] = d
    return out


def stats(pm, keys):
    res = {}
    for k in keys:
        v = np.array([x[k] for x in pm.values()])
        res[k] = {"mean": float(v.mean()), "median": float(np.median(v)), "p10": float(np.quantile(v, 0.1)),
                  "p90": float(np.quantile(v, 0.9)), "min": float(v.min()), "max": float(v.max())}
    return res


def pooled(rows):
    a = aggregate(rows, "x")["orders"]["default"]
    inc = {k: RAW * v / 100 for k, v in a["increment_raw"].items()}
    inc["total"] = RAW * a["total"] / 100
    return inc


def main():
    plan = json.load(open(R / "plan_552.json"))
    want = [e for t in plan["tasks"] for e in t["entries"]]
    new = rows_of(R / "increment_552" / "items.jsonl")
    old = rows_of(R / "increment_36" / "items.jsonl")
    rows, src = [], {}
    for e in want:
        if e in new:
            rows.append(new[e])
            src[e] = "increment_552"
        elif e in old:
            rows.append(old[e])
            src[e] = "increment_36 (reused)"
    missing = [e for e in want if e not in src]
    params = {json.dumps(r["params"], sort_keys=True) for r in rows}
    assert len(params) == 1, "chain parameters differ between rows"
    pm = per_matrix(rows)
    keys = [k for k in rows[0]["chains"]["default"]["increment"]] + ["total"]
    allp = pooled(rows)
    groups = {}
    for lo in range(1, 93, 10):
        hi = min(lo + 9, 92)
        sel = [r for r in rows if lo <= int(r["item"].split(".")[0][1:]) <= hi]
        if sel:
            g = pooled(sel)
            g["n_matrices"] = len(sel)
            groups[f"{lo}-{hi}"] = g
    roles = {}
    for role in ("w1", "w2", "w3"):
        sel = [r for r in rows if r["item"].endswith(role)]
        g = pooled(sel)
        g["n_matrices"] = len(sel)
        roles[role] = g
    d8 = json.load(open(D8 / "fp4_increments_B.json"))
    ds = dict(d8["orders"]["default"]["increment_bits_per_weight"])
    ds["total"] = d8["orders"]["default"]["total_bits_per_weight"]
    ds_out = {"n_matrices": d8["n_items"], "pooled": ds, "params": d8["params"]}
    dirs8 = [D8 / "fp4" / d for d in ("increment_B", "increment_B_s1", "increment_B_s2", "increment_B_s3")]
    all8 = {}
    for d in dirs8:
        if (d / "items.jsonl").exists():
            all8.update(rows_of(d / "items.jsonl"))
    r8 = [all8[i] for i in d8["items"] if i in all8]
    if len(r8) == d8["n_items"]:
        pooled8 = pooled(r8)
        assert abs(pooled8["total"] - ds["total"]) < 1e-6, (pooled8["total"], ds["total"])
        ds_out["per_matrix_stats"] = stats(per_matrix(r8), keys)
        ds_out["per_matrix_source"] = [str(d) for d in dirs8]
    out = {
        "dataset": "Kimi-K3 FP4, layers 1-92, rounds 1-2 (seed 20261118), 184 experts x (w3, w1, w2)",
        "unit": "bit/weight (raw 4.25); percent_of_4.25 = value / 4.25 x 100",
        "order": rows[0]["chains"]["default"]["order"],
        "params": rows[0]["params"],
        "n_matrices": len(rows), "missing": missing,
        "sources": {"increment_552": sum(v == "increment_552" for v in src.values()),
                    "increment_36 (reused)": sum(v != "increment_552" for v in src.values())},
        "pooled": allp,
        "pooled_percent_of_4.25": {k: 100 * v / RAW for k, v in allp.items()},
        "per_matrix_stats": stats(pm, keys),
        "switch_off_counts": {p: sum(1 for r in rows if not r["chains"]["default"]["used"].get(p, True))
                              for p in rows[0]["chains"]["default"]["used"]},
        "by_layer_group": groups,
        "by_role": roles,
        "kimi_36_first_test": {k: v for k, v in json.load(open(R / "kimi_increments_36.json"))["orders"]["default"]
                               ["increment_bits_per_weight"].items()},
        "deepseek_008_516": ds_out,
        "per_matrix": pm,
    }
    out["kimi_36_first_test"]["total"] = json.load(open(R / "kimi_increments_36.json"))["orders"]["default"]["total_bits_per_weight"]
    json.dump(out, open(R / "chain_552.json", "w"), indent=1)
    print("matrices:", len(rows), "missing:", len(missing), out["sources"])
    print("pattern | pooled | % of 4.25 | mean | median | p10 | p90 | DeepSeek 516")
    for k in keys:
        s = out["per_matrix_stats"][k]
        print(f"{k} | {allp[k]:.4f} | {100 * allp[k] / RAW:.2f}% | {s['mean']:.4f} | {s['median']:.4f} | {s['p10']:.4f} | "
              f"{s['p90']:.4f} | {ds.get(k, float('nan')):.4f}")
    print("switch-off:", out["switch_off_counts"])
    for g, v in groups.items():
        print(g, v["n_matrices"], {k: round(x, 4) for k, x in v.items() if k != "n_matrices"})
    for g, v in roles.items():
        print(g, round(v["total"], 4), round(v["R7"], 4))
    if "per_matrix_stats" in ds_out:
        print("DeepSeek per-matrix total:", ds_out["per_matrix_stats"]["total"])


if __name__ == "__main__":
    main()
