"""Spec 009 (first test, 36 Kimi-K3 matrices): aggregate indicators and chain, draw the figures, compare
with DeepSeek (007's 36 matrices and 008's 516-matrix sample).

    PYTHONPATH=. python experiments/009-kimi-k3/finalize.py

Writes results/indicators/ (aggregated records), figures/ (007-v2 indicator figures + kimi_summary),
results/kimi_increments_36.json and results/summary.json. All numbers come from result files.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from redundancy import increment_plot
from redundancy.increment_run import aggregate

E = Path("experiments/009-kimi-k3")
R = E / "results"
F = E / "figures"
D8 = Path("experiments/008-scale-media")
PY = sys.executable

spec = importlib.util.spec_from_file_location("analyze_fp4", D8 / "analyze_fp4.py")
A8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A8)


def rows_of(path):
    out = {}
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["item"]] = r
    return list(out.values())


def chain():
    rows = rows_of(R / "increment_36" / "items.jsonl")
    if not rows:
        return None
    agg = aggregate(rows, "Kimi-K3 FP4")
    agg["items"] = sorted(r["item"] for r in rows)
    agg["params"] = rows[0]["params"]
    agg["switch_off_counts"] = {p: sum(1 for r in rows if not r["chains"]["default"]["used"].get(p, True))
                                for p in rows[0]["chains"]["default"]["used"]}
    agg["if_forced"] = {o: {p: 100 * float(np.average([r["chains"][o]["if_forced"][p] for r in rows],
                                                     weights=[r["raw_bits"] for r in rows]))
                            for p in rows[0]["chains"][o]["if_forced"]} for o in rows[0]["chains"]}
    for o, v in agg["orders"].items():
        v["increment_bits_per_weight"] = {k: x * 4.25 / 100 for k, x in v["increment_raw"].items()}
        v["total_bits_per_weight"] = v["total"] * 4.25 / 100
    layers = {}
    for L in sorted({int(r["item"].split(".")[0][1:]) for r in rows}):
        sel = [r for r in rows if int(r["item"].split(".")[0][1:]) == L]
        w = np.array([r["raw_bits"] for r in sel])
        inc = {k: 4.25 * float(np.average([r["chains"]["default"]["increment"][k] for r in sel], weights=w))
               for k in sel[0]["chains"]["default"]["increment"]}
        inc["total"] = 4.25 * float(np.average([r["chains"]["default"]["total"] for r in sel], weights=w))
        inc["n_matrices"] = len(sel)
        layers[str(L)] = inc
    agg["by_layer_bits_per_weight"] = layers
    json.dump(agg, open(R / "kimi_increments_36.json", "w"), indent=1)
    return agg


def indicators():
    run = R / "indicators_run"
    if not (run / "entries.jsonl").exists():
        return None
    subprocess.run([PY, "-m", "redundancy.sweep", "aggregate", "--config", "redundancy/configs/fp4_kimi.json",
                    "--out", str(run), "--results", str(R / "indicators"), "--plan", str(R / "plan_36.json")],
                   check=True)
    subprocess.run([PY, "-m", "redundancy.plot", "--results", str(R / "indicators"), "--out", str(F),
                    "--overview", str(F), "--synthetic", "experiments/007-figures-v2/results",
                    "--config", "redundancy/configs/fp4_kimi.json"], check=True)
    man = json.load(open(R / "indicators" / "manifest.json"))
    rows = A8.load(run / "entries.jsonl")
    per = {n: {"layer": r["attrs"]["layer"], "role": r["attrs"]["role"], **A8.key_numbers(r["results"])}
           for n, r in rows.items()}
    keys = ["R1", "R2", "R3", "R4", "R6", "R7", "R10"]
    med = {k: float(np.median([v[k] for v in per.values() if k in v])) for k in keys
           if any(k in v for v in per.values())}
    by_layer = {}
    for L in sorted({v["layer"] for v in per.values()}):
        sel = [v for v in per.values() if v["layer"] == L]
        by_layer[str(L)] = {k: float(np.median([v[k] for v in sel if k in v])) for k in keys if any(k in v for v in sel)}
    m007 = json.load(open(D8 / "results" / "fp4" / "part1_summary.json"))
    v007 = json.load(open("experiments/007-figures-v2/results/manifest.json"))["verdicts"]
    return {"verdicts": man["verdicts"], "verdicts_deepseek_007": {k: v007.get(k) for k in man["verdicts"]},
            "n_entries": man["n_entries"], "median_key_numbers": med, "median_by_layer": by_layer,
            "deepseek_median_007_36": m007["median_007_36"], "deepseek_median_008_774": m007["median_all"],
            "per_matrix": per}


def main():
    F.mkdir(parents=True, exist_ok=True)
    ch = chain()
    ind = indicators()
    out = {"chain": None, "indicators": None}
    if ch:
        increment_plot.draw(ch, f"Kimi-K3 FP4 ({ch['n_items']} matrices)", F / "kimi_summary.png")
        ds36 = json.load(open(D8 / "results" / "fp4_increments_v6_36.json"))
        ds516 = json.load(open(D8 / "results" / "fp4_increments_B.json"))
        out["chain"] = {
            "kimi_36": {o: {"bits_per_weight": v["increment_bits_per_weight"], "total_bits_per_weight": v["total_bits_per_weight"],
                            "total_percent": v["total"]} for o, v in ch["orders"].items()},
            "deepseek_007_36": {"bits_per_weight": ds36["orders"]["default"]["increment_bits_per_weight"],
                                "total_bits_per_weight": ds36["orders"]["default"]["total_bits_per_weight"]},
            "deepseek_008_516": {"bits_per_weight": ds516["orders"]["default"]["increment_bits_per_weight"],
                                 "total_bits_per_weight": ds516["orders"]["default"]["total_bits_per_weight"]},
            "kimi_by_layer": ch["by_layer_bits_per_weight"], "switch_off_counts": ch["switch_off_counts"],
            "if_forced_percent": ch["if_forced"]["default"], "params": ch["params"],
            "params_deepseek": ds516["params"]}
    if ind:
        out["indicators"] = {k: v for k, v in ind.items() if k != "per_matrix"}
        json.dump(ind["per_matrix"], open(R / "indicator_key_numbers.json", "w"), indent=1)
    json.dump(out, open(R / "summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
