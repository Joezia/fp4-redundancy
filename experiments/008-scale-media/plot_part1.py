"""Spec 008 Part 1: aggregate the A and B indicator runs and redraw the 007-v2 figures with the existing plot CLI.

    PYTHONPATH=. python experiments/008-scale-media/plot_part1.py

Only experts of complete rounds enter (every layer then has the same number of experts):
B rounds 1..E_B (all indicators) and every complete A round (cheap indicators only).
Writes results/fp4/indicators/ (aggregated records), results/fp4/indicators_plan.json,
results/fp4/figures_008.json (figure settings of configs/fp4.json with the sample sizes of this run) and
figures/fp4_indicators/.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

E = Path("experiments/008-scale-media")
R = E / "results" / "fp4"
PY = sys.executable


def entries(path):
    out = set()
    if path.exists():
        for line in open(path):
            try:
                out.add(json.loads(line)["entry"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def complete_rounds(plan, done):
    ok = defaultdict(lambda: True)
    for t in plan["tasks"]:
        ok[t["round"]] &= all(e in done for e in t["entries"])
    return sorted(r for r, v in ok.items() if v)


def main():
    pB, pA = json.load(open(R / "B" / "plan.json")), json.load(open(R / "A" / "plan.json"))
    rB = complete_rounds(pB, entries(R / "B" / "entries.jsonl"))
    eb = 0
    while eb + 1 in rB:
        eb += 1
    rA = complete_rounds(pA, entries(R / "A" / "entries.jsonl"))
    tasks = [t for t in pB["tasks"] if t["round"] <= eb] + [t for t in pA["tasks"] if t["round"] in rA]
    plan = {"note": f"complete rounds only: B rounds 1-{eb}, A rounds {rA}", "E_B": eb, "E_A": eb + len(rA),
            "tasks": tasks}
    json.dump(plan, open(R / "indicators_plan.json", "w"), indent=1)
    nB = sum(len(t["entries"]) for t in pB["tasks"] if t["round"] <= eb)
    nAll = sum(len(t["entries"]) for t in tasks)
    ea = eb + len(rA)
    print(f"E_B={eb} ({nB} matrices), E_A={ea} ({nAll} matrices)", flush=True)

    subprocess.run([PY, "-m", "redundancy.sweep", "aggregate", "--config", "redundancy/configs/fp4_008.json",
                    "--out", f"{R}/A,{R}/B", "--results", f"{R}/indicators", "--plan", f"{R}/indicators_plan.json"],
                   check=True)
    man = json.load(open(R / "indicators" / "manifest.json"))
    n = man["n_entries"]
    rp = json.load(open(R / "indicators" / "data" / "R7a_partner.json"))["entries"]
    n_pairs = sum(1 for v in rp.values() if v.get("applicable", True) is not False)   # w1 rows only
    figs = json.load(open("redundancy/configs/fp4.json"))["figures"]
    # notes quoting numbers measured on 007's 36 matrices do not hold for this sample
    for k in ("R6", "R7a_row"):
        figs[k].pop("note_zh", None)
    figs["overview_title"] = (f"FP4 weights: {n.get('R1', 0)} matrices, {ea} experts per layer "
                              f"(R7a corresponding rows: {n_pairs} W1–W3 pairs; "
                              f"R5, R7a rows, R7b, R8: {nB} matrices, {eb} experts per layer)")
    json.dump({"figures": figs}, open(R / "figures_008.json", "w"), indent=1, ensure_ascii=False)
    out = E / "figures" / "fp4_indicators"
    subprocess.run([PY, "-m", "redundancy.plot", "--results", f"{R}/indicators", "--out", str(out),
                    "--overview", str(out), "--synthetic", "experiments/007-figures-v2/results",
                    "--config", f"{R}/figures_008.json"], check=True)
    json.dump({"E_B": eb, "E_A": ea, "B_matrices": nB, "all_matrices": nAll, "n_entries": n,
               "A_rounds": rA, "W1_W3_pairs": n_pairs}, open(R / "indicators_samples.json", "w"), indent=1)


if __name__ == "__main__":
    main()
