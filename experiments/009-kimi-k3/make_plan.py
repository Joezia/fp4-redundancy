"""Spec 009: sampling plan for Kimi-K3 (first test: 36 matrices) and the dev matrices.

    python experiments/009-kimi-k3/make_plan.py

Rule (same as 008): in every MoE layer the experts are ordered by numpy default_rng([seed, 9, layer])
.permutation; round r = r-th expert of every layer; one task = w3, w1, w2 of one expert.
The 36-matrix test takes rounds 1-3 in 4 layers spread over the 92 MoE layers (1, 31, 62, 92), the
way 007 took 3 experts in layers 0, 13, 26, 39 of DeepSeek. A later, larger sample extends the same
rounds to all layers, so the test experts stay part of it.
Dev matrices (parameter selection only, never reported): the last expert of the permutation in 7
other layers, roles mixed.
"""
import json
from pathlib import Path

import numpy as np

SEED = 20261118
OUT = Path("experiments/009-kimi-k3/results")
FIGS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7a_unit", "R7a_row", "R7a_partner", "R7b", "R8", "R9", "R10"]


def main():
    mats = json.load(open("data/009/kimi/matrices.json"))["matrices"]
    experts = {}
    shard = {}
    for m in mats:
        experts.setdefault(m["layer"], set()).add(m["expert"])
        shard[(m["layer"], m["expert"])] = m["weight_file"]
    layers = sorted(experts)
    order = {L: [int(e) for e in np.random.default_rng([SEED, 9, L]).permutation(sorted(experts[L]))]
             for L in layers}
    test_layers = [1, 31, 62, 92]
    tasks = []
    for r in (1, 2, 3):
        for L in test_layers:
            e = order[L][r - 1]
            tasks.append({"id": f"round{r}.L{L}.E{e}", "round": r, "layer": L, "expert": e,
                          "shard": shard[(L, e)], "entries": [f"L{L}.E{e}.w3", f"L{L}.E{e}.w1", f"L{L}.E{e}.w2"],
                          "figs": FIGS})
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {"seed": SEED, "rule": "per layer, experts in the order of numpy default_rng([seed, 9, layer]).permutation; "
                                  "round r = r-th expert of every layer; one task = w3, w1, w2 of one expert",
            "layers": test_layers, "rounds": [1, 2, 3], "experts_per_layer": {str(L): len(experts[L]) for L in layers},
            "figs": FIGS, "tasks": tasks}
    json.dump(plan, open(OUT / "plan_36.json", "w"), indent=1)
    json.dump(order, open(OUT / "expert_order.json", "w"))
    dev_layers, roles = [10, 25, 45, 55, 70, 80, 88], ["w1", "w2", "w3", "w1", "w2", "w3", "w1"]
    dev = [f"L{L}.E{order[L][-1]}.{r}" for L, r in zip(dev_layers, roles)]
    json.dump({"entries": dev, "rule": "last expert of the permutation in layers " + str(dev_layers)},
              open(OUT / "dev_entries.json", "w"), indent=1)
    print(len(tasks) * 3, "matrices;", [t["id"] for t in tasks])
    print("dev:", ",".join(dev))


if __name__ == "__main__":
    main()
