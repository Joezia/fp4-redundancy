"""Spec 009, enlarged sample for the chain model: layers 1-92, rounds 1 and 2 (184 experts, 552 matrices).

    python experiments/009-kimi-k3/make_plan_552.py

Same rule as make_plan.py (seed 20261118; per layer, experts in the order of numpy default_rng([seed, 9, layer])
.permutation; round r = r-th expert of every layer; one task = w3, w1, w2 of one expert). The 24 matrices of
plan_36.json that belong to rounds 1-2 are part of this plan; their chain results are reused (--skip).
"""
import json
from pathlib import Path

import numpy as np

SEED = 20261118
OUT = Path("experiments/009-kimi-k3/results")


def main():
    mats = json.load(open("data/009/kimi/matrices.json"))["matrices"]
    experts, shard = {}, {}
    for m in mats:
        experts.setdefault(m["layer"], set()).add(m["expert"])
        shard[(m["layer"], m["expert"])] = m["weight_file"]
    layers = sorted(experts)
    assert layers == list(range(1, 93)), layers[:3] + layers[-3:]
    order = {L: [int(e) for e in np.random.default_rng([SEED, 9, L]).permutation(sorted(experts[L]))] for L in layers}
    saved = json.load(open(OUT / "expert_order.json"))
    assert all(saved[str(L)] == order[L] for L in layers), "expert order differs from 009's expert_order.json"
    tasks = []
    for r in (1, 2):
        for L in layers:
            e = order[L][r - 1]
            tasks.append({"id": f"round{r}.L{L}.E{e}", "round": r, "layer": L, "expert": e, "shard": shard[(L, e)],
                          "entries": [f"L{L}.E{e}.w3", f"L{L}.E{e}.w1", f"L{L}.E{e}.w2"]})
    plan = {"seed": SEED, "rule": "per layer, experts in the order of numpy default_rng([seed, 9, layer]).permutation; "
                                  "round r = r-th expert of every layer; one task = w3, w1, w2 of one expert",
            "layers": layers, "rounds": [1, 2], "tasks": tasks}
    json.dump(plan, open(OUT / "plan_552.json", "w"), indent=1)
    old = {e for t in json.load(open(OUT / "plan_36.json"))["tasks"] for e in t["entries"]}
    new = {e for t in tasks for e in t["entries"]}
    print(len(tasks), "experts,", len(new), "matrices; reused from plan_36:", len(old & new))


if __name__ == "__main__":
    main()
