"""Spec 010 P1b (moved to 011): net gain over the 008 chain from results/p1b/{test,train,verify,synthetic}.json(l).

    python experiments/011-limits-and-models/p1b_summary.py      -> results/p1b/summary.json
"""
import json
from pathlib import Path

R = Path("experiments/011-limits-and-models/results/p1b")


def tier(g):
    if g < 0.005:
        return "< 0.005: no missed structure found within this probe's reach"
    if g <= 0.02:
        return "0.005-0.02: a little missed structure; see the ablation"
    return "> 0.02: clearly missed structure; needs a dedicated analysis"


def main():
    rows = [json.loads(l) for l in open(R / "test.jsonl")]
    tr = json.load(open(R / "train.json"))
    n = sum(r["n"] for r in rows)
    chain = sum(r["chain_code_bits"] for r in rows)
    off = sum(r["offline_bits"] for r in rows)
    on = sum(r["online_bits"] for r in rows)
    mats = json.load(open("data/008/fp4/matrices.json"))["matrices"]
    all_w = sum(m["codewords"] for m in mats)
    experts = len({(m["layer"], m["expert"]) for m in mats})
    cost = 16.0 * tr["n_params"]
    g_off = (chain - off) / n
    g_on = (chain - on) / n
    out = {"test_matrices": len(rows), "test_weights": n, "n_params": tr["n_params"],
           "chain_code_bits_per_weight": chain / n, "offline_bits_per_weight": off / n, "online_bits_per_weight": on / n,
           "offline_gain_before_model_cost": g_off,
           "model_cost_over_test_matrices": cost / n, "offline_net_gain_cost_over_test": g_off - cost / n,
           "model_cost_over_whole_model": cost / all_w, "whole_model_experts": experts, "whole_model_weights": all_w,
           "offline_net_gain_cost_over_whole_model": g_off - cost / all_w,
           "online_net_gain": g_on, "online_lr": tr["online_lr_choice"]["chosen"], "offline_lr": tr["lr"],
           "per_matrix": {r["task"]: {"offline_gain": (r["chain_code_bits"] - r["offline_bits"]) / r["n"],
                                      "online_gain": (r["chain_code_bits"] - r["online_bits"]) / r["n"]} for r in rows}}
    for k in ("offline_net_gain_cost_over_test", "offline_net_gain_cost_over_whole_model", "online_net_gain"):
        out[k + "_percent"] = 100 * out[k] / 4.25
    best = max(out["offline_net_gain_cost_over_whole_model"], out["online_net_gain"])
    out["tier"] = tier(best)
    if (R / "ablate.jsonl").exists():
        ab = [json.loads(l) for l in open(R / "ablate.jsonl")]
        na = sum(r["n"] for r in ab)
        ch = sum(r["chain_bits"] for r in ab)
        res = {}
        for d in ab[0]["bits"]:
            g = (ch - sum(r["bits"][d] for r in ab)) / na
            res[d] = {"offline_gain_without_group": g, "gain_lost": g_off - g}
        out["ablation"] = {"note": "model retrained without the group (group set to a constant in training and test); "
                                   "offline gain before model cost, same 24 test matrices",
                           "full_offline_gain": g_off, "groups": res}
    if (R / "diag.json").exists():
        out["diagnostics"] = json.load(open(R / "diag.json"))["aggregate"]
    out["verify"] = {k: v for k, v in json.load(open(R / "verify.json")).items() if k != "rows"}
    out["synthetic"] = json.load(open(R / "synthetic.json"))["checks"]
    json.dump(out, open(R / "summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_matrix"}, indent=1))


if __name__ == "__main__":
    main()
