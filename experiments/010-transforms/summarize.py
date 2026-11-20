"""Spec 010: net saving of every trial relative to the 008 chain, from the result files.

    python experiments/010-transforms/summarize.py      -> results/summary.json (and a printout)

Net saving = (008 chain code length of the affected matrices) - (code length with the trial, including its
side information), in bit per weight of the affected matrices; % = of the raw 4.25 bit per weight.
"""
import json
from pathlib import Path

import numpy as np

R = Path("experiments/010-transforms/results")


def jl(name):
    out = {}
    p = R / name
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            out[r["task"]] = r
    return out


def Lf(ch):
    return ch["L"][-1]


def w(ch):
    return ch["raw_bits"] / 4.25


def q(v):
    v = np.asarray(v, dtype=float)
    return {"median": float(np.median(v)), "p99": float(np.percentile(v, 99)) if len(v) else None}


def main():
    base = {k: v["chain"] for k, v in jl("base.jsonl").items()}
    out = {"n_base_matrices": len(base)}
    # T1
    t1 = jl("t1.jsonl")
    if t1:
        d = {"w1": 0.0, "w2": 0.0, "w3": 0.0, "w3_given_w1": 0.0}
        W = 0.0
        for k, r in t1.items():
            n = k + "."
            d["w1"] += Lf(base[n + "w1"]) - Lf(r["w1_given_w3_after"])
            d["w2"] += Lf(base[n + "w2"]) - Lf(r["w2_after"])
            d["w3"] += Lf(base[n + "w3"]) - Lf(r["w3_after"])
            d["w3_given_w1"] += Lf(r["w3_given_w1_before"]) - Lf(r["w3_given_w1_after"])
            W += w(base[n + "w1"])
        net = (d["w1"] + d["w2"] + d["w3"]) / (3 * W)
        out["T1"] = {"experts": len(t1), "all_bitwise_identical": all(r["invariance"]["bitwise_identical"] for r in t1.values()),
                     "rows_flipped_fraction": float(np.mean([r["invariance"]["rows_flipped"] / r["rows"] for r in t1.values()])),
                     "w3_given_w1_saving_bits_per_weight": d["w3_given_w1"] / W,
                     "w2_saving_bits_per_weight": d["w2"] / W, "w1_saving_bits_per_weight": d["w1"] / W,
                     "w3_alone_saving_bits_per_weight": d["w3"] / W,
                     "net_bits_per_weight": net, "net_percent": 100 * net / 4.25,
                     "affected": "w1, w2, w3 of each expert"}
    # T2
    t2 = jl("t2.jsonl")
    if t2:
        res = {}
        for kind in ("real", "shuffled"):
            num, W, gain3, gain1 = 0.0, 0.0, 0.0, 0.0
            mc, mp = [], []
            for k, r in t2.items():
                L_, pair = k.split(".", 1)
                b = pair.split("-")[1]
                x = r[kind]
                if kind == "real":
                    b1, b3 = Lf(base[f"{L_}.{b}.w1"]), Lf(base[f"{L_}.{b}.w3"])
                else:
                    b1, b3 = Lf(x["w1_008"]), Lf(x["w3_008"])
                n1 = min(b1, Lf(x["w1_given_A"])) + 1.0          # 1 bit: which partner w1 uses
                n3 = Lf(x["w3_given_A"])
                num += (b1 + b3) - (n1 + n3) - r["perm_bits"]
                gain3 += b3 - n3
                gain1 += b1 - n1
                W += 2 * w(x["w1_given_A"])
                mc.append(x["matched_corr"]["median"])
                mp.append(x["matched_corr"]["p99"])
            res[kind] = {"pairs": len(t2), "net_bits_per_weight": num / W, "net_percent": 100 * num / W / 4.25,
                         "saving_before_permutation_cost_bits_per_weight": (gain1 + gain3) / W,
                         "w3_saving_bits_per_weight_of_w3": gain3 / (W / 2),
                         "permutation_bits_per_weight": sum(r["perm_bits"] for r in t2.values()) / W,
                         "matched_corr_median_of_pair_medians": float(np.median(mc)),
                         "matched_corr_median_of_pair_p99": float(np.median(mp)),
                         "matched_corr_pair_medians_range": [float(min(mc)), float(max(mc))]}
        res["identity_fraction_real"] = float(np.mean([r["real"]["identity_fraction"] for r in t2.values()]))
        res["affected"] = "w1 and w3 of expert B in each pair"
        out["T2"] = res
    # T3
    t3 = jl("t3.jsonl")
    if t3:
        res = {}
        for o in ("original", "nn_code_corr", "nn_scale_dist", "random"):
            num, W = 0.0, 0.0
            inc = {}
            sb = {"none": 0.0, "above": 0.0, "left": 0.0}
            for k, r in t3.items():
                x = r[o]
                L008 = Lf(base[k + ".w1"]) + Lf(base[k + ".w3"])
                Lc = Lf(x["w1"]) + Lf(x["w3"]) + (0.0 if o == "original" else r["perm_bits"])
                num += L008 - Lc
                ww = w(x["w1"]) + w(x["w3"])
                W += ww
                for p in ("R3", "R4", "R6", "R7"):
                    inc[p] = inc.get(p, 0.0) + (x["w1"]["increment"][p] * x["w1"]["raw_bits"] + x["w3"]["increment"][p] * x["w3"]["raw_bits"])
                for m in sb:
                    sb[m] += x["scale_bits"][m]["w1"] + x["scale_bits"][m]["w3"]
            res[o] = {"net_bits_per_weight": num / W, "net_percent": 100 * num / W / 4.25,
                      "column_increments_bits_per_weight": {p: v / W for p, v in inc.items()},
                      "scale_stream_bits_per_weight": {m: v / W for m, v in sb.items()}}
        res["permutation_bits_per_weight"] = next(iter(t3.values()))["perm_bits"] / (2 * w(next(iter(t3.values()))["original"]["w1"]))
        res["affected"] = "w1 and w3 of each expert, coded column by column (008: row by row)"
        out["T3"] = res
    t3r = jl("t3row.jsonl")
    if t3r:
        res = {}
        for o in ("nn_code_corr", "nn_scale_dist", "random"):
            num, W, codes, scale = 0.0, 0.0, 0.0, 0.0
            for k, r in t3r.items():
                x = r[o]
                L008 = Lf(base[k + ".w1"]) + Lf(base[k + ".w3"])
                Ln = x["w1_scale"]["L_with_scale_switch"] + x["w3_scale"]["L_with_scale_switch"] + r["perm_bits"]
                num += L008 - Ln
                codes += L008 - (Lf(x["w1"]) + Lf(x["w3"]))
                scale += sum(x[m + "_scale"]["order0"] - min(x[m + "_scale"]["order0"], x[m + "_scale"]["above"]) - 1.0
                             for m in ("w1", "w3"))
                W += w(x["w1"]) + w(x["w3"])
            perm = sum(r["perm_bits"] for r in t3r.values()) / W
            res[o] = {"net_bits_per_weight": num / W, "net_percent": 100 * num / W / 4.25,
                      "code_chain_change_bits_per_weight": codes / W, "scale_saving_bits_per_weight": scale / W,
                      "permutation_bits_per_weight": perm}
        res["affected"] = "w1 and w3 of each expert; codes row by row as in 008, scale stream with the row above as context"
        out["T3_row_major"] = res
    # P1 (neural probe) moved to experiments/011-limits-and-models (results/p1/p1.json)
    json.dump(out, open(R / "summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
