"""Spec 008 Part 1: sample actually completed, per-matrix key numbers, layer table, comparison with 007.

    python experiments/008-scale-media/analyze_fp4.py

Reads results/fp4/{A,B}/entries.jsonl and the 007 result files; writes results/fp4/part1_summary.json.
Key number per matrix (all from the indicator results, no new computation):
  R1   saving = raw bits - H(code) (bit per code)
  R2   I(code; block scale) (bit per code), net of nothing (the indicator's mutual information)
  R3   I(C_j; C_j+1) (bit), lag 1
  R4   autocorrelation of |v| at lag 1
  R6   repeated fraction of length-8 windows minus the whole-shuffle baseline
  R7   median |correlation| of corresponding W1 and W3 rows (w1 only)
  R10  bits per code saved by merging +0 and -0
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

E = Path("experiments/008-scale-media")
R = E / "results" / "fp4"


def load(path):
    rows = {}
    if not path.exists():
        return rows
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["entry"] in rows:
            rows[r["entry"]]["results"].update(r["results"])
        else:
            rows[r["entry"]] = r
    return rows


def key_numbers(res):
    k = {}
    if "R1" in res and res["R1"].get("saving") is not None:
        k["R1"] = res["R1"]["saving"]
    try:
        k["R2"] = res["R2"]["groups"]["scale"]["mi"]
    except (KeyError, TypeError):
        pass
    try:
        k["R3"] = res["R3"]["mi"][0]
    except (KeyError, TypeError, IndexError):
        pass
    try:
        k["R4"] = res["R4"]["abs"][0]
    except (KeyError, TypeError, IndexError):
        pass
    try:
        k["R6"] = res["R6"]["frac"][0] - res["R6"]["base_mean"][0]
    except (KeyError, TypeError, IndexError):
        pass
    p = res.get("R7a_partner") or {}
    if p.get("applicable", True) is not False and "real" in p:
        k["R7"] = p["real"]["median"]
    try:
        k["R10"] = res["R10"]["rows"][0]["bits_per_symbol"]
    except (KeyError, TypeError, IndexError):
        pass
    return k


def main():
    A = load(R / "A" / "entries.jsonl")
    B = load(R / "B" / "entries.jsonl")
    order = json.load(open(R / "expert_order.json"))
    layers = sorted(int(x) for x in order)
    # completed experts (all three matrices present)
    def done_rounds(rows, plan):
        by_round = defaultdict(lambda: defaultdict(bool))
        for t in json.load(open(plan))["tasks"]:
            by_round[t["round"]][t["layer"]] = all(e in rows for e in t["entries"])
        return by_round
    bB = done_rounds(B, R / "B" / "plan.json")
    bA = done_rounds(A, R / "A" / "plan.json")
    full_B = [r for r in sorted(bB) if all(bB[r].values())]
    eb = 0
    while eb + 1 in full_B:
        eb += 1
    # A plan holds rounds 11+ (expert positions 11.. of each layer's fixed random permutation), so the
    # A sample is the complete B rounds plus every complete A round: still a uniform random sample of
    # experts per layer, without replacement.
    full_A = [r for r in sorted(bA) if all(bA[r].values())]
    ea = eb + len(full_A)
    nB = sum(1 for r in bB for L in bB[r] if bB[r][L])
    nA_extra = sum(1 for r in bA for L in bA[r] if bA[r][L])
    out = {"E_B_complete_rounds": eb, "E_A_complete_rounds": ea,
           "B_experts_done": nB, "A_extra_experts_done": nA_extra,
           "B_matrices": len(B), "A_extra_matrices": len(A),
           "partial_rounds_B": {str(r): sum(bB[r].values()) for r in bB if 0 < sum(bB[r].values()) < len(bB[r])},
           "partial_rounds_A": {str(r): sum(bA[r].values()) for r in bA if 0 < sum(bA[r].values()) < len(bA[r])}}
    # only complete rounds enter the statistics, so every layer has the same number of experts
    keepB = {e for t in json.load(open(R / "B" / "plan.json"))["tasks"] if t["round"] <= eb for e in t["entries"]}
    keepA = {e for t in json.load(open(R / "A" / "plan.json"))["tasks"] if t["round"] in full_A for e in t["entries"]}
    allrows = {**{n: r for n, r in A.items() if n in keepA}, **{n: r for n, r in B.items() if n in keepB}}
    out["full_A_rounds"] = full_A
    out["matrices_in_statistics"] = {"B": len(keepB), "A_extra": len(keepA)}
    per = {n: {"layer": r["attrs"]["layer"], "role": r["attrs"]["role"], **key_numbers(r["results"])}
           for n, r in allrows.items()}
    groups = [("0-9", range(0, 10)), ("10-19", range(10, 20)), ("20-29", range(20, 30)), ("30-39", range(30, 40)),
              ("MTP 40-42", range(40, 43))]
    keys = ["R1", "R2", "R3", "R4", "R6", "R7", "R10"]
    table = {}
    for g, rng in groups:
        sel = [v for v in per.values() if v["layer"] in rng]
        table[g] = {k: (float(np.median([v[k] for v in sel if k in v])) if any(k in v for v in sel) else None)
                    for k in keys}
        table[g]["n_matrices"] = len(sel)
    allm = {k: float(np.median([v[k] for v in per.values() if k in v])) for k in keys if any(k in v for v in per.values())}
    # 007: same numbers on its 36 matrices
    d7 = {}
    for fig in ["R1", "R2", "R3", "R4", "R6", "R7a_partner", "R10"]:
        rec = json.load(open(f"experiments/007-figures-v2/results/data/{fig}.json"))
        for n, res in rec["entries"].items():
            d7.setdefault(n, {})[fig] = res
    k7 = [key_numbers(v) for v in d7.values()]
    m7 = {k: float(np.median([v[k] for v in k7 if k in v])) for k in keys}
    # same numbers restricted to the matrices that have them, with counts
    counts = {k: sum(1 for v in per.values() if k in v) for k in keys}
    # chain increments (bit per weight, default order) by layer group, from the four chain shards
    chain = {}
    for d in ("increment_B", "increment_B_s1", "increment_B_s2", "increment_B_s3"):
        f = R / d / "items.jsonl"
        if f.exists():
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chain[r["item"]] = r
    ctab = {}
    for g, rng in groups:
        sel = [r for n, r in chain.items() if int(n.split(".")[0][1:]) in rng]
        if not sel:
            continue
        w = np.array([r["raw_bits"] for r in sel])
        inc = {k: float(np.average([r["chains"]["default"]["increment"][k] for r in sel], weights=w)) * 4.25
               for k in sel[0]["chains"]["default"]["increment"]}
        inc["total"] = float(np.average([r["chains"]["default"]["total"] for r in sel], weights=w)) * 4.25
        inc["n_matrices"] = len(sel)
        ctab[g] = inc
    out["chain_layer_table_bits_per_weight"] = ctab
    out.update({"layer_table": table, "median_all": allm, "median_007_36": m7, "counts": counts,
                "per_matrix": per})
    json.dump(out, open(R / "part1_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_matrix"}, indent=1))


if __name__ == "__main__":
    main()
