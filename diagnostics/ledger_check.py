"""Spec 002 section 4: synthetic checks U1-U4 (no model data).

Every dataset goes through ledger.ledger_matrix / shared_lengths / ledger_rows, the same code path
as the model matrices. Usage: python -m diagnostics.ledger_check [--out .../validation.json]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import est, ledger
from .synth import R_SYN, W_SYN, gen_T1, quantize

B = 32
SEED = 20261112


def chk(name, value, criterion, ok):
    return {"check": name, "value": value, "criterion": criterion, "pass": bool(ok)}


def gen_U3(rng):
    y = rng.standard_normal((R_SYN, W_SYN // B, B)) * 1.68
    amax = np.abs(y).max(axis=2)
    e = np.floor(np.log2(amax)).astype(np.int64) - 2
    C = quantize((y / np.exp2(e)[:, :, None]).reshape(R_SYN, W_SYN))
    return C, (e + 127).astype(np.uint8)


def run():
    out = {"shape": [R_SYN, W_SYN], "B": B, "seed": SEED, "checks": {}}
    # U1
    c = np.arange(16, dtype=np.uint8)
    out["checks"]["U1"] = [chk("v(C') == v(C) for all 16 codewords",
                               [float(x) for x in ledger.E2M1[ledger.merge_zero(c)]],
                               "elementwise equal (+0 == -0)", ledger.value_check())]

    # U2: 001 T1 (15 codewords iid uniform, constant scale)
    rng = np.random.default_rng(SEED + 21)
    C, _ = gen_T1(rng)
    E = np.full((R_SYN, W_SYN // B), 127, dtype=np.uint8)
    r = ledger.ledger_matrix(C, E, B, rng)
    rows = ledger.ledger_rows(r, None)
    expect = (4 - math.log2(15)) + 8 / B
    d = r["ent"]["H(C)"] - r["ent"]["H(Cp)"]
    out["checks"]["U2"] = [
        chk("|H(C) - H(C')|", abs(d), "< 1e-9", abs(d) < 1e-9),
        chk("I(C'; E)", r["I_Cp_E"], "== 0", r["I_Cp_E"] == 0.0),
        chk("M5 row 4 saving - ((4 - log2 15) + 8/B)", rows[4]["saving"] - expect, "|.| <= 1e-4",
            abs(rows[4]["saving"] - expect) <= 1e-4),
    ]
    out["U2_detail"] = {"row4_saving": rows[4]["saving"], "expected": expect,
                        "H(C')": r["ent"]["H(Cp)"], "row4_table_costs":
                        rows[4]["cw_tab"] + rows[4]["sc_tab"] / B}

    # U3: iid N(0, 1.68^2), e = floor(log2 amax) - 2 per block
    rng = np.random.default_rng(SEED + 31)
    C, E = gen_U3(rng)
    r = ledger.ledger_matrix(C, E, B, rng)
    rows = ledger.ledger_rows(r, None)
    e = r["ent"]
    id1 = (e["H(C)"] - e["H(Cp)"]) - r["P_m0"] * r["H_s_m0"]
    id2 = (e["H(Cp)"] - e["H(Cp|E)"]) - r["I_Cp_E"]
    s0_i = [x["I(Cp;E)"] for x in r["s0"]]
    s0_net = [x["inc_net"] for x in r["s0"]]
    sop = ledger.sum_of_parts(r)
    diff = rows[4]["saving"] - sop
    out["checks"]["U3"] = [
        chk("H(C)-H(C') - P(m=0)H(s|m=0)", id1, "|.| < 1e-9", abs(id1) < 1e-9),
        chk("H(C')-H(C'|E) - I(C';E)", id2, "|.| < 1e-9", abs(id2) < 1e-9),
        chk("I(C'; E)", r["I_Cp_E"], "> 0.05", r["I_Cp_E"] > 0.05),
        chk("I(C'; E) on S0 (3 reps)", s0_i, "each < 1e-4", max(s0_i) < 1e-4),
        chk("M2 incremental net on S0 (3 reps)", s0_net, "each <= 0", max(s0_net) <= 0),
        chk("M5 row 4 saving - 001 item sum", diff, "|.| < 1e-3", abs(diff) < 1e-3),
    ]
    out["U3_detail"] = {"K_E": r["K_E"], "E_values": [p["E"] for p in r["per_scale"]],
                        "P_m0": r["P_m0"], "H_s_m0": r["H_s_m0"], "I_C_E": r["I_C_E"],
                        "I_Cp_E": r["I_Cp_E"], "inc_net_real": r["m2"]["inc_net"],
                        "row4_saving": rows[4]["saving"], "sum_of_parts": sop,
                        "S0_analytic_bias_I": est.analytic_bias((r["K_E"] - 1) * 14, r["N"])}

    # U4: U3 copied into 3 "matrices"; scheme b vs scheme a
    res = [r, r, r]
    sh = ledger.shared_lengths(res)
    dd = [x["cw_bits"] - e["H(Cp|E)"] for x in sh]
    out["checks"]["U4"] = [chk("scheme b cross-entropy - H(C'|E) (3 copies)", dd, "each in [0, 1e-3]",
                               all(0 <= x <= 1e-3 for x in dd))]
    sha = ledger.shared_lengths(res, key="abs")
    da = [x["cw_bits"] - e["H(Cp|E)"] for x in sha]
    out["checks"]["U4"].append(chk("supplement: absolute-E shared table - H(C'|E) (3 copies)", da,
                                   "each in [0, 1e-3]", all(0 <= x <= 1e-3 for x in da)))
    out["all_pass"] = all(c["pass"] for v in out["checks"].values() for c in v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/002-ledger/validation.json")
    args = ap.parse_args()
    out = run()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out["checks"].items():
        for c in v:
            print(k, "PASS" if c["pass"] else "FAIL", c["check"], c["value"])
    print("all_pass", out["all_pass"])


if __name__ == "__main__":
    main()
