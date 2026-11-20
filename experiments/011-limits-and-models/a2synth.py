"""Spec 011 A2 synthetic checks (V-a / V-b) for the row Transformer, through the same data path.

V-a: magnitudes from the block scale only (the chain sees it) -> gain about 0.
V-b: in addition the sign of a non-zero code = sign(c - 50) XOR sign(c - 100) with probability 0.9 (otherwise
     random) -- far outside the chain's context and outside the MLP's 32-code window; the theory is the chain's
     code length minus the generator's exact code length (a1.true_bits).
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SD = HERE.parents[1] / "data" / "011_synth2"


def build(xor, rng):
    import common as K
    import a1
    import p1b
    for d in ("codes", "rows", "test"):
        (SD / d).mkdir(parents=True, exist_ok=True)
    reg, info = [], []
    keep = None
    for i in range(10):
        C, E = a1.gen_matrix(rng, 256, 4096, xor)
        ch = K.tx.chain(C, E)
        sym, p, Kk, side, scale, _ = p1b.chain_probs(C, E, p1b.active(ch))
        name = f"L0.E{i}.w3"
        split = "pool" if i < 8 else "val" if i == 8 else "T1"
        reg.append({"id": i, "name": name, "layer": 0, "expert": i, "role": "w3", "split": split, "cross": None, "cross_kind": None})
        np.savez(SD / "codes" / f"{name}.npz", sym=sym.reshape(C.shape).astype(np.uint8), E=E)
        n = len(sym)
        info.append({"name": name, "n": n, "chain_code_bits": float(-np.log2(p[np.arange(n), sym]).sum())})
        if split == "T1":
            np.save(SD / "test" / f"{name}.npy", np.log(p).astype(np.float32))
            keep = (C, E)
        else:
            np.save(SD / "rows" / f"{name}.npy", np.log(p).astype(np.float16))
    json.dump(reg, open(SD / "registry.json", "w"))
    with open(SD / "build_info.jsonl", "w") as f:
        for r in info:
            f.write(json.dumps(r) + "\n")
    return keep


def main():
    os.environ["D011"] = str(SD)
    import torch
    import amodel as A
    import a1
    import a2
    dev = "cuda:0"
    out = {}
    for kind, xor in (("V-a", None), ("V-b", (50, 100, 0.9))):
        rng = np.random.default_rng(43)
        C, E = build(xor, rng)
        store = A.Store(dev)
        names = [r["name"] for r in store.reg]
        data = a2.Rows(store, [n for n in names if store.reg[store.ids[n]]["split"] == "pool"], SD / "rows")
        val = a2.Rows(store, [n for n in names if store.reg[store.ids[n]]["split"] == "val"], SD / "rows")
        val_sel = {W: lst[:128] for W, lst in val.byW.items()}
        torch.manual_seed(0)
        model = a2.RowModel(**a2.SIZES["1M"]).to(dev)
        test0 = a2.evaluate_test(model, store, SD, ["L0.E9.w3"], False, dev)["L0.E9.w3"]
        tr = a2.train(model, store, data, val, val_sel, 120_000_000, 1e-3, False, dev, log=lambda s: None,
                      eval_every=10_000_000, patience=4)
        te = a2.evaluate_test(model, store, SD, ["L0.E9.w3"], False, dev)["L0.E9.w3"]
        n = te["n"]
        tb = a1.true_bits(C, E, xor)
        out[kind] = {"init_identity_diff": (test0["bits"] - test0["chain_bits_exact"]) / n,
                     "gain": (te["chain_bits_exact"] - te["bits"]) / n, "theory": (te["chain_bits_exact"] - tb) / n,
                     "best_val": tr["best_val"], "seen": tr["seen"]}
        print(kind, out[kind], flush=True)
        del store, model
        torch.cuda.empty_cache()
    out["checks"] = {"init_identity": bool(max(abs(out[k]["init_identity_diff"]) for k in ("V-a", "V-b")) < 1e-4),
                     "V-a_about_0": bool(-0.01 < out["V-a"]["gain"] < 0.005),
                     "V-b_recovers": bool(0.8 * out["V-b"]["theory"] < out["V-b"]["gain"] <= out["V-b"]["theory"] + 1e-3)}
    out["all_pass"] = all(out["checks"].values())
    json.dump(out, open(HERE / "results" / "a" / "synthetic_a2.json", "w"), indent=1)
    print(out["checks"], out["all_pass"])
