"""Spec 011 A3: is the MLP's information exchangeable inside a block (depends only on the set of codes of the
block) or tied to the order? Shuffle the codes inside every 32-block of a test matrix (scales unchanged), recompute
the chain on the shuffled matrix, and compare the gain of the best A1 MLP before / after. The cross matrix of w1 /
w2 is kept unshuffled. Writes results/a/a3_shuffle.json.

    PYTHONPATH=. python experiments/011-limits-and-models/a3_shuffle.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SD = HERE.parents[1] / "data" / "011_shuffle"
NAMES = ["L13.E127.w3", "L26.E336.w3", "L39.E70.w2", "L0.E191.w2", "L13.E127.w1", "L39.E70.w1"]


def build():
    import common as K
    sys.path.insert(0, str(K.ROOT / "experiments" / "010-transforms"))
    import p1b
    src = K.ROOT / "data" / "011"
    reg_all = {r["name"]: r for r in json.load(open(src / "registry.json"))}
    for d in ("codes", "test"):
        (SD / d).mkdir(parents=True, exist_ok=True)
    reg, info = [], []
    byid = {x["id"]: x for x in reg_all.values()}
    need = set()
    todo = list(NAMES)
    while todo:                                    # every matrix plus the chain of cross matrices it refers to
        n = todo.pop()
        if n in need:
            continue
        need.add(n)
        if reg_all[n]["cross"] is not None:
            todo.append(byid[reg_all[n]["cross"]]["name"])
    ids = {}
    for i, n in enumerate(sorted(need)):
        ids[n] = i
    for kind in ("orig", "shuf"):
        pass
    rng = np.random.default_rng(5)
    for n in sorted(need):
        r = dict(reg_all[n])
        r["id"] = ids[n]
        if r["cross"] is not None:
            cn = [x for x in reg_all.values() if x["id"] == reg_all[n]["cross"]][0]["name"]
            r["cross"] = ids[cn]
        reg.append(r)
        shutil.copy(src / "codes" / f"{n}.npz", SD / "codes" / f"{n}.npz")
    for n in NAMES:
        C, E = K.read(n)
        R, W = C.shape
        blk = C.reshape(R, W // 32, 32)
        perm = np.argsort(rng.random(blk.shape), axis=2)
        Cs = np.take_along_axis(blk, perm, axis=2).reshape(R, W)
        pn = K.partner_of(n)
        part = K.read(pn)[0] if pn else None
        ch = K.tx.chain(Cs, E, part)
        sym, p, Kk, side, scale, _ = p1b.chain_probs(Cs, E, p1b.active(ch), part)
        sn = n.replace(".w", ".s")                     # shuffled copy, same layer / role
        r = dict(reg_all[n])
        r["id"] = len(reg)
        r["name"] = sn
        r["cross"] = reg[[x["name"] for x in reg].index(n)]["cross"]
        reg.append(r)
        np.savez(SD / "codes" / f"{sn}.npz", sym=sym.reshape(R, W).astype(np.uint8), E=E)
        np.save(SD / "test" / f"{sn}.npy", np.log(p).astype(np.float32))
        info.append({"name": sn, "n": int(R * W), "chain_code_bits": float(-np.log2(p[np.arange(len(sym)), sym]).sum())})
        shutil.copy(src / "test" / f"{n}.npy", SD / "test" / f"{n}.npy")
    for line in open(src / "build_info.jsonl"):
        x = json.loads(line)
        if x["name"] in NAMES:
            info.append(x)
    json.dump(reg, open(SD / "registry.json", "w"))
    with open(SD / "build_info.jsonl", "w") as f:
        for x in info:
            f.write(json.dumps(x) + "\n")


def main():
    import common as K  # noqa: F401
    build()
    os.environ["D011"] = str(SD)
    import torch
    import amodel as A
    dev = "cuda:0"
    store = A.Store(dev)
    model = A.MLP(**A.SIZES["160k"]).to(dev)
    model.load_state_dict(torch.load(HERE.parents[1] / "data" / "011" / "models" / "a1_160k_1000000000.pt"))
    names = NAMES + [n.replace(".w", ".s") for n in NAMES]
    res = A.evaluate_test(model, store, names, dev)
    out = {}
    for n in NAMES:
        o, s = res[n], res[n.replace(".w", ".s")]
        out[n] = {"gain_original": (o["chain_bits_exact"] - o["bits"]) / o["n"],
                  "gain_block_shuffled": (s["chain_bits_exact"] - s["bits"]) / s["n"],
                  "chain_original": o["chain_bits_exact"] / o["n"], "chain_block_shuffled": s["chain_bits_exact"] / s["n"]}
    json.dump(out, open(HERE / "results" / "a" / "a3_shuffle.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
