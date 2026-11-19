"""Spec 011 A3 (first round, on the A1 MLP): feature-group ablation and context-length scan.

    PYTHONPATH=. python experiments/011-limits-and-models/a3.py <size> <N> [config ...]

Every configuration is retrained from scratch with the same size, training set (first N shuffled positions),
learning rate and budget as the reference run, with the masked feature group set to padding / 0 in training
and in testing. Results: results/a/a3_mlp.jsonl.
Groups (amodel.apply_mask): inblock = earlier codes of the same 32-block; crossblock = codes of earlier blocks
(within the 32-code window); above; partner; scale; position. Context scan: keep only the L previous codes.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import json
import time

CONFIGS = {"full": [], "-inblock": ["inblock"], "-crossblock": ["crossblock"], "-above": ["above"],
           "-partner": ["partner"], "-scale": ["scale"], "-position": ["position"],
           "ctx0": [("ctx", 0)], "ctx2": [("ctx", 2)], "ctx4": [("ctx", 4)], "ctx8": [("ctx", 8)], "ctx16": [("ctx", 16)],
           "inblock_only": ["crossblock", "above", "partner"]}


def main():
    import torch
    import amodel as A
    import common as K
    import a1
    size, N = sys.argv[1], int(float(sys.argv[2]))
    names = sys.argv[3:] or list(CONFIGS)
    dev = "cuda:0"
    out = HERE / "results" / "a" / "a3_mlp.jsonl"
    done = {(json.loads(l)["size"], json.loads(l)["N"], json.loads(l)["config"]) for l in open(out)} if out.exists() else set()
    store = A.Store(dev)
    for name in names:
        if (size, N, name) in done:
            continue
        mask = CONFIGS[name]
        torch.manual_seed(0)
        model = A.MLP(**A.SIZES[size]).to(dev)
        print(f"== {name} ({size}, N {N:.1e})  {time.strftime('%H:%M:%S')}", flush=True)
        info = A.train(model, store, N, dev, a1.lr_for(size), mask=mask, log=lambda s: print(s, flush=True))
        test = A.evaluate_test(model, store, K.T1 + K.T2, dev, mask=mask)
        res = {"size": size, "N": N, "config": name, "mask": [list(m) if isinstance(m, tuple) else m for m in mask],
               "train": info, "test": test, "T1": a1.summarize(test, K.T1), "T2": a1.summarize(test, K.T2)}
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"   T1 gain {res['T1']['gain']:.5f}  T2 gain {res['T2']['gain']:.5f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
