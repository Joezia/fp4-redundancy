"""Spec 011 line C2: learned coordinate embeddings added to the A1 correction MLP (160k configuration).

    PYTHONPATH=. python experiments/011-limits-and-models/c2_coord.py synth
    PYTHONPATH=. python experiments/011-limits-and-models/c2_coord.py run [config ...]

Configurations (embedding width 8; the tables are model parameters, billed at 16 bit each):
  base               : the A1 160k model, retrained here as the reference (same seed, data, learning rate)
  hidden@layer       : one vector per (layer, hidden dim)           43 x 5120
  hidden             : one vector per hidden dim, shared by layers   5120
  hidden@layer+role  : one vector per (layer, role, hidden dim)      43 x 3 x 5120
  neuron@layer       : one vector per (layer, neuron index)          43 x 2304 -- negative control: test experts'
                       neurons have no counterpart in the training experts (the neuron order is arbitrary), so
                       this should gain about nothing; a clear gain would point to a leak.
An expert-id embedding cannot be tested: the test experts are not in the training set (C4's per-matrix latent
plays that role).
Training: the first 1e9 shuffled positions (as A1, not A3's 1.6e8: each (layer, hidden dim) entry sees only
about 4,500 training codes even then), learning rate 3e-3 (a1.lr_for), seed 0.
Results: results/c/c2_coord.jsonl, synthetic check results/c/synthetic_c2.json.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import json
import time

import numpy as np

RES = HERE / "results" / "c"
N_REAL = 10 ** 9
CONFIGS = {
    "base": {},
    "hidden@layer": {"hl": (lambda f: f["L"] * 5120 + f["h"], 43 * 5120, 8)},
    "hidden": {"h": (lambda f: f["h"], 5120, 8)},
    "hidden@layer+role": {"hlr": (lambda f: (f["L"] * 3 + f["role"]) * 5120 + f["h"], 43 * 3 * 5120, 8)},
    "neuron@layer": {"nl": (lambda f: f["L"] * 2304 + f["n"], 43 * 2304, 8)},
}


def make_model(cfg, dev):
    import torch
    import amodel as A
    import cx
    torch.manual_seed(0)
    return cx.MLPx(embs=CONFIGS[cfg], **A.SIZES["160k"]).to(dev)


def stage_run(names):
    import torch
    import amodel as A
    import common as K
    import cx
    dev = "cuda:0"
    out = RES / "c2_coord.jsonl"
    done = {json.loads(l)["config"] for l in open(out)} if out.exists() else set()
    store = cx.StoreX(dev)
    for cfg in names:
        if cfg in done:
            continue
        print(f"== {cfg}  {time.strftime('%H:%M:%S')}", flush=True)
        model = make_model(cfg, dev)
        cx.run(model, store, N_REAL, dev, 3e-3, out, {"tag": f"c2_{cfg.replace('@', '_at_').replace('+', '_')}", "config": cfg},
               K.T1 + K.T2, log=lambda s: print(s, flush=True))
        del model
        torch.cuda.empty_cache()
    print("C2_DONE", flush=True)


def stage_synth():
    """Columns (hidden dims) with a shared boost: lambda x 2 on 10% of the hidden dims, the same dims in every
    expert of the layer (V-b); no boost (V-a). Six training experts, one validation, one test (512 x 4096, role w3).
    The chain cannot know which column is boosted (only the 8 codes above hint at it); the exact code length of the
    generator is the floor. Checks: V-a every configuration about 0; V-b hidden@layer recovers >= 80% of
    (chain - floor); neuron@layer gains no more than base + 0.003."""
    sd = HERE.parents[1] / "data" / "011_synth_c2"
    os.environ["D011"] = str(sd)
    import torch
    import amodel as A
    import cx
    dev = "cuda:0"
    out = {}
    for kind in ("V-a", "V-b"):
        rng = np.random.default_rng(51)
        R, W = 512, 4096
        boost = np.where(rng.random(W) < 0.1, 2.0, 1.0) if kind == "V-b" else np.ones(W)
        mats, lams = [], {}
        for i in range(8):
            E = rng.integers(117, 123, size=(R, W // 32)).astype(np.uint8)
            lam = (0.8 + 0.4 * (np.repeat(E.astype(np.float64), 32, axis=1) - 117)) * boost[None, :]
            C = cx.poisson_codes(rng, lam)
            name = f"L0.E{i}.w3"
            mats.append({"name": name, "layer": 0, "expert": i, "role": "w3", "C": C, "E": E,
                         "split": "pool" if i < 6 else "val" if i == 6 else "T1"})
            lams[name] = (C, lam)
        chain = cx.build_synth(sd, mats, seed=1)
        A.D = sd
        store = cx.StoreX(dev)
        Ct, lt = lams["L0.E7.w3"]
        floor = cx.poisson_bits(Ct, lt)
        n = Ct.size
        rec = {"theory": (chain["L0.E7.w3"] - floor) / n}
        for cfg in ("base", "hidden@layer", "neuron@layer"):
            model = make_model(cfg, dev)
            A.train(model, store, 6 * R * W, dev, 3e-3, budget=60_000_000, eval_every=5_000_000, log=lambda s: None)
            te = A.evaluate_test(model, store, ["L0.E7.w3"], dev)["L0.E7.w3"]
            rec[cfg] = (te["chain_bits_exact"] - te["bits"]) / n
            del model
            torch.cuda.empty_cache()
        out[kind] = rec
        print(kind, rec, flush=True)
        del store
    out["checks"] = {"V-a_about_0": bool(all(-0.01 < out["V-a"][c] < 0.005 for c in ("base", "hidden@layer", "neuron@layer"))),
                     "V-b_hidden_recovers": bool(0.8 * out["V-b"]["theory"] < out["V-b"]["hidden@layer"] <= out["V-b"]["theory"] + 1e-3),
                     "V-b_neuron_control": bool(out["V-b"]["neuron@layer"] <= out["V-b"]["base"] + 0.003)}
    out["all_pass"] = all(out["checks"].values())
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES / "synthetic_c2.json", "w"), indent=1)
    print(out["checks"], out["all_pass"], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "synth":
        stage_synth()
    else:
        stage_run(sys.argv[2:] or list(CONFIGS))
