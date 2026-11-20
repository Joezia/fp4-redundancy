"""Spec 011 A1: scaling curve of the correction MLP (logits = log p_chain + f), plus the checks of the pipeline.

    PYTHONPATH=. python experiments/011-limits-and-models/a1.py synth     # V-a / V-b through the GPU pipeline
    PYTHONPATH=. python experiments/011-limits-and-models/a1.py causal    # features never see undecoded codes
    PYTHONPATH=. python experiments/011-limits-and-models/a1.py run [size:N ...]

Results: experiments/011-limits-and-models/results/a/a1.jsonl (one line per run); models in data/011/models/.
Learning rates are fixed beforehand (3e-3 up to 640k parameters, 1e-3 above), never chosen on T1 / T2.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import json
import math
import time

import numpy as np

RES = HERE / "results" / "a"
RUNS = [("160k", 10 ** 9), ("40k", 10 ** 9), ("640k", 10 ** 9), ("2.56M", 10 ** 9), ("10M", 10 ** 9),
        ("10M", 10 ** 7), ("10M", 160 * 10 ** 6)]


def lr_for(size):
    return 3e-3 if size in ("40k", "160k", "640k") else 1e-3


def summarize(test, names):
    n = sum(test[k]["n"] for k in names)
    bits = sum(test[k]["bits"] for k in names)
    chain = sum(test[k]["chain_bits_exact"] for k in names)
    return {"n": n, "bits_per_code": bits / n, "chain_bits_per_code": chain / n, "gain": (chain - bits) / n}


def stage_run(specs):
    import torch
    import amodel as A
    import common as K
    dev = "cuda:0"
    RES.mkdir(parents=True, exist_ok=True)
    (A.D / "models").mkdir(exist_ok=True)
    done = set()
    if (RES / "a1.jsonl").exists():
        done = {(json.loads(l)["size"], json.loads(l)["N"]) for l in open(RES / "a1.jsonl")}
    store = A.Store(dev)
    all_w = 557171343360
    for size, N in specs:
        if (size, N) in done:
            continue
        torch.manual_seed(0)
        model = A.MLP(**A.SIZES[size]).to(dev)
        npar = A.n_params(model)
        print(f"== {size} ({npar} parameters), N = {N:.2e}, lr {lr_for(size)}  {time.strftime('%H:%M:%S')}", flush=True)
        info = A.train(model, store, N, dev, lr_for(size), time_limit=4 * 3600)
        path = A.D / "models" / f"a1_{size}_{N}.pt"
        torch.save(model.state_dict(), path)
        t0 = time.time()
        test = A.evaluate_test(model, store, K.T1 + K.T2, dev)
        res = {"size": size, "N": N, "params": npar, "cfg": A.SIZES[size], "model_file": str(path.relative_to(A.D.parent.parent)),
               "sha256": A.sha256(path), "train": info, "test_wall_s": time.time() - t0, "test": test}
        cost = 16.0 * npar
        for split, names in (("T1", K.T1), ("T2", K.T2)):
            s = summarize(test, names)
            s["net_gain_cost_over_test"] = s["gain"] - cost / s["n"]
            s["net_gain_cost_over_whole_model"] = s["gain"] - cost / all_w
            res[split] = s
        with open(RES / "a1.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"   T1 gain {res['T1']['gain']:.5f}  T2 gain {res['T2']['gain']:.5f}  ({res['T1']['bits_per_code']:.5f} vs chain "
              f"{res['T1']['chain_bits_per_code']:.5f})", flush=True)
        del model
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ checks

def stage_causal():
    """Changing a code and everything decoded after it must not change the features of that code."""
    import torch
    import amodel as A
    import common as K
    dev = "cuda:0"
    store = A.Store(dev, names=K.T1[:3])           # L0.E191.w1, w2, w3 (+ their cross matrices)
    rng = np.random.default_rng(3)
    report = []
    order = {"w3": 0, "w1": 1, "w2": 2}
    for name in K.T1[:3]:
        mid = store.ids[name]
        R, W = int(store.R[mid]), int(store.W[mid])
        for _ in range(20):
            t = int(rng.integers(0, R * W))
            m = torch.tensor([mid], device=dev)
            j = torch.tensor([t], device=dev)
            f0 = {k: v.clone() for k, v in store.features(m, j).items()}
            saved = store.buf.clone()
            # overwrite the code at t and every later code of this matrix, and every matrix decoded later
            Wp = W + A.LEFT + A.RIGHT
            P = store.buf[store.off[mid]: store.off[mid] + (R + A.TOP) * Wp].view(R + A.TOP, Wp)
            r, c = divmod(t, W)
            P[A.TOP + r, A.LEFT + c: A.LEFT + W] = torch.randint(0, 15, (W - c,), device=dev, dtype=torch.uint8)
            P[A.TOP + r + 1:, A.LEFT: A.LEFT + W] = torch.randint(0, 15, (R - r - 1, W), device=dev, dtype=torch.uint8)
            for other in K.T1[:3]:
                if order[other.split(".")[2]] > order[name.split(".")[2]]:
                    o = store.ids[other]
                    Ro, Wo = int(store.R[o]), int(store.W[o])
                    Wpo = Wo + A.LEFT + A.RIGHT
                    Q = store.buf[store.off[o]: store.off[o] + (Ro + A.TOP) * Wpo].view(Ro + A.TOP, Wpo)
                    Q[A.TOP:, A.LEFT: A.LEFT + Wo] = torch.randint(0, 15, (Ro, Wo), device=dev, dtype=torch.uint8)
            f1 = store.features(m, j)
            same = all(torch.equal(f0[k], f1[k]) for k in f0)
            store.buf.copy_(saved)
            report.append({"matrix": name, "t": t, "unchanged": bool(same)})
    ok = all(r["unchanged"] for r in report)
    json.dump({"pass": ok, "tests": report}, open(RES / "causal_check.json", "w"), indent=1)
    print("causal check pass:", ok, f"({len(report)} tests)")


def gen_matrix(rng, R, W, xor):
    """Magnitudes ~ Poisson(lambda(block scale)) capped at 7; signs random, except that (V-b) for c >= d2 the sign
    of a non-zero code equals sign(c - d1) XOR sign(c - d2) with probability rho (otherwise random). The sign of
    a zero code is set to 0 before it is used, so every sign the rule reads is visible to the decoder and the
    generator's exact code length (true_bits) is the floor for any model."""
    E = rng.integers(117, 123, size=(R, W // 32)).astype(np.uint8)
    lam = 0.8 + 0.4 * (np.repeat(E.astype(np.int64), 32, axis=1) - 117)
    mag = np.minimum(7, rng.poisson(lam))
    s = rng.integers(0, 2, size=(R, W))
    s[mag == 0] = 0
    if xor:
        d1, d2, rho = xor
        for c in range(d2, W):
            rule = s[:, c - d1] ^ s[:, c - d2]
            s[:, c] = np.where(rng.random(R) < rho, rule, rng.integers(0, 2, size=R))
            s[mag[:, c] == 0, c] = 0
    return mag + 8 * s, E


def true_bits(C, E, xor):
    """Exact code length of gen_matrix for the merged symbols: the magnitude from the block scale (Poisson capped
    at 7) and, for a non-zero code, its sign (1 bit; under the XOR rule -log2 of the rule's probability)."""
    R, W = C.shape
    S = np.repeat(E.astype(np.int64), 32, axis=1)
    lam = 0.8 + 0.4 * (S - 117)
    mag = C & 7
    k = np.minimum(mag, 7)
    logpmf = -lam + k * np.log(lam) - np.vectorize(math.lgamma)(k + 1)
    cdf6 = np.zeros_like(lam)
    for j in range(7):
        cdf6 += np.exp(-lam + j * np.log(lam) - math.lgamma(j + 1))
    lp = np.where(mag == 7, np.log(np.maximum(1 - cdf6, 1e-300)), logpmf)
    bits = float(-(lp / math.log(2)).sum())
    nz = mag > 0
    sign_bits = nz.astype(np.float64)
    if xor:
        d1, d2, rho = xor
        rho = rho + (1 - rho) / 2
        s = (C >> 3) & 1
        rule = np.zeros_like(s)
        rule[:, d2:] = s[:, d2 - d1:W - d1] ^ s[:, :W - d2]
        hit = s == rule
        after = np.zeros_like(nz)
        after[:, d2:] = True
        sign_bits = np.where(nz & after, np.where(hit, -math.log2(rho), -math.log2(1 - rho)), sign_bits)
    return bits + float(sign_bits.sum())


def stage_synth():
    """Synthetic V-a / V-b through the same GPU pipeline (separate data directory)."""
    sd = HERE.parents[1] / "data" / "011_synth"
    os.environ["D011"] = str(sd)
    import torch
    import amodel as A
    import common as K
    import p1b
    dev = "cuda:0"
    out = {}
    for kind, xor in (("V-a", None), ("V-b", (3, 20, 0.9))):
        for d in ("codes", "val", "test", "train_all", "models"):
            (sd / d).mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(41)
        reg, info, parts = [], [], []
        for i in range(8):
            C, E = gen_matrix(rng, 512, 4096, xor)
            ch = K.tx.chain(C, E)
            sym, p, Kk, side, scale, _ = p1b.chain_probs(C, E, p1b.active(ch))
            name = f"L0.E{i}.w3"
            split = "pool" if i < 6 else "val" if i == 6 else "T1"
            reg.append({"id": i, "name": name, "layer": 0, "expert": i, "role": "w3", "split": split, "cross": None,
                        "cross_kind": None})
            np.savez(sd / "codes" / f"{name}.npz", sym=sym.reshape(C.shape).astype(np.uint8), E=E)
            n = len(sym)
            info.append({"name": name, "n": n, "chain_code_bits": float(-np.log2(p[np.arange(n), sym]).sum())})
            if split == "pool":
                parts.append((i, np.arange(n, dtype=np.uint32), np.log(p).astype(np.float16)))
            elif split == "val":
                idx = np.sort(rng.choice(n, 100000, replace=False)).astype(np.uint32)
                np.savez(sd / "val" / f"{name}.npz", idx=idx, logp=np.log(p[idx]).astype(np.float32))
            else:
                np.save(sd / "test" / f"{name}.npy", np.log(p).astype(np.float32))
                Cte, Ete = C, E
        json.dump(reg, open(sd / "registry.json", "w"))
        with open(sd / "build_info.jsonl", "w") as f:
            for r in info:
                f.write(json.dumps(r) + "\n")
        mid = np.concatenate([np.full(len(x[1]), x[0], dtype=np.uint16) for x in parts])
        idx = np.concatenate([x[1] for x in parts])
        lp = np.concatenate([x[2] for x in parts])
        perm = rng.permutation(len(mid))
        np.save(sd / "train_all" / "mid.npy", mid[perm])
        np.save(sd / "train_all" / "idx.npy", idx[perm])
        np.save(sd / "train_all" / "logp.npy", lp[perm])
        store = A.Store(dev)
        torch.manual_seed(0)
        model = A.MLP(**A.SIZES["160k"]).to(dev)
        init = A.evaluate_test(model, store, ["L0.E7.w3"], dev)["L0.E7.w3"]
        tr = A.train(model, store, len(mid), dev, 3e-3, budget=60_000_000, eval_every=5_000_000, log=lambda s: None)
        te = A.evaluate_test(model, store, ["L0.E7.w3"], dev)["L0.E7.w3"]
        n = te["n"]
        rec = {"init_identity_diff": (init["bits"] - init["chain_bits_exact"]) / n,
               "gain": (te["chain_bits_exact"] - te["bits"]) / n, "best_val": tr["best_val"]}
        tb = true_bits(Cte, Ete, xor)
        rec["true_bits_per_code"] = tb / n
        rec["theory"] = (te["chain_bits_exact"] - tb) / n      # all a perfect model could gain over the chain
        out[kind] = rec
        print(kind, rec, flush=True)
        del store, model
        torch.cuda.empty_cache()
    out["checks"] = {"init_identity": bool(max(abs(out[k]["init_identity_diff"]) for k in ("V-a", "V-b")) < 1e-4),
                     "V-a_about_0": bool(-0.01 < out["V-a"]["gain"] < 0.005),
                     "V-b_recovers": bool(0.8 * out["V-b"]["theory"] < out["V-b"]["gain"] <= out["V-b"]["theory"] + 1e-3)}
    out["all_pass"] = all(out["checks"].values())
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES / "synthetic_mlp.json", "w"), indent=1)
    print(out["checks"], out["all_pass"])


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "run":
        specs = [(a.split(":")[0], int(float(a.split(":")[1]))) for a in sys.argv[2:]] or RUNS
        stage_run(specs)
    else:
        {"synth": stage_synth, "causal": stage_causal}[st]()
