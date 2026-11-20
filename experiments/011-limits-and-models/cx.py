"""Spec 011 line C: shared GPU code -- the A1 store and correction MLP extended with extra inputs.

StoreX(A.Store) adds to every feature dict the coordinates of the code:
  h     : hidden dim (residual stream index): column of w1 / w3, row of w2
  n     : neuron index: row of w1 / w3, column of w2
  L     : layer (0..42), role (0 = w1, 1 = w2, 2 = w3)
and, for each registered "extra" function, more tensors (summaries of C3 / C4); the extras only read what is
decoded before the code (or side information that is billed). With no extras, the A1 features are unchanged.
MLPx(A.MLP) concatenates learned embeddings of chosen index features and a vector of continuous features to the
A1 input; the output layer still starts at 0 (the model starts as the chain).
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import amodel as A  # noqa: E402

ALL_W = 557171343360


class StoreX(A.Store):
    def __init__(self, dev, names=None, extras=()):
        super().__init__(dev, names)
        self.g_layer_i = self.g_layer.long()
        self.extras = list(extras)

    def features(self, mid, idx, mask=None):
        f = super().features(mid, idx, mask)
        W = self.g_W[mid]
        r, c = idx // W, idx % W
        role = self.g_role[mid]
        f["h"] = torch.where(role == 1, r, c)
        f["n"] = torch.where(role == 1, c, r)
        f["L"] = self.g_layer_i[mid]
        f["role"] = role
        for ex in self.extras:
            ex(self, mid, idx, r, c, f)
        return f


class MLPx(A.MLP):
    """embs: {name: (index function f -> LongTensor, table size, dim)}; n_cont: width of f['cont'] (0 = none)."""

    def __init__(self, embs=None, n_cont=0, **cfg):
        super().__init__(**cfg)
        self.embs = dict(embs or {})
        self.tables = torch.nn.ModuleDict({k: torch.nn.Embedding(v[1], v[2]) for k, v in self.embs.items()})
        for t in self.tables.values():
            torch.nn.init.normal_(t.weight, std=0.01)
        self.n_cont = n_cont
        extra = sum(v[2] for v in self.embs.values()) + n_cont
        first = self.body[0]
        new = torch.nn.Linear(first.in_features + extra, first.out_features)
        self.body[0] = new
        self.cfg.update({"embs": {k: [v[1], v[2]] for k, v in self.embs.items()}, "n_cont": n_cont})

    def forward(self, f, logp):
        toks = torch.cat([f["prev"], f["above"], f["cross"]], 1).long()
        parts = [self.emb(toks).flatten(1), self.semb(f["sc"].long()).flatten(1), self.pemb(f["pos"].long()),
                 f["num"], logp.clamp(min=-30.0)]
        for k, v in self.embs.items():
            parts.append(self.tables[k](v[0](f)))
        if self.n_cont:
            parts.append(f["cont"])
        return logp + self.out(self.body(torch.cat(parts, 1))).float()


def summarize(test, names):
    n = sum(test[k]["n"] for k in names)
    bits = sum(test[k]["bits"] for k in names)
    chain = sum(test[k]["chain_bits_exact"] for k in names)
    return {"n": n, "bits_per_code": bits / n, "chain_bits_per_code": chain / n, "gain": (chain - bits) / n}


def run(model, store, N, dev, lr, out_path, record, names_test, extra_side_bits=None, log=print, budget=None):
    """Train, test on names_test, append one JSON line. extra_side_bits: {matrix name: billed side bits}."""
    t0 = time.time()
    info = A.train(model, store, N, dev, lr, budget=budget, log=log)
    mdir = A.D / "models"
    mdir.mkdir(exist_ok=True)
    path = mdir / f"{record['tag']}.pt"
    torch.save(model.state_dict(), path)
    test = A.evaluate_test(model, store, names_test, dev)
    if extra_side_bits:
        for k in test:
            test[k]["side_bits"] = float(extra_side_bits.get(k, 0.0))
            test[k]["bits"] += test[k]["side_bits"]
    npar = A.n_params(model)
    res = {**record, "params": npar, "N": N, "lr": lr, "model_file": str(path), "sha256": A.sha256(path), "train": info,
           "test": test, "wall_s": time.time() - t0}
    import common as K
    for split, ns in (("T1", K.T1), ("T2", K.T2)):
        ns = [x for x in ns if x in test]
        if not ns:
            continue
        s = summarize(test, ns)
        s["net_gain_cost_over_test"] = s["gain"] - 16.0 * npar / s["n"]
        s["net_gain_cost_over_whole_model"] = s["gain"] - 16.0 * npar / ALL_W
        res[split] = s
    with open(out_path, "a") as f:
        f.write(json.dumps(res) + "\n")
    log(f"   {record['tag']}: " + "  ".join(f"{s} gain {res[s]['gain']:.5f}" for s in ("T1", "T2") if s in res))
    return res


# ------------------------------------------------------------------ synthetic data directories (as a1.stage_synth)

def build_synth(sd, mats, seed=0):
    """mats: list of dicts {name, layer, expert, role, C (raw codes), E, split, cross (id or None), cross_kind}.
    Writes registry, codes, chain probabilities (val / test / shuffled training positions) into sd."""
    import common as K
    import p1b
    for d in ("codes", "val", "test", "train_all", "models"):
        (sd / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    reg, info, parts = [], [], []
    byid = {}
    for i, m in enumerate(mats):
        byid[m["name"]] = i
    for i, m in enumerate(mats):
        C, E = m["C"], m["E"]
        part = None
        if m.get("cross") is not None and m["role"] == "w1":
            part = mats[byid[m["cross"]]]["C"]
        ch = K.tx.chain(C, E, partner=part)
        flags = p1b.active(ch) - {"R6"}          # p1b.chain_probs cannot code with R6 (synthetic data only)
        sym, p, Kk, side, scale, _ = p1b.chain_probs(C, E, flags, part)
        reg.append({"id": i, "name": m["name"], "layer": m["layer"], "expert": m["expert"], "role": m["role"],
                    "split": m["split"], "cross": None if m.get("cross") is None else byid[m["cross"]],
                    "cross_kind": m.get("cross_kind")})
        np.savez(sd / "codes" / f"{m['name']}.npz", sym=sym.reshape(C.shape).astype(np.uint8), E=E)
        n = len(sym)
        info.append({"name": m["name"], "n": n, "chain_code_bits": float(-np.log2(p[np.arange(n), sym]).sum())})
        if m["split"] == "pool":
            parts.append((i, np.arange(n, dtype=np.uint32), np.log(p).astype(np.float16)))
        elif m["split"] == "val":
            idx = np.sort(rng.choice(n, min(n, 100000), replace=False)).astype(np.uint32)
            np.savez(sd / "val" / f"{m['name']}.npz", idx=idx, logp=np.log(p[idx]).astype(np.float32))
        else:
            np.save(sd / "test" / f"{m['name']}.npy", np.log(p).astype(np.float32))
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
    return {m["name"]: info[i]["chain_code_bits"] for i, m in enumerate(mats)}


def poisson_bits(C, lam):
    """exact code length of magnitudes ~ Poisson(lam) capped at 7 plus 1 sign bit per non-zero (merged symbols)."""
    mag = C & 7
    k = mag.astype(np.float64)
    lg = np.vectorize(math.lgamma)
    logpmf = -lam + k * np.log(lam) - lg(k + 1)
    cdf6 = sum(np.exp(-lam + j * np.log(lam) - math.lgamma(j + 1)) for j in range(7))
    lp = np.where(mag == 7, np.log(np.maximum(1 - cdf6, 1e-300)), logpmf)
    return float(-(lp / math.log(2)).sum() + (mag > 0).sum())


def poisson_codes(rng, lam):
    mag = np.minimum(7, rng.poisson(lam))
    s = rng.integers(0, 2, size=mag.shape)
    s[mag == 0] = 0
    return (mag + 8 * s).astype(np.int64)
