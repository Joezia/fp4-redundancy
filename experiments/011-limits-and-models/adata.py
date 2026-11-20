"""Spec 011 line A: data generation (not committed; regenerate with this script).

    PYTHONPATH=. python experiments/011-limits-and-models/adata.py chains     # 008 chain of every matrix used
    PYTHONPATH=. python experiments/011-limits-and-models/adata.py build      # per-matrix codes and chain probabilities
    PYTHONPATH=. python experiments/011-limits-and-models/adata.py shuffle    # 1e9 training positions, shuffled once

Output in data/011/ (git-ignored):
  registry.json          every matrix with an integer id, shape, role, layer and its cross-matrix source
  codes/<name>.npz       merged symbols (15 letters, uint8) and scale bytes
  train_parts/<name>.npz sampled positions (flat index) and log p_chain (float16) of a training-pool matrix
  val/<name>.npz         100,000 positions of a validation matrix (log p_chain float32)
  test/<name>.npy        log p_chain (float32) of every position of a T1 / T2 matrix; chain bits in test_bits.json
  rows/<name>.npy        log p_chain (float16) of every position: pool round 1 and validation (sequence models)
  train_all/             mid (uint16), idx (uint32), logp (float16, 15) of 1e9 positions in one random order;
                         the first N rows are a random subset of size N (nested subsets for the data scaling)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

import p1b  # noqa: E402

D = K.ROOT / "data" / "011"
POOL_ROUNDS = (1, 2)
N_TRAIN = 1_000_000_000
N_VAL_PER = 100_000


def pool():
    return [n for r in POOL_ROUNDS for n in K.pool_round(r)]


def registry():
    names = pool() + K.validation() + K.T1 + K.T2
    reg = []
    for i, n in enumerate(names):
        L, e, role = n.split(".")
        reg.append({"id": i, "name": n, "layer": int(L[1:]), "expert": int(e[1:]), "role": role,
                    "split": ("pool" if i < len(pool()) else "val" if n in K.validation() else
                              "T1" if n in K.T1 else "T2")})
    ids = {r["name"]: r["id"] for r in reg}
    for r in reg:
        pre = f"L{r['layer']}.E{r['expert']}."
        r["cross"] = ids[pre + "w3"] if r["role"] == "w1" else ids[pre + "w1"] if r["role"] == "w2" else None
        r["cross_kind"] = "same" if r["role"] == "w1" else "transposed" if r["role"] == "w2" else None
    return reg


def build_one(args):
    name, split, q = args
    chain = K._load_chains()[name]
    C, E = K.read(name)
    pn = K.partner_of(name)
    part = K.read(pn)[0] if pn else None
    sym, p, Kk, side, scale, _ = p1b.chain_probs(C, E, p1b.active(chain), part)
    assert Kk == 15
    n = len(sym)
    bits = float(-np.log2(p[np.arange(n), sym]).sum())
    np.savez(D / "codes" / f"{name}.npz", sym=sym.reshape(C.shape).astype(np.uint8), E=E)
    rng = np.random.default_rng([K.SEED, 7, zlib.crc32(name.encode())])
    if split == "pool":
        k = int(round(q * n))
        idx = np.sort(rng.choice(n, size=k, replace=False)).astype(np.uint32)
        np.savez(D / "train_parts" / f"{name}.npz", idx=idx, logp=np.log(p[idx]).astype(np.float16))
    elif split == "val":
        idx = np.sort(rng.choice(n, size=N_VAL_PER, replace=False)).astype(np.uint32)
        np.savez(D / "val" / f"{name}.npz", idx=idx, logp=np.log(p[idx]).astype(np.float32))
    else:
        np.save(D / "test" / f"{name}.npy", np.log(p).astype(np.float32))
    return {"name": name, "n": n, "chain_code_bits": bits, "side": side, "scale": scale}


def stage_chains():
    K.ensure_chains([r["name"] for r in registry()], workers=64)


def stage_build(workers=32):
    for d in ("codes", "train_parts", "val", "test"):
        (D / d).mkdir(parents=True, exist_ok=True)
    reg = registry()
    json.dump(reg, open(D / "registry.json", "w"), indent=1)
    n_pool = sum(1 for r in reg if r["split"] == "pool")
    q = N_TRAIN / (n_pool * 2304 * 5120)
    done = {p.stem for p in (D / "codes").glob("*.npz")}
    jobs = [(r["name"], r["split"], q) for r in reg if r["name"] not in done]
    info_path = D / "build_info.jsonl"
    with ProcessPoolExecutor(workers) as ex, open(info_path, "a") as f:
        for r in ex.map(build_one, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print("built", r["name"], flush=True)


def stage_shuffle():
    reg = json.load(open(D / "registry.json"))
    parts = [(r["id"], np.load(D / "train_parts" / f"{r['name']}.npz")) for r in reg if r["split"] == "pool"]
    tot = sum(len(z["idx"]) for _, z in parts)
    print("training positions", tot, flush=True)
    out = D / "train_all"
    out.mkdir(exist_ok=True)
    perm = np.random.default_rng([K.SEED, 8]).permutation(tot)
    mid = np.concatenate([np.full(len(z["idx"]), i, dtype=np.uint16) for i, z in parts])
    np.save(out / "mid.npy", mid[perm])
    del mid
    idx = np.concatenate([z["idx"] for _, z in parts])
    np.save(out / "idx.npy", idx[perm])
    del idx
    lp = np.lib.format.open_memmap(out / "logp.npy", mode="w+", dtype=np.float16, shape=(tot, 15))
    allp = np.concatenate([z["logp"] for _, z in parts])
    step = 50_000_000
    for i in range(0, tot, step):
        lp[i:i + step] = allp[perm[i:i + step]]
    lp.flush()
    json.dump({"n": int(tot), "seed": [K.SEED, 8]}, open(out / "info.json", "w"))
    print("done", flush=True)


def rows_one(name):
    chain = K._load_chains()[name]
    C, E = K.read(name)
    pn = K.partner_of(name)
    part = K.read(pn)[0] if pn else None
    sym, p, Kk, side, scale, _ = p1b.chain_probs(C, E, p1b.active(chain), part)
    np.save(D / "rows" / f"{name}.npy", np.log(p).astype(np.float16))
    return name


def stage_rows(workers=32):
    """Full-row data for the sequence models (A2): log p_chain (float16) of every position of the training-pool
    round-1 matrices and of the validation matrices."""
    (D / "rows").mkdir(exist_ok=True)
    names = [n for n in K.pool_round(1) + K.validation() if not (D / "rows" / f"{n}.npy").exists()]
    with ProcessPoolExecutor(workers) as ex:
        for n in ex.map(rows_one, names):
            print("rows", n, flush=True)


if __name__ == "__main__":
    {"chains": stage_chains, "build": stage_build, "shuffle": stage_shuffle, "rows": stage_rows}[sys.argv[1]]()
