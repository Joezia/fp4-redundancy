"""Spec 011: data splits, matrix access and the 008 chain reference (shared by every stage).

T1  = 010's 24 test matrices: layers 0, 13, 26, 39; the 3rd and 4th sampled expert of each (w1, w2, w3).
T2  = 24 matrices: layers 5, 20, 33 and one MTP layer (chosen with the seed), 2 experts each, fixed seed.
Pool (training) = every other expert of all 43 layers, in rounds: round r = the r-th expert of each layer in the
      order default_rng([SEED, 1, layer]).permutation, with T1 and T2 experts removed.
Validation = the round-1 expert of 8 layers (3, 11, 17, 23, 29, 35, 38, 41); never used for training.
Decode order (as P1b): w3 -> w1 -> w2; w1 may see w3, w2 may see w1; the scale stream first.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "010-transforms"))
import tx  # noqa: E402

E11 = Path("experiments/011-limits-and-models")
RES = E11 / "results"
SEED = 20261121
N_EXPERTS = {L: (384 if L < 40 else 128) for L in range(43)}
VAL_LAYERS = [3, 11, 17, 23, 29, 35, 38, 41]
ROLES = ("w1", "w2", "w3")


def t1_experts():
    ex = tx.sample_experts()
    return {L: es[2:] for L, es in ex.items()}


def t2_experts():
    rng = np.random.default_rng([SEED, 2])
    mtp = int(rng.choice([40, 41, 42]))
    return {L: [int(e) for e in np.random.default_rng([SEED, 2, L]).permutation(N_EXPERTS[L])[:2]]
            for L in (5, 20, 33, mtp)}


def names(experts):
    return [f"L{L}.E{e}.{r}" for L, es in experts.items() for e in es for r in ROLES]


T1 = names(t1_experts())
T2 = names(t2_experts())


def pool_order():
    excl = {}
    for d in (t1_experts(), t2_experts()):
        for L, es in d.items():
            excl.setdefault(L, set()).update(es)
    return {L: [int(e) for e in np.random.default_rng([SEED, 1, L]).permutation(N_EXPERTS[L]) if int(e) not in excl.get(L, set())]
            for L in range(43)}


def validation():
    o = pool_order()
    return names({L: [o[L][0]] for L in VAL_LAYERS})


def pool_round(r):
    """training experts of round r (1-based); the validation experts are skipped."""
    o = pool_order()
    out = {}
    for L in range(43):
        lst = [e for i, e in enumerate(o[L]) if not (L in VAL_LAYERS and i == 0)]
        if r - 1 < len(lst):
            out[L] = [lst[r - 1]]
    return names(out)


read = tx.read


def partner_of(name):
    L, e, role = name.split(".")
    return f"{L}.{e}.w3" if role == "w1" else None


# ------------------------------------------------------------------ 008 chain reference, cached per matrix

CHAINS = RES / "chains.jsonl"


def _load_chains():
    out = {}
    for line in open(ROOT / "experiments/010-transforms/results/base.jsonl"):     # 010 results, read only
        r = json.loads(line)
        out[r["task"]] = r["chain"]
    if CHAINS.exists():
        for line in open(CHAINS):
            r = json.loads(line)
            out[r["task"]] = r["chain"]
    return out


def chain_task(name):
    t0 = time.process_time()
    C, E = read(name)
    pn = partner_of(name)
    part = read(pn)[0] if pn else None
    return {"task": name, "partner": pn, "chain": tx.chain(C, E, part), "cpu": time.process_time() - t0}


def ensure_chains(names_, workers=48):
    from concurrent.futures import ProcessPoolExecutor
    have = _load_chains()
    todo = [n for n in names_ if n not in have]
    if todo:
        RES.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(min(workers, len(todo))) as ex, open(CHAINS, "a") as f:
            for r in ex.map(chain_task, todo):
                f.write(json.dumps(r) + "\n")
                f.flush()
    return _load_chains()


if __name__ == "__main__":
    print("T1", T1[:3], len(T1))
    print("T2", T2[:3], len(T2), t2_experts())
    print("val", validation()[:3], len(validation()))
    print("pool round 1", len(pool_round(1)), pool_round(1)[:3])
    s = set(T1) | set(T2)
    for r in range(1, 6):
        assert not (set(pool_round(r)) & s) and not (set(pool_round(r)) & set(validation()))
    print("no overlap ok")
