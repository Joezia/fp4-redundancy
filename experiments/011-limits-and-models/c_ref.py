"""Spec 011 line C: reference experts (read once, shared by C1 and C3).

    set -a && source .env && set +a
    PYTHONPATH=. python experiments/011-limits-and-models/c_ref.py [workers]

Reference experts = pool rounds 3..10 of every layer (8 experts x 3 roles per layer, 1,032 matrices); they are not
used for training (rounds 1-2), validation, T1 or T2. Each is stored as data/011/ref/<name>.npz with the merged
15-symbol codes (sym, uint8, 0 = zero, 1..7 = +0.5..+6, 8..14 = -0.5..-6) and the scale bytes (E).
Also writes data/011/ref/index.json (layer -> list of reference experts).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as K  # noqa: E402

D = Path(os.environ.get("D011", str(HERE.parents[1] / "data" / "011")))
OUT = D / "ref"
ROUNDS = range(3, 11)


def merge(C):
    """raw E2M1 code (bit 3 = sign) -> merged symbol (-0 joins +0)."""
    S = np.where(C == 8, 0, C)
    return np.where(S > 8, S - 1, S).astype(np.uint8)


def ref_experts():
    out = {}
    for r in ROUNDS:
        for n in K.pool_round(r):
            L, e, _ = n.split(".")
            out.setdefault(int(L[1:]), set()).add(int(e[1:]))
    return {L: sorted(es) for L, es in sorted(out.items())}


def one(name):
    p = OUT / f"{name}.npz"
    if p.exists():
        return name
    C, E = K.read(name)
    np.savez(p, sym=merge(C), E=E)
    return name


def main(workers=32):
    OUT.mkdir(parents=True, exist_ok=True)
    ex = ref_experts()
    json.dump({str(L): es for L, es in ex.items()}, open(OUT / "index.json", "w"))
    names = [f"L{L}.E{e}.{r}" for L, es in ex.items() for e in es for r in K.ROLES]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(one, names)):
            if i % 100 == 0:
                print(f"{i}/{len(names)}  {time.time() - t0:.0f}s", flush=True)
    print("REF_DONE", len(names), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
