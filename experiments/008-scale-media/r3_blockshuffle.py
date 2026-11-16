"""Spec 008 V6 follow-up: is the chain's R3 increment (after R2 = scale context) block-level information?
Shuffles codes within each 32-code block (keeps every block's code set and scale) and compares the R3 increment.
    PYTHONPATH=. python experiments/008-scale-media/r3_blockshuffle.py <matrix name from experiments/r1-r4/samples.json>
"""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json
import numpy as np
from redundancy import increment as I
from redundancy.increment_run import FP4_ORDERS
from redundancy.adapters.fp4 import FP4
name = sys.argv[1]
P = json.load(open("experiments/008-scale-media/results/fp4/increment_dev/params.json"))
P = P.get("params", P)
ad = FP4({"samples": "experiments/r1-r4/samples.json", "snapshot_env": "SNAPSHOT"})
C, E = ad._read(ad.samples[name]); C = C.astype(np.int64)
rng = np.random.default_rng(0)
B = P["block"]
Cs = C.reshape(-1, B).copy()
idx = np.argsort(rng.random(Cs.shape), axis=1)
Cs = np.take_along_axis(Cs, idx, axis=1).reshape(C.shape)
order = FP4_ORDERS["default"]
out = {"item": name}
for tag, X in (("original", C), ("within_block_shuffled", Cs)):
    a = I.fp4_codelen(X, E, frozenset({"R1c", "R1s", "R2"}), P, order)[0]
    b = I.fp4_codelen(X, E, frozenset({"R1c", "R1s", "R2", "R3"}), P, order)[0]
    out[tag] = {"R3_bits_per_weight": (a - b) / C.size}
print(json.dumps(out), flush=True)
