"""Spec 011 B4: context mixing (b4/cm.c) -- driver, synthetic checks and the test matrices.

    gcc -O2 -march=native -o experiments/011-limits-and-models/b4/cm experiments/011-limits-and-models/b4/cm.c -lm
    PYTHONPATH=. python experiments/011-limits-and-models/b4.py synth
    PYTHONPATH=. python experiments/011-limits-and-models/b4.py run

The C program encodes a matrix with a binary arithmetic coder, decodes it again and checks the result;
it reports the ideal code length (sum of -log2 p), the actual coded size and the encode / decode times
(single thread). B4 adapts on each matrix from scratch, so it has no stored model or table.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BIN = HERE / "b4" / "cm"
RES = HERE / "results" / "b"


def code(S, E, P=None, kind=0):
    R, W = S.shape
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        S.astype(np.uint8).tofile(d / "s.bin")
        E.astype(np.uint8).tofile(d / "e.bin")
        if P is not None:
            P.astype(np.uint8).tofile(d / "p.bin")
        r = subprocess.run([str(BIN), str(R), str(W), str(d / "s.bin"), str(d / "e.bin"),
                            str(d / "p.bin") if P is not None else "-", str(kind), str(d / "o.json")])
        out = json.load(open(d / "o.json"))
    out["returncode"] = r.returncode
    return out


def one(name):
    import bfeat as BF
    M = BF.Matrices()
    S, E = M.codes(name)
    r = M.reg[name]
    P, kind = None, 0
    if r["cross"] is not None:
        P, _ = M.codes(M.byid[r["cross"]]["name"])
        kind = 1 if r["cross_kind"] == "same" else 2
    o = code(S, E, P, kind)
    o["task"] = name
    return o


def stage_synth():
    import common as K
    import s0
    import p1b
    out = {}
    for kind, cm in (("V-a", 0.0), ("V-b", 0.5)):
        rng = np.random.default_rng(61)
        C, E, tb = s0.synth_matrix(rng, 2304, 5120, cm)    # the size of a real matrix
        ch = K.tx.chain(C, E)
        sym, p, Kk, side, scale, _ = p1b.chain_probs(C, E, p1b.active(ch))
        chain_bits = float(-np.log2(p[np.arange(len(sym)), sym]).sum())
        o = code(sym.reshape(C.shape), E)
        n = C.size
        out[kind] = {"gain": (chain_bits - o["ideal_bits"]) / n, "theory": (chain_bits - tb) / n, "lossless": o["lossless"],
                     "coded_vs_ideal_bits_per_code": (8 * o["coded_bytes"] - o["ideal_bits"]) / n,
                     "decode_codes_per_s": n / o["decode_s"]}
        print(kind, out[kind], flush=True)
    out["checks"] = {"lossless": bool(out["V-a"]["lossless"] and out["V-b"]["lossless"]),
                     "V-a_about_0": bool(-0.01 < out["V-a"]["gain"] < 0.005),
                     "V-b_recovers": bool(0.8 * out["V-b"]["theory"] < out["V-b"]["gain"] <= out["V-b"]["theory"] + 0.002)}
    out["all_pass"] = all(out["checks"].values())
    json.dump(out, open(RES / "synthetic_b4.json", "w"), indent=1)
    print(out["checks"], out["all_pass"])


def stage_run(workers=24):
    import common as K
    import bfeat as BF
    names = K.T1 + K.T2
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(one, names))
    with open(RES / "b4_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    chain = {json.loads(l)["name"]: json.loads(l)["chain_code_bits"] for l in open(BF.D / "build_info.jsonl")}
    out = {"all_lossless": all(r["lossless"] for r in rows)}
    for split, ns in (("T1", K.T1), ("T2", K.T2)):
        sel = [r for r in rows if r["task"] in ns]
        n = sum(r["n"] for r in sel)
        ch = sum(chain[r["task"]] for r in sel)
        ib = sum(r["ideal_bits"] for r in sel)
        cb = sum(8 * r["coded_bytes"] for r in sel)
        out[split] = {"gain_ideal": (ch - ib) / n, "gain_actual_coded": (ch - cb) / n, "chain_bits_per_code": ch / n,
                      "decode_codes_per_s_median": float(np.median([r["n"] / r["decode_s"] for r in sel])),
                      "encode_codes_per_s_median": float(np.median([r["n"] / r["encode_s"] for r in sel]))}
    json.dump(out, open(RES / "b4.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    {"synth": stage_synth, "run": stage_run}[sys.argv[1]]()
