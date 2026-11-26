"""Spec 016, part C: joint general-purpose compression of many matrices (zstd long-distance matching, 2 GiB window)
against the sum of the matrices compressed one by one.

    PYTHONPATH=.:experiments/016-cross-matrix python experiments/016-cross-matrix/joint.py {deepseek|kimi} [--workers N]

Each group is written as one stream (code bytes, packed 2 codes per byte, or unpacked 1 code per byte) and
compressed with `zstd -9 --long=31 -T1`; the separate baseline compresses every matrix of the group on its own with
the same command. joint - separate is what the compressor gained across matrices (any-offset exact matches within
the window). Control: the same group with every matrix's codes randomly permuted inside the matrix.
"""
import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as K

SEED = 20261125
CMD = ["zstd", "-9", "--long=31", "-T1", "-q", "-c"]


def groups(model):
    ents = K.entries(model)
    layers = sorted({e["layer"] for e in ents})
    mid = layers[len(layers) // 2]
    g = {}
    for L in (layers[0], mid):
        ids = [i for i, e in enumerate(ents) if e["layer"] == L]                 # layer -> expert -> role (list order)
        g[f"layer{L}_all_experts"] = ids
    by = {}
    for i, e in enumerate(ents):
        by.setdefault((e["role"], e["expert"]), []).append(i)
    # the same expert index across all layers, adjacent in the stream (role -> expert -> layer)
    g["w1_expert0-7_all_layers"] = [i for ex in range(8) for i in sorted(by[("w1", ex)], key=lambda i: ents[i]["layer"])]
    g["w2_expert0-7_all_layers"] = [i for ex in range(8) for i in sorted(by[("w2", ex)], key=lambda i: ents[i]["layer"])]
    return g


def matrix_bytes(model, i, rep, shuffle):
    C, _ = K.load(model, K.entries(model)[i])
    if shuffle:
        C = C.ravel()[np.random.default_rng(SEED + i).permutation(C.size)].reshape(C.shape)
    if rep == "unpacked":
        return np.ascontiguousarray(C).tobytes()
    f = C.ravel()
    return ((f[0::2] & 15) | (f[1::2] << 4)).astype(np.uint8).tobytes()


def separate(model, i, rep, shuffle):
    b = matrix_bytes(model, i, rep, shuffle)
    return len(b), len(subprocess.run(CMD, input=b, capture_output=True, check=True).stdout)


def joint(model, ids, rep, shuffle):
    p = subprocess.Popen(CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    import threading
    out = []
    th = threading.Thread(target=lambda: out.append(len(p.stdout.read())))
    th.start()
    n = 0
    for i in ids:
        b = matrix_bytes(model, i, rep, shuffle)
        n += len(b)
        p.stdin.write(b)
    p.stdin.close()
    th.join(); p.wait()
    return n, out[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(K.MODELS))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    path = K.RES / f"joint_{a.model}.jsonl"
    done = {(r["group"], r["rep"], r["shuffled"]) for r in map(json.loads, open(path))} if path.exists() else set()
    G = groups(a.model)
    jobs = [(g, rep, sh) for g in G for rep in ("packed", "unpacked") for sh in (False, True)
            if not (rep == "unpacked" and not g.startswith("layer"))]
    if a.only:
        jobs = [j for j in jobs if j[0] == a.only]
    with open(path, "a") as fo, ProcessPoolExecutor(a.workers) as ex:
        for g, rep, sh in jobs:
            if (g, rep, sh) in done:
                continue
            ids = G[g]
            t0 = time.time()
            fj = ex.submit(joint, a.model, ids, rep, sh)
            seps = list(ex.map(separate, [a.model] * len(ids), ids, [rep] * len(ids), [sh] * len(ids), chunksize=4))
            n, cj = fj.result()
            cs = sum(c for _, c in seps)
            r = {"group": g, "rep": rep, "shuffled": sh, "n_matrices": len(ids), "raw_bytes": n,
                 "n_codes": n * (2 if rep == "packed" else 1), "joint_bytes": cj, "separate_sum_bytes": cs,
                 "joint_gain_bits_per_code": 8 * (cs - cj) / (n * (2 if rep == "packed" else 1)),
                 "separate_bits_per_code": 8 * cs / (n * (2 if rep == "packed" else 1)),
                 "seconds": time.time() - t0, "command": " ".join(CMD)}
            fo.write(json.dumps(r) + "\n"); fo.flush()
            print(json.dumps(r), flush=True)
    print("JOINT_DONE", a.model, flush=True)


if __name__ == "__main__":
    main()
