"""Spec 011 B3: grouped parallel decoding inside the 32-blocks (checkerboard / channel-group idea), on B2's tree.

    PYTHONPATH=. python experiments/011-limits-and-models/b3.py synth
    PYTHONPATH=. python experiments/011-limits-and-models/b3.py run

Every block of a matrix is decoded in k rounds; in round g all positions of group g of all blocks are decoded in
parallel. A code may use: the scale bytes, the partner matrix (decoded before), its position, role and layer, and
the codes of its own block that belong to earlier groups (max level, count of |v| >= 4, mean level, and the left /
right neighbour if already decoded). No codes of other blocks of the same matrix (they are decoded in parallel).
Groupings: one group (fully parallel); 2 (even / odd); 4 (position mod 4); first 8 + last 24; and the serial
reference (32 groups: every earlier position of the block, blocks in parallel).
The tree (best-first, entropy criterion, model-wide, 1024 leaves, 12-bit tables) is B2's best setting.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bfeat as BF  # noqa: E402
import b2  # noqa: E402

RES = HERE / "results" / "b"
NTRAIN = 20_000_000
GROUPINGS = {"1 (all parallel)": np.zeros(32, dtype=int), "2 (even/odd)": np.arange(32) % 2,
             "4 (every 4th)": np.arange(32) % 4, "8+24": (np.arange(32) >= 8).astype(int),
             "32 (serial in block)": np.arange(32)}
FNAMES = ["scale", "rel_scale", "partner_mag", "partner_sign", "role", "layer", "pos", "max_A", "cnt4_A", "mean_A",
          "left_A", "right_A"]


def grouped_features(S, E, X, role, layer, g):
    R, W = S.shape
    NB = W // 32
    mag, sign = BF.mag_sign(S)
    lv = mag.reshape(R, NB, 32)
    F = np.zeros((R, NB, 32, len(FNAMES)), dtype=np.int16)
    Eb = E.astype(np.int16)[:, :, None]
    F[..., 0] = Eb
    F[..., 1] = (np.clip(Eb - np.round(E.astype(np.float64).mean(1))[:, None, None].astype(np.int16), -8, 8) + 8)
    if X is not None:
        xm, xs = BF.mag_sign(X)
        F[..., 2], F[..., 3] = xm.reshape(R, NB, 32), xs.reshape(R, NB, 32)
    else:
        F[..., 2], F[..., 3] = 8, 2
    F[..., 4] = {"w1": 0, "w2": 1, "w3": 2}[role]
    F[..., 5] = layer
    F[..., 6] = np.arange(32)
    for j in range(32):
        A = g < g[j]
        if A.any():
            sub = lv[..., A]
            F[:, :, j, 7] = sub.max(-1) + 1
            F[:, :, j, 8] = np.minimum((sub >= 6).sum(-1), 8)
            F[:, :, j, 9] = np.clip(np.round(2 * sub.mean(-1)), 0, 14)
        else:
            F[:, :, j, 7], F[:, :, j, 8], F[:, :, j, 9] = 0, 0, 15
        F[:, :, j, 10] = lv[:, :, j - 1] if j > 0 and A[j - 1] else 8
        F[:, :, j, 11] = lv[:, :, j + 1] if j < 31 and A[j + 1] else 8
    return F.reshape(R * W, -1), S.ravel()


def _feat(args):
    name, gkey, idx = args
    M = BF.Matrices()
    S, E = M.codes(name)
    r = M.reg[name]
    X = None
    if r["cross"] is not None:
        X, _ = M.codes(M.byid[r["cross"]]["name"])
        if r["cross_kind"] == "transposed":
            X = X.T
    F, y = grouped_features(S, E, X, r["role"], r["layer"], GROUPINGS[gkey])
    return (F, y) if idx is None else (F[idx], y[idx])


def train_data(gkey, workers=32):
    D = BF.D
    mids = np.load(D / "train_all" / "mid.npy", mmap_mode="r")[:NTRAIN].astype(np.int64)
    idx = np.load(D / "train_all" / "idx.npy", mmap_mode="r")[:NTRAIN].astype(np.int64)
    reg = {r["id"]: r for r in json.load(open(D / "registry.json"))}
    jobs, rows = [], []
    for m in np.unique(mids):
        sel = np.flatnonzero(mids == m)
        jobs.append((reg[int(m)]["name"], gkey, idx[sel]))
        rows.append(sel)
    F = np.zeros((NTRAIN, len(FNAMES)), dtype=np.int16)
    y = np.zeros(NTRAIN, dtype=np.int64)
    with ProcessPoolExecutor(workers) as ex:
        for sel, (f, yy) in zip(rows, ex.map(_feat, jobs)):
            F[sel], y[sel] = f, yy
    return F, y


def _eval(args):
    name, gkey, tree = args
    F, y = _feat((name, gkey, None))
    return {"task": name, "n": len(y), "static": float(-np.log2(tree.table[tree.leaf(F), y]).sum())}


def stage_run():
    import common as K
    chain = {json.loads(l)["name"]: json.loads(l)["chain_code_bits"] for l in open(BF.D / "build_info.jsonl")}
    out = {}
    for gkey, g in GROUPINGS.items():
        F, y = train_data(gkey)
        tree = b2.Tree(F, y, 1024)
        del F, y
        with ProcessPoolExecutor(24) as ex:
            rows = list(ex.map(_eval, [(n, gkey, tree) for n in K.T1 + K.T2]))
        res = {"rounds": int(g.max() + 1), "leaves": tree.n_leaves, "table_cost_bits": tree.cost_bits()}
        for split, ns in (("T1", K.T1), ("T2", K.T2)):
            sel = [r for r in rows if r["task"] in ns]
            n = sum(r["n"] for r in sel)
            ch = sum(chain[r["task"]] for r in sel)
            st = sum(r["static"] for r in sel)
            res[split] = {"static_gain": (ch - st) / n, "net_cost_over_test": (ch - st - tree.cost_bits()) / n,
                          "net_cost_over_whole_model": (ch - st) / n - tree.cost_bits() / b2.ALL_W}
        out[gkey] = res
        print(gkey, res, flush=True)
        json.dump(out, open(RES / "b3.json", "w"), indent=1)


def stage_synth():
    """V-a: codes depend only on the block scale -> grouping changes nothing, gains about 0 against the chain.
    V-b: magnitudes repeat the block's running maximum with probability 1/2 (s0.synth_matrix): the serial-in-block
    tree recovers most of (chain - exact code length); fewer rounds see less of the block and recover less."""
    import common as K
    import s0
    sys.path.insert(0, str(K.ROOT / "experiments" / "010-transforms"))
    import p1b
    out = {}
    for kind, cm in (("V-a", 0.0), ("V-b", 0.5)):
        rng = np.random.default_rng(91)
        mats = [s0.synth_matrix(rng, 512, 2048, cm) for _ in range(5)]
        merged = []
        for C, E, tb in mats:
            S = np.where(C == 8, 0, C)
            merged.append((np.where(S > 8, S - 1, S), E, tb, C))
        res = {}
        C, E, tb = mats[4]
        ch = K.tx.chain(C, E)
        sym, p, Kk, side, scale, _ = p1b.chain_probs(C, E, p1b.active(ch))
        chain_bits = float(-np.log2(p[np.arange(len(sym)), sym]).sum())
        for gkey, g in GROUPINGS.items():
            Fs, ys = zip(*[grouped_features(S, E_, None, "w3", 0, g) for S, E_, _, _ in merged[:4]])
            tree = b2.Tree(np.concatenate(Fs), np.concatenate(ys), 256)
            F, y = grouped_features(merged[4][0], merged[4][1], None, "w3", 0, g)
            st = float(-np.log2(tree.table[tree.leaf(F), y]).sum())
            res[gkey] = (chain_bits - st) / len(y)
        res["theory"] = (chain_bits - tb) / C.size
        out[kind] = res
        print(kind, res, flush=True)
    out["checks"] = {"V-a_about_0": bool(all(-0.01 < v < 0.005 for k, v in out["V-a"].items() if k != "theory")),
                     "V-b_serial_recovers": bool(0.8 * out["V-b"]["theory"] < out["V-b"]["32 (serial in block)"]
                                                 <= out["V-b"]["theory"] + 0.002),
                     "V-b_parallel_le_serial": bool(out["V-b"]["1 (all parallel)"] <= out["V-b"]["32 (serial in block)"])}
    out["all_pass"] = all(out["checks"].values())
    json.dump(out, open(RES / "synthetic_b3.json", "w"), indent=1)
    print(out["checks"], out["all_pass"])


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    {"run": stage_run, "synth": stage_synth}[sys.argv[1]]()
