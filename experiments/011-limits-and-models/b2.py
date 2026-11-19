"""Spec 011 B2: decision tree -> static probability table (JPEG XL MA tree / CALIC / LOCO-I context quantisation).

    PYTHONPATH=. python experiments/011-limits-and-models/b2.py data       # features of 50M training positions
    PYTHONPATH=. python experiments/011-limits-and-models/b2.py synth      # V-a / V-b
    PYTHONPATH=. python experiments/011-limits-and-models/b2.py run        # 4 leaf counts x 4 sharing levels

The encoder grows a tree on the cheap features of bfeat.py (best-first, split = threshold on one feature,
criterion = empirical entropy, at least 2,000 training positions per leaf); every leaf holds a static table of
15 probabilities with 12-bit precision. The decoder walks the tree (about log2(leaves) comparisons) and decodes
the code with that table (e.g. rANS). Code length = sum of -log2 p_table(code) (ideal static-table code length);
tree and tables are billed: 16 bit per internal node (feature id + threshold), 15 x 12 bit per leaf.
"Adaptive version": the same leaves used as contexts of the chain's sequential adaptive code (hier_bits) on each
matrix (no tables to store) -- how much the static, shared tables lose against adapting to each matrix.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import heapq
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bfeat as BF  # noqa: E402

RES = HERE / "results" / "b"
NTRAIN = 50_000_000
LEAVES = [16, 64, 256, 1024]
SHARING = ["model", "role", "layer", "layer_role"]
ALL_W = 557171343360
KQ = 15


def ent(counts):
    n = counts.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(counts > 0, counts * np.log2(np.where(n[..., None] > 0, n[..., None], 1) / np.where(counts > 0, counts, 1)), 0.0)
    return h.sum(-1)


def best_split(F, y, idx, min_leaf):
    best = (0.0, None, None)
    yy = y[idx]
    base = ent(np.bincount(yy, minlength=KQ).astype(np.float64))
    for f in range(F.shape[1]):
        v = F[idx, f].astype(np.int64)
        V = int(v.max()) + 1
        if V < 2:
            continue
        cnt = np.bincount(v * KQ + yy, minlength=V * KQ).reshape(V, KQ).astype(np.float64)
        left = np.cumsum(cnt, 0)[:-1]
        right = cnt.sum(0)[None] - left
        nl, nr = left.sum(1), right.sum(1)
        ok = (nl >= min_leaf) & (nr >= min_leaf)
        if not ok.any():
            continue
        gain = base - ent(left) - ent(right)
        gain = np.where(ok, gain, -1.0)
        t = int(np.argmax(gain))
        if gain[t] > best[0]:
            best = (float(gain[t]), f, t)
    return best


class Tree:
    """nodes: list of (feature, threshold, left, right) or leaf id; table: (leaves, 15) probabilities."""

    def __init__(self, F, y, n_leaves, min_leaf=2000):
        self.nodes = [None]
        leaves = {0: np.arange(len(y))}
        heap = []
        g, f, t = best_split(F, y, leaves[0], min_leaf)
        if f is not None:
            heapq.heappush(heap, (-g, 0, f, t))
        while heap and len(leaves) < n_leaves:
            g, node, f, t = heapq.heappop(heap)
            idx = leaves.pop(node)
            m = F[idx, f] <= t
            l, r = len(self.nodes), len(self.nodes) + 1
            self.nodes += [None, None]
            self.nodes[node] = (f, t, l, r)
            for child, sel in ((l, idx[m]), (r, idx[~m])):
                leaves[child] = sel
                gg, ff, tt = best_split(F, y, sel, min_leaf)
                if ff is not None:
                    heapq.heappush(heap, (-gg, child, ff, tt))
        self.leaf_of = {}
        tabs = []
        for node, idx in leaves.items():
            self.leaf_of[node] = len(tabs)
            c = np.bincount(y[idx], minlength=KQ).astype(np.float64) + 0.5
            tabs.append(quantize(c / c.sum()))
        self.table = np.array(tabs)
        self.n_leaves = len(tabs)
        self.n_internal = sum(1 for n in self.nodes if n is not None)

    def leaf(self, F):
        nn = len(self.nodes)
        feat = np.zeros(nn, dtype=np.int64)
        thr = np.zeros(nn, dtype=np.int64)
        lc = np.arange(nn)
        rc = np.arange(nn)
        internal = np.zeros(nn, dtype=bool)
        for n, v in enumerate(self.nodes):
            if v is not None:
                feat[n], thr[n], lc[n], rc[n] = v
                internal[n] = True
        node = np.zeros(len(F), dtype=np.int64)
        rows = np.arange(len(F))
        while internal[node].any():
            go = F[rows, feat[node]] <= thr[node]
            node = np.where(internal[node], np.where(go, lc[node], rc[node]), node)
        lid = np.array([self.leaf_of.get(n, -1) for n in range(nn)])
        return lid[node]

    def cost_bits(self):
        return 16.0 * self.n_internal + KQ * 12.0 * self.n_leaves


def quantize(p, bits=12):
    tot = 1 << bits
    q = np.maximum(1, np.round(p * tot)).astype(np.int64)
    q[np.argmax(q)] += tot - q.sum()
    return q / tot


def group_key(sharing, layer, role):
    return {"model": "all", "role": role, "layer": str(layer), "layer_role": f"{layer}.{role}"}[sharing]


# ------------------------------------------------------------------ data

def _feat_part(args):
    mid, rows, idx = args
    M = BF.Matrices()
    F, y = BF.matrix_features(M, M.byid[mid]["name"])
    return rows, F[idx], y[idx]


def stage_data(workers=32):
    D = BF.D
    mids = np.load(D / "train_all" / "mid.npy", mmap_mode="r")[:NTRAIN].astype(np.int64)
    idx = np.load(D / "train_all" / "idx.npy", mmap_mode="r")[:NTRAIN].astype(np.int64)
    jobs = []
    for m in np.unique(mids):
        rows = np.flatnonzero(mids == m)
        jobs.append((int(m), rows, idx[rows]))
    F = np.zeros((NTRAIN, len(BF.NAMES)), dtype=np.int16)
    y = np.zeros(NTRAIN, dtype=np.int8)
    with ProcessPoolExecutor(workers) as ex:
        for rows, f, yy in ex.map(_feat_part, jobs):
            F[rows] = f
            y[rows] = yy
    np.savez(D / "b_train_features.npz", F=F, y=y, mid=mids)
    print("saved", F.shape, flush=True)


# ------------------------------------------------------------------ evaluation

def _eval_matrix(args):
    name, trees_path = args
    import pickle
    sys.path.insert(0, str(HERE.parents[1] / "experiments" / "010-transforms"))
    from redundancy import increment as I
    import p1b
    trees = pickle.load(open(trees_path, "rb"))
    M = BF.Matrices()
    F, y = BF.matrix_features(M, name)
    r = M.reg[name]
    out = {"task": name, "n": len(y)}
    for key, tr in trees.items():
        sharing, leaves = key
        t = tr[group_key(sharing, r["layer"], r["role"])]
        lf = t.leaf(F)
        static = float(-np.log2(t.table[lf, y]).sum())
        adapt = I.hier_bits(y, [np.zeros(len(y), dtype=np.int64), lf], KQ, p1b.P["beta"], p1b.P["nblocks"])
        out[f"{sharing}:{leaves}"] = {"static": static, "adaptive": adapt}
    return out


def stage_run(workers=48):
    import pickle
    import common as K
    D = BF.D
    z = np.load(D / "b_train_features.npz")
    F, y, mid = z["F"], z["y"].astype(np.int64), z["mid"]
    reg = {r["id"]: r for r in json.load(open(D / "registry.json"))}
    lay = np.array([reg[int(m)]["layer"] for m in range(max(reg) + 1)])[mid]
    rol = np.array([reg[int(m)]["role"] for m in range(max(reg) + 1)])[mid]
    trees = {}
    info = {}
    for sharing in SHARING:
        groups = {}
        allk = np.char.add(np.char.add(lay.astype(str), "."), rol) if sharing == "layer_role" else \
            (lay.astype(str) if sharing == "layer" else (rol if sharing == "role" else np.full(len(lay), "all")))
        for g in np.unique(allk):
            groups[str(g)] = np.flatnonzero(allk == g)
        for leaves in LEAVES:
            jobs = {g: ix for g, ix in groups.items()}
            with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
                futs = {g: ex.submit(Tree, F[ix], y[ix], leaves) for g, ix in jobs.items()}
                trees[(sharing, leaves)] = {g: f.result() for g, f in futs.items()}
            tr = trees[(sharing, leaves)]
            info[f"{sharing}:{leaves}"] = {"trees": len(tr), "leaves_total": int(sum(t.n_leaves for t in tr.values())),
                                           "cost_bits_all": float(sum(t.cost_bits() for t in tr.values()))}
            print(sharing, leaves, info[f"{sharing}:{leaves}"], flush=True)
    tp = D / "b2_trees.pkl"
    pickle.dump(trees, open(tp, "wb"))
    test = K.T1 + K.T2
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_eval_matrix, [(n, str(tp)) for n in test]))
    with open(RES / "b2_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    chain = {json.loads(l)["name"]: json.loads(l)["chain_code_bits"] for l in open(D / "build_info.jsonl")}
    reg_n = {r["name"]: r for r in reg.values()}
    out = {}
    for key, tr in trees.items():
        k = f"{key[0]}:{key[1]}"
        res = {"trees": info[k]}
        for split, names in (("T1", K.T1), ("T2", K.T2)):
            sel = [r for r in rows if r["task"] in names]
            n = sum(r["n"] for r in sel)
            ch = sum(chain[r["task"]] for r in sel)
            st = sum(r[k]["static"] for r in sel)
            ad = sum(r[k]["adaptive"] for r in sel)
            used = {group_key(key[0], reg_n[r["task"]]["layer"], reg_n[r["task"]]["role"]) for r in sel}
            cost_test = sum(tr[g].cost_bits() for g in used)
            res[split] = {"static_gain": (ch - st) / n, "static_net_cost_over_test": (ch - st - cost_test) / n,
                          "static_net_cost_over_whole_model": (ch - st) / n - info[k]["cost_bits_all"] / ALL_W,
                          "adaptive_gain": (ch - ad) / n, "static_minus_adaptive_loss": (st - ad) / n,
                          "chain_bits_per_code": ch / n}
        out[k] = res
        print(k, {s: {a: round(b, 5) for a, b in res[s].items()} for s in ("T1", "T2")}, flush=True)
    json.dump(out, open(RES / "b2.json", "w"), indent=1)


def stage_synth():
    """V-a: magnitudes from the block scale only, random signs (the chain sees everything) -> gain about 0
    (static tables lose a little against adaptation, plus their storage). V-b: in addition each code repeats the
    block's running maximum with probability 1/2 (s0.synth_matrix) -> the tree (features include the running
    maximum) should find most of (chain code length - exact generator code length)."""
    import common as K
    import s0
    sys.path.insert(0, str(K.ROOT / "experiments" / "010-transforms"))
    import p1b
    from redundancy import increment as I
    out = {}
    for kind, cm in (("V-a", 0.0), ("V-b", 0.5)):
        rng = np.random.default_rng(51)
        mats = [s0.synth_matrix(rng, 512, 2048, cm) for _ in range(5)]
        Fs, ys = [], []
        for C, E, _ in mats[:4]:
            S = np.where(C == 8, 0, C)
            S = np.where(S > 8, S - 1, S)
            F, y = BF.features_from_arrays(S, E, None, "w3", 0)
            Fs.append(F)
            ys.append(y)
        t = Tree(np.concatenate(Fs), np.concatenate(ys), 256)
        C, E, tb = mats[4]
        ch = K.tx.chain(C, E)
        sym, p, Kk, side, scale, _ = p1b.chain_probs(C, E, p1b.active(ch))
        chain_bits = float(-np.log2(p[np.arange(len(sym)), sym]).sum())
        F, y = BF.features_from_arrays(sym.reshape(C.shape), E, None, "w3", 0)
        lf = t.leaf(F)
        st = float(-np.log2(t.table[lf, y]).sum())
        n = len(y)
        out[kind] = {"static_gain": (chain_bits - st) / n, "static_net_of_table_cost": (chain_bits - st - t.cost_bits()) / n,
                     "theory": (chain_bits - tb) / n, "leaves": t.n_leaves}
        print(kind, out[kind], flush=True)
    out["checks"] = {"V-a_about_0": bool(-0.01 < out["V-a"]["static_gain"] < 0.005),
                     "V-b_recovers": bool(0.8 * out["V-b"]["theory"] < out["V-b"]["static_gain"] <= out["V-b"]["theory"] + 0.002)}
    out["all_pass"] = all(out["checks"].values())
    json.dump(out, open(RES / "synthetic_b2.json", "w"), indent=1)
    print(out["checks"], out["all_pass"])


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    {"data": stage_data, "run": stage_run, "synth": stage_synth}[sys.argv[1]]()
