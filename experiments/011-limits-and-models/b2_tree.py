"""Spec 011 B2 supplement: retrain the final B2 tree (model-wide, 256 leaves, v2 features) and export its structure.

    PYTHONPATH=. python experiments/011-limits-and-models/b2_tree.py train    # retrain (once; --retrain to redo) + T1/T2 check
    PYTHONPATH=. python experiments/011-limits-and-models/b2_tree.py export   # structure json + figure

train: the tree is grown with b2.Tree on the same 50M training positions (data/011/b_train_features.npz, causal
features only), exactly as b2.py stage_run does for ("model", 256); it is evaluated on T1 / T2 with b2._eval_matrix
and compared with results/b/b2_v2.json (must agree within 0.001 bit/weight). It is also compared node by node with
the tree saved by the original run (data/011/b2_trees.pkl).
export: the training positions are routed through the tree; for every node the sample count and the 15 counts are
recorded, and the split gain of an internal node is H(node) - H(left) - H(right) (empirical entropy, bits), the
criterion the tree was grown with. Writes results/b/b2_tree_structure.json and figures/b2_tree_top.png.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bfeat as BF  # noqa: E402
import b2  # noqa: E402

RES = HERE / "results" / "b"
FIG = HERE / "figures"
TREE = BF.D / "b2_tree_model256.pkl"
EVAL = BF.D / "b2_tree_model256_eval.pkl"
LEVELS = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
SYMS = ["0"] + [f"+{v:g}" for v in LEVELS[1:]] + [f"-{v:g}" for v in LEVELS[1:]]

# what each feature value means (all features use only codes decoded before the current one)
DESC = {
    "scale": "块 scale 字节 E",
    "rel_scale": "块 scale 相对本行平均 scale 的差（+8 偏移，截到 0..16）",
    "prev1_mag": "前 1 个码值的幅度等级（0..7 = 0, 0.5, 1, 1.5, 2, 3, 4, 6；8 = 行首无前值）",
    "prev1_sign": "前 1 个码值的符号（0 正、1 负、2 零或无）",
    "prev2_mag": "前 2 个码值的幅度等级（8 = 无）",
    "max_so_far": "块内已解码值的最大幅度等级 + 1（0 = 块内第 1 个）",
    "cnt4_so_far": "块内已出现幅度 ≥ 4 的个数（截到 8）",
    "pos": "在 32 个码值块内的位置 0..31",
    "partner_mag": "配对矩阵同位置码值的幅度等级（w1 看 w3；8 = 无配对）",
    "partner_sign": "配对矩阵同位置码值的符号",
    "above_mag": "上一行同列码值的幅度等级（8 = 第一行）",
    "role": "矩阵角色（0 = w1、1 = w2、2 = w3）",
    "layer": "层号",
    "mean_mag_so_far": "块内已解码值幅度等级平均值的 2 倍（15 = 块内第 1 个）",
    "odd_so_far": "块内已出现的 0.5、1.5（奇数等级）个数（截到 8）",
    "nz_so_far": "块内已出现的非零个数",
}


def train():
    if TREE.exists() and "--retrain" not in sys.argv:      # already retrained by an earlier call
        t = pickle.load(open(TREE, "rb"))
    else:
        z = np.load(BF.D / "b_train_features.npz")
        F, y = z["F"][:, BF.CAUSAL], z["y"].astype(np.int64)
        t = b2.Tree(F, y, 256)
        pickle.dump(t, open(TREE, "wb"))
    print("tree:", t.n_leaves, "leaves,", t.n_internal, "internal nodes", flush=True)
    sys.modules["__main__"].Tree = b2.Tree               # the original pickle was written by b2.py run as __main__
    old = pickle.load(open(BF.D / "b2_trees.pkl", "rb"))[("model", 256)]["all"]
    same = old.nodes == t.nodes and np.array_equal(old.table, t.table)
    pickle.dump({("model", 256): {"all": t}}, open(EVAL, "wb"))
    import common as K
    with ProcessPoolExecutor(48) as ex:
        rows = list(ex.map(b2._eval_matrix, [(n, str(EVAL)) for n in K.T1 + K.T2]))
    chain = {json.loads(l)["name"]: json.loads(l)["chain_code_bits"] for l in open(BF.D / "build_info.jsonl")}
    ref = json.load(open(RES / "b2_v2.json"))["model:256"]
    out = {"leaves": t.n_leaves, "internal_nodes": t.n_internal, "cost_bits": t.cost_bits(),
           "identical_to_original_run": bool(same)}
    for split, names in (("T1", K.T1), ("T2", K.T2)):
        sel = [r for r in rows if r["task"] in names]
        n = sum(r["n"] for r in sel)
        ch = sum(chain[r["task"]] for r in sel)
        st = sum(r["model:256"]["static"] for r in sel)
        res = {"static_gain": (ch - st) / n, "static_net_cost_over_test": (ch - st - t.cost_bits()) / n,
               "static_net_cost_over_whole_model": (ch - st) / n - t.cost_bits() / b2.ALL_W}
        res["reported"] = {k: ref[split][k] for k in res}
        res["max_abs_diff"] = max(abs(res[k] - ref[split][k]) for k in ref[split] if k in res)
        out[split] = res
    out["pass"] = all(out[s]["max_abs_diff"] < 1e-3 for s in ("T1", "T2"))
    json.dump(out, open(RES / "b2_tree_retrain_check.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def H(c):
    c = np.asarray(c, dtype=np.float64)
    n = c.sum()
    nz = c[c > 0]
    return float((nz * np.log2(n / nz)).sum()) if n > 0 else 0.0


def export():
    t = pickle.load(open(TREE, "rb"))
    z = np.load(BF.D / "b_train_features.npz")
    F, y = z["F"][:, BF.CAUSAL], z["y"].astype(np.int64)
    N = len(y)
    names = [BF.NAMES[i] for i in BF.CAUSAL]
    nn = len(t.nodes)
    # route every training position, recording the node reached at each depth
    node = np.zeros(N, dtype=np.int64)
    counts = np.zeros((nn, b2.KQ), dtype=np.int64)
    depth = {0: 0}
    parent = {0: None}
    active = np.ones(N, dtype=bool)
    counts[0] = np.bincount(y, minlength=b2.KQ)
    while active.any():
        idx = np.flatnonzero(active)
        cur = node[idx]
        nxt = cur.copy()
        for n in np.unique(cur):
            v = t.nodes[n]
            if v is None:
                active[idx[cur == n]] = False
                continue
            f, thr, l, r = v
            sel = idx[cur == n]
            go = F[sel, f] <= thr
            nxt[cur == n] = np.where(go, l, r)
            for ch, m in ((l, go), (r, ~go)):
                counts[ch] = np.bincount(y[sel[m]], minlength=b2.KQ)
                depth[ch] = depth[n] + 1
                parent[ch] = n
        node[idx] = nxt
    leaf_nodes = [n for n in range(nn) if t.nodes[n] is None and n in depth]
    assert len(leaf_nodes) == t.n_leaves
    total_bits_root = H(counts[0])

    def cond(n):
        """path from the root to node n as a list of conditions, merged into one interval per feature."""
        path = []
        c = n
        while parent[c] is not None:
            p = parent[c]
            f, thr, l, r = t.nodes[p]
            path.append((f, thr, c == l))
            c = p
        path.reverse()
        lo, hi = {}, {}
        for f, thr, left in path:
            if left:
                hi[f] = min(hi.get(f, 10 ** 9), thr)
            else:
                lo[f] = max(lo.get(f, -10 ** 9), thr + 1)
        out = []
        for f in dict.fromkeys(f for f, _, _ in path):
            a, b = lo.get(f), hi.get(f)
            txt = (f"{a} ≤ {names[f]} ≤ {b}" if a is not None and b is not None else
                   f"{names[f]} ≤ {b}" if b is not None else f"{names[f]} ≥ {a}")
            out.append({"feature": names[f], "min": a, "max": b, "text": txt})
        return out, [{"feature": names[f], "threshold": int(thr), "branch": "≤" if left else ">"} for f, thr, left in path]

    internal = []
    usage = {}
    for n in range(nn):
        v = t.nodes[n]
        if v is None or n not in depth:
            continue
        f, thr, l, r = v
        g = H(counts[n]) - H(counts[l]) - H(counts[r])
        internal.append({"node": n, "depth": depth[n], "feature": names[f], "threshold": int(thr),
                         "samples": int(counts[n].sum()), "left": l, "right": r,
                         "gain_bits": g, "gain_bits_per_training_code": g / N})
        u = usage.setdefault(names[f], {"uses": 0, "gain_bits_per_training_code": 0.0, "min_depth": 99})
        u["uses"] += 1
        u["gain_bits_per_training_code"] += g / N
        u["min_depth"] = min(u["min_depth"], depth[n])
    total_gain = sum(x["gain_bits_per_training_code"] for x in internal)
    for u in usage.values():
        u["share_of_total_gain"] = u["gain_bits_per_training_code"] / total_gain
    ranking = sorted(({"feature": k, "description": DESC[k], **v} for k, v in usage.items()),
                     key=lambda x: -x["gain_bits_per_training_code"])
    for k in names:
        if k not in usage:
            ranking.append({"feature": k, "description": DESC[k], "uses": 0, "gain_bits_per_training_code": 0.0,
                            "share_of_total_gain": 0.0, "min_depth": None})

    leaves = []
    for n in leaf_nodes:
        c = counts[n]
        lid = t.leaf_of[n]
        p = t.table[lid]
        bits = float(-(c * np.log2(p)).sum() / max(c.sum(), 1))
        leaves.append({"node": n, "leaf_id": int(lid), "depth": depth[n], "samples": int(c.sum()),
                       "share": float(c.sum() / N), "bits_per_code_with_table": bits,
                       "empirical_entropy": H(c) / max(c.sum(), 1)})
    ld = np.array([x["depth"] for x in leaves])
    ls = np.array([x["samples"] for x in leaves])
    depth_dist = {int(d): int((ld == d).sum()) for d in sorted(set(ld))}
    size_edges = [2000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10 ** 9]
    size_dist = [{"from": a, "to_below": b, "leaves": int(((ls >= a) & (ls < b)).sum()),
                  "share_of_samples": float(ls[(ls >= a) & (ls < b)].sum() / N)} for a, b in zip(size_edges[:-1], size_edges[1:])]

    def detail(x):
        merged, path = cond(x["node"])
        lid = x["leaf_id"]
        return {**x, "conditions": merged, "path": path,
                "probabilities": {s: float(p) for s, p in zip(SYMS, t.table[lid])}}

    by_size = sorted(leaves, key=lambda x: -x["samples"])[:3]
    by_bits = sorted(leaves, key=lambda x: x["bits_per_code_with_table"])[:3]
    top3 = sorted([x for x in internal if x["depth"] <= 2], key=lambda x: (x["depth"], x["node"]))
    for x in top3:
        x["condition"] = f"{x['feature']} ≤ {x['threshold']}"
        x["share_of_samples"] = x["samples"] / N
    out = {
        "source": "b2.Tree, model-wide, 256 leaves, causal v2 features, 50M training positions (b2_tree.py)",
        "training_positions": int(N),
        "root_entropy_bits_per_code": total_bits_root / N,
        "leaf_entropy_bits_per_code": sum(H(counts[n]) for n in leaf_nodes) / N,
        "total_split_gain_bits_per_training_code": total_gain,
        "features": {k: DESC[k] for k in names},
        "depth": {"max": int(ld.max()), "mean_leaf_depth": float(ld.mean()),
                  "sample_weighted_mean_leaf_depth": float((ld * ls).sum() / ls.sum()), "leaf_depth_distribution": depth_dist},
        "top3_levels": top3,
        "feature_usage_ranking": ranking,
        "leaf_size": {"min": int(ls.min()), "median": float(np.median(ls)), "max": int(ls.max()), "distribution": size_dist},
        "representative_leaves": {"largest": [detail(x) for x in by_size], "lowest_code_length": [detail(x) for x in by_bits]},
        "all_internal_nodes": internal,
        "all_leaves": leaves,
    }
    json.dump(out, open(RES / "b2_tree_structure.json", "w"), indent=1, ensure_ascii=False)
    plot(t, counts, depth, names, N)
    print(json.dumps({k: out[k] for k in ("root_entropy_bits_per_code", "leaf_entropy_bits_per_code", "depth")}, indent=1))
    for x in top3:
        print(x["depth"], x["node"], x["condition"], x["samples"], round(x["gain_bits_per_training_code"], 5))
    for r in ranking:
        print(r["feature"], r["uses"], round(r["gain_bits_per_training_code"], 5))


def plot(t, counts, depth, names, N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    ink, muted, line = "#2b2b29", "#6b6a63", "#b9b8b1"
    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    pos = {}

    def place(n, d, x0, x1):
        pos[n] = ((x0 + x1) / 2, -d)
        v = t.nodes[n]
        if d < 3 and v is not None:
            place(v[2], d + 1, x0, (x0 + x1) / 2)
            place(v[3], d + 1, (x0 + x1) / 2, x1)

    place(0, 0, 0, 1)
    for n, (x, yy) in pos.items():
        v = t.nodes[n]
        if -yy < 3 and v is not None:
            for ch, lab in ((v[2], "是"), (v[3], "否")):
                cx, cy = pos[ch]
                ax.plot([x, cx], [yy - 0.18, cy + 0.18], color=line, lw=1, zorder=1)
                ax.text((x + cx) / 2, (yy + cy) / 2, lab, fontsize=7, color=muted, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc="#fcfcfb", ec="none"))
    for n, (x, yy) in pos.items():
        v = t.nodes[n]
        share = counts[n].sum() / N * 100
        if -yy < 3 and v is not None:
            txt = f"{names[v[0]]} ≤ {v[1]}\n样本 {share:.1f}%"
            fc, ec = "#eef3fb", "#2a78d6"
        else:
            kind = "叶子" if v is None else "子树"
            txt = f"{kind}\n样本 {share:.1f}%"
            fc, ec = "#f4f3ee", muted
        ax.text(x, yy, txt, fontsize=8 if -yy < 3 else 7.5, ha="center", va="center", color=ink,
                bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=1), zorder=3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-3.5, 0.5)
    ax.axis("off")
    ax.set_title("011 B2 决策树（全模型一棵，256 叶）的前 3 层：节点为“特征 ≤ 阈值”（是 → 左），底行为第 4 层的子树或叶子及其训练样本占比",
                 fontsize=10, color=ink)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "b2_tree_top.png", facecolor=fig.get_facecolor())
    print("figure:", FIG / "b2_tree_top.png")


if __name__ == "__main__":
    {"train": train, "export": export}[sys.argv[1]]()
