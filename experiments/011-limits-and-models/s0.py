"""Spec 011 stage 0 (S0): block-internal information.

    PYTHONPATH=. python experiments/011-limits-and-models/s0.py chains    # 008 chain on T2 (T1 from 010)
    PYTHONPATH=. python experiments/011-limits-and-models/s0.py stats     # S0-1 block max-level statistics
    PYTHONPATH=. python experiments/011-limits-and-models/s0.py synth     # V-a / V-b for R9b
    PYTHONPATH=. python experiments/011-limits-and-models/s0.py r9b       # S0-2 on T1 and T2

R9b = one more context level appended after the chain's own levels (after the R10 merge, i.e. on the 15
merged symbols), with the chain's hierarchical smoothing (hier_bits) and one switch bit per matrix, as for
R2-R8. Features of "the part of the current 32-block already decoded":
  max  : largest magnitude level so far in the block (9 states: none yet, 0..7)
  cnt  : number of codes with |value| >= 4 so far in the block (0, 1, 2, 3+)
  pos4 : position in the block in 4 segments of 8;  pos32: exact position (extra variant)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

sys.path.insert(0, str(K.ROOT / "experiments" / "010-transforms"))
import p1b  # noqa: E402

from redundancy import increment as I  # noqa: E402

RES = K.RES / "s0"
B = 32
LEVELS = ["0", "0.5", "1", "1.5", "2", "3", "4", "6"]
COMBOS = {"max": ["max"], "cnt": ["cnt"], "pos4": ["pos4"], "max+pos4": ["max", "pos4"], "cnt+pos4": ["cnt", "pos4"],
          "max+cnt": ["max", "cnt"], "max+cnt+pos4": ["max", "cnt", "pos4"], "max+cnt+pos32": ["max", "cnt", "pos32"],
          "mean (extra)": ["mean"], "mean+pos4 (extra)": ["mean", "pos4"], "mean+max (extra)": ["mean", "max"]}
SEQ = [["max"], ["max", "cnt"], ["max", "cnt", "pos4"]]     # added one after another


def block_feats(C):
    """causal block features from the codes (sign of zero irrelevant: magnitudes only)."""
    R, W = C.shape
    lv = (C & 7).reshape(R, W // B, B)
    mx = np.maximum.accumulate(lv, axis=2)
    prev_max = np.zeros_like(lv)
    prev_max[:, :, 1:] = mx[:, :, :-1] + 1          # 0 = nothing decoded yet in this block
    big = (lv >= 6).astype(np.int64)
    cnt = np.zeros_like(lv)
    cnt[:, :, 1:] = np.cumsum(big, axis=2)[:, :, :-1]
    pos = np.broadcast_to(np.arange(B), lv.shape)
    # extra (not in the spec): mean magnitude level of the codes decoded so far in the block, 8 bins + none
    csum = np.zeros(lv.shape, dtype=np.float64)
    csum[:, :, 1:] = np.cumsum(lv, axis=2)[:, :, :-1]
    mean = csum / np.maximum(pos, 1)
    mbin = np.where(pos == 0, 0, 1 + np.digitize(mean, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]))
    return {"mean": (mbin.reshape(R, W), 9),
            "max": (prev_max.reshape(R, W), 9), "cnt": (np.minimum(cnt, 3).reshape(R, W), 4),
            "pos4": ((pos // 8).reshape(R, W), 4), "pos32": (pos.reshape(R, W).copy(), 32)}


def combine(feats, keys):
    f = np.zeros(feats[keys[0]][0].shape, dtype=np.int64)
    for k in keys:
        a, n = feats[k]
        f = f * n + a
    return f.ravel()


def place(levels, flags, f, where):
    """where = 'last': finest level after all chain levels; 'after_R2': nested right after the scale context
    (before R3 etc.), so R9b is learned per scale value and the later levels refine it."""
    if where == "last":
        return levels + [f]
    i = 2 if "R2" in flags else 1
    return levels[:i] + [f] + levels[i:]


def r9b_bits(C, E, flags, partner, combos, where="after_R2", seq=True):
    sym, levels, Kk, side, scale = p1b.chain_levels(C, E, flags, partner)
    base = I.hier_bits(sym, levels, Kk, p1b.P["beta"], p1b.P["nblocks"])
    feats = block_feats(C)
    out = {"base_code_bits": base, "n": int(C.size), "where": where}
    for name, keys in combos.items():
        out[name] = I.hier_bits(sym, place(levels, flags, combine(feats, keys), where), Kk, p1b.P["beta"], p1b.P["nblocks"])
    if seq:
        out["sequential"] = [I.hier_bits(sym, place(levels, flags, combine(feats, keys), where), Kk, p1b.P["beta"],
                                         p1b.P["nblocks"]) for keys in SEQ]
    return out


# ------------------------------------------------------------------ S0-1

def stats_one(name):
    C, E = K.read(name)
    R, W = C.shape
    lv = (C & 7).reshape(R, W // B, B)
    bm = lv.max(axis=2)
    Eb = E.astype(np.int64)
    hist = np.bincount(bm.ravel(), minlength=8)
    exc = bm < 6
    allzero = bm == 0
    emin = int(Eb.min())
    return {"task": name, "blocks": int(bm.size), "max_level_hist": hist.tolist(),
            "share_max_4_or_6": float((~exc).mean()), "all_zero_blocks": int(allzero.sum()),
            "exceptions_by_max_level": np.bincount(bm[exc], minlength=8).tolist(),
            "exceptions_at_matrix_min_scale": int((exc & (Eb == emin)).sum()), "matrix_min_scale": emin,
            "scale_median": float(np.median(Eb)), "exception_scale_median": float(np.median(Eb[exc])) if exc.any() else None,
            "count_6_per_block_hist": np.bincount((lv == 7).sum(axis=2).ravel(), minlength=33).tolist()}


def stage_stats():
    sel = {"T1": K.T1, "T2": K.T2, "pool_round1": K.pool_round(1)}
    out = {}
    with ProcessPoolExecutor(48) as ex:
        for k, ns in sel.items():
            rows = list(ex.map(stats_one, ns))
            nb = sum(r["blocks"] for r in rows)
            hist = np.sum([r["max_level_hist"] for r in rows], axis=0)
            excl = np.sum([r["exceptions_by_max_level"] for r in rows], axis=0)
            out[k] = {"matrices": len(rows), "blocks": nb,
                      "share_max_4_or_6": float(hist[6:].sum() / nb),
                      "max_level_share": {LEVELS[i]: float(hist[i] / nb) for i in range(8)},
                      "exceptions": int(excl.sum()), "all_zero_blocks": int(sum(r["all_zero_blocks"] for r in rows)),
                      "exceptions_at_matrix_min_scale": int(sum(r["exceptions_at_matrix_min_scale"] for r in rows)),
                      "matrices_with_exceptions": [(r["task"], int(sum(r["exceptions_by_max_level"])),
                                                    r["exceptions_by_max_level"]) for r in rows if sum(r["exceptions_by_max_level"])],
                      "rows": rows}
            print(k, out[k]["share_max_4_or_6"], out[k]["exceptions"], out[k]["all_zero_blocks"], flush=True)
    json.dump(out, open(RES / "s0_1_stats.json", "w"), indent=1)


# ------------------------------------------------------------------ synthetic checks

def synth_matrix(rng, R, W, copy_max=0.0):
    """magnitudes iid given the block scale (V-a); copy_max > 0: at positions j >= 1 of a block the magnitude
    equals the block's running maximum with that probability (V-b). Returns codes, scales, true code length."""
    E = rng.integers(117, 123, size=(R, W // B)).astype(np.uint8)
    probs = {}
    for s in range(117, 123):
        lam = 0.6 + 0.5 * (s - 117)
        p = np.array([math.exp(-lam) * lam ** k / math.factorial(k) for k in range(8)])
        p[7] += 1 - p.sum()
        probs[s] = p
    mag = np.zeros((R, W), dtype=np.int64)
    true_bits = 0.0
    for b in range(W // B):
        P = np.stack([probs[int(s)] for s in E[:, b]])          # (R, 8)
        cum = P.cumsum(1)
        run = np.full(R, -1)
        for j in range(B):
            c = b * B + j
            x = (rng.random(R)[:, None] > cum).sum(1).clip(0, 7)
            if copy_max > 0 and j > 0:
                cp = rng.random(R) < copy_max
                x = np.where(cp, run, x)
                pr = (1 - copy_max) * P[np.arange(R), x] + copy_max * (x == run)
            else:
                pr = P[np.arange(R), x]
            true_bits += float(-np.log2(pr).sum() + (x > 0).sum())      # +1 bit: sign of a non-zero code
            mag[:, c] = x
            run = np.maximum(run, x)
    sign = rng.integers(0, 2, size=(R, W))
    C = np.where(mag > 0, mag + 8 * sign, 0)
    return C, E, true_bits


def stage_synth():
    rng = np.random.default_rng(31)
    out = {}
    for kind, cm in (("V-a", 0.0), ("V-b", 0.5)):
        C, E, true_bits = synth_matrix(rng, 512, 4096, cm)
        ch = K.tx.chain(C, E)
        flags = p1b.active(ch)
        n = C.size
        out[kind] = {"flags": sorted(flags), "true_entropy_bits_per_weight": true_bits / n}
        for where in ("last", "after_R2"):
            r = r9b_bits(C, E, flags, None, {"max+cnt+pos4": COMBOS["max+cnt+pos4"]}, where, seq=False)
            gain = (r["base_code_bits"] - min(r["base_code_bits"], r["max+cnt+pos4"]) - 1) / n
            out[kind]["base_bits_per_weight"] = r["base_code_bits"] / n
            out[kind]["theory_gain_bits_per_weight"] = (r["base_code_bits"] - true_bits) / n
            out[kind]["r9b_gain_" + where] = gain
        out[kind]["r9b_gain_bits_per_weight"] = out[kind]["r9b_gain_after_R2"]
        print(kind, out[kind], flush=True)
    a, b = out["V-a"], out["V-b"]
    out["checks"] = {"V-a_about_0": bool(abs(a["r9b_gain_bits_per_weight"]) < 0.002),
                     "V-b_recovers": bool(0.8 * b["theory_gain_bits_per_weight"] < b["r9b_gain_bits_per_weight"]
                                          <= b["theory_gain_bits_per_weight"] + 0.002)}
    out["note"] = ("theory gain = base chain code length - exact code length of the generator (sign: 1 bit per "
                   "non-zero code); the chain's own learning cost is part of it, so the gain should be a bit below it")
    out["all_pass"] = all(out["checks"].values())
    json.dump(out, open(RES / "synthetic.json", "w"), indent=1)
    print(out["checks"], out["all_pass"])


# ------------------------------------------------------------------ S0-2

def r9b_one(name):
    chains = K._load_chains()
    ch = chains[name]
    C, E = K.read(name)
    pn = K.partner_of(name)
    part = K.read(pn)[0] if pn else None
    r = r9b_bits(C, E, p1b.active(ch), part, COMBOS, "after_R2")
    last = r9b_bits(C, E, p1b.active(ch), part, {"max+cnt+pos4": COMBOS["max+cnt+pos4"]}, "last", seq=False)
    r["last_max+cnt+pos4"] = last["max+cnt+pos4"]
    r["task"] = name
    return r


def stage_r9b():
    rows = []
    with ProcessPoolExecutor(48) as ex:
        for r in ex.map(r9b_one, K.T1 + K.T2):
            rows.append(r)
            print(r["task"], round((r["base_code_bits"] - r["max+cnt+pos4"]) / r["n"], 5), flush=True)
    with open(RES / "s0_2_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = {}
    for split, ns in (("T1", K.T1), ("T2", K.T2)):
        sel = [r for r in rows if r["task"] in ns]
        n = sum(r["n"] for r in sel)
        base = sum(r["base_code_bits"] for r in sel)
        # per matrix: the encoder switches R9b on only if it shortens the code (1 switch bit, as R2-R8)
        res = {c: (base - sum(min(r["base_code_bits"], r[c]) + 1 for r in sel)) / n for c in COMBOS}
        seq = []
        prev = base
        for i in range(len(SEQ)):
            cur = sum(min(r["base_code_bits"], r["sequential"][i]) for r in sel) + len(sel)
            seq.append((prev - cur) / n if i == 0 else (prev - cur) / n)
            prev = cur
        out[split] = {"matrices": len(sel), "chain_code_bits_per_code": base / n,
                      "gain_bits_per_weight": res, "gain_percent_of_4_25": {c: 100 * v / 4.25 for c, v in res.items()},
                      "sequential_increments": dict(zip(["+max", "+cnt", "+pos4"], seq)),
                      "by_role": {role: (sum(r["base_code_bits"] - min(r["base_code_bits"], r["max+cnt+pos4"]) - 1
                                             for r in sel if r["task"].endswith(role)) /
                                         sum(r["n"] for r in sel if r["task"].endswith(role))) for role in K.ROLES}}
    out["p1b_offline_gain_T1"] = 0.05034066452015109
    out["share_of_p1b_recovered_T1"] = out["T1"]["gain_bits_per_weight"]["max+cnt+pos4"] / out["p1b_offline_gain_T1"]
    json.dump(out, open(RES / "s0_2_r9b.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def stage_chains():
    K.ensure_chains(K.T1 + K.T2)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    {"chains": stage_chains, "stats": stage_stats, "synth": stage_synth, "r9b": stage_r9b}[sys.argv[1]]()


if __name__ == "__main__":
    main()
