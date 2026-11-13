"""Spec 003 section 7: unit checks U1-U3 and synthetic constructions T1-T6 (no model data).

Every construction goes through the same item functions of `r58` as the model matrices
(`r6_matrix`, `r63_pair`, `rx_pair`, `r5_matrix`, `r71_matrix`, `r72_r82_matrix`, `r73`).
Shape 1024 x 2048, B = 32, constant scale 127 (so S0E == S0 and only the v domain is needed).
Usage: python -m diagnostics.r58_check [--out experiments/003-r5-r8/validation.json] [--jobs 12]
"""
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from . import est, r58  # noqa: E402
from .ledger import merge_zero  # noqa: E402
from .ledger_check import gen_U3  # noqa: E402
from .synth import code_probs, quantize  # noqa: E402

R, W, B = 1024, 2048, 32
SEED = 20261113
DOM = ("v",)


def chk(name, value, criterion, ok):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    return {"check": name, "value": value, "criterion": criterion, "pass": bool(ok)}


def base(rng, shape=(R, W)):
    return quantize(1.68 * rng.standard_normal(shape))


def const_E(shape=(R, W)):
    return np.full((shape[0], shape[1] // B), 127, dtype=np.uint8)


def rows_obj(name):
    return r58.Rows({"matrix": name, "layer": "", "expert": "", "role": "", "matrix_b": "",
                     "pair_group": ""})


def find(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


# ----------------------------------------------------------------------------- units

def units():
    out = []
    rng = np.random.default_rng(SEED + 1)
    C, E = gen_U3(rng)
    Cp = merge_zero(C)
    X, E2 = r58.shuffle(Cp, E, "S0E", rng)
    ok = True
    for e in np.unique(E):
        m = np.repeat(E == e, B, axis=1)
        ok &= np.array_equal(np.bincount(Cp[m], minlength=16), np.bincount(X[m], minlength=16))
    moved = float((X != Cp).mean())
    out.append(chk("U1 S0E keeps every scale group's histogram", {"n_scale_values":
               int(len(np.unique(E))), "fraction_positions_changed": moved}, "exact; >=2 scales",
               ok and len(np.unique(E)) >= 2 and moved > 0.5 and np.array_equal(E, E2)))
    Cp = merge_zero(base(rng))
    E = const_E()
    X, _ = r58.shuffle(Cp, E, "S2", rng)
    ok2 = np.array_equal(np.sort(Cp.reshape(-1, B), axis=1), np.sort(X.reshape(-1, B), axis=1))
    X3, E3 = r58.shuffle(Cp, E, "S3", rng)
    ok3 = sorted(map(bytes, Cp)) == sorted(map(bytes, X3)) and not np.array_equal(X3, Cp)
    out.append(chk("U2 S2 keeps each block multiset; S3 keeps each row", {"S2": bool(ok2),
               "S3": bool(ok3)}, "exact", ok2 and ok3))
    X4, _ = r58.shuffle(Cp, E, "S4", rng)
    ok4 = np.array_equal(X4 & 7, Cp & 7) and bool(np.all(X4[(Cp & 7) == 0] == 0))
    flips = float(((X4 >> 3) != (Cp >> 3))[(Cp & 7) > 0].mean())
    out.append(chk("U3 S4 keeps m at every position; m=0 stays C'=0",
                   {"sign_flip_fraction_nonzero": flips}, "exact", ok4 and 0.45 < flips < 0.55))
    return out


# ----------------------------------------------------------------------------- constructions

def t1():
    rng = np.random.default_rng(SEED + 10)
    A, Bm = base(rng), base(rng)
    E = const_E()
    Cp = merge_zero(A)
    rows = rows_obj("T1")
    H_E = r58.h_cond_e(Cp, E)
    v6, _, _ = r58.r6_matrix(Cp, E, {"matrix": "T1"}, rows, 1, H_E)
    rx = r58.rx_pair(Cp, E, merge_zero(Bm), E, rows, 1)
    r5, _ = r58.r5_matrix(Cp, E, rows, 1, DOM, H_E)
    r71 = r58.r71_matrix(Cp, E, rows, 1, H_E)
    r7282 = r58.r72_r82_matrix(Cp, E, rows, 1)
    X = {"v": r58.values(Cp, E, "v")}
    C4 = r58.shuffle(Cp, E, "S4", r58.item_rng(1, "R7.3"))[0]
    r73, _ = r58.r73(X, X, {"v": r58.values(C4, E, "v")}, rows, True, {})
    out = []
    out.append(chk("R6.1 block repeats", v6["r61"]["block_repeats"], "== 0",
                   v6["r61"]["block_repeats"] == 0))
    ks = v6["r62"]["Kstar"]
    out.append(chk("R6.2 dup(16), K*", {"dup16": v6["r62"]["dup"][16], "Kstar": ks},
                   "dup16 == 0 and K* <= 14", v6["r62"]["dup"][16] == 0 and (ks is None or ks <= 14)))
    sig = {r["variable"]: r["significant"] for r in find(rows.rows, item="RX.1", baseline="S0")}
    out.append(chk("RX.1 vs S0 significant?", sig, "all False", not any(sig.values())))
    out.append(chk("R5.1 exceed layers", r5["v"]["exceed"], "== 0", r5["v"]["exceed"] == 0))
    r2 = r5["v"]["r2"]
    out.append(chk("R5.2 hold-out R2 (all k)", r2, "all < 1e-4", max(r2.values()) < 1e-4))
    d = abs(r71["real"]["median"] - r71["S0E"]["median"])
    out.append(chk("R7.1 |median best real - S0E|", d, "<= 1", d <= 1))
    out.append(chk("R7.1 fraction best >= 18", r71["real"]["frac18"], "== 0",
                   r71["real"]["frac18"] == 0))
    r72 = r7282["r72"]
    diffs, nets = {}, {}
    for dom in ("v", "absv"):
        for K in r58.KM_K:
            s0e = np.mean([r72[f"S0E_{i}"][dom][K]["ratio"] for i in (1, 2)])
            diffs[f"{dom},K={K}"] = r72["real"][dom][K]["ratio"] - s0e
            if K <= r58.KM_STRICT_MAX:
                for var in ("net_cls", "net_clsp"):
                    nets[f"{dom},K={K},{var}"] = r72["real"][dom][K][var]
    out.append(chk("R7.2 distortion ratio real - S0E", diffs, "|.| < 0.005 for every K",
                   max(abs(x) for x in diffs.values()) < 0.005))
    out.append(chk("R7.2 strict nets (real)", nets, "all <= 0", max(nets.values()) <= 0))
    out.append(chk("R7.3 rows above 1.2 x S4 max", {"rows": r73["v"]["row"]["n_exceed"],
               "cols (info)": r73["v"]["col"]["n_exceed"]}, "rows == 0",
                   r73["v"]["row"]["n_exceed"] == 0))
    r81 = r5["v"]["r81"]
    out.append(_r81_zero(r81))
    r82 = r7282["r82"]
    dd = {s: r82["real"]["dict"][s] - np.mean([r82[f"S0E_{i}"]["dict"][s] for i in (1, 2)])
          for s in r58.DICT_S}
    out.append(chk("R8.2 R2_dict real - S0E", dd, "|.| < 0.01", max(abs(x) for x in dd.values()) < 0.01))
    return "T1", out


def _r81_zero(r81):
    keys = ("left_exceed", "right_exceed", "left_exceed_spec", "right_exceed_spec")
    return chk("R8.1 exceed layers (adjusted and spec rule)", {k: r81[k] for k in keys},
               "all == 0", all(r81[k] == 0 for k in keys))


def t2a():
    rng = np.random.default_rng(SEED + 21)
    C = base(rng)
    nb = W // B
    pos = rng.choice(R * nb, size=200, replace=False)
    blocks = C.reshape(-1, B)
    blocks[pos[100:]] = blocks[pos[:100]]
    E = const_E()
    Cp = merge_zero(C)
    rows = rows_obj("T2a")
    v6, _, _ = r58.r6_matrix(Cp, E, {"matrix": "T2a"}, rows, 2, r58.h_cond_e(Cp, E))
    r = v6["r61"]
    return "T2a", [chk("R6.1 block repeats (real, S2)", [r["block_repeats"], r["S2"]],
                       "real == 100, S2 == 0", r["block_repeats"] == 100 and r["S2"] == 0)]


def t2b():
    rng = np.random.default_rng(SEED + 22)
    C = base(rng)
    L = 200
    taken = {}

    def free(r, j):
        return all(j + L <= a or b <= j for a, b in taken.get(r, []))

    spots = []
    while len(spots) < 51:
        r, j = int(rng.integers(R)), int(rng.integers(W - L + 1))
        if free(r, j):
            taken.setdefault(r, []).append((j, j + L))
            spots.append((r, j))
    (r0, j0), targets = spots[0], spots[1:]
    seg = C[r0, j0:j0 + L].copy()
    for r, j in targets:
        C[r, j:j + L] = seg
    E = const_E()
    Cp = merge_zero(C)
    rows = rows_obj("T2b")
    v6, _, _ = r58.r6_matrix(Cp, E, {"matrix": "T2b"}, rows, 3, r58.h_cond_e(Cp, E))
    g = v6["r62"]
    # Account for every repeated 64-window (same functions as r6_matrix): windows fully inside the
    # 51 copies give exactly 50 x 137 = 6850; any other repeat must overlap two copies at the same
    # offset (its out-of-copy codes coincide by chance across the copies).
    k = 64
    words, n = r58.window_words(r58.pack16(Cp), k)
    dup, (fa, fb) = r58.dup_windows(words, want_pairs=True)
    start = {}
    for r, j in spots:
        start.setdefault(r, []).append(j)

    def offset(idx):
        r, j = divmod(int(idx), n)
        for s0 in start.get(r, []):
            if s0 - k < j < s0 + L:
                return j - s0
        return None

    inside = edge = other = 0
    for a, b in zip(fa, fb):
        oa, ob = offset(a), offset(b)
        if oa is not None and oa == ob and 0 <= oa <= L - k:
            inside += 1
        elif oa is not None and oa == ob:
            edge += 1
        else:
            other += 1
    return "T2b", [
        chk("R6.2 dup(64), K*, S2 dup(64) (spec range recorded)",
            {"dup64": g["dup"][64], "Kstar": g["Kstar"], "S2_dup64": g["dup_S2"][64],
             "spec_range_[6500,6850]": 6500 <= g["dup"][64] <= 6850},
            "K* == 64, S2 dup64 == 0", g["Kstar"] == 64 and g["dup_S2"][64] == 0),
        chk("R6.2 dup(64) accounting (adjusted criterion)",
            {"inside_copies": inside, "copy_edge_same_offset": edge, "elsewhere": other,
             "total": dup},
            "inside == 6850, elsewhere == 0, total == dup(64) of R6.2",
            inside == 50 * (L - k + 1) and other == 0 and dup == g["dup"][64])]


def t3():
    rng = np.random.default_rng(SEED + 30)
    A = base(rng)
    fresh = base(rng)
    Bm = np.where(rng.random((R, W)) < 0.8, A, fresh).astype(np.uint8)
    E = const_E()
    q = code_probs(1.68)
    p = q.copy()
    p[0] += p[8]
    p[8] = 0.0
    p = p[p > 0]
    joint = 0.8 * np.diag(p) + 0.2 * np.outer(p, p)
    theory = est.mi_table(joint * 1e12)[0]
    rows = rows_obj("T3")
    rx = r58.rx_pair(merge_zero(A), E, merge_zero(Bm), E, rows, 4)
    mi, n, dof = rx["real"]["I(C'_A;C'_B)"]
    s0 = [b["I(C'_A;C'_B)"][0] for b in rx["base"]["S0"]]
    s3 = [b["I(C'_A;C'_B)"][0] for b in rx["base"]["S3"]]
    s2 = [b["I(C'_A;C'_B)"][0] for b in rx["base"]["S2"]]
    rel = abs(mi - theory) / theory
    net3 = float(np.mean(s3) - np.mean(s0))
    sig3 = r58.mi_significant(net3, dof, n, s0)
    return "T3", [
        chk("RX.1 I(C'_A;C'_B) relative error", {"value": mi, "theory": theory, "rel": rel},
            "< 1%", rel < 0.01),
        chk("RX.1 under S3 (mean S3 - mean S0, section-4 rule)", {"net": net3, "significant": sig3},
            "not significant", sig3 is False),
        chk("RX.1 under S2 / real", float(np.mean(s2)) / mi, "< 0.01", np.mean(s2) < 0.01 * mi)]


def _lowrank(sparse, seed):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((R, 4))
    if sparse:
        U = np.where(rng.random((R, 4)) < 0.9, 0.0, U / math.sqrt(0.1))
    V = rng.standard_normal((W, 4))
    y = 1.68 * (math.sqrt(0.1) * (U @ V.T) / 2 + math.sqrt(0.9) * rng.standard_normal((R, W)))
    return quantize(y)


def t4():
    Cp = merge_zero(_lowrank(False, SEED + 40))
    E = const_E()
    rows = rows_obj("T4")
    r5, _ = r58.r5_matrix(Cp, E, rows, 5, DOM, r58.h_cond_e(Cp, E))
    v = r5["v"]
    r2 = v["r2"]
    kb = max(r2, key=r2.get)
    r53 = v["r53"]
    return "T4", [
        chk("R5.1 exceed layers", v["exceed"], "== 4", v["exceed"] == 4),
        chk("R5.2 argmax k and R2(4)", {"argmax_k": kb, "R2(4)": r2[4], "R2": r2},
            "argmax k == 4, R2(4) in [0.08, 0.11]", kb == 4 and 0.08 <= r2[4] <= 0.11),
        chk("R5.3 gain / approx", None if r53 is None else {"gain": r53["gain"],
            "approx": r53["approx"], "ratio": r53["ratio"]}, "in [0.7, 1.1]",
            r53 is not None and 0.7 <= r53["ratio"] <= 1.1),
        _r81_zero(v["r81"])]


def t4b():
    Cp = merge_zero(_lowrank(True, SEED + 41))
    E = const_E()
    rows = rows_obj("T4b")
    r5, _ = r58.r5_matrix(Cp, E, rows, 6, DOM, r58.h_cond_e(Cp, E))
    v = r5["v"]
    r81 = v["r81"]
    return "T4b", [
        chk("R5.1 exceed layers", v["exceed"], "== 4", v["exceed"] == 4),
        chk("R8.1 exceed layers (adjusted rule 2 x S4 median; spec rule 2 x S4 max recorded)",
            {k: r81[k] for k in ("left_exceed", "right_exceed", "left_thr", "right_thr",
                                 "left_exceed_spec", "right_exceed_spec", "left_thr_spec",
                                 "left_max", "left_s4_max")},
            "left == 4, right == 0 (adjusted rule)",
            r81["left_exceed"] == 4 and r81["right_exceed"] == 0)]


def t5():
    rng = np.random.default_rng(SEED + 50)
    C = base(rng)
    protos = base(rng, (16, B))
    blocks = C.reshape(-1, B)
    nbk = len(blocks)
    isp = rng.random(nbk) < 0.3
    idx = np.nonzero(isp)[0]
    pb = protos[rng.integers(16, size=len(idx))]
    redraw = rng.random(pb.shape) < 0.3
    pb = np.where(redraw, base(rng, pb.shape), pb)
    blocks[idx] = pb
    E = const_E()
    Cp = merge_zero(C)
    rows = rows_obj("T5")
    H_E = r58.h_cond_e(Cp, E)
    r71 = r58.r71_matrix(Cp, E, rows, 7, H_E)
    r72 = r58.r72_r82_matrix(Cp, E, rows, 7)["r72"]
    K = 32
    real = r72["real"]["v"][K]
    s2 = [r72[f"S2_{i}"]["v"][K] for i in (1, 2)]
    return "T5", [
        chk("R7.1 fraction best >= 18 (real, S2)", [r71["real"]["frac18"], r71["S2"]["frac18"]],
            "real >= 0.2, S2 <= 0.01", r71["real"]["frac18"] >= 0.2 and r71["S2"]["frac18"] <= 0.01),
        chk("R7.2 v K=32 nets (real)", {"net_clsp": real["net_clsp"], "net_cls": real["net_cls"]},
            "net_clsp > 0.15, net_cls <= 0", real["net_clsp"] > 0.15 and real["net_cls"] <= 0),
        chk("R7.2 v K=32 nets (S2, 2 reps)", [[x["net_cls"], x["net_clsp"]] for x in s2],
            "all <= 0", all(x["net_cls"] <= 0 and x["net_clsp"] <= 0 for x in s2))]


def t5r():
    rng = np.random.default_rng(SEED + 51)
    Z = rng.standard_normal((R, W))
    pick = rng.choice(R, size=40, replace=False)
    dst, src = pick[:20], pick[20:]
    for d, s in zip(dst, src):
        Z[d] = 0.9 * Z[s] + math.sqrt(0.19) * rng.standard_normal(W)
    Cp = merge_zero(quantize(1.68 * Z))
    E = const_E()
    rows = rows_obj("T5r")
    X = {"v": r58.values(Cp, E, "v")}
    C4 = r58.shuffle(Cp, E, "S4", r58.item_rng(8, "R7.3"))[0]
    r73, ex = r58.r73(X, X, {"v": r58.values(C4, E, "v")}, rows, True, {})
    want = {frozenset((int(d), int(s))) for d, s in zip(dst, src)}
    corr = [e["corr"] for e in ex if e["axis"] == "row" and
            frozenset((e["index_a"], e["index_b"])) in want]
    med = float(np.median(corr)) if corr else float("nan")
    return "T5r", [
        chk("R7.3 rows above 1.2 x S4 max", r73["v"]["row"]["n_exceed"], "== 40",
            r73["v"]["row"]["n_exceed"] == 40),
        chk("R7.3 median corr of the 20 planted pairs", {"median": med, "found": len(corr)},
            "in [0.85, 0.92] (all 20 found)", len(corr) == 20 and 0.85 <= med <= 0.92)]


def t6():
    rng = np.random.default_rng(SEED + 60)
    atoms = rng.standard_normal((48, B))
    atoms /= np.sqrt((atoms * atoms).sum(axis=1, keepdims=True))
    nbk = R * W // B
    a1 = rng.integers(48, size=nbk)
    a2 = (a1 + rng.integers(1, 48, size=nbk)) % 48
    c = rng.standard_normal((nbk, 2))
    sig = 4 * (c[:, :1] * atoms[a1] + c[:, 1:] * atoms[a2])
    y = 1.68 * (math.sqrt(0.6) * sig + math.sqrt(0.4) * rng.standard_normal((nbk, B)))
    Cp = merge_zero(quantize(y).reshape(R, W))
    E = const_E()
    rows = rows_obj("T6")
    r82 = r58.r72_r82_matrix(Cp, E, rows, 9)["r82"]
    s0e = np.mean([r82[f"S0E_{i}"]["dict"][2] for i in (1, 2)])
    s2 = np.mean([r82[f"S2_{i}"]["dict"][2] for i in (1, 2)])
    real = r82["real"]["dict"][2]
    return "T6", [chk("R8.2 R2_dict(2): real - S0E, |S2 - S0E|", {"real": real, "S0E": s0e, "S2": s2},
                      "real - S0E > 0.2 and |S2 - S0E| < 0.03", real - s0e > 0.2 and abs(s2 - s0e) < 0.03)]


TESTS = {"T1": t1, "T2a": t2a, "T2b": t2b, "T3": t3, "T4": t4, "T4b": t4b, "T5": t5, "T5r": t5r,
         "T6": t6}


def _run(name):
    return TESTS[name]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/003-r5-r8/validation.json")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    names = args.only or list(TESTS)
    out = {"shape": [R, W], "B": B, "seed": SEED, "checks": {}}
    if not args.only:
        out["checks"]["units"] = units()
    with ProcessPoolExecutor(min(args.jobs, len(names))) as ex:
        for name, res in ex.map(_run, names):
            out["checks"][name] = res
    out["all_pass"] = all(c["pass"] for v in out["checks"].values() for c in v)
    for k, v in out["checks"].items():
        for c in v:
            print(k, "PASS" if c["pass"] else "FAIL", c["check"], json.dumps(c["value"], default=str)[:300])
    print("all_pass", out["all_pass"])
    if not args.only:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
