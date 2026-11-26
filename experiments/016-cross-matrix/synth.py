"""Spec 016: synthetic checks of every detector (planted regularities must be found; independent data must match
the shuffled control). Run before any real data.

    PYTHONPATH=.:experiments/016-cross-matrix python experiments/016-cross-matrix/synth.py
"""
import argparse
import json
import math
import time

import numpy as np

import common as K

# a realistic code distribution (roughly that of an FP4 expert matrix)
P16 = np.array([10, 16, 14, 8, 6, 3, 1.5, .5, 10, 16, 14, 8, 6, 3, 1.5, .5]); P16 = P16 / P16.sum()
RESULTS = {}


def check(name, ok, info):
    RESULTS[name] = {"pass": bool(ok), **info}
    print(("PASS " if ok else "FAIL ") + name, json.dumps(info)[:600], flush=True)


def rand_codes(rng, R, W):
    return rng.choice(16, size=(R, W), p=P16).astype(np.uint8)


def noisy(rng, C, frac):
    C = C.copy()
    m = rng.random(C.shape) < frac
    C[m] = rand_codes(rng, 1, int(m.sum()))[0] if m.sum() else C[m]
    return C


# ------------------------------------------------------------------ part A
def test_pairs():
    import torch
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    import pairs as PA
    rng = np.random.default_rng(1)
    R, W = 256, 512
    A = rand_codes(rng, R, W)
    Ea = rng.integers(120, 125, size=(R, W // 32)).astype(np.uint8)
    Eaf = np.repeat(Ea, 32, axis=1)
    dy, dx = 5, -37
    Bs = np.roll(A, (dy, dx), axis=(0, 1))            # B[i,j] = A[i-dy, j-dx]  ->  A[i+5... ] see below
    Bs = noisy(rng, Bs, 0.3)
    Ind = rand_codes(rng, R, W)
    Sa = K.gpu_signals(torch, A, Eaf)
    for name, B in (("shifted_copy", Bs), ("independent", Ind)):
        Sb = K.gpu_signals(torch, B, Eaf)
        sp = PA.same_position(torch, A, B, Eaf, Ea, Ea)
        gs = PA.global_shift(torch, Sa, Sb, A, B, False)
        if name == "shifted_copy":
            # B[i, j] = A[i - dy, j - dx]  =>  correlation peak at shift (-dy, -dx) in c[s] = sum A[i+s] B[i]
            ok = gs["eq_argmax_shift"] == [-dy, -dx] and gs["gain_code_at_best_shift_per_code"] > 0.3 and abs(sp["gain_code"]) < 0.01
            check("A.global_shift finds planted shift", ok, {"argmax": gs["eq_argmax_shift"], "want": [-dy, -dx],
                  "gain_at_shift": gs["gain_code_at_best_shift_per_code"], "gain_same_pos": sp["gain_code"],
                  "z": gs["eq_max_z_nonzero_shift"], "v_shift": gs["v_argmax_shift"], "sign_shift": gs["sign_argmax_shift"]})
        else:
            ok = gs["gain_code_at_best_shift_per_code"] < 0 and sp["gain_code"] < 0.002 and gs["eq_max_z_nonzero_shift"] < 7
            check("A.independent pair shows nothing", ok, {"gain_at_shift": gs["gain_code_at_best_shift_per_code"],
                  "gain_same_pos": sp["gain_code"], "z": gs["eq_max_z_nonzero_shift"]})
    # same position: B = A with 50% noise
    Bp = noisy(rng, A, 0.5)
    sp = PA.same_position(torch, A, Bp, Eaf, Ea, Ea)
    check("A.same_position detects co-located copy", sp["gain_code"] > 0.5, {"gain": sp["gain_code"], "gain_sign": sp["gain_bit_sign"]})
    # local: copy a 32x32 and a 128x128 region of A into an independent matrix at other places, 20% noise
    D = Ind.copy()
    D[10:42, 300:332] = noisy(rng, A[100:132, 40:72], 0.2)
    D[120:248, 0:128] = noisy(rng, A[3:131, 350:478], 0.2)
    Sd = K.gpu_signals(torch, D, Eaf)
    # monkeypatch tile positions: use the library's random tiles plus check the planted ones directly
    old = PA.TILES
    PA.TILES = [(32, 64), (128, 8)]

    class FixedRng:
        def __init__(self, pos):
            self.pos = list(pos)

        def integers(self, lo, hi):
            return self.pos.pop(0)
    rr = np.random.default_rng(3)
    pos = [10, 300] + sum([[int(rr.integers(0, 80)), int(rr.integers(140, 260))] for _ in range(63)], []) + [120, 0] + [int(x) for x in np.random.default_rng(4).integers(0, 120, 14)]
    loc = PA.local_search(torch, Sa, [Sd, Sd], A, [D, D], False, FixedRng(pos))
    t32, t128 = loc["T32_real"][0], loc["T128_real"][0]
    others = [r["eq_max"] for r in loc["T32_real"][1:]]
    ok = t32["eq_at"] == [100, 40] and t128["eq_at"] == [3, 350] and t32["evidence_bits"] > 0 and max(others) < t32["eq_max"] \
        and max(r["evidence_bits"] for r in loc["T32_real"][1:]) < 20
    check("A.local_search finds planted tiles", ok, {"t32": t32, "t128": t128, "max_other_eq": max(others),
          "max_other_evidence": max(r["evidence_bits"] for r in loc["T32_real"][1:])})
    PA.TILES = old
    # self pair: plant an internal translation copy
    S = rand_codes(rng, R, W)
    S[150:182, 400:432] = S[20:52, 60:92]
    Ss = K.gpu_signals(torch, S, Eaf)
    pos = [150, 400] + [int(x) for x in np.random.default_rng(5).integers(0, 200, 126)] + [0, 0] * 8
    PA.TILES = [(32, 64), (128, 8)]
    loc = PA.local_search(torch, Ss, [Ss, Ss], S, [S, S], True, FixedRng(pos))
    PA.TILES = old
    check("A.local_search self pair finds internal copy, excludes own position", loc["T32_real"][0]["eq_at"] == [20, 60]
          and loc["T32_real"][0]["eq_max"] == 1.0, {"t": loc["T32_real"][0]})


# ------------------------------------------------------------------ part B (synthetic model)
def synth_model(rng):
    ents, data = [], {}
    for L in range(4):
        for e in range(6):
            for role in ("w1", "w3", "w2"):
                R, W = (256, 512) if role != "w2" else (512, 256)
                C = rand_codes(rng, R, W)
                E = rng.integers(115, 131, size=(R, W // 32)).astype(np.uint8)
                name = f"L{L}.E{e}.{role}"
                ents.append({"matrix": name, "layer": L, "expert": e, "role": role, "codeword_shape": [R, W]})
                data[name] = [C, E]
    return ents, data


def install(ents, data):
    K._ENTRIES["syn"] = ents
    K.MODELS["syn"] = dict(env="", label="synthetic")

    def load(model, e):
        C, E = data[e["matrix"]]
        return C.copy(), E.copy()

    def load_packed(model, e):
        C, E = data[e["matrix"]]
        return ((C[:, 0::2] & 15) | (C[:, 1::2] << 4)).astype(np.uint8), E.copy()
    K.load = load
    K.load_packed = load_packed


def test_hash(tmp):
    import scan as S
    rng = np.random.default_rng(11)
    ents, data = synth_model(rng)
    names = [e["matrix"] for e in ents]
    # plants
    data["L1.E2.w1"][0][17] = data["L0.E0.w1"][0][3]                          # row copy (codes only)
    data["L2.E3.w3"][0][40, 64:96] = data["L0.E1.w3"][0][7, 128:160]           # one 32-code block, other position
    data["L3.E4.w1"][0][16:24, 32:64] = data["L1.E1.w1"][0][80:88, 448:480]    # one aligned 8x32 patch
    data["L0.E5.w2"][0][9, 0:32] = data["L0.E5.w2"][0][300, 64:96]             # within-matrix block duplicate
    data["L3.E5.w3"] = [data["L2.E0.w3"][0].copy(), data["L2.E0.w3"][1].copy()]   # whole matrix duplicate
    install(ents, data)
    K.RES = tmp; K.DATA = tmp
    S.K = K
    a = argparse.Namespace(model="syn", workers=4, limit=None, sub=1, part=0)
    S.cmd_hash(a)
    out = json.load(open(tmp / "b1_syn_s1p0.json"))["units"]
    # the same with two disjoint hash subsets: counts must add up to the full count
    o2 = []
    for part in (0, 1):
        S.cmd_hash(argparse.Namespace(model="syn", workers=4, limit=None, sub=2, part=part))
        o2.append(json.load(open(tmp / f"b1_syn_s2p{part}.json"))["units"])
    split = {u: o2[0][u]["cross_matrix"]["repeat_units"] + o2[1][u]["cross_matrix"]["repeat_units"] for u in ("block32", "patch8x32")}
    check("B1 two hash subsets add up", split == {u: out[u]["cross_matrix"]["repeat_units"] for u in split},
          {"split": split, "parts": [{u: o[u]["cross_matrix"]["repeat_units"] for u in split} for o in o2]})
    R, W = 256, 512
    want = {"matrix": 1, "row_codes": R + 1, "col_codes": W, "row_scales": R, "block32": R * W // 32 + 1 + 8 + W // 32, "patch8x32": R // 8 * W // 32 + 1}
    got = {u: out[u]["cross_matrix"]["repeat_units"] for u in want}
    check("B1 cross-matrix repeat counts", got == want, {"got": got, "want": want})
    wg = {u: out[u]["within_matrix"]["dup_units"] for u in want}
    check("B1 within-matrix duplicate counts", wg["block32"] == 1 and sum(wg.values()) == 1, {"within": wg})
    return ents, data


def test_gpu(tmp):
    import torch
    import scan as S
    rng = np.random.default_rng(21)
    ents, data = synth_model(rng)
    install(ents, data)
    K.RES = tmp; K.DATA = tmp
    a = argparse.Namespace(model="syn", qmats=16, nq_vec=32, nq_patch=[16, 16, 16], workers=4, limit=None, tag="syn", random_order=False)
    S.cmd_queries(a)
    q = dict(np.load(tmp / "queries_syn.npz"))
    names = [e["matrix"] for e in ents]
    # plant: neuron query 0 -> copy (30% noise, negated) as neuron vector 77 of another matrix (different layer)
    i0, j0 = q["neuron_meta"][0]
    src = q["neuron_C"][0]
    tgt = next(k for k, e in enumerate(ents) if e["layer"] != ents[i0]["layer"] and e["role"] == "w2")
    cpy = noisy(rng, src[None], 0.3)[0] ^ 8
    data[names[tgt]][0][:, 77] = cpy                                    # w2: neuron vectors are columns
    # hidden query 0 -> exact copy into another matrix's hidden vector 5
    ih, jh = q["hidden_meta"][0]
    tgh = next(k for k, e in enumerate(ents) if e["layer"] != ents[ih]["layer"] and e["role"] == "w1"
               and len(q["hidden_C"][0]) == 256)
    data[names[tgh]][0][:, 5] = q["hidden_C"][0]
    # patches: query 0 of each shape -> into another matrix at a fixed place, 20% noise
    plant_p = {}
    at = {"1x64": (20, 10), "8x8": (60, 100), "16x16": (120, 150)}
    for s, (h, w) in S.SHAPES.items():
        r0, c0 = at[s]
        ip, rp, cp = q[f"p{s}_meta"][0]
        tp = (ip + 7) % len(ents)
        data[names[tp]][0][r0:r0 + h, c0:c0 + w] = noisy(rng, q[f"p{s}_C"][0], 0.2)
        plant_p[s] = (int(tp), r0, c0)
    S.K = K
    S.cmd_gpu(a)
    z = np.load(tmp / "scan_syn.npz")
    ti = z["vec_neuron_top_i_v"][0, 0]
    check("B2 finds noisy negated neuron-vector copy", (ti // 100000, ti % 100000) == (tgt, 77),
          {"top": [int(ti // 100000), int(ti % 100000)], "want": [tgt, 77], "cos": float(z["vec_neuron_top_v_v"][0, 0]),
           "bits_top": float(z["vec_neuron_top_v_bits"][0, 0]),
           "null_max_cos": float(z["vec_neuron_top_v_v"][0, 32:].max()), "real_other_max_cos": float(z["vec_neuron_top_v_v"][0, 1:32].max())})
    th = z["vec_hidden_top_i_bits"][0, 0]
    check("B2 finds exact hidden-vector copy (bits = 1)", (th // 100000, th % 100000) == (tgh, 5) and abs(z["vec_hidden_top_v_bits"][0, 0] - 1) < 2e-3,
          {"top": [int(th // 100000), int(th % 100000)], "want": [tgh, 5], "bits_dot": float(z["vec_hidden_top_v_bits"][0, 0])})
    for s in S.SHAPES:
        wb = z[f"pat_{s}_where_bits_other"][0].tolist()
        check(f"B3 finds planted patch {s}", tuple(wb) == plant_p[s] and z[f"pat_{s}_best_bits_other"][0] > z[f"pat_{s}_best_bits_other"][16:].max(),
              {"where": wb, "want": list(plant_p[s]), "best_bits": float(z[f"pat_{s}_best_bits_other"][0]),
               "null_max_bits": float(z[f"pat_{s}_best_bits_other"][16:].max())})
    # self exclusion: real queries' "same" best must not be their own position
    for s, (h, w) in S.SHAPES.items():
        m = z[f"pat_{s}_meta"][:16]
        wh = z[f"pat_{s}_where_bits_same"][:16]
        own = [(abs(int(wh[k, 1]) - int(m[k, 1])) < h and abs(int(wh[k, 2]) - int(m[k, 2])) < w) for k in range(16)]
        check(f"B3 excludes own window {s}", not any(own), {"n_own": int(sum(own))})


def main():
    from pathlib import Path
    import tempfile
    t0 = time.time()
    test_pairs()
    tmp = Path(tempfile.mkdtemp(prefix="syn016_", dir="/tmp/claude-10029"))
    test_hash(tmp)
    test_gpu(tmp)
    out = {"results": RESULTS, "seconds": time.time() - t0, "all_pass": all(r["pass"] for r in RESULTS.values())}
    Path(__file__).resolve().parent.joinpath("results").mkdir(exist_ok=True)
    json.dump(out, open(Path(__file__).resolve().parent / "results/synth.json", "w"), indent=1)
    print("ALL_PASS" if out["all_pass"] else "SOME_FAIL", f"{out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
