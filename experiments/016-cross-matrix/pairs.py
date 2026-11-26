"""Spec 016, part A: pairwise full-matrix and local tests on stratified matrix pairs drawn from the whole model.

    PYTHONPATH=.:experiments/016-cross-matrix python experiments/016-cross-matrix/pairs.py plan  {deepseek|kimi}
    PYTHONPATH=.:experiments/016-cross-matrix python experiments/016-cross-matrix/pairs.py gpu   {deepseek|kimi} [--limit N]
    PYTHONPATH=.:experiments/016-cross-matrix python experiments/016-cross-matrix/pairs.py comp  {deepseek|kimi} [--workers N]

All matrices are compared in neuron-major orientation (w1 / w3 as stored, w2 transposed), shape R x W.
For every pair (A, B) the same statistics are computed for B and for B_shuf (all elements of B, each with its own
block scale, randomly permuted; seed = SEED + pair index) as the no-structure control.
"""
import argparse
import json
import math
import subprocess
import sys
import time

import numpy as np

import common as K

# torch is imported lazily; cudnn autotuning helps the fixed-shape convolutions

SEED = 20261125
N_PER = 30
CATS = ["A1_same_expert_w1w3", "A2_same_expert_w2", "A3_same_layer_same_role", "A4_same_layer_other_role",
        "A5_adjacent_layer_same_expert", "A6_adjacent_layer_random_expert", "A7_far_layer", "A8_self"]
TILES = [(32, 64), (128, 8)]          # (tile side, number of tiles) for the local search
SHIFT_FRAC = 0.5                      # global shifts |dy| <= R/2, |dx| <= W/2


def index(model):
    ix = {}
    for i, e in enumerate(K.entries(model)):
        ix[(e["layer"], e["expert"], e["role"])] = i
    return ix


def make_plan(model):
    rng = np.random.default_rng(SEED)
    ents = K.entries(model)
    ix = index(model)
    layers = sorted({e["layer"] for e in ents})
    experts = {L: sorted({e["expert"] for e in ents if e["layer"] == L}) for L in layers}
    roles = ["w1", "w3", "w2"]
    pairs = []

    def rl():
        return int(rng.choice(layers))

    def re(L, but=None):
        while True:
            x = int(rng.choice(experts[L]))
            if x != but:
                return x
    for cat in CATS:
        for _ in range(N_PER):
            if cat == "A1_same_expert_w1w3":
                L = rl(); e = re(L); a, b = (L, e, "w1"), (L, e, "w3")
            elif cat == "A2_same_expert_w2":
                L = rl(); e = re(L); a, b = (L, e, str(rng.choice(["w1", "w3"]))), (L, e, "w2")
            elif cat == "A3_same_layer_same_role":
                L = rl(); e = re(L); r = str(rng.choice(roles)); a, b = (L, e, r), (L, re(L, e), r)
            elif cat == "A4_same_layer_other_role":
                L = rl(); e = re(L); r1, r2 = rng.choice(roles, 2, replace=False)
                a, b = (L, e, str(r1)), (L, re(L, e), str(r2))
            elif cat in ("A5_adjacent_layer_same_expert", "A6_adjacent_layer_random_expert"):
                while True:
                    L = rl()
                    if L + 1 in experts:
                        break
                r = str(rng.choice(roles))
                if cat.startswith("A5"):
                    e = int(rng.choice(sorted(set(experts[L]) & set(experts[L + 1]))))
                    a, b = (L, e, r), (L + 1, e, r)
                else:
                    a, b = (L, re(L), r), (L + 1, re(L + 1), r)
            elif cat == "A7_far_layer":
                while True:
                    L1, L2 = rl(), rl()
                    if abs(L1 - L2) >= len(layers) / 2:
                        break
                r = str(rng.choice(roles)); a, b = (L1, re(L1), r), (L2, re(L2), r)
            else:
                L = rl(); a = (L, re(L), str(rng.choice(roles))); b = a
            pairs.append({"id": len(pairs), "cat": cat, "a": ents[ix[a]]["matrix"], "b": ents[ix[b]]["matrix"],
                          "ia": ix[a], "ib": ix[b]})
    K.RES.mkdir(parents=True, exist_ok=True)
    json.dump({"seed": SEED, "n_per_category": N_PER, "pairs": pairs}, open(K.RES / f"pairs_plan_{model}.json", "w"), indent=0)
    print(model, len(pairs), "pairs")


def load_nm(model, i):
    e = K.entries(model)[i]
    C, E = K.load(model, e)
    Cn, En = K.neuron_major(C, E, e["role"])
    return Cn, En, C, E, e


# ------------------------------------------------------------------ 1. same position
def same_position(torch, Ca, Cb, Eb_el, Ea_blk=None, Eb_blk=None):
    """code-length gains (bits per code) of B given A at the same position; see spec item A.1."""
    dev = "cuda"
    a = torch.as_tensor(Ca, device=dev).long().ravel()
    b = torch.as_tensor(Cb, device=dev).long().ravel()
    eb = torch.as_tensor(Eb_el, device=dev).long().ravel()
    N = b.numel()

    def table(ctx, sym, nctx, k):
        return torch.bincount(ctx * k + sym, minlength=nctx * k).reshape(nctx, k).cpu().numpy()
    LB = float(K.kt_bits(np.bincount(b.cpu().numpy(), minlength=16), 16))
    LBA = float(K.kt_bits(table(a, b, 16, 16), 16).sum())
    LBE = float(K.kt_bits(table(eb, b, 256, 16), 16).sum())
    LBEA = float(K.kt_bits(table(eb * 16 + a, b, 4096, 16), 16).sum())
    out = {"gain_code": (LB - LBA) / N, "gain_code_given_own_scale": (LBE - LBEA) / N,
           "L_B_per_code": LB / N, "L_B_given_own_scale_per_code": LBE / N}
    for k, nm in enumerate(K.PLANES):
        bit = (b >> (3 - k)) & 1
        Lk = float(K.kt_bits(np.bincount(bit.cpu().numpy(), minlength=2), 2))
        Lka = float(K.kt_bits(table(a, bit, 16, 2), 2).sum())
        out[f"gain_bit_{nm}"] = (Lk - Lka) / N
    if Ea_blk is not None:
        ea = torch.as_tensor(Ea_blk, device=dev).long().ravel()
        ebb = torch.as_tensor(Eb_blk, device=dev).long().ravel()
        LE = float(K.kt_bits(np.bincount(ebb.cpu().numpy(), minlength=256), 256))
        LEA = float(K.kt_bits(table(ea, ebb, 256, 256), 256).sum())
        out["gain_scale_per_weight"] = (LE - LEA) / N
        out["gain_scale_per_scale"] = (LE - LEA) / ebb.numel()
    return out


# ------------------------------------------------------------------ 2. global shifts
def xcorr_window(torch, X, Y, R, W):
    """sum over channels of the linear cross-correlation c[dy,dx] = sum_i X[i+dy, j+dx] Y[i,j] for
    |dy| <= R*SHIFT_FRAC, |dx| <= W*SHIFT_FRAC. X, Y: (ch, R, W) float32 on GPU. Returns (2hy+1, 2hx+1)."""
    hy, hx = int(R * SHIFT_FRAC), int(W * SHIFT_FRAC)
    acc = None
    for c in range(X.shape[0]):
        FX = torch.fft.rfft2(X[c], s=(2 * R, 2 * W))
        FY = torch.fft.rfft2(Y[c], s=(2 * R, 2 * W))
        p = FX * FY.conj()
        acc = p if acc is None else acc + p
        del FX, FY
    c = torch.fft.irfft2(acc, s=(2 * R, 2 * W))
    c = torch.roll(c, shifts=(hy, hx), dims=(0, 1))[: 2 * hy + 1, : 2 * hx + 1]
    return c, hy, hx


def overlap(torch, R, W, hy, hx):
    dy = torch.arange(-hy, hy + 1, device="cuda").abs()
    dx = torch.arange(-hx, hx + 1, device="cuda").abs()
    return ((R - dy)[:, None] * (W - dx)[None, :]).float()


def global_shift(torch, Sa, Sb, Ca, Cb, self_pair):
    R, W = Ca.shape
    out = {}
    hist_a = np.bincount(Ca.ravel(), minlength=16) / Ca.size
    hist_b = np.bincount(Cb.ravel(), minlength=16) / Cb.size
    p0 = float((hist_a * hist_b).sum())
    best_eq = None
    for nm in ["eq", "sign", "e1", "e0", "m", "v", "absv", "w"]:
        X, Y = Sa[nm], Sb[nm]
        if nm != "eq":
            X = (X - X.mean()) / X.std()
            Y = (Y - Y.mean()) / Y.std()
        c, hy, hx = xcorr_window(torch, X, Y, R, W)
        ov = overlap(torch, R, W, hy, hx)
        if nm == "eq":
            z = (c - ov * p0) / torch.sqrt(ov * p0 * (1 - p0))
            z0 = float(z[hy, hx])
        else:
            z = c / torch.sqrt(ov)        # = r * sqrt(overlap)
            z0 = float(z[hy, hx])
            z = z.abs()
        if self_pair:
            z[hy, hx] = -1e9
        else:
            out[f"{nm}_z_at_zero_shift"] = z0
        zz = z.clone()
        zz[hy, hx] = -1e9
        m = int(torch.argmax(zz))
        dy, dx = m // z.shape[1] - hy, m % z.shape[1] - hx
        out[f"{nm}_max_z_nonzero_shift"] = float(zz.max())
        out[f"{nm}_argmax_shift"] = [dy, dx]
        if nm == "eq":
            best_eq = (dy, dx)
            # exact recount at that shift
            ia, ib = slice(max(0, dy), R + min(0, dy)), slice(max(0, -dy), R - max(0, dy))
            ja, jb = slice(max(0, dx), W + min(0, dx)), slice(max(0, -dx), W - max(0, dx))
            sa, sb = Ca[ia, ja], Cb[ib, jb]
            out["eq_frac_at_best_shift_exact"] = float((sa == sb).mean())
            out["eq_chance_p0"] = p0
            out["n_shifts"] = int((2 * hy + 1) * (2 * hx + 1))
        del c, z, zz, ov
    # conditional code length at the best eq shift (overlap only), minus the cost of naming the shift
    dy, dx = best_eq
    ia, ib = slice(max(0, dy), R + min(0, dy)), slice(max(0, -dy), R - max(0, dy))
    ja, jb = slice(max(0, dx), W + min(0, dx)), slice(max(0, -dx), W - max(0, dx))
    sa, sb = Ca[ia, ja].ravel(), Cb[ib, jb].ravel()
    LB = float(K.kt_bits(np.bincount(sb, minlength=16), 16))
    LBA = K.cond_bits(sa, sb, 16, 16)
    out["gain_code_at_best_shift_per_code"] = (LB - LBA - math.log2(out["n_shifts"])) / Cb.size
    return out


# ------------------------------------------------------------------ 3. local tile search
def local_search(torch, Sa, Sb_list, Ca, Cb_list, self_pair, rng):
    """tiles of B (real and shuffled, same positions) searched at every position of A (stride 1)."""
    F = torch.nn.functional
    R, W = Ca.shape
    hist_a = np.bincount(Ca.ravel(), minlength=16) / Ca.size
    res = {}
    for T, n in TILES:
        pos = [(int(rng.integers(0, R - T + 1)), int(rng.integers(0, W - T + 1))) for _ in range(n)]
        npos = (R - T + 1) * (W - T + 1)
        for tag, Sb, Cb in zip(("real", "shuffled"), Sb_list, Cb_list):
            rows = []
            for nm, ch in (("eq", 16), ("bits", 4)):
                if nm == "eq":
                    X = Sa["eq"]
                    Wt = torch.stack([Sb["eq"][:, r:r + T, c:c + T] for r, c in pos])
                else:
                    X = torch.cat([Sa[p] for p in K.PLANES])
                    Wt = torch.stack([torch.cat([Sb[p] for p in K.PLANES])[:, r:r + T, c:c + T] for r, c in pos])
                best, arg = [], []
                for i0 in range(0, n, 16):
                    # fp16 tensor-core convolution (fp32 accumulation); counts <= 16384, rounding < 0.1 % of a tile
                    o = F.conv2d(X[None].half(), Wt[i0:i0 + 16].half())[0].float()      # (t, R-T+1, W-T+1)
                    if self_pair:
                        for t in range(o.shape[0]):
                            r, c = pos[i0 + t]
                            o[t, max(0, r - T + 1): r + T, max(0, c - T + 1): c + T] = -1e9
                    v, a = o.reshape(o.shape[0], -1).max(dim=1)
                    best += v.tolist(); arg += a.tolist()
                    del o
                for t, (r, c) in enumerate(pos):
                    if len(rows) <= t:
                        rows.append({"pos": [r, c]})
                    if nm == "eq":
                        n_el = T * T
                        a_hat = best[t] / n_el
                        ht = np.bincount(Cb[r:r + T, c:c + T].ravel(), minlength=16) / n_el
                        p0 = float((ht * hist_a).sum())
                        llr = 0.0
                        if a_hat > p0:
                            llr = n_el * (a_hat * math.log2(a_hat / p0) +
                                          ((1 - a_hat) * math.log2((1 - a_hat) / (1 - p0)) if a_hat < 1 else -math.log2(1 - p0)))
                        rows[t].update(eq_max=a_hat, eq_p0=p0, eq_at=[arg[t] // (W - T + 1), arg[t] % (W - T + 1)],
                                       evidence_bits=llr - math.log2(npos))
                    else:
                        rows[t].update(bits_max=(best[t] / (4 * T * T) + 1) / 2)
            res[f"T{T}_{tag}"] = rows
    return res


def cmd_gpu(a):
    import torch
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    plan = json.load(open(K.RES / f"pairs_plan_{a.model}.json"))["pairs"]
    path = K.RES / f"pairs_gpu_{a.model}.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
    todo = [p for p in plan if p["id"] not in done][: a.limit]
    t0 = time.time()
    with open(path, "a") as fo:
        for k, p in enumerate(todo):
            t1 = time.time()
            self_pair = p["ia"] == p["ib"]
            Ca, Ea, Ca_st, Ea_st, ea = load_nm(a.model, p["ia"])
            if self_pair:
                Cb, Eb, Cb_st, Eb_st, eb = Ca, Ea, Ca_st, Ea_st, ea
            else:
                Cb, Eb, Cb_st, Eb_st, eb = load_nm(a.model, p["ib"])
            rng = np.random.default_rng(SEED + p["id"])
            perm = rng.permutation(Cb.size)
            Cs = Cb.ravel()[perm].reshape(Cb.shape)
            Es = Eb.ravel()[perm].reshape(Eb.shape)
            Eb_blk_s = Eb_st.ravel()[rng.permutation(Eb_st.size)].reshape(Eb_st.shape)
            rec = {"id": p["id"], "cat": p["cat"], "a": p["a"], "b": p["b"], "shape": list(Ca.shape)}
            Sa = K.gpu_signals(torch, Ca, Ea)
            Sb = K.gpu_signals(torch, Cb, Eb)
            Ss = K.gpu_signals(torch, Cs, Es)
            same_orient = ea["role"] == eb["role"] or {ea["role"], eb["role"]} == {"w1", "w3"}
            for tag, C2, E2, S2, Eblk in (("real", Cb, Eb, Sb, Eb_st), ("shuffled", Cs, Es, Ss, Eb_blk_s)):
                r = {}
                if not self_pair:
                    r["same_position"] = same_position(torch, Ca, C2, E2, Ea_st if same_orient else None,
                                                       Eblk if same_orient else None)
                r["global_shift"] = global_shift(torch, Sa, S2, Ca, C2, self_pair and tag == "real")
                rec[tag] = r
            rec["local"] = local_search(torch, Sa, [Sb, Ss], Ca, [Cb, Cs], self_pair, np.random.default_rng(SEED + 7 + p["id"]))
            rec["seconds"] = time.time() - t1
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            del Sa, Sb, Ss
            torch.cuda.empty_cache()
            print(f"{k + 1}/{len(todo)} {p['cat']} {p['a']} {p['b']} {rec['seconds']:.1f}s total {time.time() - t0:.0f}s", flush=True)
    print("GPU_DONE", a.model, f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ 4. general-purpose compressors
COMPRESSORS = {"zstd19": ["zstd", "-19", "--long=27", "-q", "-c", "-T1"], "xz9e": ["xz", "-9e", "-T1", "-c", "-q"]}


def csize(name, data):
    return len(subprocess.run(COMPRESSORS[name], input=data, capture_output=True, check=True).stdout)


def pack(C):
    C = C.ravel()
    return ((C[0::2] & 15) | (C[1::2] << 4)).astype(np.uint8).tobytes()


def comp_task(model, p):
    t0 = time.time()
    Ca, _, _, _, _ = load_nm(model, p["ia"])
    Cb, _, _, _, _ = load_nm(model, p["ib"])
    rng = np.random.default_rng(SEED + 100000 + p["id"])
    Cas = Ca.ravel()[rng.permutation(Ca.size)].reshape(Ca.shape)
    rec = {"id": p["id"], "cat": p["cat"], "a": p["a"], "b": p["b"], "n_codes": int(Cb.size)}
    for rep, f in (("packed", pack), ("unpacked", lambda C: np.ascontiguousarray(C).tobytes())):
        A, B, As = f(Ca), f(Cb), f(Cas)
        for cn in COMPRESSORS:
            cA, cB, cAB, cAs, cAsB = csize(cn, A), csize(cn, B), csize(cn, A + B), csize(cn, As), csize(cn, As + B)
            rec[f"{rep}_{cn}"] = {"C_A": cA, "C_B": cB, "C_AB": cAB, "C_Ashuf": cAs, "C_AshufB": cAsB,
                                  "cond_gain_bits_per_code": 8 * (cB - (cAB - cA)) / Cb.size,
                                  "cond_gain_bits_per_code_control": 8 * (cB - (cAsB - cAs)) / Cb.size,
                                  "NCD": (cAB - min(cA, cB)) / max(cA, cB),
                                  "NCD_control": (cAsB - min(cAs, cB)) / max(cAs, cB)}
    rec["seconds"] = time.time() - t0
    return rec


def cmd_comp(a):
    from concurrent.futures import ProcessPoolExecutor
    plan = [p for p in json.load(open(K.RES / f"pairs_plan_{a.model}.json"))["pairs"] if p["ia"] != p["ib"]]
    path = K.RES / f"pairs_comp_{a.model}.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
    todo = [p for p in plan if p["id"] not in done][: a.limit]
    t0 = time.time()
    with open(path, "a") as fo, ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(comp_task, [a.model] * len(todo), todo), 1):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            if k % 10 == 0:
                print(f"{k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("COMP_DONE", a.model, f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "gpu", "comp"])
    ap.add_argument("model", choices=list(K.MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    {"plan": lambda a: make_plan(a.model), "gpu": cmd_gpu, "comp": cmd_comp}[a.cmd](a)


if __name__ == "__main__":
    main()
