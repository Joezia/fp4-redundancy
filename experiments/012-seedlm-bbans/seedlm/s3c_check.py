"""S3-c: final check of the SeedLM-to-lossless route (user request, 2026-11-22), on the two Table 1 configurations
(4 bit: C=8, P=3, K=16; 3 bit: C=12, P=4, K=16), range exponent rule. Everything else as in S3-b (same residual
coder, same sub-bin count chosen on train8 for the base configuration, same 008 scale stream, same reference).

  side  SET C P K   entropy-code the side information of the saved S3-a solutions (no new search):
                    seeds, exponents, coefficients with adaptive Dirichlet-1/2 count coders and a few contexts.
  e0    C P K       choose the single shared exponent of the 'normalised, fixed exponent' variant on train8.
  share SET C P K   sharing: the seed shared by G consecutive blocks (G = 1, 2, 8, 32, 128, whole matrix), each
                    block keeping its own coefficients; on the raw weights (per-block exponent, as in the paper)
                    and on weights divided by their MXFP4 block scale (the scale reused as the shared part of the
                    exponent) with a per-block exponent or with one exponent for all matrices (0 bits).
  bb    SET C P K   seed choice by code length instead of MSE (encoder-only change, really coded) and the
                    bits-back bound side - log2 sum_s 2^-L_s (ideal, frozen train8 table, table cost not charged).

Code lengths of side streams are ideal adaptive code lengths (sequential Dirichlet-1/2 counts, exact
log-gamma form; the real rANS coder of S3-b is within 0.001% of this); the residual codes are really
encoded and decoded (racoder.c) and compared bit for bit.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s3_run as R  # noqa: E402
import s3b_lossless as B  # noqa: E402
import slm as S  # noqa: E402
import seedlm_gpu as G  # noqa: E402
import s3c_gpu as X  # noqa: E402
import racoder  # noqa: E402
from txcore import dirichlet  # noqa: E402

OUT = os.path.join(HERE, "../results/s3c")
DATA = os.path.join(R.ROOT, "data/012/s3c")
MODE = "range"


def base_sub(C, P, K):
    tr = [json.loads(l) for l in open(os.path.join(HERE, "../results/s3/s3b_train8.jsonl"))]
    tr = [r for r in tr if (r["C"], r["P"], r["K"], r["mode"]) == (C, P, K, MODE)]
    assert len(tr) == 8
    return min(B.SUBS, key=lambda s: sum(r["ideal_bits"][str(s)] for r in tr))


def ref_of(which, name):
    return json.load(open(B.REF))[which]["per_matrix"][name]


def blocks_of(a, C):
    flat = a.reshape(-1)
    assert len(flat) % C == 0
    return flat.reshape(-1, C)


def entropy0(x):
    _, c = np.unique(x, return_counts=True)
    p = c / c.sum()
    return float(-(c * np.log2(p)).sum())


def side_costs(seeds, q, e, Eblk, K, G_, fixed_e):
    """Side information of one matrix in bits. seeds/q/e per block; seeds shared by groups of G_ blocks."""
    nb, P = q.shape
    sg = seeds[::G_] if G_ > 0 else seeds[:1]
    ng = len(sg)
    out = dict(seed_fixed=float(ng * K), seed_adapt=float(dirichlet(sg - 1, None, (1 << K) - 1)),
               seed_h0=entropy0(sg))
    ei = e.astype(np.int64) + 8
    if fixed_e:
        out.update(e_fixed=0.0, e_adapt_none=0.0, e_adapt_E=0.0, e_adapt_Eprev=0.0)
    else:
        prev = np.concatenate([[16], ei[:-1]])
        out.update(e_fixed=float(4 * nb), e_adapt_none=float(dirichlet(ei, None, 16)),
                   e_adapt_E=float(dirichlet(ei, Eblk, 16)),
                   e_adapt_Eprev=float(dirichlet(ei, Eblk * 17 + prev, 16)))
    qi = q.astype(np.int64).ravel() + 8
    pos = np.tile(np.arange(P), nb)
    er = np.repeat(ei, P)
    out.update(q_fixed=float(4 * P * nb), q_adapt_none=float(dirichlet(qi, None, 16)),
               q_adapt_p=float(dirichlet(qi, pos, 16)), q_adapt_pe=float(dirichlet(qi, pos * 16 + er, 16)))
    return out


def lossless(sym, What, scale, sub, coded=True):
    """S3-b residual coder on w_hat: ideal adaptive bits for each sub-bin count and real coding with `sub`."""
    u = What.astype(np.float64) / scale
    E_w = np.round(np.log2(scale)).astype(np.int64) + 127
    res = dict(ideal_bits={str(sub): float(dirichlet(sym.ravel(), B.contexts(u, E_w, sub).ravel(), 15))})
    if coded:
        ctx = B.contexts(u, E_w, sub).ravel()
        words, nctx = racoder.encode(sym.ravel(), ctx)
        dec = racoder.decode(words, ctx, sym.size, nctx)
        res.update(code_bits=int(32 * len(words)), roundtrip_ok=bool(np.array_equal(dec, sym.ravel())))
    return res


def lossy(W, What, sym, scale):
    W64 = W.astype(np.float64)
    R64 = What.astype(np.float64)
    return dict(rel_mse=float(((W64 - R64) ** 2).sum() / (W64 ** 2).sum()),
                code_mismatch=float(np.mean(R.nearest_code(R64, scale) != sym)))


def block_E(scale, C):
    """scale byte of the first weight of every block."""
    E_w = np.round(np.log2(scale)).astype(np.int64) + 127
    return blocks_of(E_w, C)[:, 0]


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec)[:300], flush=True)


def done_keys(path, keys):
    out = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            out.add(tuple(r[k] for k in keys))
    return out


# ---------------------------------------------------------------- side
def cmd_side(which, C, P, K):
    path = os.path.join(OUT, f"side_{which}.jsonl")
    done = done_keys(path, ("matrix", "C", "P", "K"))
    for name in R.names(which):
        if (name, C, P, K) in done:
            continue
        W, sym, scale = R.load(name)
        z = np.load(os.path.join(R.DATA, f"{name}_C{C}P{P}K{K}_{MODE}.npz"))
        rec = dict(matrix=name, C=C, P=P, K=K, mode=MODE, n=int(W.size), n_blocks=int(len(z["seeds"])))
        rec.update(side_costs(z["seeds"].astype(np.int64), z["q"], z["e"], block_E(scale, C), K, 1, False))
        append(path, rec)


# ---------------------------------------------------------------- e0
def gcb_for(C, P, K):
    return G.GPUCodebook(S.Codebook(K, C, P, device="cuda"))


def cmd_e0(C, P, K):
    gcb = gcb_for(C, P, K)
    blocks = []
    for name in R.names("train8"):
        _, sym, _ = R.load(name)
        blocks.append(blocks_of(R.VAL[sym].astype(np.float32), C)[::64])
    Wn = np.concatenate(blocks)
    res = {}
    for m, e0 in [("range", 0)] + [("fixed", e) for e in range(-4, 4)]:
        sol = X.group_search(Wn, gcb, m, e0)[1]
        Rn = S.reconstruct(*sol, K, C, P)
        res[f"{m}{'' if m == 'range' else e0}"] = float(((Wn - Rn) ** 2).sum() / (Wn ** 2).sum())
        print(m, e0, res, flush=True)
    best = min((k for k in res if k.startswith("fixed")), key=res.get)
    rec = dict(C=C, P=P, K=K, sample="train8, every 64th block", rel_mse_normalised=res, e0=int(best[5:]))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(os.path.join(OUT, f"e0_C{C}P{P}K{K}.json"), "w"), indent=1)
    print(rec)


# ---------------------------------------------------------------- share
class OneSeed:
    def __init__(self, gcb, s):
        self.C, self.P = gcb.C, gcb.P
        _, U, Pi = gcb.chunks[0]
        self.chunks = [(s - 1, U[s - 1:s], Pi[s - 1:s])]


def best_single_seed(blocks, gcb, mode, e0, n_sample=4096, chunk=8192):
    """seed minimising the summed error of an evenly spaced sample of blocks (whole-matrix shared seed)."""
    idx = np.linspace(0, len(blocks) - 1, n_sample).astype(np.int64)
    Ws = torch.from_numpy(blocks[idx]).cuda()
    _, U, Pi = gcb.chunks[0]
    tot = []
    for s0 in range(0, U.shape[0], chunk):
        u, pi = U[s0:s0 + chunk], Pi[s0:s0 + chunk]
        t = torch.einsum("spc,bc->sbp", pi, Ws)
        amax = t.abs().amax(-1)
        zero = amax == 0
        if mode == "range":
            m, E = torch.frexp(amax / 7.5)
            e = E - (m == 0.5).int()
            e = torch.where(zero, torch.full_like(e, -8), e.clamp(-8, 7))
        else:
            e = torch.full_like(amax, e0, dtype=torch.int32)
        qq = torch.round(torch.ldexp(t, -e[..., None].float())).clamp(-8, 7)
        qq = torch.where(zero[..., None], torch.zeros_like(qq), qq)
        th = torch.ldexp(qq, e[..., None].float())
        Rr = torch.einsum("scp,sbp->sbc", u, th)
        tot.append(((Ws[None] - Rr) ** 2).sum((1, 2)))
    tot = torch.cat(tot)
    return int(torch.argmin(tot).item()) + 1


def cmd_share(which, C, P, K):
    path = os.path.join(OUT, f"share_{which}.jsonl")
    done = done_keys(path, ("matrix", "C", "P", "K"))
    sub = base_sub(C, P, K)
    e0 = json.load(open(os.path.join(OUT, f"e0_C{C}P{P}K{K}.json")))["e0"]
    gcb = gcb_for(C, P, K)
    os.makedirs(DATA, exist_ok=True)
    for name in R.names(which):
        if (name, C, P, K) in done:
            continue
        W, sym, scale = R.load(name)
        ref = ref_of(which, name)
        n = W.size
        Eblk = block_E(scale, C)
        base = np.load(os.path.join(R.DATA, f"{name}_C{C}P{P}K{K}_{MODE}.npz"))
        variants = [("raw_range", blocks_of(W, C), "range", 0, False),
                    ("norm_range", blocks_of((W / scale).astype(np.float32), C), "range", 0, True),
                    ("norm_fixed", blocks_of((W / scale).astype(np.float32), C), "fixed", e0, True)]
        for vname, blk, mode, ee, norm in variants:
            t0 = time.time()
            sols = X.group_search(blk, gcb, mode, ee)
            torch.cuda.synchronize()
            t_search = time.time() - t0
            s1 = best_single_seed(blk, gcb, mode, ee)
            sols["all"] = X.group_search(blk, OneSeed(gcb, s1), mode, ee)[1]
            if vname == "raw_range":
                same = float(np.mean(sols[1][0] == base["seeds"].astype(np.int64)))
            for Gk, (sd, qq, eq) in sols.items():
                What = S.reconstruct(sd, qq, eq, K, C, P).reshape(W.shape).astype(np.float64)
                if norm:
                    What = What * scale
                np.savez_compressed(os.path.join(DATA, f"{name}_C{C}P{P}K{K}_{vname}_G{Gk}.npz"),
                                    seeds=sd.astype(np.uint32), q=qq, e=eq)
                Gn = len(sd) if Gk == "all" else Gk
                sc = side_costs(sd, qq, eq, Eblk, K, Gn, mode == "fixed")
                ll = lossless(sym, What, scale, sub)
                side_fixed = sc["seed_fixed"] + sc["e_fixed"] + sc["q_fixed"]
                total = side_fixed + ll["code_bits"] + ref["scale_bits"]
                rec = dict(matrix=name, C=C, P=P, K=K, variant=vname, G=Gk, n=int(n), sub=sub, e0=ee if mode == "fixed" else None,
                           side_fixed_bits=side_fixed, **sc, **lossy(W, What, sym, scale), **ll,
                           scale_bits=ref["scale_bits"], ref_total_bits=ref["total"], total_bits=float(total),
                           bpw_total=float(total / n), net_gain_bpw=float((ref["total"] - total) / n),
                           search_s=t_search, seed_whole=s1 if Gk == "all" else None)
                if vname == "raw_range" and Gk == 1:
                    rec["same_seed_as_s3a"] = same
                append(path, rec)


# ---------------------------------------------------------------- bb
def static_table(C, P, K, sub):
    """-log2 p(code | context) from the train8 S3-a solutions (Dirichlet 1/2), dense context b * 256 + E."""
    f = os.path.join(DATA, f"table_C{C}P{P}K{K}_sub{sub}.npy")
    if os.path.exists(f):
        return np.load(f)
    nb = 16 * sub
    cnt = np.zeros((nb * 256, 15))
    only = os.environ.pop("S3_ONLY", None)          # the training set must not be filtered by the test subset
    train = R.names("train8")
    if only is not None:
        os.environ["S3_ONLY"] = only
    assert len(train) == 8
    for name in train:
        W, sym, scale = R.load(name)
        z = np.load(os.path.join(R.DATA, f"{name}_C{C}P{P}K{K}_{MODE}.npz"))
        What = S.reconstruct(z["seeds"].astype(np.int64), z["q"], z["e"], K, C, P).reshape(W.shape)
        key = dense_ctx(What.astype(np.float64) / scale, scale, sub)
        cnt += np.bincount((key * 15 + sym).ravel(), minlength=cnt.size).reshape(cnt.shape)
    assert cnt.sum() == 8 * 11796480
    T = -np.log2((cnt + 0.5) / (cnt.sum(1, keepdims=True) + 7.5))
    os.makedirs(DATA, exist_ok=True)
    np.save(f, T.astype(np.float32))
    return T.astype(np.float32)


def dense_ctx(u, scale, sub):
    i = np.searchsorted(B.GRID, u, side="right")
    lo = B.GRID[np.clip(i - 1, 0, 14)]
    hi = B.GRID[np.clip(i, 0, 14)]
    inner = (i > 0) & (i < 15)
    frac = np.where(inner, (u - lo) / np.where(inner, hi - lo, 1), 0)
    sb = np.where(inner, np.minimum((frac * sub).astype(np.int64), sub - 1), 0)
    E_w = np.round(np.log2(scale)).astype(np.int64) + 127
    return (i * sub + sb) * 256 + E_w


def cmd_bb(which, C, P, K):
    path = os.path.join(OUT, f"bb_{which}.jsonl")
    done = done_keys(path, ("matrix", "C", "P", "K"))
    sub = base_sub(C, P, K)
    T = static_table(C, P, K, sub)
    gcb = gcb_for(C, P, K)
    for name in R.names(which):
        if (name, C, P, K) in done:
            continue
        W, sym, scale = R.load(name)
        ref = ref_of(which, name)
        n = W.size
        E_w = np.round(np.log2(scale)).astype(np.int64) + 127
        t0 = time.time()
        r = X.codelen_search(blocks_of(W, C), blocks_of((1.0 / scale).astype(np.float32), C), blocks_of(E_w, C),
                             blocks_of(sym, C), T.ravel(), gcb, MODE, sub)
        torch.cuda.synchronize()
        t_search = time.time() - t0
        # check: static code length of the saved MSE solution, recomputed in numpy
        z = np.load(os.path.join(R.DATA, f"{name}_C{C}P{P}K{K}_{MODE}.npz"))
        What0 = S.reconstruct(z["seeds"].astype(np.int64), z["q"], z["e"], K, C, P).reshape(W.shape)
        l_mse_np = float(T[dense_ctx(What0.astype(np.float64) / scale, scale, sub), sym].astype(np.float64).sum())
        What = S.reconstruct(r["seeds"], r["q"], r["e"], K, C, P).reshape(W.shape)
        l_min_np = float(T[dense_ctx(What.astype(np.float64) / scale, scale, sub), sym].astype(np.float64).sum())
        np.savez_compressed(os.path.join(DATA, f"{name}_C{C}P{P}K{K}_codelen.npz"), seeds=r["seeds"].astype(np.uint32),
                            q=r["q"], e=r["e"])
        nb = len(r["seeds"])
        side_fixed = nb * (K + 4 + 4 * P)
        sc = side_costs(r["seeds"], r["q"], r["e"], block_E(scale, C), K, 1, False)
        ll = lossless(sym, What, scale, sub)
        total = side_fixed + ll["code_bits"] + ref["scale_bits"]
        rec = dict(matrix=name, C=C, P=P, K=K, n=int(n), sub=sub, n_blocks=nb,
                   static_l_mse=float(r["l_mse"].sum()), static_l_mse_numpy=l_mse_np, static_l_min_numpy=l_min_np,
                   static_l_min=float(r["l_min"].sum()), static_mixture=float((r["l_min"] - r["lse"]).sum()),
                   log2_eff_seeds_mean=float(r["lse"].mean()),
                   log2_eff_seeds_q=[float(x) for x in np.quantile(r["lse"], [.1, .5, .9])],
                   side_fixed_bits=float(side_fixed), **sc, **lossy(W, What.astype(np.float64), sym, scale), **ll,
                   scale_bits=ref["scale_bits"], ref_total_bits=ref["total"], total_bits=float(total),
                   bpw_total=float(total / n), net_gain_bpw=float((ref["total"] - total) / n), search_s=t_search)
        append(path, rec)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "e0":
        cmd_e0(*map(int, sys.argv[2:5]))
    else:
        {"side": cmd_side, "share": cmd_share, "bb": cmd_bb}[cmd](sys.argv[2], *map(int, sys.argv[3:6]))
