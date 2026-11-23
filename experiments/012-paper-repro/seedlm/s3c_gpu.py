"""S3-c kernels (final check of the SeedLM-to-lossless route).

1. _group_kernel: Algorithm 3 with the seed shared by G consecutive blocks, for G in (1, 2, 8, 32, 128) in one pass.
   For every seed each block gets its own least-squares coefficients, quantised exactly as in seedlm_gpu
   (same arithmetic); the group's error is the sum of its blocks' errors and the group keeps the seed with the
   smallest sum (strict '<', so the smallest seed wins ties). G = 1 is the original search (checked against S3-a).
   Exponent modes: 0 paper, 1 range (per block, 4 bits), 2 fixed (one exponent E0 for the whole matrix, 0 bits).
2. _codelen_kernel: for every block and every seed, the code length of the block's true FP4 codes under a frozen
   table -log2 p(code | context(w_hat_s)), where the context is the S3-b context (interval of w_hat among the FP4
   grid with SUB sub-bins, block-scale byte E). Tracks, per block: the code length at the MSE-chosen seed; the
   seed with the smallest code length (and its q, e); and log2 sum_s 2^-L_s (the bits-back / mixture bound).
"""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

GS = (1, 2, 8, 32, 128)
BB = 128


@triton.jit
def _fit(W, Pi, U, MODE: tl.constexpr, E0: tl.constexpr):
    t = tl.sum(W[:, None, :] * Pi[None, :, :], axis=2)                                          # (BB, PP2)
    amax = tl.max(tl.abs(t), axis=1)
    zero = amax == 0.0
    if MODE == 0:
        bits = amax.to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 0xFF) - 127
    elif MODE == 1:
        x = amax / 7.5
        bits = x.to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    else:
        e = tl.zeros_like(amax).to(tl.int32) + E0
    e = tl.minimum(tl.maximum(e, -8), 7)
    if MODE != 2:
        e = tl.where(zero, -8, e)
    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
    scl = ((127 + e) << 23).to(tl.float32, bitcast=True)
    q = libdevice.rint(t * inv[:, None])
    q = tl.minimum(tl.maximum(q, -8.0), 7.0)
    q = tl.where(zero[:, None], 0.0, q)
    th = q * scl[:, None]
    R = tl.sum(th[:, None, :] * U[None, :, :], axis=2)                                          # (BB, CP2)
    return q, e, R


@triton.jit
def _level(err, q, e, sid, best, bseed, bq, be, G: tl.constexpr, BBc: tl.constexpr):
    """best/bseed are per group (BBc // G,); bq/be per block."""
    if G == 1:
        take_g = err < best
        best = tl.where(take_g, err, best)
        bseed = tl.where(take_g, sid, bseed)
        take = take_g
    else:
        ge = tl.sum(tl.reshape(err, (BBc // G, G)), axis=1)
        take_g = ge < best
        best = tl.where(take_g, ge, best)
        bseed = tl.where(take_g, sid, bseed)
        take = tl.reshape(tl.broadcast_to(take_g[:, None], (BBc // G, G)), (BBc,))
    bq = tl.where(take[:, None], q, bq)
    be = tl.where(take, e, be)
    return best, bseed, bq, be


@triton.jit
def _group_kernel(W_ptr, U_ptr, Pi_ptr, n_seeds, seed_base, berr_ptr, bseed_ptr, bq_ptr, be_ptr,
                  C: tl.constexpr, P: tl.constexpr, CP2: tl.constexpr, PP2: tl.constexpr,
                  MODE: tl.constexpr, E0: tl.constexpr):
    # blocks are padded to a multiple of 128 (zeros); program = one tile of 128 blocks = one 128-group
    BBc: tl.constexpr = 128
    pid = tl.program_id(0)
    rb = pid * BBc + tl.arange(0, BBc)
    rc = tl.arange(0, CP2)
    rp = tl.arange(0, PP2)
    mc = rc < C
    mp = rp < P
    W = tl.load(W_ptr + rb[:, None] * C + rc[None, :], mask=mc[None, :], other=0.0)
    mpc = mp[:, None] & mc[None, :]
    mcp = mc[:, None] & mp[None, :]
    f = tl.full((BBc,), float("inf"), tl.float32)
    z = tl.zeros((BBc,), tl.int32)
    zq = tl.zeros((BBc, PP2), tl.float32)
    b1, s1, q1, e1 = f, z, zq, z
    b2, s2, q2, e2 = tl.full((BBc // 2,), float("inf"), tl.float32), tl.zeros((BBc // 2,), tl.int32), zq, z
    b3, s3, q3, e3 = tl.full((BBc // 8,), float("inf"), tl.float32), tl.zeros((BBc // 8,), tl.int32), zq, z
    b4, s4, q4, e4 = tl.full((BBc // 32,), float("inf"), tl.float32), tl.zeros((BBc // 32,), tl.int32), zq, z
    b5, s5, q5, e5 = tl.full((BBc // 128,), float("inf"), tl.float32), tl.zeros((BBc // 128,), tl.int32), zq, z
    for s in range(0, n_seeds):
        Pi = tl.load(Pi_ptr + s * (P * C) + rp[:, None] * C + rc[None, :], mask=mpc, other=0.0)
        U = tl.load(U_ptr + s * (C * P) + rc[:, None] * P + rp[None, :], mask=mcp, other=0.0)
        q, e, R = _fit(W, Pi, U, MODE, E0)
        D = W - R
        err = tl.sum(D * D, axis=1)
        sid = seed_base + s + 1
        b1, s1, q1, e1 = _level(err, q, e, sid, b1, s1, q1, e1, 1, BBc)
        b2, s2, q2, e2 = _level(err, q, e, sid, b2, s2, q2, e2, 2, BBc)
        b3, s3, q3, e3 = _level(err, q, e, sid, b3, s3, q3, e3, 8, BBc)
        b4, s4, q4, e4 = _level(err, q, e, sid, b4, s4, q4, e4, 32, BBc)
        b5, s5, q5, e5 = _level(err, q, e, sid, b5, s5, q5, e5, 128, BBc)
    # outputs: per level L, per block (seed broadcast to blocks); stride n_total per level
    n_tot = tl.num_programs(0) * BBc
    rq = rb[:, None] * P + rp[None, :]
    mq = mp[None, :] & (rb[:, None] >= 0)
    tl.store(bseed_ptr + 0 * n_tot + rb, s1)
    tl.store(bseed_ptr + 1 * n_tot + rb, tl.reshape(tl.broadcast_to(s2[:, None], (BBc // 2, 2)), (BBc,)))
    tl.store(bseed_ptr + 2 * n_tot + rb, tl.reshape(tl.broadcast_to(s3[:, None], (BBc // 8, 8)), (BBc,)))
    tl.store(bseed_ptr + 3 * n_tot + rb, tl.reshape(tl.broadcast_to(s4[:, None], (BBc // 32, 32)), (BBc,)))
    tl.store(bseed_ptr + 4 * n_tot + rb, tl.reshape(tl.broadcast_to(s5[:, None], (BBc // 128, 128)), (BBc,)))
    tl.store(berr_ptr + rb, b1)
    tl.store(bq_ptr + 0 * n_tot * P + rq, q1, mask=mq)
    tl.store(bq_ptr + 1 * n_tot * P + rq, q2, mask=mq)
    tl.store(bq_ptr + 2 * n_tot * P + rq, q3, mask=mq)
    tl.store(bq_ptr + 3 * n_tot * P + rq, q4, mask=mq)
    tl.store(bq_ptr + 4 * n_tot * P + rq, q5, mask=mq)
    tl.store(be_ptr + 0 * n_tot + rb, e1)
    tl.store(be_ptr + 1 * n_tot + rb, e2)
    tl.store(be_ptr + 2 * n_tot + rb, e3)
    tl.store(be_ptr + 3 * n_tot + rb, e4)
    tl.store(be_ptr + 4 * n_tot + rb, e5)


def _p2(x):
    return 1 << (int(x) - 1).bit_length()


def group_search(blocks, gcb, mode, e0=0, device="cuda"):
    """blocks (n_blocks, C) float32 -> dict G -> (seeds, q, e) for G in GS (blocks padded to 128 internally)."""
    nb = len(blocks)
    pad = (-nb) % BB
    Wp = np.concatenate([blocks, np.zeros((pad, blocks.shape[1]), np.float32)]).astype(np.float32)
    Wt = torch.from_numpy(Wp).to(device).contiguous()
    N = len(Wp)
    C, P = gcb.C, gcb.P
    L = len(GS)
    seed = torch.zeros((L, N), device=device, dtype=torch.int32)
    q = torch.zeros((L, N, P), device=device, dtype=torch.float32)
    e = torch.zeros((L, N), device=device, dtype=torch.int32)
    err = torch.full((N,), float("inf"), device=device, dtype=torch.float32)
    assert len(gcb.chunks) == 1
    s0, U, Pi = gcb.chunks[0]
    m = {"paper": 0, "range": 1, "fixed": 2}[mode]
    _group_kernel[(N // BB,)](Wt, U, Pi, U.shape[0], s0, err, seed, q, e, C=C, P=P, CP2=_p2(C), PP2=_p2(P),
                              MODE=m, E0=int(e0))
    seed, q, e = seed.cpu().numpy(), q.cpu().numpy(), e.cpu().numpy()
    return {G: (seed[i, :nb].astype(np.int64), q[i, :nb].astype(np.int8), e[i, :nb].astype(np.int8))
            for i, G in enumerate(GS)}


@triton.jit
def _ctx_bin(u, SUB: tl.constexpr):
    """S3-b interval index b = i * SUB + sub-bin, with the grid -6..6 (15 points)."""
    i = tl.zeros_like(u.to(tl.int32))
    lo = tl.full(u.shape, -6.0, tl.float32)
    hi = tl.full(u.shape, 6.0, tl.float32)
    i += (u >= -6.0).to(tl.int32)
    i += (u >= -4.0).to(tl.int32)
    i += (u >= -3.0).to(tl.int32)
    i += (u >= -2.0).to(tl.int32)
    i += (u >= -1.5).to(tl.int32)
    i += (u >= -1.0).to(tl.int32)
    i += (u >= -0.5).to(tl.int32)
    i += (u >= 0.0).to(tl.int32)
    i += (u >= 0.5).to(tl.int32)
    i += (u >= 1.0).to(tl.int32)
    i += (u >= 1.5).to(tl.int32)
    i += (u >= 2.0).to(tl.int32)
    i += (u >= 3.0).to(tl.int32)
    i += (u >= 4.0).to(tl.int32)
    i += (u >= 6.0).to(tl.int32)
    lo = tl.where(i == 2, -4.0, lo)
    lo = tl.where(i == 3, -3.0, lo)
    lo = tl.where(i == 4, -2.0, lo)
    lo = tl.where(i == 5, -1.5, lo)
    lo = tl.where(i == 6, -1.0, lo)
    lo = tl.where(i == 7, -0.5, lo)
    lo = tl.where(i == 8, 0.0, lo)
    lo = tl.where(i == 9, 0.5, lo)
    lo = tl.where(i == 10, 1.0, lo)
    lo = tl.where(i == 11, 1.5, lo)
    lo = tl.where(i == 12, 2.0, lo)
    lo = tl.where(i == 13, 3.0, lo)
    lo = tl.where(i == 14, 4.0, lo)
    hi = tl.where(i == 1, -4.0, hi)
    hi = tl.where(i == 2, -3.0, hi)
    hi = tl.where(i == 3, -2.0, hi)
    hi = tl.where(i == 4, -1.5, hi)
    hi = tl.where(i == 5, -1.0, hi)
    hi = tl.where(i == 6, -0.5, hi)
    hi = tl.where(i == 7, 0.0, hi)
    hi = tl.where(i == 8, 0.5, hi)
    hi = tl.where(i == 9, 1.0, hi)
    hi = tl.where(i == 10, 1.5, hi)
    hi = tl.where(i == 11, 2.0, hi)
    hi = tl.where(i == 12, 3.0, hi)
    hi = tl.where(i == 13, 4.0, hi)
    inner = (i > 0) & (i < 15)
    frac = tl.where(inner, (u - lo) / (hi - lo), 0.0)
    sb = tl.where(inner, tl.minimum((frac * SUB).to(tl.int32), SUB - 1), 0)
    return i * SUB + sb


@triton.jit
def _codelen_kernel(W_ptr, IS_ptr, E_ptr, SYM_ptr, T_ptr, U_ptr, Pi_ptr, n_blocks, n_seeds,
                    out_lmse_ptr, out_lmin_ptr, out_lse_ptr, out_seed_ptr, out_q_ptr, out_e_ptr,
                    C: tl.constexpr, P: tl.constexpr, CP2: tl.constexpr, PP2: tl.constexpr,
                    BBc: tl.constexpr, MODE: tl.constexpr, SUB: tl.constexpr):
    pid = tl.program_id(0)
    rb = pid * BBc + tl.arange(0, BBc)
    mb = rb < n_blocks
    rc = tl.arange(0, CP2)
    rp = tl.arange(0, PP2)
    mc = rc < C
    mp = rp < P
    m2 = mb[:, None] & mc[None, :]
    W = tl.load(W_ptr + rb[:, None] * C + rc[None, :], mask=m2, other=0.0)
    IS = tl.load(IS_ptr + rb[:, None] * C + rc[None, :], mask=m2, other=1.0)        # 1 / scale per weight
    Eb = tl.load(E_ptr + rb[:, None] * C + rc[None, :], mask=m2, other=0)          # scale byte per weight
    SY = tl.load(SYM_ptr + rb[:, None] * C + rc[None, :], mask=m2, other=0)        # true merged code
    mpc = mp[:, None] & mc[None, :]
    mcp = mc[:, None] & mp[None, :]
    best_err = tl.full((BBc,), float("inf"), tl.float32)
    l_mse = tl.zeros((BBc,), tl.float32)
    l_min = tl.full((BBc,), float("inf"), tl.float32)
    acc = tl.zeros((BBc,), tl.float32)                 # sum_s 2^-(L_s - l_min)
    bseed = tl.zeros((BBc,), tl.int32)
    bq = tl.zeros((BBc, PP2), tl.float32)
    be = tl.zeros((BBc,), tl.int32)
    for s in range(0, n_seeds):
        Pi = tl.load(Pi_ptr + s * (P * C) + rp[:, None] * C + rc[None, :], mask=mpc, other=0.0)
        U = tl.load(U_ptr + s * (C * P) + rc[:, None] * P + rp[None, :], mask=mcp, other=0.0)
        q, e, R = _fit(W, Pi, U, MODE, 0)
        D = W - R
        err = tl.sum(D * D, axis=1)
        b = _ctx_bin(R * IS, SUB)
        idx = ((b * 256 + Eb) * 15 + SY)
        Lw = tl.load(T_ptr + idx, mask=m2, other=0.0)
        L = tl.sum(Lw, axis=1)
        tk = err < best_err
        best_err = tl.where(tk, err, best_err)
        l_mse = tl.where(tk, L, l_mse)
        tl_ = L < l_min
        acc = tl.where(tl_, acc * libdevice.exp2(L - l_min) + 1.0, acc + libdevice.exp2(l_min - L))
        l_min = tl.where(tl_, L, l_min)
        bseed = tl.where(tl_, s + 1, bseed)
        bq = tl.where(tl_[:, None], q, bq)
        be = tl.where(tl_, e, be)
    tl.store(out_lmse_ptr + rb, l_mse, mask=mb)
    tl.store(out_lmin_ptr + rb, l_min, mask=mb)
    tl.store(out_lse_ptr + rb, tl.log2(acc), mask=mb)
    tl.store(out_seed_ptr + rb, bseed, mask=mb)
    tl.store(out_q_ptr + rb[:, None] * P + rp[None, :], bq, mask=mb[:, None] & mp[None, :])
    tl.store(out_e_ptr + rb, be, mask=mb)


def codelen_search(blocks, inv_scale, Ebyte, sym, table, gcb, mode, sub, device="cuda", BBc=64):
    """Per block: L at the MSE seed, min L over seeds (+ its seed, q, e), log2 sum 2^-(L - Lmin)."""
    nb = len(blocks)
    C, P = gcb.C, gcb.P
    dev = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt).to(device).contiguous()  # noqa: E731
    Wt, It = dev(blocks, torch.float32), dev(inv_scale, torch.float32)
    Et, St = dev(Ebyte, torch.int32), dev(sym, torch.int32)
    Tt = dev(table, torch.float32)
    out = [torch.zeros(nb, device=device, dtype=torch.float32) for _ in range(3)]
    seed = torch.zeros(nb, device=device, dtype=torch.int32)
    q = torch.zeros((nb, P), device=device, dtype=torch.float32)
    e = torch.zeros(nb, device=device, dtype=torch.int32)
    s0, U, Pi = gcb.chunks[0]
    assert len(gcb.chunks) == 1 and s0 == 0
    _codelen_kernel[(triton.cdiv(nb, BBc),)](Wt, It, Et, St, Tt, U, Pi, nb, U.shape[0], *out, seed, q, e,
                                             C=C, P=P, CP2=_p2(C), PP2=_p2(P), BBc=BBc,
                                             MODE={"paper": 0, "range": 1}[mode], SUB=sub)
    lmse, lmin, lse = (o.cpu().numpy().astype(np.float64) for o in out)
    return dict(l_mse=lmse, l_min=lmin, lse=lse, seeds=seed.cpu().numpy().astype(np.int64),
                q=q.cpu().numpy().astype(np.int8), e=e.cpu().numpy().astype(np.int8))
