"""GPU (Triton) version of Algorithm 3: exhaustive seed search with quantised coefficients.

Same arithmetic as slm.search_numpy (float32, exponent from the float bits, round half to even,
strict '<' over seeds in increasing order), one program per tile of BB blocks, looping over all seeds.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _search_kernel(W_ptr, U_ptr, Pi_ptr, n_blocks, n_seeds, seed_base,
                   best_err_ptr, best_seed_ptr, best_q_ptr, best_e_ptr,
                   C: tl.constexpr, P: tl.constexpr, CP2: tl.constexpr, PP2: tl.constexpr,
                   BB: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    rb = pid * BB + tl.arange(0, BB)
    mb = rb < n_blocks
    rc = tl.arange(0, CP2)
    rp = tl.arange(0, PP2)
    mc = rc < C
    mp = rp < P
    W = tl.load(W_ptr + rb[:, None] * C + rc[None, :], mask=mb[:, None] & mc[None, :], other=0.0)   # (BB, CP2)
    best = tl.load(best_err_ptr + rb, mask=mb, other=float("inf"))
    bseed = tl.load(best_seed_ptr + rb, mask=mb, other=0)
    bq = tl.load(best_q_ptr + rb[:, None] * P + rp[None, :], mask=mb[:, None] & mp[None, :], other=0.0)
    be = tl.load(best_e_ptr + rb, mask=mb, other=0)
    mpc = mp[:, None] & mc[None, :]          # (PP2, CP2)
    mcp = mc[:, None] & mp[None, :]          # (CP2, PP2)
    for s in range(0, n_seeds):
        Pi = tl.load(Pi_ptr + s * (P * C) + rp[:, None] * C + rc[None, :], mask=mpc, other=0.0)   # (PP2, CP2)
        U = tl.load(U_ptr + s * (C * P) + rc[:, None] * P + rp[None, :], mask=mcp, other=0.0)     # (CP2, PP2)
        t = tl.sum(W[:, None, :] * Pi[None, :, :], axis=2)                                          # (BB, PP2)
        amax = tl.max(tl.abs(t), axis=1)                                                            # (BB,)
        zero = amax == 0.0
        if MODE == 0:      # paper: floor(log2 amax)
            bits = amax.to(tl.int32, bitcast=True)
            e = ((bits >> 23) & 0xFF) - 127
        else:              # range: ceil(log2(amax / 7.5))
            x = amax / 7.5
            bits = x.to(tl.int32, bitcast=True)
            e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        e = tl.minimum(tl.maximum(e, -8), 7)
        e = tl.where(zero, -8, e)
        inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
        scl = ((127 + e) << 23).to(tl.float32, bitcast=True)
        q = libdevice.rint(t * inv[:, None])
        q = tl.minimum(tl.maximum(q, -8.0), 7.0)
        q = tl.where(zero[:, None], 0.0, q)
        th = q * scl[:, None]                                                                       # (BB, PP2)
        R = tl.sum(th[:, None, :] * U[None, :, :], axis=2)                                          # (BB, CP2)
        D = W - R
        err = tl.sum(D * D, axis=1)
        take = err < best
        best = tl.where(take, err, best)
        bseed = tl.where(take, seed_base + s + 1, bseed)
        bq = tl.where(take[:, None], q, bq)
        be = tl.where(take, e, be)
    tl.store(best_err_ptr + rb, best, mask=mb)
    tl.store(best_seed_ptr + rb, bseed, mask=mb)
    tl.store(best_q_ptr + rb[:, None] * P + rp[None, :], bq, mask=mb[:, None] & mp[None, :])
    tl.store(best_e_ptr + rb, be, mask=mb)


def _p2(x):
    return 1 << (int(x) - 1).bit_length()


class GPUCodebook:
    """Seeds are processed in chunks (all seeds of K = 16 fit in one)."""

    def __init__(self, cb, device="cuda", chunk=1 << 20):
        self.K, self.C, self.P, self.n = cb.K, cb.C, cb.P, cb.n
        self.chunks = []
        for s0 in range(0, cb.n, chunk):
            self.chunks.append((s0, torch.from_numpy(cb.U[s0:s0 + chunk]).to(device),
                                torch.from_numpy(cb.Pi[s0:s0 + chunk]).to(device)))


def search(W, gcb, mode, BB=128, device="cuda", state=None):
    """W: (n_blocks, C) float32 array or tensor. Returns seeds (1-based), q, e, err as numpy."""
    Wt = torch.as_tensor(np.asarray(W, dtype=np.float32)).to(device).contiguous()
    nb = Wt.shape[0]
    C, P = gcb.C, gcb.P
    if state is None:
        err = torch.full((nb,), float("inf"), device=device, dtype=torch.float32)
        seed = torch.zeros(nb, device=device, dtype=torch.int32)
        q = torch.zeros((nb, P), device=device, dtype=torch.float32)
        e = torch.zeros(nb, device=device, dtype=torch.int32)
    else:
        err, seed, q, e = state
    grid = (triton.cdiv(nb, BB),)
    for s0, U, Pi in gcb.chunks:
        _search_kernel[grid](Wt, U, Pi, nb, U.shape[0], s0, err, seed, q, e,
                             C=C, P=P, CP2=_p2(C), PP2=_p2(P), BB=BB, MODE=0 if mode == "paper" else 1)
    return (seed.cpu().numpy().astype(np.int64), q.cpu().numpy().astype(np.int8),
            e.cpu().numpy().astype(np.int8), err.cpu().numpy())


def search_lazy(W, K, C, P, mode, chunk=1 << 18, BB=None, device="cuda"):
    """Same as search(), but builds U(s) and U(s)^+ chunk by chunk (for K up to 24 and larger C, P)."""
    import slm as S
    Wt = torch.as_tensor(np.asarray(W, dtype=np.float32)).to(device).contiguous()
    nb = Wt.shape[0]
    if BB is None:
        BB = 128 if _p2(C) * _p2(P) <= 64 else (64 if _p2(C) * _p2(P) <= 128 else 32)
    err = torch.full((nb,), float("inf"), device=device, dtype=torch.float32)
    seed = torch.zeros(nb, device=device, dtype=torch.int32)
    q = torch.zeros((nb, P), device=device, dtype=torch.float32)
    e = torch.zeros(nb, device=device, dtype=torch.int32)
    n = (1 << K) - 1
    grid = (triton.cdiv(nb, BB),)
    for s0 in range(0, n, chunk):
        seeds = np.arange(s0 + 1, min(s0 + chunk, n) + 1)
        U64 = torch.from_numpy(S.U_from_V(S.V_by_seed(K, C, P, seeds), K)).to(device)
        Pi = torch.linalg.pinv(U64).float().contiguous()
        U = U64.float().contiguous()
        _search_kernel[grid](Wt, U, Pi, nb, U.shape[0], s0, err, seed, q, e,
                             C=C, P=P, CP2=_p2(C), PP2=_p2(P), BB=BB, MODE=0 if mode == "paper" else 1)
    return (seed.cpu().numpy().astype(np.int64), q.cpu().numpy().astype(np.int8),
            e.cpu().numpy().astype(np.int8), err.cpu().numpy())
