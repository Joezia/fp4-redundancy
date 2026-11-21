"""SeedLM (Shafipour et al., ICLR 2025), implemented from the paper text.

Sources (ICLR 2025 camera-ready, refs/012/seedlm_iclr2025.pdf; arXiv:2410.10714v2 agrees on every item):
  * LFSR update: Section 3.1 and Appendix A.3, Algorithm 1 -- feedback bit = XOR of the tap bits,
    state <- (state >> 1) | (bit << (K-1)); results[i] is the state *after* the i-th update.
  * taps: Appendix A.1, Table 6 (K = 16: (0, 1, 3, 12)).
  * V(s): the C*P states that follow the seed state s, filled row-wise into a C x P matrix
    (Figure 4 / arXiv Figure 3: "starts filling with the value generated one cycle after the seed";
    Algorithm 2 step 7 "row-wise order").
  * U(s) = (V(s) - 2^(K-1)) / (2^(K-1) - 1)       (Section 3.2, unnumbered eq. / arXiv Eq. 1).
  * t = U(s)^+ w, quantised to 4-bit two's complement integers with one shared 4-bit exponent,
    t_i = q_i 2^e, e = max_i floor(log2 |t_i|)                        (Section 3.2).
  * error eps_j = ||w - U(s_j) t_hat_j||^2, seed with the smallest error (Section 3.3, Eq. 2,
    Algorithm 3, strict '<' so the first seed in the order s_j = j wins ties).
  * pseudo-inverses are precomputed and cached for all seeds (Section 3.3).
  * bits per element M = (K + 4 + 4P) / C                              (Eq. 3).
  * blocks are C contiguous elements of the weight matrix (Section 3, first paragraph).

Two exponent rules are implemented (see report, "implementation decisions"):
  'paper' -- the formula as printed, e = floor(log2 max|t|) (q can only reach -2..2);
  'range' -- e = ceil(log2(max|t| / 7.5)), the smallest power of two that keeps round(t/2^e) inside
             [-8, 7]; this is what the stated dynamic range [-8*2^-8, 7*2^7] and the 4-bit integer imply.
In both, e is clipped to the 4-bit range [-8, 7] and q to [-8, 7].
"""
from __future__ import annotations

import numpy as np

TAPS = {2: (0, 1), 3: (0, 1), 4: (0, 1), 5: (0, 2), 6: (0, 1), 7: (0, 1), 8: (0, 2, 3, 4), 9: (0, 4),
        10: (0, 3), 11: (0, 2), 12: (0, 1, 2, 8), 13: (0, 1, 2, 5), 14: (0, 1, 2, 12), 15: (0, 1),
        16: (0, 1, 3, 12), 17: (0, 3), 18: (0, 7), 19: (0, 1, 2, 5), 20: (0, 3), 21: (0, 2), 22: (0, 1),
        23: (0, 5), 24: (0, 1, 2, 7)}
E_MIN, E_MAX = -8, 7
Q_MIN, Q_MAX = -8, 7


def lfsr_sequence(K, seed, length, taps=None):
    """Algorithm 1, literally."""
    taps = TAPS[K] if taps is None else taps
    out = np.empty(length, dtype=np.int64)
    state = int(seed)
    for i in range(length):
        bk = 0
        for t in taps:
            bk ^= (state >> t) & 1
        state = (state >> 1) | (bk << (K - 1))
        out[i] = state
    return out


def full_cycle(K):
    """States following state 1, vectorised version of Algorithm 1 (2^K - 1 entries, ends with 1)."""
    n = (1 << K) - 1
    taps = TAPS[K]
    out = np.empty(n, dtype=np.int64)
    state = 1
    # chunked Python loop is fast enough up to K = 24 (16.7M states, a few seconds with numba-free code)
    for i in range(n):
        bk = 0
        for t in taps:
            bk ^= (state >> t) & 1
        state = (state >> 1) | (bk << (K - 1))
        out[i] = state
    return out


_CYCLE = {}


def cycle(K):
    if K not in _CYCLE:
        import os
        cache = os.path.join(os.path.dirname(__file__), f"../../../data/012/lfsr_cycle_K{K}.npy")
        cache = os.path.abspath(cache)
        if os.path.exists(cache):
            _CYCLE[K] = np.load(cache)
        else:
            c = full_cycle(K)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.save(cache, c)
            _CYCLE[K] = c
    return _CYCLE[K]


def is_maximal(K):
    c = cycle(K)
    return len(np.unique(c)) == (1 << K) - 1 and c[-1] == 1


def V_by_seed(K, C, P, seeds=None):
    """V(s) for the given seeds (default all 1..2^K-1, in that order): shape (len(seeds), C, P)."""
    c = cycle(K)
    n = len(c)
    pos = np.empty(n + 1, dtype=np.int64)          # pos[state] = index of that state in the cycle
    pos[c] = np.arange(n)
    seeds = np.arange(1, n + 1) if seeds is None else np.asarray(seeds, dtype=np.int64)
    idx = (pos[seeds][:, None] + 1 + np.arange(C * P)[None, :]) % n
    return c[idx].reshape(len(seeds), C, P)


def U_from_V(V, K):
    half = 1 << (K - 1)
    return (V.astype(np.float64) - half) / (half - 1)


def shared_exponent(amax, mode):
    """amax > 0 (float32 array). Exact integer exponent rules via frexp (no log2 rounding)."""
    if mode == "paper":
        _, E = np.frexp(amax.astype(np.float32))
        return (E - 1).astype(np.int32)                       # floor(log2 amax)
    if mode == "range":
        x = (amax.astype(np.float32) / np.float32(7.5)).astype(np.float32)
        m, E = np.frexp(x)
        return (E - (m == 0.5)).astype(np.int32)              # ceil(log2(amax / 7.5))
    raise ValueError(mode)


def quantize(t, mode):
    """t (..., P) float32 -> q (..., P) int, e (...) int, t_hat (..., P) float32."""
    t = np.asarray(t, dtype=np.float32)
    amax = np.abs(t).max(-1)
    zero = amax == 0
    e = shared_exponent(np.where(zero, 1, amax), mode)
    e = np.clip(e, E_MIN, E_MAX)
    q = np.clip(np.rint(np.ldexp(t, -e[..., None])), Q_MIN, Q_MAX)
    q[zero] = 0
    e = np.where(zero, E_MIN, e)
    return q.astype(np.int8), e.astype(np.int8), np.ldexp(q, e[..., None]).astype(np.float32)


class Codebook:
    """U(s) and U(s)^+ for every seed s = 1..2^K-1 (row s-1), float32; pinv computed in float64."""

    def __init__(self, K, C, P, device=None):
        import torch
        self.K, self.C, self.P = K, C, P
        U = U_from_V(V_by_seed(K, C, P), K)
        if device is None:
            Pi = np.linalg.pinv(U)
        else:
            Pi = torch.linalg.pinv(torch.from_numpy(U).to(device)).cpu().numpy()
        self.U = np.ascontiguousarray(U.astype(np.float32))            # (N, C, P)
        self.Pi = np.ascontiguousarray(Pi.astype(np.float32))          # (N, P, C)

    @property
    def n(self):
        return len(self.U)


def search_numpy(W, cb, mode, block_chunk=512, seed_chunk=4096):
    """Reference implementation of Algorithm 3 for small inputs. W: (n_blocks, C) float32."""
    W = np.asarray(W, dtype=np.float32)
    nb = len(W)
    best_err = np.full(nb, np.inf, dtype=np.float32)
    best_s = np.zeros(nb, dtype=np.int64)
    best_q = np.zeros((nb, cb.P), dtype=np.int8)
    best_e = np.zeros(nb, dtype=np.int8)
    for b0 in range(0, nb, block_chunk):
        Wb = W[b0:b0 + block_chunk]
        for s0 in range(0, cb.n, seed_chunk):
            Pi = cb.Pi[s0:s0 + seed_chunk]
            U = cb.U[s0:s0 + seed_chunk]
            T = np.einsum("spc,bc->sbp", Pi, Wb)
            q, e, Th = quantize(T, mode)
            R = np.einsum("scp,sbp->sbc", U, Th) - Wb[None]
            err = (R * R).sum(-1)                       # (S, B)
            j = np.argmin(err, 0)                       # first minimum = smallest seed in chunk
            cand = err[j, np.arange(len(Wb))]
            take = cand < best_err[b0:b0 + len(Wb)]
            ii = np.nonzero(take)[0]
            best_err[b0 + ii] = cand[ii]
            best_s[b0 + ii] = s0 + j[ii] + 1
            best_q[b0 + ii] = q[j[ii], ii]
            best_e[b0 + ii] = e[j[ii], ii]
    return best_s, best_q, best_e, best_err


def reconstruct(seeds, q, e, K, C, P):
    """Algorithm 2: blocks = U(s) (q 2^e)."""
    U = U_from_V(V_by_seed(K, C, P, seeds), K).astype(np.float32)
    t = np.ldexp(q.astype(np.float32), e.astype(np.int32)[:, None])
    return np.einsum("ncp,np->nc", U, t)


def bits_per_element(C, P, K):
    return (K + 4 + 4 * P) / C
