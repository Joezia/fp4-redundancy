"""Break-even lines (spec 007 section 5). Gaussian approximations; used for drawing and screening.

N: symbols in the entry; b: bits per parameter; H: zero-order entropy of the symbols (bits).
"""
import math


def groups_mi(A, G, b, N, label_bits=0.0):
    """R2: a group variable pays when its MI exceeds the code-table cost (A-1)(G-1)b/N, plus,
    for labels the decoder does not have, H(labels) / symbols-per-label."""
    return (A - 1) * max(G - 1, 0) * b / N + label_bits


def pair_mi(A, b, N):
    """R3: a neighbour relation pays when its MI exceeds (A-1)·A·b/N."""
    return (A - 1) * A * b / N


AUTOCORR = 0.5  # R4: differencing starts to pay at a lag-1 correlation of 0.5


def flatness_bound(flatness):
    """R4: upper bound of linear-prediction gain, 1/2 log2(1 / spectral flatness)."""
    return 0.5 * math.log2(1.0 / flatness) if flatness > 0 else float("inf")


def layer_share(R, W, b, N):
    """R5: a rank-one layer pays when its energy share exceeds 1 - 2^(-2 (R+W) b / N)."""
    return 1.0 - 2.0 ** (-2.0 * (R + W) * b / N)


def repeat_length(N, H):
    """R6: a repeat pays when it is longer than log2(N) / H symbols."""
    return math.log2(N) / H if H > 0 else float("inf")


def short_match(n_cand, H, L):
    """R7a short segments: smallest m with m·H > log2(#candidates) + log2 C(L, m) (None if none)."""
    if n_cand <= 0:
        return None
    lc = math.log2(n_cand)
    for m in range(1, L + 1):
        if m * H > lc + math.log2(math.comb(L, m)):
            return m
    return None


def long_corr(n_seg, b, L):
    """R7a long segments: |corr| must exceed sqrt(1 - 2^(-2 (log2(#segments) + b) / L))."""
    return math.sqrt(1.0 - 2.0 ** (-2.0 * (math.log2(max(n_seg, 1)) + b) / L))


def cluster_ratio(K, L):
    """R7b: distortion ratio must be below 2^(-2 log2 K / L)."""
    return 2.0 ** (-2.0 * math.log2(K) / L)


def dictionary(K, s, b, L):
    """R8: explained share must exceed 1 - 2^(-2 s (log2 K + b) / L)."""
    return 1.0 - 2.0 ** (-2.0 * s * (math.log2(K) + b) / L)
