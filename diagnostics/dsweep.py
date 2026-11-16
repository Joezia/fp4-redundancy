"""Spec 006: block dictionary sweep over dictionary size K and atoms per block s (statistics only).

Generalises 003-R8.2 (`r58.omp`, `r58.dict_learn`, `r58.r82_variant`): the dictionary is learned
with the same s it is evaluated with (003 learned with s = 2 only), K in {16, 64, 256, 1024},
s in {1, 2, 4}. Same sample as 003: main-scale blocks, up to 100,000 drawn at random, first half fit,
second half evaluate, v domain, not centred. Baselines S0E and S2 (one shuffle each, same streams as
003's first replicate) redo the whole pipeline.

Random streams: the block sample and the shuffles use 003's streams (`r58.item_rng` with the 003
matrix index as seed), so the blocks are the same as in 003. The dictionary initialisation for
(K = 64, s = 2) uses 003's stream as well; the other (K, s) use a stream that adds (K, s).
"""
import math

import numpy as np

from . import r58
from .measure import E2M1

B = 32
KS = (16, 64, 256, 1024)
SS = (1, 2, 4)
ITERS = 15
BITS = (16, 8)
VARIANTS = ("real", "S0E", "S2")
# index of the variant in 003's r72_r82_matrix variant dict (real, S0E_1, S0E_2, S2_1, S2_2)
VI_003 = {"real": 0, "S0E": 1, "S2": 3}


def breakeven(K, s, b, L=B):
    """R2* = 1 - 2^(-2 s (log2 K + b) / L): the explained share that pays for s indices and s
    coefficients of b bits per block of L values (Gaussian approximation)."""
    return 1.0 - 2.0 ** (-2.0 * s * (math.log2(K) + b) / L)


def _renorm(D, Xf, rng):
    n = len(Xf)
    nrm = np.sqrt((D * D).sum(axis=1))
    for i in np.nonzero(nrm <= 0)[0]:
        while True:
            x = Xf[rng.integers(n)]
            if (x * x).sum() > 0:
                D[i] = x
                nrm[i] = math.sqrt(float((x * x).sum()))
                break
    return D / nrm[:, None]


def dict_learn(Xf, rng, K, s, iters=ITERS):
    """003's alternating scheme with K atoms and s atoms per block: OMP(s) coding, then the
    least-squares dictionary update D = (C^T C + 1e-6 I)^-1 C^T X (C^T C and C^T X accumulated from
    the s nonzeros per block instead of the dense code matrix), then renormalisation."""
    n = len(Xf)
    D = _renorm(Xf[rng.choice(n, size=K, replace=False)].copy(), Xf, rng)
    for _ in range(iters):
        sel, coef, _ = r58.omp(Xf, D, s)
        G = np.zeros((K, K))
        for a in range(s):
            for b in range(s):
                np.add.at(G, (sel[:, a], sel[:, b]), coef[:, a] * coef[:, b])
        CX = np.zeros((K, Xf.shape[1]))
        for a in range(s):
            np.add.at(CX, sel[:, a], coef[:, a, None] * Xf)
        D = np.linalg.solve(G + 1e-6 * np.eye(K), CX)
        D = _renorm(D, Xf, rng)
    return D


def sweep_codes(codes, seed, vi, ks=KS, ss=SS):
    """R2(K, s) on the evaluation half and R2_pca(s) for one variant. codes: (n, 32) C'."""
    n = len(codes)
    nf = n // 2
    X = E2M1[codes]
    Xf, Xe = X[:nf], X[nf:]
    ee = float((Xe * Xe).sum())
    out = {"dict": {}, "pca": {}, "n_fit": nf, "n_eval": n - nf}
    for K in ks:
        for s in ss:
            if (K, s) == (64, 2):
                rng = np.random.default_rng([r58.SEED, int(seed), r58._ITEM["R8.2"], vi])
            else:
                rng = np.random.default_rng([r58.SEED, int(seed), r58._ITEM["R8.2"], vi, K, s])
            D = dict_learn(Xf, rng, K, s)
            _, _, resid = r58.omp(Xe, D, s)
            out["dict"][(K, s)] = 1.0 - float((resid * resid).sum()) / ee
    lam, V = np.linalg.eigh(Xf.T @ Xf / nf)
    V = V[:, np.argsort(lam)[::-1]]
    for s in ss:
        Vs = V[:, :s]
        out["pca"][s] = 1.0 - float(((Xe - Xe @ Vs @ Vs.T) ** 2).sum()) / ee
    return out


def variant_codes(Cp, E, seed, variant):
    """The sampled blocks of one variant (same sample and shuffle streams as 003)."""
    pick, main = r58.main_sample(E, r58.item_rng(seed, "sample"))
    if variant == "real":
        X = Cp
    else:
        X = r58.shuffle(Cp, E, variant, r58.item_rng(seed, "R7.2", 1, int(variant == "S2")))[0]
    return X.reshape(-1, B)[pick], main


def sweep_variant(Cp, E, seed, variant, ks=KS, ss=SS):
    codes, main = variant_codes(Cp, E, seed, variant)
    res = sweep_codes(codes, seed, VI_003[variant], ks, ss)
    res["main"] = main
    return res
