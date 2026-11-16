"""Unit sample shared by R7b and R8: units of the most frequent `known` label (all units without
`known`), at most `n` of them in random order; first half fit, second half evaluate."""
import numpy as np


def unit_sample(entry, r, n):
    nu = entry.N // entry.unit_len
    if entry.known is not None:
        lab = entry.known.ravel()
        u, c = np.unique(lab, return_counts=True)
        main = u[np.argmax(c)]
        idx = np.flatnonzero(lab == main)
        label = int(main)
    else:
        idx = np.arange(nu)
        label = None
    pick = r.choice(idx, size=min(n, len(idx)), replace=False)
    return pick, label


def unit_values(V, L, pick):
    return V.reshape(-1, L)[pick].astype(np.float64)
