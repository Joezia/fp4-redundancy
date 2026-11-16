"""Synthetic data (spec 007 section 6): for every pattern one dataset without it ("none") and one
with it ("has"), shape 1024 x 2048, unit length 32. The constructions reproduce the ones already
validated in specs 001, 003 and 005. Symbols are 4-bit sign-magnitude codes of the grid
{0, 0.5, 1, 1.5, 2, 3, 4, 6} (code = 8 * sign + level); values are the grid values.

Each key names the figure it serves and the indicator that runs on it.
"""
import math

import numpy as np

from ..dataset import Entry, Group

GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
VALUES = np.concatenate([GRID, -GRID])
NAMES = [f"+{g:g}" for g in GRID] + [f"-{g:g}" for g in GRID]
EDGES = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
ODD = [1, 3, 9, 11]  # codes of magnitude 0.5 and 1.5
SIGMA = 1.68

# key -> (indicator, figure id served)
KEYS = {"R1": ("R1", "R1"), "R2": ("R2", "R2"), "R3": ("R3", "R3"), "R4": ("R4", "R4"),
        "R5": ("R5", "R5"), "R6": ("R6", "R6"), "R7a_unit": ("R7a", "R7a_unit"),
        "R7a_row": ("R7a", "R7a_row"), "R7a_partner": ("R7a", "R7a_partner"), "R7b": ("R7b", "R7b"),
        "R8": ("R8", "R8"), "R9_count_in_unit": ("R9", "R9_count_in_unit"),
        "R9_max_rank_in_unit": ("R9", "R9_max_rank_in_unit")}

DESCRIPTION = {
    ("R1", "none"): "16 symbols uniform", ("R1", "has"): "independent base",
    ("R2", "none"): "independent base, random known unit labels",
    ("R2", "has"): "two scales per unit (sigma 1.2 / 2.2), label = true unit type (known)",
    ("R3", "none"): "independent base", ("R3", "has"): "first-order Markov chain along rows",
    ("R4", "none"): "independent base", ("R4", "has"): "AR(1) along rows, rho = 0.7, quantized",
    ("R5", "none"): "independent base", ("R5", "has"): "rank-4 component, 0.3 of the variance",
    ("R6", "none"): "independent base", ("R6", "has"): "one 200-long stretch copied to 50 places",
    ("R7a_unit", "none"): "independent base",
    ("R7a_unit", "has"): "30% of units drawn from 16 prototypes (30% of positions redrawn)",
    ("R7a_row", "none"): "independent base", ("R7a_row", "has"): "20 rows are noisy copies (0.9) of 20 others",
    ("R7a_partner", "none"): "independent base, independent partner",
    ("R7a_partner", "has"): "partner rows correlated 0.5 with the same rows",
    ("R7b", "none"): "independent base", ("R7b", "has"): "as R7a unit: 16 prototypes",
    ("R8", "none"): "independent base", ("R8", "has"): "48 atoms, 2 per unit, signal share 0.6",
    ("R9_count_in_unit", "none"): "independent base; check: odd levels per unit",
    ("R9_count_in_unit", "has"): "30% of units use only even levels",
    ("R9_max_rank_in_unit", "none"): "independent base; check: largest level per unit",
    ("R9_max_rank_in_unit", "has"): "every unit scaled so that its largest |value| maps to 6",
}


def quantize(y):
    m = np.searchsorted(EDGES, np.abs(y))
    return (8 * (y < 0) + m).astype(np.uint8)


def code_probs(sigma):
    cdf = lambda x: 0.5 * (1.0 + math.erf(x / (sigma * math.sqrt(2.0))))
    lo = np.concatenate([[0.0], EDGES])
    hi = np.concatenate([EDGES, [np.inf]])
    pos = np.array([cdf(h) - cdf(l) for l, h in zip(lo, hi)])
    q = np.concatenate([pos, pos])
    return q / q.sum()


class Synthetic:
    def __init__(self, cfg):
        self.seed = int(cfg.get("seed", 20261116))
        self.R = int(cfg.get("rows", 1024))
        self.W = int(cfg.get("cols", 2048))
        self.L = int(cfg.get("unit_len", 32))
        self.keys = list(cfg.get("keys", KEYS))

    def tasks(self):
        return [(k, side) for k in self.keys for side in ("none", "has")]

    def _rng(self, key, side, part=0):
        import zlib
        return np.random.default_rng([self.seed, zlib.crc32(key.encode()), int(side == "has"), part])

    def base(self, g, shape=None):
        return quantize(SIGMA * g.standard_normal(shape or (self.R, self.W)))

    def entry(self, key, side):
        g = self._rng(key, side)
        R, W, L = self.R, self.W, self.L
        nu = R * W // L
        extra = {}
        X = None
        if key == "R1" and side == "none":
            X = g.integers(0, 16, size=(R, W)).astype(np.uint8)
        elif key == "R2":
            if side == "none":
                X = self.base(g)
                lab = g.integers(0, 2, size=(R, W // L))
            else:
                lab = (g.random((R, W // L)) < 0.5).astype(np.int64)
                sig = np.where(lab == 1, 2.2, 1.2)
                X = quantize(g.standard_normal((R, W // L, L)) * sig[:, :, None]).reshape(R, W)
            extra["groups"] = {"unit_type": Group(np.repeat(lab, L, axis=1), known=True)}
        elif key == "R3" and side == "has":
            P = 0.7 * g.dirichlet(np.full(16, 3.0), size=16) + 0.3 / 16
            w, v = np.linalg.eig(P.T)
            pi = np.real(v[:, np.argmin(np.abs(w - 1))])
            pi = pi / pi.sum()
            cP = np.cumsum(P, axis=1)
            cP[:, -1] = 1.0
            X = np.empty((R, W), dtype=np.uint8)
            X[:, 0] = np.searchsorted(np.cumsum(pi), g.random(R), side="right").clip(0, 15)
            for j in range(1, W):
                u = g.random(R)
                X[:, j] = (u[:, None] >= cP[X[:, j - 1]]).sum(axis=1)
        elif key == "R4" and side == "has":
            rho = 0.7
            x = np.empty((R, W))
            x[:, 0] = g.standard_normal(R)
            e = g.standard_normal((R, W)) * math.sqrt(1 - rho * rho)
            for j in range(1, W):
                x[:, j] = rho * x[:, j - 1] + e[:, j]
            X = quantize(SIGMA * x)
        elif key == "R5" and side == "has":
            U = g.standard_normal((R, 4))
            V = g.standard_normal((W, 4))
            X = quantize(SIGMA * (math.sqrt(0.3) * (U @ V.T) / 2 + math.sqrt(0.7) * g.standard_normal((R, W))))
        elif key == "R6" and side == "has":
            X = self.base(g)
            seg_len, taken, spots = 200, {}, []
            while len(spots) < 51:
                r, j = int(g.integers(R)), int(g.integers(W - seg_len + 1))
                if all(j + seg_len <= a or b <= j for a, b in taken.get(r, [])):
                    taken.setdefault(r, []).append((j, j + seg_len))
                    spots.append((r, j))
            (r0, j0), targets = spots[0], spots[1:]
            seg = X[r0, j0:j0 + seg_len].copy()
            for r, j in targets:
                X[r, j:j + seg_len] = seg
        elif key in ("R7a_unit", "R7b") and side == "has":
            X = self.base(g)
            protos = self.base(g, (16, L))
            blocks = X.reshape(-1, L)
            idx = np.flatnonzero(g.random(nu) < 0.3)
            pb = protos[g.integers(16, size=len(idx))]
            pb = np.where(g.random(pb.shape) < 0.3, self.base(g, pb.shape), pb)
            blocks[idx] = pb
        elif key == "R7a_row" and side == "has":
            Z = g.standard_normal((R, W))
            pick = g.choice(R, size=40, replace=False)
            for d, s in zip(pick[:20], pick[20:]):
                Z[d] = 0.9 * Z[s] + math.sqrt(0.19) * g.standard_normal(W)
            X = quantize(SIGMA * Z)
        elif key == "R7a_partner":
            Z = g.standard_normal((R, W))
            X = quantize(SIGMA * Z)
            if side == "has":
                P = quantize(SIGMA * (0.5 * Z + math.sqrt(0.75) * g.standard_normal((R, W))))
            else:
                P = self.base(g)
            extra["partner"] = VALUES[P]
            extra["partner_name"] = "partner"
        elif key == "R8" and side == "has":
            atoms = g.standard_normal((48, L))
            atoms /= np.sqrt((atoms * atoms).sum(axis=1, keepdims=True))
            a1 = g.integers(48, size=nu)
            a2 = (a1 + g.integers(1, 48, size=nu)) % 48
            c = g.standard_normal((nu, 2))
            sig = 4 * (c[:, :1] * atoms[a1] + c[:, 1:] * atoms[a2])
            y = SIGMA * (math.sqrt(0.6) * sig + math.sqrt(0.4) * g.standard_normal((nu, L)))
            X = quantize(y).reshape(R, W)
        elif key == "R9_count_in_unit":
            pn = code_probs(SIGMA)
            if side == "none":
                X = self.base(g)
            else:
                pe = pn.copy()
                pe[ODD] = 0.0
                pe /= pe.sum()
                z = g.random(nu) < 0.3
                X = np.empty((nu, L), dtype=np.uint8)
                X[~z] = g.choice(16, size=(int((~z).sum()), L), p=pn)
                X[z] = g.choice(16, size=(int(z.sum()), L), p=pe)
                X = X.reshape(R, W)
            extra["process_checks"] = [{"name": "count_in_unit", "kind": "count_in_unit", "symbols": ODD,
                                        "condition_on_known": True,
                                        "description": "number of symbols of magnitude 0.5 or 1.5 per unit"}]
        elif key == "R9_max_rank_in_unit":
            if side == "none":
                X = self.base(g)
            else:
                y = g.standard_normal((nu, L))
                y = y * (6.0 / np.abs(y).max(axis=1, keepdims=True))
                X = quantize(y).reshape(R, W)
            extra["process_checks"] = [{"name": "max_rank_in_unit", "kind": "max_rank_in_unit",
                                        "rank": [c & 7 for c in range(16)],
                                        "x_labels": [f"{v:g}" for v in GRID], "condition_on_known": True,
                                        "description": "largest magnitude level per unit"}]
        if X is None:
            X = self.base(g)
        eqs = [] if key == "R1" else [{"name": "+0 = -0", "symbol_classes": [[0, 8]]}]
        return Entry(name=f"{key}_{side}", symbols=X, alphabet=NAMES, raw_bits=4.0, values=VALUES[X],
                     unit_len=L, known=np.zeros((R, W // L), dtype=np.int64), equivalences=eqs,
                     attrs={"description": DESCRIPTION.get((key, side), "")}, **extra)
