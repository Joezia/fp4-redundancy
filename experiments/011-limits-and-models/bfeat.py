"""Spec 011 line B: cheap, decodable per-position features (numpy), shared by B1-B4.

All features of the code at (r, c) use only information decoded before it (w3 -> w1 -> w2, row-major, scales
first). Codes are the 15 merged symbols (0 = zero, 1..7 = +0.5..+6, 8..14 = -0.5..-6).
"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[2] / "data" / "011"
B = 32
NAMES = ["scale", "rel_scale", "prev1_mag", "prev1_sign", "prev2_mag", "max_so_far", "cnt4_so_far", "pos",
         "partner_mag", "partner_sign", "above_mag", "role", "layer", "mean_mag_so_far",
         "odd_so_far", "nz_so_far", "coarse_flag"]
CAUSAL = list(range(16))       # coarse_flag (whole block has no 0.5 / 1.5) is side information, not causal


def mag_sign(s):
    """merged symbol -> magnitude index 0..7 and sign (0 +, 1 -, 2 zero)."""
    mag = np.where(s == 0, 0, np.where(s <= 7, s, s - 7))
    sign = np.where(s == 0, 2, np.where(s <= 7, 0, 1))
    return mag, sign


class Matrices:
    def __init__(self):
        self.reg = {r["name"]: r for r in json.load(open(D / "registry.json"))}
        self.byid = {r["id"]: r for r in self.reg.values()}
        self._c = {}

    def codes(self, name):
        z = np.load(D / "codes" / f"{name}.npz")
        return z["sym"].astype(np.int64), z["E"]


def matrix_features(M, name):
    """features (R*W, F) int16 and targets (R*W,) for every position of a matrix (row-major order)."""
    S, E = M.codes(name)
    r = M.reg[name]
    X = None
    if r["cross"] is not None:
        X, _ = M.codes(M.byid[r["cross"]]["name"])
        if r["cross_kind"] == "transposed":
            X = X.T
    return features_from_arrays(S, E, X, r["role"], r["layer"])


def features_from_arrays(S, E, X, role, layer):
    R, W = S.shape
    mag, sign = mag_sign(S)
    F = np.zeros((R, W, len(NAMES)), dtype=np.int16)
    Eb = np.repeat(E.astype(np.int16), B, axis=1)
    F[..., 0] = Eb
    F[..., 1] = np.clip(Eb - np.round(E.astype(np.float64).mean(1))[:, None].astype(np.int16), -8, 8) + 8
    pm = np.full((R, W), 8, dtype=np.int16)
    pm[:, 1:] = mag[:, :-1]
    ps = np.full((R, W), 2, dtype=np.int16)
    ps[:, 1:] = sign[:, :-1]
    pm2 = np.full((R, W), 8, dtype=np.int16)
    pm2[:, 2:] = mag[:, :-2]
    F[..., 2], F[..., 3], F[..., 4] = pm, ps, pm2
    lv = mag.reshape(R, W // B, B)
    mx = np.maximum.accumulate(lv, axis=2)
    prev_max = np.zeros_like(lv)
    prev_max[:, :, 1:] = mx[:, :, :-1] + 1
    big = (lv >= 6).astype(np.int64)
    cnt = np.zeros_like(lv)
    cnt[:, :, 1:] = np.cumsum(big, axis=2)[:, :, :-1]
    pos = np.broadcast_to(np.arange(B), lv.shape)
    csum = np.zeros(lv.shape, dtype=np.float64)
    csum[:, :, 1:] = np.cumsum(lv, axis=2)[:, :, :-1]
    mean2 = np.where(pos == 0, 15, np.clip(np.round(2 * csum / np.maximum(pos, 1)), 0, 14))
    odd = np.isin(lv, [1, 3]).astype(np.int64)
    codd = np.zeros_like(lv)
    codd[:, :, 1:] = np.cumsum(odd, axis=2)[:, :, :-1]
    cnz = np.zeros_like(lv)
    cnz[:, :, 1:] = np.cumsum((lv > 0).astype(np.int64), axis=2)[:, :, :-1]
    F[..., 14] = np.minimum(codd, 8).reshape(R, W)
    F[..., 15] = cnz.reshape(R, W)
    F[..., 16] = np.repeat((odd.sum(-1) == 0).astype(np.int16), B, axis=1)
    F[..., 5] = prev_max.reshape(R, W)
    F[..., 6] = np.minimum(cnt, 8).reshape(R, W)
    F[..., 7] = pos.reshape(R, W)
    F[..., 13] = mean2.reshape(R, W)
    F[..., 8], F[..., 9] = 8, 2
    if X is not None:
        xm, xs = mag_sign(X)
        F[..., 8], F[..., 9] = xm, xs
    am = np.full((R, W), 8, dtype=np.int16)
    am[1:] = mag[:-1]
    F[..., 10] = am
    F[..., 11] = {"w1": 0, "w2": 1, "w3": 2}[role]
    F[..., 12] = layer
    return F.reshape(R * W, -1), S.ravel()


def sample_features(M, mids, idx):
    """features at given (matrix id, flat index) positions (grouped by matrix)."""
    order = np.argsort(mids, kind="stable")
    Fo = np.zeros((len(idx), len(NAMES)), dtype=np.int16)
    yo = np.zeros(len(idx), dtype=np.int64)
    m_sorted = mids[order]
    bounds = np.flatnonzero(np.diff(m_sorted)) + 1
    for grp in np.split(order, bounds):
        if len(grp) == 0:
            continue
        name = M.byid[int(mids[grp[0]])]["name"]
        F, y = matrix_features(M, name)
        Fo[grp] = F[idx[grp]]
        yo[grp] = y[idx[grp]]
    return Fo, yo
