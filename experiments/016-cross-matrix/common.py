"""Spec 016 (cross-matrix regularities): shared reading and GPU helpers.

Matrix lists are the full-model lists written by 015 (`data/015/{model}_matrices.json`, made by
redundancy.adapters.fp4.list_matrices); codes and scales are read with redundancy.adapters.fp4.FP4._read.
Codes: uint8 0..15 (E2M1, bit 3 = sign, bits 2-1 = exponent, bit 0 = mantissa); scales: E8M0 bytes, one per 32
codes along a row.
"""
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = HERE / "results"
DATA = ROOT / "data/016"
LISTS = ROOT / "data/015"
MODELS = {"deepseek": dict(env="SNAPSHOT", label="DeepSeek-V4.1-Flash"),
          "kimi": dict(env="KIMI_SNAPSHOT", label="Kimi-K3")}
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                dtype=np.float32)
PLANES = ["sign", "e1", "e0", "m"]          # bit 3, 2, 1, 0
_READERS = {}
_ENTRIES = {}


def entries(model):
    if model not in _ENTRIES:
        _ENTRIES[model] = json.load(open(LISTS / f"{model}_matrices.json"))["matrices"]
    return _ENTRIES[model]


def reader(model):
    if model not in _READERS:
        from redundancy.adapters.fp4 import FP4
        _READERS[model] = FP4({"samples": str(LISTS / f"{model}_matrices.json"), "snapshot_env": MODELS[model]["env"]})
    return _READERS[model]


def load(model, e):
    """codes (R, W) uint8 and scales (R, W/32) uint8 of one list entry."""
    C, E = reader(model)._read(e)
    return np.ascontiguousarray(C, dtype=np.uint8), np.ascontiguousarray(E, dtype=np.uint8)


def load_packed(model, e):
    """raw packed bytes (R, W/2) and scales, without unpacking (for streaming)."""
    from diagnostics.st import read_bytes
    ad = reader(model)
    if ad._index is None:
        from diagnostics.st import index_snapshot
        ad._index = index_snapshot(ad.snapshot)
    P = np.ascontiguousarray(read_bytes(ad._index[e["weight"]]), dtype=np.uint8)
    E = np.ascontiguousarray(read_bytes(ad._index[e["scale"]]), dtype=np.uint8)
    return P, E


def unpack(P):
    R, H = P.shape
    C = np.empty((R, 2 * H), dtype=np.uint8)
    C[:, 0::2] = P & 15
    C[:, 1::2] = P >> 4
    return C


def neuron_major(C, E, role):
    """orient as (neurons, hidden): w1 / w3 as stored, w2 transposed (scales expanded to elements)."""
    Ef = np.repeat(E, 32, axis=1)
    if role == "w2":
        return np.ascontiguousarray(C.T), np.ascontiguousarray(Ef.T)
    return C, Ef


# ------------------------------------------------------------------ code lengths
def kt_bits(counts, K):
    """Dirichlet-1/2 (KT) sequential code length in bits of a symbol sequence with these counts; order-free.
    counts: array (..., K) -> array (...)."""
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum(axis=-1)
    from scipy.special import gammaln
    nats = gammaln(n + K / 2) - gammaln(K / 2) - (gammaln(c + 0.5) - gammaln(0.5)).sum(axis=-1)
    return nats / math.log(2)


def cond_bits(ctx, sym, n_ctx, K):
    """sum over contexts of the KT code length of `sym` within each context (adaptive table per context)."""
    t = np.bincount(ctx.astype(np.int64) * K + sym.astype(np.int64), minlength=n_ctx * K).reshape(n_ctx, K)
    return float(kt_bits(t, K).sum())


# ------------------------------------------------------------------ GPU signals
def gpu_signals(torch, C, E=None, names=("eq", "sign", "e1", "e0", "m", "v", "absv", "w"), device="cuda"):
    """dict name -> float32 tensor (channels, R, W) on GPU. eq: 16 one-hot channels; planes: +-1;
    v: E2M1 value; absv: |v|; w: dequantized value (needs E expanded to elements)."""
    Ct = torch.as_tensor(C, device=device).long()
    out = {}
    if "eq" in names:
        out["eq"] = torch.nn.functional.one_hot(Ct, 16).permute(2, 0, 1).float()
    for k, nm in enumerate(PLANES):
        if nm in names:
            out[nm] = (((Ct >> (3 - k)) & 1).float() * 2 - 1)[None]
    lut = torch.as_tensor(E2M1, device=device)
    if "v" in names or "absv" in names or "w" in names:
        v = lut[Ct]
        if "v" in names:
            out["v"] = v[None]
        if "absv" in names:
            out["absv"] = v.abs()[None]
        if "w" in names:
            Et = torch.as_tensor(E, device=device).float()
            out["w"] = (v * torch.exp2(Et - 127))[None]
    return out
