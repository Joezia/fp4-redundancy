"""Generic adapter: any 2-D `.npy` array (spec 007 section 3.3).

Config keys (adapter section):
  files      list of .npy paths (one entry each) or {"name": path}
  axis       1 (default): distances run along rows; 0: along columns (the array is transposed)
  unit_len   optional fixed unit length along the distance axis
  bins       floats: number of equal-frequency bins (default 64)
  int_mode   integers with more than 256 distinct values: "quantile" (default) or "high_bits"
  int_values true: also give integer arrays as `values` (default false: integers are symbols only)
  raw_bits   bits per stored value (default: the dtype's size)
Floats keep the original numbers as `values`. No known, partner, equivalences or process_checks.
"""
import os

import numpy as np

from ..dataset import Entry, Group


def equal_frequency(x, bins):
    q = np.quantile(x, np.arange(1, bins) / bins)
    return np.searchsorted(q, x, side="right").astype(np.int64)


class ArrayFile:
    def __init__(self, cfg):
        f = cfg["files"]
        self.files = dict(f) if isinstance(f, dict) else {os.path.splitext(os.path.basename(p))[0]: p for p in f}
        self.cfg = cfg

    def entries(self):
        return list(self.files)

    def load(self, name):
        cfg = self.cfg
        X = np.load(self.files[name], allow_pickle=False)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2:
            raise ValueError("array_file: 1-D or 2-D arrays only")
        if cfg.get("axis", 1) == 0:
            X = X.T
        X = np.ascontiguousarray(X)
        raw_bits = float(cfg.get("raw_bits", X.dtype.itemsize * 8))
        values = None
        if np.issubdtype(X.dtype, np.floating):
            bins = int(cfg.get("bins", 64))
            sym = equal_frequency(X, bins)
            values = X.astype(np.float64)
            alphabet = [f"q{i}" for i in range(bins)]
        else:
            u, inv = np.unique(X, return_inverse=True)
            inv = inv.reshape(X.shape)
            if len(u) <= 256:
                sym = inv
                alphabet = [str(v) for v in u]
            elif cfg.get("int_mode", "quantile") == "high_bits":
                span = int(X.max()) - int(X.min())
                shift = max(0, int(span).bit_length() - 8)
                sym = (X.astype(np.int64) - int(X.min())) >> shift
                alphabet = [f"h{i}" for i in range(int(sym.max()) + 1)]
            else:
                sym = equal_frequency(X.astype(np.float64), 256)
                alphabet = [f"q{i}" for i in range(256)]
            if cfg.get("int_values", False):
                values = X.astype(np.float64)
        R, W = sym.shape
        groups = {"row": Group(np.broadcast_to(np.arange(R)[:, None], (R, W)), known=True),
                  "column": Group(np.broadcast_to(np.arange(W)[None, :], (R, W)), known=True)}
        return Entry(name=name, symbols=sym, alphabet=alphabet, raw_bits=raw_bits, values=values,
                     groups=groups, unit_len=cfg.get("unit_len"), attrs={"file": self.files[name]})
