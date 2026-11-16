"""Unified data representation (spec 007 section 3.1).

A dataset is a list of entries; every entry is one 2-D array of symbols plus optional fields.
Missing optional fields make the indicators that need them fall back or report "not applicable";
they never raise.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Group:
    """A grouping variable.

    labels: integer array of the entry's shape (a broadcast view is fine), or, for scope "entry",
    a single integer shared by the whole entry (compared across entries, pooled).
    known: whether the decoder already has the labels. coverage: symbols covered by one label
    (needed when not known, for the label cost).
    """
    labels: object
    known: bool = True
    coverage: Optional[int] = None
    scope: str = "symbol"  # "symbol" or "entry"


@dataclass
class Entry:
    name: str
    symbols: np.ndarray            # (R, W) integers 0..A-1
    alphabet: list                 # A names
    raw_bits: float                # bits per symbol in the raw storage
    values: Optional[np.ndarray] = None    # (R, W) float, numeric meaning of each symbol
    groups: dict = field(default_factory=dict)  # name -> Group
    unit_len: Optional[int] = None  # local units = consecutive runs of unit_len symbols in a row
    known: Optional[np.ndarray] = None  # already-coded grouping: (R, W // unit_len) per unit, or (R, W)
    partner: Optional[np.ndarray] = None  # (R, W) values whose rows are candidates for our rows
    partner_name: Optional[str] = None
    equivalences: list = field(default_factory=list)
    process_checks: list = field(default_factory=list)
    baselines: dict = field(default_factory=dict)  # indicator id -> special baseline name
    attrs: dict = field(default_factory=dict)      # free metadata, copied into result files

    def __post_init__(self):
        self.symbols = np.ascontiguousarray(self.symbols)
        if self.symbols.ndim != 2:
            raise ValueError("symbols must be 2-D")
        A = len(self.alphabet)
        if not 1 <= A <= 256:
            raise ValueError("alphabet size must be in 1..256")
        if self.symbols.size and (self.symbols.min() < 0 or self.symbols.max() >= A):
            raise ValueError("symbols outside 0..A-1")
        if self.symbols.dtype != np.uint8:
            self.symbols = self.symbols.astype(np.uint8)
        R, W = self.symbols.shape
        if self.values is not None and self.values.shape != (R, W):
            raise ValueError("values must have the shape of symbols")
        if self.partner is not None and self.partner.shape != (R, W):
            raise ValueError("partner must have the shape of symbols")
        if self.unit_len is not None and (self.unit_len < 2 or W % self.unit_len):
            self.unit_len = None  # units must tile the rows exactly; otherwise treat as absent
        if self.known is not None:
            ok = (R, W // self.unit_len) if self.unit_len else (R, W)
            if self.known.shape != ok:
                raise ValueError(f"known must have shape {ok}")

    # ------------------------------------------------------------------ derived
    @property
    def A(self):
        return len(self.alphabet)

    @property
    def N(self):
        return int(self.symbols.size)

    @property
    def shape(self):
        return self.symbols.shape

    def canonical_map(self):
        """symbol -> representative of its equivalence class (equivalences of kind "symbols")."""
        m = np.arange(self.A)
        for eq in self.equivalences:
            for cls in eq.get("symbol_classes", []):
                rep = min(cls)
                for s in cls:
                    m[s] = rep
        return m.astype(np.uint8)

    def canonical(self, symbols=None):
        s = self.symbols if symbols is None else symbols
        m = self.canonical_map()
        return s if np.array_equal(m, np.arange(self.A)) else m[s]

    def known_per_symbol(self):
        if self.known is None:
            return None
        if self.unit_len:
            return np.repeat(self.known, self.unit_len, axis=1)
        return self.known

    def unit_view(self, X):
        """(R, W) -> (R * W // L, L) rows = units."""
        return X.reshape(-1, self.unit_len)


def entropy(counts):
    c = np.asarray(counts, dtype=np.float64).ravel()
    n = c.sum()
    if n <= 0:
        return float("nan")
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def mutual_information(table):
    """Plug-in mutual information (bits) of a 2-D count table."""
    t = np.asarray(table, dtype=np.float64)
    n = t.sum()
    if n <= 0:
        return float("nan")
    px, py = t.sum(axis=1), t.sum(axis=0)
    nz = t > 0
    return float((t[nz] / n * np.log2(t[nz] * n / np.outer(px, py)[nz])).sum())
