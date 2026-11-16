"""Shuffle baselines (spec 007 section 3.2). Each returns new (symbols, values) arrays; the same
permutation is applied to both, so values keep their symbols.

whole   - all symbols of the entry permuted
known   - symbols permuted only among units (or symbols) with the same `known` label
unit    - every unit permuted internally
Special baselines an adapter may name for one indicator:
random_sign - values multiplied by independent random signs (symbols unchanged)
misalign    - partner rows randomly permuted (used through `misalign_partner`)
"""
import numpy as np

ENTRY = object()  # default: take the array from the entry; pass None to skip an array


def _arrays(entry, symbols, values):
    return (entry.symbols if symbols is ENTRY else symbols), (entry.values if values is ENTRY else values)


def _take(X, idx):
    return None if X is None else X.ravel()[idx].reshape(X.shape)


def whole(entry, rng, symbols=ENTRY, values=ENTRY):
    s, v = _arrays(entry, symbols, values)
    idx = rng.permutation(s.size)
    return _take(s, idx), _take(v, idx)


def unit(entry, rng, symbols=ENTRY, values=ENTRY):
    s, v = _arrays(entry, symbols, values)
    if not entry.unit_len:
        return whole(entry, rng, s, v)
    L = entry.unit_len
    nu = s.size // L
    order = np.argsort(rng.random((nu, L)), axis=1)
    idx = (order + (np.arange(nu) * L)[:, None]).ravel()
    return _take(s, idx), _take(v, idx)


def known(entry, rng, symbols=ENTRY, values=ENTRY):
    s, v = _arrays(entry, symbols, values)
    if entry.known is None:
        return whole(entry, rng, s, v)
    lab = entry.known_per_symbol().ravel()
    idx = np.arange(s.size)
    order = np.argsort(lab, kind="stable")
    sl = lab[order]
    cut = np.flatnonzero(np.diff(sl)) + 1
    out = np.empty_like(idx)
    for seg in np.split(order, cut):
        out[seg] = seg[rng.permutation(len(seg))]
    return _take(s, out), _take(v, out)


def random_sign(entry, rng, symbols=ENTRY, values=ENTRY):
    s, v = _arrays(entry, symbols, values)
    if v is None:
        return s, None
    return s, v * np.where(rng.random(v.shape) < 0.5, -1.0, 1.0)


def misalign_partner(partner, rng):
    return partner[rng.permutation(partner.shape[0])]


BASELINES = {"whole": whole, "unit": unit, "known": known, "random_sign": random_sign}


def default_name(entry, preferred):
    """Resolve a preferred generic baseline to one the entry supports."""
    if preferred == "unit" and not entry.unit_len:
        return "whole"
    if preferred == "known" and entry.known is None:
        return "whole"
    return preferred


def pick(entry, indicator, preferred):
    """Baseline name for an indicator: the adapter's special choice, else the generic preference."""
    name = entry.baselines.get(indicator, preferred)
    return default_name(entry, name)


def run(name, entry, rng, symbols=ENTRY, values=ENTRY):
    return BASELINES[name](entry, rng, symbols, values)
