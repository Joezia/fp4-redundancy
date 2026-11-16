"""FP4 adapter: DeepSeek-style FP4 (E2M1) expert weights with E8M0 block scales.

The only module of the framework that knows the format. It reads tensors through the verified
readers of `diagnostics/` (safetensors index, packed-nibble unpacking) and maps one weight matrix to
one `Entry`:

symbols   the 4-bit codeword C in 0..15 (bit 3 = sign), raw_bits = 4
values    the E2M1 code value (without the scale), as in specs 001-006
unit      the 32-codeword scale block; known = the block's E8M0 scale byte
groups    scale (known), row, column, position in block, layer and role (entry-level, known),
          block maximum level (unknown, one label per 32 codewords)
partner   for w1: the w3 matrix of the same expert (row i of w3 is the candidate for row i of w1)
equivalences  +0 / -0; permutation of the expert's neurons; paired sign flip (w3 row, w2 column)
process_checks  odd levels (0.5, 1.5) per block; largest level per block
baselines R5 uses a sign-only shuffle
"""
import json
import math
import os

import numpy as np

from ..dataset import Entry, Group

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
NAMES = ["+0", "+0.5", "+1", "+1.5", "+2", "+3", "+4", "+6", "-0", "-0.5", "-1", "-1.5", "-2", "-3", "-4", "-6"]
ROLES = {"w1": 1, "w2": 2, "w3": 3}
B = 32


class FP4:
    def __init__(self, cfg):
        snap = cfg.get("snapshot") or os.environ.get(cfg.get("snapshot_env", "SNAPSHOT"), "")
        if not snap:
            raise SystemExit("FP4 adapter: set the snapshot path (config 'snapshot' or env var)")
        self.snapshot = snap
        with open(cfg["samples"]) as f:
            self.samples = {s["matrix"]: s for s in json.load(f)["matrices"]}
        self.names = list(cfg.get("entries") or self.samples)
        self.partner = cfg.get("partner", {"w1": "w3"})
        self._index = None

    def entries(self):
        return list(self.names)

    def _read(self, s):
        from diagnostics.measure import unpack_codes
        from diagnostics.st import index_snapshot, read_bytes
        if self._index is None:
            self._index = index_snapshot(self.snapshot)
        C = unpack_codes(read_bytes(self._index[s["weight"]]))
        E = read_bytes(self._index[s["scale"]])
        assert list(C.shape) == s["codeword_shape"] and E.shape == (C.shape[0], C.shape[1] // B)
        return C, np.ascontiguousarray(E)

    def load(self, name):
        s = self.samples[name]
        C, E = self._read(s)
        R, W = C.shape
        role = s["role"]
        partner = partner_name = None
        if role in self.partner:
            pn = f"L{s['layer']}.E{s['expert']}.{self.partner[role]}"
            if pn in self.samples:
                Cp, _ = self._read(self.samples[pn])
                partner, partner_name = E2M1[Cp], pn
        lvl = (C & 7).reshape(R, W // B, B).max(axis=2)
        groups = {
            "scale": Group(np.repeat(E, B, axis=1), known=True),
            "row": Group(np.broadcast_to(np.arange(R)[:, None], (R, W)), known=True),
            "column": Group(np.broadcast_to(np.arange(W)[None, :], (R, W)), known=True),
            "position_in_block": Group(np.broadcast_to((np.arange(W) % B)[None, :], (R, W)), known=True),
            "layer": Group(int(s["layer"]), known=True, scope="entry"),
            "role": Group(ROLES[role], known=True, scope="entry"),
            "block_max_level": Group(np.repeat(lvl, B, axis=1), known=False, coverage=B),
        }
        n_neurons = R if role in ("w1", "w3") else W
        expert_symbols = 3 * R * W
        eqs = [
            {"name": "+0 = -0", "symbol_classes": [[0, 8]], "note": "sign of a zero codeword"},
            {"name": "neuron permutation", "log2_count": math.lgamma(n_neurons + 1) / math.log(2),
             "n_symbols": expert_symbols,
             "note": f"{n_neurons}! orders of the expert's neurons (w1/w3 rows with w2 columns), "
                     "spread over the expert's 3 matrices"},
            {"name": "paired sign flip", "log2_count": float(n_neurons), "n_symbols": expert_symbols,
             "note": "flip w3 row i and w2 column i together; spread over the expert's 3 matrices"},
        ]
        checks = [
            {"name": "odd_count", "kind": "count_in_unit", "symbols": [1, 3, 9, 11],
             "condition_on_known": True,
             "description": "codes of magnitude 0.5 or 1.5 per block; expected binomial per scale"},
            {"name": "max_level", "kind": "max_rank_in_unit", "rank": [c & 7 for c in range(16)],
             "x_labels": ["0", "0.5", "1", "1.5", "2", "3", "4", "6"], "condition_on_known": True,
             "description": "largest magnitude per block; expected for independent codes per scale"},
        ]
        return Entry(name=name, symbols=C, alphabet=NAMES, raw_bits=4.0, values=E2M1[C], groups=groups,
                     unit_len=B, known=E, partner=partner, partner_name=partner_name, equivalences=eqs,
                     process_checks=checks, baselines={"R5": "random_sign"},
                     attrs={"layer": s["layer"], "expert": s["expert"], "role": role, "shape": [R, W]})
