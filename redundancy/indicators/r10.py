"""R10: equivalent ways of storing the same content. No figure of its own: a table, and the
symbol classes are shaded in the R1 figure.

An equivalence is either a set of symbol classes (every occurrence of a symbol in a class of size k
has k equivalent spellings: log2 k bits each, counted from the data), or a declared count
(`log2_count` spread over `n_symbols`, both given by the adapter)."""
import math

import numpy as np

from .common import NONE, PRESENT, applicable, not_applicable_summary

FIGURES = ("R10",)


def compute(entry, cfg, seed):
    if not entry.equivalences:
        return {"R10": {"applicable": False, "reason": "no equivalences"}}
    counts = np.bincount(entry.symbols.ravel(), minlength=entry.A)
    rows = []
    for eq in entry.equivalences:
        classes = [sorted(int(s) for s in c) for c in eq.get("symbol_classes", [])]
        if classes:
            l2 = float(sum(counts[c].sum() * math.log2(len(c)) for c in classes))
            n = entry.N
        else:
            l2, n = float(eq["log2_count"]), int(eq["n_symbols"])
        rows.append({"name": eq["name"], "symbol_classes": classes, "log2_count": l2,
                     "n_symbols": n, "bits_per_symbol": l2 / n if n else None,
                     "note": eq.get("note", "")})
    return {"R10": {"rows": rows}}


def aggregate(fig, items, cfg, seed):
    items = applicable(items)
    if not items:
        return not_applicable_summary(items)
    names = [r["name"] for r in items[0][1]["rows"]]
    table = []
    for i, nm in enumerate(names):
        vals = [r["rows"][i] for _, r in items]
        table.append({"name": nm, "symbol_classes": vals[0]["symbol_classes"],
                      "log2_count": float(np.median([v["log2_count"] for v in vals])),
                      "bits_per_symbol": float(np.median([v["bits_per_symbol"] for v in vals])),
                      "note": vals[0]["note"]})
    tot = sum(t["bits_per_symbol"] for t in table)
    return {"summary": {"table": table, "total_bits_per_symbol": tot, "n_entries": len(items)},
            "verdict": PRESENT if tot > 0 else NONE,
            "detail": {"rule": "present when some equivalence has a positive count"},
            "key": f"{len(table)} equivalences, {tot:.4g} bit/symbol in total"}
