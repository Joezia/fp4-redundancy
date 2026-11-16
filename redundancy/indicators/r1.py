"""R1: symbol probabilities are not uniform. Two-level verdict: present when raw_bits - H exceeds
the configured minimum amplitude (no shuffle baseline: shuffles keep the histogram)."""
import numpy as np

from ..dataset import entropy
from .common import NONE, PRESENT, applicable, band, not_applicable_summary

FIGURES = ("R1",)


def compute(entry, cfg, seed):
    counts = np.bincount(entry.symbols.ravel(), minlength=entry.A)
    H = entropy(counts)
    return {"R1": {"counts": counts.tolist(), "N": entry.N, "H": H, "raw_bits": entry.raw_bits,
                   "saving": entry.raw_bits - H, "alphabet": list(entry.alphabet)}}


def aggregate(fig, items, cfg, seed):
    items = applicable(items)
    if not items:
        return not_applicable_summary(items)
    A = max(len(r["counts"]) for _, r in items)
    probs = [np.pad(np.asarray(r["counts"], dtype=np.float64) / r["N"], (0, A - len(r["counts"])))
             for _, r in items]
    H = float(np.median([r["H"] for _, r in items]))
    sav = float(np.median([r["saving"] for _, r in items]))
    raw = float(np.median([r["raw_bits"] for _, r in items]))
    alphabet = max((r["alphabet"] for _, r in items), key=len)
    same = all(r["alphabet"] == alphabet for _, r in items)
    exists = sav > cfg.get("min_amp", 0.01)
    return {"summary": {"alphabet": alphabet, "prob": band(probs), "uniform": 1.0 / len(alphabet),
                        "H": H, "saving": sav, "raw_bits": raw, "n_entries": len(items)},
            "verdict": PRESENT if exists else NONE,
            "detail": {"rule": "raw_bits - H > min_amp", "min_amp": cfg.get("min_amp", 0.01),
                       "alphabets_identical": same},
            "key": f"H = {H:.3f} bit, saving = {sav:.3f}"}
