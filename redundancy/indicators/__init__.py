"""Indicator registry. Every module has `compute(entry, cfg, seed) -> {figure id: per-entry result}`
and `aggregate(figure id, [(entry name, result)], cfg, seed) -> {summary, verdict, detail, key}`.
A per-entry result {"applicable": False, "reason": ...} marks the indicator as not applicable."""
from . import r1, r2, r3, r4, r5, r6, r7a, r7b, r8, r9, r10

MODULES = {"R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5, "R6": r6, "R7a": r7a, "R7b": r7b,
           "R8": r8, "R9": r9, "R10": r10}
ORDER = ["R1", "R2", "R3", "R4", "R5", "R6", "R7a", "R7b", "R8", "R9", "R10"]
PATTERNS = {"R1": "R1", "R2": "R2", "R3": "R3", "R4": "R4", "R5": "R5", "R6": "R6", "R7a_unit": "R7",
            "R7a_row": "R7", "R7a_partner": "R7", "R7b": "R7", "R8": "R8", "R10": "R10"}


def pattern_of(fig):
    if fig.startswith("R9"):
        return "R9"
    return PATTERNS[fig]
