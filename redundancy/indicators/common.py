"""Shared helpers of the indicators: random streams, the fixed judgement rule (spec 007 section 5),
and aggregation of per-entry curves into median / min / max.

Verdicts: "none", "present" (exceeds the baseline but no point beyond the break-even line),
"present_breakeven" (exceeds the baseline and some point is beyond the line), "not_applicable".
R1 and R9 use only "none" / "present".
"""
import zlib

import numpy as np

NONE, PRESENT, PASS, NA = "none", "present", "present_breakeven", "not_applicable"
REPS = 3


def rng(seed, *parts):
    """Independent stream for (seed, parts); string parts are hashed (crc32)."""
    key = [int(seed)]
    for p in parts:
        key.append(zlib.crc32(p.encode()) if isinstance(p, str) else int(p))
    return np.random.default_rng(key)


def na(reason):
    return {"applicable": False, "reason": reason}


def exceed(real, base_reps, min_amp, sign=1):
    """Fixed rule: a point exceeds the baseline when (real - baseline mean), in the direction `sign`
    (+1: real larger, -1: real smaller, 0: either), is larger than 5 x the baseline standard
    deviation over the replicates and larger than `min_amp`.
    real: (P,) values; base_reps: (reps, P). Returns (mask, diff, mean, sd) as arrays."""
    real = np.atleast_1d(np.asarray(real, dtype=np.float64))
    B = np.atleast_2d(np.asarray(base_reps, dtype=np.float64))
    mu = B.mean(axis=0)
    sd = B.std(axis=0, ddof=1) if B.shape[0] > 1 else np.zeros_like(mu)
    d = real - mu
    diff = np.abs(d) if sign == 0 else sign * d
    ok = (diff > 5.0 * sd) & (diff > min_amp) & np.isfinite(diff)
    return ok, diff, mu, sd


def exceed_stats(real, mu, sd, min_amp, sign=1, curve=False):
    """Same rule from precomputed baseline mean and sd (used on medians across entries).
    curve=True (indicators drawn as a curve): the per-point sd of 3 replicates has only 2 degrees
    of freedom and is often far too small at some point of a long curve, so the sd used at each
    point is the larger of its own and the median sd over the curve."""
    real, mu, sd = (np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in (real, mu, sd))
    if curve and sd.size > 1 and np.isfinite(sd).any():
        sd = np.maximum(sd, np.nanmedian(sd))
    d = real - mu
    diff = np.abs(d) if sign == 0 else sign * d
    return (diff > 5.0 * sd) & (diff > min_amp) & np.isfinite(diff), diff


def verdict(exists, passes=False, two_level=False):
    if not exists:
        return NONE
    if two_level:
        return PRESENT
    return PASS if passes else PRESENT


def band(arrs):
    """Per-point median, min, max across entries (lists of equal length; NaN ignored)."""
    A = np.asarray(arrs, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    with np.errstate(all="ignore"):
        return {"median": nanlist(np.nanmedian(A, axis=0)), "min": nanlist(np.nanmin(A, axis=0)),
                "max": nanlist(np.nanmax(A, axis=0))}


def nanlist(x):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(x, dtype=np.float64).ravel()]


def fnum(x):
    return None if x is None or not np.isfinite(x) else float(x)


def applicable(items):
    return [(n, r) for n, r in items if r.get("applicable", True)]


def not_applicable_summary(items):
    reasons = sorted({r.get("reason", "") for _, r in items})
    return {"summary": {}, "verdict": NA, "detail": {"reason": "; ".join(reasons)}, "key": ""}


def med_baseline(items, real_key, mu_key, sd_key):
    """Median across entries of the real curve, the baseline mean and the baseline sd."""
    R = np.median(np.asarray([r[real_key] for _, r in items], dtype=np.float64), axis=0)
    M = np.median(np.asarray([r[mu_key] for _, r in items], dtype=np.float64), axis=0)
    S = np.median(np.asarray([r[sd_key] for _, r in items], dtype=np.float64), axis=0)
    return R, M, S


def mean_sd(reps):
    B = np.asarray(reps, dtype=np.float64)
    return B.mean(axis=0), (B.std(axis=0, ddof=1) if B.shape[0] > 1 else np.zeros(B.shape[1:]))
