"""Synthetic checks V1-V5 of the incremental-saving module (spec 008 Part 0).

    python -m redundancy.validate_increment --out <validation.json> [--workers N] [--image <png>] [--audio <flac>]

Every check states its theory value, measured value and tolerance; `pass` is the comparison.
Increments are in % of raw bits.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from . import increment as I  # noqa: E402

ORDER_1D = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R10"]
ALT_1D = ["R10", "R1", "R2", "R4", "R3", "R5", "R6", "R7", "R8"]
ZERO_TOL = 0.2      # % of raw bits: "about zero" (learning cost of extra contexts and side information)
N = 2_000_000


def dlaplace_entropy(b):
    q = math.exp(-1.0 / b)
    ks = np.arange(-int(60 * b) - 10, int(60 * b) + 11)
    p = (1 - q) / (1 + q) * q ** np.abs(ks)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def dlaplace(rng, b, n):
    # difference of two geometric variables = discrete Laplace with P(k) ~ q^|k|
    q = math.exp(-1.0 / b)
    return (rng.geometric(1 - q, n) - rng.geometric(1 - q, n)).astype(np.int64)


def dgauss_entropy(sig):
    ks = np.arange(-int(12 * sig) - 5, int(12 * sig) + 6)
    p = np.exp(-0.5 * (ks / sig) ** 2)
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()), ks, p


def chain(x, order, P=None):
    P = P or dict(I.AUDIO_DEFAULT)
    x = np.asarray(x, dtype=np.int64)
    if x.ndim == 1:
        x = x[None, :]
    return I.run_chain(lambda f: I.audio_codelen(x, f, P), order, 16 * x.size)


FORCED = {}


def pct(r):
    FORCED.setdefault("last", []).append({k: 100 * v for k, v in r["if_forced"].items()})
    return {k: 100 * v for k, v in r["increment"].items()}


def zeros_ok(inc, skip):
    bad = {k: v for k, v in inc.items() if k not in skip and abs(v) > ZERO_TOL}
    return not bad, bad


def v1(seed):
    rng = np.random.default_rng(seed)
    b = 40.0
    x = dlaplace(rng, b, N)
    H = dlaplace_entropy(b)
    r = chain(x, ORDER_1D)
    inc = pct(r)
    th = 100 * (16 - H) / 16
    ok0, bad = zeros_ok(inc, {"R1"})
    tol = 0.05
    return {"name": "V1 i.i.d. discrete Laplace (scale 40), 2e6 samples, 16-bit",
            "theory": {"R1": th, "others": 0.0}, "measured": inc,
            "tolerance": {"R1": f"|measured - theory| <= {tol} % of raw bits", "others": f"|x| <= {ZERO_TOL} %"},
            "pass": bool(abs(inc["R1"] - th) <= tol and ok0), "violations": bad,
            "note": f"theory R1 = (16 - H) / 16 with H = {H:.4f} bit (entropy of the discrete Laplace law)"}


def v2(seed):
    rng = np.random.default_rng(seed)
    rho, sig = 0.9, 40.0
    n = N // 2
    w = rng.normal(0, sig, n)
    # AR(1) by an IIR filter written as a cumulative product-free recursion (lfilter is not available)
    y = np.empty(n)
    acc = 0.0
    for t in range(n):
        acc = rho * acc + w[t]
        y[t] = acc
    x = np.rint(y).astype(np.int64)
    th = 100 * 0.5 * math.log2(1 / (1 - rho * rho)) / 16
    alt = pct(chain(x, ALT_1D))
    dft = pct(chain(x, ORDER_1D))
    tol = 0.15
    ok = abs(alt["R4"] - th) <= tol and abs(dft["R3"] + dft["R4"] - th) <= tol
    return {"name": "V2 quantised Gaussian AR(1), rho = 0.9, innovation sd 40, 1e6 samples",
            "theory": {"R4": th}, "measured": {"alt_order_R4_before_R3": alt, "default_order": dft},
            "tolerance": f"|R4 - theory| <= {tol} % in the order with R4 before R3; in the default order R3 (quantised "
                         f"previous-sample context) comes first and takes part of the same dependence, so R3 + R4 is "
                         f"compared with the theory, same tolerance",
            "pass": bool(ok),
            "note": "theory = 1/2 log2(1 / (1 - rho^2)) bit per sample / 16 bit"}


def v3(seed):
    rng = np.random.default_rng(seed)
    F = 1024
    nf = N // F
    s1, s2 = 4.0, 64.0
    pick = rng.integers(0, 2, nf)
    sig = np.where(pick == 0, s1, s2)
    x = np.rint(rng.normal(0, 1, (nf, F)) * sig[:, None]).astype(np.int64).ravel()
    H1, k1, p1 = dgauss_entropy(s1)
    H2, k2, p2 = dgauss_entropy(s2)
    ks = np.arange(min(k1.min(), k2.min()), max(k1.max(), k2.max()) + 1)
    pm = np.zeros(len(ks))
    pm[k1 - ks[0]] += 0.5 * p1
    pm[k2 - ks[0]] += 0.5 * p2
    pm = pm[pm > 0]
    Hm = float(-(pm * np.log2(pm)).sum())
    label = 1.0 / F
    th = 100 * (Hm - 0.5 * (H1 + H2) - label) / 16
    inc = pct(chain(x, ORDER_1D))
    tol = 0.1 * th
    return {"name": "V3 piecewise variance: each 1024-sample frame Gaussian with sd 4 or 64 (random), 2e6 samples",
            "theory": {"R2": th}, "measured": inc, "tolerance": f"|R2 - theory| <= 10 % of theory = {tol:.3f} %",
            "pass": bool(abs(inc["R2"] - th) <= tol),
            "note": f"theory = [H(mixture) - mean H(component) - 1 bit label per frame] / 16 with H(mixture) = {Hm:.4f}, "
                    f"H = {H1:.4f} / {H2:.4f} bit"}


def v4(seed):
    rng = np.random.default_rng(seed)
    b = 40.0
    x = dlaplace(rng, b, N)
    H = dlaplace_entropy(b)
    Lr, frac = 256, 0.2
    nseg = int(frac * N / Lr)
    starts = np.sort(rng.choice(np.arange(N // 10 // Lr, (N - Lr) // Lr), nseg, replace=False)) * Lr   # distinct slots
    for s0 in starts:
        src = rng.integers(0, s0 - Lr)
        x[s0:s0 + Lr] = x[src:src + Lr]
    k = I.AUDIO_DEFAULT["lzp_k"]
    th = 100 * nseg * (Lr - k) * H / (16 * N)
    inc = pct(chain(x, ORDER_1D))
    tol = 0.05 * th
    return {"name": f"V4 exact repeats: i.i.d. discrete Laplace (scale 40) with {nseg} copied segments of {Lr} samples "
                    f"(20 % of 2e6 samples) from random earlier positions",
            "theory": {"R6": th}, "measured": inc, "tolerance": f"|R6 - theory| <= 5 % of theory = {tol:.3f} %",
            "pass": bool(abs(inc["R6"] - th) <= tol),
            "note": f"theory = segments x (length - k) x H / raw bits with the match context k = {k} (the first k samples "
                    f"of a copy cannot be predicted), H = {H:.4f} bit; the cost of the hit/miss flags is not in the theory"}


def v5(seed):
    rng = np.random.default_rng(seed)
    # structured signal: AR(1) with frame-wise variance and repeats, then fully shuffled
    F = 1024
    n = (N // 2 // F) * F
    sig = np.repeat(np.where(rng.integers(0, 2, n // F) == 0, 10.0, 80.0), F)
    w = rng.normal(0, 1, n) * sig
    y = np.empty(n)
    acc = 0.0
    for t in range(n):
        acc = 0.9 * acc + w[t]
        y[t] = acc
    x = np.rint(y).astype(np.int64)
    for s0 in range(n // 10, n - 512, 8192):
        x[s0:s0 + 256] = x[s0 - 4096:s0 - 3840]
    xs = rng.permutation(x)
    inc_struct = pct(chain(x, ORDER_1D))
    inc = pct(chain(xs, ORDER_1D))
    ok0, bad = zeros_ok(inc, {"R1"})
    return {"name": "V5 shuffle baseline: AR(1) with frame-wise variance and repeats (1e6 samples), all samples shuffled",
            "theory": {"all but R1": 0.0}, "measured": inc, "measured_before_shuffle": inc_struct,
            "tolerance": f"|x| <= {ZERO_TOL} % for every pattern but R1", "pass": bool(ok0), "violations": bad}


def v1_image(seed):
    rng = np.random.default_rng(seed)
    H, W = 512, 768
    img = np.clip(np.rint(128 + rng.laplace(0, 12, (H, W, 3))), 0, 255).astype(np.uint8)
    v, c = np.unique(img.reshape(-1, 3)[:, 0], return_counts=True)
    ent = []
    for ch in range(3):
        _, c = np.unique(img[..., ch], return_counts=True)
        p = c / c.sum()
        ent.append(float(-(p * np.log2(p)).sum()))
    th = 100 * (8 - np.mean(ent)) / 8
    P = dict(I.IMAGE_DEFAULT)
    r = I.run_chain(lambda f: I.image_codelen(img, f, P), ORDER_1D, img.size * 8)
    inc = pct(r)
    ok0, bad = zeros_ok({k: v for k, v in inc.items() if k != "R10"}, {"R1"})
    tol = 0.1
    return {"name": "V1 (2-D chain) i.i.d. image, 512 x 768 x 3, clipped Laplace (scale 12) around 128",
            "theory": {"R1": th, "others": 0.0}, "measured": inc,
            "tolerance": {"R1": f"|measured - theory| <= {tol} %", "others": f"|x| <= {ZERO_TOL} % (R10 excluded: the colour "
                          "transform of independent channels mixes them and is expected to cost)"},
            "pass": bool(abs(inc["R1"] - th) <= tol and ok0), "violations": bad,
            "note": "theory uses the empirical per-channel entropy of the generated image"}


def v5_image(seed, path):
    from PIL import Image
    rng = np.random.default_rng(seed)
    img = np.asarray(Image.open(path).convert("RGB"))
    H, W, _ = img.shape
    sh = np.stack([rng.permutation(img[..., c].ravel()).reshape(H, W) for c in range(3)], axis=-1).astype(np.uint8)
    P = dict(I.IMAGE_DEFAULT)
    inc = pct(I.run_chain(lambda f: I.image_codelen(sh, f, P), ORDER_1D, sh.size * 8))
    ok0, bad = zeros_ok({k: v for k, v in inc.items() if k != "R10"}, {"R1"})
    return {"name": f"V5 (2-D chain) shuffle baseline on {os.path.basename(path)}: each channel's pixels fully shuffled",
            "theory": {"all but R1": 0.0}, "measured": inc,
            "tolerance": f"|x| <= {ZERO_TOL} % for every pattern but R1 and R10 (R10 is reported; shuffling each channel "
                         "separately makes the channels independent, so the colour transform is expected to cost)",
            "pass": bool(ok0), "violations": bad}


def v1_fp4(seed):
    rng = np.random.default_rng(seed)
    R, W = 1024, 2048
    p = np.array([6, 7, 10, 5, 8, 6, 5, 2, 6, 7, 10, 5, 8, 6, 5, 2], dtype=np.float64)
    p /= p.sum()
    C = rng.choice(16, size=(R, W), p=p)
    E = rng.choice(np.arange(118, 124), size=(R, W // 32))
    Pm = rng.choice(16, size=(R, W), p=p)
    Hc = float(-(p * np.log2(p)).sum())
    _, c = np.unique(E, return_counts=True)
    q = c / c.sum()
    Hs = float(-(q * np.log2(q)).sum())
    order = ["R1c", "R1s", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R10"]
    P = dict(I.FP4_DEFAULT)
    cache = {}
    r = I.run_chain(lambda f: I.fp4_codelen(C, E, f, P, order, partner=Pm, cache=cache), order, R * W * 4.25)
    inc = pct(r)
    th_c = 100 * (4 - Hc) / 4.25
    th_s = 100 * (8 - Hs) / 32 / 4.25
    th_10 = 100 * (p[0] + p[8]) * 1.0 / 4.25   # sign of a zero: 1 bit when +0 and -0 are equally likely
    ok0, bad = zeros_ok(inc, {"R1c", "R1s", "R10"})
    tol = 0.05
    ok = abs(inc["R1c"] - th_c) <= tol and abs(inc["R1s"] - th_s) <= tol and abs(inc["R10"] - th_10) <= tol and ok0
    return {"name": "V1 (FP4 chain) i.i.d. codes (fixed skewed law), i.i.d. scales over 6 values, independent partner",
            "theory": {"R1c": th_c, "R1s": th_s, "R10": th_10, "others": 0.0}, "measured": inc,
            "tolerance": {"R1c, R1s, R10": f"|measured - theory| <= {tol} %", "others": f"|x| <= {ZERO_TOL} %"},
            "pass": bool(ok), "violations": bad,
            "note": "theory: code R1 = (4 - H(code)) / 4.25, scale R1 = (8 - H(scale)) / 32 / 4.25, R10 = P(zero) x 1 bit / 4.25"}


TESTS = {"V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5, "V1_image": v1_image, "V1_fp4": v1_fp4}


def run_one(name, seed, extra):
    t0 = time.time()
    try:
        r = v5_image(seed, extra) if name == "V5_image" else TESTS[name](seed)
    except Exception:
        import traceback
        r = {"name": name, "pass": False, "error": traceback.format_exc()}
    r["seconds"] = time.time() - t0
    r["if_forced_on"] = FORCED.pop("last", [])
    return name, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--image", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    names = list(TESTS) + (["V5_image"] if a.image else [])
    if a.only:
        names = [n for n in names if n in a.only.split(",")]
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, n, 8000 + i, a.image) for i, n in enumerate(names)]
        for f in futs:
            n, r = f.result()
            res[n] = r
            print(n, "PASS" if r["pass"] else "FAIL", round(r["seconds"]), "s", flush=True)
    out = {"unit": "% of raw bits", "zero_tolerance": ZERO_TOL, "checks": res,
           "all_pass_1d": all(res[k]["pass"] for k in ("V1", "V2", "V3", "V4", "V5") if k in res)}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
