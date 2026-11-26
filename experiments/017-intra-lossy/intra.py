"""Spec 017: does an FP4 expert matrix carry intra-frame regularities that H.265 intra coding (LLM.265 style) or a
block DCT can exploit? Lossy rate-distortion, real vs shuffled vs a plain scalar quantizer.

    PYTHONPATH=.:experiments/016-cross-matrix python experiments/017-intra-lossy/intra.py synth
    PYTHONPATH=.:experiments/016-cross-matrix python experiments/017-intra-lossy/intra.py run {deepseek|kimi} [--workers N]
    PYTHONPATH=.:experiments/016-cross-matrix python experiments/017-intra-lossy/intra.py summarize
"""
import argparse
import json
import math
import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.fft import dctn  # noqa: E402

import common as K  # noqa: E402  (experiments/016-cross-matrix/common.py: matrix lists and reading)

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FF = "/home/zhouziang/.local/share/miniforge3/envs/codec/bin/ffmpeg"
SEED = 20261126
QPS = [1, 6, 12, 18, 24, 30, 36, 42, 48]
STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
LAYERS = {"deepseek": [0, 13, 26, 39], "kimi": [1, 31, 61, 92]}
BLOCKS = [(1, 32), (8, 8), (16, 16), (32, 32)]
TMP = Path("/tmp/claude-10029/intra017")


def kt_bits(counts):
    from scipy.special import gammaln
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0] if (c > 0).sum() else c
    K_ = max(len(c), 1)
    return float((gammaln(c.sum() + K_ / 2) - gammaln(K_ / 2) - (gammaln(c + 0.5) - gammaln(0.5)).sum()) / math.log(2))


def sample(model):
    rng = np.random.default_rng(SEED)
    ents = K.entries(model)
    out = []
    for L in LAYERS[model]:
        exs = sorted({e["expert"] for e in ents if e["layer"] == L})
        for ex in rng.choice(exs, 2, replace=False):
            for i, e in enumerate(ents):
                if e["layer"] == L and e["expert"] == int(ex):
                    out.append(i)
    return out


# ------------------------------------------------------------------ views
def tiles(img):
    R, W = img.shape
    h, w = R // 2, W // 2
    return [img[a * h:(a + 1) * h, b * w:(b + 1) * w] for a in (0, 1) for b in (0, 1)], (h, w)


def untile(frames, shape):
    R, W = shape
    h, w = R // 2, W // 2
    out = np.empty(shape, dtype=frames[0].dtype)
    k = 0
    for a in (0, 1):
        for b in (0, 1):
            out[a * h:(a + 1) * h, b * w:(b + 1) * w] = frames[k]; k += 1
    return out


def hevc(img, qp, tmpdir):
    """all-intra libx265 encode / decode of a 12-bit monochrome image; returns (bits, decoded image)."""
    fr, (h, w) = tiles(img)
    raw = Path(tmpdir) / "in.raw"; bs = Path(tmpdir) / "out.hevc"; dec = Path(tmpdir) / "dec.raw"
    np.concatenate([f.ravel() for f in fr]).astype("<u2").tofile(raw)
    params = "keyint=1:pools=1:frame-threads=1:log-level=error:" + ("lossless=1" if qp is None else f"qp={qp}")
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray12le", "-s", f"{w}x{h}",
                    "-r", "1", "-i", str(raw), "-c:v", "libx265", "-preset", "medium", "-x265-params", params, "-f", "hevc", str(bs)], check=True)
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-i", str(bs), "-f", "rawvideo", "-pix_fmt", "gray12le", str(dec)], check=True)
    d = np.fromfile(dec, dtype="<u2").reshape(4, h, w)
    return 8 * bs.stat().st_size, untile(list(d), img.shape)


def nmse_db(w, wh):
    return float(10 * np.log10(((w - wh) ** 2).sum() / (w ** 2).sum() + 1e-300))


def dct_stats(x):
    """coding gain and top-10% energy share of block DCT coefficients for several block shapes."""
    R, W = x.shape
    out = {}
    for bh, bw in BLOCKS:
        b = x[: R // bh * bh, : W // bw * bw].reshape(R // bh, bh, W // bw, bw).transpose(0, 2, 1, 3).reshape(-1, bh, bw)
        c = dctn(b - b.mean(), axes=(1, 2), norm="ortho")
        var = (c ** 2).mean(axis=0).ravel()
        var = np.maximum(var, 1e-300)
        top = np.sort(var)[::-1]
        out[f"{bh}x{bw}"] = {"coding_gain": float(var.mean() / np.exp(np.log(var).mean())),
                             "top10pct_energy": float(top[: max(1, len(top) // 10)].sum() / top.sum())}
    return out


def matrix_job(C, E, name, tag, view):
    """all measurements for one matrix (block scales E) and one view."""
    return matrix_job_elem(C, np.repeat(E, 32, axis=1), name, tag, view)


def real_job(model, i, view, tag):
    e = K.entries(model)[i]
    C, E = K.load(model, e)
    if tag == "shuffled":
        rng = np.random.default_rng(SEED + i)
        Ef = np.repeat(E, 32, axis=1)
        p = rng.permutation(C.size)
        C = C.ravel()[p].reshape(C.shape)
        Ef = Ef.ravel()[p].reshape(C.shape)           # every element keeps its own scale
        return matrix_job_elem(C, Ef, e["matrix"], tag, view)
    return matrix_job(C, E, e["matrix"], tag, view)


def matrix_job_elem(C, Ef, name, tag, view):
    """all measurements for one matrix with a per-element exponent Ef (block scales expanded, or shuffled) and one view."""
    t0 = time.time()
    v = K.E2M1[C].astype(np.float64)
    sc = np.exp2(Ef.astype(np.float64) - 127)
    w = v * sc
    N = C.size
    if view == "v":
        lum = (2048 + 160 * np.round(2 * v)).astype(np.uint16)
        side = kt_bits(np.bincount(Ef[:, ::32].ravel(), minlength=256))     # same scale-stream cost as the real matrix
        rec = lambda L_: (L_.astype(np.float64) - 2048) / 320 * sc
    else:
        m = np.abs(w).max()
        lum = np.round(2048 + 2047 * w / m).astype(np.uint16)
        side = 32.0
        rec = lambda L_: (L_.astype(np.float64) - 2048) * m / 2047
    res = {"matrix": name, "tag": tag, "view": view, "shape": list(C.shape), "side_bits": side, "hevc": [], "scalar": []}
    res["dct"] = dct_stats(v if view == "v" else w)
    TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TMP) as td:
        for qp in [None] + QPS:
            bits, d = hevc(lum, qp, td)
            res["hevc"].append({"qp": "lossless" if qp is None else qp, "bpw": (bits + side) / N, "nmse_db": nmse_db(w, rec(d)),
                                "exact": bool(qp is None and np.array_equal(d, lum))})
    x = lum.astype(np.int64) - 2048
    for s in STEPS:
        q = np.round(x / s).astype(np.int64)
        bits = kt_bits(np.bincount(q.ravel() - q.min()))
        res["scalar"].append({"step": s, "bpw": (bits + side) / N, "nmse_db": nmse_db(w, rec(2048 + q * s))})
    res["mapping_nmse_db"] = nmse_db(w, rec(lum))
    res["seconds"] = time.time() - t0
    return res


def cmd_run(a):
    RES.mkdir(parents=True, exist_ok=True)
    ids = sample(a.model)
    jobs = [(a.model, i, view, tag) for i in ids for view in ("v", "w") for tag in ("real", "shuffled")]
    path = RES / f"rd_{a.model}.jsonl"
    done = {(r["matrix"], r["view"], r["tag"]) for r in map(json.loads, open(path))} if path.exists() else set()
    names = [e["matrix"] for e in K.entries(a.model)]
    jobs = [j for j in jobs if (names[j[1]], j[2], j[3]) not in done]
    print(a.model, len(jobs), "jobs", flush=True)
    t0 = time.time()
    with open(path, "a") as fo, ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(real_job, *zip(*jobs)), 1):
            fo.write(json.dumps(r) + "\n"); fo.flush()
            print(f"{k}/{len(jobs)} {r['matrix']} {r['view']} {r['tag']} {r['seconds']:.0f}s total {time.time() - t0:.0f}s", flush=True)
    print("RUN_DONE", a.model, f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ synthetic checks
def mxfp4(x):
    """quantize a float matrix to MXFP4 codes and E8M0 scales (per row, 32-element blocks), nearest E2M1 level."""
    R, W = x.shape
    b = x.reshape(R, W // 32, 32)
    amax = np.abs(b).max(axis=2)
    E = np.clip(np.floor(np.log2(np.maximum(amax, 1e-30))) - 2 + 127, 1, 254).astype(np.uint8)
    s = np.exp2(E.astype(np.float64) - 127)[:, :, None]
    lv = K.E2M1[:8]
    mag = np.abs(b) / s
    idx = np.abs(mag[..., None] - lv).argmin(axis=-1)
    C = (idx + 8 * (b < 0)).astype(np.uint8).reshape(R, W)
    return C, E


def cmd_synth(a):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(1)
    R, W = 512, 1024
    smooth = gaussian_filter(rng.standard_normal((R, W)), 6) * 30 + 0.05 * rng.standard_normal((R, W))
    iid = rng.standard_normal((R, W))
    out = {}
    for nm, x in (("smooth_field", smooth), ("iid", iid)):
        C, E = mxfp4(x)
        for view in ("v", "w"):
            rr = matrix_job(C, E, nm, "real", view)
            p = rng.permutation(C.size)
            rs = matrix_job_elem(C.ravel()[p].reshape(C.shape), np.repeat(E, 32, axis=1).ravel()[p].reshape(C.shape), nm, "shuffled", view)
            out[f"{nm}_{view}"] = {"real": rr, "shuffled": rs}
            g = rr["dct"]["8x8"]["coding_gain"], rs["dct"]["8x8"]["coding_gain"]
            d = rd_gap(rr["hevc"], rs["hevc"])
            print(nm, view, "DCT 8x8 gain real %.2f shuffled %.2f" % g, "HEVC real-vs-shuffled NMSE gap at equal rate (dB):", d,
                  "lossless exact:", rr["hevc"][0]["exact"], flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES / "synth.json", "w"), indent=1)


def interp(curve, rate):
    """NMSE (dB) of a rate-distortion curve at `rate` bpw (linear interpolation in rate; None outside the range)."""
    # exact reconstructions (NMSE -> -inf, stored as about -3000 dB) are end points, not interpolable
    pts = sorted((p["bpw"], p["nmse_db"]) for p in curve if p["nmse_db"] > -150)
    r = np.array([p[0] for p in pts]); d = np.array([p[1] for p in pts])
    if rate < r.min() or rate > r.max():
        return None
    return float(np.interp(rate, r, d))


RATES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]


def rd_gap(c1, c2):
    """c1 - c2 in dB at common rates (negative: c1 has lower distortion)."""
    out = {}
    for r in RATES:
        a, b = interp(c1, r), interp(c2, r)
        if a is not None and b is not None:
            out[r] = round(a - b, 3)
    return out


def cmd_summarize(a):
    S = {}
    for model in ("deepseek", "kimi"):
        p = RES / f"rd_{model}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p)]
        by = {(r["matrix"], r["view"], r["tag"]): r for r in rows}
        m = {}
        for view in ("v", "w"):
            mats = sorted({r["matrix"] for r in rows if r["view"] == view})
            gaps = {"hevc_real_minus_shuffled": [], "hevc_real_minus_scalar_real": [], "scalar_real_minus_shuffled": []}
            dct = {f"{bh}x{bw}": {"real": [], "shuffled": []} for bh, bw in BLOCKS}
            lossless = {"real": [], "shuffled": []}
            for mt in mats:
                r, s = by.get((mt, view, "real")), by.get((mt, view, "shuffled"))
                if not (r and s):
                    continue
                gaps["hevc_real_minus_shuffled"].append(rd_gap(r["hevc"], s["hevc"]))
                gaps["hevc_real_minus_scalar_real"].append(rd_gap(r["hevc"], r["scalar"]))
                gaps["scalar_real_minus_shuffled"].append(rd_gap(r["scalar"], s["scalar"]))
                for k in dct:
                    dct[k]["real"].append(r["dct"][k]); dct[k]["shuffled"].append(s["dct"][k])
                lossless["real"].append(r["hevc"][0]["bpw"]); lossless["shuffled"].append(s["hevc"][0]["bpw"])
            def med(lst):
                out = {}
                for rate in RATES:
                    v = [g[rate] for g in lst if rate in g]
                    if v:
                        out[rate] = {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v)), "n": len(v)}
                return out
            m[view] = {"n_matrices": len(lossless["real"]),
                       "gap_db": {k: med(v) for k, v in gaps.items()},
                       "dct": {k: {t: {"coding_gain_median": float(np.median([x["coding_gain"] for x in v[t]])),
                                       "coding_gain_max": float(max(x["coding_gain"] for x in v[t])),
                                       "top10pct_energy_median": float(np.median([x["top10pct_energy"] for x in v[t]]))}
                                   for t in ("real", "shuffled")} for k, v in dct.items()},
                       "hevc_lossless_bpw_median": {t: float(np.median(v)) for t, v in lossless.items()},
                       "lossless_exact_all": all(by[(mt, view, t)]["hevc"][0]["exact"] for mt in mats for t in ("real", "shuffled") if (mt, view, t) in by)}
        S[model] = m
    json.dump(S, open(RES / "summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1)[:6000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["synth", "run", "summarize"])
    ap.add_argument("model", nargs="?", choices=["deepseek", "kimi"])
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    {"synth": cmd_synth, "run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
