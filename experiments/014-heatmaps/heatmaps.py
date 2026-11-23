"""Spec 014: heat maps of FP4 weight matrices (observation only; no compression, no new method id).

    PYTHONPATH=. python experiments/014-heatmaps/heatmaps.py run [--workers N] [--only NAME,...]
    PYTHONPATH=. python experiments/014-heatmaps/heatmaps.py stats      # merge per-matrix stats into results/stats.json

Matrices (18): DeepSeek layers 0, 20, 39 and Kimi-K3 layers 1, 46, 92; in each layer the first expert of the 008
plan (plan B) / the 009 plan (plan_552), roles w1, w3, w2. Codes C and block scales E are read with
`redundancy.adapters.fp4` exactly as in 008 / 009.

Two value sets per matrix, drawn in the stored shape (not transposed):
  code  v = E2M1 value of the code (0, +-0.5, ..., +-6), no scale;
  deq   w = v * 2^(E - 127) (exact in BF16).
Variants: signed (diverging map, symmetric, limits +-p99.5 of |x|, clipped), magnitude |x| (sequential map,
0..p99.5, clipped), and for deq also log10 |x| (sequential map over [min nonzero, max]; 0 has its own colour).

Figures per matrix (each written as PNG and PDF, only the smaller file kept, as the user asked): full resolution (one pixel per element, margins with labels and a colour bar
around it), preview (mean |x| per g x g cell), 256 x 256 zooms (top left, centre) with 32-column block lines,
block-scale map E (R x W/32), shuffle control (all elements permuted with a fixed seed, same colour scales),
row / column RMS curves (real and shuffled), autocorrelation of |x| (row and column shifts 1..64).
Full-resolution images of the matrices in PUSH_FULL go to figures/; the others to data/014/full/ (not in git).
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from redundancy.adapters.fp4 import E2M1, FP4  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "figures"
RES = HERE / "results"
LOCAL_FULL = ROOT / "data/014/full"
SEED = 20261123
CELL = 8                       # preview cell size (rows x columns)
ZOOM = 256
SHIFTS = [1, 2, 4, 8, 16, 32, 64]
Q = 0.995
PUSH_FULL = {"L20.E86.w1", "L20.E86.w2", "L46.E484.w1"}   # user: full-resolution images of 3 matrices go into git
MODELS = {
    "deepseek": dict(plan="experiments/008-scale-media/results/fp4/B/plan.json", layers=[0, 20, 39],
                     samples="data/008/fp4/matrices.json", snapshot_env="SNAPSHOT", label="DeepSeek"),
    "kimi": dict(plan="experiments/009-kimi-k3/results/plan_552.json", layers=[1, 46, 92],
                 samples="data/009/kimi/matrices.json", snapshot_env="KIMI_SNAPSHOT", label="Kimi-K3"),
}
FONT = os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans.ttf")


def selection():
    out = []
    for model, m in MODELS.items():
        plan = json.load(open(ROOT / m["plan"]))
        for L in m["layers"]:
            t = next(t for t in plan["tasks"] if t["layer"] == L)
            ents = {e.rsplit(".", 1)[1]: e for e in t["entries"]}
            for role in ("w1", "w3", "w2"):
                out.append((model, ents[role], t["id"]))
    return out


def axes_meaning(role, shape):
    R, W = shape
    if role == "w2":
        return f"rows: hidden dimension ({R})", f"columns: neurons ({W})"
    return f"rows: neurons ({R})", f"columns: hidden dimension ({W})"


# ---------------------------------------------------------------- colour scales
def variants(v, w):
    """name -> (array, kind, norm parameters)."""
    out = {}
    for vs, x, lab in (("code", v, "FP4 code value v (no scale)"), ("deq", w, "dequantized w = v*2^(E-127) (exact in BF16)")):
        a = np.abs(x)
        hi = float(np.quantile(a, Q))
        out[f"{vs}_signed"] = dict(x=x, kind="signed", lo=-hi, hi=hi, label=lab, valueset=vs,
                                   clip_note=f"clipped at +-{hi:.6g} (p99.5 of |x|); "
                                             f"{np.mean(a > hi) * 100:.3f}% of elements beyond")
        out[f"{vs}_abs"] = dict(x=a, kind="abs", lo=0.0, hi=hi, label="|" + lab + "|", valueset=vs,
                                clip_note=f"clipped at {hi:.6g} (p99.5 of |x|); {np.mean(a > hi) * 100:.3f}% of elements beyond")
    a = np.abs(w)
    nz = a[a > 0]
    out["deq_log"] = dict(x=a, kind="log", lo=float(nz.min()), hi=float(nz.max()), label="|dequantized w| (log scale)",
                          valueset="deq", clip_note=f"log10 colour scale over [{nz.min():.6g}, {nz.max():.6g}] "
                          f"(min nonzero, max), no clipping; 0 drawn grey ({np.mean(a == 0) * 100:.2f}% of elements)")
    return out


CMAPS = {"signed": "RdBu_r", "abs": "viridis", "log": "viridis"}
ZERO_RGB = (128, 128, 128)


def norm_of(var):
    if var["kind"] == "log":
        return LogNorm(var["lo"], var["hi"])
    return Normalize(var["lo"], var["hi"], clip=True)


def cmap_of(var):
    cm = matplotlib.colormaps[CMAPS[var["kind"]]].copy()
    if var["kind"] == "log":
        cm.set_bad(np.array(ZERO_RGB) / 255)
        cm.set_under(np.array(ZERO_RGB) / 255)
    return cm


def to_index(x, var, nlev=253):
    """0..nlev-1 colour levels; nlev = zero colour (log); 254 white, 255 black."""
    if var["kind"] == "log":
        a = np.where(x > 0, x, var["lo"])
        t = (np.log10(a) - np.log10(var["lo"])) / (np.log10(var["hi"]) - np.log10(var["lo"]))
        idx = np.clip(np.rint(t * (nlev - 1)), 0, nlev - 1).astype(np.uint8)
        return np.where(x > 0, idx, nlev).astype(np.uint8)
    t = (np.clip(x, var["lo"], var["hi"]) - var["lo"]) / (var["hi"] - var["lo"])
    return np.clip(np.rint(t * (nlev - 1)), 0, nlev - 1).astype(np.uint8)


def palette(var, nlev=253):
    cm = matplotlib.colormaps[CMAPS[var["kind"]]]
    lut = (cm(np.linspace(0, 1, nlev))[:, :3] * 255).round().astype(np.uint8).tolist()
    lut.append(list(ZERO_RGB))            # index 253: zero (log only)
    lut.append([255, 255, 255])           # 254: background
    lut.append([0, 0, 0])                 # 255: text
    return [c for rgb in lut for c in rgb]


def full_image(x, var, title, rowlab, collab):
    """Exact one pixel per element inside a labelled frame; palette PNG."""
    R, W = x.shape
    top, left, bottom, right = 110, 70, 70, 230
    img = Image.new("P", (left + W + right, top + R + bottom), 254)
    img.putpalette(palette(var))
    img.paste(Image.fromarray(to_index(x, var), "P"), (left, top))
    d = ImageDraw.Draw(img)
    f1, f2 = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 22)
    d.text((left, 12), title, fill=255, font=f1)
    d.text((left, 52), f"one pixel per element; {var['clip_note']}", fill=255, font=f2)
    d.text((left, top + R + 18), collab + "  (column index increases to the right)", fill=255, font=f2)
    lab = Image.new("P", (R, 40), 254)
    ImageDraw.Draw(lab).text((0, 6), rowlab + "  (row index increases downwards)", fill=255, font=f2)
    lab = lab.rotate(90, expand=True)
    img.paste(lab, (14, top))
    d.rectangle([left - 1, top - 1, left + W, top + R], outline=255)
    # colour bar
    cb_h = min(R, 1200)
    x0 = left + W + 40
    levels = np.arange(252, -1, -1)
    bar = np.repeat(levels[np.linspace(0, 252, cb_h).astype(int)][:, None], 36, axis=1).astype(np.uint8)
    img.paste(Image.fromarray(bar, "P"), (x0, top))
    d.rectangle([x0 - 1, top - 1, x0 + 36, top + cb_h], outline=255)
    if var["kind"] == "log":
        ticks = [var["hi"], var["lo"]]
        labs = [f"{var['hi']:.3g}", f"{var['lo']:.3g}"]
        d.rectangle([x0, top + cb_h + 30, x0 + 36, top + cb_h + 60], fill=253, outline=255)
        d.text((x0 + 44, top + cb_h + 32), "0", fill=255, font=f2)
    else:
        labs = [f"{var['hi']:.3g}", f"{(var['hi'] + var['lo']) / 2:.3g}", f"{var['lo']:.3g}"]
    for i, s in enumerate(labs):
        y = top + int(i * (cb_h - 1) / (len(labs) - 1))
        d.text((x0 + 44, y - 12), s, fill=255, font=f2)
    return img


def keep_smaller(base):
    """user: keep only one of PNG / PDF per figure, the smaller file."""
    png, pdf = Path(str(base) + ".png"), Path(str(base) + ".pdf")
    drop = pdf if png.stat().st_size <= pdf.stat().st_size else png
    drop.unlink()
    return (png if drop == pdf else pdf).name


def save_full(img, base):
    img.save(str(base) + ".png", optimize=True)
    a = np.asarray(img.convert("RGB"))
    H, W = a.shape[:2]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.figimage(a, 0, 0)                       # 1:1 pixels, embedded losslessly (Flate)
    fig.savefig(str(base) + ".pdf", dpi=100)
    plt.close(fig)
    return keep_smaller(base)


def savefig(fig, base):
    fig.savefig(str(base) + ".png", dpi=130, bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return keep_smaller(base)


def cell_mean_abs(x, g=CELL):
    R, W = x.shape
    a = np.abs(x[:R - R % g, :W - W % g])
    return a.reshape(R // g, g, W // g, g).mean(axis=(1, 3))


def preview_axes(ax, P, norm, cmap, title, rowlab, collab, shape):
    R, W = shape
    im = ax.imshow(P, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal",
                   extent=(0, P.shape[1] * CELL, P.shape[0] * CELL, 0))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(collab, fontsize=8)
    ax.set_ylabel(rowlab, fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def zoom_axes(ax, x, r0, c0, var, title, rowlab, collab):
    sub = x[r0:r0 + ZOOM, c0:c0 + ZOOM]
    disp = np.where(sub > 0, sub, np.nan) if var["kind"] == "log" else sub
    im = ax.imshow(disp, cmap=cmap_of(var), norm=norm_of(var), interpolation="nearest",
                   extent=(c0 - .5, c0 + ZOOM - .5, r0 + ZOOM - .5, r0 - .5))
    for c in range(c0, c0 + ZOOM + 1, 32):
        ax.axvline(c - .5, color="k", lw=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(collab, fontsize=8)
    ax.set_ylabel(rowlab, fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def rms_curves(x):
    return np.sqrt((x * x).mean(axis=1)), np.sqrt((x * x).mean(axis=0))


def cv(a):
    return float(a.std() / a.mean())


def autocorr(a):
    out = {"row_shift": {}, "col_shift": {}}
    for d in SHIFTS:
        out["row_shift"][d] = float(np.corrcoef(a[:-d, :].ravel(), a[d:, :].ravel())[0, 1])
        out["col_shift"][d] = float(np.corrcoef(a[:, :-d].ravel(), a[:, d:].ravel())[0, 1])
    return out


def r7(a):
    return [float(f"{x:.7g}") for x in np.asarray(a, dtype=np.float64)]


def do_matrix(model, name, task):
    t0 = time.time()
    m = MODELS[model]
    ad = FP4({"samples": str(ROOT / m["samples"]), "snapshot_env": m["snapshot_env"]})
    C, E = ad._read(ad.samples[name])
    C = C.astype(np.int64)
    R, W = C.shape
    role = name.rsplit(".", 1)[1]
    v = E2M1[C]
    v = np.where(v == 0, 0.0, v)                                  # -0 -> 0 for display
    w = v * np.exp2(np.repeat(E.astype(np.float64), 32, axis=1) - 127)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(v.size)
    vs, ws = v.ravel()[perm].reshape(R, W), w.ravel()[perm].reshape(R, W)
    rowlab, collab = axes_meaning(role, (R, W))
    out = FIG / model / name
    out.mkdir(parents=True, exist_ok=True)
    fulldir = out if name in PUSH_FULL else LOCAL_FULL / model / name
    fulldir.mkdir(parents=True, exist_ok=True)
    head = f"{m['label']} {name} ({R} x {W})"
    V = variants(v, w)
    Vs = {k: dict(var, x=(vs if var["valueset"] == "code" else ws) if var["kind"] == "signed"
                  else np.abs(vs if var["valueset"] == "code" else ws)) for k, var in V.items()}
    r0c, c0c = R // 2 - ZOOM // 2, W // 2 - ZOOM // 2
    rec = {"model": model, "matrix": name, "expert_task": task, "role": role, "shape": [R, W], "rows": rowlab,
           "columns": collab, "full_resolution_dir": str(fulldir.relative_to(ROOT)), "variants": {},
           "centre_zoom_origin": [r0c, c0c], "preview_cell": CELL, "shuffle_seed": SEED}
    # 1. full resolution + zooms
    for k, var in V.items():
        save_full(full_image(var["x"], var, f"{head}  {var['label']}", rowlab, collab), fulldir / f"full_{k}")
        for zname, (r0, c0) in (("topleft", (0, 0)), ("centre", (r0c, c0c))):
            fig, ax = plt.subplots(figsize=(7.2, 6.4))
            im = zoom_axes(ax, var["x"], r0, c0, var, f"{head}\n{var['label']}, rows {r0}-{r0 + ZOOM - 1}, "
                           f"columns {c0}-{c0 + ZOOM - 1}; lines: 32-column scale blocks", rowlab, collab)
            fig.colorbar(im, ax=ax, shrink=0.8)
            savefig(fig, out / f"zoom_{k}_{zname}")
        rec["variants"][k] = {"kind": var["kind"], "lo": var["lo"], "hi": var["hi"], "note": var["clip_note"]}
    # 2. previews (mean |x| per cell), with the shuffled matrix next to it, same scales; shuffle zoom
    for vsname, x, xs in (("code", v, vs), ("deq", w, ws)):
        P, Ps = cell_mean_abs(x), cell_mean_abs(xs)
        scales = [("linear", Normalize(0, float(np.quantile(P, Q)), clip=True))]
        if vsname == "deq":
            scales.append(("log", LogNorm(float(P[P > 0].min()), float(P.max()))))
        for sname, nm in scales:
            cmap = matplotlib.colormaps["viridis"]
            fig, ax = plt.subplots(figsize=(8, 8 * R / W + 1.2) if W >= R else (8 * W / R + 1.6, 8))
            im = preview_axes(ax, P, nm, cmap, f"{head}  preview: mean |{vsname}| per {CELL}x{CELL} cell ({sname} colour)",
                              rowlab, collab, (R, W))
            fig.colorbar(im, ax=ax, shrink=0.8)
            savefig(fig, out / f"preview_{vsname}_{sname}")
            rec.setdefault("preview_scales", {})[f"{vsname}_{sname}"] = [float(nm.vmin), float(nm.vmax)]
        for k in [k for k in V if V[k]["valueset"] == vsname]:
            var, svar = V[k], Vs[k]
            nm = scales[1][1] if var["kind"] == "log" else scales[0][1]
            fig, axs = plt.subplots(2, 2, figsize=(13, 12))
            for j, (PP, lab) in enumerate(((P, "real"), (Ps, f"shuffled (seed {SEED})"))):
                im = preview_axes(axs[0, j], PP, nm, matplotlib.colormaps["viridis"],
                                  f"{lab}: mean |{vsname}| per {CELL}x{CELL} cell", rowlab, collab, (R, W))
                fig.colorbar(im, ax=axs[0, j], shrink=0.7)
            for j, (vv, lab) in enumerate(((var, "real"), (svar, f"shuffled (seed {SEED})"))):
                im = zoom_axes(axs[1, j], vv["x"], 0, 0, var, f"{lab}: {var['label']}, top-left {ZOOM}x{ZOOM}",
                               rowlab, collab)
                fig.colorbar(im, ax=axs[1, j], shrink=0.7)
            fig.suptitle(f"{head}  shuffle control ({k})", fontsize=11)
            savefig(fig, out / f"shuffle_{k}")
    # 3. block scale map
    Ee = E.astype(np.int64) - 127
    fig, ax = plt.subplots(figsize=(8, 8) if R >= W else (8, 6))
    im = ax.imshow(Ee, cmap="viridis", interpolation="nearest", aspect="auto")
    ax.set_title(f"{head}  block scale exponent E-127, {R} x {W // 32} (row, 32-column block)", fontsize=9)
    ax.set_xlabel(f"block index along columns (each = 32 columns; {collab})", fontsize=8)
    ax.set_ylabel(rowlab, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="E - 127")
    savefig(fig, out / "scale_map")
    rec["scale_exponent_range"] = [int(Ee.min()), int(Ee.max())]
    rec["scale_exponent_counts"] = {str(k): int(c) for k, c in zip(*np.unique(Ee, return_counts=True))}
    # 4. statistics and their figures
    rec["stats"] = {}
    for vsname, x, xs in (("code", v, vs), ("deq", w, ws)):
        rr, cr = rms_curves(x)
        rrs, crs = rms_curves(xs)
        ac, acs = autocorr(np.abs(x)), autocorr(np.abs(xs))
        rec["stats"][vsname] = {"row_rms_cv": cv(rr), "col_rms_cv": cv(cr), "row_rms_cv_shuffled": cv(rrs),
                                "col_rms_cv_shuffled": cv(crs), "autocorr_abs": ac, "autocorr_abs_shuffled": acs,
                                "row_rms": r7(rr), "col_rms": r7(cr), "row_rms_shuffled": r7(rrs),
                                "col_rms_shuffled": r7(crs)}
        fig, axs = plt.subplots(2, 1, figsize=(10, 7))
        axs[0].plot(rr, lw=0.6, label=f"real (CV {cv(rr):.4f})")
        axs[0].plot(rrs, lw=0.6, alpha=0.7, label=f"shuffled (CV {cv(rrs):.4f})")
        axs[0].set_xlabel(rowlab)
        axs[0].set_ylabel(f"RMS of {vsname} over the row")
        axs[1].plot(cr, lw=0.6, label=f"real (CV {cv(cr):.4f})")
        axs[1].plot(crs, lw=0.6, alpha=0.7, label=f"shuffled (CV {cv(crs):.4f})")
        axs[1].set_xlabel(collab)
        axs[1].set_ylabel(f"RMS of {vsname} over the column")
        for ax in axs:
            ax.legend(fontsize=8)
        fig.suptitle(f"{head}  row / column RMS ({vsname})", fontsize=10)
        fig.tight_layout()
        savefig(fig, out / f"rms_{vsname}")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for key, lab, mk in (("row_shift", "shift d rows (same column)", "o"), ("col_shift", "shift d columns (same row)", "s")):
            ax.plot(SHIFTS, [ac[key][d] for d in SHIFTS], marker=mk, label=f"real, {lab}")
            ax.plot(SHIFTS, [acs[key][d] for d in SHIFTS], marker=mk, ls="--", alpha=0.7, label=f"shuffled, {lab}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(SHIFTS, [str(d) for d in SHIFTS])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("shift d")
        ax.set_ylabel(f"correlation of |{vsname}|")
        ax.legend(fontsize=7)
        ax.set_title(f"{head}  autocorrelation of |{vsname}|", fontsize=9)
        savefig(fig, out / f"autocorr_{vsname}")
    rec["seconds"] = time.time() - t0
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "per_matrix").mkdir(exist_ok=True)
    json.dump(rec, open(RES / "per_matrix" / f"{model}_{name}.json", "w"))
    return name, rec["seconds"]


def cmd_run(a):
    sel = selection()
    if a.only:
        keep = set(a.only.split(","))
        sel = [s for s in sel if s[1] in keep]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for name, sec in ex.map(do_matrix, *zip(*sel)):
            print(f"{name} {sec:.0f} s", flush=True)
    print(f"RUN_DONE {len(sel)} matrices, wall {time.time() - t0:.0f} s", flush=True)


def cmd_stats(a):
    recs = []
    for model, name, task in selection():
        recs.append(json.load(open(RES / "per_matrix" / f"{model}_{name}.json")))
    json.dump({"seed": SEED, "preview_cell": CELL, "shifts": SHIFTS, "quantile": Q, "matrices": recs},
              open(RES / "stats.json", "w"))
    for r in recs:
        s = r["stats"]
        print(r["model"], r["matrix"], r["shape"], " ".join(
            f"{vs}: row {s[vs]['row_rms_cv']:.4f}/{s[vs]['row_rms_cv_shuffled']:.4f} col {s[vs]['col_rms_cv']:.4f}/"
            f"{s[vs]['col_rms_cv_shuffled']:.4f}" for vs in ("code", "deq")), f"{r['seconds']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--only", default=None)
    sub.add_parser("stats")
    a = ap.parse_args()
    {"run": cmd_run, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
