"""Draw the figures from result files only (spec 007 sections 4 and 8; second version of the figures).

    python -m redundancy.plot --results <results dir> --out <figure dir> [--overview <dir>] [--synthetic <dir>]

Which panels a figure shows ("synthetic_none", "synthetic_has", "data"), display names and wording
that belongs to one kind of data come from the `figures` section of the run's config (kept in
manifest.json). Without that section every figure shows all its panels with generic wording.
Writes <figure>.png/.pdf per figure, overview.png/.pdf, overview.md, figure_notes.md, r5_gain.json
(when the R5 results carry the strict gain) and figure_check.json (automatic layout checks).
This module reads JSON only; it imports no adapter and no indicator.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.transforms import blended_transform_factory, offset_copy  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from . import notes, style  # noqa: E402

LOCS = ("upper right", "upper left", "upper center", "lower right", "lower left", "center right",
        "center left", "lower center", "center")
HEADROOM = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8)
PANEL_KEYS = ("synthetic_none", "synthetic_has", "data")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def arr(x):
    return np.array([np.nan if v is None else v for v in x], dtype=np.float64)


# ----------------------------------------------------------------------------- wording

class Ctx:
    """Labels plus the `figures` section of the config."""

    def __init__(self, labels, man):
        self.L = labels
        self.man = man
        self.fc = (man.get("config") or {}).get("figures") or {}
        self.vocab = {**labels["vocab"], **self.fc.get("vocab", {})}

    def fig(self, fid):
        return self.fc.get("R9" if fid.startswith("R9") else fid) or {}

    def t(self, text, **kw):
        for k, v in {**self.vocab, **kw}.items():
            text = text.replace("{" + k + "}", str(v))
        return text

    def title(self, fid, rec=None):
        fc = self.fig(fid)
        if "title" in fc:
            return self.t(fc["title"])
        base = "R9" if fid.startswith("R9") else fid
        kw = {}
        if base == "R8":
            kw["s"] = self.r8_s(rec)
        if base == "R9":
            kw["name"] = fid[3:].replace("_", " ")
        return self.t(self.L["titles"][base], **kw)

    def axes(self, fid, rec):
        base = "R9" if fid.startswith("R9") else fid
        x, y = self.L["axes"][base]
        fc = self.fig(fid)
        kw = {}
        if base == "R2":
            kw["raw"] = f"{rec['summary'].get('raw_bits', 0):g}" if rec else ""
        x, y = fc.get("xlabel", x), fc.get("ylabel", y)
        return self.t(x, **kw), self.t(y, **kw)

    def r8_s(self, rec):
        if self.fig("R8").get("s"):
            return int(self.fig("R8")["s"])
        ss = (rec or {}).get("summary", {}).get("ss") or [None]
        return ss[-1]

    def dataset(self, rec):
        n = rec.get("summary", {}).get("n_entries") or len(rec.get("entries", {}))
        lab = self.fig(rec.get("figure", "")).get("dataset_label") or self.fc.get("dataset_label") \
            or self.L["dataset_label"]
        return self.t(lab, n=n)

    def baseline(self, name, fid=None):
        name = self.fig(fid).get("baseline_legend", name) if fid else name
        return self.t(self.L["legend"]["baseline"].get(name, name))

    def leg(self, key, **kw):
        return self.t(self.L["legend"][key], **kw)


# ----------------------------------------------------------------------------- axes helpers

def pct_text(v):
    if not np.isfinite(v) or v <= 0:
        return "0%"
    p = 100.0 * v
    s = f"{p:.6f}".rstrip("0").rstrip(".")
    return s + "%"


def percent_axis(ax, log=False):
    if log:
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pct_text(v)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pct_text(v) if v > 1e-12 else "0%"))


def band_line(ax, x, b, color, label, marker=None):
    med, lo, hi = arr(b["median"]), arr(b["min"]), arr(b["max"])
    if np.nanmax(np.abs(hi - lo)) > 0:
        ax.fill_between(x, lo, hi, color=color, alpha=style.BAND_ALPHA, lw=0)
    return ax.plot(x, med, color=color, lw=style.LW_REAL, marker=marker, ms=style.MS, label=label)[0]


def base_line(ax, x, y, label, marker=None):
    return ax.plot(x, arr(y), color=style.BASE, lw=style.LW_BASE, marker=marker, ms=style.MS - 1,
                   label=label)[0]


def hline(ax, y, label):
    return ax.axhline(y, color=style.LINE, ls="--", lw=style.LW_LINE, label=label)


def vline(ax, x, label, guide=False):
    if guide:
        return ax.axvline(x, color=style.GUIDE, ls=":", lw=2.2, label=label)
    return ax.axvline(x, color=style.LINE, ls="--", lw=style.LW_LINE, label=label)


def lag_ticks(ax, n):
    step = 16 if n > 48 else 8
    ax.set_xticks(np.arange(0, n + 1, step))
    ax.set_xlim(0, n + 1)


def log2_ticks(ax, ks):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(ks[0] / 1.25, ks[-1] * 1.25)


# ----------------------------------------------------------------------------- panels
# Each returns (legend handles, options); options: {"fixed_y": bool, "log_y": bool}.

def d_r1(ax, rec, ctx, fid, extra):
    s = rec["summary"]
    A = len(s["alphabet"])
    x = np.arange(A)
    med, lo, hi = arr(s["prob"]["median"]), arr(s["prob"]["min"]), arr(s["prob"]["max"])
    ax.bar(x, med, color=style.REAL, width=0.7, zorder=2)
    if np.nanmax(hi - lo) > 0:
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none", ecolor="black", elinewidth=1.8, capsize=4,
                    capthick=1.8, zorder=3)
    h = ax.axhline(s["uniform"], color=style.BASE, lw=style.LW_BASE, label=ctx.leg("uniform", A=A), zorder=1)
    step = 1 if A <= 32 else int(np.ceil(A / 32))
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(i) for i in x[::step]], rotation=90 if A > 16 else 0)
    ax.set_xlim(-0.6, A - 0.4)
    ax.set_ylim(0, np.nanmax(hi) * 1.05)
    percent_axis(ax)
    return [h], {}


def r2_rows(rec, ctx, r1):
    """Bars of R2: [(display name, gain clipped at 0, raw net, (min, max) or None)], skew, raw bits."""
    s = rec["summary"]
    groups = s["groups"]
    names = ctx.fig("R2").get("groups")
    sel = [g for g in groups if names is None or g in names]
    show = (names or {})
    rows = []
    for g in sel:
        r = groups[g]
        rng = (r["net_min"], r["net_max"]) if (r.get("scope") == "symbol" and "net_min" in r) else None
        rows.append((show.get(g, g.replace("_", " ").capitalize()), max(r["net"], 0.0), r["net"], rng, g))
    rows.sort(key=lambda t: -t[1])
    if r1 and r1.get("summary"):
        skew, raw = r1["summary"]["saving"], r1["summary"]["raw_bits"]
    else:
        p = np.asarray(s["overall"], dtype=np.float64)
        p = p[p > 0]
        raw = float(np.ceil(np.log2(len(s["overall"]))))
        skew = raw + float((p * np.log2(p)).sum())
    return rows, skew, raw


def d_r2(ax, rec, ctx, fid, extra):
    rows, skew, raw = r2_rows(rec, ctx, extra.get("R1"))
    rec["summary"]["raw_bits"] = raw
    cats = [ctx.leg("none_group")] + [r[0] for r in rows]
    x = np.arange(len(cats))
    up = np.array([0.0] + [r[1] for r in rows])
    rawlab = f"{raw:g}"
    h1 = ax.bar(x, np.full(len(x), skew), color=style.LIGHT, width=0.65, label=ctx.leg("skew", raw=rawlab), zorder=2)
    h2 = ax.bar(x, up, bottom=skew, color=style.REAL, width=0.65, label=ctx.leg("gain"), zorder=2)
    top = skew + up
    for i, r in enumerate(rows, start=1):
        if r[3] is not None:
            lo, hi = skew + max(r[3][0], 0.0), skew + max(r[3][1], 0.0)
            ax.errorbar([i], [top[i]], yerr=[[top[i] - lo], [hi - top[i]]], fmt="none", ecolor="black",
                        elinewidth=1.8, capsize=5, capthick=1.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([split2(c) if len(c) > 9 else c for c in cats])
    ax.set_xlim(-0.6, len(x) - 0.4)
    hi = max([skew + max(r[3][1], 0) for r in rows if r[3] is not None] + list(top))
    ax.set_ylim(0, hi * 1.05)
    return [h1, h2], {}


def d_r3(ax, rec, ctx, fid, extra):
    s = rec["summary"]
    x = np.asarray(s["lags"], dtype=np.float64)
    hs = [band_line(ax, x, s["real"], style.REAL, ctx.leg("measured_R3")),
          base_line(ax, x, s["base"]["median"], ctx.baseline(s["baseline"])),
          hline(ax, s["breakeven"], ctx.leg("breakeven"))]
    L = s.get("unit_len")
    if ctx.fig(fid).get("unit_line") and L and L <= x[-1]:
        hs.append(vline(ax, L, ctx.leg("unit_line", L=L), guide=True))
    lag_ticks(ax, int(x[-1]))
    ax.set_ylim(0, max(np.nanmax(arr(s["real"]["max"])), s["breakeven"]) * 1.05)
    return hs, {}


def d_r4(ax, rec, ctx, fid, extra):
    s = rec["summary"]
    x = np.asarray(s["lags"], dtype=np.float64)
    c = s["curves"]
    hs = [band_line(ax, x, c["v"]["real"], style.REAL, ctx.leg("value")),
          band_line(ax, x, c["abs"]["real"], style.REAL2, ctx.leg("abs")),
          base_line(ax, x, c["v"]["base"]["median"], ctx.baseline(s["baseline"]))]
    base_line(ax, x, c["abs"]["base"]["median"], None)
    hs.append(hline(ax, s["breakeven"], ctx.leg("breakeven")))
    L = s.get("unit_len")
    if ctx.fig(fid).get("unit_line") and L and L <= x[-1]:
        hs.append(vline(ax, L, ctx.leg("unit_line", L=L), guide=True))
    lag_ticks(ax, int(x[-1]))
    lo = min(np.nanmin(arr(c["v"]["real"]["min"])), np.nanmin(arr(c["abs"]["real"]["min"])), 0.0)
    hi = max(np.nanmax(arr(c["v"]["real"]["max"])), np.nanmax(arr(c["abs"]["real"]["max"])), s["breakeven"])
    ax.set_ylim(lo - 0.05, hi + 0.05)
    return hs, {}


def d_r5(ax, rec, ctx, fid, extra):
    s = rec["summary"]
    x = np.asarray(s["layers"], dtype=np.float64)
    hs = [band_line(ax, x, s["real"], style.REAL, ctx.leg("measured_R5")),
          base_line(ax, x, s["base"]["median"], ctx.baseline(s["baseline"])),
          hline(ax, s["breakeven"], ctx.leg("breakeven"))]
    ax.set_yscale("log")
    vals = np.concatenate([arr(s["real"]["min"]), arr(s["real"]["max"]), arr(s["base"]["median"]), [s["breakeven"]]])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    ax.set_ylim(vals.min() / 1.5, vals.max() * 1.5)
    percent_axis(ax, log=True)
    n = int(x[-1])
    ax.set_xticks([1] + list(range(8, n + 1, 8)))
    ax.set_xlim(0, n + 1)
    return hs, {"log_y": True}


def d_r6(ax, rec, ctx, fid, extra):
    s = rec["summary"]
    ks = [int(k) for k in s["ks"]]
    x = np.asarray(ks, dtype=np.float64)
    pos = np.concatenate([arr(s["real"][q]) for q in ("median", "min", "max")] + [arr(s["base"]["median"])])
    pos = pos[np.isfinite(pos) & (pos > 0)]
    d0 = 10.0 ** np.floor(np.log10(pos.min())) if len(pos) else 1e-6
    floor = d0 / 10.0
    fix = lambda v: np.where(~np.isfinite(arr(v)) | (arr(v) <= 0), floor, arr(v))
    b = {q: fix(s["real"][q]) for q in ("median", "min", "max")}
    hs = [band_line(ax, x, b, style.REAL, ctx.leg("measured"), marker="o"),
          base_line(ax, x, fix(s["base"]["median"]), ctx.baseline(s["baseline"]), marker="s")]
    hs.append(vline(ax, s["breakeven"], ctx.leg("breakeven")))
    ax.set_yscale("log")
    top = max(pos.max() if len(pos) else d0, d0) * 3
    ax.set_ylim(floor / 2, top)
    dec = 10.0 ** np.arange(np.log10(d0), np.log10(top) + 1e-9)
    ax.yaxis.set_major_locator(FixedLocator([floor] + list(dec)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v <= floor * 1.001 else pct_text(v)))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    # axis break between the "0" tick and the first decade: two short slashes on both spines
    yb = np.sqrt(floor * d0)
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for xs in (0.0, 1.0):
        for dy in (-3.0, 3.0):
            ax.plot([xs], [yb], transform=offset_copy(tr, fig=ax.figure, x=0, y=dy, units="points"), marker=[(-1, -0.5), (1, 0.5)], ms=14, mew=1.6,
                    color="black", ls="none", clip_on=False, zorder=5)
    ax.set_xscale("log", base=2)
    xt = sorted(set(ks))
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    lo_x = min(xt[0], s["breakeven"]) / 1.15
    ax.set_xlim(lo_x, xt[-1] * 1.15)
    return hs, {"log_y": True, "fixed_y": True}


def d_r7a(ax, rec, ctx, fid, extra):
    s = rec["summary"]
    x = np.asarray(s["x"], dtype=np.float64)
    real, base = arr(s["real"]), arr(s["base"])
    w = 1.0 if s["kind"] == "unit" else float(x[1] - x[0])
    ax.bar(x, np.where(real > 0, real, np.nan), color=style.REAL, width=w * 0.92 if s["kind"] == "unit" else w,
           label=ctx.leg("measured"), zorder=2)
    hb = ax.stairs(base, np.append(x - w / 2, x[-1] + w / 2), color=style.BASE, lw=style.LW_BASE,
                   label=ctx.baseline(s["baseline"]), zorder=3)
    hl = vline(ax, s["line"], ctx.leg("breakeven"))
    hr = Rectangle((0, 0), 1, 1, color=style.REAL, label=ctx.leg("measured"))
    ax.set_yscale("log")
    vals = np.concatenate([real[real > 0], base[base > 0]])
    ax.set_ylim(vals.min() / 2, vals.max() * 2)
    percent_axis(ax, log=True)
    nz = np.flatnonzero((real > 0) | (base > 0))
    lo = 0.0 if s["kind"] != "unit" else -0.5
    hi = max(x[nz[-1]] + 2 * w if len(nz) else x[-1], s["line"] * 1.15)
    ax.set_xlim(lo, min(hi, x[-1] + w / 2))
    return [hr, hb, hl], {"log_y": True, "x_room": True}


def d_r7b(ax, rec, ctx, fid, extra):
    s = rec["summary"]
    ks = [int(k) for k in s["ks"]]
    x = np.asarray(ks, dtype=np.float64)
    hs = [band_line(ax, x, s["real"], style.REAL, ctx.leg("measured"), marker="o"),
          base_line(ax, x, s["base"]["median"], ctx.baseline(s["baseline"], fid), marker="s"),
          ax.plot(x, s["breakeven"], color=style.LINE, ls="--", lw=style.LW_LINE, label=ctx.leg("breakeven"))[0]]
    log2_ticks(ax, ks)
    vals = np.concatenate([arr(s["real"]["min"]), arr(s["real"]["max"]), arr(s["base"]["median"]), arr(s["breakeven"])])
    vals = vals[np.isfinite(vals)]
    pad = 0.05 * (vals.max() - vals.min())
    ax.set_ylim(vals.min() - pad, vals.max() + pad)
    percent_axis(ax)
    return hs, {}


def d_r8(ax, rec, ctx, fid, extra):
    s = rec["summary"]
    sv = ctx.r8_s(rec)
    ks = [int(k) for k in s["ks"]]
    keys = s["keys"]
    idx = [keys.index(f"{k},{sv}") for k in ks]
    want = ctx.fig("R8").get("breakeven_bits", s.get("b"))
    line = s["breakeven"] if s.get("b") == want else s["breakeven_alt"]
    b = {q: [s["real"][q][j] for j in idx] for q in ("median", "min", "max")}
    hs = [band_line(ax, ks, b, style.REAL, ctx.leg("measured"), marker="o"),
          base_line(ax, ks, [s["base"]["median"][j] for j in idx], ctx.baseline(s["baseline"], fid), marker="s"),
          ax.plot(ks, [line[j] for j in idx], color=style.LINE, ls="--", lw=style.LW_LINE,
                  label=ctx.leg("breakeven"))[0]]
    log2_ticks(ax, ks)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    percent_axis(ax)
    return hs, {"fixed_y": True}


def d_r9(ax, rec, ctx, fid, extra):
    s = rec["summary"]
    x = np.arange(len(s["x"]))
    med, lo, hi = arr(s["observed"]["median"]), arr(s["observed"]["min"]), arr(s["observed"]["max"])
    ax.bar(x, med, color=style.REAL, width=0.8, zorder=2)
    if np.nanmax(hi - lo) > 0:
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none", ecolor="black", elinewidth=1.8, capsize=4)
    hr = Rectangle((0, 0), 1, 1, color=style.REAL, label=ctx.leg("observed"))
    he = ax.plot(x, arr(s["expected"]["median"]), color=style.BASE, lw=style.LW_BASE, marker="o", ms=style.MS,
                 label=ctx.leg("expected"))[0]
    labels = s.get("x_labels") or [str(v) for v in s["x"]]
    step = max(1, int(np.ceil(len(x) / 17)))
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step])
    ax.set_ylim(0, max(np.nanmax(hi), np.nanmax(arr(s["expected"]["median"]))) * 1.05)
    percent_axis(ax)
    return [hr, he], {}


DRAW = {"R1": d_r1, "R2": d_r2, "R3": d_r3, "R4": d_r4, "R5": d_r5, "R6": d_r6, "R7a": d_r7a,
        "R7b": d_r7b, "R8": d_r8, "R9": d_r9}


# ----------------------------------------------------------------------------- layout checks

def _runs(v):
    ok = np.isfinite(v).all(axis=1)
    out, cur = [], []
    for p, k in zip(v, ok):
        if k:
            cur.append(p)
        elif cur:
            out.append(np.asarray(cur))
            cur = []
    if cur:
        out.append(np.asarray(cur))
    return out


def _shapes(ax, skip):
    lines, areas, boxes = [], [], []
    for ln in ax.lines:
        if ln in skip or not ln.get_visible() or ln.get_alpha() == 0:
            continue
        v = ln.get_transform().transform(np.asarray(ln.get_path().vertices, dtype=np.float64))
        lines += _runs(v)
    for p in ax.patches:
        if not p.get_visible():
            continue
        if isinstance(p, Rectangle):
            bb = p.get_window_extent()
            if np.isfinite(bb.get_points()).all() and bb.width > 0 and bb.height > 0:
                boxes.append(bb)
        else:
            v = p.get_transform().transform(np.asarray(p.get_path().vertices, dtype=np.float64))
            lines += _runs(v)
    for c in ax.collections:
        tr = c.get_transform()
        for pth in c.get_paths():
            v = tr.transform(np.asarray(pth.vertices, dtype=np.float64))
            if isinstance(c, LineCollection):
                lines += _runs(v)
            else:
                v = v[np.isfinite(v).all(axis=1)]
                if len(v) >= 3:
                    areas.append(MPath(v, closed=True))
    return lines, areas, boxes


def legend_clear(ax, leg, pad=4.0, shapes=None):
    """True when the legend box touches no drawn line, marker, band or bar of the axes."""
    bb = leg.get_window_extent().padded(pad)
    lines, areas, boxes = shapes if shapes is not None else _shapes(ax, set())
    for v in lines:
        if len(v) == 1:
            if bb.contains(*v[0]):
                return False
            continue
        if MPath(v).intersects_bbox(bb, filled=False):
            return False
    for a in areas:
        if a.intersects_bbox(bb, filled=True):
            return False
    return not any(b.overlaps(bb) for b in boxes)


def legend_inside_figure(fig, leg):
    fb = fig.bbox
    lb = leg.get_window_extent()
    return lb.x0 >= fb.x0 - 1 and lb.x1 <= fb.x1 + 1 and lb.y0 >= fb.y0 - 1 and lb.y1 <= fb.y1 + 1


def _expand(lo, hi, h, log):
    if log:
        return lo, hi * (hi / lo) ** h
    return lo, hi + h * (hi - lo)


def _try_locs(fig, ax, hs, wrapped):
    shapes = _shapes(ax, set())
    renderer = fig.canvas.get_renderer()
    labels = [split2(h.get_label()) if wrapped and len(_plain(h.get_label())) > 24 else h.get_label() for h in hs]
    for loc in LOCS:
        leg = ax.legend(handles=hs, labels=labels, loc=loc)
        leg.draw(renderer)
        if legend_clear(ax, leg, shapes=shapes):
            return loc
    return None


def place_legends(fig, specs):
    """specs: [(ax, handles, opts)]. Tries every corner, first with the labels as they are, then with
    labels broken in two lines, at increasing head room (the same for every panel, so shared ranges
    stay shared); panels drawn from data with a full-height reference line may also widen the x
    range. Falls back to a legend outside the axes."""
    fig.canvas.draw()
    base = [ax.get_ylim() for ax, _, _ in specs]
    basex = [ax.get_xlim() for ax, _, _ in specs]
    result = [None] * len(specs)
    for xr in (0.0, 0.25, 0.5):
        if xr and not any(o.get("x_room") for _, _, o in specs):
            break
        for h in HEADROOM:
            for i, (ax, hs, opts) in enumerate(specs):
                if not opts.get("fixed_y"):
                    ax.set_ylim(*_expand(*base[i], h, opts.get("log_y")))
                if opts.get("x_room"):
                    ax.set_xlim(*_expand(*basex[i], xr, False))
            fig.canvas.draw()
            ok = True
            for i, (ax, hs, opts) in enumerate(specs):
                if not hs:
                    result[i] = {"loc": None, "clear": True}
                    continue
                loc = _try_locs(fig, ax, hs, False)
                if loc:
                    result[i] = {"loc": loc, "clear": True, "headroom": h, "x_room": xr}
                if not loc:
                    ok = False
                    break
            if ok:
                return result
            if all(opts.get("fixed_y") for _, _, opts in specs):
                break
    for i, (ax, hs, opts) in enumerate(specs):
        ax.set_ylim(*base[i])
        ax.set_xlim(*basex[i])
        if hs:
            ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
            result[i] = {"loc": "outside right", "clear": True, "headroom": 0.0}
    return result


def split2(s):
    """Split a label in two lines at the space nearest its middle, never inside $...$."""
    spaces, inside = [], False
    for i, ch in enumerate(s):
        if ch == "$":
            inside = not inside
        elif ch == " " and not inside:
            spaces.append(i)
    if not spaces:
        return s
    mid = len(s) / 2
    i = min(spaces, key=lambda j: abs(j - mid))
    return s[:i] + "\n" + s[i + 1:]


def fit_texts(fig, axes):
    """Break titles and axis labels that are wider than their axes into two lines."""
    fig.canvas.draw()
    changed = False
    for ax in axes:
        bb = ax.get_window_extent()
        t = ax.title
        yl = ax.yaxis.label
        if yl.get_text() and "\n" not in yl.get_text() and yl.get_window_extent().height > bb.height * 0.98:
            yl.set_text(split2(yl.get_text()))
            changed = True
        xl = ax.xaxis.label
        if xl.get_text() and "\n" not in xl.get_text() and xl.get_window_extent().width > bb.width * 1.1:
            xl.set_text(split2(xl.get_text()))
            changed = True
    if changed:
        fig.canvas.draw()


CJK = re.compile(r"[　-鿿＀-￯]")
CODE_NAME = re.compile(r"[A-Za-z]+_[A-Za-z]+")


def _plain(s):
    return re.sub(r"\$[^$]*\$", "", s)


def check_figure(fig, axes, legend_info):
    """Automatic part of the self-check: texts, sizes, legends, ranges."""
    fig.canvas.draw()
    hidden = set()
    for ax in axes:
        for axis in (ax.xaxis, ax.yaxis):
            shown = set(map(id, axis._update_ticks()))
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if id(tk) not in shown:
                    hidden.update((id(tk.label1), id(tk.label2)))
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and id(t) not in hidden]
    fb = fig.bbox
    out_of_fig = [t.get_text() for t in texts
                  if not (t.get_window_extent().x0 >= fb.x0 - 1 and t.get_window_extent().x1 <= fb.x1 + 1
                          and t.get_window_extent().y0 >= fb.y0 - 1 and t.get_window_extent().y1 <= fb.y1 + 1)]
    sizes = {"tick": min([lab.get_fontsize() for ax in axes for lab in ax.get_xticklabels() + ax.get_yticklabels()
                          if lab.get_text() and id(lab) not in hidden] or [0]),
             "label": min([ax.xaxis.label.get_fontsize() for ax in axes] + [ax.yaxis.label.get_fontsize() for ax in axes]),
             "title": min([ax.title.get_fontsize() for ax in axes if ax.title.get_text()] or
                          [fig._suptitle.get_fontsize() if fig._suptitle else 0]),
             "legend": min([t.get_fontsize() for ax in axes if ax.get_legend() for t in ax.get_legend().get_texts()] or [0])}
    lines = [ln.get_linewidth() for ax in axes for ln in ax.lines if ln.get_visible() and ln.get_alpha() != 0]
    legs = []
    for ax in axes:
        leg = ax.get_legend()
        if leg:
            legs.append({"clear": legend_clear(ax, leg), "inside_figure": legend_inside_figure(fig, leg)})
    allt = [t.get_text() for t in texts]
    titles = [ax.title for ax in axes if ax.title.get_text()]
    title_rows = sorted({round(t.get_window_extent().y0, 1) for t in titles})
    legend_texts = [t.get_text() for ax in axes if ax.get_legend() for t in ax.get_legend().get_texts()]
    too_wide = [t.get_text() for t in titles
                if t.get_window_extent().width > t.axes.get_window_extent().width * 1.12]
    return {"multiline_titles": [t.get_text() for t in titles if "\n" in t.get_text()] +
                                ([fig._suptitle.get_text()] if fig._suptitle and "\n" in fig._suptitle.get_text() else []),
            "titles_wider_than_axes": too_wide,
            "multiline_legend_entries": [t for t in legend_texts if "\n" in t],
            "title_baselines_px": title_rows,
            "chinese": [s for s in allt if CJK.search(s)],
            "code_names": [s for s in allt if CODE_NAME.search(_plain(s))],
            "outside_figure": out_of_fig, "font_sizes": sizes,
            "min_line_width": min(lines) if lines else None, "legends": legs, "legend_placement": legend_info,
            "ylims": [list(map(float, ax.get_ylim())) for ax in axes],
            "xlims": [list(map(float, ax.get_xlim())) for ax in axes],
            "ylabels": [ax.yaxis.label.get_text() for ax in axes],
            "xticklabels": [[t.get_text() for t in ax.get_xticklabels() if id(t) not in hidden] for ax in axes]}


# ----------------------------------------------------------------------------- figures

def draw_panel(ax, rec, ctx, fid, extra, title):
    if rec is None or rec.get("verdict") == "not_applicable" or not rec.get("summary"):
        ax.text(0.5, 0.5, ctx.L["na_text"], transform=ax.transAxes, ha="center", va="center",
                fontsize=style.FONT["label"], color=style.BASE)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        return [], {"fixed_y": True}
    base = "R9" if fid.startswith("R9") else ("R7a" if fid.startswith("R7a") else fid)
    hs, opts = DRAW[base](ax, rec, ctx, fid, extra)
    xl, yl = ctx.axes(fid, rec)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)
    ax.tick_params(width=1.2, length=5)
    return hs, opts


def unify(axes, specs, x_too):
    """Same y range on every panel (and the same x range when asked)."""
    live = [(ax, o) for ax, (_, _, o) in zip(axes, specs) if ax.has_data()]
    if len(live) < 2:
        return
    ys = [ax.get_ylim() for ax, _ in live]
    lo, hi = min(y[0] for y in ys), max(y[1] for y in ys)
    for ax, _ in live:
        ax.set_ylim(lo, hi)
    if x_too:
        xs = [ax.get_xlim() for ax, _ in live]
        for ax, _ in live:
            ax.set_xlim(min(x[0] for x in xs), max(x[1] for x in xs))


def make_figure(fid, panels, ctx, path, extra):
    """panels: [(record, panel title)]. One panel: the figure title is the axes title."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(n * style.PANEL_W, style.PANEL_H), squeeze=False, layout="constrained")
    axes = list(axes[0])
    title = ctx.title(fid, panels[-1][0])
    specs = []
    for ax, (rec, pt) in zip(axes, panels):
        hs, opts = draw_panel(ax, rec, ctx, fid, extra, title if n == 1 else pt)
        specs.append((ax, hs, opts))
    if n > 1:
        fig.suptitle(title, fontsize=style.FONT["suptitle"])
        unify(axes, specs, x_too=True)
    fit_texts(fig, axes)
    info = place_legends(fig, specs)
    fit_texts(fig, axes)
    rep = check_figure(fig, axes, info)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=style.DPI)
    plt.close(fig)
    return rep


def panel_list(fid, rec, syn, ctx):
    """[(record, title)] for the panels the config asks for (all available ones by default)."""
    fc = ctx.fig(fid)
    want = fc.get("panels")
    if want is None:
        want = [k for k in PANEL_KEYS if k == "data" or (syn and k.split("_")[1] in syn)]
    out = []
    for k in want:
        if k == "data":
            out.append((rec, ctx.dataset(rec)))
        elif syn and k.split("_")[1] in syn:
            srec = syn[k.split("_")[1]]
            desc = fc.get("synthetic") if k == "synthetic_has" and fc.get("synthetic") else srec.get("description", "")
            out.append((srec, ctx.t(ctx.L["synthetic_panel"], desc=desc)))
    if len(out) > 1:
        out = [(r, f"({chr(97 + i)}) {t}") for i, (r, t) in enumerate(out)]
    return out


def synthetic_for(fig_id, rec, manifest):
    if fig_id in manifest["synthetic"]:
        return fig_id
    if fig_id.startswith("R9"):
        k = "R9_" + rec.get("summary", {}).get("kind", "")
        return k if k in manifest["synthetic"] else None
    return None


def overview(man, res, ctx, odir, extra):
    """2 x 5 grid of the measured panels named in the config (all drawn figures by default)."""
    figs = ctx.fc.get("overview") or [f for f in man["figures"] if f != "R10"][:10]
    figs = [f for f in figs if f in man["figures"]]
    rows = 2 if len(figs) > 5 else 1
    cols = int(np.ceil(len(figs) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * style.PANEL_W, rows * style.PANEL_H + 0.4),
                             squeeze=False, layout="constrained")
    specs, used = [], []
    for ax, fid in zip(axes.ravel(), figs):
        rec = load(res / "data" / f"{fid}.json")
        specs.append((ax,) + draw_panel(ax, rec, ctx, fid, extra, ctx.title(fid, rec)))
        used.append(ax)
    for ax in axes.ravel()[len(figs):]:
        ax.axis("off")
    fig.suptitle(ctx.fc.get("overview_title") or man["label"], fontsize=style.FONT["suptitle"] + 2)
    fit_texts(fig, used)
    info = place_legends(fig, specs)
    fit_texts(fig, used)
    rep = check_figure(fig, used, info)
    for ext in ("png", "pdf"):
        fig.savefig(odir / f"overview.{ext}", dpi=style.DPI)
    plt.close(fig)
    return figs, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overview", default=None)
    ap.add_argument("--synthetic", default=None,
                    help="results directory whose synthetic panels are used when --results has none")
    ap.add_argument("--config", default=None,
                    help="config whose `figures` section replaces the one stored with the results")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    odir = Path(args.overview) if args.overview else out
    out.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    labels, msg = style.setup()
    print(msg, flush=True)
    with open(out / "plot_log.txt", "w", encoding="utf-8") as f:
        f.write(msg + "\n")
    man = load(res / "manifest.json")
    sres = res
    if not man["synthetic"] and args.synthetic:
        sres = Path(args.synthetic)
        man["synthetic"] = load(sres / "manifest.json")["synthetic"]
    syn = {k: {side: load(sres / v[side]["file"]) for side in ("none", "has") if side in v}
           for k, v in man["synthetic"].items()}
    if args.config:
        man.setdefault("config", {})["figures"] = load(args.config).get("figures", {})
    ctx = Ctx(labels, man)
    written, checks, skipped = [], {}, []
    if man["figures"]:
        extra = {"R1": load(res / "data" / "R1.json")} if "R1" in man["figures"] else {}
        for fig_id in man["figures"]:
            if fig_id == "R10":
                continue
            rec = load(res / "data" / f"{fig_id}.json")
            sk = synthetic_for(fig_id, rec, man)
            panels = panel_list(fig_id, rec, syn.get(sk) if sk else None, ctx)
            if not panels:
                skipped.append(fig_id)
                continue
            path = out / style.file_name(fig_id)
            checks[path.name] = make_figure(fig_id, panels, ctx, path, extra)
            written.append(path.name)
        ov, rep = overview(man, res, ctx, odir, extra)
        checks["overview"] = rep
        notes.write_all(man, res, syn, ctx, odir, written, skipped, ov)
    else:
        for k, v in syn.items():
            fid = k
            panels = [(v[s], ctx.t(ctx.L["synthetic_panel"], desc=v[s].get("description", "")))
                      for s in ("none", "has") if s in v]
            panels = [(r, f"({chr(97 + i)}) {t}") for i, (r, t) in enumerate(panels)]
            path = out / style.file_name(k)
            checks[path.name] = make_figure(fid, panels, ctx, path, {})
            written.append(path.name)
    with open(odir / "figure_check.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("figures:", ", ".join(written), flush=True)
    if skipped:
        print("not drawn (config):", ", ".join(skipped), flush=True)



if __name__ == "__main__":
    main()
