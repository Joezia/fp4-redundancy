"""Draw the figures from result files only (spec 007 sections 4 and 8).

    python -m redundancy.plot --results <results dir> --out <figure dir> [--overview <dir>] [--lang zh|en]

With a measured dataset in the results: one triptych per figure (synthetic none, synthetic present,
measured), plus overview.png / overview.md. Without one: two-panel synthetic figures.
This module reads JSON only; it imports no adapter and no indicator.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import style  # noqa: E402

TWO_LEVEL = ("R1", "R9", "R10")
KEY_PT = 30      # key-number line, points below the axes
TABLE_PT = 42    # first table line, points below the axes
LINE_PT = 7.5    # table line height in points


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def base_id(fig):
    if fig.startswith("R9"):
        return "R9"
    return fig


def verdict_text(rec, L):
    v = rec["verdict"]
    if v == "present" and base_id(rec["figure"]) not in TWO_LEVEL:
        v = "present_below"
    return L["verdict"][v]


def arr(x):
    return np.array([np.nan if v is None else v for v in x], dtype=np.float64)


def line_label(ax, x, y, L):
    ax.annotate(L["breakeven"], xy=(x, y), xytext=(2, 2), textcoords="offset points", fontsize=6,
                color=style.LINE)


def hline(ax, y, L, xpos=None):
    ax.axhline(y, color=style.LINE, ls="--", lw=1.0)
    xl = ax.get_xlim()
    line_label(ax, xl[0] + 0.02 * (xl[1] - xl[0]) if xpos is None else xpos, y, L)


def vline(ax, x, L):
    ax.axvline(x, color=style.LINE, ls="--", lw=1.0)
    yl = ax.get_ylim()
    ax.annotate(L["breakeven"], xy=(x, yl[1]), xytext=(2, -10), textcoords="offset points", fontsize=6,
                color=style.LINE)


def curve(ax, x, b, color=style.REAL, label=None, marker=None):
    """Median curve with a light min–max band (band only when there are several entries)."""
    med, lo, hi = arr(b["median"]), arr(b["min"]), arr(b["max"])
    if np.nanmax(np.abs(hi - lo)) > 0:
        ax.fill_between(x, lo, hi, color=color, alpha=style.BAND_ALPHA, lw=0)
    ax.plot(x, med, color=color, lw=1.4, marker=marker, ms=3, label=label)


def basecurve(ax, x, b, label=None):
    ax.plot(x, arr(b["median"]), color=style.BASE, lw=0.8, label=label)


# ----------------------------------------------------------------------------- per-figure panels

def p_r1(ax, rec, L, extra):
    s = rec["summary"]
    A = len(s["alphabet"])
    x = np.arange(A)
    for i, cls in enumerate(extra.get("classes", [])):
        for c in cls:
            ax.axvspan(c - 0.5, c + 0.5, color=style.SHADE[i % len(style.SHADE)], lw=0, zorder=0)
    med, lo, hi = arr(s["prob"]["median"]), arr(s["prob"]["min"]), arr(s["prob"]["max"])
    ax.bar(x, med, color=style.REAL, width=0.7, zorder=2)
    if np.nanmax(hi - lo) > 0:
        ax.vlines(x, lo, hi, color="#56B4E9", lw=2.5, alpha=0.8, zorder=3)
    ax.axhline(s["uniform"], color=style.BASE, lw=0.8, label=L["legend"]["uniform"])
    ax.set_xticks(x)
    ax.set_xticklabels(s["alphabet"], rotation=90 if A > 8 else 0, fontsize=5 if A > 16 else 6)
    if A > 32:
        ax.set_xticks(x[:: max(1, A // 16)])
        ax.set_xticklabels([s["alphabet"][i] for i in x[:: max(1, A // 16)]], rotation=90, fontsize=5)
    ax.legend(fontsize=6, loc="upper left")


def p_r2(ax, rec, L, extra):
    s = rec["summary"]
    A = s["alphabet_size"]
    x = np.arange(A)
    ax.plot(x, s["overall"], color=style.BASE, lw=2.0, alpha=0.6, label=L["legend"]["overall"])
    for i, (lab, c) in enumerate(list(s["curves"].items())[:4]):
        ax.plot(x, c, color=style.SERIES[i], lw=1.0, marker="o", ms=2, label=f"{s['best']} = {lab}")
    ax.legend(fontsize=5.5, loc="upper left")


def p_lines(ax, rec, L, xkey, logy=False):
    s = rec["summary"]
    x = np.asarray(s[xkey], dtype=np.float64)
    curve(ax, x, s["real"], label=L["real"])
    basecurve(ax, x, s["base"], label=L["baseline"])
    if logy:
        ax.set_yscale("log")
    return s, x


def p_r3(ax, rec, L, extra):
    s, x = p_lines(ax, rec, L, "lags")
    hline(ax, s["breakeven"], L)
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 0.88))


def p_r4(ax, rec, L, extra):
    s = rec["summary"]
    x = np.asarray(s["lags"], dtype=np.float64)
    ax.axhspan(-s["noise"], s["noise"], color=style.BASE, alpha=0.25, lw=0, label=L["legend"]["noise"])
    curve(ax, x, s["curves"]["v"]["real"], color=style.SERIES[0], label=L["legend"]["value"])
    curve(ax, x, s["curves"]["abs"]["real"], color=style.SERIES[1], label=L["legend"]["abs"])
    basecurve(ax, x, s["curves"]["v"]["base"], label=L["baseline"])
    basecurve(ax, x, s["curves"]["abs"]["base"])
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, -0.1), max(hi, 0.6))
    hline(ax, s["breakeven"], L)
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 0.88))


def p_r5(ax, rec, L, extra):
    s, x = p_lines(ax, rec, L, "layers", logy=True)
    hline(ax, s["breakeven"], L, xpos=x[len(x) // 2])
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 0.88))


def p_r6(ax, rec, L, extra):
    s = rec["summary"]
    x = np.asarray(s["ks"], dtype=np.float64)
    floor = 1e-8
    fix = lambda b: {k: [floor if (v is None or v <= floor) else v for v in b[k]] for k in ("median", "min", "max")}
    curve(ax, x, fix(s["real"]), label=L["real"], marker="o")
    basecurve(ax, x, fix(s["base"]), label=L["baseline"])
    ax.set_yscale("log")
    ax.set_ylim(floor / 2, 1.0)
    ax.text(0.98, 0.66, L["legend"]["zero_floor"], transform=ax.transAxes, ha="right", fontsize=5.5,
            color=style.BASE)
    ax.set_xlim(min(x[0], s["breakeven"]) - 2, x[-1] + 2)
    vline(ax, s["breakeven"], L)
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 0.88))


def p_r7a(ax, rec, L, extra):
    s = rec["summary"]
    x = np.asarray(s["x"], dtype=np.float64)
    if s["kind"] == "unit":
        ax.bar(x, s["real"], color=style.REAL, width=0.8, label=L["real"])
        ax.step(x, s["base"], where="mid", color=style.BASE, lw=0.9, label=L["baseline"])
    else:
        w = x[1] - x[0]
        ax.bar(x, s["real"], color=style.REAL, width=w, label=L["real"])
        ax.step(x, s["base"], where="mid", color=style.BASE, lw=0.9, label=L["baseline"])
        nz = np.flatnonzero((np.asarray(s["real"]) > 0) | (np.asarray(s["base"]) > 0))
        hi = max(x[nz[-1]] + 3 * w if len(nz) else 1.0, s["line"] * 1.3)
        ax.set_xlim(0, min(1.0, hi))
    ax.set_yscale("log")
    vline(ax, s["line"], L)
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 0.88))


def p_r7b(ax, rec, L, extra):
    s = rec["summary"]
    x = np.asarray(s["ks"], dtype=np.float64)
    curve(ax, x, s["real"], label=L["real"], marker="o")
    basecurve(ax, x, s["base"], label=L["baseline"])
    ax.plot(x, s["breakeven"], color=style.LINE, ls="--", lw=1.0)
    line_label(ax, x[0], s["breakeven"][0], L)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(k)}" for k in x], fontsize=6)
    ax.legend(fontsize=6, loc="lower left")


def p_r8(ax, rec, L, extra):
    s = rec["summary"]
    keys = s["keys"]
    ks = s["ks"]
    for i, sv in enumerate(s["ss"]):
        idx = [keys.index(f"{k},{sv}") for k in ks]
        b = {q: [s["real"][q][j] for j in idx] for q in ("median", "min", "max")}
        curve(ax, ks, b, color=style.SERIES[i], label=L["legend"]["s"].format(s=sv), marker="o")
        ax.plot(ks, [s["base"]["median"][j] for j in idx], color=style.BASE, lw=0.8)
        ax.plot(ks, [s["breakeven"][j] for j in idx], color=style.LINE, ls="--", lw=1.0)
        ax.plot(ks, [s["breakeven_alt"][j] for j in idx], color=style.LINE, ls=":", lw=0.8)
    j0 = keys.index(f"{ks[0]},{s['ss'][0]}")
    line_label(ax, ks[0], s["breakeven"][j0], L)
    ax.text(0.98, 0.03, f"-- b = {s['b']}    ··· b = {s['b_alt']}", transform=ax.transAxes, ha="right",
            fontsize=5.5, color=style.LINE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6, loc="upper left")


def p_r9(ax, rec, L, extra):
    s = rec["summary"]
    x = np.arange(len(s["x"]))
    med, lo, hi = arr(s["observed"]["median"]), arr(s["observed"]["min"]), arr(s["observed"]["max"])
    ax.bar(x, med, color=style.REAL, width=0.8, label=L["legend"]["observed"])
    if np.nanmax(hi - lo) > 0:
        ax.vlines(x, lo, hi, color="#56B4E9", lw=2.0, alpha=0.8)
    ax.plot(x, arr(s["expected"]["median"]), color=style.BASE, lw=0.9, marker="o", ms=2,
            label=L["legend"]["expected"])
    labels = s.get("x_labels") or [str(v) for v in s["x"]]
    step = max(1, len(x) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], fontsize=6)
    ax.legend(fontsize=6, loc="upper left")


DRAW = {"R1": p_r1, "R2": p_r2, "R3": p_r3, "R4": p_r4, "R5": p_r5, "R6": p_r6, "R7a": p_r7a,
        "R7b": p_r7b, "R8": p_r8, "R9": p_r9}


def draw_panel(ax, rec, title, L, extra=None):
    fig_id = rec["figure"]
    bid = base_id(fig_id)
    ax.set_title(title, fontsize=8)
    if rec["verdict"] == "not_applicable" or not rec.get("summary"):
        ax.text(0.5, 0.5, L["na_text"], transform=ax.transAxes, ha="center", va="center", fontsize=10,
                color=style.BASE)
        ax.set_xticks([])
        ax.set_yticks([])
        reason = rec.get("detail", {}).get("reason", "")
        ax.text(0.5, -0.12, reason, transform=ax.transAxes, ha="center", va="top", fontsize=6)
    else:
        DRAW["R7a" if bid.startswith("R7a") else bid](ax, rec, L, extra or {})
        ax_lbl = L["axes"].get(bid, L["axes"].get(fig_id, ["", ""]))
        if bid == "R9":
            ax_lbl = [L["axes"]["R9"][0].format(name=rec["summary"].get("name", "")), L["axes"]["R9"][1]]
        ax.set_xlabel(ax_lbl[0], fontsize=7)
        ax.set_ylabel(ax_lbl[1], fontsize=7)
        ax.tick_params(labelsize=6)
        ax.annotate(rec.get("key", ""), xy=(0.5, 0.0), xycoords="axes fraction", xytext=(0, -KEY_PT),
                    textcoords="offset points", ha="center", va="top", fontsize=6.5)
    ax.text(0.98, 0.97, verdict_text(rec, L), transform=ax.transAxes, ha="right", va="top", fontsize=7,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=style.BASE, lw=0.5))


def table_lines(rec, kind, L):
    if kind == "R2" and rec.get("summary", {}).get("groups"):
        out = [L["r2_table"]]
        for name, g in rec["summary"]["groups"].items():
            mark = "✓" if g["pays"] else ("•" if g["exceeds"] else " ")
            out.append(f"{mark} {name}: {g['net']:.3g} / {g['cost']:.3g}")
        return out
    if kind == "R10" and rec and rec.get("summary", {}).get("table"):
        out = [L["r10_table"]]
        for t in rec["summary"]["table"]:
            out.append(f"{t['name']}: {t['log2_count']:.4g}, {t['bits_per_symbol']:.3g}")
        return out
    return []


def triptych(recs, titles, L, path, suptitle, tables=None, extras=None):
    n = len(recs)
    tables = tables or [[] for _ in recs]
    extras = extras or [{} for _ in recs]
    nlines = max(len(t) for t in tables)
    below = (TABLE_PT + LINE_PT * nlines + 6) / 72 if nlines else (KEY_PT + 14) / 72
    h = style.PANEL_H + below
    fig, axes = plt.subplots(1, n, figsize=(n * style.PANEL_W, h), squeeze=False)
    for ax, rec, t, tab, ex in zip(axes[0], recs, titles, tables, extras):
        draw_panel(ax, rec, t, L, ex)
        if tab:
            ax.annotate("\n".join(tab), xy=(0.0, 0.0), xycoords="axes fraction", xytext=(0, -TABLE_PT),
                        textcoords="offset points", ha="left", va="top", fontsize=5.5, linespacing=1.1)
    fig.suptitle(suptitle, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.99, top=1 - 0.5 / h, bottom=(below + 0.05) / h, wspace=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=style.DPI)
    plt.close(fig)


def synthetic_for(fig_id, rec, manifest):
    if fig_id in manifest["synthetic"]:
        return fig_id
    if fig_id.startswith("R9"):
        k = "R9_" + rec.get("summary", {}).get("kind", "")
        return k if k in manifest["synthetic"] else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overview", default=None)
    ap.add_argument("--lang", default=None, choices=("zh", "en"))
    ap.add_argument("--synthetic", default=None,
                    help="results directory whose synthetic panels are used when --results has none")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    L, lang, font, msg = style.setup(args.lang)
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
    r10 = load(res / "data" / "R10.json") if (res / "data" / "R10.json").exists() else None
    written = []
    if man["figures"]:
        for fig_id in man["figures"]:
            if fig_id == "R10":
                continue
            rec = load(res / "data" / f"{fig_id}.json")
            sk = synthetic_for(fig_id, rec, man)
            recs, titles = [], []
            if sk:
                recs += [syn[sk]["none"], syn[sk]["has"]]
                titles += [L["panel_none"], L["panel_has"]]
            recs.append(rec)
            titles.append(man["label"])
            tables = [[] for _ in recs]
            extras = [{} for _ in recs]
            if fig_id == "R2":
                tables = [table_lines(r, "R2", L) for r in recs]
            if fig_id == "R1" and r10:
                tables[-1] = table_lines(r10, "R10", L)
                extras[-1] = {"classes": [c for t in r10["summary"].get("table", []) for c in t["symbol_classes"]]}
            title = L["titles"].get(base_id(fig_id) if fig_id.startswith("R9") else fig_id, fig_id)
            if fig_id.startswith("R9"):
                title = f"{title}: {fig_id[3:]}"
            path = out / style.file_name(fig_id)
            triptych(recs, titles, L, path, title, tables, extras)
            written.append(path.name)
        overview(man, res, L, Path(args.overview) if args.overview else out)
    else:
        for k, v in syn.items():
            recs = [v["none"], v["has"]]
            tables = [table_lines(r, "R2", L) for r in recs] if k == "R2" else None
            title = L["titles"].get("R9" if k.startswith("R9") else k, k)
            name = style.file_name(k if not k.startswith("R9") else k)
            triptych(recs, [L["panel_none"], L["panel_has"]], L, out / name, title, tables)
            written.append(name)
    print("figures:", ", ".join(written), flush=True)


def overview(man, res, L, odir):
    """2 x 5 grid of the measured panels and a markdown table (one row per figure)."""
    odir.mkdir(parents=True, exist_ok=True)
    figs = man["figures"]
    pick = []
    for want in ("R1", "R2", "R3", "R4", "R5", "R6", "R7a_unit", "R7b", "R8"):
        if want in figs:
            pick.append(want)
    r9 = [f for f in figs if f.startswith("R9")]
    if r9:
        pick.append(r9[0])
    fig, axes = plt.subplots(2, 5, figsize=(5 * style.PANEL_W, 2 * (style.PANEL_H + 0.6)))
    for ax, fid in zip(axes.ravel(), pick):
        rec = load(res / "data" / f"{fid}.json")
        t = L["titles"].get("R9" if fid.startswith("R9") else fid, fid)
        draw_panel(ax, rec, t, L)
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"{L['overview']}: {man['label']}", fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.1, wspace=0.3, hspace=0.8)
    for ext in ("png", "pdf"):
        fig.savefig(odir / f"overview.{ext}", dpi=style.DPI)
    plt.close(fig)
    lines = ["| " + " | ".join(L["md_header"]) + " |", "|" + " --- |" * len(L["md_header"])]
    for fid in figs:
        rec = load(res / "data" / f"{fid}.json")
        bid = "R9" if fid.startswith("R9") else fid
        pat = rec["pattern"]
        esc = lambda t: t.replace("|", "\\|")
        lines.append(f"| {pat} | `{style.file_name(fid) if fid != 'R10' else 'R1_distribution'}` | "
                     f"{esc(L['indicator'].get(bid, ''))} | {verdict_text(rec, L)} | {esc(rec.get('key', ''))} |")
    with open(odir / "overview.md", "w", encoding="utf-8") as f:
        f.write(f"# {L['overview']}: {man['label']}\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
