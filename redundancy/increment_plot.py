"""Summary bar chart of the incremental saving per pattern (spec 008; same style as slide 13).

    python -m redundancy.increment_plot --result <aggregate json> --dataset "<display name>" --out <png>

Bars: the patterns in the default order, then Total. Negative increments are drawn as 0 (the result
files keep the raw negative values); Total is the raw total saving.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import style  # noqa: E402

NAMES = {"R1c": "R1 codes", "R1s": "R1 scales"}
TOTAL = "#444444"
BAR = "#0072B2"


def draw(res, dataset, out, order_name="default"):
    style.setup()
    o = res["orders"][order_name]
    pats = o["order"]
    vals = [max(0.0, o["increment_raw"][p]) for p in pats] + [o["total"]]
    labels = [NAMES.get(p, p) for p in pats] + ["Total"]
    fig, ax = plt.subplots(figsize=(style.PANEL_W * 1.5, style.PANEL_H))
    xs = range(len(vals))
    ax.bar(xs, vals, color=[BAR] * len(pats) + [TOTAL], width=0.7, zorder=2)
    for x, v in zip(xs, vals):
        ax.text(x, v, f"{v:.2f}" if v < 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=style.FONT["tick"] - 2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=0 if len(labels) <= 10 else 30, ha="center" if len(labels) <= 10 else "right")
    ax.set_ylabel("Saving (% of raw bits)")
    ax.set_title(f"Incremental saving per pattern on {dataset}")
    ax.set_ylim(0, max(vals) * 1.12 if max(vals) > 0 else 1)
    ax.grid(axis="y", color="#dddddd", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=style.DPI)
    fig.savefig(str(out).rsplit(".", 1)[0] + ".pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    draw(json.load(open(a.result)), a.dataset, a.out)


if __name__ == "__main__":
    main()
