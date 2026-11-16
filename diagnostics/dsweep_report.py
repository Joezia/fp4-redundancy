"""Spec 006: aggregate the dictionary sweep into sweep.csv and dict_sweep.png; print the median table
(the summary.md text is written by hand from it).
Usage: python -m diagnostics.dsweep_report --work <dir>
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import dsweep
from .dsweep_run import probe_matrices

OUT = Path("experiments/006-dict-sweep")
FIELDS = ["matrix", "main_scale", "n_fit", "n_eval", "K", "s", "R2_real", "R2_S0E", "R2_S2",
          "real_minus_S0E", "real_minus_S2", "breakeven_b16", "breakeven_b8", "R2_pca", "R2_pca_S0E",
          "R2_003"]


def r2_003():
    """003's R2_dict at s = 2 (dictionary learned with s = 2, K = 64) per matrix."""
    out = {}
    for r in csv.DictReader(open("experiments/003-r5-r8/r8_sparse.csv")):
        if r["item"] == "R8.2" and r["variable"] == "R2_dict" and r["k"] == "2" and r["baseline"] == "S0E":
            out[r["matrix"]] = float(r["value"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    ref = r2_003()
    res = {}
    for _, s in probe_matrices():
        for v in dsweep.VARIANTS:
            with open(Path(args.work) / f"{s['matrix']}_{v}.json") as f:
                d = json.load(f)
            res[(s["matrix"], v)] = {"dict": {(K, sv): x for K, sv, x in d["dict"]},
                                     "pca": {sv: x for sv, x in d["pca"]}, "meta": d}
    mats = [s["matrix"] for _, s in probe_matrices()]
    rows = []
    for m in mats:
        r, a, b = (res[(m, v)] for v in dsweep.VARIANTS)
        for K in dsweep.KS:
            for sv in dsweep.SS:
                rows.append({"matrix": m, "main_scale": r["meta"]["main"], "n_fit": r["meta"]["n_fit"],
                             "n_eval": r["meta"]["n_eval"], "K": K, "s": sv,
                             "R2_real": r["dict"][(K, sv)], "R2_S0E": a["dict"][(K, sv)],
                             "R2_S2": b["dict"][(K, sv)],
                             "real_minus_S0E": r["dict"][(K, sv)] - a["dict"][(K, sv)],
                             "real_minus_S2": r["dict"][(K, sv)] - b["dict"][(K, sv)],
                             "breakeven_b16": dsweep.breakeven(K, sv, 16),
                             "breakeven_b8": dsweep.breakeven(K, sv, 8),
                             "R2_pca": r["pca"][sv], "R2_pca_S0E": a["pca"][sv],
                             "R2_003": ref.get(m) if (K, sv) == (64, 2) else None})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.8g}" if isinstance(v, float) else ("" if v is None else v))
                        for k, v in r.items()})

    def col(K, sv, key):
        return np.array([r[key] for r in rows if r["K"] == K and r["s"] == sv])

    print("K s | real med [min,max] | S0E med | S2 med | real-S0E med [min,max] | real-S2 med | "
          "be16 | be8 | pca med | real>be8")
    for K in dsweep.KS:
        for sv in dsweep.SS:
            re, a, b = col(K, sv, "R2_real"), col(K, sv, "R2_S0E"), col(K, sv, "R2_S2")
            d, d2 = re - a, re - b
            print(f"{K} {sv} | {np.median(re):.4f} [{re.min():.4f},{re.max():.4f}] | {np.median(a):.4f} | "
                  f"{np.median(b):.4f} | {np.median(d):+.5f} [{d.min():+.5f},{d.max():+.5f}] | "
                  f"{np.median(d2):+.5f} | {dsweep.breakeven(K, sv, 16):.4f} | "
                  f"{dsweep.breakeven(K, sv, 8):.4f} | {np.median(col(K, sv, 'R2_pca')):.4f} | "
                  f"{int((re > dsweep.breakeven(K, sv, 8)).sum())}")
    d003 = [r["R2_real"] - r["R2_003"] for r in rows if r["R2_003"] is not None]
    print("(64,2) real - 003:", f"max |.| {max(abs(x) for x in d003):.3g}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    ks = np.array(dsweep.KS)
    for ax, sv in zip(axes, dsweep.SS):
        re = np.array([col(K, sv, "R2_real") for K in ks])
        a = np.array([col(K, sv, "R2_S0E") for K in ks])
        ax.fill_between(ks, re.min(axis=1), re.max(axis=1), color="#0072B2", alpha=0.2, lw=0)
        ax.plot(ks, np.median(re, axis=1), "o-", color="#0072B2", label="real (median, range)")
        ax.plot(ks, np.median(a, axis=1), "s-", color="#999999", lw=1, ms=3, label="S0E (median)")
        for b, ls in ((16, "--"), (8, ":")):
            ax.plot(ks, [dsweep.breakeven(K, sv, b) for K in ks], ls, color="#D55E00",
                    label=f"break-even b = {b}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_title(f"s = {sv}")
        ax.set_xlabel("dictionary size K")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("held-out R²")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "dict_sweep.png", dpi=150)
    print("written", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
