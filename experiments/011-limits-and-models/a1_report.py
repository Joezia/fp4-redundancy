"""Spec 011 A1 report: scaling curves and the fit L(N) = L_inf + a * N^(-alpha).

    PYTHONPATH=. python experiments/011-limits-and-models/a1_report.py

Reads results/a/a1.jsonl; writes results/a/a1_summary.json and figures/a1_params.png, figures/a1_data.png.
L is the test code length of the codes (bit per code); the gain over the chain is chain - L.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def fit(x, y):
    from scipy.optimize import curve_fit
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = lambda n, L, a, al: L + a * n ** (-al)
    try:
        p, cov = curve_fit(f, x, y, p0=[y.min() - 1e-3, 1.0, 0.5], maxfev=20000)
        resid = y - f(x, *p)
        err = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else [None] * 3
        return {"L_inf": float(p[0]), "a": float(p[1]), "alpha": float(p[2]), "rms_residual": float(np.sqrt(np.mean(resid ** 2))),
                "L_inf_std": None if err[0] is None else float(err[0]), "points": len(x)}
    except Exception as e:   # too few points or no convergence
        return {"error": str(e), "points": len(x)}


def main():
    rows = [json.loads(l) for l in open(HERE / "results" / "a" / "a1.jsonl")]
    out = {"runs": [{"size": r["size"], "N": r["N"], "params": r["params"], "sha256": r["sha256"],
                     "T1_bits": r["T1"]["bits_per_code"], "T2_bits": r["T2"]["bits_per_code"],
                     "T1_gain": r["T1"]["gain"], "T2_gain": r["T2"]["gain"],
                     "T1_net_test": r["T1"]["net_gain_cost_over_test"], "T1_net_model": r["T1"]["net_gain_cost_over_whole_model"],
                     "T2_net_test": r["T2"]["net_gain_cost_over_test"], "T2_net_model": r["T2"]["net_gain_cost_over_whole_model"],
                     "best_val": r["train"]["best_val"], "wall_s": r["train"]["wall_s"]} for r in rows]}
    big = [r for r in rows if r["N"] == 10 ** 9]
    big.sort(key=lambda r: r["params"])
    top = max(rows, key=lambda r: r["params"])["size"]
    dat = sorted([r for r in rows if r["size"] == top], key=lambda r: r["N"])
    for split in ("T1", "T2"):
        out[f"fit_params_{split}"] = fit([r["params"] for r in big], [r[split]["bits_per_code"] for r in big])
        out[f"fit_data_{split}"] = fit([r["N"] for r in dat], [r[split]["bits_per_code"] for r in dat])
        ch = rows[0][split]["chain_bits_per_code"]
        for k in ("fit_params", "fit_data"):
            fr = out[f"{k}_{split}"]
            if "L_inf" in fr:
                fr["gain_at_L_inf"] = ch - fr["L_inf"]
    json.dump(out, open(HERE / "results" / "a" / "a1_summary.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from redundancy import style
        style.setup()
    except Exception:
        pass
    (HERE / "figures").mkdir(exist_ok=True)
    for kind, sel, xkey, xlabel in (("params", big, "params", "Parameters"), ("data", dat, "N", "Training positions")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for split, mk in (("T1", "o-"), ("T2", "s--")):
            x = [r[xkey] for r in sel]
            y = [r[split]["gain"] for r in sel]
            ax.plot(x, y, mk, label=f"{split} (test)")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Gain over the chain model (bit per weight)")
        ax.set_title("Correction MLP: gain vs " + ("model size" if kind == "params" else "training data"))
        ax.legend()
        fig.tight_layout()
        fig.savefig(HERE / "figures" / f"a1_{kind}.png", dpi=150)
        plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
