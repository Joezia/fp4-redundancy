"""Spec 011 line C: tables of C2 / C3 / C4 (gain over the chain, both cost amortisations) and the C3 figure.

    PYTHONPATH=. python experiments/011-limits-and-models/c_report.py

Writes results/c/c_summary.json and figures/c3_scope.png.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "c"


def rows(fn):
    p = RES / fn
    return [json.loads(l) for l in open(p)] if p.exists() else []


def slim(r):
    out = {"config": r["config"], "params": r["params"], "N": r["N"], "sha256": r["sha256"]}
    for s in ("T1", "T2"):
        if s in r:
            out[s] = {k: round(v, 5) for k, v in r[s].items() if k != "n" and isinstance(v, float)}
    return out


def main():
    out = {}
    for key, fn in (("C2", "c2_coord.jsonl"), ("C3", "c3_scope.jsonl"), ("C4", "c4_hyperprior.jsonl")):
        out[key] = [slim(r) for r in rows(fn)]
    json.dump(out, open(RES / "c_summary.json", "w"), indent=1)
    for k, v in out.items():
        print(k)
        for r in v:
            print("  ", r["config"], r.get("T1"), r.get("T2"))
    c3 = {r["config"]: r for r in out["C3"]}
    ladder = [c for c in ("base", "M", "M+E", "M+E+L", "M+E+L+G") if c in c3]
    if len(ladder) >= 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for s, mk in (("T1", "o-"), ("T2", "s--")):
            ax.plot(range(len(ladder)), [c3[c][s]["gain"] for c in ladder], mk, label=s)
        ax.set_xticks(range(len(ladder)))
        ax.set_xticklabels(["local (A1)", "+ matrix", "+ expert", "+ layer", "+ model"][:len(ladder)])
        ax.set_ylabel("Gain over the chain model (bit per weight)")
        ax.set_title("Correction MLP: gain vs what the model can see")
        ax.legend()
        fig.tight_layout()
        (HERE / "figures").mkdir(exist_ok=True)
        fig.savefig(HERE / "figures" / "c3_scope.png", dpi=150)


if __name__ == "__main__":
    main()
