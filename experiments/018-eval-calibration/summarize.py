"""Tables and figure for 018 from results/<method>.json.

  PYTHONPATH=. python experiments/018-eval-calibration/summarize.py
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
PPL = ["wikitext2", "c4"]
MC = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "mmlu"]


def load():
    runs = []
    for f in sorted(RES.glob("*.json")):
        if f.name.startswith(("selfcheck", "summary")):
            continue
        d = json.loads(f.read_text())
        if "A_weights" in d:
            runs.append(d)
    key = lambda d: (d["method"] != "identity", xdb(d) if xdb(d) is not None else -1e9)
    return sorted(runs, key=key)


def xdb(d):
    """x axis: median over matrices of the per-matrix weight NMSE. The energy-weighted total is dominated by the
    deep-layer down matrices (their mean W^2 is 10^3-10^5 times that of gate_up), so it mostly reports those."""
    return d["A_weights"].get("matrix_nmse_db_median")


def f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    runs = load()
    rows, lines = [], []
    lines.append("| 扰动 | 权重 NMSE，矩阵中位数 / 按能量合计 (dB) | 局部 MoE 误差中位数 (dB) | 末层隐状态误差 (dB) | 路由改变比例（中位数） | "
                 + " | ".join(f"{p} KL / top-1 / PPL" for p in PPL) + " |")
    lines.append("|" + "---|" * (5 + len(PPL)))
    for d in runs:
        B = d["B_layers"]
        loc = [x for x in B["local_moe_nmse_db"] if x > -290]
        cells = [d["spec"], f(xdb(d)) + " / " + f(d["A_weights"]["nmse_db"]),
                 f(float(np.median(loc))) if loc else "—",
                 f(B["hidden_nmse_db"][-1]) if B["hidden_nmse_db"][-1] > -290 else "—",
                 f(float(np.median(B["route_changed_frac"])), 3)]
        for p in PPL:
            c = d["C_output"].get(p)
            cells.append("—" if c is None else f"{c['kl_mean']:.4f} / {c['top1_agree']:.3f} / {c['ppl_ref']:.1f}→{c['ppl_pert']:.1f}")
        lines.append("| " + " | ".join(cells) + " |")
        rows.append(d)
    lines.append("")
    lines.append("| 扰动 | " + " | ".join(MC) + " | 平均精度变化 |")
    lines.append("|" + "---|" * (len(MC) + 2))
    for d in runs:
        cells, deltas = [d["spec"]], []
        for t in MC:
            x = d["D_tasks"].get(t)
            if x is None:
                cells.append("—"); continue
            cells.append(f"{x['acc_ref']:.3f}→{x['acc_pert']:.3f}（−{x['acc_flip_right_to_wrong']}/+{x['acc_flip_wrong_to_right']}）")
            deltas.append(x["acc_pert"] - x["acc_ref"])
        cells.append(f"{np.mean(deltas):+.3f}" if deltas else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("多选题 KL / top-1 一致率（续写位置）：")
    lines.append("")
    lines.append("| 扰动 | " + " | ".join(MC) + " |")
    lines.append("|" + "---|" * (len(MC) + 1))
    for d in runs:
        cells = [d["spec"]] + [("—" if t not in d["C_output"] else f"{d['C_output'][t]['kl_mean']:.4f} / {d['C_output'][t]['top1_agree']:.3f}") for t in MC]
        lines.append("| " + " | ".join(cells) + " |")
    (RES / "tables.md").write_text("\n".join(lines) + "\n")
    summary = [dict(spec=d["spec"], weight_nmse_db=d["A_weights"]["nmse_db"], bits_per_weight=d["A_weights"].get("bits_per_weight"),
                    local_moe_nmse_db=d["B_layers"]["local_moe_nmse_db"], hidden_nmse_db=d["B_layers"]["hidden_nmse_db"],
                    route_changed_frac=d["B_layers"]["route_changed_frac"],
                    C={k: v for k, v in d["C_output"].items()}, D=d["D_tasks"], seconds=d["seconds"]) for d in runs]
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    plot(runs)
    print("\n".join(lines))


def plot(runs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    wn = [d for d in runs if d["method"] == "white_noise"]
    rtn = [d for d in runs if d["method"] == "block_rtn"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for grp, mk, lab in ((wn, "o-", "white noise"), (rtn, "s", "block RTN (k bit)")):
        if not grp:
            continue
        x = [xdb(d) for d in grp]
        ax[0].plot(x, [np.median([v for v in d["B_layers"]["local_moe_nmse_db"]]) for d in grp], mk, label=lab)
        for p, ls in (("wikitext2", "-"), ("c4", "--")):
            if all(p in d["C_output"] for d in grp):
                ax[1].plot(x, [d["C_output"][p]["kl_mean"] for d in grp], mk[0] + ls, label=f"{lab}, {p}")
        ax[2].plot(x, [np.mean([d["D_tasks"][t]["acc_pert"] - d["D_tasks"][t]["acc_ref"] for t in MC if t in d["D_tasks"]]) for d in grp], mk, label=lab)
    ax[0].plot([-45, 2], [-45, 2], ":", color="gray", label="y = x")
    ax[0].set(xlabel="weight NMSE, median over matrices (dB)", ylabel="median local MoE output NMSE (dB)", title="B: layer output")
    ax[1].set(xlabel="weight NMSE, median over matrices (dB)", ylabel="mean KL(p_orig || p_pert) (nats)", yscale="log", title="C: output distribution")
    ax[2].set(xlabel="weight NMSE, median over matrices (dB)", ylabel="mean accuracy change (6 tasks)", title="D: tasks")
    ax[2].axvline(-2.4, color="gray", ls=":")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle("gpt-oss-20b, perturbing all MXFP4 expert weights")
    fig.tight_layout()
    fig.savefig(RES / "calibration.png", dpi=130)


if __name__ == "__main__":
    main()
