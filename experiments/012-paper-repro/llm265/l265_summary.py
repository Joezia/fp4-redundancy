"""LLM.265 on FP4: summary per QP (sums of bits / sums of weights over the matrices of a set).

usage: python l265_summary.py [SET] [--plot]
Writes results/llm265/l265_summary_{SET}.json, prints a markdown table, and with --plot draws
results/llm265/lossy_vs_total_{SET}.png (lossy bits vs lossless total).
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "../results/llm265")


def order(t):
    return (0, int(t[2:])) if t.startswith("qp") else (1, t)


def summarise(which):
    rows = defaultdict(dict)
    for line in open(os.path.join(R, f"l265_{which}.jsonl")):
        r = json.loads(line)
        rows[r["target"]][r["matrix"]] = r
    out = []
    for t in sorted(rows, key=order):
        rs = list(rows[t].values())
        n = sum(r["n"] for r in rs)
        row = dict(target=t, matrices=len(rs),
                   lossy_bpw=sum(r["hevc_bits"] for r in rs) / n,
                   qparam_bpw=sum(r["qparam_bits"] for r in rs) / n,
                   mse_uint8=sum(r["mse_uint8"] * r["n"] for r in rs) / n,
                   rel_mse=sum(r["rel_mse"] * r["n"] for r in rs) / n,
                   code_mismatch=sum(r["code_mismatch"] * r["n"] for r in rs) / n,
                   decode_twice_equal=all(r["decode_twice_equal"] for r in rs))
        if all("code_bits_actual" in r for r in rs):
            row.update(sub=sorted({r["sub"] for r in rs}),
                       code_bpw=sum(r["code_bits_actual"] for r in rs) / n,
                       scale_bpw=sum(r["scale_bits"] for r in rs) / n,
                       total_bpw=sum(r["total_bits"] for r in rs) / n,
                       ref_total_bpw=sum(r["ref_total_bits"] for r in rs) / n,
                       net_gain=sum(r["net_gain_bits"] for r in rs) / n,
                       roundtrip=all(r["roundtrip_ok"] for r in rs))
        else:
            best = min(rs[0]["ideal_bits"], key=lambda s: sum(r["ideal_bits"][s] for r in rs))
            row.update(sub_ideal=int(best), code_bpw_ideal=sum(r["ideal_bits"][best] for r in rs) / n)
        out.append(row)
    return out


def main():
    which = next((a for a in sys.argv[1:] if not a.startswith("--")), "T1")
    out = summarise(which)
    json.dump(out, open(os.path.join(R, f"l265_summary_{which}.json"), "w"), indent=1)
    if "total_bpw" in out[0]:
        print("| setting | HEVC (lossy) | + FP16 scale/zero | rel. MSE | MSE (uint8) | codes changed | S | conditional code"
              " | scale stream | total (lossless) | net vs reference |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in out:
            print(f"| {r['target']} | {r['lossy_bpw']:.4f} | {r['lossy_bpw'] + r['qparam_bpw']:.4f} | {r['rel_mse']:.5f} | "
                  f"{r['mse_uint8']:.3f} | {100 * r['code_mismatch']:.2f}% | {r['sub']} | {r['code_bpw']:.4f} | "
                  f"{r['scale_bpw']:.4f} | {r['total_bpw']:.4f} | {r['net_gain']:+.4f} |")
        print("reference total:", round(out[0]["ref_total_bpw"], 4), "round trips:", all(r["roundtrip"] for r in out),
              "two decodes equal:", all(r["decode_twice_equal"] for r in out))
    else:
        for r in out:
            print(r)
    if "--plot" in sys.argv and "total_bpw" in out[0]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        qs = [r for r in out if r["target"].startswith("qp")]
        x = [r["lossy_bpw"] + r["qparam_bpw"] for r in qs]
        ax.plot(x, [r["total_bpw"] for r in qs], "o-", color="#2a6fbb", label="无损总计")
        ax.plot(x, [r["code_bpw"] for r in qs], "s--", color="#d9822b", label="条件码（残差）")
        for r, xi in zip(qs, x):
            ax.annotate(r["target"].upper(), (xi, r["total_bpw"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.axhline(out[0]["ref_total_bpw"], color="#444", lw=1, ls=":", label=f"参考基准 {out[0]['ref_total_bpw']:.3f}")
        ax.axhline(4.25, color="#999", lw=1, ls="-.", label="MXFP4 原始 4.25")
        ax.set_xlabel("有损部分 bit/权重（HEVC 码流 + FP16 scale/zero）")
        ax.set_ylabel("bit/权重")
        ax.set_title(f"LLM.265 用于 FP4：有损码率与无损总计（{which}）")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(R, f"lossy_vs_total_{which}.png"), dpi=150)


if __name__ == "__main__":
    main()
