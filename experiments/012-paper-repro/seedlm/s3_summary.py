"""S3 summary: S3-a (lossy, SeedLM alone) and S3-b (lossless: SeedLM side information + conditional code) per configuration.

All numbers are aggregated over the matrices of one set (sums of bits / sums of weights; rel. MSE weighted by n).
usage: python s3_summary.py [SET] [--plot]
Writes results/s3/s3_summary_{SET}.json, prints markdown tables, and with --plot draws
results/s3/side_vs_code_{SET}.png (side information vs conditional code length).
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "../results/s3")
PAPER_CFG = {(8, 3, 16), (12, 4, 16)}
MXFP4_BPW = 4.25


def load(which):
    a = defaultdict(list)
    b = defaultdict(list)
    for l in open(os.path.join(R, f"s3a_{which}.jsonl")):
        r = json.loads(l)
        a[(r["C"], r["P"], r["K"], r["mode"])].append(r)
    p = os.path.join(R, f"s3b_{which}.jsonl")
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            b[(r["C"], r["P"], r["K"], r["mode"])].append(r)
    return a, b


def summarise(which):
    a, b = load(which)
    out = []
    for key in sorted(a):
        C, P, K, mode = key
        ra = {r["matrix"]: r for r in a[key]}
        rb = {r["matrix"]: r for r in b.get(key, [])}
        n = sum(r["n"] for r in ra.values())
        row = dict(C=C, P=P, K=K, mode=mode, table1=(C, P, K) in PAPER_CFG, matrices=len(ra),
                   bits_per_element=next(iter(ra.values()))["bits_per_element"],
                   rel_mse=sum(r["rel_mse"] * r["n"] for r in ra.values()) / n,
                   code_mismatch=sum(r["code_mismatch"] * r["n"] for r in ra.values()) / n,
                   floor_frac=sum(r["floor_frac"] * r["n_blocks"] for r in ra.values()) / sum(r["n_blocks"] for r in ra.values()))
        if len(rb) == len(ra):
            m = sum(r["n"] for r in rb.values())
            subs = defaultdict(int)
            for r in rb.values():
                subs[r["sub"]] += 1
            row.update(side_bpw=sum(r["side_bits"] for r in rb.values()) / m,
                       code_bpw=sum(r["code_bits_actual"] for r in rb.values()) / m,
                       scale_bpw=sum(r["scale_bits"] for r in rb.values()) / m,
                       total_bpw=sum(r["total_bits"] for r in rb.values()) / m,
                       ref_total_bpw=sum(r["ref_total_bits"] for r in rb.values()) / m,
                       net_gain=sum(r["net_gain_bits"] for r in rb.values()) / m,
                       roundtrip=all(r["roundtrip_ok"] for r in rb.values()),
                       sub=dict(sorted(subs.items())),
                       decode_codes_per_s=m / sum(r["decode_s"] for r in rb.values()))
        out.append(row)
    return out


def plot(rows, which):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cols = {8: "#2a78d6", 12: "#eda100", 16: "#eb6834", 32: "#1baf7a"}
    ink, muted, grid = "#2b2b29", "#6b6a63", "#e4e3dd"
    rows = [r for r in rows if "side_bpw" in r]
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ref_code = rows[0]["ref_total_bpw"] - rows[0]["scale_bpw"]
    xs = [r["side_bpw"] for r in rows]
    x0, x1 = 0, max(xs) * 1.05
    ytop = max(r["code_bpw"] for r in rows) * 1.08
    for tot in (4, 4.5, 5, 5.5):
        ax.plot([x0, x1], [tot - x0, tot - x1], color=grid, lw=1, zorder=0)
        ax.text(tot - ytop * 0.96, ytop * 0.96, f" {tot}", color=muted, fontsize=7, va="center")
    ax.plot([x0, x1], [ref_code - x0, ref_code - x1], color=ink, lw=1.2, ls="--", zorder=1)
    ax.text(0.04, 0.12, f"参考基准：边信息 + 码值 = {ref_code:.3f}（灰线：其他等和线，均不含 scale 流）", color=ink, fontsize=8)
    for r in rows:
        face = cols[r["C"]] if r["mode"] == "range" else "#fcfcfb"
        ax.scatter(r["side_bpw"], r["code_bpw"], s=46 if r["table1"] else 26, marker="o",
                   facecolor=face, edgecolor=cols[r["C"]], linewidth=1.6, zorder=3)
        if r["table1"]:
            ax.annotate(f"表1 ({r['C']},{r['P']},{r['K']}) {r['mode']}", (r["side_bpw"], r["code_bpw"]),
                        xytext=(6, 4), textcoords="offset points", fontsize=7, color=ink)
    for C in sorted({r["C"] for r in rows}):
        ax.scatter([], [], color=cols[C], label=f"C = {C}")
    ax.scatter([], [], facecolor="#888", edgecolor="#888", label="range（实心）")
    ax.scatter([], [], facecolor="#fcfcfb", edgecolor="#888", linewidth=1.6, label="paper（空心）")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_xlim(x0, x1)
    ax.set_ylim(0, ytop)
    ax.set_xlabel("SeedLM 边信息（bit/权重）", color=ink)
    ax.set_ylabel("条件码长（bit/权重）", color=ink)
    ax.set_title(f"S3-b：边信息与条件码长（{which}，虚线以下才比参考基准短）", fontsize=10, color=ink)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted)
    ax.grid(False)
    fig.tight_layout()
    p = os.path.join(R, f"side_vs_code_{which}.png")
    fig.savefig(p, facecolor=fig.get_facecolor())
    print("figure:", p)


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    which = args[0] if args else "T1"
    rows = summarise(which)
    json.dump(rows, open(os.path.join(R, f"s3_summary_{which}.json"), "w"), indent=1)
    print(f"S3-a（{which}，有损；对照 MXFP4 本身 {MXFP4_BPW} bit/权重）")
    print("| C | P | K | 指数规则 | 表 1 | bit/权重 | 占 4.25 bit | 相对均方误差 | 码值不一致率 | 指数在下限的块 |")
    print("|---:|---:|---:|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['C']} | {r['P']} | {r['K']} | {r['mode']} | {'是' if r['table1'] else ''} | {r['bits_per_element']:.4f} | {r['bits_per_element'] / MXFP4_BPW * 100:.1f}% | "
              f"{r['rel_mse']:.4f} | {r['code_mismatch'] * 100:.1f}% | {r['floor_frac'] * 100:.1f}% |")
    print()
    print(f"S3-b（{which}，无损：边信息 + 条件码 + scale 流）")
    print("| C | P | K | 指数规则 | 子区间数（矩阵数） | 边信息 | 条件码 | 总码长 | 占 4.25 bit | 相对参考基准 | 逐位一致 |")
    print("|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|")
    for r in sorted([r for r in rows if "side_bpw" in r], key=lambda r: -r["net_gain"]):
        sub = "、".join(f"{k}（{v}）" for k, v in r["sub"].items())
        print(f"| {r['C']} | {r['P']} | {r['K']} | {r['mode']} | {sub} | {r['side_bpw']:.4f} | {r['code_bpw']:.4f} | "
              f"{r['total_bpw']:.4f} | {r['total_bpw'] / MXFP4_BPW * 100:.1f}% | {r['net_gain']:+.4f} | {'是' if r['roundtrip'] else '否'} |")
    if "--plot" in sys.argv:
        plot(rows, which)


if __name__ == "__main__":
    main()
