"""B1 summary table (markdown) from results/b1/b1_*.json.

The first runs stored the official test() value divided by 784 twice (fixed in b1_mnist.py afterwards); such
values (< 0.05) are multiplied back by 784 here and flagged.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "../results/b1")
PAPER = {"bin": dict(bbans=0.19, elbo=0.19), "full": dict(bbans=1.41, elbo=1.39)}


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(R, "b1_*.json"))):
        if p.endswith("summary.json"):
            continue
        r = json.load(open(p))
        off = r["neg_elbo_official_test"]
        fixed = off < 0.05
        if fixed:
            off *= 784
        bp = r["bpd_official"]
        rows.append(dict(kind=r["kind"], mode=r["mode"], n=r["n_images"], bbans=bp, with_init=r["bpd_with_initial"],
                         elbo_official=off, elbo_official_rescaled=fixed, elbo_mc=r["neg_elbo_mc20"],
                         gap_mc_pct=100 * (bp / r["neg_elbo_mc20"] - 1), gap_off_pct=100 * (bp / off - 1),
                         paper_bbans=PAPER[r["kind"]]["bbans"], paper_elbo=PAPER[r["kind"]]["elbo"],
                         diff_paper=bp - PAPER[r["kind"]]["bbans"], pass_002=abs(bp - PAPER[r["kind"]]["bbans"]) <= 0.02,
                         roundtrip=r["roundtrip_ok"] and r["other_bits_recovered"], init_bits=r["init_bits"],
                         first_pop_bits=r["first_image_posterior_pop_bits"], encode_s=r["encode_s"],
                         decode_s=r["decode_s"], ppf_fix=r.get("ppf_fix", False)))
    json.dump(rows, open(os.path.join(R, "b1_summary.json"), "w"), indent=1)
    print("| 数据 | 图像数 | BB-ANS（bit/维） | 含初始比特 | 负 ELBO（官方 test） | 负 ELBO（20 次采样） | 码长/负 ELBO − 1 | 论文 BB-ANS / ELBO | 与论文差 | 逐位一致 | 第一张图后验弹出的比特 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for r in rows:
        name = {"bin": "二值 MNIST", "full": "原始 MNIST"}[r["kind"]] + ("（三份打乱拷贝）" if r["mode"] == "test3x" else "") + ("，ppf 修正" if r["ppf_fix"] else "")
        print(f"| {name} | {r['n']:,} | {r['bbans']:.4f} | {r['with_init']:.4f} | {r['elbo_official']:.4f} | "
              f"{r['elbo_mc']:.4f} | {r['gap_mc_pct']:+.2f}% | {r['paper_bbans']} / {r['paper_elbo']} | "
              f"{r['diff_paper']:+.4f} | {'是' if r['roundtrip'] else '否'} | {r['first_pop_bits']} |")


if __name__ == "__main__":
    main()
