"""B2 summary (markdown): single chain (official) and N streams (Craystack) for the selected VAE, T1 and T2.

Totals: BB-ANS message (gross, incl. the initial words of the single chain) + the 008 scale stream; the model
(params x 16 bit) is shown separately, amortised over the coded matrices and over the whole model.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "../results/b2")
REF = os.path.join(HERE, "../../010-transforms/results/tx/reference.json")
ALL_W = 557171343360
TAG = "vae_h500_d20_e20"
PARAMS = 290600


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "T1"
    ref = json.load(open(REF))[which]["per_matrix"]
    code = [json.loads(l) for l in open(os.path.join(R, "code", TAG + ".jsonl"))]
    code = [r for r in code if r["matrix"] in ref]
    st = [json.loads(l) for l in open(os.path.join(R, "streams", TAG + ".jsonl"))]
    st = [r for r in st if r["matrix"] in ref]
    n = sum(r["n_weights"] for r in code)
    scale = sum(ref[r["matrix"]]["scale_bits"] for r in code) / n
    reft = sum(ref[r["matrix"]]["total"] for r in code) / n
    elbo = sum(r["neg_elbo_bits"] for r in code) / n
    cost_here, cost_all = PARAMS * 16 / n, PARAMS * 16 / ALL_W
    out = dict(set=which, matrices=len(code), neg_elbo_bpw=elbo, scale_bpw=scale, ref_total_bpw=reft,
               model_cost_these=cost_here, model_cost_whole=cost_all, runs=[])
    g = sum(r["bits_gross"] for r in code) / n
    out["runs"].append(dict(N=1, impl="official bits-back", device="cpu", matrices=len(code), gross_bpw=g,
                            net_official_bpw=sum(r["bits_net_official"] for r in code) / n,
                            total_bpw=g + scale, net_gain=reft - g - scale,
                            roundtrip=all(r["roundtrip_ok"] and r["other_bits_recovered"] for r in code),
                            decode_codes_per_s=n / sum(r["decode_s"] for r in code)))
    for dev in ("cpu", "cuda"):
        for N in (64, 1024, 16384):
            rr = [r for r in st if r["device"] == dev and r["N"] == N]
            if not rr:
                continue
            m = sum(r["n_weights"] for r in rr)
            gg = sum(r["bits_gross"] for r in rr) / m
            sc = sum(ref[r["matrix"]]["scale_bits"] for r in rr) / m
            rt = sum(ref[r["matrix"]]["total"] for r in rr) / m
            out["runs"].append(dict(N=N, impl="craystack", device=dev, matrices=len(rr), gross_bpw=gg,
                                    total_bpw=gg + sc, net_gain=rt - gg - sc,
                                    roundtrip=all(r["roundtrip_ok"] for r in rr),
                                    decode_codes_per_s=m / sum(r["decode_s"] for r in rr)))
    json.dump(out, open(os.path.join(R, f"b2_summary_{which}.json"), "w"), indent=1)
    print(f"{which}（{out['matrices']} 个矩阵）：负 ELBO {elbo:.5f}，scale 流 {scale:.5f}，参考基准 {reft:.5f}，"
          f"模型成本 {cost_here:.5f}（摊到这些矩阵）/ {cost_all:.1e}（摊到全模型）")
    print("| 流数 N | 实现 | 网络所在 | 矩阵数 | 码值部分（实际，含初始比特） | 总码长（+scale） | 相对参考基准（不含模型） | 逐位一致 | 解码（码值/秒） |")
    print("|---:|---|---|---:|---:|---:|---:|---|---:|")
    for r in out["runs"]:
        print(f"| {r['N']:,} | {r['impl']} | {r['device'].upper()} | {r['matrices']} | {r['gross_bpw']:.5f} | {r['total_bpw']:.5f} | "
              f"{r['net_gain']:+.5f} | {'是' if r['roundtrip'] else '否'} | {r['decode_codes_per_s']:,.0f} |")


if __name__ == "__main__":
    main()
