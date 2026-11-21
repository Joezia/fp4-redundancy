"""B3 summary (markdown) from results/b3/b3_K*.jsonl, against the reference (008 chain + 011 R9c, incl. the scale stream).

Totals: BB-ANS message (gross: everything in the flattened message, incl. the initial words for the official single
chain) + model cost (sum_E K_E * 16 * 16 bits) + the 008 scale stream (reference.json scale_bits). Units bit/weight.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "../results/b3")
REF = os.path.join(HERE, "../../010-transforms/results/tx/reference.json")
ALL_W = 557171343360


def summary(K, which):
    suffix = "" if which == "T1" else "_" + which
    rows = [json.loads(l) for l in open(os.path.join(R, f"b3_K{K}{suffix}.jsonl"))]
    ref = json.load(open(REF))[which]["per_matrix"]
    n = sum(r["n_weights"] for r in rows)
    marg = sum(r["marginal_bits"] for r in rows)
    k1 = sum(r["k1_bits"] for r in rows)
    cost = sum(r["model_cost_bits"] for r in rows)
    scale = sum(ref[r["matrix"]]["scale_bits"] for r in rows)
    reft = sum(ref[r["matrix"]]["total"] for r in rows)
    out = dict(K=K, set=which, matrices=len(rows), n=n, marginal_bpw=marg / n, k1_bpw=k1 / n,
               gain_vs_k1=(k1 - marg) / n, model_cost_bpw=cost / n, scale_bpw=scale / n, ref_total_bpw=reft / n,
               runs=[])
    for N in (1, 64, 1024, 16384):
        rr = [x for r in rows for x in r["runs"] if x["N"] == N]
        if len(rr) != len(rows):
            continue
        g = sum(x["bits_gross"] for x in rr)
        tot = g + cost + scale
        dec = sum(x["decode_s"] for x in rr)
        enc = sum(x["encode_s"] for x in rr)
        out["runs"].append(dict(N=N, impl=rr[0]["impl"], gross_bpw=g / n,
                                net_bpw=(sum(x["bits_net"] for x in rr) / n) if N == 1 else None,
                                overhead_vs_marginal=(g - marg) / n, total_bpw=tot / n, net_gain=(reft - tot) / n,
                                roundtrip=all(x["roundtrip_ok"] for x in rr),
                                borrowed_words=sum(x.get("borrowed_words", 0) for x in rr),
                                decode_codes_per_s_cpu=n / dec, encode_codes_per_s=n / enc))
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "T1"
    res = []
    for K in (16, 32):
        p = os.path.join(R, f"b3_K{K}{'' if which == 'T1' else '_' + which}.jsonl")
        if os.path.exists(p):
            res.append(summary(K, which))
    json.dump(res, open(os.path.join(R, f"b3_summary_{which}.json"), "w"), indent=1)
    for s in res:
        print(f"K={s['K']} ({s['set']}, {s['matrices']} 个矩阵)：边际似然码长 {s['marginal_bpw']:.5f}，K=1 码长 {s['k1_bpw']:.5f}，"
              f"收益 {s['gain_vs_k1']:.5f}，模型成本 {s['model_cost_bpw']:.5f}，scale 流 {s['scale_bpw']:.5f}，参考基准 {s['ref_total_bpw']:.5f}")
        print("| 流数 N | 实现 | 码值部分（实际，含初始比特） | 比边际似然多 | 总码长（+模型+scale） | 相对参考基准的净收益 | 逐位一致 | CPU 解码（码值/秒） |")
        print("|---:|---|---:|---:|---:|---:|---|---:|")
        for r in s["runs"]:
            print(f"| {r['N']:,} | {r['impl']} | {r['gross_bpw']:.5f} | {r['overhead_vs_marginal']:+.5f} | {r['total_bpw']:.5f} | "
                  f"{r['net_gain']:+.5f} | {'是' if r['roundtrip'] else '否'} | {r['decode_codes_per_s_cpu']:,.0f} |")


if __name__ == "__main__":
    main()
