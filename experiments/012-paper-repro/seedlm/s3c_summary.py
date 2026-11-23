"""S3-c summary on T1 (24 matrices, weighted by weights): writes results/s3c/s3c_summary_T1.md.

Entropy-coded side information = per stream the cheapest of {fixed bits, the adaptive coders}:
seeds min(fixed, adaptive), exponent min(fixed, ctx E, ctx (E, previous e)), coefficients min(fixed, ctx (p, e)).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "../results/s3c")
CFG = [(8, 3, 16), (12, 4, 16)]


def load(f):
    return [json.loads(l) for l in open(os.path.join(D, f))]


def ent(r, fixed_e=False):
    return (min(r["seed_fixed"], r["seed_adapt"]) + min(r["e_fixed"], r["e_adapt_E"], r["e_adapt_Eprev"])
            + min(r["q_fixed"], r["q_adapt_pe"]))


def main():
    side, share, bb = load("side_T1.jsonl"), load("share_T1.jsonl"), load("bb_T1.jsonl")
    s3b = [json.loads(l) for l in open(os.path.join(HERE, "../results/s3/s3b_T1.jsonl"))]
    out = ["# S3-c summary, T1 (24 matrices), bit/weight; reference 3.6056; all residual streams round-trip exactly\n"]
    for C, P, K in CFG:
        out.append(f"\n## C={C}, P={P}, K={K} (range)\n")
        rs = [r for r in side if (r["C"], r["P"], r["K"]) == (C, P, K)]
        n = sum(r["n"] for r in rs)
        out.append("\n### side information (S3-a solutions)\n\n| stream | fixed | adaptive, no ctx | ctx 1 | ctx 2 | H0 |\n|---|---:|---:|---:|---:|---:|")
        g = lambda k: sum(r[k] for r in rs) / n  # noqa: E731
        out.append(f"| seed | {g('seed_fixed'):.4f} | {g('seed_adapt'):.4f} | | | {g('seed_h0'):.4f} |")
        out.append(f"| exponent | {g('e_fixed'):.4f} | {g('e_adapt_none'):.4f} | {g('e_adapt_E'):.4f} (E) | {g('e_adapt_Eprev'):.4f} (E, prev e) | |")
        out.append(f"| coefficients | {g('q_fixed'):.4f} | {g('q_adapt_none'):.4f} | {g('q_adapt_p'):.4f} (p) | {g('q_adapt_pe'):.4f} (p, e) | |")
        out.append("\n### sharing\n\n| variant | G | rel MSE | code mismatch | side fixed | code | total fixed | net fixed | side entropy-coded | total | net | round trip |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for v in ["raw_range", "norm_range", "norm_fixed"]:
            for G in [1, 2, 8, 32, 128, "all"]:
                xs = [r for r in share if (r["C"], r["P"], r["K"], r["variant"], r["G"]) == (C, P, K, v, G)]
                assert len(xs) == 24, (v, G, len(xs))
                n = sum(r["n"] for r in xs)
                f = lambda k: sum(r[k] for r in xs) / n  # noqa: E731
                w = lambda k: sum(r[k] * r["n"] for r in xs) / n  # noqa: E731
                se = sum(ent(r) for r in xs) / n
                te = se + f("code_bits") + f("scale_bits")
                out.append(f"| {v} | {G} | {w('rel_mse'):.4f} | {w('code_mismatch'):.3f} | {f('side_fixed_bits'):.4f} | {f('code_bits'):.4f} | "
                           f"{f('total_bits'):.4f} | {f('ref_total_bits') - f('total_bits'):+.4f} | {se:.4f} | {te:.4f} | {f('ref_total_bits') - te:+.4f} | "
                           f"{all(r['roundtrip_ok'] for r in xs)} |")
        xs = [r for r in bb if (r["C"], r["P"], r["K"]) == (C, P, K)]
        assert len(xs) == 24
        n = sum(r["n"] for r in xs)
        f = lambda k: sum(r[k] for r in xs) / n  # noqa: E731
        w = lambda k: sum(r[k] * r["n"] for r in xs) / n  # noqa: E731
        base = [r for r in s3b if (r["C"], r["P"], r["K"], r["mode"]) == (C, P, K, "range")]
        bcode = sum(r["code_bits_actual"] for r in base) / sum(r["n"] for r in base)
        se = sum(ent(r) for r in xs) / n
        te = se + f("code_bits") + f("scale_bits")
        out.append("\n### seed chosen by code length; bits-back bound (frozen train8 table, table not charged)\n")
        out.append(f"- static code length of the residual: MSE seed {f('static_l_mse'):.4f} (numpy check {f('static_l_mse_numpy'):.4f}); "
                   f"code-length seed {f('static_l_min'):.4f} (numpy {f('static_l_min_numpy'):.4f}); mixture / bits-back bound {f('static_mixture'):.4f}")
        out.append(f"- log2 effective number of seeds per block: mean {sum(r['log2_eff_seeds_mean'] for r in xs) / len(xs):.3f}, "
                   f"10/50/90% {[round(sum(r['log2_eff_seeds_q'][i] for r in xs) / len(xs), 3) for i in range(3)]}")
        out.append(f"- code-length seed, really coded: rel MSE {w('rel_mse'):.4f}, mismatch {w('code_mismatch'):.3f}, code {f('code_bits'):.4f} "
                   f"(MSE seed {bcode:.4f}), total fixed side {f('total_bits'):.4f} ({f('ref_total_bits') - f('total_bits'):+.4f}), "
                   f"entropy-coded side {se:.4f}, total {te:.4f} ({f('ref_total_bits') - te:+.4f}), round trip {all(r['roundtrip_ok'] for r in xs)}")
    open(os.path.join(D, "s3c_summary_T1.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
