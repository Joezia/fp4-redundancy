"""Spec 005: aggregate the work pickles into experiments/005-r9/ (CSV files, PNG figures and the
tables used by summary.md). Usage: python -m diagnostics.r9_report --work <dir>"""
import argparse
import csv
import glob
import json
import math
import os
import pickle
import statistics as st
from pathlib import Path

import numpy as np

from .r9_run import B3_GROUPS, B3_K, SAMPLES, names

OUT = Path("experiments/005-r9")


def load_all(work, prefix):
    out = {}
    for p in sorted(glob.glob(os.path.join(work, f"{prefix}*.pkl"))):
        with open(p, "rb") as f:
            out[Path(p).stem] = pickle.load(f)
    return out


def fmt(x, nd=7):
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{nd}g}"
    return x


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in fields})


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return st.median(xs) if xs else float("nan")


def rng_(xs):
    return (min(xs), max(xs)) if xs else (float("nan"), float("nan"))


def layer_of(m):
    return m.split(".")[0]


# ----------------------------------------------------------------------------- A

def report_a(ab, nm):
    fields = ["matrix", "set", "E", "n_blocks", "q_odd", "frac0_real", "frac0_binom", "frac0_S0E",
              "frac0_ratio_real_binom", "tv_real_binom", "tv_S0E_binom"] + \
             [f"g{j}_real" for j in range(5)] + [f"g{j}_S0E" for j in range(5)] + \
             ["A3_h_E", "A3_h_Eg", "A3_gain", "A3_label_cost", "A3_net"]
    rows = []
    for key, r in ab.items():
        m = r["spec"]["matrix"]
        s = "full" if m in nm["full"] else "probe"
        for x in r["A_rows"]:
            rows.append({"matrix": m, "set": s, **x,
                         "frac0_ratio_real_binom": x["frac0_real"] / x["frac0_binom"]
                         if x["frac0_binom"] > 0 else None})
        a = r["A"]
        rows.append({"matrix": m, "set": s, "E": "all", "A3_h_E": a["h_E"], "A3_h_Eg": a["h_Eg"],
                     "A3_gain": a["gain"], "A3_label_cost": a["label_cost"], "A3_net": a["net"]})
    write_csv(OUT / "a_grid.csv", fields, rows)
    return rows


def report_a4(a4, ab, nm):
    fields = ["source", "dist", "tau", "rule", "max3", "max4", "max6", "top1_E", "top1_share",
              "top1_q_odd", "top2_E", "top2_share", "top2_q_odd", "frac0", "frac0_binom",
              "frac0_ratio", "share_0.5", "share_1", "share_1.5", "count_order"]
    rows = [{"source": "synthetic", **x} for x in a4["A4"]]
    real = [r["rules"] for r in ab.values() if r["spec"]["matrix"] in nm["full"]]
    mrow = {"source": f"real median ({len(real)} full-layer matrices)"}
    for k in fields[4:]:
        vals = [x.get(k) for x in real]
        if k == "count_order":
            from collections import Counter
            mrow[k] = " / ".join(f"{o}:{c}" for o, c in Counter(vals).most_common())
        elif k in ("top1_E", "top2_E"):
            mrow[k] = int(med(vals))
        else:
            mrow[k] = med(vals)
    rows.append(mrow)
    write_csv(OUT / "a_rules.csv", fields, rows)
    # distance of each candidate to the real median (L1 over the numeric statistics, normalized)
    keys = ["max3", "max4", "max6", "top1_q_odd", "top2_q_odd", "frac0_ratio"]
    dist = []
    for x in rows[:-1]:
        d = 0.0
        for k in keys:
            a, b = x[k], mrow[k]
            if k == "frac0_ratio":
                a, b = math.log(max(a, 1e-9)), math.log(max(b, 1e-9))
            d += abs(a - b)
        dist.append((d, x))
    dist.sort(key=lambda t: t[0])
    return rows, mrow, dist


# ----------------------------------------------------------------------------- B

def report_b(ab, nm, dprobe):
    fields = ["matrix", "set", "variant", "K", "codelen", "gain", "cost", "net", "sum_K_E",
              "codelen_half0", "codelen_half1", "d_block_composition_004"]
    rows = []
    b1 = {}
    for r in ab.values():
        m = r["spec"]["matrix"]
        s = "probe" if m in nm["probe"] else "full"
        for var, res in (("real", r["B1"]), ("S0E", r["B1_S0E"])):
            for K, x in res.items():
                rows.append({"matrix": m, "set": s, "variant": var, "K": K, "codelen": x["codelen"],
                             "gain": x["gain"], "cost": x["cost"], "net": x["net"],
                             "sum_K_E": max(sum(d.values()) for d in x["K_E"]),
                             "codelen_half0": x["codelen_halves"][0],
                             "codelen_half1": x["codelen_halves"][1]})
        b1[m] = r["B1"]
    for m in nm["probe"]:
        if m in dprobe and m in b1:
            rows.append({"matrix": m, "set": "probe", "variant": "B4", "K": 32,
                         "gain": b1[m][32]["gain"], "net": b1[m][32]["net"],
                         "d_block_composition_004": dprobe[m]})
    write_csv(OUT / "b_mixture.csv", fields, rows)
    cf = ["matrix", "E", "K", "K_E", "n_blocks", "type", "prior", "odd_share", "zero_share",
          "mean_absv", "p_max6"] + [f"theta_{c}" for c in range(16)]
    crow = []
    for r in ab.values():
        for x in r["B2"]:
            crow.append({"matrix": r["spec"]["matrix"], **{k: v for k, v in x.items() if k != "theta"},
                         **{f"theta_{c}": x["theta"][c] for c in range(16)}})
    write_csv(OUT / "b_classes.csv", cf, crow)
    return rows, b1, crow


def report_b3(b3e, b3fit, b1, nm, sel):
    fields = ["scope", "group", "K", "fit_slot", "eval_matrix", "codelen_shared", "codelen_B1",
              "loss_vs_B1", "gain_vs_B1_K1", "unseen_blocks", "shared_model_cost_amortized"]
    rows = []
    N = 2304 * 5120
    for key, r in b3e.items():
        sp = r["spec"]
        scope, slot = sp["scope"], sp["fit_slot"]
        for (group, K), x in r["results"].items():
            kes = x["K_E"]
            n_eval = sum(1 for m in r["eval_matrices"] if group == "all" or m.endswith(group))
            cost_bits = sum(kes.values()) * 16 * 16
            for i, m in enumerate(r["eval_matrices"]):
                if group != "all" and not m.endswith("." + group):
                    continue
                cl = x["bits"][i] / (x["n_blocks"][i] * 32)
                ref = b1.get(m, {}).get(K, {}).get("codelen")
                ref1 = b1.get(m, {}).get(1, {}).get("codelen")
                rows.append({"scope": scope, "group": group, "K": K, "fit_slot": slot,
                             "eval_matrix": m, "codelen_shared": cl, "codelen_B1": ref,
                             "loss_vs_B1": None if ref is None else cl - ref,
                             "gain_vs_B1_K1": None if ref1 is None else ref1 - cl,
                             "unseen_blocks": int(x["unseen_blocks"][i]),
                             "shared_model_cost_amortized": cost_bits / (N * n_eval)})
    write_csv(OUT / "b_shared.csv", fields, rows)
    return rows


# ----------------------------------------------------------------------------- C

def report_c1(c1):
    fields = ["matrix", "item", "k", "r2", "r2_baseline", "baseline", "strict_k", "strict_gain",
              "strict_cost", "strict_net", "H_E", "H_E_bin", "gain_per_scale"]
    rows = []
    for r in c1.values():
        m = r["spec"]["matrix"]
        for item, res, bk, bl in (("C1", r["C1"], "r2_S0E", "S0E"), ("C1b", r["C1b"], "r2_perm",
                                                                       "permutation")):
            for k, v in res["r2"].items():
                rows.append({"matrix": m, "item": item, "k": k, "r2": v,
                             "r2_baseline": res[bk][k], "baseline": bl})
            s = res["strict"]
            if s:
                if item == "C1":
                    rows.append({"matrix": m, "item": "C1 strict", "strict_k": s["k"],
                                 "strict_gain": s["gain"], "strict_cost": s["cost"],
                                 "strict_net": s["net"], "H_E": s["h_E"], "H_E_bin": s["h_Ebin"]})
                else:
                    rows.append({"matrix": m, "item": "C1b strict", "strict_k": s["k"],
                                 "strict_gain": s["gain"], "strict_cost": s["factor_cost"],
                                 "strict_net": s["net"], "H_E": s["H_E"], "H_E_bin": s["H_E_bin"],
                                 "gain_per_scale": s["gain_per_scale"]})
            else:
                rows.append({"matrix": m, "item": f"{item} strict",
                             "strict_k": f"not run: max R2 {res['max_r2']:.3g} <= 0.001"})
    write_csv(OUT / "c_magnitude.csv", fields, rows)
    return rows


def report_c2(c2, c2c):
    fields = ["matrix", "set", "variant", "base_test", "net_test", "gain", "cost", "net",
              "best_epoch", "best_val", "base_val", "n_emb", "n_params_net"]
    rows = []
    for coll, setname in ((c2, "dev"), (c2c, "conf")):
        for r in coll.values():
            sp = r["spec"]
            x = r["C2"]
            rows.append({"matrix": sp["matrix"], "set": setname, "variant": sp["variant"],
                         **{k: x[k] for k in fields[3:]}})
    write_csv(OUT / "c_inr.csv", fields, rows)
    return rows


# ----------------------------------------------------------------------------- figures

def figures(ab, nm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # n_odd histograms: 3 probe matrices (L0 w1, L13 w2, L39 w3), top-2 E each
    picks = [m for m in nm["probe"] if m.startswith("L0.") and m.endswith("w1")] + \
            [m for m in nm["probe"] if m.startswith("L13.") and m.endswith("w2")] + \
            [m for m in nm["probe"] if m.startswith("L39.") and m.endswith("w3")]
    fig, axes = plt.subplots(len(picks), 2, figsize=(10, 3.0 * len(picks)), squeeze=False)
    for i, m in enumerate(picks):
        r = ab[f"ab_{m}"]
        top = sorted(r["A_rows"], key=lambda x: -x["n_blocks"])[:2]
        for j, x in enumerate(top):
            ax = axes[i][j]
            k = np.arange(33)
            ax.bar(k, x["hist_real"], color="#4C72B0", label="real", alpha=0.8)
            ax.plot(k, x["hist_binom"], "o-", color="#DD8452", ms=3, label="binomial Bin(32, q_E)")
            ax.plot(k, x["hist_S0E"], "--", color="#55A868", lw=1, label="S0E")
            ax.set_title(f"{m}  E={x['E']}  (share {x['n_blocks'] / sum(y['n_blocks'] for y in r['A_rows']):.2f}, "
                         f"q={x['q_odd']:.3f})", fontsize=9)
            ax.set_xlabel("n_odd (codewords with |v| in {0.5, 1.5} per block)")
            ax.set_xlim(-0.5, 32.5)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "n_odd_hist.png", dpi=110)
    plt.close(fig)
    # mixture gain vs K
    Ks = [1, 2, 4, 8, 16, 32]
    real = {K: [r["B1"][K]["gain"] for r in ab.values() if r["spec"]["matrix"] in nm["full"]] for K in Ks}
    s0 = {K: [r["B1_S0E"][K]["gain"] for r in ab.values() if r["spec"]["matrix"] in nm["full"]]
          for K in (1, 2, 8, 32)}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.array(Ks)
    ax.fill_between(x, [min(real[K]) for K in Ks], [max(real[K]) for K in Ks], color="#4C72B0",
                    alpha=0.2, label="real: range")
    ax.plot(x, [st.median(real[K]) for K in Ks], "o-", color="#4C72B0", label="real: median")
    xs = np.array([1, 2, 8, 32])
    ax.fill_between(xs, [min(s0[K]) for K in xs], [max(s0[K]) for K in xs], color="#C44E52", alpha=0.2)
    ax.plot(xs, [st.median(s0[K]) for K in xs], "s--", color="#C44E52", label="S0E: median (range shaded)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K (types per scale value)")
    ax.set_ylabel("held-out gain over K = 1 (bit/weight)")
    ax.set_title(f"B1 mixture gain, {len(real[1])} full-layer matrices")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "mixture_gain.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- variants

def report_variants(work, meta):
    """meta: list of dicts (id, part, hyp, struct, conf_selected) in the order tried."""
    rows = []
    for v in meta:
        vid = v["id"]
        sp = Path(work) / f"synth_{vid}.pkl"
        syn = pickle.load(open(sp, "rb"))["result"] if sp.exists() else None
        if v.get("orig") == "B":
            # original design = B1 with K = 32 (dev: probe matrices; confirmation: full-layer matrices)
            ab = load_all(work, "ab_")
            nm = names(json.load(open(SAMPLES)))

            def b1(ms):
                return {m: {"result": ab[f"ab_{m}"]["B1"][32]} for m in ms if f"ab_{m}" in ab}
            devs, confs = b1(nm["b_dev"]), b1(nm["b_conf"])
        elif v.get("orig") == "C":
            devs = {k: {"result": r["C2"]} for k, r in load_all(work, "c2_").items()
                    if r["spec"]["variant"] == "real"}
            confs = {k: {"result": r["C2"]} for k, r in load_all(work, "c2conf_").items()}
        else:
            devs = load_all(work, f"dev_{vid}_")
            confs = load_all(work, f"conf_{vid}_")

        def agg(coll):
            if not coll:
                return {}
            g = [x["result"]["gain"] for x in coll.values()]
            c = [x["result"]["cost"] for x in coll.values()]
            n = [x["result"]["net"] for x in coll.values()]
            return {"gain": float(np.mean(g)), "cost": float(np.mean(c)), "net": float(np.mean(n)),
                    "net_median": med(n), "net_min": min(n), "net_max": max(n), "n": len(n)}

        d, c = agg(devs), agg(confs)
        syn_txt = ""
        if syn:
            syn_txt = (f"独立底 净值 {syn['null']['net']:.4g}（{'通过' if syn['null_pass'] else '不通过'}）；"
                       f"埋入：{syn['planted']['construction']}，收益 {syn['planted']['gain']:.4g}"
                       + (f"，理论 {syn['planted']['theory']:.4g}" if "theory" in syn["planted"] else "")
                       + f"，判据 {syn['planted']['criterion']}（{'通过' if syn['planted_pass'] else '不通过'}）")
        rows.append({"id": vid, "part": v["part"], "hypothesis": v["hyp"], "structure": v["struct"],
                     "synthetic": syn_txt, "synthetic_pass": None if syn is None else syn["pass"],
                     "dev_gain": d.get("gain"), "dev_cost": d.get("cost"), "dev_net": d.get("net"),
                     "dev_net_median": d.get("net_median"), "dev_n": d.get("n"),
                     "confirmed": "是" if v.get("conf") else "否",
                     "conf_gain": c.get("gain"), "conf_cost": c.get("cost"), "conf_net": c.get("net"),
                     "conf_net_median": c.get("net_median"), "conf_n": c.get("n"),
                     "note": v.get("note", "")})
    fields = ["id", "part", "hypothesis", "structure", "synthetic", "synthetic_pass", "dev_gain",
              "dev_cost", "dev_net", "dev_net_median", "dev_n", "confirmed", "conf_gain", "conf_cost",
              "conf_net", "conf_net_median", "conf_n", "note"]
    write_csv(OUT / "variants.csv", fields, rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--variants-meta", default=None)
    args = ap.parse_args()
    with open(SAMPLES) as f:
        sel = json.load(f)
    nm = names(sel)
    ab = load_all(args.work, "ab_")
    ab = {k: v for k, v in ab.items()}
    ab_by = {k: v for k, v in ab.items()}
    dprobe = {}
    dcsv = Path("experiments/004-gaps/d_probe.csv")
    if dcsv.exists():
        g = {}
        for r in csv.DictReader(open(dcsv)):
            if r["component"] == "mixture" and r["metric"] == "gain":
                g[(r["matrix"], r["variant"])] = float(r["value"])
        for m in nm["probe"]:
            if (m, "S2") in g and (m, "S0E") in g:
                dprobe[m] = g[(m, "S2")] - g[(m, "S0E")]
    report_a(ab, nm)
    a4 = load_all(args.work, "a4")
    rules = report_a4(a4["a4"], ab, nm) if "a4" in a4 else None
    brows, b1, crow = report_b(ab, nm, dprobe)
    b3 = report_b3(load_all(args.work, "b3eval_"), None, b1, nm, sel)
    c1 = report_c1(load_all(args.work, "c1_"))
    c2 = report_c2(load_all(args.work, "c2_"), load_all(args.work, "c2conf_"))
    figures(ab_by, nm)
    if args.variants_meta:
        report_variants(args.work, json.load(open(args.variants_meta)))
    print("written", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
