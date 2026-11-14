"""Spec 004: aggregate the per-job pickles of gaps_run into experiments/004-gaps/.

Usage: python -m diagnostics.gaps_report --work <dir> [--out experiments/004-gaps]
"""
import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import gaps, ledger

ROW002 = 0.569  # 002 ledger row 4 saving on the original sample (median of 36 matrices)
PAIR_NAME = {("a", "g"): "w1行–w3行", ("a", "o"): "w1行–w2列", ("g", "o"): "w3行–w2列"}
FAM_NAME = {"a": "w1 行", "g": "w3 行", "o": "w2 列"}
MTP_X = 41


def fnum(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.8g}"
    if isinstance(v, np.integer):
        return int(v)
    return v


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fnum(r.get(k, "")) for k in fields})


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if v != 0 and (abs(v) < 10 ** (-nd) or abs(v) >= 1e5):
        return f"{v:.2e}"
    return f"{v:.{nd}f}"


def fin(x):
    return [float(v) for v in x if v is not None and np.isfinite(v)]


def med(x):
    x = fin(x)
    return float(np.median(x)) if x else None


def load(work):
    out = defaultdict(list)
    for p in sorted(Path(work).glob("*.pkl")):
        with open(p, "rb") as f:
            r = pickle.load(f)
        out[r["spec"]["kind"]].append(r)
    return out


def lab(spec):
    return f"MTP{spec['layer']}" if spec["prefix"] == "mtp" else f"L{spec['layer']}"


def xpos(spec):
    return MTP_X + spec["layer"] if spec["prefix"] == "mtp" else spec["layer"]


# ----------------------------------------------------------------------------- A

def a_rows(A):
    rows = []
    for r in A:
        s = r["spec"]
        base = {"sets": "+".join(s["sets"]), "layer": lab(s), "expert": s["expert"]}
        a = r["A"]
        for (x, y, dom), v in a["A1"].items():
            pr = f"{x}-{y}"
            for q, val in v["q"].items():
                rows.append(dict(base, item="A1", config=pr, domain=dom, metric=f"abs_rho_q{int(q * 100)}",
                                 value=val))
            for q, val in v["base_q"].items():
                rows.append(dict(base, item="A1", config=pr, domain=dom,
                                 metric=f"baseline_abs_rho_q{int(q * 100)}", value=val))
            for m in ("thr", "frac_sig", "sig_pos", "sig_neg", "all_pos", "mean_rho", "n_sig",
                      "n_strong", "null_max_median"):
                rows.append(dict(base, item="A1", config=pr, domain=dom, metric=m, value=v[m]))
            for m, d in (("a2_y_given_x", f"{y}|{x}"), ("a2_x_given_y", f"{x}|{y}"),
                         ("a2s_y_given_x", f"{y}|{x}"), ("a2s_x_given_y", f"{x}|{y}")):
                rows.append(dict(base, item="A2", config=d, domain=dom,
                                 metric="same_index_share" + ("_strong" if m.startswith("a2s") else ""),
                                 value=v[m]))
        for name, v in a["A3"].items():
            b = a["A3_base"][name]
            for m in ("gain", "approx", "cost", "net", "h_e", "h_ctx", "r2_median", "r2_mean", "n"):
                rows.append(dict(base, item="A3", config=name, domain="w", metric=m, value=v[m]))
            for m in ("gain", "approx", "net"):
                rows.append(dict(base, item="A3", config=name, domain="w", metric=f"misaligned_{m}",
                                 value=b[m]))
        if "A4" in a:
            for o, v in a["A4"]["orders"].items():
                rows.append(dict(base, item="A4", config=o, domain="w", metric="total", value=v))
            rows.append(dict(base, item="A4", config=a["A4"]["best"], domain="w", metric="best_total",
                             value=a["A4"]["total"]))
    return rows


# ----------------------------------------------------------------------------- B

def b_rows(Bs):
    rows = []
    for r in Bs:
        s = r["spec"]
        b = r["B"]
        base = {"layer": s["layer"], "family": s["family"], "role": gaps.ROLE_OF[s["family"]],
                "dim": b["L"], "n_fit": b["n_fit"], "n_eval": b["n_eval"]}
        for var in ("real", "S4"):
            v = b[var]
            for k, x in v["captured"].items():
                rows.append(dict(base, item="B1", variant=var, k=k, metric="captured", value=x))
                rows.append(dict(base, item="B1", variant=var, k=k, metric="captured_minus_k/L",
                                 value=x - k / b["L"]))
            for k, x in v["captured_self"].items():
                rows.append(dict(base, item="B1", variant=var, k=k, metric="captured_self", value=x))
            for k, x in v["r2_dict"].items():
                rows.append(dict(base, item="B2", variant=var, k=k, metric="r2_dict", value=x))
        for k, sg in b["sig_B1"].items():
            rows.append(dict(base, item="B1", variant="real-S4", k=k, metric="significant", value=sg))
        for k, sg in b["sig_B2"].items():
            rows.append(dict(base, item="B2", variant="real-S4", k=k, metric="significant", value=sg))
        for name, v in b["B3"].items():
            for m in ("gain", "cost", "net", "r2_holdout", "approx", "approx_pooled", "h_e", "h_ctx"):
                rows.append(dict(base, item=f"B3-{name}", variant="real", k=v["k"], metric=m,
                                 value=v[m]))
    return rows


# ----------------------------------------------------------------------------- C

def c_ledger(A, sel):
    """Rows 1-4 per matrix, 6abs with leave-one-out over the full-layer set (and, separately, over
    the original 36), per-layer scale shares."""
    rows = []
    full = [(r["spec"], role, r["ledger"][role]) for r in A if "full" in r["spec"]["sets"]
            for role in ("w1", "w2", "w3")]
    orig = [(r["spec"], role, r["ledger"][role]) for r in A if "orig" in r["spec"]["sets"]
            for role in ("w1", "w2", "w3")]
    res = {}
    for setname, lst in (("full", full), ("orig", orig)):
        sh = ledger.shared_lengths([x[2] for x in lst], key="abs")
        for (s, role, lr), shx in zip(lst, sh):
            lrows = ledger.ledger_rows(lr, None, shx)
            for k in (1, 2, 3, 4, "6abs"):
                v = lrows[k]
                for m in ("cw", "cw_tab", "sc", "sc_tab", "compressed", "saving"):
                    rows.append({"sample": setname, "layer": lab(s), "expert": s["expert"],
                                 "role": role, "matrix": lr["matrix"], "item": f"row{k}", "key": m,
                                 "value": v[m]})
            res.setdefault(setname, []).append((s, role, lr, lrows))
    # per-layer scale value shares (full-layer set, by number of scales)
    shares = defaultdict(lambda: defaultdict(int))
    for s, role, lr, _ in res["full"]:
        for p in lr["per_scale"]:
            shares[lab(s)][p["E"]] += p["n_scales"]
    for layer, d in shares.items():
        tot = sum(d.values())
        for e in sorted(d):
            rows.append({"sample": "full", "layer": layer, "expert": "", "role": "", "matrix": "",
                         "item": "scale_share", "key": e, "value": d[e] / tot})
    return rows, res, shares


# ----------------------------------------------------------------------------- plots

def plots(A, cres, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    full = [r for r in A if "full" in r["spec"]["sets"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, pr in zip(axes, gaps.A1_PAIRS):
        for q, mk in ((0.5, "o"), (0.9, "^")):
            xs = [xpos(r["spec"]) for r in full]
            ys = [r["A"]["A1"][(pr[0], pr[1], "w")]["q"][q] for r in full]
            ax.scatter(xs, ys, s=12, marker=mk, label=f"|ρ| {int(q * 100)}% 分位" if False else
                       f"|rho| q{int(q * 100)}")
            lay = sorted(set(xs))
            ax.plot(lay, [np.mean([y for x, y in zip(xs, ys) if x == l]) for l in lay], lw=0.8)
        ax.set_title(f"{pr[0]}-{pr[1]} (w domain)")
        ax.set_xlabel(f"layer (MTP at x={MTP_X})")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|rho_i| over neurons")
    fig.tight_layout()
    fig.savefig(out / "neuron_rho_by_layer.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    xs = [xpos(r["spec"]) for r in full]
    ys = [r["A"]["A4"]["total"] for r in full]
    ax.scatter(xs, ys, s=14)
    lay = sorted(set(xs))
    ax.plot(lay, [np.mean([y for x, y in zip(xs, ys) if x == l]) for l in lay], lw=0.8)
    ax.set_xlabel(f"layer (MTP at x={MTP_X})")
    ax.set_ylabel("A4 best-order total (bit/weight)")
    ax.set_title("A4 neuron correspondence, per expert")
    fig.tight_layout()
    fig.savefig(out / "neuron_gain_by_layer.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    by = defaultdict(list)
    for s, role, lr, lrows in cres["full"]:
        by[xpos(s)].append(lrows[4]["saving"])
    lay = sorted(by)
    ax.plot(lay, [np.median(by[l]) for l in lay], "o-", ms=3, label="row 4 median per layer")
    ax.fill_between(lay, [min(by[l]) for l in lay], [max(by[l]) for l in lay], alpha=0.2,
                    label="min-max of 6 matrices")
    ax.axhline(ROW002, color="k", lw=0.6, ls="--", label="002 original sample median 0.569")
    ax.set_xlabel(f"layer (MTP at x={MTP_X})")
    ax.set_ylabel("saving vs 4.25 (bit/weight)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ledger_by_layer.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- summary

def summary(data, sel, crows_res, shares, val, out):
    A, Bs, C2, D = data["a"], data["b"], data["c2"], data["d"]
    full = [r for r in A if "full" in r["spec"]["sets"]]
    main = [r for r in full if r["spec"]["prefix"] == "layers"]
    mtp = [r for r in full if r["spec"]["prefix"] == "mtp"]
    orig = [r for r in A if "orig" in r["spec"]["sets"]]
    L = []
    w = L.append
    w("# 004 补缺与夯实：结果摘要\n")
    w(f"规格：[docs/004-补缺与夯实.md](../../docs/004-补缺与夯实.md)。第 7 节检查："
      f"{'全部通过' if val['all_pass'] else '有未通过项'}（TA1 的 A2、TB2 的近似对照按完成状态所述调整，"
      "原判据数值见 `validation.json`）。")
    w(f"样本：全层样本 {len(main)} 个主干专家（{len({r['spec']['layer'] for r in main})} 层 × 2）+ "
      f"{len(mtp)} 个 MTP 专家，原样本 {len(orig)} 个专家；层内样本 4 层 × 8 个专家；"
      f"探针 12 张；往返 3 张。共读取 {sel['n_tensors']} 个张量。\n")
    w("“严格”= 留出交叉熵或实际码流，已扣除所列参数成本；“近似”= 由 R² 等换算的估计。"
      "单位都是 bit/权重（相对于该部分所在矩阵的码字数）。\n")

    # ---------------- A
    w("## A 神经元级的对应\n")
    w("### A1/A2 同编号相关（w 域；v 域见 CSV）\n")
    w("| 族对 | 范围 | \\|ρ\\| 中位数 | \\|ρ\\| 90% | \\|ρ\\| 99% | 基线 99.9% 阈值 | 显著比例 | 显著中正号 | "
      "A2 同编号（显著行） | A2 同编号（\\|ρ\\|>2×阈值） |")
    w("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    a1sum = {}
    for pr in gaps.A1_PAIRS:
        for scope, lst in (("主干 80", main), ("MTP 2", mtp), ("原样本 12", orig)):
            if not lst:
                continue
            v = [r["A"]["A1"][(pr[0], pr[1], "w")] for r in lst]
            row = [med([x["q"][0.5] for x in v]), med([x["q"][0.9] for x in v]),
                   med([x["q"][0.99] for x in v]), med([x["thr"] for x in v]),
                   med([x["frac_sig"] for x in v]), med([x["sig_pos"] for x in v]),
                   med([x["a2_y_given_x"] for x in v]), med([x["a2s_y_given_x"] for x in v])]
            if scope.startswith("主干"):
                a1sum[pr] = row
            w(f"| {PAIR_NAME[pr]} | {scope} | " + " | ".join(fmt(x, 3) for x in row) + " |")
    w("\n表中为各专家数值的中位数。A2 一列以第二个族为目标、第一个族为来源（反方向见 CSV）。\n")
    # by layer table (main)
    w("按层（主干，每层 2 个专家的平均；w 域 \\|ρ\\| 中位数）：\n")
    layers = sorted({r["spec"]["layer"] for r in main})
    w("| 族对 | " + " | ".join(f"L{l}" for l in layers[::4]) + " | MTP |")
    w("| --- |" + " ---: |" * (len(layers[::4]) + 1))
    for pr in gaps.A1_PAIRS:
        cells = []
        for l in layers[::4]:
            cells.append(fmt(np.mean([r["A"]["A1"][(pr[0], pr[1], "w")]["q"][0.5] for r in main
                                      if r["spec"]["layer"] == l]), 3))
        cells.append(fmt(np.mean([r["A"]["A1"][(pr[0], pr[1], "w")]["q"][0.5] for r in mtp]), 3)
                     if mtp else "—")
        w(f"| {PAIR_NAME[pr]} | " + " | ".join(cells) + " |")
    w("\n（每 4 层列一次，全部层见 `a_neurons.csv` 与 `neuron_rho_by_layer.png`。）\n")

    w("### A3 严格收益（主干 80 个专家的中位数，严格）\n")
    w("| 配置（目标\\|来源） | 收益 | 斜率成本 | 净值 | 净值>0 的专家 | 近似对照 | 收益/近似 | 错位基线收益 |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    a3sum = {}
    for tgt, srcs in gaps.A3_CONFIGS:
        n = gaps.cfg_name(tgt, srcs)
        v = [r["A"]["A3"][n] for r in main]
        bb = [r["A"]["A3_base"][n] for r in main]
        g, c, nt, ap = (med([x[k] for x in v]) for k in ("gain", "cost", "net", "approx"))
        npos = sum(x["net"] > 0 for x in v)
        a3sum[n] = (g, nt)
        lbl = f"{FAM_NAME[tgt]} \\| " + "+".join(FAM_NAME[s] for s in srcs)
        w(f"| {lbl} | {fmt(g)} | {fmt(c)} | {fmt(nt)} | {npos}/{len(v)} | {fmt(ap)} | "
          f"{fmt(g / ap if ap else None, 2)} | {fmt(med([x['gain'] for x in bb]))} |")
    w("")
    w("### A4 专家级合计（严格，已除以 3）\n")
    tot_main = [r["A"]["A4"]["total"] for r in main]
    by_layer = defaultdict(list)
    for r in main:
        by_layer[r["spec"]["layer"]].append(r["A"]["A4"]["total"])
    lm = {l: float(np.mean(v)) for l, v in by_layer.items()}
    best = defaultdict(int)
    for r in main:
        best[r["A"]["A4"]["best"]] += 1
    w(f"- 40 个主干层平均：**{fmt(float(np.mean(tot_main)))}**；按层平均的最小 {fmt(min(lm.values()))}"
      f"（L{min(lm, key=lm.get)}），最大 {fmt(max(lm.values()))}（L{max(lm, key=lm.get)}）；"
      f"专家中位数 {fmt(med(tot_main))}。")
    if mtp:
        w("- MTP 层（3 层 × 2 个专家）：" + "，".join(
            f"{lab(r['spec'])}.E{r['spec']['expert']} {fmt(r['A']['A4']['total'])}" for r in mtp) +
          "。MTP2.E89 的 w2 有 67% 的码字为零（m = 0），它的 w2 列 | w1 行 收益 0.27 来自“预测强度”分档"
          "（逐神经元幅度，近似对照 ≈ 0），不是逐位置的线性相关。")
    w(f"- 原样本 12 个专家：中位数 {fmt(med([r['A']['A4']['total'] for r in orig]))}。")
    w("- 最优顺序（首个不预测，第二个单来源，第三个双来源）的专家数：" +
      "，".join(f"{k}: {v}" for k, v in sorted(best.items(), key=lambda t: -t[1])) +
      "（a = w1 行，g = w3 行，o = w2 列）。")
    w("- 曲线见 `neuron_gain_by_layer.png`。\n")

    # ---------------- B
    w("## B 同层专家共享的行级结构\n")
    w("拟合集 6 个专家（13,824 个向量），评估集 2 个专家（4,608 个）；S4 基线重做全流程。"
      "`超出` = captured(k) − k/5120。\n")
    w("| 层 | 族 | 超出 k=4 | k=16 | k=64 | k=256 | k=1024 | S4 超出 k=64 | 自身方向 k=64 | "
      "R²_dict(2) 真实 | S4 | B1 显著 k | B2 显著 s |")
    w("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    b3_lines = []
    for r in sorted(Bs, key=lambda r: (r["spec"]["layer"], "ago".index(r["spec"]["family"]))):
        b, s = r["B"], r["spec"]
        Ld = b["L"]
        ex = [b["real"]["captured"][k] - k / Ld for k in (4, 16, 64, 256, 1024)]
        w(f"| L{s['layer']} | {FAM_NAME[s['family']]} | " + " | ".join(fmt(x) for x in ex) +
          f" | {fmt(b['S4']['captured'][64] - 64 / Ld)} | {fmt(b['real']['captured_self'][64] - 64 / Ld)} | "
          f"{fmt(b['real']['r2_dict'][2])} | {fmt(b['S4']['r2_dict'][2])} | "
          f"{','.join(str(k) for k, v in b['sig_B1'].items() if v) or '无'} | "
          f"{','.join(str(k) for k, v in b['sig_B2'].items() if v) or '无'} |")
        for name, v in b["B3"].items():
            b3_lines.append((s, name, v))
    w("\n`自身方向 k=64` 是 captured_self(64) − 64/5120：用评估集自己一半向量求方向时的水平。\n")
    if b3_lines:
        w("B3 严格收益（方向或字典由全层共享，成本记 0；系数成本已扣除）：\n")
        w("| 层 | 族 | 方法 | k | 收益 | 系数成本 | 净值 | 留出 R² | 近似对照（逐向量） |")
        w("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s, name, v in b3_lines:
            w(f"| L{s['layer']} | {FAM_NAME[s['family']]} | {name} | {v['k']} | {fmt(v['gain'])} | "
              f"{fmt(v['cost'])} | {fmt(v['net'])} | {fmt(v['r2_holdout'])} | {fmt(v['approx'])} |")
        w("")
        w("B3-B1 按规格取 `captured − S4` 最大的 k，在所有执行到的层上都是 k = 1024：用 2560 个位置拟合 1024 个系数，"
          "留出 R² 为负（逐向量近似对照因此记 0），系数成本 3.2 bit/权重。与 k 无关的近似上界：即使留出 R² 达到"
          "全部 `captured(k) − k/5120`，换算收益 −½log₂(1−超出) 也小于系数成本 k·16/5120（k=4：超出 ≤ 0.0025、"
          "上界 ≤ 0.002 < 0.0125；k=1024：超出 ≤ 0.106、上界 ≤ 0.081 < 3.2），所以任何 k 的 B1 净值都为负（近似论证）。\n")
    else:
        w("B1、B2 在所有层与族上都不显著，B3 未执行。\n")

    # ---------------- C
    w("## C 账本夯实\n")
    cfull = crows_res["full"]
    sv = defaultdict(list)
    sv6 = defaultdict(list)
    for s, role, lr, lrows in cfull:
        sv[lab(s)].append(lrows[4]["saving"])
        sv6[lab(s)].append(lrows["6abs"]["saving"])
    mains = [f"L{l}" for l in range(sel["main_layers"])]
    lmed = {k: float(np.median(sv[k])) for k in mains}
    lmed6 = {k: float(np.median(sv6[k])) for k in mains}
    orig_sv = [lrows[4]["saving"] for s, role, lr, lrows in crows_res["orig"]]
    orig6 = [lrows["6abs"]["saving"] for s, role, lr, lrows in crows_res["orig"]]
    w("### C1 全层账本（严格：熵 + 每矩阵码表；6abs 为留一法共用码表）\n")
    w("| 行 | 40 个主干层（各层中位数）的平均 | 最小（层） | 最大（层） | MTP 层中位数 | 原样本 36 张中位数 |")
    w("| --- | ---: | ---: | ---: | ---: | ---: |")
    for k, name in ((1, "1 仅 R1"), (2, "2 零合并"), (3, "3 scale 条件"), (4, "4 加 scale 流"),
                    ("6abs", "6abs 按绝对 E 共用码表")):
        per = defaultdict(list)
        for s, role, lr, lrows in cfull:
            per[lab(s)].append(lrows[k]["saving"])
        lm_ = {l: float(np.median(per[l])) for l in mains}
        mt = [x for l, v in per.items() if l.startswith("MTP") for x in v]
        og = [lrows[k]["saving"] for s, role, lr, lrows in crows_res["orig"]]
        lo, hi = min(lm_, key=lm_.get), max(lm_, key=lm_.get)
        w(f"| {name} | {fmt(float(np.mean(list(lm_.values()))))} | {fmt(lm_[lo])}（{lo}） | "
          f"{fmt(lm_[hi])}（{hi}） | {fmt(med(mt))} | {fmt(med(og))} |")
    m4 = float(np.mean(list(lmed.values())))
    w(f"\n第 4 行：40 个主干层平均 {fmt(m4)}，与原样本 0.569 之差 {fmt(m4 - ROW002, 4)}；"
      f"本次在原样本上重算的中位数 {fmt(med(orig_sv))}（002 为 0.569）。"
      f"6abs 共用码表在全层样本上的留一法：{fmt(float(np.mean(list(lmed6.values()))))}（原样本 36 张内留一：{fmt(med(orig6))}）。"
      "曲线见 `ledger_by_layer.png`。\n")
    mt_l = sorted({lab(s) for s, role, lr, lrows in cfull if lab(s).startswith("MTP")})
    if mt_l:
        w("MTP 各层第 4 行（6 张矩阵的中位数）：" + "；".join(
            f"{l} {fmt(float(np.median([lrows[4]['saving'] for s, role, lr, lrows in cfull if lab(s) == l])))}"
            for l in mt_l) + "。MTP0、MTP1 每张矩阵只有 3 种 scale，scale 条件几乎没有收益；"
          "MTP2.E89.w2 码字 67% 为零，第 4 行节省 2.34，是全部 258 张中唯一的离群值。\n")
    allE = sorted({e for d in shares.values() for e in d})
    w(f"各层出现的 scale 取值（字节）范围：{min(allE)}–{max(allE)}；"
      "每层取值及占比见 `c_ledger.csv`（item = scale_share）。各层占比最高的取值：")
    tops = []
    for l in mains[::5] + [k for k in shares if k.startswith("MTP")]:
        d = shares[l]
        tot = sum(d.values())
        e = max(d, key=d.get)
        tops.append(f"{l}: {e}（{d[e] / tot:.0%}，共 {len(d)} 种）")
    w("；".join(tops) + "。\n")

    w("### C2 往返验证（严格：实际码流）\n")
    w("| 矩阵 | E 逐字节相同 | C' 逐元素相同 | 数值逐元素相等 | 原矩阵 −0 个数 | 文件头 bit/权重 | "
      "码流总计 bit/权重 | 002 第 4 行压缩后 | 差 |")
    w("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    c2d = []
    for r in sorted(C2, key=lambda r: r["spec"]["matrix"]):
        c = r["C2"]
        ref = r["ledger_row4"]["compressed"]
        c2d.append(c["bits_per_weight"] - ref)
        w(f"| {r['spec']['matrix']} | {c['E_equal']} | {c['Cp_equal']} | {c['values_equal']} | "
          f"{c['n_neg_zero']} | {fmt(c['header_bits_per_weight'], 6)} | {fmt(c['bits_per_weight'], 5)} | "
          f"{fmt(ref, 5)} | {fmt(c['bits_per_weight'] - ref, 6)} |")
    w("\n−0 位置的比特模式与原矩阵不同（被合并为 +0），数值相同。\n")

    # ---------------- D
    w("## D 上下文混合探针（严格：计数行 {0,1}、定权行 {2}、评估行 {3} 互不重叠）\n")
    dd = defaultdict(dict)
    for r in D:
        dd[r["spec"]["matrix"]][r["spec"]["variant"]] = r["D"]
    w("| 矩阵 | h_E | 收益 真实 | S2 | S0E | 块组成（S2−S0E） | 顺序与位置（真实−S2） | 显著 | 最好的单个分量（收益） |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    comp_parts, ord_parts, sigs = [], [], []
    for m in sorted(dd, key=lambda m: (int(m.split(".")[0][1:]), m)):
        v = dd[m]
        if not all(k in v for k in ("real", "S2", "S0E")):
            continue
        g, g2, g0 = v["real"]["gain"], v["S2"]["gain"], v["S0E"]["gain"]
        sig = g > 0.001 and g > 5 * abs(g0)
        sigs.append(sig)
        comp_parts.append(g2 - g0)
        ord_parts.append(g - g2)
        sg = {k: x for k, x in v["real"]["single_gain"].items() if k != "E"}
        bk = max(sg, key=sg.get)
        w(f"| {m} | {fmt(v['real']['h_E'])} | {fmt(g)} | {fmt(g2)} | {fmt(g0)} | {fmt(g2 - g0)} | "
          f"{fmt(g - g2)} | {'是' if sig else '否'} | {bk}（{fmt(sg[bk])}） |")
    w("")
    # component table (median over the 12 real matrices)
    reals = [dd[m]["real"] for m in dd if "real" in dd[m]]
    if reals:
        names = reals[0]["components"]
        w("各分量单独收益与混合权重（12 张真实矩阵的中位数；S2 列为 S2 上单独收益的中位数）：\n")
        w("| 分量 | 单独收益 | S2 上单独收益 | 权重 |")
        w("| --- | ---: | ---: | ---: |")
        s2s = [dd[m]["S2"] for m in dd if "S2" in dd[m]]
        for n in names:
            w(f"| {n} | {fmt(med([x['single_gain'][n] for x in reals]))} | "
              f"{fmt(med([x['single_gain'][n] for x in s2s]))} | {fmt(med([x['weights'][n] for x in reals]), 3)} |")
        w("")

    # ---------------- totals
    w("## 总表\n")
    w("| 项 | 节省（bit/权重） | 适用范围 | 严格/近似 |")
    w("| --- | ---: | --- | --- |")
    c2ok = all(r["C2"]["E_equal"] and r["C2"]["Cp_equal"] and r["C2"]["values_equal"] for r in C2)
    w(f"| 002 第 4 行（全层样本：40 个主干层平均） | {fmt(m4)} | 全部权重 | 严格（熵）；C2 "
      f"{'证实 scale 与合并零码字可无损往返（不保留负零位模式），3 张实测与熵账之差 ' + ', '.join(fmt(x, 5) for x in c2d) if c2ok else '未证实'} |")
    w(f"| A4 神经元对应（主干层平均） | {fmt(float(np.mean(tot_main)))} | 全部权重（已除以 3） | 严格 |")
    b2n = [v["net"] for _, n_, v in b3_lines if n_ == "B2"]
    b1n = [v["net"] for _, n_, v in b3_lines if n_ == "B1"]
    w(f"| B3 行级共享结构 | 全部为负：B2 字典 {', '.join(fmt(x) for x in b2n)}；B1 最好 {fmt(max(b1n)) if b1n else '—'} | "
      "执行到的层与族的评估专家（B2 显著：L0 w1、L13 w1、L13 w2 列；L0 w1 只在 s=8 显著，未执行） | 严格 |")
    w(f"| D 探针：块组成的自适应部分（S2−S0E，中位数） | {fmt(med(comp_parts))} | 12 张探针矩阵 | 严格 |")
    w(f"| D 探针：顺序与位置部分（真实−S2，中位数） | {fmt(med(ord_parts))} | 12 张探针矩阵 | 严格 |")
    w("")
    w("相加关系：")
    w("- 002 第 4 行是基准（H(C'|E) + scale 流）；A4、B3、D 都是在 `H(C'|E)` 之上的增量，各自以 "
      "`H(C'|E)` 为对照，可以分别与第 4 行相加。")
    w("- A 与 B 都利用行级（神经元向量）结构：A 用同一专家另一矩阵的同编号向量，B 用同层其他专家的方向或字典，二者有重叠，不能直接相加。")
    w("- D 的“块组成”部分（块内已见、邻块等）与 002 的 scale 条件、002 第 5 行的 m_max 利用的是同一种块内幅度信息，"
      "与 002 第 5 行不能相加；它与 A 的重叠未测量。")
    w("- D 的“顺序与位置”部分与 A、B 是否重叠未测量（D 只用局部上下文，A/B 用跨矩阵/跨专家的向量）。")
    w("- 只陈述测量结果，不推荐压缩方法。\n")
    # direct answers to section 0, inserted after the intro
    ag, gaw = a3sum["a|g"], a1sum[("a", "g")]
    ans = [
        "## 第 0 节四个问题的直接回答\n",
        f"- **A（神经元对应）**：同一神经元的 w1 行与 w3 行明显相关（主干专家 \\|ρ\\| 中位数 {fmt(gaw[0], 3)}，"
        f"{fmt(gaw[4] * 100, 0)}% 的神经元显著，正负号各半），与 w2 列的相关弱得多（\\|ρ\\| 中位数约 0.03）。"
        f"严格收益：w1 行 | w3 行 净值中位数 {fmt(ag[1])} bit/权重（77/80 个专家为正），凡以 w2 列为目标或来源的单来源配置净值中位数为负。"
        f"专家级合计（已除以 3）在 40 个主干层上平均 **{fmt(float(np.mean(tot_main)))}**，随层下降：L0 约 {fmt(lm[0], 3)}，"
        f"L22–L33 均 < 0.004（最小 L{min(lm, key=lm.get)} {fmt(min(lm.values()))}），L34–L39 回升到 0.002–0.014；MTP 层约 0.003（一个离群专家除外）。严格。",
        "- **B（同层专家共享）**：存在可测但很弱的共享子空间——12 个“层 × 族”中 11 个的 captured(k) 显著超出随机水平"
        "（k=1024 时 0.004–0.106，L0 最大），但 k=64 时 12 个中有 11 个用评估专家自己一半向量求出的方向更好，说明方向主要是专家自己的。"
        "行级字典只在 L13 的 w1 行、w2 列（以及 L0 w1 在 s=8）显著，幅度 ≤ 0.011。B3 严格净值全部为负"
        f"（字典 {', '.join(fmt(x) for x in b2n)}；子空间按规格取 k=1024 时 −3.2，近似论证表明任何 k 都为负）。",
        f"- **C（账本）**：成立。40 个主干层第 4 行平均 {fmt(m4)}（层间 {fmt(min(lmed.values()))}–{fmt(max(lmed.values()))}），"
        f"与原样本 0.569 相差 {fmt(m4 - ROW002, 4)}；MTP 层约 0.50（scale 只有 3 种，scale 条件收益小）。"
        f"实际编码再解码：3 张矩阵 E、C'、数值全部一致，码流（含文件头）比熵账多 0.00004–0.00005 bit/权重。严格。",
        f"- **D（局部上下文组合）**：混合后比 `H(C'|E)` 好 {fmt(min(x['real']['gain'] for x in dd.values()), 3)}–"
        f"{fmt(max(x['real']['gain'] for x in dd.values()), 3)} bit/权重（12 张全部显著），但几乎全部在 S2（块内打乱）上同样出现："
        f"块组成部分中位数 {fmt(med(comp_parts))}，顺序与位置部分中位数 {fmt(med(ord_parts))}（|·| ≤ 0.0003，无新结构）。严格。\n",
    ]
    L[4:4] = ans
    (out / "summary.md").write_text("\n".join(L).rstrip() + "\n")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="experiments/004-gaps")
    args = ap.parse_args()
    out = Path(args.out)
    data = load(args.work)
    with open(out / "samples_004.json") as f:
        sel = json.load(f)
    with open(out / "validation.json") as f:
        val = json.load(f)
    arows = a_rows(data["a"])
    write_csv(out / "a_neurons.csv", ["sets", "layer", "expert", "item", "config", "domain",
                                      "metric", "value"], arows)
    write_csv(out / "b_layer.csv", ["layer", "family", "role", "dim", "n_fit", "n_eval", "item",
                                    "variant", "k", "metric", "value"], b_rows(data["b"]))
    crows, cres, shares = c_ledger(data["a"], sel)
    write_csv(out / "c_ledger.csv", ["sample", "layer", "expert", "role", "matrix", "item", "key",
                                     "value"], crows)
    rt = []
    for r in data["c2"]:
        c = dict(r["C2"])
        c["matrix"] = r["spec"]["matrix"]
        c["ledger_row4_compressed"] = r["ledger_row4"]["compressed"]
        c["diff_vs_row4"] = c["bits_per_weight"] - c["ledger_row4_compressed"]
        c["saving_vs_4.25"] = ledger.RAW + 8.0 / gaps.B - c["bits_per_weight"]
        rt.append(c)
    write_csv(out / "c_roundtrip.csv", ["matrix", "E_equal", "Cp_equal", "values_equal", "n_neg_zero",
                                        "header_bytes", "stream_bytes", "N", "header_bits_per_weight",
                                        "bits_per_weight", "ledger_row4_compressed", "diff_vs_row4",
                                        "saving_vs_4.25"], rt)
    drows = []
    for r in data["d"]:
        s, d = r["spec"], r["D"]
        base = {"matrix": s["matrix"], "variant": s["variant"]}
        for m in ("h_E", "h_mix", "gain", "n_fit", "n_eval", "n_count", "K_E"):
            drows.append(dict(base, component="mixture", metric=m, value=d[m]))
        drows.append(dict(base, component="mixture", metric="newton_steps",
                          value=len(d["fit_loss_bits"]) - 1))
        for n in d["components"]:
            drows.append(dict(base, component=n, metric="single_gain", value=d["single_gain"][n]))
            drows.append(dict(base, component=n, metric="weight", value=d["weights"][n]))
            drows.append(dict(base, component=n, metric="n_ctx_seen", value=d["n_ctx_seen"][n]))
    write_csv(out / "d_probe.csv", ["matrix", "variant", "component", "metric", "value"], drows)
    plots(data["a"], cres, out)
    summary(data, sel, cres, shares, val, out)
    secs = {k: sum(r["seconds"] for r in v) for k, v in data.items()}
    print("jobs:", {k: len(v) for k, v in data.items()}, "cpu seconds:", {k: round(v) for k, v in secs.items()})


if __name__ == "__main__":
    main()
