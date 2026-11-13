"""Spec 002: run the ledger on the 001 sample (36 matrices) and write experiments/002-ledger/.

Refuses to start unless the section-4 validation.json reports all_pass. Per-matrix results are
small count tables and scalars held in memory; nothing intermediate is written.

Usage: python -m diagnostics.ledger_run --snapshot $SNAPSHOT [--jobs 12]
"""
import argparse
import csv
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from . import ledger  # noqa: E402
from .measure import unpack_codes  # noqa: E402
from .st import index_snapshot, read_bytes  # noqa: E402

B = 32
SEED_BASE = 20261112
RAW = 4.0 + 8.0 / B
CP_LABELS = ["+0", "+0.5", "+1", "+1.5", "+2", "+3", "+4", "+6",
             "-0.5", "-1", "-1.5", "-2", "-3", "-4", "-6"]
META = ("matrix", "layer", "expert", "role")
LEDGER_FIELDS = list(META) + ["row", "row_name", "codeword_bits", "codeword_table",
                              "scale_bits_per_scale", "scale_table_per_scale", "label_bits",
                              "label_table", "compressed", "saving", "saving_pct_of_4.25", "note"]


def work_one(snapshot, sample, idx):
    t0 = time.time()
    index = index_snapshot(snapshot)
    C = unpack_codes(read_bytes(index[sample["weight"]]))
    E = read_bytes(index[sample["scale"]])
    assert list(C.shape) == sample["codeword_shape"]
    assert E.shape == (C.shape[0], C.shape[1] // B)
    rng = np.random.default_rng(SEED_BASE * 1000 + 2 * 100 + idx)
    r = ledger.ledger_matrix(C, E, B, rng)
    r["meta"] = {k: sample[k] for k in META}
    r["seconds"] = time.time() - t0
    return r


def fnum(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.10g}"
    return v


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fnum(r.get(k, "")) for k in fields})


# ----------------------------------------------------------------------------- tables

def build(results):
    roles = [r["meta"]["role"] for r in results]
    sh_b = ledger.shared_lengths(results)
    sh_c = ledger.shared_lengths(results, groups=roles)
    sh_ba = ledger.shared_lengths(results, key="abs")
    sh_ca = ledger.shared_lengths(results, groups=roles, key="abs")
    ledgers, cond, hist, tables, scalars = [], [], [], [], []
    for r, b, c, ba, ca in zip(results, sh_b, sh_c, sh_ba, sh_ca):
        meta = r["meta"]
        B_, N, nS, K = r["B"], r["N"], r["n_scales"], r["K_E"]
        rows = ledger.ledger_rows(r, b, ba)
        sop = ledger.sum_of_parts(r)
        for k, v in rows.items():
            d = dict(meta, row=k, row_name=ledger.ROW_NAMES[k])
            if v is None:
                d["note"] = "shared table unavailable"
            else:
                d.update(codeword_bits=v["cw"], codeword_table=v["cw_tab"],
                         scale_bits_per_scale=v["sc"], scale_table_per_scale=v["sc_tab"],
                         label_bits=v["lab"], label_table=v["lab_tab"],
                         compressed=v["compressed"], saving=v["saving"],
                         saving_pct_of_4_25=100 * v["saving"] / RAW)
                d["saving_pct_of_4.25"] = d.pop("saving_pct_of_4_25")
            ledgers.append(d)
        ledgers.append(dict(meta, row="sum001", row_name="001 分项相加（无码表成本）",
                            saving=sop, **{"saving_pct_of_4.25": 100 * sop / RAW},
                            note=f"row4_saving - sum = {rows[4]['saving'] - sop:.6g}"))

        e = r["ent"]
        q = {**{k.replace("Cp", "C'"): v for k, v in e.items()},
             "P(m=0)": r["P_m0"], "H(s|m=0)": r["H_s_m0"],
             "I(C;E)": r["I_C_E"], "I(C';E)": r["I_Cp_E"],
             "H(m_max)": r["H_mmax"], "H(m_max|E)": r["m2"]["H_mmax_E"],
             "H0(E) [bit/scale]": r["H0_E"], "K_E": K, "K_(E,m_max)": r["K_E_mmax"],
             "mode_E": r["mode_E"], "scales_outside_mode±3": r["n_escape"],
             "M2 inc_net": r["m2"]["inc_net"]}
        for i, s in enumerate(r["s0"]):
            q[f"M2 inc_net S0 rep{i + 1}"] = s["inc_net"]
            q[f"I(C';E) S0 rep{i + 1}"] = s["I(Cp;E)"]
        for k, v in q.items():
            cond.append(dict(meta, E="", quantity=k, value=v))
        for p in r["per_scale"]:
            for k in ("P(m=0|E)", "H(s|m=0,E)", "n_weights", "n_scales"):
                cond.append(dict(meta, E=p["E"], quantity=k, value=p[k],
                                 note="" if np.isfinite(p[k]) else "no m=0 codeword under this E"))
            h = dict(meta, E=p["E"], exponent=p["E"] - 127,
                     E_rel=p["E"] - r["mode_E"], n_scales=p["n_scales"], n_weights=p["n_weights"],
                     **{"P(m=0|E)": p["P(m=0|E)"], "H(s|m=0,E)": p["H(s|m=0,E)"]})
            h.update({f"C'={lab}": int(x) for lab, x in zip(CP_LABELS, p["hist_Cp"])})
            hist.append(h)

        a_cw = e["H(Cp|E)"]
        a_tab = ledger.table_cost(K, 15, N)
        a_sc, a_sct = r["H0_E"], K * ledger.TABLE_BITS / nS
        tables.append(dict(meta, stream="codeword C'|E", scheme="a", length=a_cw, table_cost=a_tab,
                           total_bit_per_weight=a_cw + a_tab, excess_vs_a=0.0))
        tables.append(dict(meta, stream="scale E", scheme="a", length=a_sc, table_cost=a_sct,
                           total_bit_per_weight=(a_sc + a_sct) / B_, excess_vs_a=0.0))
        for name, sh in (("b", b), ("c", c), ("b_absE (supplement)", ba),
                         ("c_absE (supplement)", ca)):
            if sh is None:
                tables.append(dict(meta, stream="codeword C'|E", scheme=name,
                                   note="no other matrix in group"))
                continue
            tables.append(dict(meta, stream="codeword C'|E", scheme=name, length=sh["cw_bits"],
                               table_cost=0.0, total_bit_per_weight=sh["cw_bits"],
                               excess_vs_a=sh["cw_bits"] - (a_cw + a_tab),
                               train_matrices=sh["train_matrices"]))
        if b is not None:
            tables.append(dict(meta, stream="scale E", scheme="b", length=b["scale_bits"],
                               table_cost=0.0, total_bit_per_weight=b["scale_bits"] / B_,
                               excess_vs_a=(b["scale_bits"] - a_sc - a_sct) / B_,
                               train_matrices=b["train_matrices"], escapes=b["scale_escapes"],
                               note="length includes 8 bit/matrix for the mode and 8 raw bits "
                                    "per escaped scale"))
        if ba is not None:
            tables.append(dict(meta, stream="scale E", scheme="b_absE (supplement)",
                               length=ba["scale_bits"], table_cost=0.0,
                               total_bit_per_weight=ba["scale_bits"] / B_,
                               excess_vs_a=(ba["scale_bits"] - a_sc - a_sct) / B_,
                               train_matrices=ba["train_matrices"],
                               note="256 scale-byte symbols, no mode or escape needed"))

        sc = {"row": rows, "sop": sop, "q": q, "b": b, "c": c, "ba": ba, "ca": ca,
              "a_cw_total": a_cw + a_tab, "a_sc_total": (a_sc + a_sct) / B_}
        scalars.append(sc)
    return ledgers, cond, hist, tables, scalars


# ----------------------------------------------------------------------------- summary

def stats(x):
    x = np.array([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if x.size == 0:
        return None
    return float(np.median(x)), float(x.min()), float(x.max())


def f4(v, nd=4):
    return "" if v is None else f"{v:.{nd}f}"


def line(label, xs, pct=True, nd=4):
    s = stats(xs)
    if s is None:
        return f"| {label} | — | — | — |" + (" — |" if pct else "")
    out = f"| {label} | {f4(s[0], nd)} | {f4(s[1], nd)} | {f4(s[2], nd)} |"
    if pct:
        out += f" {100 * s[0] / RAW:.2f}% |"
    return out


def group_table(results, items):
    layers = sorted({r["meta"]["layer"] for r in results})
    roles = ["w1", "w2", "w3"]
    head = "| 项 | " + " | ".join(f"L{l}" for l in layers) + " | " + " | ".join(roles) + " |"
    sep = "| --- |" + " ---: |" * (len(layers) + len(roles))
    out = [head, sep]
    for label, xs in items:
        cells = []
        for l in layers:
            v = [x for r, x in zip(results, xs) if r["meta"]["layer"] == l]
            cells.append(f4(stats(v)[0]) if stats(v) else "—")
        for ro in roles:
            v = [x for r, x in zip(results, xs) if r["meta"]["role"] == ro]
            cells.append(f4(stats(v)[0]) if stats(v) else "—")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return out


def summary(results, scalars, out, meta):
    L = []
    w = L.append
    rw = lambda k, f: [s["row"][k][f] if s["row"][k] else None for s in scalars]  # noqa: E731
    q = lambda k: [s["q"][k] for s in scalars]  # noqa: E731
    w("# 002 账本核对：结果摘要\n")
    w(f"样本：001 的 {len(results)} 张矩阵（层 {meta['layers']}，每层 3 个专家 × w1/w2/w3），"
      f"每张 N = {results[0]['N']:,} 码字、{results[0]['n_scales']:,} 个 scale。全部量为 plug-in 估计，"
      "单位 bit/权重（另注明者除外），百分比按原始成本 4.25 bit/权重。"
      "码表成本：每个上下文 (字母表−1) 个概率 × 12 bit（第 3 节 M4 方案 a 的写法，推广到第 1、2、5 行）。\n")
    w("## M5 总账\n")
    w("| 行 | 组合 | 压缩后 中位数 | 节省 中位数 | 最小 | 最大 | 节省占 4.25 |")
    w("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for k in list(range(1, 7)) + ["6abs"]:
        sv = stats(rw(k, "saving"))
        cv = stats(rw(k, "compressed"))
        if sv is None:
            w(f"| {k} | {ledger.ROW_NAMES[k]} | — | — | — | — | — |")
            continue
        w(f"| {k} | {ledger.ROW_NAMES[k]} | {cv[0]:.4f} | {sv[0]:.4f} | {sv[1]:.4f} | {sv[2]:.4f} "
          f"| {100 * sv[0] / RAW:.2f}% |")
    sop = [s["sop"] for s in scalars]
    s_ = stats(sop)
    w(f"| — | 001 分项相加（不含码表成本） | — | {s_[0]:.4f} | {s_[1]:.4f} | {s_[2]:.4f} "
      f"| {100 * s_[0] / RAW:.2f}% |")
    diff = [s["row"][4]["saving"] - s["sop"] for s in scalars]
    ent_diff = [s["q"]["I(C';E)"] - s["q"]["I(C;E)"] for s in scalars]
    tab4 = [s["row"][4]["cw_tab"] + s["row"][4]["sc_tab"] / B for s in scalars]
    w("")
    w("各行的组成（中位数，bit/权重；scale 两列为 bit/scale，计入总数时 ÷32）：\n")
    w("| 行 | 码字流 | 码字码表 | scale 流 | scale 码表 | m_max 标签 | 标签码表 |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for k in list(range(1, 7)) + ["6abs"]:
        if scalars[0]["row"][k] is None:
            continue
        cols = [stats(rw(k, f))[0] for f in ("cw", "cw_tab", "sc", "sc_tab", "lab", "lab_tab")]
        w(f"| {k} | " + " | ".join(f"{c:.6f}" for c in cols) + " |")
    w("")
    w("## M1 码字流条件熵（bit/权重）\n")
    w("| 量 | 中位数 | 最小 | 最大 |")
    w("| --- | ---: | ---: | ---: |")
    for k in ("H(C)", "H(C')", "H(C|E)", "H(C'|E)", "H(C|m_max)", "H(C'|m_max)",
              "H(C|E,m_max)", "H(C'|E,m_max)", "P(m=0)", "H(s|m=0)", "I(C;E)", "I(C';E)"):
        w(line(k, q(k), pct=False))
    pm0 = [p["P(m=0|E)"] for r in results for p in r["per_scale"]]
    hs0 = [(p["H(s|m=0,E)"], p["n_weights"] * p["P(m=0|E)"]) for r in results
           for p in r["per_scale"] if np.isfinite(p["H(s|m=0,E)"])]
    hs_w = sum(h * n for h, n in hs0) / sum(n for _, n in hs0)
    hs_big = [h for h, n in hs0 if n >= 1000]
    w("")
    w(f"每个 scale 取值（{len(results)} 张矩阵共 {len(pm0)} 个 (矩阵, E) 组合）：P(m=0|E) 范围 "
      f"{min(pm0):.4f}–{max(pm0):.4f}；H(s|m=0,E) 按零码字数加权平均 {hs_w:.6f}，"
      f"在零码字 ≥ 1000 个的组合上范围 {min(hs_big):.6f}–{max(hs_big):.6f}。"
      "逐组合数值见 `cond_entropy.csv`，直方图见 `hist_by_scale.csv` / `hist_by_scale.png`。\n")
    w("## M2 m_max 标签\n")
    w("| 量 | 中位数 | 最小 | 最大 |")
    w("| --- | ---: | ---: | ---: |")
    gain = [s["q"]["H(C'|E)"] - s["q"]["H(C'|E,m_max)"] for s in scalars]
    lab = [s["q"]["H(m_max|E)"] / B for s in scalars]
    w(line("H(C'|E) − H(C'|E,m_max)", gain, pct=False))
    w(line("标签成本 H(m_max|E)/B", lab, pct=False))
    w(line("增量净收益（真实）", q("M2 inc_net"), pct=False))
    s0 = [s["q"][f"M2 inc_net S0 rep{i}"] for s in scalars for i in (1, 2, 3)]
    w(line(f"增量净收益（S0，{len(results)}×3 次）", s0, pct=False))
    s0i = [s["q"][f"I(C';E) S0 rep{i}"] for s in scalars for i in (1, 2, 3)]
    w(line(f"I(C';E)（S0，{len(results)}×3 次）", s0i, pct=False, nd=6))
    w("")
    w("## M3 scale 流\n")
    w("| 量 | 中位数 | 最小 | 最大 |")
    w("| --- | ---: | ---: | ---: |")
    w(line("H0(E)（bit/scale）", q("H0(E) [bit/scale]"), pct=False))
    w(line("收益 (8 − H0(E))/B（bit/权重）", [(8 - x) / B for x in q("H0(E) [bit/scale]")],
           pct=False))
    w(line("出现的 scale 取值数 K_E", q("K_E"), pct=False, nd=0))
    esc = sum(q("scales_outside_mode±3"))
    w(line("超出 众数±3 的 scale 数", q("scales_outside_mode±3"), pct=False, nd=0))
    w("")
    w("## M4 码表成本与共用（bit/权重）\n")
    w("| 量 | 中位数 | 最小 | 最大 |")
    w("| --- | ---: | ---: | ---: |")
    w(line("方案 a 码字码表成本 14·K_E·12/N", [s["row"][4]["cw_tab"] for s in scalars], pct=False,
           nd=6))
    w(line("方案 a scale 码表成本 K_E·12/(N/B)/B", [s["row"][4]["sc_tab"] / B for s in scalars],
           pct=False, nd=6))
    w(line("方案 b 码字流 − 方案 a（含码表）", [s["b"]["cw_bits"] - s["a_cw_total"] for s in scalars],
           pct=False, nd=6))
    w(line("方案 c 码字流 − 方案 a（含码表）", [s["c"]["cw_bits"] - s["a_cw_total"] for s in scalars],
           pct=False, nd=6))
    w(line("共用 scale 表 − 每矩阵 scale 表", [s["b"]["scale_bits"] / B - s["a_sc_total"]
                                          for s in scalars], pct=False, nd=6))
    w(line("补充：按绝对 E 全局共用码字表 − 方案 a", [s["ba"]["cw_bits"] - s["a_cw_total"]
                                           for s in scalars], pct=False, nd=6))
    w(line("补充：按绝对 E 按角色共用码字表 − 方案 a", [s["ca"]["cw_bits"] - s["a_cw_total"]
                                            for s in scalars], pct=False, nd=6))
    w(line("补充：按绝对 E 共用 scale 表 − 每矩阵 scale 表",
           [s["ba"]["scale_bits"] / B - s["a_sc_total"] for s in scalars], pct=False, nd=6))
    modes = [s["q"]["mode_E"] - 127 for s in scalars]
    top2 = []
    for r in results:
        ps = sorted(r["per_scale"], key=lambda p: -p["n_scales"])
        top2.append(ps[1]["n_scales"] / ps[0]["n_scales"] if len(ps) > 1 else 0.0)
    flips = {m: modes.count(m) for m in sorted(set(modes))}
    w("")
    w("众数（E_rel 的零点）指数分布：" + "，".join(f"2^{m}：{n} 张" for m, n in flips.items())
      + f"。次多与最多 scale 取值的块数之比中位数 {np.median(top2):.3f}（范围 {min(top2):.3f}–"
      f"{max(top2):.3f}）：两个取值几乎各占一半时，众数在两者之间换边，E_rel 整体平移一格，"
      "共用表的行就错位。补充列改用 scale 字节本身作上下文和符号（256 个符号，无需众数与转义）。"
      "补充不替代规格的方案 b/c，只用来区分“分布不能共用”与“对齐方式不合适”。")
    w("")
    w("## 按层、按角色的中位数（bit/权重）\n")
    items = [(f"第{k}行节省", rw(k, "saving")) for k in list(range(1, 7)) + ["6abs"]]
    items += [("4 − H(C)", [4 - x for x in q("H(C)")]),
              ("H(C) − H(C')", [a - b for a, b in zip(q("H(C)"), q("H(C')"))]),
              ("I(C';E)", q("I(C';E)")),
              ("(8 − H0(E))/B", [(8 - x) / B for x in q("H0(E) [bit/scale]")]),
              ("M2 增量净收益", q("M2 inc_net")),
              ("方案 b − a（码字）", [s["b"]["cw_bits"] - s["a_cw_total"] for s in scalars]),
              ("补充 绝对 E 共用 − a（码字）",
               [s["ba"]["cw_bits"] - s["a_cw_total"] for s in scalars])]
    L.extend(group_table(results, items))
    w("")
    w("## 第 0 节三个问题\n")
    r4 = stats(rw(4, "saving"))
    d = stats(diff)
    w(f"1. **直接测量的总数**：四项放在一起（第 4 行，含码表成本）节省中位数 {r4[0]:.4f} bit/权重"
      f"（{100 * r4[0] / RAW:.2f}%，范围 {r4[1]:.4f}–{r4[2]:.4f}）。与 001 分项相加之差中位数 "
      f"{d[0]:+.6f}（范围 {d[1]:+.6f}–{d[2]:+.6f}）：其中 I(C';E) − I(C;E) 中位数 "
      f"{stats(ent_diff)[0]:+.2e}，码表成本中位数 {stats(tab4)[0]:.2e}。"
      + ("差的绝对值全部 < 10⁻³，分项相加成立。" if max(abs(d[1]), abs(d[2])) < 1e-3
         else "差的绝对值超过 10⁻³，分项相加不成立。"))
    rb = stats([s["b"]["cw_bits"] - s["a_cw_total"] for s in scalars])
    rc = stats([s["c"]["cw_bits"] - s["a_cw_total"] for s in scalars])
    r6 = stats(rw(6, "saving"))
    rba = stats([s["ba"]["cw_bits"] - s["a_cw_total"] for s in scalars])
    w(f"2. **码表成本与共用**：每矩阵一套码表（方案 a）的成本中位数为 "
      f"{stats(tab4)[0]:.2e} bit/权重。改用其余 {len(results) - 1} 张矩阵的共用表（方案 b），码字流比方案 a 多 "
      f"{rb[0]:+.5f}（范围 {rb[1]:+.5f}–{rb[2]:+.5f}）；按角色共用（方案 c）多 {rc[0]:+.5f}"
      f"（范围 {rc[1]:+.5f}–{rc[2]:+.5f}）。第 6 行（全部改用共用表）节省中位数 {r6[0]:.4f}。"
      f"多出的部分主要来自 E_rel 的对齐：补充的按绝对 E 共用表只比方案 a 多 {rba[0]:+.5f}"
      f"（范围 {rba[1]:+.5f}–{rba[2]:+.5f}），对应总账节省中位数 {stats(rw('6abs', 'saving'))[0]:.4f}。")
    ni = stats(q("M2 inc_net"))
    w(f"3. **再加 m_max**：扣除标签成本后增量净收益中位数 {ni[0]:+.5f}（范围 {ni[1]:+.5f}–"
      f"{ni[2]:+.5f}）；S0 上 {stats(s0)[0]:+.5f}（最大 {stats(s0)[2]:+.5f}）。"
      f"第 5 行（含全部码表）节省中位数 {stats(rw(5, 'saving'))[0]:.4f}，对第 4 行 "
      f"{stats([a - b for a, b in zip(rw(5, 'saving'), rw(4, 'saving'))])[0]:+.5f}。")
    w("")
    w("## 说明\n")
    w(f"- 共用 scale 表的字母表：E_rel = E − 众数，取 −3…+3 共 7 个符号，另设 1 个转义符号；"
      f"超出 ±3 的 scale 在转义后另付 8 bit 原值（规格未定义，裁剪本身不可逆）。{len(results)} 张矩阵中超出的 "
      f"scale 共 {esc} 个。每矩阵另加 8 bit 传众数；共用表各格加 0.5。")
    w("- 共用码字表 P(C'|E_rel) 的条件 E_rel 按规格裁剪到 [−3, +3]（解码端已知 E，裁剪只合并上下文，"
      "仍可逆）。")
    w("- 众数并列时取较小的 scale 字节。")
    w(f"- 运行：{meta['seconds']:.0f} s（{meta['jobs']} 进程），失败 {meta['failures']} 张。")
    (out / "summary.md").write_text("\n".join(L) + "\n")


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = ["L0.E164.w1", "L13.E135.w2", "L39.E183.w3"]
    sel = [r for r in results if r["meta"]["matrix"] in pick] or results[:3]
    fig, axes = plt.subplots(1, len(sel), figsize=(5 * len(sel), 4), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), sel):
        cmap = plt.get_cmap("viridis")
        ps = r["per_scale"]
        for i, p in enumerate(ps):
            h = p["hist_m"] / p["n_weights"]
            h = np.where(h > 0, h, np.nan)  # zero counts are not drawn on the log axis
            ax.plot(range(8), h, marker="o", ms=3, color=cmap(i / max(1, len(ps) - 1)),
                    label=f"E=2^{p['E'] - 127} ({p['n_scales']} blk)")
        ax.set_title(r["meta"]["matrix"])
        ax.set_xticks(range(8))
        ax.set_xticklabels(["0", ".5", "1", "1.5", "2", "3", "4", "6"])
        ax.set_xlabel("|value| (magnitude m)")
        ax.set_yscale("log")
        ax.legend(fontsize=6)
    np.atleast_1d(axes)[0].set_ylabel("P(m | E)")
    fig.tight_layout()
    fig.savefig(out / "hist_by_scale.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--samples", default="experiments/r1-r4/samples.json")
    ap.add_argument("--validation", default="experiments/002-ledger/validation.json")
    ap.add_argument("--out", default="experiments/002-ledger")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    with open(args.validation) as f:
        if not json.load(f)["all_pass"]:
            raise SystemExit("section 4 checks have failures; refusing to read model data")
    with open(args.samples) as f:
        sdoc = json.load(f)
    samples = sdoc["matrices"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results, failures = [None] * len(samples), {}
    with ProcessPoolExecutor(min(args.jobs, len(samples))) as ex:
        futs = [ex.submit(work_one, args.snapshot, s, i) for i, s in enumerate(samples)]
        for i, fut in enumerate(futs):
            try:
                results[i] = fut.result()
                print(f"done {samples[i]['matrix']}: {results[i]['seconds']:.0f}s", flush=True)
            except Exception:
                failures[samples[i]["matrix"]] = traceback.format_exc()
                print(f"FAILED {samples[i]['matrix']}\n{failures[samples[i]['matrix']]}", flush=True)
    if failures:
        (out / "failures.json").write_text(json.dumps(failures, indent=1))
    results = [r for r in results if r is not None]
    ledgers, cond, hist, tables, scalars = build(results)
    write_csv(out / "ledger.csv", LEDGER_FIELDS, ledgers)
    write_csv(out / "cond_entropy.csv", list(META) + ["E", "quantity", "value", "note"], cond)
    write_csv(out / "hist_by_scale.csv",
              list(META) + ["E", "exponent", "E_rel", "n_scales", "n_weights", "P(m=0|E)",
                            "H(s|m=0,E)"] + [f"C'={x}" for x in CP_LABELS], hist)
    write_csv(out / "tables.csv",
              list(META) + ["stream", "scheme", "length", "table_cost", "total_bit_per_weight",
                            "excess_vs_a", "train_matrices", "escapes", "note"], tables)
    summary(results, scalars, out, {"layers": sdoc["layers"], "seconds": time.time() - t0,
                                    "jobs": args.jobs, "failures": len(failures)})
    plot(results, out)
    print(f"{len(results)} ok, {len(failures)} failed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
