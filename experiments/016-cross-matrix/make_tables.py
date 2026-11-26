"""Spec 016: markdown tables for report.md (pattern strength, not code lengths; user request 2026-11-26).

    PYTHONPATH=.:experiments/016-cross-matrix python experiments/016-cross-matrix/make_tables.py > experiments/016-cross-matrix/results/tables.md
Reads results/summary.json (run summarize.py first), the pair JSONL files and patterns_<model>.json.
"""
import json

import numpy as np

import common as K

S = json.load(open(K.RES / "summary.json"))
LABEL = {"deepseek": "DeepSeek-V4.1-Flash", "kimi": "Kimi-K3"}
CATN = {"A1_same_expert_w1w3": "A1 同专家 w1–w3", "A2_same_expert_w2": "A2 同专家 w1/w3–w2ᵀ",
        "A3_same_layer_same_role": "A3 同层异专家同角色", "A4_same_layer_other_role": "A4 同层异专家异角色",
        "A5_adjacent_layer_same_expert": "A5 相邻层同编号", "A6_adjacent_layer_random_expert": "A6 相邻层随机专家",
        "A7_far_layer": "A7 远层", "A8_self": "A8 自身（非零位移）"}
SIG = ["eq", "sign", "e1", "e0", "m", "v", "absv", "w"]


def mx(v):
    return f"{np.median(v):.1f} / {np.max(v):.1f}"


def part_a(m):
    rows = [json.loads(l) for l in open(K.RES / f"pairs_gpu_{m}.jsonl")]
    cats = list(CATN)
    print(f"\n#### {LABEL[m]}：同位置（零位移）的相似度 z（真实 中位数 / 最大；打乱 最大）\n")
    print("z：码字相同（eq）为（相同数 − 偶然期望）/ 标准差；其余信号为皮尔逊相关 × √N。|z| 在无结构时约为 1。\n")
    print("| 类别 | " + " | ".join(SIG) + " |")
    print("|---|" + "---:|" * len(SIG))
    for c in cats[:-1]:
        rs = [r for r in rows if r["cat"] == c]
        cells = []
        for s in SIG:
            re_ = [r["real"]["global_shift"][f"{s}_z_at_zero_shift"] for r in rs]
            sh = [abs(r["shuffled"]["global_shift"][f"{s}_z_at_zero_shift"]) for r in rs]
            cells.append(f"{np.median(re_):+.1f} / {re_[int(np.argmax(np.abs(re_)))]:+.1f}（{max(sh):.1f}）")
        print(f"| {CATN[c]} | " + " | ".join(cells) + " |")
    print(f"\n#### {LABEL[m]}：全局平移——所有二维位移（|dy| ≤ R/2，|dx| ≤ W/2，约 1.2×10⁷ 个）中最大的 |z|（真实 中位数 / 最大；打乱 最大）\n")
    print("| 类别 | " + " | ".join(SIG) + " | 最佳 eq 位移处 相同率 / 偶然 |")
    print("|---|" + "---:|" * (len(SIG) + 1))
    for c in cats:
        rs = [r for r in rows if r["cat"] == c]
        cells = []
        for s in SIG:
            re_ = [r["real"]["global_shift"][f"{s}_max_z_nonzero_shift"] for r in rs]
            sh = [r["shuffled"]["global_shift"][f"{s}_max_z_nonzero_shift"] for r in rs]
            cells.append(f"{mx(re_)}（{max(sh):.1f}）")
        ratio = [r["real"]["global_shift"]["eq_frac_at_best_shift_exact"] / r["real"]["global_shift"]["eq_chance_p0"] for r in rs]
        print(f"| {CATN[c]} | " + " | ".join(cells) + f" | {np.median(ratio):.4f} / {max(ratio):.4f} |")
    print(f"\n#### {LABEL[m]}：局部块运动搜索——B 的随机块在 A 的所有位置（步长 1）中最相似处（A1–A7 合计）\n")
    print("| 块 | 块数 | 最佳处相同率 / 偶然 中位数 真实（打乱） | 最大 真实（打乱） | 比特一致率 最大 真实（打乱） | 超过打乱块最大值的真实块 |")
    print("|---|---:|---:|---:|---:|---:|")
    for T in (32, 128):
        tr = [t for r in rows if r["cat"] != "A8_self" for t in r["local"][f"T{T}_real"]]
        ts = [t for r in rows if r["cat"] != "A8_self" for t in r["local"][f"T{T}_shuffled"]]
        rr = np.array([t["eq_max"] / t["eq_p0"] for t in tr]); rs_ = np.array([t["eq_max"] / t["eq_p0"] for t in ts])
        print(f"| {T}×{T} | {len(tr)} | {np.median(rr):.3f}（{np.median(rs_):.3f}） | {rr.max():.3f}（{rs_.max():.3f}） | "
              f"{max(t['bits_max'] for t in tr):.4f}（{max(t['bits_max'] for t in ts):.4f}） | {int((rr > rs_.max()).sum())} |")
    sr = [t for r in rows if r["cat"] == "A8_self" for t in r["local"]["T32_real"]]
    print(f"\n同一矩阵内（A8，排除自身及重叠位置）32×32 块的最佳处相同率 / 偶然：中位数 {np.median([t['eq_max'] / t['eq_p0'] for t in sr]):.3f}，最大 {max(t['eq_max'] / t['eq_p0'] for t in sr):.3f}。")
    c = S[m].get("A_comp")
    if c:
        print(f"\n#### {LABEL[m]}：通用压缩器的归一化压缩距离 NCD（中位数；越接近 1 越无关，对照 = A 打乱）\n")
        print("| 类别 | zstd packed 真实（对照） | xz packed 真实（对照） | zstd 展开 真实（对照） | xz 展开 真实（对照） |")
        print("|---|---:|---:|---:|---:|")
        for k, d in c["by_cat"].items():
            print(f"| {CATN[k]} | " + " | ".join(f"{d[f'{x}.NCD']['median']:.5f}（{d[f'{x}.NCD_control']['median']:.5f}）"
                                                 for x in ("packed_zstd19", "packed_xz9e", "unpacked_zstd19", "unpacked_xz9e")) + " |")


def part_b1(m):
    b = S[m].get("B1")
    if not b:
        return
    v = S[m].get("B1_verify")
    print(f"\n#### {LABEL[m]}：B1 全模型精确重复（单元在任意矩阵、任意位置）\n")
    print("| 单元 | 全模型单元数 | 跨矩阵检查的单元数 | 矩阵内重复 | 跨矩阵哈希相同 | 64 位哈希预期偶然碰撞 | 读回原始字节核对后的真重复 | 最大重复组 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for u, d in b["units"].items():
        if v and u in v["n_true_repeats"]:
            tr = str(v["n_true_repeats"][u])
        elif d["cross_repeat_units_by_hash"] == 0:
            tr = "0"
        else:
            tr = f"{d['cross_repeat_units_by_hash']:,}（碰撞预期 {d['expected_false_hash_collisions']:.1g}，无需核对）"
        print(f"| {u} | {d['n_units_model']:,} | {d['n_units_checked_cross']:,} | {d['within_matrix']['dup_units']:,} | "
              f"{d['cross_repeat_units_by_hash']:,} | {d['expected_false_hash_collisions']:.3g} | {tr} | {d['max_group_size']} |")


def part_b23(m):
    b = S[m].get("B2_B3")
    if not b:
        return
    print(f"\n#### {LABEL[m]}：B2 全模型向量最近邻（数据库 = {b['matrices_scanned']:,} / {b['matrices_total']:,} 个矩阵的全部向量"
          f"{'，按固定种子随机顺序扫描' if b['random_order'] else ''}）\n")
    print("| 向量 | 信号 | 查询数 | 最佳相似度 真实 中位数 / p90 / 最大 | 对照 中位数 / p90 / 最大 | 超过对照最大值的真实查询 | 同专家 / 同层其他专家 / 其他层 的最佳中位数 真实（对照） |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for typ, d in b["vec"].items():
        for sig in ("v", "w", "bits"):
            x = d[sig]
            r, n_ = x["top1_real"], x["top1_null"]
            cr, cn = x["by_category_real"], x["by_category_null"]
            cats = " / ".join(f"{cr[c]['median']:.3f}（{cn[c]['median']:.3f}）" for c in ("same_expert", "same_layer", "other_layer"))
            print(f"| {'神经元' if typ == 'neuron' else '隐藏维'} | {sig} | {d['n_real_queries']} | {r['median']:.3f} / {r['p90']:.3f} / {r['max']:.3f} | "
                  f"{n_['median']:.3f} / {n_['p90']:.3f} / {n_['max']:.3f} | {x['real_above_null_max']} | {cats} |")
    print("\nv、w 为 |余弦|；bits 为 4 个比特平面合计的比特一致率。")
    print(f"\n#### {LABEL[m]}：有损视角——以伙伴向量做最优缩放预测时残差能量的平均降幅（mean cos²，真实 / 对照）\n")
    print("| 向量 | 信号 | 全模型最佳 | 同矩阵 | 同专家 | 同层其他专家 | 其他层 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for typ, d in b["vec"].items():
        for sig in ("v", "w"):
            r, n_ = d[sig]["mean_cos2_real"], d[sig]["mean_cos2_null"]
            print(f"| {'神经元' if typ == 'neuron' else '隐藏维'} | {sig} | " + " | ".join(
                f"{100 * r[c]:.2f}% / {100 * n_[c]:.2f}%" for c in ("best", "same_matrix", "same_expert", "same_layer", "other_layer")) + " |")
    print(f"\n#### {LABEL[m]}：B3 局部块全模型运动搜索（比特一致率，步长 1，二维任意位置）\n")
    print("| 形状 | 查询数 | 其他矩阵最佳 真实 中位数 / 最大 | 对照 中位数 / 最大 | 超过对照最大值的真实查询 |")
    print("|---|---:|---:|---:|---:|")
    for s, d in b["pat"].items():
        print(f"| {s} | {d['n_real_queries']} | {d['bits_other_real']['median']:.4f} / {d['bits_other_real']['max']:.4f} | "
              f"{d['bits_other_null']['median']:.4f} / {d['bits_other_null']['max']:.4f} | {d['real_other_above_null_max']} |")


def patterns(m):
    f = K.RES / f"patterns_{m}.json"
    if not f.exists():
        return
    p = json.load(open(f))
    print(f"\n#### {LABEL[m]}：相似神经元的形态（超过对照最大值的查询；{p['n_queries']} 个随机神经元查询）\n")
    print("| 伙伴所在 | 对照阈值 |cos w| | 超过阈值的比例 | |cos w| 中位数 / p90 / 最大 | 负余弦比例 | 码字相同率 / 偶然 中位数 / 最大 | 非零位置符号一致率 中位数 / 最大 | |v| 相关 中位数 | 同神经元编号 | 同专家编号 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    nm = {"same_expert": "同专家其他矩阵", "same_layer": "同层其他专家", "other_layer": "其他层"}
    for c, d in p["cats"].items():
        if not d.get("abs_cos_w"):
            continue
        print(f"| {nm[c]} | {p['control_threshold_abs_cos_w'][c]:.3f} | {d['frac_above_control']:.1%} | "
              f"{d['abs_cos_w'][0]:.3f} / {d['abs_cos_w'][1]:.3f} / {d['abs_cos_w'][2]:.3f} | {d['frac_negative_cos']:.1%} | "
              f"{d['eq_rate_over_chance'][0]:.3f} / {d['eq_rate_over_chance'][2]:.2f} | {d['sign_agree'][0]:.3f} / {d['sign_agree'][2]:.3f} | "
              f"{d['absv_corr'][0]:.3f} | {d['same_neuron_index']:.1%} | {d['same_expert_index']:.1%} |")
    for c in ("same_layer", "other_layer"):
        d = p["cats"][c]
        if d.get("role_pairs"):
            print(f"\n{nm[c]}的角色配对：" + "，".join(f"{k} {v}" for k, v in sorted(d["role_pairs"].items(), key=lambda x: -x[1])) + "。")
    d = p["cats"]["other_layer"]
    if d.get("layer_distance"):
        print("\n其他层伙伴的层距分布（层距：个数）：" + "，".join(f"{k}：{v}" for k, v in d["layer_distance"].items()) + "。")
    print("\n按查询所在层，超过对照阈值的比例（同专家 / 同层其他专家 / 其他层；n）：\n")
    print("| 层 | n | 同专家 | 同层其他专家 | 其他层 |")
    print("|---:|---:|---:|---:|---:|")
    for L, d in p["frac_above_control_by_layer"].items():
        print(f"| {L} | {d['n']} | {d['same_expert']:.0%} | {d['same_layer']:.0%} | {d['other_layer']:.0%} |")
    print("\n按查询角色：" + "；".join(f"{r} {d['same_expert']:.0%} / {d['same_layer']:.0%} / {d['other_layer']:.0%}" for r, d in p["frac_above_control_by_role"].items()) + "。")
    q = p["top1_to_top4"]
    print("\n第 1–4 相似的向量 |cos w|（中位数）：真实 " + " / ".join(f"{x[0]:.3f}" for x in q["real_quantiles_50_90_99"]) +
          "；对照 " + " / ".join(f"{x[0]:.3f}" for x in q["control_quantiles_50_90_99"]) + "。")


def shared(m):
    f = K.RES / f"shared_dirs_{m}.json"
    if not f.exists():
        return
    d = json.load(open(f))
    print(f"\n#### {LABEL[m]}：相似是不是少数公共方向造成的（每层取上下三层、每层 {d['n_experts_per_layer_for_directions']} 个专家的全部神经元向量求主方向，排除查询和伙伴所在专家）\n")
    print("| 层 | 查询数 | 前 64 个主方向占能量 | 查询在前 64 方向的能量 真实（打乱） | 同层伙伴 |cos| 去掉前 0 / 4 / 16 / 64 个方向 | 其他层伙伴 |cos| 去掉前 0 / 4 / 16 / 64 个方向 |")
    print("|---:|---:|---:|---:|---:|---:|")
    for L, v in d["layers"].items():
        s_ = v["summary"]
        f4 = lambda key: " / ".join(f"{s_[key][k]:.3f}" for k in ("0", "4", "16", "64")) if key in s_ else "—"
        print(f"| {L} | {s_['n_queries']} | {100 * s_['eigen_share_top_k_cumulative']['64']:.2f}% | "
              f"{100 * s_['query_energy_topk_median']['64']:.2f}%（{100 * s_['shuffled_energy_topk_median']['64']:.2f}%） | "
              f"{f4('same_layer_abs_cos_median_before_and_after_removing_k')} | {f4('other_layer_abs_cos_median_before_and_after_removing_k')} |")


def joint(m):
    c = S[m].get("C_joint")
    if not c:
        return
    print(f"\n#### {LABEL[m]}：C 联合压缩（zstd −9 --long=31，联合压缩大小 / 逐个单独压缩大小之和）\n")
    print("| 组 | 矩阵数 | 表示 | 真实 | 对照（每个矩阵内部打乱） |")
    print("|---|---:|---|---:|---:|")
    by = {}
    for r in c:
        by.setdefault((r["group"], r["rep"]), {})["shuf" if r["shuffled"] else "real"] = r
    for (g, rep), d in by.items():
        if "real" in d and "shuf" in d:
            print(f"| {g} | {d['real']['n_matrices']} | {rep} | {d['real']['joint_bytes'] / d['real']['separate_sum_bytes']:.7f} | "
                  f"{d['shuf']['joint_bytes'] / d['shuf']['separate_sum_bytes']:.7f} |")


def p99_table():
    f = K.RES / "b2_threshold_p99.json"
    if not f.exists():
        return
    d = json.load(open(f))
    print("\n### 两个模型对比：神经元查询中最佳伙伴超过对照 99 百分位的比例（纯偶然约 1%）\n")
    print("| 模型 | 伙伴所在 | 对照 p99 |cos w| | 超过比例 | 浅 / 中 / 深三分之一层 | 最佳 |cos w| 中位数 真实（对照） |")
    print("|---|---|---:|---:|---:|---:|")
    nm = {"same_expert": "同专家其他矩阵", "same_layer": "同层其他专家", "other_layer": "其他层"}
    for m, r in d.items():
        for c, x in r.items():
            t = x["frac_above_p99_by_depth_third"]
            print(f"| {LABEL[m]} | {nm[c]} | {x['null_p99']:.3f} | {x['frac_real_above_null_p99']:.1%} | "
                  f"{t['浅']:.0%} / {t['中']:.0%} / {t['深']:.0%} | {x['real_median']:.3f}（{x['null_median']:.3f}） |")


p99_table()
for m in S:
    print(f"\n### {LABEL[m]}")
    part_b1(m); part_a(m); joint(m); part_b23(m); patterns(m); shared(m)
