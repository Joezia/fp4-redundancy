"""TX: tables of every variant (report_tx_variants.md) from results/tx/stage*/summary.json.

    PYTHONPATH=. python experiments/010-transforms/tx/tx_report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import txcore as T  # noqa: E402

OUT = T.E10 / "report_tx_variants.md"

NAMES = {"A1_wht": "A1 WHT（S 变换蝶形）", "A2_dct": "A2 DCT（近似 + 残差）", "A3_dft": "A3 DFT（近似 + 残差）",
         "A4_dwt": "A4 DWT（整数提升）", "A5_lapped": "A5 重叠变换（近似 + 残差）", "B1_lorenzo": "B1 Lorenzo / 差分",
         "B2_xor": "B2 异或差分", "B3_regress": "B3 按块回归", "B4_interp": "B4 插值预测", "B5_multilevel": "B5 二维层级插值",
         "C_order": "C1 转置 / C2 空间填充曲线", "D1_split": "D1 字段 / 位平面拆分",
         "A7_ica": "A7 ICA（单矩阵）", "C3_seriation": "C3 行重排（seriation）", "C4_bwt": "C4 BWT + MTF",
         "A6_klt": "A6 KLT / PCA（跨专家）", "A7_ica_shared": "A7 ICA（跨专家共享）", "A8_tucker": "A8 Tucker / HOSVD",
         "C3_expert_order": "C3 专家顺序（以前一个专家为上下文）"}


def f(x, d=4, sign=True):
    if x is None:
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def per_matrix_table(meth, d):
    lines = [f"### {NAMES.get(meth, meth)}", ""]
    if "variants" not in d:
        lines += [f"状态：{d.get('status')}", ""]
        return lines
    best = d.get("best_on_train")
    lines += ["| 变体 | 基准视图 | C0 | C1 | 边信息 | 显现度 C0 | 显现度 C1 | 净收益 | T2 净收益 | 编码增益 | 非零比例 | 往返 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for vn, x in d["variants"].items():
        te = x.get("test")
        if not te:
            continue
        t2 = x.get("T2", {})
        mark = " **（训练集最好）**" if vn == best else ""
        lines.append(f"| `{vn}`{mark} | {x['base_view']} | {f(te['C0'], sign=False)} | {f(te['C1'], sign=False)} | "
                     f"{f(te['side_C1'], sign=False)} | {f(te['appear_C0'])} | {f(te['appear_C1'])} | {f(te.get('net_gain'))} | "
                     f"{f(t2.get('net_gain'))} | {f(te.get('coding_gain_mean'), 2, False)} | {f(te.get('nonzero_frac_mean'), 3, False)} | "
                     f"{'通过' if te['roundtrip_all'] else '失败'} |")
    lines.append("")
    return lines


def group_table(meth, d):
    lines = [f"### {NAMES.get(meth, meth)}", ""]
    if "variants" not in d:
        lines += [f"状态：{d.get('status')}", ""]
        return lines
    lines += [f"选参（选参专家、8 个专家共享）选出：`{d.get('best_on_xsel')}`", "",
              "| 变体 | 设置 | C0 | C1 | 边信息 | 显现度 C0 | 显现度 C1 | 净收益 | 往返 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for vn, x in d["variants"].items():
        for key in sorted(k for k in x if k.startswith("n")):
            s = x[key]
            if not s or not key.endswith("xrep"):
                continue
            lines.append(f"| `{vn}` | {key} | {f(s['C0'], sign=False)} | {f(s['C1'], sign=False)} | {f(s['side_C1'], sign=False)} | "
                         f"{f(s['appear_C0'])} | {f(s['appear_C1'])} | {f(s.get('net_gain'))} | {'通过' if s['roundtrip_all'] else '失败'} |")
    lines.append("")
    return lines


def main():
    lines = ["# TX 各方法全部变体的结果", "",
             "由 `tx/tx_report.py` 从 `results/tx/stage*/summary.json` 生成。单位：bit/权重。",
             "C0、C1 为变换后数据的码长（含边信息，不含 scale 流）；显现度相对“基准视图”的原始码长；净收益相对参考基准"
             "（008 链式 + R9c，含 scale 流），变换方案取 C0、C1 中较好的一个。测试集为 010 的 24 个测试矩阵（= 011 T1）；"
             "跨专家方法为跨专家样本中报告用的 8 个专家。", ""]
    for st in ("1", "2", "3"):
        p = T.RES / f"stage{st}" / "summary.json"
        if not p.exists():
            continue
        s = json.load(open(p))
        lines += [f"## 阶段 {st}", ""]
        if st == "3":
            lines += ["| 方法 | 集合 | 参数量 | C0 | C1 | 显现度 C0 | 显现度 C1 | 净收益（摊到测试矩阵） | 净收益（摊到全模型） | 往返 |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
            for k, d in s["methods"].items():
                for sp in ("train", "test", "T2"):
                    x = d.get(sp)
                    if x:
                        lines.append(f"| {k} | {sp} | {x['params']} | {f(x['C0'], sign=False)} | {f(x['C1'], sign=False)} | "
                                     f"{f(x['appear_C0'])} | {f(x['appear_C1'])} | {f(x.get('net_gain_model_over_test'))} | "
                                     f"{f(x.get('net_gain_model_over_model'))} | {'通过' if x['roundtrip_all'] else '失败'} |")
            lines += ["", "E4 坐标网络：", "", "| 设置 | 参数量 | C0 | C1 | 显现度 C0 | 显现度 C1 | 相对均方误差 | 净收益 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for k, x in s.get("E4_inr", {}).items():
                lines.append(f"| `{k}` | {x['params']} | {f(x['C0'], sign=False)} | {f(x['C1'], sign=False)} | {f(x['appear_C0'])} | "
                             f"{f(x['appear_C1'])} | {f(x['relative_mse_mean'], 4, False)} | {f(x.get('net_gain'))} |")
            lines.append("")
            continue
        for meth, d in s.items():
            if meth in ("note", "sweep", "A8_tucker_layer_mode") or not isinstance(d, dict):
                if meth == "sweep":
                    lines += ["### 共享专家数扫描（报告专家）", "", "| 文件 | 集合 | 显现度 C0 | 显现度 C1 | 边信息 | 净收益 |",
                              "|---|---|---:|---:|---:|---:|"]
                    for k, dd in d.items():
                        for sp, x in dd.items():
                            lines.append(f"| `{k}` | {sp} | {f(x['appear_C0'])} | {f(x['appear_C1'])} | {f(x['side_C1'], sign=False)} | {f(x.get('net_gain'))} |")
                    lines.append("")
                if meth == "A8_tucker_layer_mode":
                    lines += ["### A8 层模式（相邻 4 层同一专家编号）", "", "| 变体 | 显现度 C0 | 显现度 C1 | 边信息 | 净收益 |", "|---|---:|---:|---:|---:|"]
                    for k, x in d.items():
                        lines.append(f"| `{k}` | {f(x['appear_C0'])} | {f(x['appear_C1'])} | {f(x['side_C1'], sign=False)} | {f(x.get('net_gain'))} |")
                    lines.append("")
                continue
            if "best_on_xsel" in d or (st == "2" and meth in NAMES and meth in ("A6_klt", "A7_ica_shared", "A8_tucker", "C3_expert_order")):
                lines += group_table(meth, d)
            else:
                lines += per_matrix_table(meth, d)
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
