"""Text companions of the figures (second version): figure_notes.md (one section per figure, in
Chinese, for writing slides), overview.md (verdict table) and r5_gain.json. Reads result files only.
Wording specific to one kind of data comes from the config (`figures.vocab`, `figures.<fig>.note_zh`).
"""
import json

import numpy as np

TWO_LEVEL = ("R1", "R9", "R10")

PATTERN_ZH = {"R1": "符号概率不均匀", "R2": "不同分组的分布不同", "R3": "与邻近符号有关系",
              "R4": "与邻近数值有线性关系", "R5": "各行是少数图案的混合", "R6": "一段内容原样再现",
              "R7": "段与段相似", "R8": "各段只用字典里少数原子", "R9": "产生过程的约束", "R10": "等价存法"}
INDICATOR_ZH = {"R1": "各{sym_zh}的占比", "R2": "I({sym_zh}; 分组) − 完全打乱基线",
                "R3": "I(C_j; C_{j+d})", "R4": "自相关 ρ(d)", "R5": "奇异值能量份额",
                "R6": "长度 k 窗口的重复比例", "R7a_unit": "与最相似的另一{unit_zh}的相同位置数",
                "R7a_row": "与最相关的另一行的 |ρ|", "R7a_partner": "与对应行的 |ρ|",
                "R7b": "k-means 留出失真比", "R8": "字典学习留出 R²", "R9": "实际与预期分布的总变差",
                "R10": "等价存法个数"}


def g(x, d=4):
    if x is None:
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if not np.isfinite(x):
        return "—"
    return f"{x:.{d}g}"


def pct(x, d=3):
    return "—" if x is None else f"{100 * x:.{d}g}%"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class Notes:
    def __init__(self, man, res, syn, ctx):
        self.man, self.res, self.syn, self.ctx = man, res, syn, ctx
        self.zh = ctx.L["zh"]

    def z(self, s, **kw):
        return self.ctx.t(s, **kw)

    def verdict(self, rec, fid=None):
        v = rec.get("verdict", "not_applicable")
        base = "R9" if (fid or rec["figure"]).startswith("R9") else (fid or rec["figure"])
        if v == "present":
            v = "present_two" if base in TWO_LEVEL else "present_below"
        return self.zh["verdict"][v]

    def base_zh(self, name):
        return self.z(self.zh["baseline"].get(name, name))

    def data(self, fid):
        p = self.res / "data" / f"{fid}.json"
        return load(p) if p.exists() else None

    def syn_has(self, fid):
        return (self.syn.get(fid) or {}).get("has")

    def extra_notes(self, fid):
        return [self.z(s) for s in self.ctx.fig(fid).get("note_zh", [])]

    def alt_line(self, s, fid):
        out = []
        for b, a in (s.get("alt") or {}).items():
            out.append(f"另算的基线「{self.base_zh(b)}」（只存于结果文件，不画）下的判定：{self.zh['verdict'][a['verdict'] if a['verdict'] != 'present' else 'present_below']}。")
        return out

    # ------------------------------------------------------------------ per figure

    def r1(self, rec):
        s = rec["summary"]
        p = s["prob"]["median"]
        top = sorted(range(len(p)), key=lambda i: -p[i])[:3]
        return {
            "axes": ["横轴：{sym_zh}编号 c，按编码顺序 0–{A}，每个都标出刻度。".replace("{A}", str(len(p) - 1)),
                     "纵轴：该{sym_zh}在条目中的占比（%）。"],
            "lines": [f"蓝柱：{s['n_entries']} 个条目的中位数；黑色误差线：各条目的最小值–最大值。",
                      f"灰色水平线：均匀分布 1/{len(p)}（{pct(s['uniform'])}）。"],
            "baseline": "无打乱基线（打乱不改变直方图）。",
            "breakeven": f"不画回本线。判定为两档：原始位宽 − H 大于最小幅度 {rec['detail'].get('min_amp')} 即为存在。",
            "keys": [f"零阶熵 H = {g(s['H'])} bit（中位数），原始 {g(s['raw_bits'])} bit，可省 {g(s['saving'])} bit/{{sym_zh}}。",
                     "占比最高的三个编号：" + "，".join(f"c = {i}（{pct(p[i])}）" for i in top) + "。"]}

    def r2(self, rec):
        s = rec["summary"]
        r1 = self.data("R1")
        skew = r1["summary"]["saving"] if r1 else None
        raw = r1["summary"]["raw_bits"] if r1 else None
        names = self.ctx.fig("R2").get("groups")
        rows = ["| 分组变量 | 图中名称 | 净互信息中位数（bit） | 最小 | 最大 | 回本线 | 超出基线 | 过回本线 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        neg, dropped = [], []
        for gname, r in sorted(s["groups"].items(), key=lambda t: -t[1]["net"]):
            shown = (names or {}).get(gname) if names is not None else gname
            if names is not None and gname not in names:
                dropped.append(gname)
            if r["net"] < 0:
                neg.append(f"{shown or gname} = {g(r['net'])}")
            mm = (g(r.get("net_min")), g(r.get("net_max"))) if r.get("scope") == "symbol" else ("跨条目合并", "—")
            rows.append(f"| `{gname}` | {shown or '不画'} | {g(r['net'])} | {mm[0]} | {mm[1]} | {g(r['cost'], 3)} | "
                        f"{'是' if r['exceeds'] else '否'} | {'是' if r['pays'] else '否'} |")
        keys = [f"边缘偏斜 {g(raw)} − H(C) = {g(skew)} bit/{{sym_zh}}（各柱下段，全部相同）。"] if skew is not None else []
        keys += rows
        if neg:
            keys.append("净互信息为负、图中显示为 0 的变量：" + "，".join(neg) + "。")
        if dropped:
            keys.append("按配置不画的变量：" + "，".join(f"`{d}`" for d in dropped) + "（数值见上表）。")
        return {
            "axes": ["横轴：分组变量 G。None 在最左（不分组），其余按柱高从高到低。",
                     f"纵轴：相对 {g(raw)} bit 原始存储的节省（bit/{{sym_zh}}），从 0 开始。"],
            "lines": ["堆叠柱：下段（浅色）= 边缘偏斜 4 − H(C)，各柱相同；上段（深色）= 该变量的净互信息 I(C; G)，"
                      "即真实互信息减去完全打乱基线的均值，负值显示为 0。",
                      "误差线（只在逐条目计算的变量上）：各条目净互信息的最小–最大值，画在下段中位数之上。"
                      "跨条目合并计算的变量（只有一个合并值）不加误差线。"],
            "baseline": "完全打乱：条目内全部{sym_zh}随机置换、标签不动，3 次重复；跨条目合并的变量在条目之间置换标签。",
            "breakeven": "图中不画回本线。每个变量的成本 = (A − 1)(G − 1)·16 / N（码表），"
                         "解码端没有的标签再加 H(标签) / 覆盖的{sym_zh}数；数值见表。",
            "keys": keys}

    def r3(self, rec):
        s = rec["summary"]
        d = rec["detail"]["exceed_lags"]
        su = s.get("same_unit_d1", {})
        return {
            "axes": ["横轴：距离 d（同一行内相隔的位置数），1–" + str(s["lags"][-1]) + "，每 16 一个刻度。",
                     "纵轴：互信息 I(C_j; C_{j+d})（bit），C_j 为一行中第 j 个{sym_zh}。"],
            "lines": [f"蓝线：全部{{sym_zh}}对的互信息，{s['n_entries']} 个条目的中位数，浅色阴影为最小–最大。",
                      f"灰线：基线（{self.base_zh(s['baseline'])}）均值的中位数。",
                      "橙色虚线：回本线。灰色点线：d = {L}，{unit_zh}长。".replace("{L}", str(s.get("unit_len")))],
            "baseline": f"{self.base_zh(s['baseline'])}，3 次重复。",
            "breakeven": f"一张 A × A 码表（A = 16，每参数 16 bit）的成本 (A − 1)·A·16 / N = {g(s['breakeven'], 3)} bit。",
            "keys": [f"d = 1：实测 {g(s.get('real_d1'))} bit，基线 {g(s.get('base_d1'))} bit。",
                     f"超出基线的距离：{len(d)} 个（{d[0]}–{d[-1]}）。" if d else "没有超出基线的距离。",
                     f"同一{{unit_zh}}内相邻对（d = 1）：实测 {g(su.get('real'))} bit。"]
                    + [f"{{unit_zh}}内打乱基线下 d = 1 的基线值：{g(a.get('base_d1'))} bit。"
                       for b, a in (s.get('alt') or {}).items() if b == 'unit']
                    + self.alt_line(s, "R3")}

    def r4(self, rec):
        s = rec["summary"]
        c = s["curves"]
        d = rec["detail"]["exceed_lags"]
        return {
            "axes": ["横轴：距离 d（同一行内相隔的位置数），每 16 一个刻度。", "纵轴：自相关 ρ(d)。"],
            "lines": [f"蓝线：带符号的数值 v；橙线：幅度 |v|。各为 {s['n_entries']} 个条目的中位数，浅色阴影为最小–最大。",
                      f"两条灰线：两者的基线（{self.base_zh(s['baseline'])}），图例合为一项。",
                      "橙色虚线：回本线 ρ = 0.5。灰色点线：d = {L}。".replace("{L}", str(s.get("unit_len")))],
            "baseline": f"{self.base_zh(s['baseline'])}，3 次重复；相关系数用全局均值与方差。",
            "breakeven": "|ρ(1)| ≥ 0.5 时差分预测开始有利（与参数位数无关）。另有谱平坦度给出的线性预测上限 "
                         f"½·log2(1/平坦度) = {g(s['bound'], 3)} bit。",
            "keys": [f"ρ(1)：v = {g(c['v']['real']['median'][0], 3)}（基线 {g(c['v']['base']['median'][0], 3)}），"
                     f"|v| = {g(c['abs']['real']['median'][0], 3)}（基线 {g(c['abs']['base']['median'][0], 3)}）。",
                     f"超出基线的距离：v {len(d.get('v', []))} 个，|v| {len(d.get('abs', []))} 个。"]
                    + self.alt_line(s, "R4")}

    def _r5_one(self, rec, tag):
        s = rec["summary"]
        out = [f"{tag}最大层份额 {pct(s['top'])}（基线 {pct(s['top_base'])}），回本线 {pct(s['breakeven'])}；"
               f"超出基线的层：{rec['detail']['exceed_layers'][:8]}{'…' if len(rec['detail']['exceed_layers']) > 8 else ''}；"
               f"判定：{self.verdict(rec, 'R5')}。"]
        gn = s.get("gain") or {}
        if gn:
            out.append(f"{tag}严格收益（k = {gn['k']}，{gn['bins']} 档）：毛收益 {g(gn['gain']['median'])}，"
                       f"因子成本 {g(gn['cost']['median'])}，净收益 {g(gn['net']['median'])} bit/{{sym_zh}}"
                       + (f"（{gn['n']} 个条目中位数；净收益范围 {g(gn['net']['min'])} … {g(gn['net']['max'])}）"
                          if gn["n"] > 1 else "") + "。")
        return out

    def r5(self, rec):
        s = rec["summary"]
        sh = self.syn_has("R5")
        keys = (self._r5_one(sh, "(a) ") if sh else []) + self._r5_one(rec, "(b) ")
        return {
            "axes": ["横轴：奇异值序号 i（从大到小），1–" + str(s["layers"][-1]) + "。",
                     "纵轴：归一化能量 σ_i² / Σ_j σ_j²（%，对数轴），σ_i 为去均值后数值矩阵的第 i 个奇异值。"],
            "lines": ["蓝线：实测（(b) 为条目中位数，浅色阴影为最小–最大）；灰线：基线；橙色虚线：各联自己的回本线。"],
            "baseline": f"{self.base_zh(s['baseline'])}，3 次重复，两联相同。",
            "breakeven": "一层 rank-1 的因子成本 (R + W)·16 / N 换算成能量份额：1 − 2^(−2(R + W)·16 / N)。"
                         "两联矩阵大小不同，回本线不同。",
            "keys": keys + ["严格收益的做法：象限留出得到 rank-k 预测，按 16 分位分档作上下文，"
                            "留出条件熵的下降为毛收益，因子成本 k(R + W)·16 / N；详见 r5_gain.json。"]}

    def r6(self, rec):
        s = rec["summary"]
        d = rec["detail"]["exceed_k"]
        return {
            "axes": ["横轴：窗口长度 k（{sym_zh}数），标出全部计算过的 k：" + ", ".join(str(k) for k in s["ks"]) + "（对数轴）。",
                     "纵轴：重复窗口占全部窗口的比例（%，对数轴）；最下方的刻度为 0，轴断裂记号以下的点表示 0。"],
            "lines": [f"蓝线：实测（{s['n_entries']} 个条目中位数，浅色阴影为最小–最大）；灰线：基线；橙色竖虚线：回本长度。"],
            "baseline": f"{self.base_zh(s['baseline'])}，3 次重复。",
            "breakeven": f"复制一段要记位置，约 log2 N bit；长度 L* = log2 N / H 以上才划算，L* = {g(s['breakeven'], 3)}。",
            "keys": [f"最长出现重复的 k（中位数）：{g(s.get('Kstar_median'))}。",
                     f"超出基线的 k：{d}。" if d else "没有超出基线的 k。"] + self.alt_line(s, "R6")}

    def _r7(self, rec, tag):
        s = rec["summary"]
        return (f"{tag}中位数 {g(s['median'], 3)}（基线 {g(s['base_median'], 3)}）；越过回本线的比例 {pct(s['share'])}"
                f"（基线 {pct(s['base_share'])}）；回本线 {g(s['line'], 3)}；判定：{self.verdict(rec)}。")

    def r7a_partner(self, rec):
        sh = self.syn_has("R7a_partner")
        return {
            "axes": ["横轴：" + self.ctx.axes("R7a_partner", rec)[0].replace("$", "") + "，0–1 分 100 格。",
                     "纵轴：行的比例（%，对数轴）。"],
            "lines": ["蓝柱：实测直方图（(b) 为各条目归一化直方图的逐格中位数）；灰色阶梯线：基线；橙色竖虚线：回本线。",
                      "随机置换偶尔把某一行配回它自己的对应行（每次约 1 行），所以 (a) 的基线在实测峰附近有一个高约 1/(3R) 的小格（3 次重复的平均）；"
                      "这是基线做法本身的性质，不是数据的结构。"],
            "baseline": "对应行随机配对（把配对矩阵的行随机置换），3 次重复。",
            "breakeven": "用另一行预测本行、只付 16 bit 系数：|ρ| > sqrt(1 − 2^(−2·16 / W))，W 为行长。",
            "keys": ([self._r7(sh, "(a) ")] if sh else []) + [self._r7(rec, "(b) ")]}

    def r7a_row(self, rec):
        return {
            "axes": ["横轴：一行与本条目中任一其他行的最大 |相关系数|，0–1 分 100 格。", "纵轴：行的比例（%，对数轴）。"],
            "lines": ["蓝柱：实测直方图（各条目归一化直方图的逐格中位数）；灰色阶梯线：基线；橙色竖虚线：回本线。"],
            "baseline": f"{self.base_zh(rec['summary']['baseline'])}，3 次重复。",
            "breakeven": "记下参考行编号（log2 行数 bit）和 16 bit 系数：|ρ| > sqrt(1 − 2^(−2(log2(R − 1) + 16) / W))。",
            "keys": [self._r7(rec, "")]}

    def _r7b(self, rec, tag):
        s = rec["summary"]
        i = s["ks"].index(64) if 64 in s["ks"] else 0
        return (f"{tag}K = {s['ks'][i]}：失真比 {pct(s['real']['median'][i])}（基线 {pct(s['base']['median'][i])}），"
                f"回本线 {pct(s['breakeven'][i])}；超出基线的 K：{rec['detail']['exceed_K']}；判定：{self.verdict(rec)}。")

    def r7b(self, rec):
        sh = self.syn_has("R7b")
        return {
            "axes": ["横轴：原型（质心）个数 K = " + ", ".join(str(k) for k in rec["summary"]["ks"]) + "（对数轴，全部标出）。",
                     "纵轴：留出失真比（%）= 留出{unit_zh}到最近原型的平方距离 / 到拟合半的均值的平方距离。"],
            "lines": ["蓝线：实测（(b) 为条目中位数，浅色阴影为最小–最大）；灰线：基线；橙色虚线：回本线。"],
            "baseline": f"{self.base_zh(rec['summary']['baseline'])}，整条流程重复 3 次。",
            "breakeven": "每个{unit_zh}只付 log2 K bit 原型编号：失真比 < 2^(−2·log2 K / L)，L 为{unit_zh}长。"
                         "这条线不含参数位数（原型表的成本未计入），因此与 16 bit 的统一约定无关。",
            "keys": ([self._r7b(sh, "(a) ")] if sh else []) + [self._r7b(rec, "(b) ")]}

    def r8(self, rec):
        s = rec["summary"]
        sv = self.ctx.r8_s(rec)
        want = self.ctx.fig("R8").get("breakeven_bits", s.get("b"))
        line = s["breakeven"] if s.get("b") == want else s["breakeven_alt"]
        idx = [s["keys"].index(f"{k},{sv}") for k in s["ks"]]
        keys = [f"K = {k}：R² {pct(s['real']['median'][j])}（基线 {pct(s['base']['median'][j])}），回本线 {pct(line[j])}"
                for k, j in zip(s["ks"], idx)]
        return {
            "axes": [f"横轴：字典大小 K（原子数）= {', '.join(str(k) for k in s['ks'])}（对数轴）。",
                     "纵轴：留出解释方差 R²（%，0–100%）。"],
            "lines": [f"只画每{{unit_zh}} {sv} 个原子：蓝线为实测（{s['n_entries']} 个条目中位数，浅色阴影为最小–最大），"
                      "灰线为基线，橙色虚线为回本线。"],
            "baseline": f"{self.base_zh(s['baseline'])}，整条流程重复 3 次。",
            "breakeven": f"每{{unit_zh}} s 个原子，各付 log2 K bit 编号和 {want} bit 系数："
                         f"R² > 1 − 2^(−2s(log2 K + {want}) / L)。",
            "keys": ["；".join(keys) + "。",
                     f"判定沿用配置中的 b = {s.get('b')}（合成数据验收所用）；本条目在任何 (K, s) 上都"
                     + ("没有超出基线，因此在 16 bit 下判定同样为「无」。" if not rec["detail"]["exceed"]
                        else f"超出基线：{rec['detail']['exceed']}。")]}

    FIGS = {"R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5, "R6": r6, "R7a_partner": r7a_partner,
            "R7a_row": r7a_row, "R7b": r7b, "R8": r8}

    def section(self, fid, fname):
        rec = self.data(fid)
        fn = self.FIGS.get(fid)
        title = self.ctx.title(fid, rec).replace("$", "").replace("\\,", " ")
        for a, b in (("\\rho", "ρ"), ("\\sigma", "σ"), ("\\", "")):
            title = title.replace(a, b)
        out = [f"## {fid}　{title}", "", f"文件：`figures/{fname}.png`、`.pdf`。"]
        panels = self.ctx.fig(fid).get("panels")
        if panels and len(panels) > 1:
            out.append("两联：(a) 合成数据（有此结构）；(b) " + self.ctx.dataset(rec) + "。两联纵轴范围相同，横轴范围相同。")
        else:
            out.append("单联：" + self.ctx.dataset(rec) + "。")
        if fn is None or rec.get("verdict") == "not_applicable":
            out += ["", f"判定：{self.verdict(rec)}。", ""]
            return out
        d = fn(self, rec)
        out += ["", "**坐标轴**", ""] + [f"- {self.z(x)}" for x in d["axes"]]
        out += ["", "**画了什么**", ""] + [f"- {self.z(x)}" for x in d["lines"]]
        out += ["", f"**基线**：{self.z(d['baseline'])}", "", f"**回本线**：{self.z(d['breakeven'])}", ""]
        out += ["**关键数字**", ""]
        prev_table = False
        for x in d["keys"]:
            table = x.startswith("|")
            if table != prev_table:
                out.append("")
            out.append(self.z(x) if table else f"- {self.z(x)}")
            prev_table = table
        out += ["", f"**判定**：{self.verdict(rec)}。"]
        ex = self.extra_notes(fid)
        if ex:
            out += [""] + [f"> {x}" for x in ex]
        out.append("")
        return out


def write_all(man, res, syn, ctx, odir, written, skipped, overview_figs):
    from . import style
    nt = Notes(man, res, syn, ctx)
    label = man["label"]
    lines = [f"# 图的说明：{label}", "",
             "每张图一节：标题、坐标轴、画了什么、基线、回本线（每个参数 16 bit）、关键数字和判定。"
             "多条目时线和柱为各条目的中位数，浅色阴影或误差线为最小–最大。", "",
             "判定规则：某点真实值与基线 3 次重复均值之差大于 5 倍标准差且大于最小幅度，即为“超出基线”；"
             "曲线类图每点的标准差取该点与整条曲线中位数中较大的一个；多条目时先对真实值、基线均值和标准差各取中位数。"
             "某点既超出基线、又越过回本线，即为“过回本线”。", ""]
    by_file = {style.file_name(f): f for f in man["figures"] if f != "R10"}
    for fname in written:
        lines += nt.section(by_file[fname], fname)
    if skipped:
        lines += ["## 未出图", "", "按配置不出图（结果文件保留）：" + "，".join(f"`{f}`" for f in skipped) + "。", ""]
    with open(odir / "figure_notes.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # overview.md
    hdr = "| 规律 | 图 | 指标 | 判定 | 关键数字 | 其他基线下的判定 |"
    rows = [f"# 判定总表：{label}", "", hdr, "|" + " --- |" * 6]
    for fid in man["figures"]:
        rec = nt.data(fid)
        pat = rec["pattern"]
        name = style.file_name(fid) if fid != "R10" else "—"
        fig_cell = f"`{name}`" if name in written else ("不出图" if fid != "R10" else "—")
        alt = "；".join(f"{nt.base_zh(b).split('（')[0]}：{nt.zh['verdict'][a['verdict'] if a['verdict'] != 'present' else 'present_below']}"
                       for b, a in (rec.get("summary", {}).get("alt") or {}).items()) or ""
        ind = ctx.t(INDICATOR_ZH.get("R9" if fid.startswith("R9") else fid, ""))
        esc = lambda t: t.replace("|", "\\|")
        rows.append(f"| {pat} {PATTERN_ZH.get(pat, '')} | {fig_cell} | {esc(ind)} | {nt.verdict(rec)} | "
                    f"{esc(rec.get('key', ''))} | {alt} |")
    rows += [""] + [f"- {fid}：" + x for fid in ("R3", "R4", "R6") if fid in man["figures"]
                    for x in nt.extra_notes(fid)]
    rows += ["", f"总览图 `overview.png` 的各联：{', '.join(overview_figs)}。", ""]
    with open(odir / "overview.md", "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    # r5_gain.json
    r5 = nt.data("R5")
    if r5 and r5.get("summary", {}).get("gain"):
        out = {"method": "spec 003 R5.3: quadrant hold-out rank-k prediction of the mean-removed values; the "
                         "prediction cut into quantile bins is a context next to the known label; gain = held-out "
                         "H(symbol | known) - H(symbol | known, bin), symbols with equivalent symbols merged, halves = "
                         "even / odd rows; factor cost = k (R + W) b / N; net = gain - cost; bit per symbol",
               "data": {"label": label, "summary": r5["summary"]["gain"],
                        "entries": {n: e.get("gain") for n, e in r5["entries"].items() if e.get("gain")}}}
        for side in ("has", "none"):
            srec = (syn.get("R5") or {}).get(side)
            if srec and srec["summary"].get("gain"):
                out[f"synthetic_{side}"] = {"description": srec.get("description", ""),
                                            "gain": list(srec["entries"].values())[0]["gain"]}
        with open(odir / "r5_gain.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
            f.write("\n")
