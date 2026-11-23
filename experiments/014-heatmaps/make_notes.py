"""Spec 014: write figure_notes.md from results/stats.json and the files actually present in figures/."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE / "results/stats.json"))
CELL = S["preview_cell"]

KINDS = [
    ("full_<variant>", "全图（原始分辨率）", """每个元素正好一个像素，行列按存储形状（不转置），行号向下增大、列号向右增大。矩阵外有白边：上方是标题和色标说明，左边竖排写行的含义，下方写列的含义，右侧是色标条。
- `code_signed`：FP4 码值 v，发散色图 RdBu_r，以 0 为中心对称，上下限为 ±（|v| 的 99.5 百分位）。所有矩阵的这个值都是 6，即码值的最大值，所以实际没有截断。
- `code_abs`：|v|，顺序色图 viridis，范围 0 到 6。
- `deq_signed`：反量化值 w = v·2^(E−127)（dequantized, exact in BF16），RdBu_r，上下限为 ±（|w| 的 99.5 百分位，所有矩阵都是 0.0625），超出部分截断成端点颜色。截断比例见文末表。
- `deq_abs`：|w|，viridis，0 到 0.0625，超出部分截断。
- `deq_log`：|w| 的对数色标，viridis，范围是 [最小非零 |w|, 最大 |w|]，不截断；0 单独画成灰色 (128,128,128)。0 的比例见文末表。
- 色标量化为 253 级（调色板 PNG）。码值只有 15 个取值，所以每个码值都有自己的颜色。
- 存放位置：DeepSeek L20.E86.w1、L20.E86.w2 和 Kimi-K3 L46.E484.w1 三个矩阵的全图在 `figures/` 中（用户要求只推送 3 个矩阵的全图）；其余 15 个矩阵的全图在本地 `data/014/full/<模型>/<矩阵>/`，不在仓库里。"""),
    ("preview_<code|deq>_<linear|log>", "缩小版预览", f"""每个 {CELL}×{CELL} 格子（{CELL} 行 × {CELL} 列）画一个值，即格内 |值| 的平均，所以带符号和幅值两种画法共用这张预览。坐标轴仍以元素为单位。
- `linear`：viridis，范围是 0 到（真实矩阵格均值的 99.5 百分位），超出部分截断。
- `log`（只画反量化值）：对数色标，范围是格均值的 [最小非零, 最大]。
各矩阵的范围见 `results/stats.json` 的 `preview_scales`。"""),
    ("zoom_<variant>_<topleft|centre>", "局部放大 256×256", """`topleft` 是第 0–255 行、第 0–255 列；`centre` 从（R/2−128, W/2−128）开始，该位置在所有形状下都是 32 的倍数。元素一一对应，每个元素画成一个小方格，坐标是矩阵中的绝对行号和列号。
- 黑色竖线是每 32 列一个的 scale 块边界。
- 色标与同一变体的全图相同（带符号、幅值、对数），对数版中 0 为灰色。"""),
    ("scale_map", "块 scale 图", """把块 scale 的指数 E−127 排成 R ×（W/32）的矩阵：横轴是沿列的块序号（每块覆盖 32 列），纵轴是行。色图 viridis，范围是该矩阵 E−127 的最小值到最大值（见文末表）。这张图没有截断，也没有打乱基线。"""),
    ("shuffle_<variant>", "打乱对照", f"""把同一矩阵的全部元素用固定种子 {S['seed']} 随机置换（码值与反量化值用同一个置换，scale 跟着元素一起移动），然后与真实矩阵并排画。
- 上排是 {CELL}×{CELL} 格均值 |值| 的预览：左边真实、右边打乱，两边共用真实矩阵的色标（对数变体用对数色标）。
- 下排是左上角 256×256：左边真实、右边打乱，色标与该变体的全图相同。
- 打乱后只剩下元素取值的分布，没有任何位置结构，所以它就是“没有结构”时的样子。"""),
    ("rms_<code|deq>", "行 / 列 RMS 曲线", """上图是每行的 RMS（对该行全部元素求 √(mean x²)），横轴为行号；下图是每列的 RMS，横轴为列号。
- 蓝线是真实矩阵，橙线是打乱后的矩阵；打乱后的曲线只剩抽样波动，作为基线。
- 图例里写着各自的变异系数（标准差 / 均值）。"""),
    ("autocorr_<code|deq>", "|值| 的二维自相关", """纵轴是 |值| 与位移后的 |值| 之间的皮尔逊相关系数，横轴是位移 d = 1, 2, 4, 8, 16, 32, 64（以 2 为底的对数刻度）。
- 圆点（“shift d rows”）是同一列内相隔 d 行的两个元素；方块（“shift d columns”）是同一行内相隔 d 列的两个元素。
- 实线是真实矩阵，虚线是打乱基线。"""),
]


def main():
    L = ["# 014 图注", "",
         "每个矩阵的图在 `figures/<模型>/<矩阵名>/`，模型是 `deepseek` 或 `kimi`。每张图都同时生成过 PNG 和 PDF，按用户要求只保留较小的一个：全图都是 PNG，其余图都是 PDF。",
         "所有图都按存储形状画，行列不转置。w1、w3 的行是神经元、列是隐藏维；w2 的行是隐藏维、列是神经元。每张图的坐标轴标签都写了含义和长度。",
         "“variant”是 `code_signed`、`code_abs`、`deq_signed`、`deq_abs`、`deq_log` 之一。", ""]
    for i, (pat, title, body) in enumerate(KINDS, 1):
        L += [f"## {i}. {title}（`{pat}`）", "", body, ""]
    L += ["## 各矩阵的色标值与截断", "",
          "| 模型 | 矩阵 | 形状 | 码值 p99.5 | 反量化 p99.5（截断值） | 反量化超出截断的比例 | 对数色标范围 | |w| = 0 的比例 | E−127 范围 | 全图位置 |",
          "|---|---|---|---:|---:|---:|---|---:|---|---|"]
    for r in S["matrices"]:
        v = r["variants"]
        beyond = v["deq_signed"]["note"].split(";")[1].strip().split(" ")[0]
        zero = v["deq_log"]["note"].split("grey (")[1].split(" ")[0]
        L.append(f"| {r['model']} | {r['matrix']} | {r['shape'][0]} × {r['shape'][1]} | {v['code_signed']['hi']:g} | "
                 f"{v['deq_signed']['hi']:g} | {beyond} | [{v['deq_log']['lo']:.4g}, {v['deq_log']['hi']:.4g}] | {zero} | "
                 f"{r['scale_exponent_range'][0]} … {r['scale_exponent_range'][1]} | `{r['full_resolution_dir']}` |")
    (HERE / "figure_notes.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-20:]))


if __name__ == "__main__":
    main()
