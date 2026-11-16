# redundancy：可移植的冗余分析框架

规格：[docs/007-指标图与框架.md](../docs/007-指标图与框架.md)；图在第二版中按“修订 007 的指标图”的要求改过（结果见 [experiments/007-figures-v2/](../experiments/007-figures-v2/summary.md)）。十种规律（R1–R10）各一张图，每张可含合成数据（无 / 有此规律）和被测数据几联，带基线和回本线。图中全为英文；判定（无 / 存在，未过回本线 / 存在，过回本线，R1、R9 只有“存在 / 无”）和关键数字不画在图上，写进 `figure_notes.md` 和 `overview.md`。

只用 NumPy 与 Matplotlib。不 import `fp4_redundancy`；只有 `adapters/fp4.py` 调用 `diagnostics/` 中读取权重文件的代码。

## 两条命令

```bash
python -m redundancy.run  --config redundancy/configs/<名称>.json --out <结果目录>
python -m redundancy.plot --results <结果目录> --out <图目录>
```

在仓库根目录运行。`run` 写结果文件：`data/<图>.json`、`synthetic/<规律>_{none,has}.json`、`manifest.json`。`plot` 只读这些文件，写 `<图名>.png/.pdf`（300 dpi；单联 6.5 × 4.5 英寸），以及 `--overview` 目录中的 `overview.png/.pdf`、`overview.md`（判定总表）、`figure_notes.md`（每张图的中文说明）、`r5_gain.json`（R5 算了严格收益时）和 `figure_check.json`（自动版面检查：图例是否压住数据、文字是否出界、字号、中文与变量名）。

- 结果目录里没有合成数据时（配置里没有 `synthetic` 一节），可用 `--synthetic <另一个结果目录>` 借用那里的合成面板。
- 图例位置自动选择：依次试各个位置，压住任何线、点、阴影或柱就换位置；都不行时加大纵轴上方余量（直方图还可加宽横轴）；仍不行才放到坐标区外。图例条目和标题不折行，过长时在配置里改短；`figure_check.json` 会列出折行或宽于坐标区的标题。

## 三层

| 层 | 文件 | 只能做什么 |
| --- | --- | --- |
| 适配器 | `adapters/*.py` | 把一种数据变成 `dataset.Entry` |
| 指标 | `indicators/r1.py … r10.py`，以及 `baselines.py`、`breakeven.py` | 只认 `Entry`，写结果文件；不含任何数据格式的词 |
| 画图 | `plot.py`、`notes.py`、`style.py`、`labels.json` | 只读结果文件和配置中的 `figures` 一节 |

`tests/test_isolation.py` 检查两件事：指标层、画图层（`plot.py`、`notes.py`、`style.py`、`labels.json`）、`baselines.py`、`breakeven.py` 中不出现 FP4 专有的词；画图层不 import 适配器或指标。

## 为新数据写适配器

适配器是一个类：构造时接收配置中 `adapter` 一节，提供 `entries() -> [名称]` 和 `load(名称) -> Entry`。`load` 在工作进程中执行，要能独立运行。在 `adapters/__init__.py` 的 `get` 里登记一个 `type` 名。

`Entry` 的字段（`dataset.py`）：

| 字段 | 必需 | 含义 | 缺失时 |
| --- | --- | --- | --- |
| `symbols` | 是 | 行 × 列的整数，0..A−1，A ≤ 256 | — |
| `alphabet`、`raw_bits` | 是 | 符号名称；原始存储中每个符号占的 bit 数 | — |
| `values` | 否 | 每个符号的数值 | R4、R5、R7a 行与 partner、R7b、R8 标为不适用 |
| `groups` | 否 | 名称 → `Group(labels, known, coverage, scope)`。`scope="entry"` 表示整个条目共用一个标签，在条目之间合并比较 | R2 不适用 |
| `unit_len` | 否 | 每行按此长度切成局部单位 | “单位内打乱”退化为整体打乱；R7a 短段、R7b、R8、R9 不适用 |
| `known` | 否 | 已经记过账的分组，每个单位一个标签（无单位时每个符号一个） | “已知分组内打乱”退化为整体打乱 |
| `partner` | 否 | 同形状的另一组数值：第 i 行是本条目第 i 行的候选 | R7a partner 图不适用 |
| `equivalences` | 否 | `{"name", "symbol_classes": [[…]]}`（等价符号，计数由数据得出）或 `{"name", "log2_count", "n_symbols"}` | R10 不适用 |
| `process_checks` | 否 | R9 假设：`kind` 为 `count_in_unit`（`symbols`）或 `max_rank_in_unit`（`rank`），`condition_on_known` | R9 不适用 |
| `baselines` | 否 | 某个指标改用的专用基线，如 `{"R5": "random_sign"}` | 用通用基线 |

如果数据能存成 `.npy` 二维数组，就不需要写适配器，直接用 `array_file`：

```json
{"name": "my_audio", "label": "音频 10 段", "seed": 1, "workers": 8,
 "adapter": {"type": "array_file", "files": ["a.npy", "b.npy"], "bins": 64, "unit_len": 32, "axis": 1},
 "indicators": {"R1": {}, "R2": {}, "R3": {}, "R4": {}, "R5": {}, "R6": {}, "R7a": {}, "R7b": {}, "R8": {}, "R9": {}, "R10": {}}}
```

- 整数数组直接作符号。取值超过 256 种时，按 `int_mode` 等频分档（`quantile`），或取高位（`high_bits`）。
- 浮点数组按 `bins` 等频分档，原值保留作 `values`。
- `array_file` 不提供 `known`、`partner`、`equivalences`、`process_checks`，相应的图标为“不适用”。

一个数据集内的条目应使用同一套字母表；不同时，R1、R2 按最长的字母表补零，并在结果里注明。

## 配置

`configs/fp4.json` 是 FP4 的配置，`configs/synthetic.json` 只算合成数据。字段如下：

- `seed`：全部随机数的种子。同一配置连跑两次，结果文件逐字节相同；`run` 把数值库固定为单线程。
- `workers`：并行进程数。
- `synthetic`：合成数据的形状与种子。省略这一节则不算合成面板。`synthetic.indicators.<Rk>` 只对合成数据覆盖指标设置（FP4 配置用它让合成数据的 R8 按 b = 8 验收，被测数据按 16 bit）。
- `indicators.<Rk>`：各指标的参数、最小幅度（`min_amp*`）、参数位数 `b`、基线名 `baseline`（`whole` / `known` / `unit` / `random_sign`）。R7a 可用 `baseline_row` 给“矩阵内行”单独指定基线。R3、R4、R6 可加 `alt_baselines`：另算的基线只存进结果文件（含它下的判定），不参与判定、不画。R5 可加 `gain: {"k", "bins"}`，按 003 R5.3 的方法算严格收益。
- `figures`（只给画图层用，可省略）：
  - `dataset_label`：被测数据一联的标题，`{n}` 换成条目数；各图也可单独给。
  - `vocab`：图中的通用词，如 `sym`（符号叫什么）、`var`（符号变量的字母）、`unit`（局部单位叫什么）、`known`（已知分组叫什么），以及 `*_zh`（中文说明用）。
  - `<图>.panels`：保留哪几联，取 `synthetic_none`、`synthetic_has`、`data`；空列表表示不出图。省略时画出全部可用的联。
  - `<图>.title`、`xlabel`、`synthetic`（合成一联的结构描述，标题为 “Synthetic: <描述>”）、`baseline_legend`（图例改用另一种基线的名称，如取样已使两种打乱等价时）、`unit_line`（R3、R4 在单位长度处画点线）、`groups`（R2 保留的分组及显示名）、`s` 与 `breakeven_bits`（R8）、`note_zh`（写进说明的补充）。
  - `overview`：总览图的各联。

## 判定规则（写在 `indicators/common.py`）

- **超出基线**：真实值与基线 3 次重复的均值之差，大于基线标准差的 5 倍，并且大于配置中的最小幅度。
  - 曲线类图（R3–R6、R7b、R8）每点的标准差取“该点的标准差”与“整条曲线的标准差中位数”中较大的一个。3 次重复的逐点标准差只有 2 个自由度，长曲线上总有个别点被低估。
  - 多个条目时，对各条目的真实值、基线均值和标准差分别取中位数后再判定。
- **过回本线**：某一点既超出基线、又越过回本线。
- R1：`raw_bits − H` 大于最小幅度即为存在。
- R9：实际与预期分布的总变差距离 > 0.05 即为存在。

## 测试

```bash
python -m unittest redundancy.tests.test_isolation redundancy.tests.test_portable redundancy.tests.test_reproducible redundancy.tests.test_synthetic
python -m redundancy.tests.acceptance --results <FP4 结果> --rerun <第二次 FP4 结果> --work <临时目录> --out <validation.json>
```

`acceptance` 依次做规格第 7 节的 5 项检查。其中与 001–006 结果的逐张比较只适用于 FP4。
