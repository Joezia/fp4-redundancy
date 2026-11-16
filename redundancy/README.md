# redundancy：可移植的冗余分析框架

规格：[docs/007-指标图与框架.md](../docs/007-指标图与框架.md)。十种规律（R1–R10）各一张三联图：合成数据（无此规律）、合成数据（有此规律）、被测数据。每张图带参照线和回本线，并在右上角给出固定规则的判定：无 / 存在，未过回本线 / 存在，过回本线（R1、R9 只有“存在 / 无”）。

只用 NumPy 与 Matplotlib。不 import `fp4_redundancy`；只有 `adapters/fp4.py` 调用 `diagnostics/` 中读取权重文件的代码。

## 两条命令

```bash
python -m redundancy.run  --config redundancy/configs/<名称>.json --out <结果目录>
python -m redundancy.plot --results <结果目录> --out <图目录>
```

在仓库根目录运行。`run` 写结果文件：`data/<图>.json`、`synthetic/<规律>_{none,has}.json`、`manifest.json`。`plot` 只读这些文件，写 `<图名>.png/.pdf`，以及 `overview.png`、`overview.md`。

- `plot` 默认使用中文标签（系统里有 CJK 字体时）。用了哪种语言写在 `plot_log.txt` 里。`--lang en` 强制使用英文。
- 结果目录里没有合成数据时（配置里没有 `synthetic` 一节），可用 `--synthetic <另一个结果目录>` 借用那里的合成面板。

## 三层

| 层 | 文件 | 只能做什么 |
| --- | --- | --- |
| 适配器 | `adapters/*.py` | 把一种数据变成 `dataset.Entry` |
| 指标 | `indicators/r1.py … r10.py`，以及 `baselines.py`、`breakeven.py` | 只认 `Entry`，写结果文件；不含任何数据格式的词 |
| 画图 | `plot.py`、`style.py`、`labels.json` | 只读结果文件 |

`tests/test_isolation.py` 检查两件事：指标层、`plot.py`、`baselines.py`、`breakeven.py` 中不出现 FP4 专有的词；`plot.py` 不 import 适配器或指标。

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
- `synthetic`：合成数据的形状与种子。省略这一节则不算合成面板。
- `indicators.<Rk>`：各指标的参数、最小幅度（`min_amp*`）、参数位数 `b`、基线名（`whole` / `known` / `unit`）。

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
