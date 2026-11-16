# 被测数据总览: FP4 原样本 36 张

| 规律 | 图 | 指标 | 判定 | 关键数字 |
| --- | --- | --- | --- | --- |
| R1 | `R1_distribution` | 各符号占比 | 存在 | H = 3.894 bit, saving = 0.106 |
| R2 | `R2_groups` | I(符号; 分组) − 打乱基线 | 存在，过回本线 | best = scale: net MI = 0.1269, cost = 6.1e-05 |
| R3 | `R3_mi_distance` | I(x_j; x_j+d) | 无 | MI(d=1) = 0.02621, baseline = 0.02625, break-even = 0.000326 |
| R4 | `R4_autocorr` | 自相关 ρ(d) | 无 | ρ(1) = 0.000, \|x\| ρ(1) = 0.105, bound = 0.000174 bit |
| R5 | `R5_spectrum` | 奇异值能量份额 | 存在，未过回本线 | top share = 0.002474 (baseline 0.001217), break-even = 0.0139 |
| R6 | `R6_repeats` | 长度 k 窗口的重复比例 | 无 | K* = 12, L* = 6.2 |
| R7 | `R7a_neighbors_partner` | 对应行的 \|ρ\| | 存在，过回本线 | median \|ρ\| = 0.152 (baseline 0.010), ρ* = 0.066 |
| R7 | `R7a_neighbors_row` | 最近邻行的 \|ρ\| | 存在，过回本线 | median max\|ρ\| = 0.078 (baseline 0.051), ρ* = 0.086 |
| R7 | `R7a_neighbors_unit` | 最近邻的相同位置数 | 无 | median best = 12 (baseline 12), m* = 13 of 32 |
| R7 | `R7b_clusters` | k-means 留出失真比 | 无 | ratio(K=64) = 0.831 (baseline 0.831), break-even 0.771 |
| R8 | `R8_dictionary` | 字典留出 R² | 无 | R2(K=64, s=2) = 0.356 (baseline 0.356), break-even 0.703 (b = 8) |
| R9 | `R9_max_level` | 实际与预期分布的总变差 | 存在 | TV = 0.114; share at 0: 0.000 (expected 3.56e-24) |
| R9 | `R9_odd_count` | 实际与预期分布的总变差 | 存在 | TV = 0.444; share at 0: 0.301 (expected 0.0097) |
| R10 | `R1_distribution` | 等价存法个数 | 存在 | 3 equivalences, 0.1176 bit/symbol in total |
