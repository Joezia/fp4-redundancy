# 判定总表：Kimi-K3 FP4 weights (spec 009 sample)

| 规律 | 图 | 指标 | 判定 | 关键数字 | 其他基线下的判定 |
| --- | --- | --- | --- | --- | --- |
| R1 符号概率不均匀 | `R1_distribution` | 各码字的占比 | 存在 | H = 3.755 bit, saving = 0.245 |  |
| R2 不同分组的分布不同 | `R2_groups` | I(码字; 分组) − 完全打乱基线 | 存在，过回本线 | best = block_max_level: net MI = 0.06476, cost = 0.0442 |  |
| R3 与邻近符号有关系 | `R3_mi_distance` | I(C_j; C_{j+d}) | 存在，过回本线 | MI(d=1) = 0.005896, baseline = 1.457e-05, break-even = 0.000349 | 块内打乱：无 |
| R4 与邻近数值有线性关系 | `R4_autocorr` | 自相关 ρ(d) | 存在，未过回本线 | ρ(1) = 0.000, \|x\| ρ(1) = 0.081, bound = 0.000122 bit | 块内打乱：无 |
| R5 各行是少数图案的混合 | `R5_spectrum` | 奇异值能量份额 | 无 | top share = 0.00139 (baseline 0.001208), break-even = 0.0133 |  |
| R6 一段内容原样再现 | `R6_repeats` | 长度 k 窗口的重复比例 | 存在，过回本线 | K* = 12, L* = 6.4 | 块内打乱：无 |
| R7 段与段相似 | 不出图 | 与最相似的另一块的相同位置数 | 无 | median best = 12 (baseline 12), m* = 13 of 32 |  |
| R7 段与段相似 | `R7a_neighbors_row` | 与最相关的另一行的 \|ρ\| | 存在，未过回本线 | median max\|ρ\| = 0.066 (baseline 0.062), ρ* = 0.103 |  |
| R7 段与段相似 | `R7a_neighbors_partner` | 与对应行的 \|ρ\| | 存在，过回本线 | median \|ρ\| = 0.115 (baseline 0.012), ρ* = 0.079 |  |
| R7 段与段相似 | `R7b_clusters` | k-means 留出失真比 | 无 | ratio(K=64) = 0.832 (baseline 0.832), break-even 0.771 |  |
| R8 各段只用字典里少数原子 | `R8_dictionary` | 字典学习留出 R² | 无 | R2(K=64, s=2) = 0.355 (baseline 0.355), break-even 0.851 (b = 16) |  |
| R9 产生过程的约束 | 不出图 | 实际与预期分布的总变差 | 无 | TV = 0.033; share at 0: 0.000 (expected 3.46e-23) |  |
| R9 产生过程的约束 | 不出图 | 实际与预期分布的总变差 | 无 | TV = 0.004; share at 0: 0.000 (expected 1.34e-05) |  |
| R10 等价存法 | — | 等价存法个数 | 存在 | 3 equivalences, 0.1164 bit/symbol in total |  |

- R3：相对完全打乱存在的依赖，来自同一块 32 个码字共享同一个 scale：块内打乱（保留每块的码字集合）后消失。它与 R2 中 scale 分组的收益是同一来源，二者不能相加。
- R4：相对完全打乱存在的相关（主要是 |v|），来自同一块共享同一个 scale：块内打乱后消失。它与 R2 中 scale 分组的收益是同一来源，二者不能相加。
- R7b：取样只取同一 scale 值（每张矩阵中出现最多的那个）的块，在这些块上“同 scale 的块之间打乱”与完全打乱等价，因此图例统一写 “Shuffled baseline”。
- R8：取样只取同一 scale 值（每张矩阵中出现最多的那个）的块，在这些块上“同 scale 的块之间打乱”与完全打乱等价，因此图例统一写 “Shuffled baseline”。

总览图 `overview.png` 的各联：R1, R2, R3, R4, R5, R6, R7a_partner, R7a_row, R7b, R8。

