# R1–R4 格式事实（spec 001 第 2 节）

快照：`2cba9e42aa026125f3ed06c6d98c1db82f7ca027`（DeepseekV41ForCausalLM）；由 `diagnostics/format_facts.py` 生成，只读文件头、`config.json` 与参考代码文本，不执行快照代码。

## F1 清单

自行解析 48 个分片的 safetensors 文件头（8 字节小端长度 + JSON），共 96085 个张量。

| dtype | 张量数 | 字节数 |
| --- | ---: | ---: |
| BF16 | 520 | 3,952,883,712 |
| F32 | 387 | 169,229,128 |
| F8_E4M3 | 357 | 204,015,223,296 |
| F8_E8M0 | 47589 | 23,563,015,184 |
| I8 | 47232 | 278,585,671,680 |

**FP4 权重识别规则**：文件头 dtype 为 `I8`，且名称匹配 `^(layers|mtp).<i>.ffn.experts.<e>.w{1,2,3}.weight$`；对应 scale 为同名把 `weight` 换成 `scale` 的张量，dtype `F8_E8M0`，形状 `[行数, 2·字节列数/32]`。
- 全部 47232 个 I8 张量中，不符合名称规则的 0 个，scale 缺失/dtype/形状不符的 0 个。按前缀：{'layers': 46080, 'mtp': 1152}（`layers` = 主干路由专家，`mtp` = MTP 层路由专家）。
- I8 形状（字节）× 角色：[2304, 2560] w1: 15744；[2304, 2560] w3: 15744；[5120, 1152] w2: 15744
- `F8_E8M0` 张量按同名 `.weight` 的 dtype 归类：{'F8_E4M3': 357, 'I8': 47232}。I8 的 scale 即上述规则对应者，其余 `F8_E8M0` 属于 `F8_E4M3` 权重（共享专家等），不在本规格范围。
- 依据：`inference/convert.py:174` — `elif "experts" in name and state_dicts[i][name].dtype == torch.int8:`；`inference/convert.py:181` — `state_dicts[i][name] = state_dicts[i][name].view(torch.float4_e2m1fn_x2)`（路由专家的 int8 张量按 FP4×2 解释）；`inference/convert.py:176` — `scale_name = name.replace("weight", "scale")`（scale 为同名 `.scale`）；`config.json` `quantization_config`: `{"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"}`。

## F2 参考代码

1. **字节内码字顺序：低 4 位在前。** `inference/convert.py:32` — `low = x & 0x0F`；`inference/convert.py:33` — `high = (x >> 4) & 0x0F`；`inference/convert.py:34` — `x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)`——`stack(..., dim=-1).flatten` 使每个字节展开为 `[low, high]`，即逻辑列 `2k` 为低 nibble、`2k+1` 为高 nibble。打包沿最后一维（K）：`inference/kernel.py:494` — `The FP4 values are packed along the K (last) dimension.`；`inference/model.py:220` — `# two values per byte: [out, in] logically, [out, in//2] stored`。
2. **码字→数值表（E2M1）**：`inference/convert.py:14` — `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32`（`FP4_TABLE`，起于第 13 行）。索引 0–7 = {0, 0.5, 1, 1.5, 2, 3, 4, 6}，8–15 为其相反数，故最高位（bit 3）为符号位，低 3 位为幅值索引；码字 8 为 −0（数值 0）。kernel 中的 dtype 名称：`inference/kernel.py:15` — `FP4 = "float4_e2m1fn"`。
3. **scale**：块大小 B = 32，沿行（K，最后一维）：`inference/model.py:28` — `fp4_block_size = 32  # one fp4 scale per 32 elements along K`；`inference/model.py:223` — `torch.empty(out_features, in_features // fp4_block_size, dtype=torch.float8_e8m0fnu)`；`inference/convert.py:29` — `assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size`；`inference/kernel.py:491` — `Weight: 1x32 quant on K (reduce dim), FP4 with E8M0 scale`。字节→数值：E8M0（`float8_e8m0fnu`，`inference/model.py:30` — `scale_dtype = torch.float8_e8m0fnu`），纯 8 位指数、偏置 127，值 = 2^(byte−127)；代码中同一偏置见 `inference/kernel.py:32` — `bits_x = (x + 127) << 23`（构造 2^x 的 fp32 位模式）。e8m0 的 NaN 编码（0xFF）由 PyTorch dtype 定义，快照代码中未出现。反量化：权重 = FP4_TABLE[C]·2^(E−127)，见 `inference/convert.py:51` — `offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(fp4_block_size, dim=-1)`。
4. **量化过程**：快照中**未找到权重**的 FP4 量化代码（权重以已量化形式发布）。仅找到激活/KV 用的 `fp4_quant_kernel`：scale = 2^ceil(log2(amax/6))，`inference/kernel.py:166` — `s_local[i] = fast_round_scale(amax_local[i], fp4_max_inv)`，`inference/kernel.py:22` — `def fast_log2_ceil(x):`；取值 `T.Cast(FP4, T.clamp(x/s, -6, 6))`，`inference/kernel.py:171` — `T.Cast(compute_dtype, T.Cast(FP4, T.clamp(x_local[i, j] / s_local[i], -fp4_max, fp4_max))) * s_local[i],`；舍入规则由 FP4 类型转换决定，代码中未写明（未找到）。该过程是否用于权重未知。

## F3 数据侧核对（样本第一张矩阵 `layers.0.ffn.experts.164.w1.weight`）

1. **形状一致性**：码字矩阵 2304×5120（字节 [2304, 2560]），scale [2304, 160]；`[行数, 每行码字数/B]` = [2304, 160] → 一致。
2. **块对齐**（每种假设只用完整块）：

| 假设 | 块数 | 块内最大幅值索引 ≥ 6 的比例 | I(C; 块id) bit | n | 解析偏差 dof/(2n ln2) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 沿行、偏移 0 | 368,640 | 1.0000 | 0.54355 | 11,796,480 | 0.33813 |
| 沿行、偏移 B/2 | 366,336 | 0.9577 | 0.45757 | 11,722,752 | 0.33813 |
| 沿列、偏移 0 | 368,640 | 0.9861 | 0.38864 | 11,796,480 | 0.33813 |

   ≥6 比例最高的假设：沿行、偏移 0；I(C; 块id) 最大的假设：沿行、偏移 0。三种假设的块大小和组数相近，解析偏差相同量级（每组仅 32 个样本时该公式偏小，仅作量级参考），差别如上表如实记录。
3. **符号对称**：

| m | hist[m]（s=0） | hist[m^8]（s=1） | 比值 s=1/s=0 |
| ---: | ---: | ---: | ---: |
| 0 | 696,557 | 696,350 | 0.9997 |
| 1 | 873,639 | 874,910 | 1.0015 |
| 2 | 1,207,730 | 1,205,051 | 0.9978 |
| 3 | 604,200 | 602,868 | 0.9978 |
| 4 | 994,005 | 992,943 | 0.9989 |
| 5 | 715,593 | 711,294 | 0.9940 |
| 6 | 572,499 | 571,457 | 0.9982 |
| 7 | 238,638 | 238,746 | 1.0005 |

   H(s) = 1.000000 bit，I(s;m) = 0.000001 bit，H(s|m) = 0.999999 bit（N = 11,796,480）。

## F4 停止条件

- 块沿行：F2.3 + F3.2 → 是。B = 32：F2.3 → 是。数值表为 E2M1 且最高位为符号：F2.2 → 是。scale 形状：F3.1 → 一致。
- 结论：未触发停止条件，继续第 3 节之后的测量。
- 字节内顺序只由 F2.1 确定（低 nibble 在前）；C1 的 d ≤ 4 奇偶拆分作为旁证，见 summary.md。
