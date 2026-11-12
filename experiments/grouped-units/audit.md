# 联合 word 扩展支持审计

审计和实现范围独立于研究价值；所有非等价候选均已接入，并通过微小合成输入检查。
32-bit rANS 受65536个已出现符号的实现上限约束，遇到时记录resource_limited；ADPCM32需要重新定义步长/样本范围，列为待复核；JPEG-LS32/64超出CharLS原生精度。
不执行同一字节序列的类型别名配置；Inter-frame仅关联Motion Compensation。原版、原结果及4bit行为保留。

| 方法 | 审计 | unit_bits | 实现/检查 | 具体范围与适配 |
| --- | --- | --- | --- | --- |
| adpcm_like_fp4 | 可直接扩展 | 4,8,16 | implemented_and_synthetic_verified | 8/16-bit word减去中心映射至IMA有符号16-bit范围，输入不截断；保留原饱和重建反馈及精确模补偿。32-bit不执行，需另定义自适应范围。 |
| aq_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| arithmetic | 可直接扩展 | 4,8,16,32 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 32-bit稀疏观察字母表，24-bit CDF；64仅作所选精确整数方法的匹配对照。 |
| bitplane | 可直接扩展 | 4,8,16,32 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| block_dedup | 等价无变化 | 4 | equivalent_reference | 原literal已packed；固定128原始bytes边界时，4/8/16-bit表示的相等关系、引用和大小不变，不另跑。 |
| bwt_mtf | 可直接扩展 | 4,8,16 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 unsigned字典序，初始MTF为0..2^b−1；内部65536word块，覆盖65536*b/4个原码字。 |
| cabac | 可直接扩展 | 4,8,16 | implemented_and_synthetic_verified | 内部逐bit概率构造整word CDF；宽度改变previous-word/前缀/深度特征或状态更新边界，是真实新配置，不是只加标签。 |
| cavlc | 可直接扩展 | 4,8,16 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| cdc_dedup | 需要格式适配 | 4,8 | implemented_and_synthetic_verified | 原Gear按nibble更新。新增真正packed字节Gear，保持32/128/512原始bytes边界尺度，记录必要mask适配；16/32仅再序列化等价。 |
| cdf53 | 可直接扩展 | 4,8,16,32 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| circulant_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| cp_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| ctw | 可直接扩展 | 4,8,16 | implemented_and_synthetic_verified | 三阶word上下文，覆盖3*unit_bits/4个历史FP4权重；完整终端域、escape/KT先验同步扩至65536。 |
| dct_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| delta | 可直接扩展 | 4,8,16,32,64 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| dpcm_lossless | 可直接扩展 | 4,8,16,32 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| dst_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| fpc_like_fp4 | 需要格式适配 | 4,8,16,32,64 | implemented_and_synthetic_verified | 8/16保留原前缀；32/64新格式v2，保留FCM/DFCM/XOR，64-bit字节数类7代表8（7字节残差补足8），每word前缀4bit。 |
| golomb_rice | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 重新选k<=unit_bits；128-bit成本计数，先算完整位数再有界分配，按实际word宽度检查商。 |
| haar_lifting | 可直接扩展 | 4,8,16,32 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| huffman | 可直接扩展 | 4,8,16,32 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| inter_frame_reference | 等价无变化 | 关联motion | equivalent_reference | 关联motion_compensation的同unit_bits配置，不重复编码或统计。 |
| intra_directional | 可直接扩展 | 4,8,16,32 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| jpeg_ls_like_fp4 | 可直接扩展 | 4,8,16 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| kronecker_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| ksvd_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| lpc_lossless | 可直接扩展 | 4,8,16,32 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| ltp_lossless | 可直接扩展 | 4,8,16,32 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| lz77 | 需要格式适配 | 4,8,16 | implemented_and_synthetic_verified | 窗口原始字节跨度保持：LZ77约32767.5bytes，LZSS2048bytes；最短匹配仍3word，记录原权重跨度。 |
| lz78 | 可直接扩展 | 4,8,16 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| lzss | 需要格式适配 | 4,8,16 | implemented_and_synthetic_verified | 窗口原始字节跨度保持：LZ77约32767.5bytes，LZSS2048bytes；最短匹配仍3word，记录原权重跨度。 |
| lzw | 可直接扩展 | 4,8,16 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| mdct_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| motion_compensation | 可直接扩展 | 4,8,16,32 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| neural_entropy | 可直接扩展 | 4,8,16 | implemented_and_synthetic_verified | 内部逐bit概率构造整word CDF；宽度改变previous-word/前缀/深度特征或状态更新边界，是真实新配置，不是只加标签。 |
| opq_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| paq | 可直接扩展 | 4,8,16 | implemented_and_synthetic_verified | 内部逐bit概率构造整word CDF；宽度改变previous-word/前缀/深度特征或状态更新边界，是真实新配置，不是只加标签。 |
| pca_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| png_filtering | 可直接扩展 | 4,8,16,32 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| ppm | 可直接扩展 | 4,8,16 | implemented_and_synthetic_verified | 三阶word上下文，覆盖3*unit_bits/4个历史FP4权重；完整终端域、escape/KT先验同步扩至65536。 |
| pq_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| rans | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 4/8/16完整word；32稀疏观察字母表，原rANS16-bit归一化频率上限65536，超过时明确resource_limited，不能零收益或截断。 |
| repair | 可直接扩展 | 4,8,16 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| rle | 可直接扩展 | 4,8,16,32,64 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 8bit起符号按实际宽度小端存储、run用LEB128，版本2；4bit保留原格式。 |
| rlz | 可直接扩展 | 4,8,16 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| rvq_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| sequitur | 可直接扩展 | 4,8,16 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |
| sparse_coding | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| svd_lowrank | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| sz_like_lossless | 可直接扩展 | 4,8,16,32 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 32-bit用已出现word的稀疏映射＋原SZ3 Huffman/Zstd level3，保留Lorenzo及精确反馈。 |
| tt_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| tucker_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| vcdiff | 需要格式适配 | 4,8 | implemented_and_synthetic_verified | 现版wire每nibble扩一byte；8-bit版真正packed字节；16/32序列化同字节时等价，不另跑。 |
| vq_codeword | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 保持medoid与码长目标；宽字母表需稀疏频率/候选成本，不能分配2^32表。 16/32使用稀疏Laplace码长成本、同medoid训练；纠错改用原nibble序列，auto比较共享/中心×word位置×nibble位置的完整计费。 |
| vq_lossless | 需要格式适配 | 4,8,16,32 | implemented_and_synthetic_verified | 8/16/32-bit无符号word精确转float64数值；保留原算法/形状/秩/Q20参数，用实际整数逆向路径和原nibble单位精确XOR纠错。 |
| xor | 可直接扩展 | 4,8,16,32,64 | implemented_and_synthetic_verified | 4-bit原版保留；8/16-bit使用真正联合word。 |

JPEG-LS 8/16 的首次模型抽样因封装使用预估容量失败；独立缓冲区修正库已扩容重跑通过，原CharLS无损机制/行布局不变；旧失败状态保留。
