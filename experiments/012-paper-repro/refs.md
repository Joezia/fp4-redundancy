# 012 论文与代码来源

论文 PDF 存在 `refs/012/`（不提交），文字版由 `pdftotext` 生成。

| 文件 | URL | sha256 |
|---|---|---|
| seedlm_iclr2025.pdf（ICLR 2025 会议版） | https://proceedings.iclr.cc/paper_files/paper/2025/file/222a2a46018a1e7b55ba48ba11932d04-Paper-Conference.pdf | 26a66589744f5df665ef0e7f2270a11ded849c8cf418a3652e8f4dfedf3c9117 |
| 2410.10714v2.pdf（arXiv 最新版 v2） | https://arxiv.org/pdf/2410.10714v2 | 445816d21cfc5353d247e410625fe62f9ac3ed9585efbf0a4ca769c969f7ca9c |
| 1901.04866.pdf（BB-ANS，arXiv v1，ICLR 2019 会议版） | https://arxiv.org/pdf/1901.04866v1 | b84a5f607446274fde7d84a251bc5439c01a28a1724d5bcb199564e993a56351 |
| llm265_micro2025.pdf（LLM.265，MICRO 2025） | https://users.cs.duke.edu/~ml579/papers/llm265_micro2025.pdf | 91d68c37028893381f7ba38612e211d339ab411323501a890e8b2aa4c0cfbecf |

OpenReview 的 BB-ANS PDF（https://openreview.net/pdf?id=ryE98iR5tm）对自动下载返回网页，未取得。

代码（克隆在 `data/012/src/`，不提交）：

| 仓库 | commit | 用途 |
|---|---|---|
| https://github.com/bits-back/bits-back | 45aef5ad4785ac290a92b728b3dcd2a3d6f36b0a | BB-ANS 官方代码（B1；B2、B3 的单条链） |
| https://github.com/j-towns/craystack | 992e14d1ecedb1de56eba041bc2d7b72d1084ef3 | BB-ANS 作者的通用库（B2、B3 的并行流）；需要 `autograd`，装在独立环境 `data/012/venv`（基于 lab 环境，另装 autograd） |
| https://github.com/qiuyu-ren/seedlm-repro | 5f42af6f0509a73b7edd9194556c5611b7114ad9 | SeedLM 第三方复现，只作对照 |
| https://github.com/alexander-camuto/seedlm-sim | 4b8f449557794caab5f5a29ea657e245b2f3d2bc | SeedLM 第三方复现（抽头与论文表 6 不同），只作参考 |
| https://github.com/MyasnikovNA/SeedLM-Reproduction | 33d34b2b12ff00f3bc7d627f81d1eaa51855fe87 | SeedLM 第三方复现（课程作业），只作参考 |
| https://github.com/Entropy-xcy/llm.265 | 262e588b11937801a2b3bd0ce760ba05abc4da32 | LLM.265 官方代码，只读（克隆在 `data/012/src/Entropy-xcy_llm.265`）；复现用 ffmpeg 7.1.1（conda-forge GPL 版，含 `hevc_nvenc`、`libx265`），装在独立环境 `llm265` |
