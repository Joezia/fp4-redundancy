# 008 数据来源

数据文件都不进 git（放在 `data/008/`，`data/` 已在 `.gitignore` 中）。时间为北京时间（UTC+8）。

## Kodak PhotoCD PCD0992（图像）

- 来源页：http://r0k.us/graphics/kodak/ （跳转到 https://r0k.us/graphics/kodak/）；文件 URL 为 `https://r0k.us/graphics/kodak/kodak/kodimNN.png`（该页每张图的详情页 `kodimNN.html` 中给出的 PNG）。
- 下载时间：2026-11-16 19:59–20:00。24 个文件共 15,394,305 字节。
- 官方校验值：无（该页未提供）。
- 使用说明原文摘录（来源页）：“The pictures below link to lossless, true color (24 bits per pixel, aka "full color") images. It is my understanding they have been released by the Eastman Kodak Company for unrestricted usage. Many sites use them as a standard test suite for compression testing, etc.”
- 划分：kodim01–12 只用来选参数，kodim13–24 出报告；另给出全部 24 张的结果。

| 文件 | 字节 | sha256 |
| --- | --- | --- |
| kodim01.png | 736,501 | `a56e27cbf5f843c048b6af1d6e090760e9c92fadba88b7dee0205918a37523bd` |
| kodim02.png | 617,995 | `4f4b74a79237e311d72cad958237b5f7088d8bce1c82305ebefe1a70e3022dfd` |
| kodim03.png | 502,888 | `e25ca1ff2f0c0cb5fdfd5f9b0a0bb21ac4c3de3c84a67f35b09a85d3306249db` |
| kodim04.png | 637,432 | `e3b946107c5d3441c022f678d0c3caf1e224d81b1604ba840a4f88e562de61aa` |
| kodim05.png | 785,610 | `10349e963c5c813d327852f82c1795fa4148d69fedffc4c589bee458e3ac3d53` |
| kodim06.png | 618,959 | `363510303b715d4cbc384e1ce227e466b613a09e1b71ae985882bf8e7fbd9b18` |
| kodim07.png | 566,322 | `b77d3f006f42414bb242222e0482e750c0fb9e5ee8d4bed2f6f11c5605fe54a4` |
| kodim08.png | 788,470 | `ba23983c76b4832ee0e8af0592664756841a16779acd69f792e268fb6d13d6e7` |
| kodim09.png | 582,899 | `6a4361c2fc194feb4edaa9f9a4a0620fb9943e460ac7fdf037fb0f6dd6607a7d` |
| kodim10.png | 593,463 | `9dfb70f5867c29ff9ed6313683f19b3d867849e40fbc0c4c54a4a89df341cf23` |
| kodim11.png | 621,023 | `7936814b58b5387fce2e4e2488b4ec830dadd95fa9520f358ddb30990b50f2b6` |
| kodim12.png | 531,024 | `d78c37c2f04f23761ed2367dd77e2db584ddd4c3950833fecf89f199a8126980` |
| kodim13.png | 822,712 | `bc34a3ce58dea09dce1704c997171602de90cb34d0c8503a988b77f473d39b08` |
| kodim14.png | 692,201 | `55a94550ff18f3246c4074fd32b77b0c74447c26b6ad274d564d999c0450ba6e` |
| kodim15.png | 612,582 | `7538cbb80cb9103606c48b806eae57d56c885c7f90b9b3be70a41160f9cbb683` |
| kodim16.png | 534,247 | `a89c7268ccd4718ba424a99fc4643c572cf692ca6eae887185ceb4e9f11d2e54` |
| kodim17.png | 602,078 | `37afcc89fbdcb76d9518e04b2fc011027e2f4cd14b3b2f83cefd721641a47c5b` |
| kodim18.png | 780,947 | `1a9258c365988961d87a0598725b609139c303ad48a5aad6c503c3b1a87849aa` |
| kodim19.png | 671,476 | `b7450b264b1b0a411390d8931b112c27905a992520fc90569dc4b920aa32bbdc` |
| kodim20.png | 492,462 | `3b46c71e3b92a563820ba32936be8330c586c41f938efd94be938386aae4328a` |
| kodim21.png | 637,051 | `ac958597c82073f6bb65129c68f72b651db5b9efd82e11547d07350214bc268b` |
| kodim22.png | 701,970 | `1cee58eb1f2d9c7ebb254d208a03c783ce6cf2c4d8c2cf45e235dd23b4ce1b29` |
| kodim23.png | 557,596 | `e3111a2fd4da24af15d6459ef9eacfe54106b38e27b4a21821b75c3f5d2d5baf` |
| kodim24.png | 706,397 | `1071c68372cc5a01435c2c225a5cf7d4bb803846ec08bb6b3d6721b156d7cb96` |

## EBU SQAM（EBU Tech 3253，音乐/通用音频）

- 规格指定的官方页面：https://tech.ebu.ch/publications/sqamcd 。该页对命令行请求返回 Cloudflare 拦截页（HTTP 403，“Sorry, you have been blocked”），同站的 `https://tech.ebu.ch/docs/tech/tech3253.pdf` 可以下载，但 zip 包的链接同样返回 403。
- 实际下载自 EBU 自己的测试素材站 EBU QC：https://qc.ebu.io/testmaterials/523/ （条目 “TECH3253_SQAM_FLAC”，由 EBU 提供，页面写明 “See for the Terms of Use: https://tech.ebu.ch/publications/sqamcd”），下载链接 `https://qc.ebu.io/testmaterials/523/1/download/`，重定向到该站的 S3 存储。**这是 EBU 运营的站点，但不是规格写的 tech.ebu.ch 页面，属于来源上的偏差，在此如实说明。**
- 下载时间：2026-11-16 20:00:52。文件 `SQAM_FLAC.zip`，175,545,976 字节；EBU QC 页面标注 167.4 MB（= 175,545,976 / 2^20，一致）。
- sha256：`7d6fcd0fc42354637291792534b61bf129612f221f8efef97b62e8942a8686aa`。官方校验值：无（EBU QC 页面写明 “MD5 not provided / SHA-256 not provided”）。
- 内容：70 个 FLAC 曲目（01–70），44.1 kHz，立体声，16 bit，共 3517.0 秒；另有一个 `EBU SQAM.m3u` 播放列表。
- 许可原文摘录（EBU QC 页面与 tech.ebu.ch 页面存档中的同一句）：“By downloading this file you agree not to use this material for commercial purposes other than as a R&D tool.”（只允许非商业研发用途。）
- 划分：奇数号曲目只用来选参数，偶数号曲目出报告。

## LibriSpeech（OpenSLR SLR12，语音）

- 官方页面：https://www.openslr.org/12 。下载用的是该页列出的第一个链接 `https://openslr.trmal.net/resources/12/<文件>`（页面上的 [EU] 镜像之一）。
- 下载时间：md5sum.txt 19:59:11，test-clean.tar.gz 20:00:45，dev-clean.tar.gz 20:01:38。

| 文件 | 字节 | sha256 | md5（本地） | 官方 md5（md5sum.txt） | 一致 |
| --- | --- | --- | --- | --- | --- |
| dev-clean.tar.gz | 337,926,286 | `76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3` | `42e2234ba48799c1f50f24a7926300a1` | `42e2234ba48799c1f50f24a7926300a1` | 是 |
| test-clean.tar.gz | 346,663,984 | `39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23` | `32fa31d27d2e1cad72775fee3f4849a9` | `32fa31d27d2e1cad72775fee3f4849a9` | 是 |
| md5sum.txt | 600 | `efe6522682d078fdcfd2c3b2d418a53a89683b342dd5147a769a41a7e021fe3f` | — | — | — |

- 许可原文（压缩包内 `LibriSpeech/LICENSE.TXT`）：“LibriSpeech (c) 2014 by Vassil Panayotov. LibriSpeech ASR corpus is licensed under a Creative Commons Attribution 4.0 International License.”；官方页面写 “License: CC BY 4.0”。
- 划分：dev-clean 只用来选参数，test-clean 出报告。

## 新装的依赖

- `soundfile` 0.14.0（`pip install soundfile`，自带 libsndfile 1.2.2），用于解码 FLAC 和产生 FLAC -8 对照。只装了这一个；系统里没有 `flac`、`ffmpeg`，所以没有 FFV1 / JPEG-LS 对照。
