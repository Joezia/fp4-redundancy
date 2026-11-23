# S3-c summary, T1 (24 matrices), bit/weight; reference 3.6056; all residual streams round-trip exactly


## C=8, P=3, K=16 (range)


### side information (S3-a solutions)

| stream | fixed | adaptive, no ctx | ctx 1 | ctx 2 | H0 |
|---|---:|---:|---:|---:|---:|
| seed | 2.0000 | 2.0056 | | | 1.9890 |
| exponent | 0.5000 | 0.1589 | 0.1554 (E) | 0.1554 (E, prev e) | |
| coefficients | 1.5000 | 1.4576 | 1.4524 (p) | 1.4401 (p, e) | |

### sharing

| variant | G | rel MSE | code mismatch | side fixed | code | total fixed | net fixed | side entropy-coded | total | net | round trip |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_range | 1 | 0.0151 | 0.219 | 4.0000 | 0.7055 | 4.7363 | -1.1307 | 3.5954 | 4.3317 | -0.7261 | True |
| raw_range | 2 | 0.0762 | 0.553 | 3.0000 | 1.8060 | 4.8368 | -1.2312 | 2.6002 | 4.4370 | -0.8314 | True |
| raw_range | 8 | 0.3082 | 0.771 | 2.2500 | 2.8084 | 5.0892 | -1.4836 | 1.8516 | 4.6908 | -1.0852 | True |
| raw_range | 32 | 0.4676 | 0.821 | 2.0625 | 3.1062 | 5.1995 | -1.5939 | 1.6417 | 4.7787 | -1.1731 | True |
| raw_range | 128 | 0.5492 | 0.840 | 2.0156 | 3.2198 | 5.2663 | -1.6606 | 1.5740 | 4.8246 | -1.2190 | True |
| raw_range | all | 0.6283 | 0.858 | 2.0000 | 3.3133 | 5.3441 | -1.7385 | 1.5115 | 4.8556 | -1.2500 | True |
| norm_range | 1 | 0.0151 | 0.218 | 4.0000 | 0.7048 | 4.7355 | -1.1299 | 3.6079 | 4.3434 | -0.7378 | True |
| norm_range | 2 | 0.0762 | 0.553 | 3.0000 | 1.8058 | 4.8366 | -1.2309 | 2.6110 | 4.4476 | -0.8420 | True |
| norm_range | 8 | 0.3318 | 0.773 | 2.2500 | 2.8255 | 5.1062 | -1.5006 | 1.9005 | 4.7568 | -1.1511 | True |
| norm_range | 32 | 0.4924 | 0.824 | 2.0625 | 3.1196 | 5.2128 | -1.6072 | 1.7202 | 4.8705 | -1.2649 | True |
| norm_range | 128 | 0.5623 | 0.842 | 2.0156 | 3.2270 | 5.2734 | -1.6678 | 1.6728 | 4.9306 | -1.3250 | True |
| norm_range | all | 0.6281 | 0.852 | 2.0000 | 3.3074 | 5.3381 | -1.7325 | 1.6397 | 4.9779 | -1.3722 | True |
| norm_fixed | 1 | 0.0194 | 0.273 | 3.5000 | 0.8406 | 4.3714 | -0.7658 | 3.3853 | 4.2567 | -0.6511 | True |
| norm_fixed | 2 | 0.0846 | 0.574 | 2.5000 | 1.8714 | 4.4022 | -0.7966 | 2.3760 | 4.2782 | -0.6726 | True |
| norm_fixed | 8 | 0.3448 | 0.777 | 1.7500 | 2.8486 | 4.6294 | -1.0238 | 1.5483 | 4.4277 | -0.8221 | True |
| norm_fixed | 32 | 0.5069 | 0.827 | 1.5625 | 3.1370 | 4.7303 | -1.1247 | 1.2848 | 4.4526 | -0.8470 | True |
| norm_fixed | 128 | 0.5767 | 0.843 | 1.5156 | 3.2421 | 4.7885 | -1.1829 | 1.1920 | 4.4649 | -0.8593 | True |
| norm_fixed | all | 0.6388 | 0.849 | 1.5000 | 3.3080 | 4.8388 | -1.2331 | 1.1398 | 4.4786 | -0.8730 | True |

### seed chosen by code length; bits-back bound (frozen train8 table, table not charged)

- static code length of the residual: MSE seed 0.7035 (numpy check 0.7035); code-length seed 0.6433 (numpy 0.6433); mixture / bits-back bound 0.4466
- log2 effective number of seeds per block: mean 1.574, 10/50/90% [0.356, 1.417, 2.911]
- code-length seed, really coded: rel MSE 0.0171, mismatch 0.202, code 0.6367 (MSE seed 0.7055), total fixed side 4.6674 (-1.0618), entropy-coded side 3.5958, total 4.2632 (-0.6576), round trip True

## C=12, P=4, K=16 (range)


### side information (S3-a solutions)

| stream | fixed | adaptive, no ctx | ctx 1 | ctx 2 | H0 |
|---|---:|---:|---:|---:|---:|
| seed | 1.3333 | 1.3426 | | | 1.3276 |
| exponent | 0.3333 | 0.0957 | 0.0934 (E) | 0.0934 (E, prev e) | |
| coefficients | 1.3333 | 1.2894 | 1.2870 (p) | 1.2741 (p, e) | |

### sharing

| variant | G | rel MSE | code mismatch | side fixed | code | total fixed | net fixed | side entropy-coded | total | net | round trip |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_range | 1 | 0.0509 | 0.474 | 3.0000 | 1.5163 | 4.5471 | -0.9415 | 2.7008 | 4.2479 | -0.6423 | True |
| raw_range | 2 | 0.1520 | 0.679 | 2.3333 | 2.2999 | 4.6640 | -1.0584 | 2.0327 | 4.3633 | -0.7577 | True |
| raw_range | 8 | 0.3876 | 0.807 | 1.8333 | 2.9750 | 4.8391 | -1.2335 | 1.5260 | 4.5317 | -0.9261 | True |
| raw_range | 32 | 0.5307 | 0.843 | 1.7083 | 3.1992 | 4.9384 | -1.3327 | 1.3808 | 4.6108 | -1.0052 | True |
| raw_range | 128 | 0.6023 | 0.856 | 1.6771 | 3.2888 | 4.9966 | -1.3910 | 1.3311 | 4.6507 | -1.0451 | True |
| raw_range | all | 0.6708 | 0.868 | 1.6667 | 3.3642 | 5.0617 | -1.4561 | 1.2836 | 4.6786 | -1.0730 | True |
| norm_range | 1 | 0.0533 | 0.478 | 3.0000 | 1.5350 | 4.5658 | -0.9602 | 2.7102 | 4.2760 | -0.6704 | True |
| norm_range | 2 | 0.1617 | 0.681 | 2.3333 | 2.3218 | 4.6859 | -1.0803 | 2.0447 | 4.3972 | -0.7916 | True |
| norm_range | 8 | 0.4239 | 0.810 | 1.8333 | 2.9979 | 4.8620 | -1.2564 | 1.5648 | 4.5934 | -0.9878 | True |
| norm_range | 32 | 0.5624 | 0.845 | 1.7083 | 3.2159 | 4.9551 | -1.3494 | 1.4438 | 4.6905 | -1.0849 | True |
| norm_range | 128 | 0.6236 | 0.858 | 1.6771 | 3.3001 | 5.0080 | -1.4024 | 1.4116 | 4.7425 | -1.1369 | True |
| norm_range | all | 0.6807 | 0.870 | 1.6667 | 3.3705 | 5.0679 | -1.4623 | 1.3904 | 4.7916 | -1.1860 | True |
| norm_fixed | 1 | 0.0620 | 0.513 | 2.6667 | 1.6269 | 4.3244 | -0.7187 | 2.5122 | 4.1699 | -0.5643 | True |
| norm_fixed | 2 | 0.1746 | 0.692 | 2.0000 | 2.3668 | 4.3976 | -0.7919 | 1.8230 | 4.2206 | -0.6150 | True |
| norm_fixed | 8 | 0.4423 | 0.812 | 1.5000 | 3.0210 | 4.5518 | -0.9462 | 1.2433 | 4.2951 | -0.6895 | True |
| norm_fixed | 32 | 0.5831 | 0.847 | 1.3750 | 3.2355 | 4.6413 | -1.0356 | 1.0509 | 4.3171 | -0.7115 | True |
| norm_fixed | 128 | 0.6445 | 0.860 | 1.3438 | 3.3182 | 4.6927 | -1.0871 | 0.9764 | 4.3254 | -0.7198 | True |
| norm_fixed | all | 0.6961 | 0.870 | 1.3333 | 3.3815 | 4.7456 | -1.1400 | 0.9090 | 4.3213 | -0.7157 | True |

### seed chosen by code length; bits-back bound (frozen train8 table, table not charged)

- static code length of the residual: MSE seed 1.5151 (numpy check 1.5151); code-length seed 1.5019 (numpy 1.5019); mixture / bits-back bound 1.3856
- log2 effective number of seeds per block: mean 1.395, 10/50/90% [0.275, 1.283, 2.606]
- code-length seed, really coded: rel MSE 0.0521, mismatch 0.470, code 1.5001 (MSE seed 1.5163), total fixed side 4.5309 (-0.9253), entropy-coded side 2.7000, total 4.2309 (-0.6253), round trip True
