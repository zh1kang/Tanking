## Phase 1: does thinking time leak the card?

| game | model | tokens/decision | truncated | leak (mbb/hand) [95% CI] | null [95% CI] | card headroom | bits gained |
|---|---|---:|---:|---|---|---:|---:|
| kuhn | Qwen3-0.6B | 4379 | 4% | -1.0 [-3.5, 3.0] | -0.6 [-2.9, 2.2] | 451 | -0.019 |
| kuhn | Qwen3-1.7B | 6409 | 19% | 0.3 [-7.7, 8.2] | -1.6 [-8.1, 0.0] | 428 | -0.003 |
| kuhn | Qwen3-4B | 4984 | 8% | 17.7 [11.7, 27.7] | -2.5 [-8.4, 2.5] | 236 | 0.056 |
| kuhn | Qwen3-8B | 3982 | 3% | 0.9 [-0.3, 3.2] | -0.3 [-0.9, 0.0] | 196 | 0.116 |
| leduc | Qwen3-0.6B | 4210 | 3% | -65.1 [-112.2, 10.9] | -55.3 [-110.9, -10.1] | 2353 | 0.005 |
| leduc | Qwen3-1.7B | 7610 | 49% | -16.2 [-70.0, 75.3] | -62.8 [-124.9, -4.9] | 2660 | 0.027 |
| leduc | Qwen3-4B | 7578 | 43% | 70.5 [-10.6, 143.2] | -41.5 [-84.1, -4.6] | 2699 | 0.159 |

## Strength and cost

| game | model | mode | exploitability (mbb/hand) | EV vs Nash (mbb/hand) | tokens/decision |
|---|---|---|---:|---:|---:|
| kuhn | Qwen3-0.6B | think | 330.4 | -129.7 | 4379 |
| kuhn | Qwen3-1.7B | think | 264.7 | -122.5 | 6409 |
| kuhn | Qwen3-4B | think | 140.7 | -42.7 | 4984 |
| kuhn | Qwen3-8B | think | 166.9 | -22.7 | 3982 |
| kuhn | Qwen3-0.6B | nothink | 332.3 | -176.2 | 0 |
| kuhn | Qwen3-1.7B | nothink | 333.3 | -111.2 | 0 |
| kuhn | Qwen3-4B | nothink | 333.3 | -116.1 | 0 |
| kuhn | Qwen3-8B | nothink | 334.2 | -114.8 | 0 |
| kuhn | distill_0.6B | nothink | 12.1 | -0.0 | 0 |
| kuhn | distill_1.7B | nothink | 4.7 | -0.0 | 0 |
| kuhn4 | Qwen3-0.6B | nothink | 307.7 | -144.8 | 0 |
| kuhn4 | Qwen3-1.7B | nothink | 250.0 | -106.8 | 0 |
| kuhn4 | Qwen3-4B | nothink | 250.0 | -117.9 | 0 |
| kuhn4 | Qwen3-8B | nothink | 251.3 | -108.0 | 0 |
| kuhn4 | distill_0.6B | nothink | 5.9 | -0.1 | 0 |
| kuhn4 | distill_1.7B | nothink | 4.5 | -0.0 | 0 |
| kuhn4_b2 | Qwen3-0.6B | nothink | 616.6 | -188.5 | 0 |
| kuhn4_b2 | Qwen3-1.7B | nothink | 500.0 | -111.7 | 0 |
| kuhn4_b2 | Qwen3-4B | nothink | 531.9 | -126.8 | 0 |
| kuhn4_b2 | Qwen3-8B | nothink | 504.2 | -114.2 | 0 |
| kuhn4_b2 | distill_0.6B | nothink | 82.0 | -2.1 | 0 |
| kuhn4_b2 | distill_1.7B | nothink | 117.1 | -1.6 | 0 |
| kuhn5 | Qwen3-0.6B | nothink | 289.2 | -134.9 | 0 |
| kuhn5 | Qwen3-1.7B | nothink | 200.0 | -67.1 | 0 |
| kuhn5 | Qwen3-4B | nothink | 217.4 | -78.2 | 0 |
| kuhn5 | Qwen3-8B | nothink | 201.9 | -68.5 | 0 |
| kuhn5 | distill_0.6B | nothink | 70.3 | -11.8 | 0 |
| kuhn5 | distill_1.7B | nothink | 87.3 | -12.2 | 0 |
| kuhn_b2 | Qwen3-0.6B | nothink | 635.7 | -243.6 | 0 |
| kuhn_b2 | Qwen3-1.7B | nothink | 666.7 | -333.3 | 0 |
| kuhn_b2 | Qwen3-4B | nothink | 666.7 | -340.5 | 0 |
| kuhn_b2 | Qwen3-8B | nothink | 667.6 | -338.4 | 0 |
| kuhn_b2 | distill_0.6B | nothink | 0.1 | -0.0 | 0 |
| kuhn_b2 | distill_1.7B | nothink | 0.0 | -0.0 | 0 |
| leduc | Qwen3-0.6B | think | 1486.5 | -535.6 | 4210 |
| leduc | Qwen3-1.7B | think | 1707.8 | -497.4 | 7610 |
| leduc | Qwen3-4B | think | 1587.4 | -452.4 | 7578 |
| leduc | Qwen3-0.6B | nothink | 1508.6 | -651.4 | 0 |
| leduc | Qwen3-1.7B | nothink | 2366.8 | -524.8 | 0 |
| leduc | Qwen3-4B | nothink | 2360.8 | -514.7 | 0 |
| leduc | distill_0.6B | nothink | 241.8 | -35.4 | 0 |
| leduc | distill_1.7B | nothink | 325.4 | -50.8 | 0 |
| leduc4 | Qwen3-0.6B | nothink | 1491.4 | -528.6 | 0 |
| leduc4 | Qwen3-1.7B | nothink | 2333.4 | -443.5 | 0 |
| leduc4 | Qwen3-4B | nothink | 2328.8 | -442.4 | 0 |
| leduc4 | Qwen3-8B | nothink | 2257.6 | -518.8 | 0 |
| leduc4 | distill_0.6B | nothink | 201.5 | -49.2 | 0 |
| leduc4 | distill_1.7B | nothink | 277.7 | -51.7 | 0 |
| leduc4_b12 | Qwen3-0.6B | nothink | 777.0 | -360.4 | 0 |
| leduc4_b12 | Qwen3-1.7B | nothink | 1142.9 | -248.2 | 0 |
| leduc4_b12 | Qwen3-4B | nothink | 1140.3 | -251.9 | 0 |
| leduc4_b12 | Qwen3-8B | nothink | 1087.3 | -282.5 | 0 |
| leduc4_b12 | distill_0.6B | nothink | 390.1 | -54.9 | 0 |
| leduc4_b12 | distill_1.7B | nothink | 337.4 | -60.1 | 0 |
| leduc_b12 | Qwen3-0.6B | nothink | 787.7 | -364.6 | 0 |
| leduc_b12 | Qwen3-1.7B | nothink | 1133.4 | -341.7 | 0 |
| leduc_b12 | Qwen3-4B | nothink | 1131.1 | -351.2 | 0 |
| leduc_b12 | Qwen3-8B | nothink | 1174.1 | -385.4 | 0 |
| leduc_b12 | distill_0.6B | nothink | 289.3 | -39.4 | 0 |
| leduc_b12 | distill_1.7B | nothink | 260.6 | -50.2 | 0 |
| leduc_b36 | Qwen3-0.6B | nothink | 2266.5 | -870.9 | 0 |
| leduc_b36 | Qwen3-1.7B | nothink | 3600.1 | -994.6 | 0 |
| leduc_b36 | Qwen3-4B | nothink | 3591.6 | -990.7 | 0 |
| leduc_b36 | Qwen3-8B | nothink | 3758.3 | -1059.2 | 0 |
| leduc_b36 | distill_0.6B | nothink | 388.5 | -100.3 | 0 |
| leduc_b36 | distill_1.7B | nothink | 595.1 | -169.0 | 0 |

## Defenses: kuhn_0.6B

| defense | parameter | leak (mbb/hand, ± SE over repeats) | added tokens/decision |
|---|---|---:|---:|
| none | 0 | -1.3 ± 0.7 | 0 |
| constant | 8192 | 0.0 ± 0.0 | 3819 |
| random | 1024 | -1.0 ± 0.4 | 515 |
| random | 2048 | -0.3 ± 0.5 | 1017 |
| random | 4096 | -0.8 ± 0.5 | 2047 |
| random | 8192 | -0.7 ± 0.6 | 4105 |
| random | 16384 | -1.3 ± 0.5 | 8220 |
| lp | 0 | -0.6 ± 0.6 | 217 |
| lp | 0.02 | -0.7 ± 0.5 | 145 |
| lp | 0.05 | -0.8 ± 0.4 | 99 |
| lp | 0.1 | -1.1 ± 0.4 | 73 |
| lp | 0.2 | -1.3 ± 0.3 | 63 |

| λ (mbb per token) | exploitability | reported tokens | computed tokens |
|---|---:|---:|---:|
| 0 | 327.9 | 3886 | 2042 |
| 0.003 | 324.0 | 128 | 128 |
| 0.01 | 324.0 | 128 | 128 |
| 0.03 | 324.0 | 128 | 128 |
| 0.1 | 324.0 | 128 | 128 |
| full thinking | 332.2 | 4377 | 4377 |

## Defenses: kuhn_1.7B

| defense | parameter | leak (mbb/hand, ± SE over repeats) | added tokens/decision |
|---|---|---:|---:|
| none | 0 | -3.2 ± 1.5 | 0 |
| constant | 8192 | 0.0 ± 0.0 | 1787 |
| random | 1024 | -0.3 ± 1.2 | 515 |
| random | 2048 | 0.4 ± 1.6 | 1016 |
| random | 4096 | -3.1 ± 1.5 | 2047 |
| random | 8192 | -6.0 ± 1.8 | 4101 |
| random | 16384 | -4.0 ± 0.7 | 8210 |
| lp | 0 | -4.6 ± 1.1 | 388 |
| lp | 0.02 | -4.7 ± 1.1 | 296 |
| lp | 0.05 | -3.4 ± 1.6 | 185 |
| lp | 0.1 | -1.5 ± 1.7 | 103 |
| lp | 0.2 | -2.1 ± 0.7 | 57 |

| λ (mbb per token) | exploitability | reported tokens | computed tokens |
|---|---:|---:|---:|
| 0 | 263.7 | 7673 | 6108 |
| 0.003 | 263.7 | 7673 | 6108 |
| 0.01 | 296.8 | 4096 | 4020 |
| 0.03 | 333.3 | 128 | 128 |
| 0.1 | 333.3 | 128 | 128 |
| full thinking | 263.8 | 6377 | 6377 |

## Defenses: kuhn_4B

| defense | parameter | leak (mbb/hand, ± SE over repeats) | added tokens/decision |
|---|---|---:|---:|
| none | 0 | 15.2 ± 2.1 | 0 |
| constant | 8192 | 0.0 ± 0.0 | 3206 |
| random | 1024 | 15.8 ± 1.5 | 515 |
| random | 2048 | 14.6 ± 1.1 | 1015 |
| random | 4096 | 10.1 ± 2.1 | 2047 |
| random | 8192 | 0.5 ± 1.9 | 4110 |
| random | 16384 | -5.3 ± 1.4 | 8219 |
| lp | 0 | -4.6 ± 1.3 | 990 |
| lp | 0.02 | -8.1 ± 1.6 | 838 |
| lp | 0.05 | -6.5 ± 1.2 | 649 |
| lp | 0.1 | 4.4 ± 1.7 | 400 |
| lp | 0.2 | 17.2 ± 1.6 | 139 |

| λ (mbb per token) | exploitability | reported tokens | computed tokens |
|---|---:|---:|---:|
| 0 | 127.5 | 7251 | 4691 |
| 0.003 | 127.5 | 7251 | 4691 |
| 0.01 | 127.5 | 7251 | 4691 |
| 0.03 | 233.6 | 2106 | 1595 |
| 0.1 | 333.3 | 128 | 128 |
| full thinking | 145.0 | 5005 | 5005 |

## Defenses: kuhn_8B

| defense | parameter | leak (mbb/hand, ± SE over repeats) | added tokens/decision |
|---|---|---:|---:|
| none | 0 | 0.3 ± 0.6 | 0 |
| constant | 8192 | 0.0 ± 0.0 | 4207 |
| random | 1024 | 0.6 ± 0.3 | 515 |
| random | 2048 | 1.0 ± 0.3 | 1015 |
| random | 4096 | 0.1 ± 0.3 | 2047 |
| random | 8192 | -0.2 ± 0.1 | 4112 |
| random | 16384 | -0.3 ± 0.1 | 8224 |
| lp | 0 | 0.0 ± 0.3 | 1935 |
| lp | 0.02 | 0.4 ± 0.2 | 1707 |
| lp | 0.05 | 0.0 ± 0.3 | 1455 |
| lp | 0.1 | -0.7 ± 0.6 | 1097 |
| lp | 0.2 | 0.3 ± 0.5 | 588 |

| λ (mbb per token) | exploitability | reported tokens | computed tokens |
|---|---:|---:|---:|
| 0 | 148.7 | 5571 | 3283 |
| 0.003 | 150.4 | 4096 | 3235 |
| 0.01 | 150.4 | 4096 | 3235 |
| 0.03 | 159.5 | 3359 | 3079 |
| 0.1 | 279.4 | 578 | 754 |
| full thinking | 168.7 | 3994 | 3994 |

## Defenses: leduc_0.6B

| defense | parameter | leak (mbb/hand, ± SE over repeats) | added tokens/decision |
|---|---|---:|---:|
| none | 0 | -125.1 ± 10.3 | 0 |
| constant | 8192 | 0.0 ± 0.0 | 3970 |
| random | 1024 | -131.8 ± 11.3 | 507 |
| random | 2048 | -128.6 ± 9.5 | 1024 |
| random | 4096 | -118.4 ± 7.5 | 2046 |
| random | 8192 | -140.3 ± 9.7 | 4089 |
| random | 16384 | -144.9 ± 7.8 | 8176 |
| lp | 0 | -107.6 ± 8.3 | 494 |
| lp | 0.02 | -128.3 ± 9.6 | 413 |
| lp | 0.05 | -120.0 ± 7.0 | 310 |
| lp | 0.1 | -132.5 ± 9.5 | 199 |
| lp | 0.2 | -131.5 ± 9.7 | 107 |

| λ (mbb per token) | exploitability | reported tokens | computed tokens |
|---|---:|---:|---:|
| 0 | 1332.7 | 5853 | 3460 |
| 0.003 | 1348.6 | 4908 | 2965 |
| 0.01 | 1370.3 | 3865 | 2520 |
| 0.03 | 1406.3 | 2633 | 1727 |
| 0.1 | 1443.7 | 1595 | 1051 |
| full thinking | 1471.0 | 4170 | 4170 |

## Defenses: leduc_1.7B

| defense | parameter | leak (mbb/hand, ± SE over repeats) | added tokens/decision |
|---|---|---:|---:|
| none | 0 | -69.3 ± 10.9 | 0 |
| constant | 8192 | 0.0 ± 0.0 | 589 |
| random | 1024 | -74.5 ± 8.4 | 508 |
| random | 2048 | -70.4 ± 7.6 | 1024 |
| random | 4096 | -84.0 ± 9.0 | 2045 |
| random | 8192 | -110.4 ± 11.0 | 4092 |
| random | 16384 | -109.0 ± 8.7 | 8165 |
| lp | 0 | -75.8 ± 12.6 | 189 |
| lp | 0.02 | -78.9 ± 6.9 | 160 |
| lp | 0.05 | -75.1 ± 4.8 | 121 |
| lp | 0.1 | -63.2 ± 11.2 | 71 |
| lp | 0.2 | -53.2 ± 8.8 | 36 |

| λ (mbb per token) | exploitability | reported tokens | computed tokens |
|---|---:|---:|---:|
| 0 | 1664.9 | 7463 | 6948 |
| 0.003 | 1703.4 | 6920 | 6412 |
| 0.01 | 1677.3 | 4761 | 4668 |
| 0.03 | 1689.9 | 4145 | 4383 |
| 0.1 | 1817.0 | 485 | 1435 |
| full thinking | 1663.6 | 7603 | 7603 |

## Defenses: leduc_4B

| defense | parameter | leak (mbb/hand, ± SE over repeats) | added tokens/decision |
|---|---|---:|---:|
| none | 0 | 18.7 ± 11.6 | 0 |
| constant | 8192 | 0.0 ± 0.0 | 619 |
| random | 1024 | -4.7 ± 9.5 | 506 |
| random | 2048 | -16.7 ± 8.3 | 1024 |
| random | 4096 | -42.1 ± 9.5 | 2047 |
| random | 8192 | -72.5 ± 8.7 | 4087 |
| random | 16384 | -71.2 ± 5.2 | 8170 |
| lp | 0 | -56.9 ± 10.0 | 215 |
| lp | 0.02 | -68.6 ± 6.4 | 197 |
| lp | 0.05 | -58.0 ± 7.1 | 160 |
| lp | 0.1 | -51.1 ± 11.1 | 115 |
| lp | 0.2 | -21.8 ± 7.6 | 54 |

| λ (mbb per token) | exploitability | reported tokens | computed tokens |
|---|---:|---:|---:|
| 0 | 1517.8 | 6587 | 6024 |
| 0.003 | 1506.5 | 3816 | 3342 |
| 0.01 | 1506.2 | 3802 | 3266 |
| 0.03 | 1584.4 | 1665 | 1928 |
| 0.1 | 1605.8 | 1277 | 1557 |
| full thinking | 1497.8 | 7585 | 7585 |

## Serving benchmark (one A100, vLLM)

| game | model | mode | decisions/s | GPU-s per 1k hands | p50 ms | p99 ms | tokens/decision | ms/token | R² |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| kuhn | Qwen3-4B | think | 0.5 | 5029 | 35078 | 48170 | 4769 | 7.87 | 1.000 |
| kuhn | Qwen3-4B | budget | 2.2 | 1100 | 958 | 31230 | 1602 | 7.61 | 1.000 |
| kuhn | Qwen3-4B | nothink | 2393.1 | 1 | 10 | 10 | 0 | 0.00 | – |
| leduc | Qwen3-0.6B | think | 1.1 | 3577 | 8567 | 23160 | 3957 | 2.88 | 0.999 |
| leduc | Qwen3-1.7B | think | 0.4 | 10875 | 36257 | 36299 | 7507 | 4.64 | 1.000 |
| leduc | Qwen3-0.6B | budget | 3.5 | 1145 | 1602 | 16709 | 1603 | 2.73 | 0.999 |
| leduc | Qwen3-0.6B | nothink | 2780.0 | 1 | 7 | 8 | 0 | 0.00 | – |
| leduc | Qwen3-1.7B | nothink | 2208.3 | 2 | 7 | 8 | 0 | 0.00 | – |
| leduc | distill_0.6B | nothink | 2583.3 | 2 | 7 | 9 | 0 | 0.00 | – |
| leduc | distill_1.7B | nothink | 2255.2 | 2 | 7 | 8 | 0 | 0.00 | – |
