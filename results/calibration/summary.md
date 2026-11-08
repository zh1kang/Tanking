# Leak estimator calibration (synthetic tells)

Synthetic thinking times log t = 8.0 - delta * rank / (n_ranks - 1) + N(0, 0.5^2) on real information sets and action distributions. Reference = point estimate at n = 4096 per information set (0 when delta = 0).

| game | delta | n per infoset | reference | mean estimate | bias | CI covers reference | tell rule fires | null range contains 0 |
|---|---|---|---|---|---|---|---|---|
| kuhn | 0 | 64 | 0.0 | -10.3 | -10.3 | 100% | 0% | 90% |
| kuhn | 0 | 512 | 0.0 | -3.0 | -3.0 | 100% | 0% | 95% |
| kuhn | 0.15 | 64 | 2.0 | -8.5 | -10.5 | 100% | 0% | 90% |
| kuhn | 0.15 | 512 | 2.0 | -0.7 | -2.7 | 100% | 0% | 85% |
| kuhn | 0.3 | 64 | 7.6 | -1.7 | -9.3 | 100% | 0% | 85% |
| kuhn | 0.3 | 512 | 7.6 | 6.3 | -1.3 | 100% | 10% | 95% |
| kuhn | 0.6 | 64 | 30.2 | 19.7 | -10.5 | 100% | 30% | 100% |
| kuhn | 0.6 | 512 | 30.2 | 29.0 | -1.2 | 95% | 90% | 90% |
| kuhn | 1 | 64 | 55.4 | 45.4 | -10.1 | 100% | 100% | 100% |
| kuhn | 1 | 512 | 55.4 | 52.5 | -2.9 | 80% | 75% | 75% |
| leduc | 0 | 64 | 0.0 | -57.7 | -57.7 | 85% | 0% | 5% |
| leduc | 0 | 512 | 0.0 | -4.5 | -4.5 | 100% | 0% | 90% |
| leduc | 0.15 | 64 | 0.9 | -58.6 | -59.5 | 90% | 0% | 10% |
| leduc | 0.15 | 512 | 0.9 | -2.3 | -3.2 | 100% | 0% | 95% |
| leduc | 0.3 | 64 | 12.5 | -49.1 | -61.6 | 100% | 0% | 15% |
| leduc | 0.3 | 512 | 12.5 | 8.3 | -4.2 | 100% | 10% | 95% |
| leduc | 0.6 | 64 | 92.2 | 16.2 | -76.0 | 100% | 0% | 0% |
| leduc | 0.6 | 512 | 92.2 | 84.7 | -7.5 | 100% | 90% | 90% |
| leduc | 1 | 64 | 239.8 | 150.9 | -88.9 | 95% | 5% | 5% |
| leduc | 1 | 512 | 239.8 | 233.8 | -6.0 | 100% | 85% | 85% |
