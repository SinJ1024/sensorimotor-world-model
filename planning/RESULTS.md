# Results

Generated 2026-10-07 13:41 from `scripts/summarize_runs.py` on: DelftBlue, Leonardo. Success = paper CEM protocol (100 tasks, goal 25 primitive steps ahead, budget 50), in %. Regenerate with `scripts/results_table.py` (see its docstring); do not edit by hand.

## Policy-head input ablation (seed 0)

Every regularizer uses the paper λ of its environment; `noact` and `c1` feed no actions to the head. `–` = not run; *italics* = queued or running. Upstream columns: mean of 5 seeds from `experiments/planning_eval/aggregated_results.csv` (same protocol).

| env | λ | inverse<br>`a_t = h(z_t, z_{t+1})` | policy +a_t<br>`a_{t+1} = π(z_t, z_{t+1}, a_t)` | policy noact<br>`a_{t+1} = π(z_t, z_{t+1})` | policy c1<br>`a_{t+1} = π(z_{t+1})` | upstream inverse | upstream forward-only | upstream random |
|---|---|---|---|---|---|---|---|---|
| TwoRoom | 0.1 | 100 (DelftBlue) / 100 (Leonardo) | 89 | 100 | 88 | 100.0 | 34.6 | 30.0 |
| Reacher | 5 | 68 | 11 | 10 | 14 | 71.4 | 9.8 | 13.6 |
| Push-T | 30 | 89 | 37 | 69 (DelftBlue) / 69 (Leonardo) | 57 | 84.4 | 2.6 | 2.8 |
| OGBench-Cube | 1 | 81 | 69 | 80 (DelftBlue) / 79 (Leonardo) | 66 | 83.0 | 46.8 | 43.0 |

## Across seeds

Run families with at least two evaluated seeds; mean ± sample standard deviation.

| env | run family | cluster | n | success | per seed |
|---|---|---|---|---|---|
| TwoRoom | tworoom_policy_mlp_c2k1 | DelftBlue | 2 | 88.0 ± 1.4 | s0: 89, s2: 87 |
| TwoRoom | tworoom_policy_mlp_c2k1_lam0p005 | DelftBlue | 3 | 52.3 ± 8.0 | s0: 44, s1: 53, s2: 60 |
| TwoRoom | tworoom_policy_mlp_c2k1_lam0p01 | DelftBlue | 3 | 66.0 ± 25.2 | s0: 79, s1: 37, s2: 82 |
| TwoRoom | tworoom_policy_mlp_c2k1_lam0p02 | DelftBlue | 3 | 73.0 ± 21.2 | s0: 49, s1: 81, s2: 89 |
| TwoRoom | tworoom_policy_mlp_c2k1_lam0p03 | DelftBlue | 3 | 68.3 ± 26.5 | s0: 94, s1: 41, s2: 70 |
| TwoRoom | tworoom_policy_mlp_c2k1_lam0p05 | DelftBlue | 3 | 59.3 ± 18.6 | s0: 68, s1: 38, s2: 72 |
| Reacher | reacher_inverse_lambda_1 | DelftBlue | 2 | 71.0 ± 4.2 | s0: 68, s1: 74 |
| Push-T | pusht_policy_mlp_c2k1_lam10p0 | DelftBlue | 2 | 39.5 ± 7.8 | s0: 45, s1: 34 |
| Push-T | pusht_policy_mlp_c2k1_lam20p0 | DelftBlue | 2 | 39.5 ± 0.7 | s0: 40, s1: 39 |
| Push-T | pusht_policy_mlp_c2k1_lam5p0 | DelftBlue | 2 | 35.5 ± 0.7 | s0: 36, s1: 35 |
| Push-T | pusht_policy_mlp_c2k1_noact | DelftBlue | 3 | 65.3 ± 3.2 | s0: 69, s1: 63, s2: 64 |
| OGBench-Cube | cube_policy_mlp_c2k1_noact | DelftBlue | 3 | 78.0 ± 4.4 | s0: 80, s1: 81, s2: 73 |

## All runs

### TwoRoom

| cluster | run | method | λ | step | eff rank | pred loss | inv loss | policy loss | goal loss | success | verdict | other evals |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DelftBlue | tworoom_goalpolicy_g123_seed0 | goal_policy(G=[1,2,3],k1) | 0.1 | 25690 | 10.7 | 4e-05 |  |  | 0.83 | 100 | PASS |  |
| DelftBlue | tworoom_goalpolicy_g2_seed0 | goal_policy(G=[2],k1) | 0.1 | 27440 | 9.01 | 4.13e-05 |  |  | 0.854 | 100 | PASS |  |
| DelftBlue | tworoom_goalpolicy_g3_seed0 | goal_policy(G=[3],k1) | 0.1 | 25690 | 13 | 6.82e-05 |  |  | 0.882 | 100 | PASS |  |
| DelftBlue | tworoom_inverse_lambda_0p1_seed0 | inverse | 0.1 | 29200 | 12.3 | 6.97e-05 | 0.754 |  |  | 100 | PASS |  |
| Leonardo | tworoom_inverse_lambda_0p1_seed0 | inverse | 0.1 | 29200 | 12.3 | 7.28e-05 | 0.753 |  |  | 100 | PASS |  |
| DelftBlue | tworoom_invpolicy_mlp_c2k1_seed0 | inverse+policy(mlp,c2,k1) | 0.1/0.1 | 29200 | 16.8 | 8.27e-05 | 0.753 | 0.951 |  | 100 | PASS |  |
| DelftBlue | tworoom_policy_mamba_c16k1_seed0 | policy(mamba,c16,k1) | 0.1 | 22010 | 117 | 0.000206 |  | 0.911 |  | 45 | PASS |  |
| DelftBlue | tworoom_policy_mamba_c16k2_seed0 | policy(mamba,c16,k2) | 0.1 | 16410 | 126 | 0.000141 |  | 0.92 |  | 50 | PASS |  |
| DelftBlue | tworoom_policy_mamba_c16k3_seed0 | policy(mamba,c16,k3) | 0.1 | 11160 | 79.8 | 7.72e-05 |  | 0.926 |  | 47 | PASS |  |
| DelftBlue | tworoom_policy_mamba_c2k1_seed0 | policy(mamba,c2,k1) | 0.1 | 29200 | 37.7 | 3.37e-05 |  | 0.951 |  | 79 | PASS |  |
| DelftBlue | tworoom_policy_mamba_c4k1_seed0 | policy(mamba,c4,k1) | 0.1 | 25690 | 55.6 | 5.1e-05 |  | 0.931 |  | 79 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c1k1_seed0 | policy(mlp,c1,k1) | 0.1 | 29200 | 21.5 | 3.24e-05 |  | 0.971 |  | 88 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c16k1_seed0 | policy(mlp,c16,k1) | 0.1 | 22010 | 81.4 | 0.000154 |  | 0.916 |  | 50 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c16k2_seed0 | policy(mlp,c16,k2) | 0.1 | 16410 | 68.8 | 0.000104 |  | 0.926 |  | 45 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c16k3_seed0 | policy(mlp,c16,k3) | 0.1 | 11160 | 33.5 | 8.38e-05 |  | 0.931 |  | 39 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p0001_seed0 | policy(mlp,c2,k1) | 0.0001 | 29200 | 182 | 1.53e-06 |  | 0.982 |  | 31 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p0003_seed0 | policy(mlp,c2,k1) | 0.0003 | 29200 | 184 | 1.5e-06 |  | 0.982 |  | 38 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p001_seed0 | policy(mlp,c2,k1) | 0.001 | 29200 | 173 | 1.51e-06 |  | 0.982 |  | 31 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p003_seed0 | policy(mlp,c2,k1) | 0.003 | 29200 | 153 | 1.44e-06 |  | 0.982 |  | 31 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p005_seed0 | policy(mlp,c2,k1) | 0.005 | 29200 | 6.09 | 8.19e-06 |  | 0.963 |  | 44 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p005_seed1 | policy(mlp,c2,k1) | 0.005 | 29200 | 15 | 6.24e-06 |  | 0.955 |  | 53 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p005_seed2 | policy(mlp,c2,k1) | 0.005 | 29200 | 10.2 | 6.54e-06 |  | 0.954 |  | 60 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p01_seed0 | policy(mlp,c2,k1) | 0.01 | 29200 | 11.9 | 9.37e-06 |  | 0.953 |  | 79 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p01_seed1 | policy(mlp,c2,k1) | 0.01 | 29200 | 23.4 | 1.21e-05 |  | 0.954 |  | 37 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p01_seed2 | policy(mlp,c2,k1) | 0.01 | 29200 | 12.2 | 1.03e-05 |  | 0.953 |  | 82 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p02_seed0 | policy(mlp,c2,k1) | 0.02 | 29200 | 17.2 | 1.57e-05 |  | 0.952 |  | 49 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p02_seed1 | policy(mlp,c2,k1) | 0.02 | 29200 | 16.3 | 1.76e-05 |  | 0.953 |  | 81 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p02_seed2 | policy(mlp,c2,k1) | 0.02 | 29200 | 12.5 | 1.61e-05 |  | 0.952 |  | 89 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p03_seed0 | policy(mlp,c2,k1) | 0.03 | 29200 | 17.2 | 1.98e-05 |  | 0.952 |  | 94 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p03_seed1 | policy(mlp,c2,k1) | 0.03 | 29200 | 19.6 | 2.37e-05 |  | 0.953 |  | 41 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p03_seed2 | policy(mlp,c2,k1) | 0.03 | 29200 | 16.9 | 1.83e-05 |  | 0.952 |  | 70 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p05_seed0 | policy(mlp,c2,k1) | 0.05 | 29200 | 20 | 2.88e-05 |  | 0.952 |  | 68 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p05_seed1 | policy(mlp,c2,k1) | 0.05 | 29200 | 25.1 | 3.74e-05 |  | 0.953 |  | 38 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p05_seed2 | policy(mlp,c2,k1) | 0.05 | 29200 | 23.8 | 3e-05 |  | 0.952 |  | 72 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p1_seed1 | policy(mlp,c2,k1) | 0.1 | 29200 | 25.6 | 5.03e-05 |  | 0.952 |  | 95 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1) | 0.1 | 29200 | 13.2 | 5.53e-05 |  | 0.952 |  | 100 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_seed0 | policy(mlp,c2,k1) | 0.1 | 29200 | 27.2 | 4.79e-05 |  | 0.952 |  | 89 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_seed2 | policy(mlp,c2,k1) | 0.1 | 29200 | 28.4 | 4.9e-05 |  | 0.952 |  | 87 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c4k1_seed0 | policy(mlp,c4,k1) | 0.1 | 25690 | 35 | 6.22e-05 |  | 0.934 |  | 82 | PASS |  |
| DelftBlue | tworoom_policy_transformer_c16k1_seed0 | policy(transformer,c16,k1) | 0.1 | 22010 | 103 | 0.000144 |  | 0.918 |  | 40 | PASS |  |
| DelftBlue | tworoom_policy_transformer_c16k2_seed0 | policy(transformer,c16,k2) | 0.1 | 16410 | 124 | 0.000115 |  | 0.924 |  | 35 | PASS |  |
| DelftBlue | tworoom_policy_transformer_c16k3_seed0 | policy(transformer,c16,k3) | 0.1 | 11160 | 54.1 | 3.34e-05 |  | 0.928 |  | 38 | PASS |  |
| DelftBlue | tworoom_policy_transformer_c2k1_seed0 | policy(transformer,c2,k1) | 0.1 | 29200 | 27.9 | 2.74e-05 |  | 0.951 |  | 84 | PASS |  |
| DelftBlue | tworoom_policy_transformer_c4k1_seed0 | policy(transformer,c4,k1) | 0.1 | 25690 | 64.2 | 5.54e-05 |  | 0.933 |  | 45 | PASS |  |

### Reacher

| cluster | run | method | λ | step | eff rank | pred loss | inv loss | policy loss | goal loss | success | verdict | other evals |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DelftBlue | reacher_goalpolicy_g123_seed0 | goal_policy(G=[1,2,3],k1) | 5 | 63980 | 18.2 | 0.001 |  |  | 0.884 | 69 | PASS |  |
| DelftBlue | reacher_goalpolicy_g2_seed0 | goal_policy(G=[2],k1) | 5 | 65740 | 18.7 | 0.00147 |  |  | 0.892 | 71 | PASS |  |
| DelftBlue | reacher_goalpolicy_g3_seed0 | goal_policy(G=[3],k1) | 5 | 63980 | 19.9 | 0.00145 |  |  | 0.928 | 69 | PASS |  |
| DelftBlue | reacher_inverse_lambda_1_seed0 | inverse | 5 | 67500 | 17.1 | 0.00132 | 0.832 |  |  | 68 | PASS |  |
| DelftBlue | reacher_inverse_lambda_1_seed1 | inverse | 5 | 67500 | 19.3 | 0.00134 | 0.832 |  |  | 74 | PASS | seed_0=64 |
| DelftBlue | reacher_invpolicy_mlp_c2k1_seed0 | inverse+policy(mlp,c2,k1) | 5/5 | 67500 | 18 | 0.0013 | 0.832 | 0.999 |  | 72 | PASS |  |
| DelftBlue | reacher_policy_mamba_c4k1_seed0 | policy(mamba,c4,k1) | 5 | 63980 | 124 | 0.000672 |  | 0.998 |  | 10 | PASS |  |
| DelftBlue | reacher_policy_mlp_c1k1_seed0 | policy(mlp,c1,k1) | 5 | 67500 | 160 | 5.56e-06 |  | 1 |  | 14 | PASS |  |
| DelftBlue | reacher_policy_mlp_c2k1_lam0p5_seed0 | policy(mlp,c2,k1) | 0.5 | 67500 | 169 | 3.97e-06 |  | 0.999 |  | 16 | PASS |  |
| DelftBlue | reacher_policy_mlp_c2k1_lam500p0_seed0 | policy(mlp,c2,k1) | 500 | 67500 | 141 | 0.019 |  | 0.999 |  | 8 | PASS |  |
| DelftBlue | reacher_policy_mlp_c2k1_lam50p0_seed0 | policy(mlp,c2,k1) | 50 | 67500 | 145 | 0.00448 |  | 0.999 |  | 16 | PASS |  |
| DelftBlue | reacher_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1) | 5 | 67500 | 154 | 3.29e-06 |  | 0.999 |  | 10 | PASS |  |
| DelftBlue | reacher_policy_mlp_c2k1_seed0 | policy(mlp,c2,k1) | 5 | 67500 | 149 | 8.48e-06 |  | 0.999 |  | 11 | PASS |  |
| DelftBlue | reacher_policy_transformer_c4k1_seed0 | policy(transformer,c4,k1) | 5 | 63980 | 163 | 1.01e-05 |  | 0.999 |  | 10 | PASS |  |

### Push-T

| cluster | run | method | λ | step | eff rank | pred loss | inv loss | policy loss | goal loss | success | verdict | other evals |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DelftBlue | pusht_goalpolicy_g123_seed0 | goal_policy(G=[1,2,3],k1) | 30 | 69640 | 66 | 0.0331 |  |  | 0.109 | 85 | PASS |  |
| DelftBlue | pusht_goalpolicy_g2_seed0 | goal_policy(G=[2],k1) | 30 | 72920 | 68.9 | 0.042 |  |  | 0.12 | 87 | PASS |  |
| DelftBlue | pusht_goalpolicy_g3_seed0 | goal_policy(G=[3],k1) | 30 | 69640 | 80 | 0.0503 |  |  | 0.13 | 84 | PASS |  |
| DelftBlue | pusht_inverse_lambda_30_seed0 | inverse | 30 | 76210 | 49 | 0.0287 | 0.0925 |  |  | 89 | PASS |  |
| DelftBlue | pusht_invpolicy_mlp_c2k1_seed0 | inverse+policy(mlp,c2,k1) | 30/30 | 76210 | 81.8 | 0.0492 | 0.0927 | 0.144 |  | 82 | PASS |  |
| DelftBlue | pusht_policy_mamba_c4k1_seed0 | policy(mamba,c4,k1) | 30 | 69640 | 114 | 0.0237 |  | 0.0981 |  | 17 | PASS |  |
| DelftBlue | pusht_policy_mlp_c1k1_seed0 | policy(mlp,c1,k1) | 30 | 76210 | 110 | 0.0639 |  | 0.206 |  | 57 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam100p0_seed0 | policy(mlp,c2,k1) | 100 | 76210 | 119 | 0.0639 |  | 0.136 |  | 43 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam10p0_seed0 | policy(mlp,c2,k1) | 10 | 76210 | 113 | 0.0251 |  | 0.136 |  | 45 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam10p0_seed1 | policy(mlp,c2,k1) | 10 | 76210 | 118 | 0.026 |  | 0.137 |  | 34 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam20p0_seed0 | policy(mlp,c2,k1) | 20 | 76210 | 116 | 0.0363 |  | 0.136 |  | 40 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam20p0_seed1 | policy(mlp,c2,k1) | 20 | 76210 | 118 | 0.0373 |  | 0.134 |  | 39 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam300p0_seed0 | policy(mlp,c2,k1) | 300 | 76210 | 120 | 0.0761 |  | 0.137 |  | 40 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam3p0_seed0 | policy(mlp,c2,k1) | 3 | 76210 | 108 | 0.0112 |  | 0.138 |  | 39 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam5p0_seed0 | policy(mlp,c2,k1) | 5 | 76210 | 111 | 0.0163 |  | 0.137 |  | 36 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam5p0_seed1 | policy(mlp,c2,k1) | 5 | 76210 | 114 | 0.0167 |  | 0.139 |  | 35 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1) | 30 | 76210 | 92.2 | 0.0529 |  | 0.152 |  | 69 | PASS |  |
| Leonardo | pusht_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1) | 30 | 76210 | 90.5 | 0.0524 |  | 0.15 |  | 69 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_noact_seed1 | policy(mlp,c2,k1) | 30 | 76210 | 92 | 0.0539 |  | 0.146 |  | 63 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_noact_seed2 | policy(mlp,c2,k1) | 30 | 76210 | 89.8 | 0.053 |  | 0.148 |  | 64 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_seed0 | policy(mlp,c2,k1) | 30 | 76210 | 117 | 0.0435 |  | 0.136 |  | 37 | PASS |  |
| DelftBlue | pusht_policy_transformer_c4k1_seed0 | policy(transformer,c4,k1) | 30 | 69640 | 112 | 0.0217 |  | 0.0962 |  | 25 | PASS |  |

### OGBench-Cube

| cluster | run | method | λ | step | eff rank | pred loss | inv loss | policy loss | goal loss | success | verdict | other evals |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DelftBlue | cube_inverse_lambda_1_seed0 | inverse | 1 | 67500 | 39.4 | 0.00166 | 0.0863 |  |  | 81 | PASS |  |
| DelftBlue | cube_policy_mlp_c1k1_seed0 | policy(mlp,c1,k1) | 1 | 67500 | 77.4 | 0.0024 |  | 0.369 |  | 66 | PASS |  |
| DelftBlue | cube_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1) | 1 | 67500 | 39.4 | 0.00329 |  | 0.269 |  | 80 | PASS |  |
| Leonardo | cube_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1) | 1 | 67500 | 41.7 | 0.00319 |  | 0.268 |  | 79 | PASS |  |
| DelftBlue | cube_policy_mlp_c2k1_noact_seed1 | policy(mlp,c2,k1) | 1 | 67500 | 40.1 | 0.00325 |  | 0.268 |  | 81 | PASS |  |
| DelftBlue | cube_policy_mlp_c2k1_noact_seed2 | policy(mlp,c2,k1) | 1 | 67500 | 38.5 | 0.00341 |  | 0.271 |  | 73 | PASS |  |
| DelftBlue | cube_policy_mlp_c2k1_seed0 | policy(mlp,c2,k1) | 1 | 67500 | 51.1 | 0.00143 |  | 0.224 |  | 69 | PASS |  |

## Queued / running

(none)

