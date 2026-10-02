# Results

Generated 2026-10-02 14:08 from `scripts/summarize_runs.py` on: DelftBlue. Success = paper CEM protocol (100 tasks, goal 25 primitive steps ahead, budget 50), in %. Regenerate with `scripts/results_table.py` (see its docstring); do not edit by hand.

## Policy-head input ablation (seed 0)

Every regularizer uses the paper λ of its environment; `noact` and `c1` feed no actions to the head. `–` = not run; *italics* = queued or running.

| env | λ | inverse<br>`a_t = h(z_t, z_{t+1})` | policy +a_t<br>`a_{t+1} = π(z_t, z_{t+1}, a_t)` | policy noact<br>`a_{t+1} = π(z_t, z_{t+1})` | policy c1<br>`a_{t+1} = π(z_{t+1})` | forward-only (paper) |
|---|---|---|---|---|---|---|
| TwoRoom | 0.1 | 100 (DelftBlue) / *pending · Leonardo* | 89 | 100 | 88 | ~60 |
| Reacher | 5 | 68 | 11 | *pending · Leonardo* | *pending · Leonardo* | ~40 |
| Push-T | 30 | 89 | 37 | 69 (DelftBlue) / *pending · Leonardo* | *pending · Leonardo* | ~35 |
| OGBench-Cube | 1 | 81 | 69 | 80 (DelftBlue) / *pending · Leonardo* | *pending · Leonardo* | – |

## All runs

### TwoRoom

| cluster | run | method | λ | step | eff rank | pred loss | inv loss | policy loss | goal loss | success | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DelftBlue | tworoom_goalpolicy_g123_seed0 | goal_policy(G=[1,2,3],k1) | 0.1 | 25690 | 10.7 | 4e-05 |  |  | 0.83 | 100 | PASS |
| DelftBlue | tworoom_goalpolicy_g2_seed0 | goal_policy(G=[2],k1) | 0.1 | 27440 | 9.01 | 4.13e-05 |  |  | 0.854 | 100 | PASS |
| DelftBlue | tworoom_goalpolicy_g3_seed0 | goal_policy(G=[3],k1) | 0.1 | 25690 | 13 | 6.82e-05 |  |  | 0.882 | 100 | PASS |
| DelftBlue | tworoom_inverse_lambda_0p1_seed0 | inverse | 0.1 | 29200 | 12.3 | 6.97e-05 | 0.754 |  |  | 100 | PASS |
| DelftBlue | tworoom_invpolicy_mlp_c2k1_seed0 | inverse+policy(mlp,c2,k1) | 0.1/0.1 | 29200 | 16.8 | 8.27e-05 | 0.753 | 0.951 |  | 100 | PASS |
| DelftBlue | tworoom_policy_mamba_c16k1_seed0 | policy(mamba,c16,k1) | 0.1 | 22010 | 117 | 0.000206 |  | 0.911 |  | 45 | PASS |
| DelftBlue | tworoom_policy_mamba_c16k2_seed0 | policy(mamba,c16,k2) | 0.1 | 16410 | 126 | 0.000141 |  | 0.92 |  | 50 | PASS |
| DelftBlue | tworoom_policy_mamba_c16k3_seed0 | policy(mamba,c16,k3) | 0.1 | 11160 | 79.8 | 7.72e-05 |  | 0.926 |  | 47 | PASS |
| DelftBlue | tworoom_policy_mamba_c2k1_seed0 | policy(mamba,c2,k1) | 0.1 | 29200 | 37.7 | 3.37e-05 |  | 0.951 |  | 79 | PASS |
| DelftBlue | tworoom_policy_mamba_c4k1_seed0 | policy(mamba,c4,k1) | 0.1 | 25690 | 55.6 | 5.1e-05 |  | 0.931 |  | 79 | PASS |
| DelftBlue | tworoom_policy_mlp_c1k1_seed0 | policy(mlp,c1,k1) | 0.1 | 29200 | 21.5 | 3.24e-05 |  | 0.971 |  | 88 | PASS |
| DelftBlue | tworoom_policy_mlp_c16k1_seed0 | policy(mlp,c16,k1) | 0.1 | 22010 | 81.4 | 0.000154 |  | 0.916 |  | 50 | PASS |
| DelftBlue | tworoom_policy_mlp_c16k2_seed0 | policy(mlp,c16,k2) | 0.1 | 16410 | 68.8 | 0.000104 |  | 0.926 |  | 45 | PASS |
| DelftBlue | tworoom_policy_mlp_c16k3_seed0 | policy(mlp,c16,k3) | 0.1 | 11160 | 33.5 | 8.38e-05 |  | 0.931 |  | 39 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p0001_seed0 | policy(mlp,c2,k1) | 0.0001 | 29200 | 182 | 1.53e-06 |  | 0.982 |  | 31 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p0003_seed0 | policy(mlp,c2,k1) | 0.0003 | 29200 | 184 | 1.5e-06 |  | 0.982 |  | 38 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p001_seed0 | policy(mlp,c2,k1) | 0.001 | 29200 | 173 | 1.51e-06 |  | 0.982 |  | 31 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p003_seed0 | policy(mlp,c2,k1) | 0.003 | 29200 | 153 | 1.44e-06 |  | 0.982 |  | 31 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p005_seed0 | policy(mlp,c2,k1) | 0.005 | 29200 | 6.09 | 8.19e-06 |  | 0.963 |  | 44 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p005_seed1 | policy(mlp,c2,k1) | 0.005 | 29200 | 15 | 6.24e-06 |  | 0.955 |  | 53 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p005_seed2 | policy(mlp,c2,k1) | 0.005 | 29200 | 10.2 | 6.54e-06 |  | 0.954 |  | 60 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p01_seed0 | policy(mlp,c2,k1) | 0.01 | 29200 | 11.9 | 9.37e-06 |  | 0.953 |  | 79 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p01_seed1 | policy(mlp,c2,k1) | 0.01 | 29200 | 23.4 | 1.21e-05 |  | 0.954 |  | 37 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p01_seed2 | policy(mlp,c2,k1) | 0.01 | 29200 | 12.2 | 1.03e-05 |  | 0.953 |  | 82 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p02_seed0 | policy(mlp,c2,k1) | 0.02 | 29200 | 17.2 | 1.57e-05 |  | 0.952 |  | 49 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p02_seed1 | policy(mlp,c2,k1) | 0.02 | 29200 | 16.3 | 1.76e-05 |  | 0.953 |  | 81 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p02_seed2 | policy(mlp,c2,k1) | 0.02 | 29200 | 12.5 | 1.61e-05 |  | 0.952 |  | 89 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p03_seed0 | policy(mlp,c2,k1) | 0.03 | 29200 | 17.2 | 1.98e-05 |  | 0.952 |  | 94 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p03_seed1 | policy(mlp,c2,k1) | 0.03 | 29200 | 19.6 | 2.37e-05 |  | 0.953 |  | 41 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p03_seed2 | policy(mlp,c2,k1) | 0.03 | 29200 | 16.9 | 1.83e-05 |  | 0.952 |  | 70 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p05_seed0 | policy(mlp,c2,k1) | 0.05 | 29200 | 20 | 2.88e-05 |  | 0.952 |  | 68 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p05_seed1 | policy(mlp,c2,k1) | 0.05 | 29200 | 25.1 | 3.74e-05 |  | 0.953 |  | 38 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p05_seed2 | policy(mlp,c2,k1) | 0.05 | 29200 | 23.8 | 3e-05 |  | 0.952 |  | 72 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_lam0p1_seed1 | policy(mlp,c2,k1) | 0.1 | 29200 | 25.6 | 5.03e-05 |  | 0.952 |  | 95 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1) | 0.1 | 29200 | 13.2 | 5.53e-05 |  | 0.952 |  | 100 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_seed0 | policy(mlp,c2,k1) | 0.1 | 29200 | 27.2 | 4.79e-05 |  | 0.952 |  | 89 | PASS |
| DelftBlue | tworoom_policy_mlp_c2k1_seed2 | policy(mlp,c2,k1) | 0.1 | 29200 | 28.4 | 4.9e-05 |  | 0.952 |  | 87 | PASS |
| DelftBlue | tworoom_policy_mlp_c4k1_seed0 | policy(mlp,c4,k1) | 0.1 | 25690 | 35 | 6.22e-05 |  | 0.934 |  | 82 | PASS |
| DelftBlue | tworoom_policy_transformer_c16k1_seed0 | policy(transformer,c16,k1) | 0.1 | 22010 | 103 | 0.000144 |  | 0.918 |  | 40 | PASS |
| DelftBlue | tworoom_policy_transformer_c16k2_seed0 | policy(transformer,c16,k2) | 0.1 | 16410 | 124 | 0.000115 |  | 0.924 |  | 35 | PASS |
| DelftBlue | tworoom_policy_transformer_c16k3_seed0 | policy(transformer,c16,k3) | 0.1 | 11160 | 54.1 | 3.34e-05 |  | 0.928 |  | 38 | PASS |
| DelftBlue | tworoom_policy_transformer_c2k1_seed0 | policy(transformer,c2,k1) | 0.1 | 29200 | 27.9 | 2.74e-05 |  | 0.951 |  | 84 | PASS |
| DelftBlue | tworoom_policy_transformer_c4k1_seed0 | policy(transformer,c4,k1) | 0.1 | 25690 | 64.2 | 5.54e-05 |  | 0.933 |  | 45 | PASS |

### Reacher

| cluster | run | method | λ | step | eff rank | pred loss | inv loss | policy loss | goal loss | success | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DelftBlue | reacher_goalpolicy_g2_seed0 | goal_policy(G=[2],k1) | 5 | 65740 | 18.7 | 0.00147 |  |  | 0.892 | 71 | PASS |
| DelftBlue | reacher_inverse_lambda_1_seed0 | inverse | 5 | 67500 | 17.1 | 0.00132 | 0.832 |  |  | 68 | PASS |
| DelftBlue | reacher_inverse_lambda_1_seed1 | inverse | 5 | 67500 | 19.3 | 0.00134 | 0.832 |  |  | 64/74 | PASS |
| DelftBlue | reacher_invpolicy_mlp_c2k1_seed0 | inverse+policy(mlp,c2,k1) | 5/5 | 67500 | 18 | 0.0013 | 0.832 | 0.999 |  | 72 | PASS |
| DelftBlue | reacher_policy_mamba_c4k1_seed0 | policy(mamba,c4,k1) | 5 | 63980 | 124 | 0.000672 |  | 0.998 |  | 10 | PASS |
| DelftBlue | reacher_policy_mlp_c2k1_lam0p5_seed0 | policy(mlp,c2,k1) | 0.5 | 67500 | 169 | 3.97e-06 |  | 0.999 |  | 16 | PASS |
| DelftBlue | reacher_policy_mlp_c2k1_lam500p0_seed0 | policy(mlp,c2,k1) | 500 | 67500 | 141 | 0.019 |  | 0.999 |  | 8 | PASS |
| DelftBlue | reacher_policy_mlp_c2k1_lam50p0_seed0 | policy(mlp,c2,k1) | 50 | 67500 | 145 | 0.00448 |  | 0.999 |  | 16 | PASS |
| DelftBlue | reacher_policy_mlp_c2k1_seed0 | policy(mlp,c2,k1) | 5 | 67500 | 149 | 8.48e-06 |  | 0.999 |  | 11 | PASS |
| DelftBlue | reacher_policy_transformer_c4k1_seed0 | policy(transformer,c4,k1) | 5 | 63980 | 163 | 1.01e-05 |  | 0.999 |  | 10 | PASS |

### Push-T

| cluster | run | method | λ | step | eff rank | pred loss | inv loss | policy loss | goal loss | success | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DelftBlue | pusht_goalpolicy_g2_seed0 | goal_policy(G=[2],k1) | 30 | 72920 | 68.9 | 0.042 |  |  | 0.12 | 87 | PASS |
| DelftBlue | pusht_inverse_lambda_30_seed0 | inverse | 30 | 76210 | 49 | 0.0287 | 0.0925 |  |  | 89 | PASS |
| DelftBlue | pusht_invpolicy_mlp_c2k1_seed0 | inverse+policy(mlp,c2,k1) | 30/30 | 76210 | 81.8 | 0.0492 | 0.0927 | 0.144 |  | 82 | PASS |
| DelftBlue | pusht_policy_mamba_c4k1_seed0 | policy(mamba,c4,k1) | 30 | 69640 | 114 | 0.0237 |  | 0.0981 |  | 17 | PASS |
| DelftBlue | pusht_policy_mlp_c2k1_lam100p0_seed0 | policy(mlp,c2,k1) | 100 | 76210 | 119 | 0.0639 |  | 0.136 |  | 43 | PASS |
| DelftBlue | pusht_policy_mlp_c2k1_lam10p0_seed0 | policy(mlp,c2,k1) | 10 | 76210 | 113 | 0.0251 |  | 0.136 |  | 45 | PASS |
| DelftBlue | pusht_policy_mlp_c2k1_lam300p0_seed0 | policy(mlp,c2,k1) | 300 | 76210 | 120 | 0.0761 |  | 0.137 |  | 40 | PASS |
| DelftBlue | pusht_policy_mlp_c2k1_lam3p0_seed0 | policy(mlp,c2,k1) | 3 | 76210 | 108 | 0.0112 |  | 0.138 |  | 39 | PASS |
| DelftBlue | pusht_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1) | 30 | 76210 | 92.2 | 0.0529 |  | 0.152 |  | 69 | PASS |
| DelftBlue | pusht_policy_mlp_c2k1_seed0 | policy(mlp,c2,k1) | 30 | 76210 | 117 | 0.0435 |  | 0.136 |  | 37 | PASS |
| DelftBlue | pusht_policy_transformer_c4k1_seed0 | policy(transformer,c4,k1) | 30 | 69640 | 112 | 0.0217 |  | 0.0962 |  | 25 | PASS |

### OGBench-Cube

| cluster | run | method | λ | step | eff rank | pred loss | inv loss | policy loss | goal loss | success | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DelftBlue | cube_inverse_lambda_1_seed0 | inverse | 1 | 67500 | 39.4 | 0.00166 | 0.0863 |  |  | 81 | PASS |
| DelftBlue | cube_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1) | 1 | 67500 | 39.4 | 0.00329 |  | 0.269 |  | 80 | PASS |
| DelftBlue | cube_policy_mlp_c2k1_seed0 | policy(mlp,c2,k1) | 1 | 67500 | 51.1 | 0.00143 |  | 0.224 |  | 69 | PASS |

## Queued / running

| cluster | job | name | state | elapsed |
|---|---|---|---|---|
| Leonardo | 59127409 | te-tworoom_inverse_lambda_0p1_seed0 | PENDING | 0:00 |
| Leonardo | 59127422 | te-cube_policy_mlp_c1k1_seed0 | PENDING | 0:00 |
| Leonardo | 59127416 | te-cube_policy_mlp_c2k1_noact_seed0 | PENDING | 0:00 |
| Leonardo | 59127413 | te-reacher_policy_mlp_c1k1_seed0 | PENDING | 0:00 |
| Leonardo | 59127412 | te-reacher_policy_mlp_c2k1_noact_seed0 | PENDING | 0:00 |
| Leonardo | 59127415 | te-pusht_policy_mlp_c1k1_seed0 | PENDING | 0:00 |
| Leonardo | 59127414 | te-pusht_policy_mlp_c2k1_noact_seed0 | PENDING | 0:00 |

