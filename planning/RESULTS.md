# Results

Generated 2026-10-26 13:53 from `scripts/summarize_runs.py` on: DelftBlue, Leonardo. Success = paper CEM protocol (100 tasks, goal 25 primitive steps ahead, budget 50), in %. Regenerate with `scripts/results_table.py` (see its docstring); do not edit by hand.

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
| TwoRoom | tworoom_policy_mlp_c1k1 | DelftBlue | 3 | 85.7 ± 14.6 | s0: 88, s1: 70, s2: 99 |
| TwoRoom | tworoom_policy_mlp_c2k1 | DelftBlue | 2 | 88.0 ± 1.4 | s0: 89, s2: 87 |
| TwoRoom | tworoom_policy_mlp_c2k1_lam0p005 | DelftBlue | 3 | 52.3 ± 8.0 | s0: 44, s1: 53, s2: 60 |
| TwoRoom | tworoom_policy_mlp_c2k1_lam0p01 | DelftBlue | 3 | 66.0 ± 25.2 | s0: 79, s1: 37, s2: 82 |
| TwoRoom | tworoom_policy_mlp_c2k1_lam0p02 | DelftBlue | 3 | 73.0 ± 21.2 | s0: 49, s1: 81, s2: 89 |
| TwoRoom | tworoom_policy_mlp_c2k1_lam0p03 | DelftBlue | 3 | 68.3 ± 26.5 | s0: 94, s1: 41, s2: 70 |
| TwoRoom | tworoom_policy_mlp_c2k1_lam0p05 | DelftBlue | 3 | 59.3 ± 18.6 | s0: 68, s1: 38, s2: 72 |
| TwoRoom | tworoom_policy_mlp_c2k1_noact | DelftBlue | 3 | 100.0 ± 0.0 | s0: 100, s1: 100, s2: 100 |
| Reacher | reacher_expert_inverse | DelftBlue | 3 | 59.3 ± 3.1 | s0: 60, s1: 62, s2: 56 |
| Reacher | reacher_expert_policy_mlp_c2k1_noact | DelftBlue | 3 | 59.0 ± 3.0 | s0: 62, s1: 56, s2: 59 |
| Reacher | reacher_expert_sigreg | DelftBlue | 3 | 58.0 ± 3.6 | s0: 61, s1: 59, s2: 54 |
| Reacher | reacher_inverse_lambda_1 | DelftBlue | 2 | 71.0 ± 4.2 | s0: 68, s1: 74 |
| Reacher | reacher_policy_mlp_c1k1 | DelftBlue | 3 | 11.0 ± 3.0 | s0: 14, s1: 11, s2: 8 |
| Reacher | reacher_policy_mlp_c2k1_noact | DelftBlue | 3 | 12.7 ± 2.5 | s0: 10, s1: 13, s2: 15 |
| Reacher | reacher_policy_mlp_c2k1_noact_delta_goalfinal | DelftBlue | 3 | 35.3 ± 2.1 | s0: 33, s1: 36, s2: 37 |
| Push-T | pusht_policy_mlp_c1k1 | DelftBlue | 3 | 59.3 ± 2.5 | s0: 57, s1: 59, s2: 62 |
| Push-T | pusht_policy_mlp_c2k1_lam10p0 | DelftBlue | 2 | 39.5 ± 7.8 | s0: 45, s1: 34 |
| Push-T | pusht_policy_mlp_c2k1_lam20p0 | DelftBlue | 2 | 39.5 ± 0.7 | s0: 40, s1: 39 |
| Push-T | pusht_policy_mlp_c2k1_lam5p0 | DelftBlue | 2 | 35.5 ± 0.7 | s0: 36, s1: 35 |
| Push-T | pusht_policy_mlp_c2k1_noact | DelftBlue | 3 | 65.3 ± 3.2 | s0: 69, s1: 63, s2: 64 |
| Push-T | pusht_policy_mlp_c2k1_noact_delta_goalfinal | Leonardo | 3 | 83.7 ± 2.3 | s0: 85, s1: 85, s2: 81 |
| Push-T | pusht_policy_mlp_c2k1_noact_goalfinal | Leonardo | 3 | 77.3 ± 3.1 | s0: 78, s1: 74, s2: 80 |
| OGBench-Cube | cube_policy_mlp_c1k1 | DelftBlue | 3 | 68.7 ± 3.1 | s0: 66, s1: 72, s2: 68 |
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
| Leonardo | tworoom_h3_inverse_seed0 | inverse [h=3] | 0.1 | 51380 | 13.1 | 7.29e-05 | 0.754 |  |  | 100 | DEVIATE |  |
| DelftBlue | tworoom_invpolicy_mlp_c2k1_seed0 | inverse+policy(mlp,c2,k1) | 0.1/0.1 | 29200 | 16.8 | 8.27e-05 | 0.753 | 0.951 |  | 100 | PASS |  |
| DelftBlue | tworoom_policy_mamba_c16k1_seed0 | policy(mamba,c16,k1) | 0.1 | 22010 | 117 | 0.000206 |  | 0.911 |  | 45 | PASS |  |
| DelftBlue | tworoom_policy_mamba_c16k2_seed0 | policy(mamba,c16,k2) | 0.1 | 16410 | 126 | 0.000141 |  | 0.92 |  | 50 | PASS |  |
| DelftBlue | tworoom_policy_mamba_c16k3_seed0 | policy(mamba,c16,k3) | 0.1 | 11160 | 79.8 | 7.72e-05 |  | 0.926 |  | 47 | PASS |  |
| DelftBlue | tworoom_policy_mamba_c2k1_seed0 | policy(mamba,c2,k1) | 0.1 | 29200 | 37.7 | 3.37e-05 |  | 0.951 |  | 79 | PASS |  |
| DelftBlue | tworoom_policy_mamba_c4k1_seed0 | policy(mamba,c4,k1) | 0.1 | 25690 | 55.6 | 5.1e-05 |  | 0.931 |  | 79 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c1k1_seed0 | policy(mlp,c1,k1,noact) | 0.1 | 29200 | 21.5 | 3.24e-05 |  | 0.971 |  | 88 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c1k1_seed1 | policy(mlp,c1,k1,noact) | 0.1 | 29200 | 28.7 | 3.93e-05 |  | 0.971 |  | 70 | PASS | rerun_job_930925=68 |
| DelftBlue | tworoom_policy_mlp_c1k1_seed2 | policy(mlp,c1,k1,noact) | 0.1 | 29200 | 20 | 2.74e-05 |  | 0.971 |  | 99 | PASS | rerun_job_930926=98 |
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
| DelftBlue | tworoom_policy_mlp_c2k1_seed0 | policy(mlp,c2,k1) | 0.1 | 29200 | 27.2 | 4.79e-05 |  | 0.952 |  | 89 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_seed2 | policy(mlp,c2,k1) | 0.1 | 29200 | 28.4 | 4.9e-05 |  | 0.952 |  | 87 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_noact_delta_seed0 | policy(mlp,c2,k1,delta,noact) | 0.1 | 29200 | 18.1 | 0.000176 |  | 0.964 |  | 66 | PASS |  |
| Leonardo | tworoom_h3_policy_mlp_c2k1_noact_delta_seed0 | policy(mlp,c2,k1,delta,noact) [h=3] | 0.1 | 51380 | 10.8 | 0.000143 |  | 0.954 |  | 66 | DEVIATE |  |
| DelftBlue | tworoom_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | policy(mlp,c2,k1,delta,noact,Gfinal) | 0.1 | 29200 | 16.5 | 6.24e-05 |  | 0.908 |  | 100 | PASS |  |
| Leonardo | tworoom_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | policy(mlp,c2,k1,delta,noact,Gfinal) [h=3] | 0.1 | 51380 | 29.1 | 7.85e-05 |  | 0.917 |  | 88 | DEVIATE |  |
| DelftBlue | tworoom_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) | 0.1 | 29200 | 13.2 | 5.53e-05 |  | 0.952 |  | 100 | PASS |  |
| DelftBlue | tworoom_policy_mlp_c2k1_noact_seed1 | policy(mlp,c2,k1,noact) | 0.1 | 29200 | 13.3 | 7.49e-05 |  | 0.953 |  | 100 | PASS | rerun_job_930921=100 |
| DelftBlue | tworoom_policy_mlp_c2k1_noact_seed2 | policy(mlp,c2,k1,noact) | 0.1 | 29200 | 11.3 | 5.23e-05 |  | 0.952 |  | 100 | PASS | rerun_job_930922=100 |
| Leonardo | tworoom_h3_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) [h=3] | 0.1 | 51380 | 13.1 | 6.1e-05 |  | 0.948 |  | 99 | DEVIATE |  |
| DelftBlue | tworoom_policy_mlp_c2k1_noact_goalfinal_seed0 | policy(mlp,c2,k1,noact,Gfinal) | 0.1 | 29200 | 15.5 | 6.39e-05 |  | 0.909 |  | 99 | PASS |  |
| Leonardo | tworoom_h3_policy_mlp_c2k1_noact_goalfinal_seed0 | policy(mlp,c2,k1,noact,Gfinal) [h=3] | 0.1 | 51380 | 16.9 | 4.98e-05 |  | 0.915 |  | 99 | DEVIATE |  |
| DelftBlue | tworoom_policy_mlp_c4k1_seed0 | policy(mlp,c4,k1) | 0.1 | 25690 | 35 | 6.22e-05 |  | 0.934 |  | 82 | PASS |  |
| DelftBlue | tworoom_policy_transformer_c16k1_seed0 | policy(transformer,c16,k1) | 0.1 | 22010 | 103 | 0.000144 |  | 0.918 |  | 40 | PASS |  |
| DelftBlue | tworoom_policy_transformer_c16k2_seed0 | policy(transformer,c16,k2) | 0.1 | 16410 | 124 | 0.000115 |  | 0.924 |  | 35 | PASS |  |
| DelftBlue | tworoom_policy_transformer_c16k3_seed0 | policy(transformer,c16,k3) | 0.1 | 11160 | 54.1 | 3.34e-05 |  | 0.928 |  | 38 | PASS |  |
| DelftBlue | tworoom_policy_transformer_c2k1_seed0 | policy(transformer,c2,k1) | 0.1 | 29200 | 27.9 | 2.74e-05 |  | 0.951 |  | 84 | PASS |  |
| DelftBlue | tworoom_policy_transformer_c4k1_seed0 | policy(transformer,c4,k1) | 0.1 | 25690 | 64.2 | 5.54e-05 |  | 0.933 |  | 45 | PASS |  |
| Leonardo | tworoom_h3_sigreg_seed0 | sigreg [h=3] | 0.09 | 51380 | 74.1 | 0.0141 |  |  |  | 90 | DEVIATE |  |

### Reacher

| cluster | run | method | λ | step | eff rank | pred loss | inv loss | policy loss | goal loss | success | verdict | other evals |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DelftBlue | reacher_goalpolicy_g123_seed0 | goal_policy(G=[1,2,3],k1) | 5 | 63980 | 18.2 | 0.001 |  |  | 0.884 | 69 | PASS |  |
| DelftBlue | reacher_goalpolicy_g2_seed0 | goal_policy(G=[2],k1) | 5 | 65740 | 18.7 | 0.00147 |  |  | 0.892 | 71 | PASS |  |
| DelftBlue | reacher_goalpolicy_g3_seed0 | goal_policy(G=[3],k1) | 5 | 63980 | 19.9 | 0.00145 |  |  | 0.928 | 69 | PASS |  |
| DelftBlue | reacher_inverse_lambda_1_seed0 | inverse | 5 | 67500 | 17.1 | 0.00132 | 0.832 |  |  | 68 | PASS |  |
| DelftBlue | reacher_inverse_lambda_1_seed1 | inverse | 5 | 67500 | 19.3 | 0.00134 | 0.832 |  |  | 74 | PASS | seed_0=64 |
| DelftBlue | reacher_expert_inverse_seed0 | inverse [data=reacher_expert_train] | 5 | 67500 | 17.3 | 0.00422 | 0.369 |  |  | 60 | PASS |  |
| DelftBlue | reacher_expert_inverse_seed1 | inverse [data=reacher_expert_train] | 5 | 67500 | 18.1 | 0.00425 | 0.369 |  |  | 62 | PASS |  |
| DelftBlue | reacher_expert_inverse_seed2 | inverse [data=reacher_expert_train] | 5 | 67500 | 19.9 | 0.00435 | 0.369 |  |  | 56 | PASS |  |
| DelftBlue | reacher_expert_h3_inverse_seed0 | inverse [data=reacher_expert_train] [h=3] | 5 | 127960 | 21.8 | 0.00281 | 0.37 |  |  | 90 | DEVIATE |  |
| DelftBlue | reacher_mix_inverse_seed0 | inverse [data=reacher_mix_train] | 5 | 67500 | 18.2 | 0.00252 | 0.613 |  |  | 62 | PASS |  |
| DelftBlue | reacher_mix_h3_inverse_seed0 | inverse [data=reacher_mix_train] [h=3] | 5 | 127960 | 20.6 | 0.00169 | 0.614 |  |  | 96 | DEVIATE |  |
| DelftBlue | reacher_rand2x_inverse_seed0 | inverse [data=reacher_rand2x_train] | 5 | 135000 | 23.4 | 0.000848 | 0.832 |  |  | 67 | PASS |  |
| DelftBlue | reacher_randlocal9k_inverse_seed0 | inverse [data=reacher_randlocal9k_train] | 5 | 67500 | 24.7 | 0.00168 | 0.828 |  |  | 75 | PASS |  |
| DelftBlue | reacher_randlocal_inverse_seed0 | inverse [data=reacher_randlocal_train] | 5 | 135000 | 22.2 | 0.00102 | 0.828 |  |  | 70 | PASS |  |
| DelftBlue | reacher_h3_inverse_seed0 | inverse [h=3] | 5 | 127960 |  |  |  |  |  | 96 | DEVIATE |  |
| Leonardo | reacher_h3_inverse_seed0 | inverse [h=3] | 5 | 127960 | 24.1 | 0.0012 | 0.832 |  |  |  | DEVIATE |  |
| DelftBlue | reacher_invpolicy_mlp_c2k1_seed0 | inverse+policy(mlp,c2,k1) | 5/5 | 67500 | 18 | 0.0013 | 0.832 | 0.999 |  | 72 | PASS |  |
| DelftBlue | reacher_policy_mamba_c4k1_seed0 | policy(mamba,c4,k1) | 5 | 63980 | 124 | 0.000672 |  | 0.998 |  | 10 | PASS |  |
| DelftBlue | reacher_policy_mlp_c1k1_seed0 | policy(mlp,c1,k1,noact) | 5 | 67500 | 160 | 5.56e-06 |  | 1 |  | 14 | PASS |  |
| DelftBlue | reacher_policy_mlp_c1k1_seed1 | policy(mlp,c1,k1,noact) | 5 | 67500 | 151 | 5.11e-06 |  | 1 |  | 11 | PASS | rerun_job_930930=15 |
| DelftBlue | reacher_policy_mlp_c1k1_seed2 | policy(mlp,c1,k1,noact) | 5 | 67500 | 151 | 4.3e-06 |  | 1 |  | 8 | PASS | rerun_job_930931=12 |
| DelftBlue | reacher_policy_mlp_c2k1_lam0p5_seed0 | policy(mlp,c2,k1) | 0.5 | 67500 | 169 | 3.97e-06 |  | 0.999 |  | 16 | PASS |  |
| DelftBlue | reacher_policy_mlp_c2k1_lam500p0_seed0 | policy(mlp,c2,k1) | 500 | 67500 | 141 | 0.019 |  | 0.999 |  | 8 | PASS |  |
| DelftBlue | reacher_policy_mlp_c2k1_lam50p0_seed0 | policy(mlp,c2,k1) | 50 | 67500 | 145 | 0.00448 |  | 0.999 |  | 16 | PASS |  |
| DelftBlue | reacher_policy_mlp_c2k1_seed0 | policy(mlp,c2,k1) | 5 | 67500 | 149 | 8.48e-06 |  | 0.999 |  | 11 | PASS |  |
| DelftBlue | reacher_expert_policy_mlp_c2k1_seed0 | policy(mlp,c2,k1) [data=reacher_expert_train] | 5 | 67500 | 10.6 | 0.00166 |  | 0.553 |  | 17 | PASS |  |
| DelftBlue | reacher_policy_mlp_c2k1_noact_delta_seed0 | policy(mlp,c2,k1,delta,noact) | 5 | 67500 | 80.8 | 1.24e-05 |  | 0.999 |  | 4 | PASS |  |
| DelftBlue | reacher_expert_policy_mlp_c2k1_noact_delta_seed0 | policy(mlp,c2,k1,delta,noact) [data=reacher_expert_train] | 5 | 67500 | 32.2 | 0.00654 |  | 0.681 |  | 52 | PASS |  |
| DelftBlue | reacher_expert_h3_policy_mlp_c2k1_noact_delta_seed0 | policy(mlp,c2,k1,delta,noact) [data=reacher_expert_train] [h=3] | 5 | 127960 | 36.6 | 0.00597 |  | 0.68 |  | 83 | DEVIATE |  |
| DelftBlue | reacher_mix_h3_policy_mlp_c2k1_noact_delta_seed0 | policy(mlp,c2,k1,delta,noact) [data=reacher_mix_train] [h=3] | 5 | 127960 | 30.3 | 0.00369 |  | 0.862 |  | 85 | DEVIATE |  |
| DelftBlue | reacher_h3_policy_mlp_c2k1_noact_delta_seed0 | policy(mlp,c2,k1,delta,noact) [h=3] | 5 | 127960 | 115 | 0.000209 |  | 1 |  | 10 | DEVIATE |  |
| DelftBlue | reacher_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | policy(mlp,c2,k1,delta,noact,Gfinal) | 5 | 67500 | 19.6 | 0.00107 |  | 0.989 |  | 33 | PASS |  |
| DelftBlue | reacher_policy_mlp_c2k1_noact_delta_goalfinal_seed1 | policy(mlp,c2,k1,delta,noact,Gfinal) | 5 | 67500 | 16.6 | 0.00103 |  | 0.989 |  | 36 | PASS |  |
| DelftBlue | reacher_policy_mlp_c2k1_noact_delta_goalfinal_seed2 | policy(mlp,c2,k1,delta,noact,Gfinal) | 5 | 67500 | 17.8 | 0.000989 |  | 0.989 |  | 37 | PASS |  |
| DelftBlue | reacher_expert_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | policy(mlp,c2,k1,delta,noact,Gfinal) [data=reacher_expert_train] | 5 | 67500 | 34.4 | 0.00719 |  | 0.677 |  | 52 | PASS |  |
| DelftBlue | reacher_expert_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | policy(mlp,c2,k1,delta,noact,Gfinal) [data=reacher_expert_train] [h=3] | 5 | 127960 | 36.8 | 0.00656 |  | 0.677 |  | 84 | DEVIATE |  |
| DelftBlue | reacher_mix_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | policy(mlp,c2,k1,delta,noact,Gfinal) [data=reacher_mix_train] [h=3] | 5 | 127960 | 37.4 | 0.00504 |  | 0.851 |  | 85 | DEVIATE |  |
| DelftBlue | reacher_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | policy(mlp,c2,k1,delta,noact,Gfinal) [h=3] | 5 | 127960 | 136 | 0.000537 |  | 0.999 |  | 10 | DEVIATE |  |
| DelftBlue | reacher_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) | 5 | 67500 | 154 | 3.29e-06 |  | 0.999 |  | 10 | PASS |  |
| DelftBlue | reacher_policy_mlp_c2k1_noact_seed1 | policy(mlp,c2,k1,noact) | 5 | 67500 | 156 | 4.04e-06 |  | 0.999 |  | 13 | PASS | rerun_job_930923=13 |
| DelftBlue | reacher_policy_mlp_c2k1_noact_seed2 | policy(mlp,c2,k1,noact) | 5 | 67500 | 166 | 2.5e-06 |  | 0.999 |  | 15 | PASS | rerun_job_930924=15 |
| DelftBlue | reacher_expert_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) [data=reacher_expert_train] | 5 | 67500 | 20.9 | 0.00485 |  | 0.68 |  | 62 | PASS |  |
| DelftBlue | reacher_expert_policy_mlp_c2k1_noact_seed1 | policy(mlp,c2,k1,noact) [data=reacher_expert_train] | 5 | 67500 | 22 | 0.00484 |  | 0.681 |  | 56 | PASS |  |
| DelftBlue | reacher_expert_policy_mlp_c2k1_noact_seed2 | policy(mlp,c2,k1,noact) [data=reacher_expert_train] | 5 | 67500 | 21.5 | 0.00491 |  | 0.681 |  | 59 | PASS |  |
| DelftBlue | reacher_expert_h3_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) [data=reacher_expert_train] [h=3] | 5 | 127960 | 22.7 | 0.00426 |  | 0.68 |  | 81 | DEVIATE |  |
| DelftBlue | reacher_mix_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) [data=reacher_mix_train] | 5 | 67500 | 19.6 | 0.00278 |  | 0.861 |  | 66 | PASS |  |
| DelftBlue | reacher_mix_h3_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) [data=reacher_mix_train] [h=3] | 5 | 127960 | 25.6 | 0.00287 |  | 0.862 |  | 88 | DEVIATE |  |
| DelftBlue | reacher_h3_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) [h=3] | 5 | 127960 |  |  |  |  |  | 16 | DEVIATE |  |
| Leonardo | reacher_h3_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) [h=3] | 5 | 127960 | 106 | 5.19e-05 |  | 1 |  |  | DEVIATE |  |
| DelftBlue | reacher_policy_mlp_c2k1_noact_goalfinal_seed0 | policy(mlp,c2,k1,noact,Gfinal) | 5 | 67500 | 159 | 2.1e-05 |  | 0.999 |  | 8 | PASS |  |
| DelftBlue | reacher_expert_policy_mlp_c2k1_noact_goalfinal_seed0 | policy(mlp,c2,k1,noact,Gfinal) [data=reacher_expert_train] | 5 | 67500 | 26.6 | 0.00586 |  | 0.677 |  | 53 | PASS |  |
| DelftBlue | reacher_expert_h3_policy_mlp_c2k1_noact_goalfinal_seed0 | policy(mlp,c2,k1,noact,Gfinal) [data=reacher_expert_train] [h=3] | 5 | 127960 | 29.8 | 0.00531 |  | 0.678 |  | 71 | DEVIATE |  |
| DelftBlue | reacher_mix_h3_policy_mlp_c2k1_noact_goalfinal_seed0 | policy(mlp,c2,k1,noact,Gfinal) [data=reacher_mix_train] [h=3] | 5 | 127960 | 32.3 | 0.00408 |  | 0.852 |  | 88 | DEVIATE |  |
| DelftBlue | reacher_h3_policy_mlp_c2k1_noact_goalfinal_seed0 | policy(mlp,c2,k1,noact,Gfinal) [h=3] | 5 | 127960 | 117 | 6.33e-05 |  | 1 |  | 9 | DEVIATE |  |
| DelftBlue | reacher_policy_transformer_c4k1_seed0 | policy(transformer,c4,k1) | 5 | 63980 | 163 | 1.01e-05 |  | 0.999 |  | 10 | PASS |  |
| DelftBlue | reacher_expert_sigreg_seed0 | sigreg [data=reacher_expert_train] | 0.09 | 67500 | 78.4 | 0.0808 |  |  |  | 61 | PASS |  |
| DelftBlue | reacher_expert_sigreg_seed1 | sigreg [data=reacher_expert_train] | 0.09 | 67500 | 78.2 | 0.0798 |  |  |  | 59 | PASS |  |
| DelftBlue | reacher_expert_sigreg_seed2 | sigreg [data=reacher_expert_train] | 0.09 | 67500 | 78.2 | 0.081 |  |  |  | 54 | PASS |  |
| DelftBlue | reacher_expert_h3_sigreg_seed0 | sigreg [data=reacher_expert_train] [h=3] | 0.09 | 127960 | 66.4 | 0.0408 |  |  |  | 87 | DEVIATE |  |
| DelftBlue | reacher_mix_sigreg_seed0 | sigreg [data=reacher_mix_train] | 0.09 | 67500 | 79.2 | 0.0705 |  |  |  | 65 | PASS |  |
| DelftBlue | reacher_mix_h3_sigreg_seed0 | sigreg [data=reacher_mix_train] [h=3] | 0.09 | 127960 | 57.8 | 0.033 |  |  |  | 95 | DEVIATE |  |
| DelftBlue | reacher_rand2x_sigreg_seed0 | sigreg [data=reacher_rand2x_train] | 0.09 | 135000 | 85.2 | 0.0581 |  |  |  | 66 | PASS |  |
| DelftBlue | reacher_randlocal9k_sigreg_seed0 | sigreg [data=reacher_randlocal9k_train] | 0.09 | 67500 | 82.6 | 0.0588 |  |  |  | 72 | PASS |  |
| DelftBlue | reacher_randlocal_sigreg_seed0 | sigreg [data=reacher_randlocal_train] | 0.09 | 135000 | 91 | 0.0528 |  |  |  | 74 | PASS |  |
| DelftBlue | reacher_h3_sigreg_seed0 | sigreg [h=3] | 0.09 | 127960 |  |  |  |  |  | 95 | DEVIATE |  |
| Leonardo | reacher_h3_sigreg_seed0 | sigreg [h=3] | 0.09 | 127960 | 58.1 | 0.0284 |  |  |  |  | DEVIATE |  |

### Push-T

| cluster | run | method | λ | step | eff rank | pred loss | inv loss | policy loss | goal loss | success | verdict | other evals |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DelftBlue | pusht_goalpolicy_g123_seed0 | goal_policy(G=[1,2,3],k1) | 30 | 69640 | 66 | 0.0331 |  |  | 0.109 | 85 | PASS |  |
| DelftBlue | pusht_goalpolicy_g2_seed0 | goal_policy(G=[2],k1) | 30 | 72920 | 68.9 | 0.042 |  |  | 0.12 | 87 | PASS |  |
| DelftBlue | pusht_goalpolicy_g3_seed0 | goal_policy(G=[3],k1) | 30 | 69640 | 80 | 0.0503 |  |  | 0.13 | 84 | PASS |  |
| DelftBlue | pusht_inverse_lambda_30_seed0 | inverse | 30 | 76210 | 49 | 0.0287 | 0.0925 |  |  | 89 | PASS |  |
| DelftBlue | pusht_h3_inverse_seed0 | inverse [h=3] | 30 | 139280 | 47.2 | 0.026 | 0.0844 |  |  | 74 | DEVIATE |  |
| DelftBlue | pusht_invpolicy_mlp_c2k1_seed0 | inverse+policy(mlp,c2,k1) | 30/30 | 76210 | 81.8 | 0.0492 | 0.0927 | 0.144 |  | 82 | PASS |  |
| DelftBlue | pusht_policy_mamba_c4k1_seed0 | policy(mamba,c4,k1) | 30 | 69640 | 114 | 0.0237 |  | 0.0981 |  | 17 | PASS |  |
| DelftBlue | pusht_policy_mlp_c1k1_seed0 | policy(mlp,c1,k1,noact) | 30 | 76210 | 110 | 0.0639 |  | 0.206 |  | 57 | PASS |  |
| DelftBlue | pusht_policy_mlp_c1k1_seed1 | policy(mlp,c1,k1,noact) | 30 | 76210 | 110 | 0.0638 |  | 0.203 |  | 59 | PASS | rerun_job_930927=58 |
| DelftBlue | pusht_policy_mlp_c1k1_seed2 | policy(mlp,c1,k1,noact) | 30 | 76210 | 109 | 0.0634 |  | 0.202 |  | 62 | PASS | rerun_job_930928=61 |
| DelftBlue | pusht_policy_mlp_c2k1_lam100p0_seed0 | policy(mlp,c2,k1) | 100 | 76210 | 119 | 0.0639 |  | 0.136 |  | 43 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam10p0_seed0 | policy(mlp,c2,k1) | 10 | 76210 | 113 | 0.0251 |  | 0.136 |  | 45 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam10p0_seed1 | policy(mlp,c2,k1) | 10 | 76210 | 118 | 0.026 |  | 0.137 |  | 34 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam20p0_seed0 | policy(mlp,c2,k1) | 20 | 76210 | 116 | 0.0363 |  | 0.136 |  | 40 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam20p0_seed1 | policy(mlp,c2,k1) | 20 | 76210 | 118 | 0.0373 |  | 0.134 |  | 39 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam300p0_seed0 | policy(mlp,c2,k1) | 300 | 76210 | 120 | 0.0761 |  | 0.137 |  | 40 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam3p0_seed0 | policy(mlp,c2,k1) | 3 | 76210 | 108 | 0.0112 |  | 0.138 |  | 39 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam5p0_seed0 | policy(mlp,c2,k1) | 5 | 76210 | 111 | 0.0163 |  | 0.137 |  | 36 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_lam5p0_seed1 | policy(mlp,c2,k1) | 5 | 76210 | 114 | 0.0167 |  | 0.139 |  | 35 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_seed0 | policy(mlp,c2,k1) | 30 | 76210 | 117 | 0.0435 |  | 0.136 |  | 37 | PASS |  |
| Leonardo | pusht_policy_mlp_c2k1_noact_delta_seed0 | policy(mlp,c2,k1,delta,noact) | 30 | 76210 | 68.7 | 0.0368 |  | 0.163 |  | 72 | PASS |  |
| DelftBlue | pusht_h3_policy_mlp_c2k1_noact_delta_seed0 | policy(mlp,c2,k1,delta,noact) [h=3] | 30 | 139280 | 61.2 | 0.0324 |  | 0.179 |  | 76 | DEVIATE |  |
| Leonardo | pusht_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | policy(mlp,c2,k1,delta,noact,Gfinal) | 30 | 76210 | 78.2 | 0.0448 |  | 0.14 |  | 85 | PASS |  |
| Leonardo | pusht_policy_mlp_c2k1_noact_delta_goalfinal_seed1 | policy(mlp,c2,k1,delta,noact,Gfinal) | 30 | 76210 | 80.5 | 0.0452 |  | 0.136 |  | 85 | PASS |  |
| Leonardo | pusht_policy_mlp_c2k1_noact_delta_goalfinal_seed2 | policy(mlp,c2,k1,delta,noact,Gfinal) | 30 | 76210 | 80.4 | 0.0453 |  | 0.137 |  | 81 | PASS |  |
| DelftBlue | pusht_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | policy(mlp,c2,k1,delta,noact,Gfinal) [h=3] | 30 | 139280 | 71.6 | 0.0415 |  | 0.15 |  | 69 | DEVIATE |  |
| DelftBlue | pusht_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) | 30 | 76210 | 92.2 | 0.0529 |  | 0.152 |  | 69 | PASS |  |
| Leonardo | pusht_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) | 30 | 76210 | 90.5 | 0.0524 |  | 0.15 |  | 69 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_noact_seed1 | policy(mlp,c2,k1,noact) | 30 | 76210 | 92 | 0.0539 |  | 0.146 |  | 63 | PASS |  |
| DelftBlue | pusht_policy_mlp_c2k1_noact_seed2 | policy(mlp,c2,k1,noact) | 30 | 76210 | 89.8 | 0.053 |  | 0.148 |  | 64 | PASS |  |
| DelftBlue | pusht_h3_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) [h=3] | 30 | 139280 | 81.4 | 0.0483 |  | 0.161 |  | 66 | DEVIATE |  |
| Leonardo | pusht_policy_mlp_c2k1_noact_goalfinal_seed0 | policy(mlp,c2,k1,noact,Gfinal) | 30 | 76210 | 84.7 | 0.0538 |  | 0.138 |  | 78 | PASS |  |
| Leonardo | pusht_policy_mlp_c2k1_noact_goalfinal_seed1 | policy(mlp,c2,k1,noact,Gfinal) | 30 | 76210 | 86.5 | 0.0541 |  | 0.136 |  | 74 | PASS |  |
| Leonardo | pusht_policy_mlp_c2k1_noact_goalfinal_seed2 | policy(mlp,c2,k1,noact,Gfinal) | 30 | 76210 | 86.2 | 0.054 |  | 0.135 |  | 80 | PASS |  |
| DelftBlue | pusht_h3_policy_mlp_c2k1_noact_goalfinal_seed0 | policy(mlp,c2,k1,noact,Gfinal) [h=3] | 30 | 139280 | 78 | 0.0507 |  | 0.148 |  | 70 | DEVIATE |  |
| DelftBlue | pusht_policy_transformer_c4k1_seed0 | policy(transformer,c4,k1) | 30 | 69640 | 112 | 0.0217 |  | 0.0962 |  | 25 | PASS |  |
| DelftBlue | pusht_h3_sigreg_seed0 | sigreg [h=3] | 0.09 | 139280 | 32.3 | 0.00699 |  |  |  | 88 | DEVIATE |  |

### OGBench-Cube

| cluster | run | method | λ | step | eff rank | pred loss | inv loss | policy loss | goal loss | success | verdict | other evals |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DelftBlue | cube_inverse_lambda_1_seed0 | inverse | 1 | 67500 | 39.4 | 0.00166 | 0.0863 |  |  | 81 | PASS |  |
| DelftBlue | cube_h3_inverse_seed0 | inverse [h=3] | 1 | 127960 | 34.4 | 0.00159 | 0.0808 |  |  | 87 | DEVIATE |  |
| DelftBlue | cube_policy_mlp_c1k1_seed0 | policy(mlp,c1,k1,noact) | 1 | 67500 | 77.4 | 0.0024 |  | 0.369 |  | 66 | PASS |  |
| DelftBlue | cube_policy_mlp_c1k1_seed1 | policy(mlp,c1,k1,noact) | 1 | 67500 | 71.8 | 0.00268 |  | 0.374 |  | 72 | PASS | rerun_job_930929=75 |
| DelftBlue | cube_policy_mlp_c1k1_seed2 | policy(mlp,c1,k1,noact) | 1 | 67500 | 72.5 | 0.00266 |  | 0.376 |  | 68 | PASS |  |
| DelftBlue | cube_policy_mlp_c2k1_seed0 | policy(mlp,c2,k1) | 1 | 67500 | 51.1 | 0.00143 |  | 0.224 |  | 69 | PASS |  |
| Leonardo | cube_policy_mlp_c2k1_noact_delta_seed0 | policy(mlp,c2,k1,delta,noact) | 1 | 67500 | 38 | 0.00267 |  | 0.33 |  | 77 | PASS |  |
| DelftBlue | cube_h3_policy_mlp_c2k1_noact_delta_seed0 | policy(mlp,c2,k1,delta,noact) [h=3] | 1 | 127960 | 33.5 | 0.00224 |  | 0.369 |  | 73 | DEVIATE |  |
| Leonardo | cube_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | policy(mlp,c2,k1,delta,noact,Gfinal) | 1 | 67500 | 44.7 | 0.00301 |  | 0.221 |  | 72 | PASS |  |
| DelftBlue | cube_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | policy(mlp,c2,k1,delta,noact,Gfinal) [h=3] | 1 | 127960 | 40.8 | 0.00383 |  | 0.233 |  | 73 | DEVIATE |  |
| DelftBlue | cube_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) | 1 | 67500 | 39.4 | 0.00329 |  | 0.269 |  | 80 | PASS |  |
| Leonardo | cube_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) | 1 | 67500 | 41.7 | 0.00319 |  | 0.268 |  | 79 | PASS |  |
| DelftBlue | cube_policy_mlp_c2k1_noact_seed1 | policy(mlp,c2,k1,noact) | 1 | 67500 | 40.1 | 0.00325 |  | 0.268 |  | 81 | PASS |  |
| DelftBlue | cube_policy_mlp_c2k1_noact_seed2 | policy(mlp,c2,k1,noact) | 1 | 67500 | 38.5 | 0.00341 |  | 0.271 |  | 73 | PASS |  |
| DelftBlue | cube_h3_policy_mlp_c2k1_noact_seed0 | policy(mlp,c2,k1,noact) [h=3] | 1 | 127960 | 32.9 | 0.00377 |  | 0.316 |  | 84 | DEVIATE |  |
| Leonardo | cube_policy_mlp_c2k1_noact_goalfinal_seed0 | policy(mlp,c2,k1,noact,Gfinal) | 1 | 67500 | 42.9 | 0.0035 |  | 0.218 |  | 78 | PASS |  |
| DelftBlue | cube_h3_policy_mlp_c2k1_noact_goalfinal_seed0 | policy(mlp,c2,k1,noact,Gfinal) [h=3] | 1 | 127960 | 37.7 | 0.00447 |  | 0.234 |  | 86 | DEVIATE |  |
| DelftBlue | cube_h3_sigreg_seed0 | sigreg [h=3] | 0.09 | 127960 | 70.4 | 0.00629 |  |  |  | 62 | DEVIATE |  |

## Queued / running

| cluster | job | name | state | elapsed |
|---|---|---|---|---|
| DelftBlue | 1093632 | ws-cube_h3_policy_mlp_c2k1_noact_seed0 | PENDING | 0:00 |
| DelftBlue | 1093633 | ws-reacher_expert_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | PENDING | 0:00 |
| DelftBlue | 1093631 | ws-cube_h3_policy_mlp_c2k1_noact_goalfinal_seed0 | PENDING | 0:00 |
| DelftBlue | 1093630 | ws-pusht_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | PENDING | 0:00 |
| DelftBlue | 1093629 | ws-pusht_h3_policy_mlp_c2k1_noact_delta_seed0 | PENDING | 0:00 |
| DelftBlue | 1093626 | ws-reacher_mix_h3_policy_mlp_c2k1_noact_seed0 | PENDING | 0:00 |
| DelftBlue | 1093627 | ws-reacher_mix_h3_policy_mlp_c2k1_noact_goalfinal_seed0 | PENDING | 0:00 |
| DelftBlue | 1093628 | ws-reacher_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | PENDING | 0:00 |
| DelftBlue | 1093625 | ws-reacher_expert_h3_policy_mlp_c2k1_noact_seed0 | PENDING | 0:00 |
| DelftBlue | 1093624 | ws-reacher_expert_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0 | RUNNING | 11:21 |

## Not PASS

- **reacher_expert_h3_inverse_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_mix_h3_inverse_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_h3_inverse_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16); metrics.csv missing
- **reacher_expert_h3_policy_mlp_c2k1_noact_delta_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_mix_h3_policy_mlp_c2k1_noact_delta_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_h3_policy_mlp_c2k1_noact_delta_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_expert_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_mix_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_expert_h3_policy_mlp_c2k1_noact_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_mix_h3_policy_mlp_c2k1_noact_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_h3_policy_mlp_c2k1_noact_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16); metrics.csv missing
- **reacher_expert_h3_policy_mlp_c2k1_noact_goalfinal_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_mix_h3_policy_mlp_c2k1_noact_goalfinal_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_h3_policy_mlp_c2k1_noact_goalfinal_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_expert_h3_sigreg_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_mix_h3_sigreg_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_h3_sigreg_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16); metrics.csv missing
- **pusht_h3_inverse_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **pusht_h3_policy_mlp_c2k1_noact_delta_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **pusht_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **pusht_h3_policy_mlp_c2k1_noact_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **pusht_h3_policy_mlp_c2k1_noact_goalfinal_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **pusht_h3_sigreg_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **cube_h3_inverse_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **cube_h3_policy_mlp_c2k1_noact_delta_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **cube_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **cube_h3_policy_mlp_c2k1_noact_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **cube_h3_policy_mlp_c2k1_noact_goalfinal_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **cube_h3_sigreg_seed0** (DelftBlue, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **tworoom_h3_inverse_seed0** (Leonardo, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **tworoom_h3_policy_mlp_c2k1_noact_delta_seed0** (Leonardo, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **tworoom_h3_policy_mlp_c2k1_noact_delta_goalfinal_seed0** (Leonardo, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **tworoom_h3_policy_mlp_c2k1_noact_seed0** (Leonardo, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **tworoom_h3_policy_mlp_c2k1_noact_goalfinal_seed0** (Leonardo, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **tworoom_h3_sigreg_seed0** (Leonardo, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_h3_inverse_seed0** (Leonardo, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_h3_policy_mlp_c2k1_noact_seed0** (Leonardo, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)
- **reacher_h3_sigreg_seed0** (Leonardo, DEVIATE): optimizer.lr=5e-05 (paper 0.0001); wm.history_size=3 (paper 1); batch_size=128 (paper 256; 64 only tolerated for c16)

