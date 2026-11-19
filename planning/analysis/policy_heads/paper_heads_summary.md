# 论文里的四种 policy head：成功率、effective rank 与各项 loss

> 由 `planning/analysis/policy_heads/make_paper_heads_summary.py` 生成（2026-11-17），请勿手改；重新生成：`python3 planning/analysis/policy_heads/make_paper_heads_summary.py`。
> 数据：h1 设定（history size 1）的全部运行，评测与训练日志于 2026-11-13 从 DelftBlue 收集（`~/Downloads/iclr2027/results/h1_*.csv`）。逐 seed 的数值在同目录的 `paper_heads_per_seed.csv`。

## 1. 四种 policy head 与两个参照

| 名称 | 运行名后缀 | head 的输入 | 预测目标 |
|---|---|---|---|
| endpoints | `policy_mlp_c2k1_noact` | (z_t, z_{t+1}) | a_{t+1} |
| displacement | `…_noact_delta` | (z_{t+1} − z_t) | a_{t+1} |
| endpoints + goal（论文默认，“Ours”） | `…_noact_goalfinal` | (z_t, z_{t+1}, z_g) | a_{t+1} |
| displacement + goal | `…_noact_delta_goalfinal` | (z_{t+1} − z_t, z_g − z_{t+1}) | a_{t+1} |
| SMWM（参照） | `…_inverse…` | (z_t, z_{t+1}) | a_t（inverse dynamics） |
| LeWM（参照） | `…_sigreg` | 无 head，SIGReg 正则 | – |

- z_g 是同一条轨迹最后一帧的 latent（`goal_source: final`）。四种 head 的网络相同（MLP，宽 256，不输入 a_t），只有输入不同。
- 正则权重 λ 取各环境 SMWM 的值：TwoRoom 0.1，Push-T 30，Cube 1，Reacher 5；LeWM 的 SIGReg 权重 0.09。脚本用日志核对过：total loss = prediction loss + λ × 正则项。

## 2. 指标与口径

- **成功率**：100 个评测任务，目标在 25 个环境步之后，步数预算 50。CEM-30 = 标准 CEM（30 次迭代）；warm-start-30 = 用 policy head 的输出作为 CEM 的初始均值；head only = 不搜索，直接执行 head 的输出。
- **effective rank**：exp(归一化奇异值的熵)，192 维 latent。logged = 训练中验证环节记录的最后一个值（bfloat16 混合精度，前 2560 个 held-out clip，论文表 mech 用的就是它）；fp32 = 用保存的最终 latent 在单精度下重算（同样的 2560 个 clip）。
- **latent std**：最终 latent 每一维的标准差（单精度），用来判断是否坍缩。
- **prediction loss**：预测 latent 与目标 latent 的 MSE。train = 最后一个 epoch 的训练均值（forward model 带 dropout）；val = held-out 上最后一次验证的值。**相对预测误差** = val prediction loss / latent 方差。
- **action loss**：标准化动作上的 MSE（policy head 是 `policy_loss`，SMWM 是 `inv_loss`），所以 1 − val loss 就是 held-out 上动作的**可解释方差**；loss ≈ 1 表示 head 只能输出均值。
- 每格是训练 seed 间的均值 ± 样本标准差（四舍五入）；上标是该格的 seed 数（不足 5 个时标出）。成功率和日志的 seed 数可能不同（第 7 节）。

## 3. 成功率（%）

### 3.1 默认预算：CEM 30 次迭代

| 评测方式 | 方法 / head | TwoRoom | Push-T | Cube | Reacher-PD | Reacher-Random |
|---|---|:---:|:---:|:---:|:---:|:---:|
| CEM-30 | LeWM（SIGReg，无 head） | 93.8 ± 1.8 | 87.0 ± 2.3 | 61.2 ± 2.2 | 69.8 ± 4.9 | 73.0 ± 4.7 |
| CEM-30 | SMWM（inverse head） | 97.8 ± 2.0 | 84.4 ± 2.5 | 85.2 ± 2.4 | 65.4 ± 4.4 | 72.2 ± 4.8 |
| CEM-30 | endpoints | 99.6 ± 0.9 | 67.3 ± 4.0⁴ | 74.8 ± 7.4⁴ | 67.0 ± 4.8 | 12.4 ± 4.2 |
| CEM-30 | displacement | 66.2 ± 4.2 | 77.3 ± 1.5⁴ | 77.8 ± 3.0⁴ | 68.0 ± 5.6 | 8.2 ± 2.5 |
| CEM-30 | endpoints + goal（默认） | 99.6 ± 0.5 | 79.4 ± 2.1 | 78.2 ± 3.4 | 65.2 ± 3.6 | 8.8 ± 2.9 |
| CEM-30 | displacement + goal | 100.0 ± 0.0⁴ | 83.0 ± 1.4² | 75.5 ± 2.9⁴ | 62.2 ± 3.2 | 31.8 ± 8.9 |
| warm-start-30 | endpoints | 99.4 ± 0.5 | 76.3 ± 1.3⁴ | 85.5 ± 3.9⁴ | 61.6 ± 3.0 | 11.0 ± 3.7 |
| warm-start-30 | displacement | 66.4 ± 3.8 | 77.8 ± 3.1⁴ | 81.3 ± 2.1⁴ | 63.4 ± 5.9 | 7.8 ± 2.7 |
| warm-start-30 | endpoints + goal（默认） | 99.4 ± 0.9 | 81.6 ± 2.1 | 88.8 ± 3.1 | 62.0 ± 3.9 | 11.0 ± 2.9 |
| warm-start-30 | displacement + goal | 100.0 ± 0.0⁴ | 87.5 ± 2.1² | 79.8 ± 3.6⁴ | 64.6 ± 4.7 | 27.4 ± 4.4 |
| head only | endpoints | 63.4 ± 1.7 | 45.0 ± 1.2⁴ | 56.3 ± 1.3⁴ | 4.0 ± 0.0 | 1.0 ± 0.0 |
| head only | displacement | 37.2 ± 2.5 | 18.5 ± 3.0⁴ | 55.0 ± 1.4⁴ | 1.4 ± 0.9 | 1.0 ± 0.0 |
| head only | endpoints + goal（默认） | 98.4 ± 1.9 | 43.0 ± 4.6 | 70.6 ± 1.5 | 15.0 ± 1.9 | 1.0 ± 0.0 |
| head only | displacement + goal | 99.8 ± 0.5⁴ | 40.0 ± 4.2² | 58.7 ± 2.2⁴ | 3.4 ± 1.1 | 15.6 ± 7.7 |

### 3.2 更小的预算：CEM 10 次和 5 次迭代

| 评测方式 | 方法 / head | TwoRoom | Push-T | Cube | Reacher-PD | Reacher-Random |
|---|---|:---:|:---:|:---:|:---:|:---:|
| CEM-10 | LeWM（SIGReg，无 head） | 93.6 ± 1.8 | 83.0 ± 4.1 | 64.2 ± 1.3 | 70.2 ± 6.5 | 72.0 ± 2.0 |
| CEM-10 | SMWM（inverse head） | 97.6 ± 2.2 | 79.8 ± 2.3 | 85.6 ± 1.7 | 73.2 ± 3.3 | 75.0 ± 2.7 |
| CEM-10 | endpoints | 99.6 ± 0.5 | 75.0 ± 3.4⁴ | 81.3 ± 3.9⁴ | 69.4 ± 5.1 | 9.6 ± 1.7 |
| CEM-10 | displacement | 67.0 ± 4.2 | 79.3 ± 3.5⁴ | 79.0 ± 4.1⁴ | 70.4 ± 4.4 | 12.0 ± 2.7 |
| CEM-10 | endpoints + goal（默认） | 99.2 ± 0.8 | 78.8 ± 1.3 | 83.6 ± 2.1 | 67.6 ± 2.7 | 7.4 ± 2.2 |
| CEM-10 | displacement + goal | 99.5 ± 1.0⁴ | 82.5 ± 6.4² | 79.5 ± 3.9⁴ | 69.2 ± 4.1 | 32.0 ± 10.7 |
| warm-start-10 | endpoints | 99.4 ± 0.9 | 80.3 ± 3.8⁴ | 87.8 ± 3.3⁴ | 71.6 ± 2.1 | 8.8 ± 1.3 |
| warm-start-10 | displacement | 64.2 ± 3.8 | 81.0 ± 2.9⁴ | 87.3 ± 0.5⁴ | 70.4 ± 4.3 | 10.8 ± 2.5 |
| warm-start-10 | endpoints + goal（默认） | 99.6 ± 0.5 | 86.6 ± 2.3 | 90.0 ± 3.2 | 72.2 ± 3.8 | 9.6 ± 2.3 |
| warm-start-10 | displacement + goal | 99.8 ± 0.5⁴ | 87.0 ± 7.1² | 80.0 ± 2.4⁴ | 70.4 ± 2.3 | 30.3 ± 8.5⁴ |
| CEM-5 | LeWM（SIGReg，无 head） | 92.4 ± 2.2 | 68.0 ± 3.7 | 62.4 ± 1.7 | 70.6 ± 2.2 | 72.4 ± 3.4 |
| CEM-5 | SMWM（inverse head） | 97.8 ± 2.0 | 64.4 ± 1.8 | 83.6 ± 2.1 | 71.0 ± 5.3 | 73.4 ± 5.1 |
| CEM-5 | endpoints | 99.0 ± 1.2⁴ | 62.3 ± 5.2⁴ | 78.8 ± 1.7⁴ | 67.5 ± 2.1⁴ | 8.5 ± 2.4⁴ |
| CEM-5 | displacement | 66.0 ± 5.5⁴ | 66.5 ± 5.4⁴ | 81.3 ± 2.1⁴ | 71.3 ± 2.8⁴ | 7.8 ± 2.1⁴ |
| CEM-5 | endpoints + goal（默认） | 99.6 ± 0.5 | 68.2 ± 1.8 | 81.4 ± 0.5 | 70.0 ± 3.5 | 8.0 ± 2.9 |
| CEM-5 | displacement + goal | 100.0 ± 0.0³ | 68.5 ± 2.1² | 79.0 ± 0.8⁴ | 70.5 ± 4.7⁴ | 29.5 ± 7.8⁴ |
| warm-start-5 | endpoints | 99.3 ± 1.5⁴ | 80.0 ± 3.7⁴ | 81.5 ± 3.7⁴ | 71.0 ± 1.4⁴ | 8.3 ± 1.7⁴ |
| warm-start-5 | displacement | 68.5 ± 4.7⁴ | 71.8 ± 4.3⁴ | 88.3 ± 1.0⁴ | 69.8 ± 4.7⁴ | 8.3 ± 2.8⁴ |
| warm-start-5 | endpoints + goal（默认） | 99.6 ± 0.5 | 85.8 ± 1.5 | 89.2 ± 2.8 | 71.2 ± 3.9 | 7.6 ± 2.5 |
| warm-start-5 | displacement + goal | 100.0 ± 0.0³ | 85.0 ± 2.8² | 78.3 ± 3.6⁴ | 71.5 ± 5.0⁴ | 29.5 ± 13.2⁴ |

## 4. Effective rank 与 latent 尺度

第 4、5 节是日志类指标，每格的 seed 数见第 7 节的表（“/” 右侧）；没有 ± 的表不再标上标。

### 4.1 Effective rank（logged，bfloat16）

| 方法 / head | TwoRoom | Push-T | Cube | Reacher-PD | Reacher-Random |
|---|:---:|:---:|:---:|:---:|:---:|
| LeWM（SIGReg，无 head） | 91.8 ± 0.9⁴ | 58.2 ± 0.3⁴ | 108.0 ± 1.6⁴ | 78.8 ± 0.2 | 81.0 ± 0.2⁴ |
| SMWM（inverse head） | 14.2 ± 1.4 | 50.0 ± 1.0⁴ | 39.0 ± 0.7 | 18.6 ± 0.6 | 19.1 ± 1.4⁴ |
| endpoints | 11.9 ± 1.4 | 90.6 ± 1.4 | 39.6 ± 0.7 | 19.7 ± 0.8 | 160.3 ± 5.3 |
| displacement | 17.4 ± 2.5 | 68.3 ± 1.0⁴ | 40.9 ± 1.7⁴ | 29.8 ± 1.0 | 140.9 ± 34.3 |
| endpoints + goal（默认） | 13.7 ± 2.3 | 85.2 ± 1.8² | 43.6 ± 0.5⁴ | 27.8 ± 0.5 | 161.8 ± 4.9 |
| displacement + goal | 17.6 ± 1.4⁴ | 79.1 ± 0.5² | 44.5 ± 0.7⁴ | 37.6 ± 0.8 | 16.3 ± 3.2 |

### 4.2 Effective rank（fp32，最终 latent）

| 方法 / head | TwoRoom | Push-T | Cube | Reacher-PD | Reacher-Random |
|---|:---:|:---:|:---:|:---:|:---:|
| LeWM（SIGReg，无 head） | 91.7 ± 0.9⁴ | 58.1 ± 0.4⁴ | 107.9 ± 1.6⁴ | 78.7 ± 0.3 | 80.7 ± 0.4⁴ |
| SMWM（inverse head） | 10.7 ± 1.2 | 49.7 ± 1.3⁴ | 38.9 ± 0.7 | 17.2 ± 0.3 | 17.5 ± 1.5⁴ |
| endpoints | 5.0 ± 0.5 | 90.5 ± 1.2 | 39.5 ± 0.6 | 16.8 ± 0.6 | 29.5 ± 7.8 |
| displacement | 11.6 ± 1.8 | 68.2 ± 1.2⁴ | 40.8 ± 1.8⁴ | 27.2 ± 0.8 | 19.9 ± 6.0 |
| endpoints + goal（默认） | 5.4 ± 0.5 | 85.2 ± 1.9² | 43.6 ± 0.6⁴ | 24.8 ± 0.5 | 23.5 ± 9.7 |
| displacement + goal | 7.0 ± 0.5⁴ | 78.9 ± 0.5² | 44.4 ± 0.6⁴ | 34.6 ± 1.3 | 12.1 ± 2.9 |

### 4.3 Latent 每维标准差（fp32）

| 方法 / head | TwoRoom | Push-T | Cube | Reacher-PD | Reacher-Random |
|---|:---:|:---:|:---:|:---:|:---:|
| LeWM（SIGReg，无 head） | 1.01 | 0.94 | 0.99 | 0.97 | 0.98 |
| SMWM（inverse head） | 0.075 | 1.03 | 0.58 | 0.31 | 0.24 |
| endpoints | 0.049 | 0.77 | 0.60 | 0.30 | 3.3e-4 |
| displacement | 0.031 | 0.97 | 0.56 | 0.34 | 1.4e-3 |
| endpoints + goal（默认） | 0.060 | 0.89 | 0.57 | 0.34 | 3.9e-4 |
| displacement + goal | 0.049 | 0.93 | 0.55 | 0.35 | 0.083 |

注意：latent std 很小时（TwoRoom 约 0.05，Reacher-Random 上坍缩的 head 约 1e-3 以下），logged 的 effective rank 被 bfloat16 的舍入噪声抬高，应看 fp32 的值和 latent std。Reacher-Random 上 endpoints、displacement、endpoints + goal 三种 head 都坍缩了，logged 的 140–160 不是真实的维度。

## 5. Loss

### 5.1 Prediction loss（train / val）

| 方法 / head | TwoRoom | Push-T | Cube | Reacher-PD | Reacher-Random |
|---|:---:|:---:|:---:|:---:|:---:|
| LeWM（SIGReg，无 head） | 1.8e-2 / 8.6e-3 | 8.7e-3 / 4.0e-3 | 9.0e-3 / 4.6e-3 | 7.1e-2 / 6.7e-2 | 6.5e-2 / 6.1e-2 |
| SMWM（inverse head） | 6.9e-5 / 4.4e-5 | 2.8e-2 / 2.2e-2 | 1.6e-3 / 1.2e-3 | 2.5e-3 / 2.2e-3 | 1.4e-3 / 1.3e-3 |
| endpoints | 6.2e-5 / 4.6e-5 | 5.3e-2 / 4.6e-2 | 3.5e-3 / 2.8e-3 | 2.8e-3 / 2.4e-3 | 3.7e-6 / 2.3e-6 |
| displacement | 1.7e-4 / 1.2e-4 | 3.7e-2 / 2.7e-2 | 3.2e-3 / 2.8e-3 | 4.1e-3 / 3.6e-3 | 9.7e-6 / 4.9e-6 |
| endpoints + goal（默认） | 5.9e-5 / 4.2e-5 | 5.5e-2 / 4.7e-2 | 3.5e-3 / 3.0e-3 | 4.2e-3 / 3.7e-3 | 8.2e-6 / 2.4e-6 |
| displacement + goal | 6.4e-5 / 4.6e-5 | 4.6e-2 / 3.7e-2 | 3.0e-3 / 2.6e-3 | 5.5e-3 / 4.8e-3 | 8.7e-4 / 7.6e-4 |

### 5.2 相对预测误差（val prediction loss / latent 方差，%）

| 方法 / head | TwoRoom | Push-T | Cube | Reacher-PD | Reacher-Random |
|---|:---:|:---:|:---:|:---:|:---:|
| LeWM（SIGReg，无 head） | 0.85 | 0.44 | 0.46 | 7.08 | 6.26 |
| SMWM（inverse head） | 0.67 | 1.47 | 0.33 | 1.77 | 2.08 |
| endpoints | 0.76 | 7.09 | 0.71 | 2.18 | >100 |
| displacement | 9.62 | 2.57 | 0.73 | 2.66 | >100 |
| endpoints + goal（默认） | 0.87 | 5.47 | 0.86 | 2.79 | >100 |
| displacement + goal | 1.07 | 3.98 | 0.77 | 3.54 | 6.07 |

坍缩的模型 prediction loss 的绝对值最低（latent 几乎是常数），相对误差却超过 100 %，所以不同 head 之间不要直接比 prediction loss 的绝对值。

### 5.3 Action loss（train / val）

| 方法 / head | TwoRoom | Push-T | Cube | Reacher-PD | Reacher-Random |
|---|:---:|:---:|:---:|:---:|:---:|
| SMWM（inverse head） | 0.754 / 0.745 | 0.091 / 0.062 | 0.087 / 0.086 | 0.613 / 0.634 | 0.832 / 0.830 |
| endpoints | 0.952 / 0.956 | 0.149 / 0.154 | 0.272 / 0.311 | 0.861 / 0.864 | 0.999 / 1.000 |
| displacement | 0.963 / 0.964 | 0.166 / 0.149 | 0.324 / 0.357 | 0.861 / 0.864 | 0.999 / 1.000 |
| endpoints + goal（默认） | 0.908 / 0.905 | 0.135 / 0.130 | 0.216 / 0.269 | 0.849 / 0.859 | 0.999 / 1.000 |
| displacement + goal | 0.908 / 0.906 | 0.136 / 0.134 | 0.221 / 0.260 | 0.848 / 0.859 | 0.990 / 0.993 |

SMWM 一行是 inverse loss（目标 a_t），其余是 policy loss（目标 a_{t+1}），两者的目标不同，只在各自内部比较。

### 5.4 动作的可解释方差（1 − val action loss，%）

| 方法 / head | TwoRoom | Push-T | Cube | Reacher-PD | Reacher-Random |
|---|:---:|:---:|:---:|:---:|:---:|
| SMWM（inverse head） | 25.5 | 93.8 | 91.4 | 36.6 | 17.0 |
| endpoints | 4.4 | 84.6 | 68.9 | 13.6 | 0.0 |
| displacement | 3.6 | 85.1 | 64.3 | 13.6 | 0.0 |
| endpoints + goal（默认） | 9.5 | 87.0 | 73.1 | 14.1 | 0.0 |
| displacement + goal | 9.4 | 86.6 | 74.0 | 14.1 | 0.7 |

### 5.5 Total loss（train / val）

| 方法 / head | TwoRoom | Push-T | Cube | Reacher-PD | Reacher-Random |
|---|:---:|:---:|:---:|:---:|:---:|
| SMWM（inverse head） | 0.075 / 0.075 | 2.76 / 1.89 | 0.088 / 0.087 | 3.07 / 3.17 | 4.16 / 4.15 |
| endpoints | 0.095 / 0.096 | 4.52 / 4.65 | 0.28 / 0.31 | 4.31 / 4.32 | 4.99 / 5.00 |
| displacement | 0.096 / 0.097 | 5.02 / 4.50 | 0.33 / 0.36 | 4.31 / 4.32 | 4.99 / 5.00 |
| endpoints + goal（默认） | 0.091 / 0.091 | 4.10 / 3.95 | 0.22 / 0.27 | 4.25 / 4.30 | 4.99 / 5.00 |
| displacement + goal | 0.091 / 0.091 | 4.12 / 4.07 | 0.22 / 0.26 | 4.25 / 4.30 | 4.95 / 4.97 |

Total loss = prediction loss + λ × action loss，λ 随环境不同，所以只能在同一数据集内比较。LeWM 的正则项是 SIGReg，量纲不同，没有列在这里；它的 SIGReg loss 和 total loss 在逐 seed 的 CSV 里。

## 6. 按数据集的总表

### TwoRoom

| 方法 / head | CEM-30 | warm-start-30 | head only | eff. rank（logged / fp32） | latent std | pred loss（train / val） | 相对误差 % | action loss（train / val） | 可解释方差 % | seed 数（评测 / 日志） |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LeWM（SIGReg，无 head） | 93.8 ± 1.8 | – | – | 91.8 / 91.7 | 1.01 | 1.8e-2 / 8.6e-3 | 0.85 | – | – | 5 / 4 |
| SMWM（inverse head） | 97.8 ± 2.0 | – | – | 14.2 / 10.7 | 0.075 | 6.9e-5 / 4.4e-5 | 0.67 | 0.754 / 0.745 | 25.5 | 5 / 5 |
| endpoints | 99.6 ± 0.9 | 99.4 ± 0.5 | 63.4 ± 1.7 | 11.9 / 5.0 | 0.049 | 6.2e-5 / 4.6e-5 | 0.76 | 0.952 / 0.956 | 4.4 | 5 / 5 |
| displacement | 66.2 ± 4.2 | 66.4 ± 3.8 | 37.2 ± 2.5 | 17.4 / 11.6 | 0.031 | 1.7e-4 / 1.2e-4 | 9.62 | 0.963 / 0.964 | 3.6 | 5 / 5 |
| endpoints + goal（默认） | 99.6 ± 0.5 | 99.4 ± 0.9 | 98.4 ± 1.9 | 13.7 / 5.4 | 0.060 | 5.9e-5 / 4.2e-5 | 0.87 | 0.908 / 0.905 | 9.5 | 5 / 5 |
| displacement + goal | 100.0 ± 0.0⁴ | 100.0 ± 0.0⁴ | 99.8 ± 0.5⁴ | 17.6 / 7.0 | 0.049 | 6.4e-5 / 4.6e-5 | 1.07 | 0.908 / 0.906 | 9.4 | 4 / 4 |

### Push-T

| 方法 / head | CEM-30 | warm-start-30 | head only | eff. rank（logged / fp32） | latent std | pred loss（train / val） | 相对误差 % | action loss（train / val） | 可解释方差 % | seed 数（评测 / 日志） |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LeWM（SIGReg，无 head） | 87.0 ± 2.3 | – | – | 58.2 / 58.1 | 0.94 | 8.7e-3 / 4.0e-3 | 0.44 | – | – | 5 / 4 |
| SMWM（inverse head） | 84.4 ± 2.5 | – | – | 50.0 / 49.7 | 1.03 | 2.8e-2 / 2.2e-2 | 1.47 | 0.091 / 0.062 | 93.8 | 5 / 4 |
| endpoints | 67.3 ± 4.0⁴ | 76.3 ± 1.3⁴ | 45.0 ± 1.2⁴ | 90.6 / 90.5 | 0.77 | 5.3e-2 / 4.6e-2 | 7.09 | 0.149 / 0.154 | 84.6 | 4 / 5 |
| displacement | 77.3 ± 1.5⁴ | 77.8 ± 3.1⁴ | 18.5 ± 3.0⁴ | 68.3 / 68.2 | 0.97 | 3.7e-2 / 2.7e-2 | 2.57 | 0.166 / 0.149 | 85.1 | 4 / 4 |
| endpoints + goal（默认） | 79.4 ± 2.1 | 81.6 ± 2.1 | 43.0 ± 4.6 | 85.2 / 85.2 | 0.89 | 5.5e-2 / 4.7e-2 | 5.47 | 0.135 / 0.130 | 87.0 | 5 / 2 |
| displacement + goal | 83.0 ± 1.4² | 87.5 ± 2.1² | 40.0 ± 4.2² | 79.1 / 78.9 | 0.93 | 4.6e-2 / 3.7e-2 | 3.98 | 0.136 / 0.134 | 86.6 | 2 / 2 |

### Cube

| 方法 / head | CEM-30 | warm-start-30 | head only | eff. rank（logged / fp32） | latent std | pred loss（train / val） | 相对误差 % | action loss（train / val） | 可解释方差 % | seed 数（评测 / 日志） |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LeWM（SIGReg，无 head） | 61.2 ± 2.2 | – | – | 108.0 / 107.9 | 0.99 | 9.0e-3 / 4.6e-3 | 0.46 | – | – | 5 / 4 |
| SMWM（inverse head） | 85.2 ± 2.4 | – | – | 39.0 / 38.9 | 0.58 | 1.6e-3 / 1.2e-3 | 0.33 | 0.087 / 0.086 | 91.4 | 5 / 5 |
| endpoints | 74.8 ± 7.4⁴ | 85.5 ± 3.9⁴ | 56.3 ± 1.3⁴ | 39.6 / 39.5 | 0.60 | 3.5e-3 / 2.8e-3 | 0.71 | 0.272 / 0.311 | 68.9 | 4 / 5 |
| displacement | 77.8 ± 3.0⁴ | 81.3 ± 2.1⁴ | 55.0 ± 1.4⁴ | 40.9 / 40.8 | 0.56 | 3.2e-3 / 2.8e-3 | 0.73 | 0.324 / 0.357 | 64.3 | 4 / 4 |
| endpoints + goal（默认） | 78.2 ± 3.4 | 88.8 ± 3.1 | 70.6 ± 1.5 | 43.6 / 43.6 | 0.57 | 3.5e-3 / 3.0e-3 | 0.86 | 0.216 / 0.269 | 73.1 | 5 / 4 |
| displacement + goal | 75.5 ± 2.9⁴ | 79.8 ± 3.6⁴ | 58.7 ± 2.2⁴ | 44.5 / 44.4 | 0.55 | 3.0e-3 / 2.6e-3 | 0.77 | 0.221 / 0.260 | 74.0 | 4 / 4 |

### Reacher-PD

| 方法 / head | CEM-30 | warm-start-30 | head only | eff. rank（logged / fp32） | latent std | pred loss（train / val） | 相对误差 % | action loss（train / val） | 可解释方差 % | seed 数（评测 / 日志） |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LeWM（SIGReg，无 head） | 69.8 ± 4.9 | – | – | 78.8 / 78.7 | 0.97 | 7.1e-2 / 6.7e-2 | 7.08 | – | – | 5 / 5 |
| SMWM（inverse head） | 65.4 ± 4.4 | – | – | 18.6 / 17.2 | 0.31 | 2.5e-3 / 2.2e-3 | 1.77 | 0.613 / 0.634 | 36.6 | 5 / 5 |
| endpoints | 67.0 ± 4.8 | 61.6 ± 3.0 | 4.0 ± 0.0 | 19.7 / 16.8 | 0.30 | 2.8e-3 / 2.4e-3 | 2.18 | 0.861 / 0.864 | 13.6 | 5 / 5 |
| displacement | 68.0 ± 5.6 | 63.4 ± 5.9 | 1.4 ± 0.9 | 29.8 / 27.2 | 0.34 | 4.1e-3 / 3.6e-3 | 2.66 | 0.861 / 0.864 | 13.6 | 5 / 5 |
| endpoints + goal（默认） | 65.2 ± 3.6 | 62.0 ± 3.9 | 15.0 ± 1.9 | 27.8 / 24.8 | 0.34 | 4.2e-3 / 3.7e-3 | 2.79 | 0.849 / 0.859 | 14.1 | 5 / 5 |
| displacement + goal | 62.2 ± 3.2 | 64.6 ± 4.7 | 3.4 ± 1.1 | 37.6 / 34.6 | 0.35 | 5.5e-3 / 4.8e-3 | 3.54 | 0.848 / 0.859 | 14.1 | 5 / 5 |

### Reacher-Random

| 方法 / head | CEM-30 | warm-start-30 | head only | eff. rank（logged / fp32） | latent std | pred loss（train / val） | 相对误差 % | action loss（train / val） | 可解释方差 % | seed 数（评测 / 日志） |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LeWM（SIGReg，无 head） | 73.0 ± 4.7 | – | – | 81.0 / 80.7 | 0.98 | 6.5e-2 / 6.1e-2 | 6.26 | – | – | 5 / 4 |
| SMWM（inverse head） | 72.2 ± 4.8 | – | – | 19.1 / 17.5 | 0.24 | 1.4e-3 / 1.3e-3 | 2.08 | 0.832 / 0.830 | 17.0 | 5 / 4 |
| endpoints | 12.4 ± 4.2 | 11.0 ± 3.7 | 1.0 ± 0.0 | 160.3 / 29.5 | 3.3e-4 | 3.7e-6 / 2.3e-6 | >100 | 0.999 / 1.000 | 0.0 | 5 / 5 |
| displacement | 8.2 ± 2.5 | 7.8 ± 2.7 | 1.0 ± 0.0 | 140.9 / 19.9 | 1.4e-3 | 9.7e-6 / 4.9e-6 | >100 | 0.999 / 1.000 | 0.0 | 5 / 5 |
| endpoints + goal（默认） | 8.8 ± 2.9 | 11.0 ± 2.9 | 1.0 ± 0.0 | 161.8 / 23.5 | 3.9e-4 | 8.2e-6 / 2.4e-6 | >100 | 0.999 / 1.000 | 0.0 | 5 / 5 |
| displacement + goal | 31.8 ± 8.9 | 27.4 ± 4.4 | 15.6 ± 7.7 | 16.3 / 12.1 | 0.083 | 8.7e-4 / 7.6e-4 | 6.07 | 0.990 / 0.993 | 0.7 | 5 / 5 |

## 7. Seed 覆盖

每格：有 CEM-30 评测的 seed / 有完整 DelftBlue 训练日志和最终 latent 的 seed。

| 方法 / head | TwoRoom | Push-T | Cube | Reacher-PD | Reacher-Random |
|---|:---:|:---:|:---:|:---:|:---:|
| LeWM（SIGReg，无 head） | 01234 / 1234 | 01234 / 1234 | 01234 / 1234 | 01234 / 01234 | 01234 / 1234 |
| SMWM（inverse head） | 01234 / 01234 | 01234 / 1234 | 01234 / 01234 | 01234 / 01234 | 01234 / 1234 |
| endpoints | 01234 / 01234 | 1234 / 01234 | 1234 / 01234 | 01234 / 01234 | 01234 / 01234 |
| displacement | 01234 / 01234 | 1234 / 1234 | 1234 / 1234 | 01234 / 01234 | 01234 / 01234 |
| endpoints + goal（默认） | 01234 / 01234 | 01234 / 34 | 01234 / 1234 | 01234 / 01234 | 01234 / 01234 |
| displacement + goal | 0124 / 0124 | 34 / 34 | 1234 / 1234 | 01234 / 01234 | 01234 / 01234 |

- 第 3–6 节的日志类指标（rank、std、loss）只用 DelftBlue 上有日志的 seed，与论文的表 mech、表 diag 口径一致。
- 缺的 seed：Push-T 和 Cube 的部分 seed 0–2 在 Leonardo 上训练（日志不在 DelftBlue）；TwoRoom 的 displacement + goal seed 3 没有训练。

### 7.1 Leonardo 上训练的 seed（来自 `planning/results.csv`，旧评测流程）

这些 seed 只有最后一个 epoch 的训练指标（3 位有效数字），没有 val loss 和 fp32 的统计。“合并”一行把它们与 DelftBlue 的 seed 一起求均值。

| 数据集 | head | seed | 训练集群 | eff. rank（logged） | pred loss（train） | policy loss（train） | 成功率（旧流程，CEM-30） |
|---|---|:---:|---|:---:|:---:|:---:|:---:|
| Push-T | displacement | 0 | Leonardo | 68.7 | 3.7e-2 | 0.163 | 72 |
| Push-T | displacement | **合并（5 个 seed）** | DelftBlue + Leonardo | **68.4 ± 0.9** | **3.7e-2** | **0.165** |  |
| Push-T | endpoints + goal（默认） | 0 | Leonardo | 84.7 | 5.4e-2 | 0.138 | 78 |
| Push-T | endpoints + goal（默认） | 1 | Leonardo | 86.5 | 5.4e-2 | 0.136 | 74 |
| Push-T | endpoints + goal（默认） | 2 | Leonardo | 86.2 | 5.4e-2 | 0.135 | 80 |
| Push-T | endpoints + goal（默认） | **合并（5 个 seed）** | DelftBlue + Leonardo | **85.6 ± 1.2** | **5.4e-2** | **0.136** |  |
| Push-T | displacement + goal | 0 | Leonardo | 78.2 | 4.5e-2 | 0.140 | 85 |
| Push-T | displacement + goal | 1 | Leonardo | 80.5 | 4.5e-2 | 0.136 | 85 |
| Push-T | displacement + goal | 2 | Leonardo | 80.4 | 4.5e-2 | 0.137 | 81 |
| Push-T | displacement + goal | **合并（5 个 seed）** | DelftBlue + Leonardo | **79.5 ± 1.0** | **4.5e-2** | **0.137** |  |
| Cube | displacement | 0 | Leonardo | 38.0 | 2.7e-3 | 0.330 | 77 |
| Cube | displacement | **合并（5 个 seed）** | DelftBlue + Leonardo | **40.3 ± 2.0** | **3.1e-3** | **0.325** |  |
| Cube | endpoints + goal（默认） | 0 | Leonardo | 42.9 | 3.5e-3 | 0.218 | 78 |
| Cube | endpoints + goal（默认） | **合并（5 个 seed）** | DelftBlue + Leonardo | **43.5 ± 0.5** | **3.5e-3** | **0.216** |  |
| Cube | displacement + goal | 0 | Leonardo | 44.7 | 3.0e-3 | 0.221 | 72 |
| Cube | displacement + goal | **合并（5 个 seed）** | DelftBlue + Leonardo | **44.5 ± 0.6** | **3.0e-3** | **0.221** |  |

旧流程（`eval.py`，同样是 100 个任务、目标 25 步、预算 50、CEM 30 次迭代）与新流程的 CEM-30 在 30 个两边都评测过的运行上平均相差 1.0 个点（最大 4，没有系统偏差：新减旧的均值为 0.0），可以互相参考，但第 3 节的均值没有混用。

### 7.2 用旧流程补齐后的 CEM-30 成功率

新流程缺评测的 seed 如果在旧流程里评测过，可以补进来看均值是否稳定（两个流程混用，仅供参考）。

| 数据集 | head | 新流程 | 旧流程补的 seed | 补齐后 |
|---|---|:---:|---|:---:|
| Push-T | endpoints | 67.3 ± 4.0⁴ | seed 0: 69 | 67.6 ± 3.6 |
| Push-T | displacement | 77.3 ± 1.5⁴ | seed 0: 72 | 76.2 ± 2.7 |
| Push-T | displacement + goal | 83.0 ± 1.4² | seed 0: 85、seed 1: 85、seed 2: 81 | 83.4 ± 1.8 |
| Cube | endpoints | 74.8 ± 7.4⁴ | seed 0: 80 | 75.8 ± 6.8 |
| Cube | displacement | 77.8 ± 3.0⁴ | seed 0: 77 | 77.6 ± 2.6 |
| Cube | displacement + goal | 75.5 ± 2.9⁴ | seed 0: 72 | 74.8 ± 2.9 |

## 8. 更早的 head 变体

MLP / Transformer / Mamba、不同 context 和 k、是否输入 a_t、goal policy、λ 扫描等 9–10 月的结果不在这里，见同目录的 `early_heads_summary.md`（由 `make_early_summary.py` 生成）。

## 9. 逐 seed 数据

`paper_heads_per_seed.csv`：每个（数据集，方法，seed）一行，包含本页所有表格背后的数值，以及 `log_source` 一列（DelftBlue = 完整日志；Leonardo = 只有 `planning/results.csv` 里的最终训练指标）。
