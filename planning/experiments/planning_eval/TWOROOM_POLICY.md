# TwoRoom: CEM 与直接 policy head 评估

基于 DelftBlue 同步提交 `4f6eb4e`。入口：`planning/eval_tworoom_policy.py`。

## 仓库实现与模型含义

- `planning/train.py::forward_step`：forward loss 始终启用；inverse、SIGReg、policy loss 按各自权重相加。policy 用前 L 帧 latent 和 L−1 个过去动作块预测后续 k 个动作块。
- `planning/module.py::PolicyModel`：支持 MLP、ResMLP、GRU、Transformer、Mamba；输出 `(batch, k, frameskip * action_dim)`。这里的 Mamba 是仓库自己的 PyTorch 实现。
- `planning/jepa.py`：ViT 编码观测，projector 输出 latent；forward predictor 用 latent 和动作预测下一状态；`get_cost` 用预测终点与目标图像的 latent 距离供 CEM 搜索。
- 原 `planning/eval.py` 仅有 CEM 和 random；它虽然加载 policy head，但 CEM 不调用该 head。新入口另加直接 head 控制路径，沿用已有模型构建和 World 构建函数。
- `planning/utils.py::get_column_normalizer`：在训练集原始动作上使用均值与 `torch.std`（样本标准差）。新入口恢复相同统计，而非在 eval 集重新拟合。
- `toy/` 是独立玩具实验，本次不使用。架构分析目录提供 loss、effective rank 等离线分析，它们不能代替闭环导航成功率。

用户提供的 `tworoom_policy_mamba_c16k1_seed0`：`wm.history_size=1`、`embed_dim=192`、policy `context=16`、`num_future=1`、`use_action=true`、`depth=2`、`weight=0.1`。inverse/SIGReg 权重为零。每步动作是 2 维，`frameskip=5`，所以 head 每次输出 10 个数，表示连续 5 步的不同动作，不能当成同一动作重复 5 次。16 帧观测覆盖 75 个原始环境步；它不是 world model 的 history_size。

## 两种评估协议

1. **cem**：规划 horizon=5 个动作块，执行 receding_horizon=5 个块后重规划；默认 300 个候选、30 次迭代、topk=30。
2. **direct**：每隔 5 个环境步重新编码观测，维持 16 个 latent 与 15 个已执行动作块的窗口。取预测的第一个动作块，反归一化、裁剪到环境动作范围后逐步执行；k>1 时同样只执行第一个块再观测。历史保存裁剪后实际发送的动作。

直接 head 没有独立的目标参数，代码不把 goal latent 冒充当前 latent，也不输入示范的未来动作。它可能从观测像素中看到环境目标线索，但没有额外训练成目标条件策略。因此其成功率衡量当前行为策略在指定任务上的表现，不能直接解释为 CEM 的规划能力。

两种模式从相同的 held-out `tworoom_eval.h5` 任务启动。直接 head 缺少历史时重复初始观测，并以物理零动作补齐历史；不进行专家预热。context=16 时，完整的自主历史要经过 75 步才能形成，这种分布偏移可能影响直接 head 分数。

默认 100 个任务、每批 10 个并行环境、150 步预算、task_seed=42025。任务按合法起点均匀抽样，同一条轨迹可贡献多个起点，因此这些任务不保证统计独立。成功判定沿用 stable-worldmodel 0.0.6：预算内任一步触发 TwoRoom `terminated`，计为成功；该版本距离阈值为 16。成功率单位为百分比。

`--goal-offset 25` 明确表示目标位于 start+25。stable-worldmodel 0.0.6 的 `load_chunk` 使用右开区间，所以向旧 API 传入 26。旧入口直接传入 25，实际取到 start+24；旧入口抽样还遗漏最后一个合法候选。因此新旧入口的数值不能视为同协议复现。新入口也使用训练集动作统计，这一差异记录在 protocol.json 中。比较不同模型时请统一使用新入口、相同参数和相同数据文件。

## DelftBlue 运行

### 与自己的 inverse 基线比较

按导师确认的实验设置，TwoRoom 比较统一使用 **150 个原始环境步**。下文中论文的 50 步仅记录已发表文字，不作为本次比较预算。

inverse checkpoint 可用同一入口的 `--mode cem` 评估，动作归一化、任务抽样、目标偏移和 CEM 均保持一致：

```bash
sbatch planning/experiments/planning_eval/eval_tworoom_policy.sbatch \
  "/scratch/$USER/smwm-runs/tworoom_inverse_lambda_0p1_seed0" \
  --mode cem --num-eval 100 --batch-size 10 --eval-budget 150 \
  --goal-offset 25 --task-seed 42025 --seed 42
```

比较前核对各次 protocol.json 的参数、episodes、start_steps、数据文件和训练设置。只有一个训练 seed 的结果不能代表跨 seed 均值。inverse 的 CEM 与 policy 模型的 CEM 是主要对照，direct 是额外的行为策略评估。没有训练过 policy head 的 inverse run 会拒绝 direct/both；旧 checkpoint 没有 head 与新 checkpoint 保存未训练 head 两种格式都支持，其他参数仍严格校验。

需要现有 Linux GPU 环境、`stable-worldmodel==0.0.6`，以及 `/scratch/$USER/smwm-data/` 中的 `tworoom_train.h5` 和 `tworoom_eval.h5`。checkpoint 必须同时有 `config.yaml` 与 `checkpoints/last.ckpt`。严格加载任何缺失、额外或形状不符的模型权重都会报错，不会用随机权重继续评分。

从仓库根目录提交，先用两个任务跑通完整路径：

```bash
RUN="/scratch/$USER/smwm-runs/tworoom_policy_mamba_c16k1_seed0"
sbatch planning/experiments/planning_eval/eval_tworoom_policy.sbatch \
  "$RUN" --mode both --num-eval 2 --batch-size 2
```

正式评估：

```bash
sbatch planning/experiments/planning_eval/eval_tworoom_policy.sbatch \
  "$RUN" --mode both --num-eval 100 --batch-size 10
```

只测一种使用 `--mode cem` 或 `--mode direct`；视频加 `--save-video`。调度分区和 account 沿用该仓库 DelftBlue 脚本，可以用 sbatch 参数覆盖。数据路径可用 `EXTERNAL_DATA_ROOT` 覆盖。输出为 `/scratch/$USER/smwm-evals/<run-name>/<job-id>/`，任务日志在提交目录的 `tworoom-policy-eval-<job-id>.out`。

已有 GPU allocation 中也可直接运行：

```bash
python planning/eval_tworoom_policy.py \
  --run-dir "$RUN" --data-root "/scratch/$USER/smwm-data" \
  --output-dir "/scratch/$USER/smwm-evals/manual-$(date +%Y%m%d-%H%M%S)" \
  --mode both --num-eval 100 --batch-size 10
```

输出目录必须不存在，避免覆盖旧结果。`protocol.json` 保存 checkpoint SHA256、训练配置、评估参数、动作统计、起点和目标索引；`summary.json` 保存两种模式的成功率、逐任务成功标记及耗时；每批另保存原始 metrics，可选保存视频。程序不进行训练。

## 验证范围

CPU 回归测试覆盖任务边界、动作归一化、闭环动作块和历史对齐、裁剪后动作回填、cold start、context=1、无动作输入，以及五种真实 head 的 c16 输入和多步输出。运行 `python -m unittest discover -s planning/tests -v`。

本地没有用户的 checkpoint、TwoRoom 数据或 DelftBlue GPU 会话；这些单元测试不代表完成了真实导航评估。首次集群运行应检查日志和少量 rollout，再进行正式实验。主指标是闭环成功率，不是离线 policy-loss/MSE。
