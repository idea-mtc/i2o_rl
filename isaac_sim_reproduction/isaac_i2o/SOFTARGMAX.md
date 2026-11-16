# SoftArgmax 模仿学习实现与验收

2026-11-12 在固定基座 Panda / Isaac Sim 适配器中实现。该修改增加可微动作路径，不代表完成论文复现或已经学会抓取。

## 实现改动

`vendor/cqn_utils.py:soft_zoom_in` 在 **5 个动作区间**上计算 `softmax(beta * Q)` 权重，再计算浮点期望索引。设边界为 `low, high`，区间宽度为 `width=(high-low)/bins`，则：

```text
index = sum(softmax(beta*Q)[i] * i)
next_low = low + width * index
next_high = next_low + width
```

最后一层动作取 `(low+high)/2`。索引位于 `[0,bins-1]`，各层边界保持嵌套，动作无需裁剪即可处于 [-1,1]。`C2FCritic.get_soft_action` 保留贯穿三层及前层动作条件输入的梯度。Q 是现有 51 个回报原子分布的期望；softmax 作用于动作区间，不是回报原子索引。模仿动作路径中没有硬 argmax、向下取整、整数类型转换或 detach。该平滑期望区间细化方式属于具体实现选择，所提供论文未完整规定所有边界更新细节。

`I2OAgent.imitation_update` 只更新 MSE，不含 TD、margin、FOSD、奖励或数据增强，用于单独验证动作回归梯度。现有联合 `update` 也将示范行的不可微 L1 分支替换为可微 MSE，保留继承的 C51、FOSD 和价值间隔损失。MSE 权重为 `bc_lambda`，此前不起作用的 L1 分支使用固定 0.5 系数。保留的分布式损失和回放流程不等同于论文的简化公式。

`act` 默认采用连续软解码；`--action-mode hard` 保留量化贪心执行以供比较。两种模式的模仿损失均使用软解码。现有 Q 目标计算仍使用硬贪心动作。`--softargmax-beta` 默认 1.0。形状兼容的旧权重可以加载，但若未明确选择硬模式，推理语义会发生变化。

在 beta=1 时，回报支撑范围 [-2,2] 限制软权重的集中程度，因此未必能精确达到动作端点 +/-1。这可能形成端点回归误差下限，应结合奖励和支撑范围设计检查，不能通过改变评估来掩盖。

## 独立验证命令

在 `/home/ubuntu/pingandog` 下执行：

```bash
scripts/run_isaac62.sh isaac_i2o/validate_softargmax.py \
  --data outputs/i2o_panda/run_20261111_132359/demonstrations.npz \
  --updates 1000 --batch-size 32 --fit-samples 64 --beta 1 --eval-episodes 2
```

历史启动器通过项目锁选择空闲 GPU。共享运行环境、Go2 和原始 `i2o_rl-master` 源码未修改。旧部署模块备份在服务器 `isaac_i2o/backups/pre_softargmax_20261112/`。

## 验收结果

实验目录：`outputs/softargmax_panda/run_20261112_035910/`。`pass=true` 表示机制验证通过，不代表任务成功。

- 三层双精度有限差分梯度检查通过。
- 确认旧硬解码动作 MSE 的 `requires_grad=false`。
- 初始化时，新输出头和三层 Q 的梯度均非零。
- 上游输出头初始值为零，因此编码器和主干初始梯度为零。20 次纯 MSE 更新后，CNN、视觉/状态投影、融合模块、Q 主干、Q 输出头及三个层级均具有有限非零梯度。这是模块级检查，不表示每个参数都有梯度。
- 从首个成功回合抽取 64 个转移拟合，批量 32，更新 1000 次，学习率 5e-5；第二回合全部 125 个转移用作保留测试。
- 训练 MSE：0.11372009 → 0.000449401，下降 99.60%。
- 训练机械臂 MSE：0.03115148 → 0.0000115403。
- 训练夹爪 MSE：0.69170034 → 0.00351443。
- 保留测试 MSE：0.11557116 → 0.04063938，下降 64.84%。
- CNN 与动作头权重发生变化。另执行了一次联合 C51/margin/FOSD/MSE 更新，指标有限且 CNN 梯度非零；执行闭环评估前已恢复该更新，因此评估的仍是纯 MSE 检查点。
- 加载检查点后，仅用网络动作运行两个各 220 步的 Isaac Sim 回合，没有调用专家：成功抓取 **0/2**。动作均有限且有界；小数据拟合尚不能证明闭环泛化。

保存内容包括 `result.json`、`metrics.jsonl`、`predictions.npz`、`policy_trace.json`、评估 PNG 和 `soft_imitation.pt`。数据划分保存为索引；拟合未使用专家奖励。训练样本拟合不能代替回合成功率或独立多种子测试。

## 从这些权重继续联合训练

```bash
scripts/run_isaac62.sh isaac_i2o/train.py \
  --resume outputs/softargmax_panda/run_20261112_035910/soft_imitation.pt \
  --steps 1000 --demos 10 --batch-size 32 --action-mode soft
```

纯模仿检查点只导出模型权重。训练入口对此报告 `model_only_new_optimizers` 并初始化新优化器；普通训练检查点保留优化器状态。回放、示范采集和计数器按 README 说明重新初始化。上述命令仅为示例，不会自动启动长实验。

联合训练续跑在 `outputs/i2o_panda/run_20261112_040128` 中验证：16 个策略环境步、4 次示范预更新及 16 次在线联合更新，从纯 MSE 权重初始化新优化器，全部集成检查通过。这不是额外的抓取成功率评估。
