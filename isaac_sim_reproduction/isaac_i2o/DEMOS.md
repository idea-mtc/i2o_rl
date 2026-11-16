# 固定锅盖示范数据与训练任务选择

锅盖任务的采集与训练使用同一个任务工厂。`train.py --task take_lid_off_saucepan --demo-dir DIRECTORY` 加载已保存回合，不再运行专家。加载前检查任务、环境与相机约定、成功状态、回合边界和 SHA256 校验和。

## 采集

```bash
cd /home/ubuntu/pingandog
scripts/run_isaac62.sh isaac_i2o/collect_demos.py \
  --episodes 10 --pilot 10 --seed 12000
```

输出的 `DEMO_DIRECTORY` 是固定数据集目录。采集具有独立清单、每次尝试的种子及结果、预采集审计和最终审计。专家失败尝试会被计数并拒绝，不会重标为成功。成功回合原子写入且不覆盖。

使用相同代码版本和配置继续采集：

```bash
scripts/run_isaac62.sh isaac_i2o/collect_demos.py \
  --resume datasets/take_lid_off_saucepan/EXISTING_COLLECTION \
  --episodes 100 --pilot 10 --seed 12000
```

续采拒绝任务、配置或源码版本发生变化，以及重复文件写入。种子范围互不重叠的已完成批次可用 `merge_demos.py` 合并。该脚本重新审计所有输入，要求采集代码与配置一致，复制所有记录并保留来源链；拒绝已存在的输出目录及重叠种子。合并清单保留所有尝试，不只保留被接受的回合。

## 数据格式

`manifest.json` 标识格式为 `isaac_i2o_raw_episode_v1`；每个 `episodes/*.npz` 包含：

| 字段 | 形状与含义 |
|---|---|
| rgb | `[T+1,4,3,84,84]`，uint8，未堆叠的原始 RGB |
| state | `[T+1,15]`，7 个关节、归一化开合度、TCP XYZ 与 XYZW |
| actions | `[T,8]`，实际执行的归一化关节增量与夹爪命令 |
| rewards, discounts | `[T]`，稀疏终端成功奖励与自举折扣 |
| terminated, truncated | `[T]`，真正任务结束与时间上限截断 |
| object_positions, knob_positions, tcp_positions | `[T+1,3]`，仅用于物理审计 |
| sim_time | `[T+1]`，实际仿真时钟，相邻动作严格相隔 0.1 秒 |
| phase | `[T]`，专家阶段，仅作元数据 |

初始观测位于第一个动作之前，每个动作连接 t 与 t+1 的观测。物体位姿与专家阶段不进入学习器观测。图像不重复存储八份，加载时在回合内部堆叠；重置后的缺失历史只用本回合首帧填充。

相机顺序为 front、wrist、left_shoulder、right_shoulder。训练和采集以 168×126 渲染，与 640×480 场景预览具有相同 4:3 比例和安装位置，再双线性缩放为 84×84。配置明确记录这些信息，数据集必须与训练环境一致。没有合成图像生成。

成功状态会根据保存的物体、把手、TCP 和状态数组独立重算：最后 30 个控制步满足抬升至少 0.15 m、XY 偏移小于 0.05 m、把手与 TCP 距离小于 0.05 m、手指开合度小于 0.025 m，且相对滑移小于 0.015 m。还检查四视角图像非空且有变化、状态有限、动作有界、终端奖励和实际物理时间戳。20 次尝试内无法完成预采集则中止；大规模采集也有尝试次数上限。

## 使用已保存数据训练

```bash
scripts/run_isaac62.sh isaac_i2o/train.py \
  --task take_lid_off_saucepan --demo-dir DATASET_DIRECTORY \
  --steps 8 --batch-size 8 --pretrain-updates 4 --eval-episodes 0
```

不传 `--demos` 时加载全部固定回合；`--demos 10` 只限制加载数量，不重新生成数据。不传 `--demo-dir` 时仍可使用旧的即时采集路径，默认两条示范。默认方块任务保留；锅盖任务默认最长 260 个控制步。

该短命令检查任务选择、示范加载、四相机输入、回放、有限损失、在线动作、权重变化及检查点重载，不是长训练或成功率实验。现有 `--pretrain-updates` 使用组合 I2O 损失，不是纯 MSE 预训练。此前单相机方块检查点与四相机形状不兼容。

## 时钟修正

初始采集审计发现每个动作实际耗时 0.125 秒，而非计划的 0.1 秒，原因是实时渲染额外推进物理仿真。这些尝试已拒绝。锅盖控制器现先执行 12 个不渲染的物理步，仅在图像渲染时暂停物理，然后明确恢复。记录的 `sim_time` 验证每动作 0.1 秒。这不同于早期笔记中的暂停后未恢复问题。方块和 Go2 控制器未改动。

## 本阶段的边界

这些是专家示范，不是已训练策略。资产为项目自建的程序化 Isaac Sim 场景，不是原始 RLBench 网格。数据采用 XY 各 +/-2 cm 随机化，不能证明广泛物体或外观泛化。批量 256 的 GPU 显存、长训练、保留测试与 VLM 奖励属于后续工作。此阶段 `audit_demos.py` 仅在 CPU 检查批量 256 和八帧填充，不表示已验证同批量 GPU 训练。

## 2026-11-12 实测交付

固定数据集：`/home/ubuntu/pingandog/datasets/take_lid_off_saucepan/fixed_100_20261112`。
共 100 个成功回合、100 个独立种子、14885 个转移；压缩回合数据 478964550 字节，约 457 MiB。五个独立完成的批次为 10+23+23+22+22，种子范围不重叠；时钟修正后的采集尝试均被接受。早期时钟调试运行单独保存，未混入。每批均通过预采集审计；完整重读数据、重算物理成功、验证哈希，并加载全部 100 回合至回放。CPU 批量 256 形状和八帧重置填充检查通过，不能作为 GPU 显存测试。

清单 SHA256：
`c9c7f39f38700d9565f8e119b7ac72afc5fadfbc7b337c5eac86897ef0d071ce`。
报告为数据集内的 `audit.json` 和 `loader_audit.json`。合并数据集是冻结产物，不用于续采。采集后审计器补强了独立时间戳检查，其哈希与历史采集源码哈希分开记录。

训练集成测试 `outputs/i2o_panda/run_20261112_072620` 从磁盘加载 10 回合预采集数据，没有重新采集；执行 8 个网络控制步及 12 次更新，有限损失、编码器/价值网络变化和检查点重载检查通过。CNN 特征展开由 view 改为 reshape，修复实时四视角张量步幅兼容问题。该测试不代表任务成功或长训练完成。

使用最终固定数据的短检查示例：

```bash
scripts/run_isaac62.sh isaac_i2o/train.py \
  --task take_lid_off_saucepan \
  --demo-dir datasets/take_lid_off_saucepan/fixed_100_20261112 \
  --steps 8 --batch-size 8 --pretrain-updates 4 --eval-episodes 0
```

该示例默认加载全部 100 回合。上述 GPU 集成冒烟测试使用 10 回合预采集数据；全部 100 回合的加载在 CPU 上另行验证。截至本历史记录形成时，长训练尚未启动。
