# 固定基座 Panda / Isaac Sim 上的 I2O-RL

本适配器把提供的 `i2o_rl-master` I2O 智能体接入 `/home/ubuntu/pingandog` 下的固定基座 Panda 方块与锅盖场景，不涉及 Go2、ReKep、RLBench 运行时或语言模型。共享 Isaac Sim 文件与原始 I2O 仓库未修改。

本文保留各阶段历史配置与运行记录，当前交付状态见上一级 [README.md](../README.md)。以下历史启动器可能自动选择 GPU；执行前须遵循当前设备约束，不应直接照抄历史卡号。

## 在 jiangxing 服务器运行

在 `/home/ubuntu/pingandog` 下执行：

```bash
scripts/run_isaac62.sh isaac_i2o/train.py \
  --steps 64 --demos 2 --batch-size 8 --pretrain-updates 8 --eval-episodes 1
```

现有启动器选择空闲 GPU、持有项目 GPU 锁，并提供相匹配的 CUDA/Vulkan 设备。已验证环境为项目本地 Isaac Sim 6.2.0、PyTorch 2.11.0+cu128 与 RTX 6000D。不要把上游 conda 环境装入该运行时。

较长实验示例，不保证成功或收敛：

```bash
scripts/run_isaac62.sh isaac_i2o/train.py \
  --steps 20000 --demos 20 --batch-size 32 --pretrain-updates 1000 \
  --eval-episodes 10 --capacity 10000 --checkpoint-every 1000
```

进程运行至完成，README 本身不会自动启动。启动器输出日志，产物位于 `outputs/i2o_panda/run_*/`。`latest` 指向最近完成的运行，包括失败情况，需读取 `result.json`。

继续训练模型与优化器：

```bash
scripts/run_isaac62.sh isaac_i2o/train.py \
  --resume /absolute/path/to/checkpoint.pt \
  --steps 1000 --demos 10 --batch-size 32
```

恢复编码器、价值网络、目标价值网络及可用优化器状态。设置 `--demo-dir` 时重新加载固定数据，否则重新采集示范。该基础入口的回放、随机数流和本地计数器重新初始化，不是回放/随机状态的精确续跑；帧堆叠维度必须匹配。

## 基础方块环境约定

- 固定基座 Panda、PhysX，物理 120 Hz，每动作 12 个物理步，即 10 Hz。
- 一路世界固定 RGB，84×84，默认两帧通道堆叠，形状 `[views=1, channels=6, height=84, width=84]`。
- 每帧状态 15 维：7 个实测机械臂关节、归一化手指开合度、TCP 位置及 XYZW 四元数；默认堆叠状态 30 维。
- 策略观测不含仿真方块坐标；专家和奖励函数可以使用。
- 8 维归一化动作：7 个命令关节增量（每控制步最多 0.04 rad），以及绝对手指开合度（-1 闭合，+1 表示每指 0.04 m）。两个耦合手指收到相同开合目标。
- 关节目标遵守 URDF 限位。专家与学习器均采用物理抓取测试中的有界关节位置偏置控制器补偿静态跟踪误差；学习器执行时不是笛卡尔逆运动学控制。
- 方块边长 4 cm、质量 50 g，默认 XY 随机 +/-2 cm；静态桌面高 0.2 m。
- 成功条件：比稳定后高度抬升大于 0.12 m，方块/TCP 距离小于 0.06 m，手指开合度小于 0.03 m，连续保持 20 个控制步（2 秒）。
- 稀疏奖励成功为 1，否则 0。成功、方块掉落或离开局部工作区时终止，默认上限 220 步。真正终止折扣为 0，继续或时间截断转移折扣为 0.99。
- 只有专家使用逆运动学，随后经过与学习器完全相同的归一化动作/环境接口。只有完整成功回合进入示范回放，失败尝试会报告；每条目标示范最多尝试 3 次，数量不足则中止。

该场景是小型单物体集成任务，未实现完整避碰轨迹规划或部署安全控制器。

## 学习实现与输出

`vendor/i2o_rl.py` 保留原实现的 CNN、交叉注意力融合、3 层/5 区间/51 回报原子及分布式 Q 损失。2026-11-12 增加可微 SoftArgmax 动作分支与有效的示范动作 MSE。相对导入避免与 Isaac Sim 无关的 `utils` 模块冲突。默认软推理；`--action-mode hard` 选择原始量化贪心执行。算法选择与实测验证见 [SOFTARGMAX.md](SOFTARGMAX.md)，原始/部署哈希见 `vendor/provenance.json`。这是提供实现的工程集成，不代表复现论文结果。

基础 `replay.py` 在有在线数据后混合 50% 示范和 50% 在线转移。16 字段批次保留原更新接口；未使用任务/ACT 辅助字段以零占位，不参与有效损失。这个基础适配器没有成功轨迹重标机制。

输出包括 `demonstrations.npz`、`metrics.jsonl`、`policy_actions.json`、`checkpoint.pt`、观测 PNG 和 `result.json`。示范包含 RGB/状态/动作/奖励/折扣/下一观测。检查点包含模型、优化器、环境形状及运行配置。

`result.json` 中 `pass=true` 仅表示集成检查通过：专家数据成功、策略动作有界、损失有限、编码器/价值网络权重变化及检查点重载编码器一致。它不表示学习策略已通过抓取任务；学习器表现由 `eval_results` 与 `eval_success_rate` 单独报告。

## 首次实测结果

`run_20261111_132359`：2 条成功示范、250 个转移，64 个在线策略步，8 次纯示范及 64 次混合更新。全部集成检查通过；价值网络参数最大变化 0.002009，编码器变化 0.000955。独立学习器评估 **0/1 成功**，属于短冒烟测试，不是已训练抓取策略。

当时单视角、两帧堆叠，与原 RLBench 四视角、八帧配置不同。仍需长训练、重复评估、保留初始场景和多种子验证任务表现。

## 锅盖任务与固定示范

独立采集、回合格式与审计见 [DEMOS.md](DEMOS.md)。`--task take_lid_off_saucepan` 选择四相机任务。`--demo-dir DIRECTORY` 加载固定数据且不重新采集；不传 `--demos` 时使用全部保存回合。[SAUCEPAN.md](SAUCEPAN.md) 说明几何与相机安装，时钟同步修正见 DEMOS.md。

## 2026-11-13 关节物体双任务

`aligned/articulated.py` 增加程序化 PhysX 开抽屉与开门近似任务。成功条件与固定版本 RLBench Python JointCondition 一致：抽屉绝对位移大于 0.15 m；门铰链绝对转角大于 25 度，把手旋转超过 25 度前铰链锁定。抽屉只对应底层抽屉近似，没有导入原生三种变体几何。不使用夹爪附着。原生任务源码保存在 `tmp/i2o_audit/upstream_open_drawer.py`、`upstream_open_door.py`、`upstream_conditions.py`。

服务器 `outputs/articulated_probe/pair_12` 中，两任务专家、无接触负对照及命令/观测夹爪一致性均通过，每任务独立重置三次。抽屉使用 20 N/m 弱回位弹簧防止无命令滑动。柜体几何、惯量、摩擦、相机及 XY +/-2 cm 复位分布均属迁移差异。机械臂 PD 增益 8000/150，手指 4000/100。离散夹爪同时检查观测状态与遗留执行目标，防止闭合命令后保留张开电机目标。

这不是原生资产转换或严格论文基准。门专家先转动/解锁把手，再释放并通过接触推门板；物体位姿、任务标记或阶段时钟不进入学习器。

`run_articulated_pair.py` 可在共享 GPU 租约下运行两个独立进程，也可通过 `--separate-gpus` 使用 `run_isaac62.sh` 为每任务取得独立项目 GPU 租约。当时运行采用独立 GPU，每张 RTX 6000D 约 83 GiB；各任务独立保留原始帧缓存、模型、回放、数据集和检查点。不停止其他用户 GPU 进程。并行只在空闲渲染/计算资源允许时节省墙钟时间，不保证两倍吞吐。

每任务新采集 100 条成功示范，从新模型开始，使用状态 120/动作 8 接口、共享粗到细输出头、beta=1，无阶段输出头，稀疏 0/1 奖励，批量 256，计划 30250 在线步。门回合上限 125，抽屉 100，遵循下载任务配置。保留锅盖实验的论文文字重构损失，与未修改仓库的硬解码/C51/FOSD 目标不同。每 1000 步评估 5 次，最终 100 次，仅一个种子。进入 RL 前要求独立复位成功至少 4/5、TCP 位移大于 3 cm，预热最多 5000 次更新。

各任务 `latest.pt/previous.pt` 恢复模型、优化器及回放元数据；恢复时物理回合重置，未完成回合截断。双任务监督器限制重试次数、监控进度停滞，在学习门槛失败或运动塌缩警报时停止对应任务，另一任务可继续。任务级 `PAUSE` 标记只暂停对应任务。

当时运行目录为 `outputs/articulated_training/run_20261113_v2/{open_drawer,open_door}`，持久用户服务 `pingandog-articulated-training.service`，活动指针 `outputs/articulated_training/active.json`。较早的 `run_20261113` 在采集阶段停止，抽屉接受 16 条；门命令/状态断言正确阻止训练。旧运行和旧示范未复用，新运行每任务重新采集 100 条。被拒绝的夹爪命令/状态样本记入 `manifest.rejections`，不进入回放。

## 2026-11-13 原生资产训练

独立新资产实验位于 `outputs/native_asset_training/run_20261113`，由 `pingandog-native-asset-training.service` 管理。不复用程序化基线数据或检查点。`--scene native_assets --separate-gpus` 为每任务取得独立 GPU 文件锁，不预留或停止无关 GPU 分配。只有专家验证成功后写入的 `ready/<task>.json` 才允许启动。上述训练门槛、每任务 100 条示范、30250 在线步、恢复及评估设置同样适用。

抽屉使用 NVIDIA Isaac Sektion 柜底层抽屉，成功位移大于 0.15 m，底座高 0.1 m，根位置 (0.95,0,0.4842)、偏航 pi，把手局部 TCP 偏移 (0.290,0,0.01)。为适应导入 Panda 手指，该位置比 Isaac Lab 的 0.305 m 坐标深入 15 mm。

门使用固定版本原始 RLBench 网格、材质和关节限位转换为 USD，底座高 0.2 m，根位置 (0.36,-0.13,0.20)，比首次探测靠近 6 cm。保留原始工具朝向，避免此前尝试对称翻滚 180 度导致开门时腕部/关节限位失败。把手超过 25 度解锁，铰链超过 25 度成功。锁定电机映射至 PhysX，锁定时最大力 1000，解锁后恢复为 1。

专家使用实际手指接触，不附着、不直接命令物体关节。门先接近原始前两个路点且保留工具朝向，再根据接触反馈沿实际把手/铰链轴运动。这替代原专家末段固定路点/路径，属于适配专家，不是同一套 RLBench 示范。

保留标准 URDF TCP；Coppelia 到 URDF 的标定随姿态变化，因此未应用拟合变换。使用四路 84×84 RGB、八帧、状态 120/动作 8、单共享动作头，无阶段观测。放置/随机化、相机标定、PhysX 和论文文字重构目标仍与严格复现不同。

2026-11-13 原生抽屉任务通过检查点从 GPU 4 迁移到 GPU 7。任务 `gpu.txt` 保存 `7`，由 `scripts/run_isaac62.sh` 读取并传入 `PINGANDOG_REQUIRED_GPU`，恢复时保持该分配。没有标记的任务仍自动选卡。原生门当时在 GPU 0，无关任务未停止。固定卡号并不阻止其他用户占用显存；这些卡号仅为历史记录。

## 2026-11-14 原始微波炉与书架任务

历史运行位于 `outputs/extended_asset_training/run_20261114`，用户服务 `pingandog-extended-asset-training.service`。`run_articulated_pair.py --tasks open_microwave put_books_on_bookshelf --scene native_assets --separate-gpus` 使用各任务就绪标记，当时固定 GPU 0/2。各任务仅在专家与无接触负对照通过后启动。之前锅盖、抽屉和门的实验与产物保持不变。

两任务模型及 Python 定义来自 RLBench 提交 `b80e51feb3694d9959cb8c0408cd385001b01382`，下载哈希、来源和许可位于 `assets/candidates/rlbench_<task>_original`。通过 CoppeliaSim 4.1 公共接口导出原始网格、材质像素/UV、动态书本质量/惯量、微波炉关节轴/限位和路点位姿，再映射为 USD。`asset_tools/build_rlbench_task_usd.py` 构建网格；`tune_contact_offsets.py` 把薄物体接触偏移设为 1 mm、静止偏移 0、求解器迭代 16/4，保留盒形书本凸包。原始物体不缩放。

微波炉：原 JointCondition 为铰链绝对位移超过 80 度，原限位 0..90 度，回合上限 125。根位置 (0.45,0,0.30)，支撑面高 0.28 m。原零目标速度电机和最大力矩 2.073600292 Nm 映射为零刚度、阻尼 2000 的 PhysX 角驱动，力矩上限保持原值。专家接近原始两个抓握路点后，依靠实际接触沿原铰链运动，无附着和直接关节驱动；安全接近插值替代原生避碰运动规划。

书架：原 variation 0 只要求 `book2` 进入原成功传感器，专家在 waypoint5 停止；variation 1/2 要求两本/三本书，此处未合并。选择 variation 0 遵循下载 wrapper 默认值与 README 单变体采集设置，论文未说明确切变体索引。回合上限 175，根位置 (0.40,0,0.04)，支撑面高 0.03 m。三本原书均保持动态、可抓取。专家沿原始 waypoint0..5，增加安全净空接近和初始静置后书本平移补偿。原夹爪/TCP 到导入 URDF 的标定只用于专家目标位姿；三个标定位姿间最大矩阵变化 0.0034168，因此是近似而非精确机器人等同。

RLBench 书本注册为可抓取并使用夹爪附着。适配器在闭合夹爪 TCP 感知盒与书相交时建立运行时固定关节，张开时释放，策略执行期间不修改物体位姿。它近似原生接近传感器：TCP 半尺寸 (.025,.025,.035) m，使用有向包围盒投影测试；附着可能在闭合开始时发生，而非 Coppelia 驱动完成后。策略只看到共享状态 120/动作 8，不接收阶段或附着标记。这与纯接触抓握的抽屉/门不同。

原书架成功条件不要求释放。成功传感器使用原有向长方体与书本碰撞盒的分离轴重叠测试，不是只看中心距离。与 `sim.checkProximitySensor(original success, original book2)` 交叉验证，175 个采样位置全部一致，见 `export/sensor_reference.json`、`sensor_validation.json`；有限测试不是所有位姿的证明。

仍未对齐的因素包括：PhysX 与 Coppelia 接触/控制、导入 Panda/TCP/夹爪感知近似、相机标定，以及原全局放置/偏航采样（当前只随机 XY +/-2 cm、偏航固定）。损失仍为论文文字重构，不是原样使用下载代码 C51/FOSD。

两任务均为四路 84×84 RGB、八帧、状态 120/动作 8、单输出头、beta=1、稀疏 0/1 奖励、无 VLP、新采 100 条接受示范、模仿门槛 4/5 且运动大于 3 cm、预热上限 5000、在线 30250 步、周期评估 5 次及最终 100 次。仅一个训练种子，不是完整三种子复现。已知专家逆运动学不可达回合按种子/原因拒绝，计入 200 次尝试上限，不作为成功；其他控制/数据断言仍直接报错。恢复检查点还原模型、优化器、回放和随机数状态，并重置未完成物理回合。
