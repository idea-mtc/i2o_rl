# 论文与原始代码及 Isaac 适配的复现核对

> 2026-11-13 更新：下文的“当前暂停”指审计时的旧训练。新对齐流水线已经单独启动，旧模型与旧 watchdog 仍暂停。新配置和状态边界见 [ALIGNED_RUN_20261113.md](ALIGNED_RUN_20261113.md)，不能把旧工程修复版的成功率用于新版本。

当前强化学习与自动恢复均已暂停，最后记录为在线第1184步。现有检查点、录像和数据保留。此前将最初的 Isaac 运行称为“按论文配置训练”不够准确：只对齐了部分超参数，动作接口、示范、解码和训练流程均有改动。先前的停滞不能当作论文原方法失败的证据。

核对基准分为三个：提供论文的9页匿名稿、本地未修改的 `i2o_rl-master`、我们编写的 Isaac 适配。最初 Isaac 运行与最新工程修复版分别列出，不把二者混为论文实现。

## 源码版本与证据范围

用户提供仓库地址为 [idea-mtc/i2o_rl](https://github.com/idea-mtc/i2o_rl)。本次 GitHub 插件、匿名 HTTPS 网页、API 和 master/main 原始文件访问均返回404，尚不能取得官方 commit，也不能判断是私有、改名还是权限问题。此处的“原始代码”特指本地副本，不把它未经核验地等同于作者产生论文结果的最终版本。

本地 README 自称 re-implementation；没有 `.git` 历史。本地 `i2o_rl.py` 的 SHA256 为 `d324f80fbe3b2d65ff595c9668d6289dee0936648c3b49eb6e274b2e7f17d9ae`，与此前保存的源文件记录一致，未被我们的适配覆盖。最初正式运行的 vendor 快照为 `b622408bf8304aa8184dda23252019793c9cea82d96fa3da8f5df47452672b00`，已经不同于原文件。

README 指定的 RLBench 版本 `b80e51feb3694d9959cb8c0408cd385001b01382` 的源码已通过公开 HTTPS 取回，用于核验动作和抓取语义。原微信临时PDF路径现已失效，本次论文核对使用此前保存的完整提取文本和第4至7页图像；没有把网上同名出版条目默认为同一版正文。

## 论文与本地原始代码本身的差异

|项目|论文描述|本地原始代码及结论|
|---|---|---|
|动作类型|第3页明确为关节角增量|`JointPosition(False)`，7个相对当前实测位置的关节增量，加离散夹爪；这一大类与论文一致，不是末端笛卡尔动作|
|动作选择|图2及式4、5：SoftArgmax，beta=1，分层连续解码|`get_action()` 用整数 argmax 或平坦Q时随机bin，再量化；对atoms维做softmax是在计算C51概率，不是对动作bin做SoftArgmax|
|显式动作模仿|式7：动作平方误差加Q margin|原代码是 `F.l1_loss`，而输入动作经过argmax和量化；独立测试证实该L1没有梯度|
|其他模仿路径|式7写margin|代码另有FOSD项和margin，两者仍可训练Q；不能因L1断梯度就说整个算法不能学|
|RL目标|式8写标量Bellman平方误差|代码采用51 atoms的C51分布交叉熵，support[-2,2]；是否为作者未写明的实现细节，需对应实验版本确认|
|奖励|VLP生成结构化距离奖励；表2写成功bonus1000|普通入口默认稀疏奖励；两个wrapper的 `shaped_rewards=True` 均被注释，未核实完整VLP奖励进入训练的链路|
|论文奖励表述|表2成功bonus1000|第5页又写成功为0、否则惩罚，存在需作者澄清的表述；不能任意挑一处解释后宣称严格复现|
|批大小|表2为256|配置写128，但两个loader各128再拼接，实际更新批为256；不能只看YAML就认定这里不一致|
|训练阶段|算法1交替IL与RL，未明确各阶段长度|入口加载示范后直接进行混合更新，没有独立1000步MSE预训练|
|最终统计|3次独立训练，每次100随机场景|默认开发评估每1000帧25回合；需另行落实论文最终评估协议|

论文第7页表1报告拿锅盖：I2O-RL为90.3%，VL-I2O为92.7%。这是两条不同的比较目标：先复现不含VLP的I2O-RL，再加入VLP复现完整框架，不能把没有VLP单独当成纯I2O复现失败的理由。

## 动作与环境差异

|项目|本地原始 RLBench 基准|最初 Isaac 版本|最新工程修复版|
|---|---|---|---|
|模拟器与资产|RLBench和原生Panda、锅盖场景|Isaac Sim 6.2，自制锅体/锅盖，导入Panda|沿用自制场景，不是原RLBench资产|
|策略动作|8维：7关节增量、离散夹爪|8维：命令目标增量、连续夹爪|4维：末端XYZ目标、夹爪；固定向下姿态|
|增量参照点|当前实测关节角|上一次命令目标，持续累加|由网络末端目标经IK转换，再限制关节目标变化|
|动作归一化|逐关节示范min/max，向外扩20%后线性映射|所有关节统一乘0.04rad|XYZ按工作空间[.38,-.08,.30]至[.54,.08,.58]米映射|
|控制时序|arm一次scene step；gripper开合可额外阻塞多个物理步，arm动作后重设当前位置目标|固定120Hz物理、10Hz控制，每步12个物理步|固定10Hz，加普通IK与目标缓存|
|电机控制|原生PyRep控制，速度2、加速度8的配置|PD及积分偏置，目标每步最多0.04rad|沿用该底层控制，加IK目标转换|
|夹爪动作|Discrete，阈值0.5开/关，完整执行开合|每指位置0.02*(a+1)米|仍为连续位置夹爪|
|抓取机制|Discrete默认启用grasp附着，任务注册可抓取锅盖|强制无附着，仅靠接触摩擦|同样无附着、纯摩擦|
|任务成功|GraspedCondition与DetectedCondition的组合|自定抬升>15cm、XY偏移<5cm、把手/TCP<5cm、闭合条件连续30步|同上；验收另外要求相对滑移<15mm|
|单回合上限|配置100个策略步|260步|260步|
|专家轨迹|RLBench原生专家采集100回合|自编固定35/40/25/45/35步时序和IK脚本|模仿标签改成原示范各阶段终点目标|
|专家动作标签|下一命令位置减当前实测关节角，或实测位置差|下一命令目标减上一次命令目标，再除0.04|阶段末端位置标签，不再是原来执行的一步动作|
|相机|原RLBench四个相机，84x84|四视角名称/顺序一致；位置、光学和腕部安装自定义，168x126缩至84x84|同最初Isaac|
|机器人状态|关节、gripper_open、tip pose，每帧15维，8帧120维|维数相同，但grip为连续指位归一化，TCP自行按URDF标定|再附加6个时序特征，共126维|
|人工阶段信息|没有提供给策略的固定时钟或阶段头|没有显式阶段输入，脚本专家却按时间切换|5阶段指示和阶段内进度，5个任务阶段输出头|

原代码的3个 coarse-to-fine 层是动作离散精度层级，不是我们增加的5个任务阶段。论文VLP的子任务奖励也不等于把任务阶段时钟直接送入策略。

## 算法与训练流程差异

|项目|本地原始基准|最初 Isaac 版本|最新工程修复版|
|---|---|---|---|
|视觉网络|4路CNN，通道32/64/128/256，融合与Q网络|基本复用，修正非连续张量flatten|复用视觉编码器，Q输出增为5个阶段头|
|分层与分布参数|3层、5bins、51atoms、support[-2,2]|相同|保留这些参数，动作维度改为4|
|动作选择|硬argmax、平坦Q随机bin、最终量化|我们添加的可微C51期望Q SoftArgmax，beta1|beta5并将最外层bin中心范围归一化到[-1,1]|
|显式模仿项|无梯度L1项，另有FOSD及margin|替换为可微MSE，并保留FOSD/margin|阶段目标MSE，加原FOSD/margin|
|独立预训练|入口没有单独MSE阶段|1000次纯MSE，工程选择|最终候选纯MSE500次，阶段均衡采样；仅迁移旧视觉编码器，重新学习动作Q输出|
|示范用途|真实一步转移同时用于TD与模仿|真实自编轨迹同时用于TD与模仿|重标注阶段目标只用于模仿，排除示范TD，TD只用新在线转移|
|回放初始化|示范同时预填普通buffer和demo buffer|示范与纯在线池分离|仍分离，旧数据BC-only|
|在线成功数据|成功轨迹会重标为demo，进入成功池|没有实现这一成功轨迹晋升机制|同样没有实现|
|早期demo比例|混合两个池，但普通池也含示范，初期接近100%示范|在线池出现后固定约50%示范|约50%旧示范，且TD只来自在线部分|
|容量与采样|每池配置1,000,000，4 worker，nstep1|在线容量50,000，NumPy进程内采样|相同工程回放实现|
|批大小|128+128=256|256|在线256；独立模仿128|
|优化器|AdamW、lr5e-5、wd0.1|一致|一致|
|gamma与目标网络|gamma.99、tau.02|一致|一致|
|探索|归一化std.01，平坦Q时可随机bin|std.01；新增soft动作路径没有随机bin机制|std.002，动作物理尺度已不同|
|增强|RandomShift pad4|联合训练pad4，独立MSE无增强|pad0，与模仿验证的原始输入一致|
|总更新计数|30250环境帧，action_repeat1，直接混合更新|30250在线加1000MSE|新30250在线加500MSE，不计入旧运行6720步|
|奖励与VLP|普通入口默认稀疏0/1；VLP进入训练链路待核|自定成功+1，没有VLP|仍然稀疏+1，没有VLP|
|评估|开发配置25回合/1000帧；论文最终3x100|每5000步10回合、最终100，只有1次训练|额外1000步10回合，仍未做论文3次独立训练|
|依赖与硬件|仓库Python3.10、torch2.3.1；论文Ubuntu22.04、4090 24GB|Isaac Python3.12、torch2.11+cu128、Ubuntu24.04、RTX6000D|相同服务器环境|

还有一项版本兼容缺口：本地wrapper在reset和step调用 `self.task.get_tip_pose()`，而README固定commit的官方TaskEnvironment没有这个方法。可能依赖作者额外修改过的RLBench，或这些调用是遗留代码；仅按README安装不能据此认定完整实验已可复现。此项与“模型不动”不同，属于首先需要解决的版本/可运行性问题。

## 为什么先前看起来不动

### 已经核实的执行差异

原始接口的关节目标关系是：

```text
q_goal = q_measured + unnormalize_per_joint(a)
```

最初Isaac接口却是：

```text
q_goal_next = q_goal_previous + 0.04 * a
```

后者多了未直接输入策略的命令积分状态，以及内部bias积累。专家标签也相对于这个命令状态生成。两者不是同一个控制问题，也不是只换模拟器后仍保持相同的动作定义。

按原代码，归一化动作0会映射到示范范围的中点，不一定是物理零增量；夹爪动作0经映射约为0.6，会执行离散“开”。最初Isaac中动作0意味着关节不再增量移动，同时每指位置约0.02米。网络输出相同的数值，实际行为可能完全不同。

对我们自己100条示范、14885个转移的审计显示，约46.93%的动作前7维全零，命令目标与实测关节角的RMS差约0.00398rad。接近/下降/抬升阶段各有约三分之一零动作，闭合/保持阶段全部零关节动作。这些数据解释了为何小动作误差或大量零动作上的低损失，不足以保证能自行启动下一阶段；它们不是原论文示范统计。

### 我添加的 SoftArgmax 路径也有区别

本地原始实现平坦Q时会随机选择bin；我添加的soft实现对相等Q取平均，输出趋近0。配合统一0.04尺度和较小探索噪声，陌生状态下容易表现为小动作或停滞。这是对执行记录和解码结构的解释，尚未以只替换这一机制的对照实验证实其独立贡献。

对于我们新增的特定实现，即每bin使用范围[-2,2]的C51期望Q，再以beta1进行SoftArgmax，3层5bins的动作幅值上界约为0.90736。这个上界只属于该组合实现，不能推广成“论文beta1一定有问题”：论文没有明确与C51 atoms如何连接，其回归头/动作分数也可能采用不同尺度。原代码按示范统计留有归一化裕量，离散夹爪也不要求连续输出精确到-1。

实际对照中，硬最大值的离线动作误差较低，但闭环仍停在预抓取位置。因此此前把问题直接归结为“SoftArgmax幅度不足”或“必须加入阶段时钟”都过早。阶段时钟和笛卡尔控制能使工程任务成功，不能作为论文缺少这些组件的证据。

### 还不能确定唯一根因

此前第5000步是我们已改动的Isaac策略，10回合均失败；其中一个复现回合距把手最近仍20.4厘米，实际是少量移动后停滞，而非控制指令完全没有执行。相同场景脚本专家成功，说明底层能抓取，但不能排除学习接口/数据/损失的问题。

目前已证明存在多项足以改变任务的差异，还没有做“只修测量基准”“只修归一化”“只修夹爪”“只换soft解码”的受控消融，不能给每项分配因果比例或宣布唯一原因已完全定位。

## 论文是否也出现了机械臂不动

论文表1报告拿锅盖I2O-RL90.3%、VL-I2O92.7%。第6页图4的其他任务学习曲线初期有接近0的成功率，但这只说明早期失败，不说明机械臂不运动；没有拿锅盖早期逐步动作日志可供同样诊断。

不能从我们5000步、10回合、不同控制器和任务中的失败，反推论文在30250步、3x100最终评估也会失败。也不能从本地副本的断梯度问题，断言作者最终实验版本有同样代码。反过来，论文报告的高成功率也不能替本地实现证明正确。

最新工程版在其自定标准下模仿20/20成功，随后在线1000步评估10/10成功；这两项均不是论文RLBench基准结果。该版现在停在1184步，自动恢复关闭。

## 恢复复现工作的顺序

1. 固定可读取的作者仓库commit和论文版本，明确拿锅盖目标是纯I2O-RL90.3那一行，还是含VLP的92.7那一行。
2. 先建立原生RLBench基线：依照固定依赖核验原始专家、相机、动作统计、实测关节增量、离散夹爪、默认抓取机制和任务成功条件。解决get_tip_pose等版本兼容问题，但不顺手改网络和任务。
3. 把论文SoftArgmax/MSE与本地hard/L1的冲突单独解决：需要与作者实验版本核对；若只能自行实现，应明确叫论文方法重实现，保留原代码基线，分别做梯度与闭环对照。
4. 保留3层离散Q的原结构，不添加固定阶段时钟和5个任务阶段头；使用原始一步专家动作与正确归一化。保留原回放预填和成功轨迹重标记。
5. 在基准中先验证动作编码/解码往返、实测关节参考、专家重放和梯度路径，再按30250步及3次独立训练、每次100随机场景评估。
6. Isaac作为另一条迁移实验：逐项保持动作、夹爪、归一化、成功判据和数据一致，报告物理引擎与资产差异，不把其绝对成功率直接当作原论文表1复现。

当前只完成核对与暂停，没有自动恢复训练、回滚代码或安装另一套模拟器。

## 代码与实测证据

- [论文保存全文](/Users/linxiaopeng/project/pingandog/tmp/pdfs/i2o/full.txt)，[公式页](/Users/linxiaopeng/project/pingandog/tmp/pdfs/i2o/page_5.png)，[实验设置页](/Users/linxiaopeng/project/pingandog/tmp/pdfs/i2o/page_6.png)，[结果与超参数页](/Users/linxiaopeng/project/pingandog/tmp/pdfs/i2o/page_7.png)。
- [原始动作及统计](/Users/linxiaopeng/project/pingandog/i2o_rl-master/rlbench_env.py:297)，[原始硬动作解码](/Users/linxiaopeng/project/pingandog/i2o_rl-master/i2o_rl.py:293)，[原始L1模仿项](/Users/linxiaopeng/project/pingandog/i2o_rl-master/i2o_rl.py:648)。
- [原始配置](/Users/linxiaopeng/project/pingandog/i2o_rl-master/cfgs/config_rlbench.yaml)，[混合批处理](/Users/linxiaopeng/project/pingandog/i2o_rl-master/utils.py:150)，[回放重标记](/Users/linxiaopeng/project/pingandog/i2o_rl-master/replay_buffer.py:103)。
- [最初Isaac控制与专家标签](/Users/linxiaopeng/project/pingandog/isaac_i2o/saucepan_env.py:248)，[当前动作转换](/Users/linxiaopeng/project/pingandog/isaac_i2o/cartesian_goals.py)，[当前时序条件](/Users/linxiaopeng/project/pingandog/isaac_i2o/phase_clock.py)。
- [原始代码梯度实测](/Users/linxiaopeng/project/pingandog/isaac_i2o/audit_evidence/original_gradient_check.json)，[自编示范统计](/Users/linxiaopeng/project/pingandog/isaac_i2o/audit_evidence/isaac_demo_action_audit.json)，[暂停状态](/Users/linxiaopeng/project/pingandog/isaac_i2o/audit_evidence/pause_status.json)。
- [本地源码哈希](/Users/linxiaopeng/project/pingandog/isaac_i2o/audit_evidence/local_source_sha256.json)，[仓库地址访问结果](/Users/linxiaopeng/project/pingandog/isaac_i2o/audit_evidence/github_provenance_check.json)。
- 固定版本RLBench：[关节控制](https://github.com/stepjam/RLBench/blob/b80e51feb3694d9959cb8c0408cd385001b01382/rlbench/action_modes/arm_action_modes.py#L97)，[夹爪控制](https://github.com/stepjam/RLBench/blob/b80e51feb3694d9959cb8c0408cd385001b01382/rlbench/action_modes/gripper_action_modes.py#L40)，[任务成功条件](https://github.com/stepjam/RLBench/blob/b80e51feb3694d9959cb8c0408cd385001b01382/rlbench/tasks/take_lid_off_saucepan.py#L11)。本地核验副本位于 `tmp/i2o_audit/upstream_*.py`。
