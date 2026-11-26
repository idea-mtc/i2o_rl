# LeWM 开抽屉预测实验

在已有的机器人开抽屉轨迹上，独立训练动作条件世界模型 LeWM。输入是四路相机画面与动作，输出是 192 维未来画面特征。训练和验证在离线轨迹上进行，评估指标是特征预测误差及优于“保持当前特征”的片段比例。

| 配置 | 观察与动作 | 评分目标 |
|---|---|---|
| 单步 | 观察第 0 步，输入随后 1 个八维动作 | 第 1 步 |
| 五步 | 观察第 0、5、10 步，输入第 0 到第 14 步的动作，按顺序分成 3 块 | 第 15 步 |

两组均从头训练 1,000 次更新，每批 32 个片段，使用 AdamW，学习率 0.00005，权重衰减 0.001。损失为未来特征均方误差加上 0.09 × SIGReg。每个采样时刻将四张 84 × 84 相机图像拼成一张图并缩放至 224 × 224。五步模型同时学习三个间隔为 5 步的预测，验证时对最后一次预测评分。

## 准备数据

设置环境变量 `LEWM_DRAWER_DATA` 指向本机数据根目录。目录应包含专家轨迹 `dataset/*.npz`、相应的 `dataset/manifest.json`、策略采样轨迹 `online_episodes/*.npz` 和动作范围文件 `action_statistics.json`。输入字段及检查逻辑见 [数据适配器](lewm_drawer_data.py)。示范和在线数据按整条轨迹划分，两个配置读取同一份清单。

Python 环境需要支持 BF16 的 CUDA PyTorch、Transformers（ViT）、einops、NumPy 和 Pillow。从本目录执行以下命令。CPU 检查会生成本地清单及动作和画面对齐样例。GPU 命令以物理 GPU 2 为例；在 jiangxing 上只可使用空闲的 2、3、4，启动器会检查占用并按 GPU UUID 隔离。训练和校准写入新的输出目录。

```bash
export LEWM_DRAWER_DATA=/path/to/open_drawer
python lewm_drawer_data.py --root "$LEWM_DRAWER_DATA" --upstream . --out audit
mkdir -p runs
./on_idle_gpu.sh 2 python train_lewm_drawer.py runs/single
./on_idle_gpu.sh 2 python train_lewm_drawer_cube.py runs/five
./on_idle_gpu.sh 2 python calibrate.py single runs/single/checkpoint.pt runs/single/calibrated
./on_idle_gpu.sh 2 python calibrate.py five runs/five/checkpoint.pt runs/five/calibrated
```

## 如何查看自己的结果

每次训练在输出目录生成 `checkpoint.pt`、配置、`result.json` 和逐片段 `validation.csv`。校准时固定模型权重，用训练集的 128 批、每批 32 个样本重估 BatchNorm 统计量，再用固定统计量评估验证集。校准脚本向新目录写结果，保留原检查点。每种配置和校准方式应与其对应的“保持不变”基线比较。拉动分组由夹爪动作和机械臂运动筛选；五步评估还检查正面相机中缝隙亮度的变化。这些指标用于评估预测，策略成功率需要另做闭环测试。

`jepa.py` 和 `module.py` 取自 [LeWM 官方代码](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac)，沿用 [原始 MIT 许可证](LICENSE-LeWM)。
