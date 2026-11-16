# 训练监控与自动恢复

`watch_training.py` 每 60 秒检查一次 `outputs/i2o_panda/active_formal_saucepan.json` 指向的实验。它接管对现有训练服务的监控，不会因此重启训练。如果服务在写入 `complete` 进度标记前退出，监控器会从本实验及其恢复链中选择最新且完整的原子保存 PyTorch ZIP 检查点，并启动独立 systemd 用户服务。每次尝试使用独立输出目录和日志。

恢复时保留训练参数，仅修改恢复路径、输出目录、运行标签，并把重复预训练次数设为零。恢复模型、目标网络、优化器、步数和更新计数。最终步检查点可以只恢复最终评估。

用户服务为 `pingandog-i2o-watchdog.service`，独立于 SSH；现有 `Linger=yes` 允许退出登录后继续运行。最多重启训练 5 次，相邻重启至少间隔 120 秒。只要没有完成标记，即使进程正常退出也按中断处理。损坏的 ZIP 文件会被跳过；若无有效检查点或重启次数耗尽，则记录原因并退出。ZIP 检查不能代替训练器执行的张量和配置校验。

进程仍存活但超过一小时没有进度时，会在监控状态中标记，不自动终止，因为渲染、评估和磁盘写入可能耗时较长。该机制主要检测进程退出，不能发现所有 GPU 死锁。不发送外部通知。

## 查看服务器状态

```bash
systemctl --user status pingandog-i2o-watchdog.service
cat /home/ubuntu/pingandog/outputs/i2o_panda/watchdog_state.json
cat /home/ubuntu/pingandog/outputs/i2o_panda/active_formal_saucepan.json
tail /home/ubuntu/pingandog/logs/i2o_watchdog.log
```

主动停止训练前，先暂停自动恢复：

```bash
touch /home/ubuntu/pingandog/outputs/i2o_panda/WATCHDOG_PAUSE
```

删除该标记后允许继续恢复。完全禁用监控服务：

```bash
systemctl --user disable --now pingandog-i2o-watchdog.service
```

恢复从最后检查点开始，不是从中断的那条指令继续。该历史版本每 1000 个在线步骤保存一次检查点；旧检查点未保存在线回放、随机数状态或物理仿真状态，这些会重新初始化，最后检查点之后未保存的进度会重复执行。若最终评估中断，评估从头重做。

验证：本地与服务器均运行了 4 项单元测试，覆盖存活进程不受干扰、异常退出恢复与配置保留、重启退避、完成状态、检查点损坏、重启上限和主动暂停。部署时对实际训练服务进行了检查，没有终止该训练。
