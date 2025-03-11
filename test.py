def update_critic(
    self,
    rgb_obs,
    low_dim_obs,
    gripper_pose,
    next_gripper_pose,
    action,
    reward,
    discount,
    next_rgb_obs,
    next_low_dim_obs,
    demos,
):
    metrics = dict()
    
    with torch.no_grad():
        next_action, mets = self.critic.get_action(next_rgb_obs, next_low_dim_obs)
        target_q_probs_a = self.critic_target.compute_target_q_dist(
            next_rgb_obs, next_low_dim_obs, next_action, reward, discount
        )
        if self.use_logger:
            metrics.update(**mets)

    # **1️⃣ 计算 Critic 相关的 Q-value**
    q_probs, q_probs_a, log_q_probs, log_q_probs_a = self.critic(
        rgb_obs, low_dim_obs, action
    )
    q_critic_loss = -torch.sum(target_q_probs_a * log_q_probs_a, 3).mean()
    critic_loss = self.critic_lambda * q_critic_loss

    if self.use_logger:
        metrics["q_critic_loss"] = q_critic_loss.item()

    # **2️⃣ 计算 BC 相关 Loss**
    if self.bc_lambda > 0.0:
        demos = demos.float().squeeze(1)
        if self.use_logger:
            metrics["ratio_of_demos"] = demos.mean().item()

        if torch.sum(demos) > 0:
            q_probs_cdf = torch.cumsum(q_probs, -1)
            q_probs_a_cdf = torch.cumsum(q_probs_a, -1)
            bc_fosd_loss = (
                (q_probs_a_cdf.unsqueeze(-2) - q_probs_cdf)
                .clamp(min=0)
                .sum(-1)
                .mean([-1, -2, -3])
            )
            bc_fosd_loss = (bc_fosd_loss * demos).sum() / demos.sum()
            critic_loss += self.bc_lambda * bc_fosd_loss

            if self.use_logger:
                metrics["bc_fosd_loss"] = bc_fosd_loss.item()

    # **3️⃣ 计算 gripper_pose 的变化趋势**
    delta_pose = next_gripper_pose - gripper_pose  # (512, 8, 7)
    delta_pos = delta_pose[:, :, :3]  # 位置变化 (512, 8, 3)
    delta_rot = delta_pose[:, :, 3:]  # 旋转变化 (512, 8, 4)

    # 计算速度（Δpos/Δt）
    velocity = delta_pos[:, 1:, :] - delta_pos[:, :-1, :]  # 速度变化 (512, 7, 3)

    # 计算加速度（ΔV/Δt）
    acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]  # 加速度变化 (512, 6, 3)

    # 计算整体趋势（是否在朝向某个方向）
    avg_velocity = torch.mean(velocity, dim=1)  # (512, 3)
    avg_acceleration = torch.mean(acceleration, dim=1)  # (512, 3)

    # **4️⃣ 通过增量变化趋势自动判断阶段**
    approach_phase = avg_velocity.norm(dim=-1) > avg_acceleration.norm(dim=-1)  # 速度大于加速度 -> 仍在移动
    push_phase = (avg_velocity[:, 2] < 0) & (avg_acceleration[:, 2] > 0)  # Z 轴速度下降但加速度上升 -> 推门
    pull_phase = (avg_velocity[:, 2] > 0) & (avg_acceleration[:, 2] < 0)  # Z 轴速度上升但加速度下降 -> 拉门

    # **5️⃣ 计算动作合理性分数**
    action_validity_score = torch.zeros_like(approach_phase.float())  # 初始化合理性分数
    action_validity_score[approach_phase] = 1.0  # 接近阶段
    action_validity_score[push_phase] = 1.0  # 推门阶段
    action_validity_score[pull_phase] = 1.0  # 拉门阶段

    # **6️⃣ 计算任务相关 Loss**
    task_loss = (1 - action_validity_score) * (avg_velocity.norm(dim=-1) + avg_acceleration.norm(dim=-1))
    task_loss = task_loss.mean()

    # **7️⃣ 调整 Critic 价值估计**
    weighted_task_loss = torch.mean(q_probs_a.detach()) * task_loss
    critic_loss += self.critic_lambda * weighted_task_loss

    if self.use_logger:
        metrics["task_loss"] = task_loss.item()
        metrics["weighted_task_loss"] = weighted_task_loss.item()

    # **8️⃣ 更新 Critic**
    self.encoder_opt.zero_grad(set_to_none=True)
    self.critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    self.critic_opt.step()
    self.encoder_opt.step()

    return metrics
