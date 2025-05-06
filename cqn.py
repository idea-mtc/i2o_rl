from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from cqn_utils import (
    random_action_if_within_delta,
    zoom_in,
    encode_action,
    decode_action,
)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class ImgChLayerNorm(nn.Module):
    def __init__(self, num_channels, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MultiViewCNNEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 4
        self.num_views = obs_shape[0]
        self.repr_dim = self.num_views * 256 * 5 * 5  # for 84,84. hard-coded

        self.conv_nets = nn.ModuleList()
        for _ in range(self.num_views):
            conv_net = nn.Sequential(
                nn.Conv2d(obs_shape[1], 32, 4, stride=2, padding=1),
                ImgChLayerNorm(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                ImgChLayerNorm(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                ImgChLayerNorm(128),
                nn.SiLU(),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                ImgChLayerNorm(256),
                nn.SiLU(),
            )
            self.conv_nets.append(conv_net)

        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor):
        # obs: [B, V, C, H, W]
        obs = obs / 255.0 - 0.5
        hs = []
        for v in range(self.num_views):
            h = self.conv_nets[v](obs[:, v])
            h = h.view(h.shape[0], -1)
            hs.append(h)
        h = torch.cat(hs, -1)
        return h

class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)  # 投影 Query
        self.key_proj = nn.Linear(feature_dim, feature_dim)  # 投影 Key
        self.value_proj = nn.Linear(feature_dim, feature_dim)  # 投影 Value
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, low_dim_feat, rgb_feat):
        """
        low_dim_feat: (batch_size, feature_dim)  # Query
        rgb_feat: (batch_size, feature_dim)      # Key & Value
        """
        Q = self.query_proj(low_dim_feat).unsqueeze(0)  # (1, batch, feature_dim)
        K = self.key_proj(rgb_feat).unsqueeze(0)  # (1, batch, feature_dim)
        V = self.value_proj(rgb_feat).unsqueeze(0)  # (1, batch, feature_dim)

        attn_output, _ = self.attn(Q, K, V)  # (1, batch, feature_dim)
        return attn_output.squeeze(0)  # (batch, feature_dim)

class BiCrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.cross_attn_1 = CrossAttentionFusion(feature_dim, num_heads)
        self.cross_attn_2 = CrossAttentionFusion(feature_dim, num_heads)

    def forward(self, low_dim_feat, rgb_feat):
        fused_1 = self.cross_attn_1(low_dim_feat, rgb_feat)  # 状态关注视觉
        fused_2 = self.cross_attn_2(rgb_feat, low_dim_feat)  # 视觉关注状态
        return torch.cat([fused_1, fused_2], dim=-1)  # 连接两个特征


class C2FCriticNetwork(nn.Module):
    def __init__(
        self,
        repr_dim: int,
        low_dim: int,
        action_shape: Tuple,
        feature_dim: int,
        hidden_dim: int,
        levels: int,
        bins: int,
        atoms: int,
    ):
        super().__init__()
        self._levels = levels
        self._actor_dim = action_shape[0]
        self._bins = bins

        # Advantage stream in Dueling network
        self.adv_rgb_encoder = nn.Sequential(
            nn.Linear(repr_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.adv_low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.adv_fusion = BiCrossAttentionFusion(feature_dim)
        self.adv_net = nn.Sequential(
            nn.Linear(
                feature_dim * 2 + self._actor_dim + levels, hidden_dim, bias=False
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.adv_head = nn.Linear(hidden_dim, self._actor_dim * bins * atoms)
        self.adv_output_shape = (self._actor_dim, bins, atoms)

        # Value stream in Dueling network
        self.value_rgb_encoder = nn.Sequential(
            nn.Linear(repr_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.value_low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.value_fusion = BiCrossAttentionFusion(feature_dim)
        self.value_net = nn.Sequential(
            nn.Linear(
                feature_dim * 2 + self._actor_dim + levels, hidden_dim, bias=False
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.value_head = nn.Linear(hidden_dim, self._actor_dim * 1 * atoms)
        self.value_output_shape = (self._actor_dim, 1, atoms)

        self.apply(utils.weight_init)
        self.adv_head.weight.data.fill_(0.0)
        self.adv_head.bias.data.fill_(0.0)
        self.value_head.weight.data.fill_(0.0)
        self.value_head.bias.data.fill_(0.0)

    def forward(
        self, level: int, rgb_obs: torch.Tensor, low_dim_obs, prev_action: torch.Tensor
    ):
        """
        Inputs:
        - level: level index
        - obs: features from visual encoder
        - low_dim_obs: low-dimensional observations
        - prev_action: actions from previous level

        Outputs:
        - q_logits: (batch_size, action_dim, bins, atoms)
        """
        level_id = (
            torch.eye(self._levels, device=rgb_obs.device, dtype=rgb_obs.dtype)[level]
            .unsqueeze(0)
            .repeat_interleave(rgb_obs.shape[0], 0)
        )
        auto = self.value_rgb_encoder(rgb_obs)
        auto2  = self.value_low_dim_encoder(low_dim_obs)
        value_h = torch.cat(
            [auto, auto2],
            -1,
        )
        value_fusion = self.value_fusion(auto, auto2)
        value_h = value_h + value_fusion
        
        value_x = torch.cat([value_h, prev_action, level_id], -1)
        values = self.value_head(self.value_net(value_x)).view(
            -1, *self.value_output_shape
        )

        auto3 = self.adv_rgb_encoder(rgb_obs)
        auto4 = self.adv_low_dim_encoder(low_dim_obs)

        adv_h = torch.cat(
            [auto3, auto4], -1
        )
        adv_fusion = self.adv_fusion(auto3, auto4)
        adv_h = adv_h+adv_fusion
        adv_x = torch.cat([adv_h, prev_action, level_id], -1)
        advs = self.adv_head(self.adv_net(adv_x)).view(-1, *self.adv_output_shape)

        q_logits = values + advs - advs.mean(-2, keepdim=True)
        return q_logits


class C2FCritic(nn.Module):
    def __init__(
        self,
        action_shape: tuple,
        repr_dim: int,
        low_dim: int,
        feature_dim: int,
        hidden_dim: int,
        levels: int,
        bins: int,
        atoms: int,
        v_min: float,
        v_max: float,
    ):
        super().__init__()

        self.levels = levels
        self.bins = bins
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        actor_dim = action_shape[0]
        self.initial_low = nn.Parameter(
            torch.FloatTensor([-1.0] * actor_dim), requires_grad=False
        )
        self.initial_high = nn.Parameter(
            torch.FloatTensor([1.0] * actor_dim), requires_grad=False
        )
        self.support = nn.Parameter(
            torch.linspace(v_min, v_max, atoms), requires_grad=False
        )
        self.delta_z = (v_max - v_min) / (atoms - 1)

        self.network = C2FCriticNetwork(
            repr_dim,
            low_dim,
            action_shape,
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
        )

    def get_action(self, rgb_obs: torch.Tensor, low_dim_obs: torch.Tensor):
        metrics = dict()
        low = self.initial_low.repeat(rgb_obs.shape[0], 1).detach()
        high = self.initial_high.repeat(rgb_obs.shape[0], 1).detach()

        for level in range(self.levels):
            q_logits = self.network(level, rgb_obs, low_dim_obs, (low + high) / 2)
            q_probs = F.softmax(q_logits, 3)
            qs = (q_probs * self.support.expand_as(q_probs).detach()).sum(3)
            argmax_q = random_action_if_within_delta(qs)
            if argmax_q is None:
                argmax_q = qs.max(-1)[1]  # [..., D]
            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)

            # for logging
            qs_a = torch.gather(qs, dim=-1, index=argmax_q.unsqueeze(-1))[
                ..., 0
            ]  # [..., D]
            metrics[f"critic_target_q_level{level}"] = qs_a.mean().item()
        continuous_action = (high + low) / 2.0  # [..., D]
        return continuous_action, metrics

    def forward(
        self,
        rgb_obs: torch.Tensor,
        low_dim_obs: torch.Tensor,
        continuous_action: torch.Tensor,
    ):
        """Compute value distributions for given obs and action.

        Args:
            obs: [B, repr_dim] shaped feature tensor
            low_dim_obs: [B, low_dim] shaped feature tensor
            continuous_action: [B, D] shaped action tensor

        Return:
            q_probs: [B, L, D, bins, atoms] for value distribution at all bins
            q_probs_a: [B, L, D, atoms] for value distribution at given bin
            log_q_probs: [B, L, D, bins, atoms] with log probabilities
            log_q_probs_a: [B, L, D, atoms] with log probabilities
        """

        discrete_action = encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )

        q_probs_per_level = []
        q_probs_a_per_level = []
        log_q_probs_per_level = []
        log_q_probs_a_per_level = []

        low = self.initial_low.repeat(rgb_obs.shape[0], 1).detach()
        high = self.initial_high.repeat(rgb_obs.shape[0], 1).detach()
        for level in range(self.levels):
            q_logits = self.network(level, rgb_obs, low_dim_obs, (low + high) / 2)
            argmax_q = discrete_action[..., level, :].long()  # [..., L, D] -> [..., D]

            # (Log) Probs [..., D, bins, atoms]
            # (Log) Probs_a [..., D, atoms]
            q_probs = F.softmax(q_logits, 3)  # [B, D, bins, atoms]
            q_probs_a = torch.gather(
                q_probs,
                dim=-2,
                index=argmax_q.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat_interleave(self.atoms, -1),
            )
            q_probs_a = q_probs_a[..., 0, :]  # [B, D, atoms]

            log_q_probs = F.log_softmax(q_logits, 3)  # [B, D, bins, atoms]
            log_q_probs_a = torch.gather(
                log_q_probs,
                dim=-2,
                index=argmax_q.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat_interleave(self.atoms, -1),
            )
            log_q_probs_a = log_q_probs_a[..., 0, :]  # [B, D, atoms]

            q_probs_per_level.append(q_probs)
            q_probs_a_per_level.append(q_probs_a)
            log_q_probs_per_level.append(log_q_probs)
            log_q_probs_a_per_level.append(log_q_probs_a)

            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)

        q_probs = torch.stack(q_probs_per_level, -4)  # [B, L, D, bins, atoms]
        q_probs_a = torch.stack(q_probs_a_per_level, -3)  # [B, L, D, atoms]
        log_q_probs = torch.stack(log_q_probs_per_level, -4)
        log_q_probs_a = torch.stack(log_q_probs_a_per_level, -3)
        return q_probs, q_probs_a, log_q_probs, log_q_probs_a

    def compute_target_q_dist(
        self,
        next_rgb_obs: torch.Tensor,
        next_low_dim_obs: torch.Tensor,
        next_continuous_action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
    ):
        """Compute target distribution for distributional critic
        based on https://github.com/Kaixhin/Rainbow/blob/master/agent.py implementation

        Args:
            next_rgb_obs: [B, repr_dim] shaped feature tensor
            next_low_dim_obs: [B, low_dim] shaped feature tensor
            next_continuous_action: [B, D] shaped action tensor
            reward: [B, 1] shaped reward tensor
            discount: [B, 1] shaped discount tensor

        Return:
            m: [B, L, D, atoms] shaped tensor for value distribution
        """
        next_q_probs_a = self.forward(
            next_rgb_obs, next_low_dim_obs, next_continuous_action
        )[1]

        shape = next_q_probs_a.shape  # [B, L, D, atoms]
        next_q_probs_a = next_q_probs_a.view(-1, self.atoms)
        batch_size = next_q_probs_a.shape[0]

        # Compute Tz for [B, atoms]
        Tz = reward + discount * self.support.unsqueeze(0).detach()
        Tz = Tz.clamp(min=self.v_min, max=self.v_max)
        # Compute L2 projection of Tz onto fixed support z
        b = (Tz - self.v_min) / self.delta_z
        lower, upper = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        # Fix disappearing probability mass when l =b = u (b is int)
        lower[(upper > 0) * (lower == upper)] -= 1
        upper[(lower < (self.atoms - 1)) * (lower == upper)] += 1

        # Repeat Tz for (L * D) times -> [B * L * D, atoms]
        multiplier = batch_size // lower.shape[0]
        b = torch.repeat_interleave(b, multiplier, 0)
        lower = torch.repeat_interleave(lower, multiplier, 0)
        upper = torch.repeat_interleave(upper, multiplier, 0)

        # Distribute probability of Tz
        m = torch.zeros_like(next_q_probs_a)
        offset = (
            torch.linspace(
                0,
                ((batch_size - 1) * self.atoms),
                batch_size,
                device=lower.device,
                dtype=lower.dtype,
            )
            .unsqueeze(1)
            .expand(batch_size, self.atoms)
        )
        m.view(-1).index_add_(
            0,
            (lower + offset).view(-1),
            (next_q_probs_a * (upper.float() - b)).view(-1),
        )  # m_l = m_l + p(s_t+n, a*)(u - b)
        m.view(-1).index_add_(
            0,
            (upper + offset).view(-1),
            (next_q_probs_a * (b - lower.float())).view(-1),
        )  # m_u = m_u + p(s_t+n, a*)(b - l)

        m = m.view(*shape)  # [B, L, D, atoms]
        return m

    def encode_decode_action(self, continuous_action: torch.Tensor):
        """Encode and decode actions"""
        discrete_action = encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )
        continuous_action = decode_action(
            discrete_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )
        return continuous_action


class CQNAgent:
    def __init__(
        self,
        rgb_obs_shape,
        low_dim_obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        levels,
        bins,
        atoms,
        v_min,
        v_max,
        bc_lambda,
        # sc_lambda, #space constraints and global strategy
        bc_margin,
        critic_lambda,
        critic_target_tau,
        weight_decay,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        use_logger,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_logger = use_logger
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.bc_lambda = bc_lambda
        # self.sc_lambda = sc_lambda
        self.bc_margin = bc_margin
        self.critic_lambda = critic_lambda

        # models
        self.encoder = MultiViewCNNEncoder(rgb_obs_shape).to(device)
        self.critic = C2FCritic(
            action_shape,
            self.encoder.repr_dim,
            low_dim_obs_shape[-1],
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
            v_min,
            v_max,
        ).to(device)
        self.critic_target = C2FCritic(
            action_shape,
            self.encoder.repr_dim,
            low_dim_obs_shape[-1],
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
            v_min,
            v_max,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.AdamW(
            self.encoder.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=lr, weight_decay=weight_decay
        )

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.eval()

        print(self.encoder)
        print(self.critic)

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.critic.train(training)

    def act(self, rgb_obs, low_dim_obs, step, eval_mode):
        rgb_obs = torch.as_tensor(rgb_obs, device=self.device).unsqueeze(0)
        low_dim_obs = torch.as_tensor(low_dim_obs, device=self.device).unsqueeze(0)
        rgb_obs = self.encoder(rgb_obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        action, _ = self.critic_target.get_action(
            rgb_obs, low_dim_obs
        )  # use critic_target
        stddev = torch.ones_like(action) * stddev
        dist = utils.TruncatedNormal(action, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        action = self.critic.encode_decode_action(action)
        return action.cpu().numpy()[0]

    def update_critic(
        self,
        rgb_obs,
        low_dim_obs,
        gripper_pose,
        next_gripper_pose,
        task_low_dim_state,
        next_task_low_dim_state,
        action,
        reward,
        discount,
        next_rgb_obs,
        next_low_dim_obs,
        demos,
        step,
    ):
        metrics = dict()
        # print("task_low_dim_state:{}".format(task_low_dim_state))
        # print("task_low_dim_state:{}".format(task_low_dim_state.shape))
        # pass
        # print("gripper_pose:{}".format(gripper_pose.shape))
        # print("next_gripper_pose:{}".format(next_gripper_pose.shape))
        with torch.no_grad():
            next_action, mets = self.critic.get_action(next_rgb_obs, next_low_dim_obs)
            target_q_probs_a = self.critic_target.compute_target_q_dist(
                next_rgb_obs, next_low_dim_obs, next_action, reward, discount
            )
            if self.use_logger:
                metrics.update(**mets)
        #     print("next_actiom:{}".format(next_action.shape))
        #     print("action{}".format(action))

        # print("rgb_obs_shape:{}".format(rgb_obs.shape))

        # Cross entropy loss for C51
        q_probs, q_probs_a, log_q_probs, log_q_probs_a = self.critic(
            rgb_obs, low_dim_obs, action
        )
        # next_q_probs, next_q_probs_a, next_log_q_probs, next_log_q_probs_a = self.critic(
        #     rgb_obs, low_dim_obs, action
        # )
        # mask = demos.float().squeeze(1)  # [B,] 
        # mask = (1 - mask).view(-1, 1, 1, 1).float()
        # masked_loss = -torch.sum(target_q_probs_a * log_q_probs_a*mask, 3).mean()
        # q_critic_loss = masked_loss.sum() / mask.sum().clamp(min=1.0)
        q_critic_loss = -torch.sum(target_q_probs_a * log_q_probs_a, 3).mean()
        critic_loss = self.critic_lambda * q_critic_loss

        if self.use_logger:
            metrics["q_critic_loss"] = q_critic_loss.item()

        # if self.sc_lambda > 0.0:
        #     next_action

        if self.bc_lambda > 0.0:
            qs = None
            demos = demos.float().squeeze(1)  # [B,]
            if self.use_logger:
                metrics["ratio_of_demos"] = demos.mean().item()

            if torch.sum(demos) > 0:
                stddev = utils.schedule(self.stddev_schedule, step)
                current_action, _ = self.critic.get_action(rgb_obs, low_dim_obs)
                stddev = torch.ones_like(current_action) * stddev
                dist = utils.TruncatedNormal(current_action, stddev)
                current_action = dist.mean
                current_action = self.critic.encode_decode_action(current_action)
                bc_loss = F.l1_loss(current_action, action, reduction='none')
                bc_loss = bc_loss.mean(dim=-1)
                # q_probs: [B, L, D, bins, atoms], q_probs_a: [B, L, D, atoms]
                q_probs_cdf = torch.cumsum(q_probs, -1)
                q_probs_a_cdf = torch.cumsum(q_probs_a, -1)
                # q_probs_{a_{i}} is stochastically dominant over q_probs_{a_{-i}}
                bc_fosd_loss = (
                    (q_probs_a_cdf.unsqueeze(-2) - q_probs_cdf)
                    .clamp(min=0)
                    .sum(-1)
                    .mean([-1, -2, -3])
                )
                bc_fosd_loss = (bc_fosd_loss * demos).sum() / demos.sum()
                bc_loss = (bc_loss * demos).sum()/demos.sum()
                critic_loss = critic_loss + self.bc_lambda * bc_fosd_loss + 0.5 * bc_loss
                if self.use_logger:
                    metrics["bc_fosd_loss"] = bc_fosd_loss.item()
                    metrics["bc_loss"] = bc_loss.item()

                if self.bc_margin > 0:
                    qs = (q_probs * self.critic.support.expand_as(q_probs)).sum(-1)
                    qs_a = (q_probs_a * self.critic.support.expand_as(q_probs_a)).sum(
                        -1
                    )
                    margin_loss = torch.clamp(
                        self.bc_margin - (qs_a.unsqueeze(-1) - qs), min=0
                    ).mean([-1, -2, -3])
                    margin_loss = (margin_loss * demos).sum() / demos.sum()
                    critic_loss = critic_loss + self.bc_lambda * margin_loss
                    if self.use_logger:
                        metrics["bc_margin_loss"] = margin_loss.item()

        # # 提取关键位置信息
        # door_frame_pos = task_low_dim_state[:, :, :3]  
        # door_handle_pos = task_low_dim_state[:, :, 3:]  
        # next_door_handle_pos = next_task_low_dim_state[:, :, 3:]  
        # grasp_pose = gripper_pose[:, :, :3]  
        # next_grasp_pose = next_gripper_pose[:, :, :3]  

        # # 计算 **相对变化率** 而不是固定阈值
        # handle_pos_change = next_door_handle_pos - door_handle_pos  
        # handle_change_rate = torch.norm(handle_pos_change, dim=-1) / (torch.norm(door_handle_pos, dim=-1) + 1e-6)  

        # grasp_to_handle_dist = torch.norm(door_handle_pos - grasp_pose, dim=-1)  
        # next_grasp_to_handle_dist = torch.norm(next_door_handle_pos - next_grasp_pose, dim=-1)  
        # approach_rate = (grasp_to_handle_dist - next_grasp_to_handle_dist) 

        # # 训练阶段随 step 变化的动态权重
        # max_steps = 30250  
        # progress = step / max_steps  
        # approach_weight = 1.0 - torch.sigmoid(torch.tensor((progress - 0.3) * 10))  # 训练前30%主要关注接近
        # grasp_weight = torch.sigmoid(torch.tensor((progress - 0.5) * 10))  # 训练到50%后开始注重抓取
        # push_weight = torch.sigmoid(torch.tensor((progress - 0.7) * 10))  # 训练到70%后主要关注推门

        # # 根据动态权重判断任务阶段
        # approach_phase = approach_rate > 0.05  
        # grasp_phase = grasp_to_handle_dist < (0.2 + 0.5 * (1 - progress))  # 初期要求宽松，后期要求精确
        # push_phase = handle_change_rate > 0.05  

        # final_phase = (approach_phase * approach_weight) + (grasp_phase * grasp_weight) + (push_phase * push_weight)

        # # 计算 task_fosd_loss
        # q_probs_cdf = torch.cumsum(q_probs, -1)
        # q_probs_a_cdf = torch.cumsum(q_probs_a, -1)

        # demos_inverse = 1 - demos
        # task_fosd_loss = (
        #     (q_probs_a_cdf.unsqueeze(-2) - q_probs_cdf)
        #     .clamp(min=0)
        #     .sum(-1)
        #     .mean([-1, -2, -3])
        # )
        # task_fosd_loss = ((1 - final_phase.float()) * task_fosd_loss * demos_inverse).sum() / (demos_inverse.sum() + 1)
        # critic_loss = critic_loss + task_fosd_loss

        # if self.use_logger:
        #     metrics["task_fosd_loss"] = task_fosd_loss.item()
        #     # metrics["weighted_task_loss"] = weighted_task_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics        
        batch = next(replay_iter)
        # 其余数据转换为 Tensor
        (
            rgb_obs,
            low_dim_obs,
            gripper_pose,
            next_gripper_pose,
            task_low_dim_state,
            next_task_low_dim_state,
            action,
            reward,
            discount,
            next_rgb_obs,
            next_low_dim_obs,
            demos,
            policy_rgb_obs,
            policy_next_rgb_obs,
            policy_q_pos,
            policy_next_q_pos,
        ) = utils.to_torch(batch, self.device)

        # augment
        rgb_obs = rgb_obs.float()
        next_rgb_obs = next_rgb_obs.float()
        rgb_obs = torch.stack(
            [self.aug(rgb_obs[:, v]) for v in range(rgb_obs.shape[1])], 1
        )
        next_rgb_obs = torch.stack(
            [self.aug(next_rgb_obs[:, v]) for v in range(next_rgb_obs.shape[1])], 1
        )
        # encode
        rgb_obs = self.encoder(rgb_obs)
        with torch.no_grad():
            next_rgb_obs = self.encoder(next_rgb_obs)

        if self.use_logger:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(
                rgb_obs,
                low_dim_obs,
                gripper_pose,
                next_gripper_pose,
                task_low_dim_state,
                next_task_low_dim_state,
                action,
                reward,
                discount,
                next_rgb_obs,
                next_low_dim_obs,
                demos,
                step,
            )
        )

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
