from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from detr.models.transformer import _get_activation_fn
from typing import Optional, List
from torch import nn, Tensor
import IPython
e = IPython.embed
import utils
from utils import get_config
from cqn_utils import (
    random_action_if_within_delta,
    zoom_in,
    encode_action,
    decode_action,
    flatten_time_dim_into_channel_dim,
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


class TransformerEncoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = self.linear2(self.activation(self.linear1(src2)))
        # src = src + self.dropout2(src2)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.linear2(self.activation(self.linear1(src2)))
        return src2

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class MultiViewCNNEncoder(nn.Module):
    def __init__(self, obs_shape, hidden_dim=256, backbones = None):
        super().__init__()

        assert len(obs_shape) == 4
        self.num_views = obs_shape[0]
        self.repr_dim = self.num_views * 256 * 5 * 5  # for 84,84. hard-coded
        self.has_backbone = False

        if backbones is not None:
            self.input_proj = self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.conv_nets = nn.ModuleList(backbones)
            self.has_backbone = True
            # self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
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
        if self.has_backbone:
            all_cam_features = []
            for v in range(self.num_views):
                features, pos = self.conv_nets[0](obs[:, v])
                features = features[0]
                all_cam_features.append(self.input_proj(features))
            return all_cam_features
        for v in range(self.num_views):
            h = self.conv_nets[v](obs[:, v])
            h = h.view(h.shape[0], -1)
            hs.append(h)
        h = torch.cat(hs, -1)
        return h

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
        gripper_pose = 7*8,
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
        self.adv_gp_encoder = nn.Sequential(
            nn.Linear(gripper_pose, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.adv_Transformer = TransformerEncoder(feature_dim, 1)
        self.adv_net = nn.Sequential(
            nn.Linear(
                feature_dim * 3 + self._actor_dim + levels, hidden_dim, bias=False
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
        self.value_gp_encoder = nn.Sequential(
            nn.Linear(gripper_pose, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.value_Transformer = TransformerEncoder(feature_dim, 4)
        self.value_net = nn.Sequential(
            nn.Linear(
                feature_dim * 3 + self._actor_dim + levels, hidden_dim, bias=False
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
        self, level: int, rgb_obs: torch.Tensor, low_dim_obs, gripper_pose, prev_action: torch.Tensor
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
        # auto3 = self.value_gp_encoder(gripper_pose)
        # batch, sequence, length = gripper_pose.shape
        # gripper_pose = gripper_pose.view(batch, -1)

        value_h = torch.stack(
            [self.value_rgb_encoder(rgb_obs), self.value_low_dim_encoder(low_dim_obs), self.value_gp_encoder(gripper_pose)],
            dim=1,
        )
        value_src = torch.cat([self.value_rgb_encoder(rgb_obs), self.value_low_dim_encoder(low_dim_obs), self.value_gp_encoder(gripper_pose)],-1)
        value_temp = self.value_Transformer(value_h)
        value_temp = value_temp.reshape(*value_temp.shape[:-2], -1) + value_src
        value_temp = value_src
        try:
            value_x = torch.cat([value_temp, prev_action, level_id], -1)
        except:
            value_temp = value_temp.unsqueeze(0)
            value_x = torch.cat([value_temp, prev_action, level_id], -1)
        values = self.value_head(self.value_net(value_x)).view(
            -1, *self.value_output_shape
        )

        adv_h = torch.stack(
            [self.adv_rgb_encoder(rgb_obs), self.adv_low_dim_encoder(low_dim_obs), self.adv_gp_encoder(gripper_pose)], dim=1
        )
        adv_src = torch.cat([self.value_rgb_encoder(rgb_obs), self.value_low_dim_encoder(low_dim_obs), self.value_gp_encoder(gripper_pose)],-1)
        adv_temp = self.adv_Transformer(adv_h)
        adv_temp = adv_temp.reshape(*adv_temp.shape[:-2], -1) + adv_src
        adv_temp = adv_src
        try:
            adv_x = torch.cat([adv_temp, prev_action, level_id], -1)
        except:
            adv_temp = adv_temp.unsqueeze(0)
            adv_x = torch.cat([adv_temp, prev_action, level_id], -1)
        # adv_x = torch.cat([adv_temp, prev_action, level_id], -1)
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

    def get_action(self, rgb_obs: torch.Tensor, low_dim_obs: torch.Tensor, gripper_pose: torch.Tensor):
        metrics = dict()
        low = self.initial_low.repeat(rgb_obs.shape[0], 1).detach()
        high = self.initial_high.repeat(rgb_obs.shape[0], 1).detach()

        for level in range(self.levels):
            q_logits = self.network(level, rgb_obs, low_dim_obs, gripper_pose, (low + high) / 2)
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
        gripper_pose: torch.Tensor,
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
            q_logits = self.network(level, rgb_obs, low_dim_obs, gripper_pose, (low + high) / 2)
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
        next_gripper_pose: torch.Tensor,
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
            next_rgb_obs, next_low_dim_obs, next_gripper_pose, next_continuous_action
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
        self.actor_grad_clip = 100.0
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
        # policy_config = get_config("/home/lenovo/project/CQN_Improve/cfgs/policy_config.yaml")
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

    def act(self, rgb_obs, low_dim_obs, gripper_pose, step, eval_mode):
        rgb_obs = torch.as_tensor(rgb_obs, device=self.device).unsqueeze(0)
        low_dim_obs = torch.as_tensor(low_dim_obs, device=self.device).unsqueeze(0)
        gripper_pose = torch.as_tensor(gripper_pose, device=self.device, dtype=torch.float).unsqueeze(0)
        rgb_obs = self.encoder(rgb_obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        action, _ = self.critic_target.get_action(
            rgb_obs, low_dim_obs, gripper_pose
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
    
    # def _compute_actor_loss(
    #     self,
    #     rgb_obs,
    #     low_dim_obs,
    #     action,
    #     critic,
    #     next_rgb_obs,
    #     next_low_dim_obs,
    #     reward,
    #     discount,
    #     critic_target : C2FCritic
    # ):
    #     q_probs, q_probs_a, log_q_probs, log_q_probs_a = critic(
    #         rgb_obs, low_dim_obs, action
    #     )
    #     next_action, mets = self.critic.get_action(next_rgb_obs, next_low_dim_obs)
    #     with torch.no_grad():
    #         target_q_probs_a = critic_target.compute_target_q_dist(
    #             next_rgb_obs, next_low_dim_obs, next_action, reward, discount
    #         )
    #     q_critic_loss = -torch.sum(target_q_probs_a * log_q_probs_a, 3).mean()
    #     min_q = q_probs_a.min(-1, keepdim=True)[0]
    #     return (-min_q*q_critic_loss).mean()

    # def get_bc_loss(self, predicted_action, buffer_action, demos):
    #     metrics = dict()
    #     bc_loss = 0
    #     if demos is not None:
    #         # Only apply loss to demo items
    #         demos = demos.float().unsqueeze(1)
    #         bs = demos.shape[0]
    #         if demos.sum() > 0:
    #             bc_loss = (
    #                 F.mse_loss(
    #                     predicted_action.view(bs, -1),
    #                     buffer_action.view(bs, -1),
    #                     reduction="none",
    #                 )
    #                 * demos
    #             )
    #             bc_loss = self.bc_lambda * bc_loss.sum() / demos.sum()
    #             if self.use_logger:
    #                 metrics["actor_bc_loss"] = bc_loss.item()
    #         if self.use_logger:
    #             metrics["ratio_of_demos"] = demos.mean().item()
    #     return metrics, bc_loss

    # def update_actor(
    #     self,
    #     rgb_obs,
    #     low_dim_obs,
    #     action,
    #     reward,
    #     discount,
    #     next_rgb_obs,
    #     next_low_dim_obs,
    #     demos,
    #     steps,
    # ):
    #     metrics = dict()
    #     dist = self.actor(rgb_obs, low_dim_obs)
    #     predicted_action = dist.sample()
    #     log_prob = dist.log_prob(action).sum(-1, keepdim=True)
    #     base_actor_loss = self._compute_actor_loss(rgb_obs, low_dim_obs, predicted_action, self.critic, next_rgb_obs, next_low_dim_obs,reward, discount, self.critic_target)
    #     bc_metrics, bc_loss = self.get_bc_loss(dist.mean, action, demos)
    #     metrics.update(bc_metrics)
    #     actor_loss = base_actor_loss+bc_loss
    #     if self.actor_grad_clip:
    #         nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
    #     self.actor_opt.zero_grad(set_to_none=True)
    #     actor_loss.backward(retain_graph=True)
    #     self.actor_opt.step()

    #     if self.use_logger:
    #         metrics["mean_act"] = dist.mean.mean().item()
    #         metrics["actor_loss"] = actor_loss.item()
    #         metrics["actor_logprob"] = log_prob.mean().item()
    #         metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()
    #     return metrics



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
        with torch.no_grad():
            next_action, mets = self.critic.get_action(next_rgb_obs, next_low_dim_obs, gripper_pose)
            target_q_probs_a = self.critic_target.compute_target_q_dist(
                next_rgb_obs, next_low_dim_obs, next_action, next_gripper_pose, reward, discount
            )
            if self.use_logger:
                metrics.update(**mets)

        # Cross entropy loss for C51
        q_probs, q_probs_a, log_q_probs, log_q_probs_a = self.critic(
            rgb_obs, low_dim_obs, gripper_pose, action
        )
        # next_q_probs, next_q_probs_a, next_log_q_probs, next_log_q_probs_a = self.critic(
        #     rgb_obs, low_dim_obs, action
        # )
        q_critic_loss = -torch.sum(target_q_probs_a * log_q_probs_a, 3).mean()
        critic_loss = self.critic_lambda * q_critic_loss

        if self.use_logger:
            metrics["q_critic_loss"] = q_critic_loss.item()

        # if self.sc_lambda > 0.0:
        #     next_action

        if self.bc_lambda > 0.0:
            qs = None
            demos = demos.float().squeeze(1)  # [B,]
            # print(demos)
            if self.use_logger:
                metrics["ratio_of_demos"] = demos.mean().item()

            if torch.sum(demos) > 0:
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
                critic_loss = critic_loss + self.bc_lambda * bc_fosd_loss
                if self.use_logger:
                    metrics["bc_fosd_loss"] = bc_fosd_loss.item()

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
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward(retain_graph=True)
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

        # if torch.sum(demos) > 0:
        #     new_demos = demos.squeeze(1)
        #     image = policy_rgb_obs.float()
        #     image = image[new_demos.bool()]
            
        #     seq_action = action[new_demos.bool()]
        #     seq_action = seq_action.unsqueeze(1)
        #     is_pad = seq_action.sum(axis=-1) == 0
        #     policy_q_pos = policy_q_pos[new_demos.bool()]
        #     loss_dict = self.actor(policy_q_pos, image, seq_action, is_pad)
        #     self.actor_opt.zero_grad(set_to_none=True)
        #     loss_dict["loss"].backward()
        #     metrics["actor_loss"] = loss_dict["loss"].item()
        #     self.actor_opt.step()
        
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
