from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

import utils
from utils import get_config


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            # actions = actions[:, :self.model.num_queries]
            # is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            # a_hat = a_hat[0][0]
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            a_hat = torch.tanh(a_hat)
            dist = utils.TruncatedNormal(a_hat, torch.zeros_like(a_hat))
            all_l1 = F.l1_loss(actions, dist.mean, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            a_hat = a_hat[0][0]
            mu = torch.tanh(a_hat)
            dist = utils.TruncatedNormal(mu, torch.zeros_like(mu))
            return dist

    def configure_optimizers(self):
        return self.optimizer


def make_policy(policy_config):
    policy = ACTPolicy(policy_config)
    return policy

class ACTactor():
    def __init__(self, policy_config):
        policy = ACTPolicy(policy_config)
        policy.cuda()
        self.policy = policy
        self.num_queries = policy['num_queries']
    
    def get_action(self, ckpt_path, qpos, curr_image) :

        self.policy.load_state_dict(torch.load(ckpt_path))
        all_actions = self.policy(qpos, curr_image)
        return all_actions


class ACTAgent:
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
        self.update_every_steps = update_every_steps
        self.use_logger = use_logger
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.bc_lambda = bc_lambda
        # self.sc_lambda = sc_lambda
        self.bc_margin = bc_margin
        self.critic_lambda = critic_lambda

        # model
        policy_config = get_config("/home/lenovo/project/CQN_Improve/cfgs/policy_config.yaml")
        self.actor = ACTPolicy(policy_config).to(device)
        # optimizers
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr = lr, weight_decay = weight_decay
        )

        # data augmentation
        self.train()

    def train(self, training=True):
        self.training = training
        # self.encoder.train(training)
        # self.critic.train(training)
        self.actor.train(training)

    
    def actor_act(self, rgb_obs, low_dim_obs, step, eval_mode=False):
        try:
            q_pos = low_dim_obs[-15:]
            qpos = q_pos[:8]
            rgb = rgb_obs[:,:,1:2,-3:,:,:]
        except:
            rgb = rgb_obs[:, -3:,:,:]
            rgb = torch.as_tensor(rgb,device=self.device)
            rgb = rgb.unsqueeze(0)
            q_pos = torch.as_tensor(low_dim_obs,device=self.device)
            q_pos = q_pos[-15:]
            q_pos = q_pos[:8]
            q_pos = q_pos.unsqueeze(0)
            
            

            # print(rgb_obs.shape)
            # print(rgb)
        image = rgb.float().detach()
        dist = self.actor(q_pos, image)
        # action = torch.clamp(action,-1.0,1.0)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.num_expl_steps:
                action.uniform(-1.0,1.0)
            # action = dist.sample()
        
        return action.cpu().numpy()

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

        if torch.sum(demos) > 0:
            new_demos = demos.squeeze(1)
            image = policy_rgb_obs.float()
            image = image[new_demos.bool()]
            
            seq_action = action[new_demos.bool()]
            seq_action = seq_action.unsqueeze(1)
            is_pad = seq_action.sum(axis=-1) == 0
            policy_q_pos = policy_q_pos[new_demos.bool()]
            loss_dict = self.actor(policy_q_pos, image, seq_action, is_pad)
            self.actor_opt.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            metrics["actor_loss"] = loss_dict["loss"].item()
            self.actor_opt.step()
        

        if self.use_logger:
            metrics["batch_reward"] = reward.mean().item()

        # update critic target

        return metrics
