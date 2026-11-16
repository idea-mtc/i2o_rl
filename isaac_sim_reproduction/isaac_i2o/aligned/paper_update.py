"""Paper-text loss reconstruction on the original shared coarse-to-fine Q body.

Explicitly not the unmodified repository's C51/FOSD/L1 objective. Uses beta1
continuous soft actions, action MSE + factorized margin, and scalar final-level
Bellman square (RL scale .1). No phase conditioning or task-specific output heads.
"""
import torch
import torch.nn.functional as F
from vendor.cqn_utils import encode_action


def update(agent,batch,rl=False):
    rgb,low,_,_,_,_,action,reward,discount,next_rgb,next_low,demo,*_=batch
    rgb=rgb.float();next_rgb=next_rgb.float()
    rgb=torch.stack([agent.aug(rgb[:,v]) for v in range(rgb.shape[1])],1)
    next_rgb=torch.stack([agent.aug(next_rgb[:,v]) for v in range(next_rgb.shape[1])],1)
    features=agent.encoder(rgb)
    prediction=agent.critic.get_soft_action(features,low,beta=1.,endpoint_rescale=False)
    probs,probs_a,*_=agent.critic(features,low,action)
    q=(probs*agent.critic.support).sum(-1);qa=(probs_a*agent.critic.support).sum(-1)
    indices=encode_action(action,agent.critic.initial_low,agent.critic.initial_high,3,5).long()
    alternatives=~F.one_hot(indices,num_classes=5).bool()
    margin=(q-qa.unsqueeze(-1)+.01).masked_fill(~alternatives,-float('inf')).max(-1).values.clamp(min=0).mean((-1,-2))
    per_sample=(prediction-action).square().mean(-1)
    mask=demo.reshape(-1).float()
    mse=(per_sample*mask).sum()/mask.sum().clamp(min=1)
    margin_loss=(margin*mask).sum()/mask.sum().clamp(min=1)
    il_loss=mse+margin_loss
    td_loss=torch.zeros((),device=rgb.device)
    if rl:
        with torch.no_grad():
            nxt=agent.encoder(next_rgb)
            greedy,_=agent.critic_target.get_action(nxt,next_low)
            _,target_probs,*_=agent.critic_target(nxt,next_low,greedy)
            target_q=(target_probs*agent.critic_target.support).sum(-1)[:,-1].mean(-1)
            target=reward.reshape(-1)+discount.reshape(-1)*target_q
        current_q=qa[:,-1].mean(-1)
        td_loss=F.mse_loss(current_q,target)
    loss=il_loss+.1*td_loss
    assert torch.isfinite(loss)
    agent.encoder_opt.zero_grad(set_to_none=True);agent.critic_opt.zero_grad(set_to_none=True)
    loss.backward()
    grad=sum(float(p.grad.detach().square().sum()) for p in agent.encoder.parameters() if p.grad is not None)**.5
    agent.encoder_opt.step();agent.critic_opt.step()
    with torch.no_grad():
        for source,target in zip(agent.critic.parameters(),agent.critic_target.parameters()):
            target.lerp_(source,.02)
    return {'loss':float(loss.detach()),'action_mse':float(mse.detach()),'margin':float(margin_loss.detach()),
            'td_square':float(td_loss.detach()),'encoder_grad_norm':grad,'soft_action_requires_grad':prediction.requires_grad}
