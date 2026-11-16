"""Run under the same torch runtime as Isaac, on CPU; no scene or GPU use."""
import torch
from vendor.i2o_rl import C2FCriticNetwork

torch.set_num_threads(2)
n=C2FCriticNetwork(16,126,(8,),8,16,3,5,51,phase_heads=5)
with torch.no_grad():
    bias=n.value_head.bias.view(5,8,1,51)
    for phase in range(5):bias[phase,:,:,0]=phase+1
low=torch.zeros(5,126)
for phase in range(5):low[phase,-6+phase]=1
out=n(0,torch.zeros(5,16),low,torch.zeros(5,8))
assert out.shape==(5,8,5,51)
for phase in range(5):assert torch.all(out[phase,:,:,0]==phase+1)
n.zero_grad();out[2].sum().backward()
grad=n.value_head.bias.grad.view(5,8,1,51)
assert grad[2].abs().sum()>0
assert sum(grad[i].abs().sum() for i in [0,1,3,4])==0
print('PHASE_HEAD_SELECTION_AND_GRADIENT_ISOLATION_PASS')

# Exercise the actual critic update: relabeled demo transitions must not affect TD.
import copy
from vendor.i2o_rl import I2OAgent,C2FCritic
agent=object.__new__(I2OAgent)
agent.critic=C2FCritic((8,),16,126,8,16,3,5,51,-2.,2.,phase_heads=5)
with torch.no_grad():
    agent.critic.network.adv_head.weight.normal_(0,.02)
    agent.critic.network.adv_head.bias.normal_(0,.1)
agent.critic_target=copy.deepcopy(agent.critic)
agent.encoder=torch.nn.Linear(16,16)
agent.encoder_opt=torch.optim.SGD(agent.encoder.parameters(),lr=0)
agent.critic_opt=torch.optim.SGD(agent.critic.parameters(),lr=0)
agent.critic_lambda=.1;agent.bc_lambda=0.;agent.use_logger=True;agent.demo_bc_only=True
raw=torch.randn(2,16);low=torch.zeros(2,126);low[0,-6]=1;low[1,-5]=1
next_raw=torch.randn(2,16);action=torch.zeros(2,8);discount=torch.full((2,1),.99)
demos=torch.tensor([[1.],[0.]])
def update(rewards):
    metrics=agent.update_critic(agent.encoder(raw),low,None,None,None,None,action,
        torch.tensor(rewards).reshape(2,1),discount,next_raw,low,demos,0)
    grads=torch.cat([p.grad.flatten() for p in agent.critic.parameters() if p.grad is not None])
    return metrics['q_critic_loss'],grads.clone()
l0,g0=update([0.,0.]);l1,g1=update([1.,0.]);l2,g2=update([0.,1.])
assert abs(l0-l1)<1e-7 and torch.allclose(g0,g1,atol=1e-7)
assert not torch.allclose(g0,g2,atol=1e-7)
print('DEMO_TARGET_EXCLUSION_FROM_ACTUAL_TD_UPDATE_PASS')
