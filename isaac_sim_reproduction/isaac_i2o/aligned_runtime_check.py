import numpy as np,torch,copy
from types import SimpleNamespace
from aligned.contract import ActionCodec,stack_episode
from aligned.gpu_replay import GPUReplay
from aligned.paper_update import update
from vendor.i2o_rl import C2FCritic

torch.set_num_threads(2)
c=ActionCodec(np.r_[np.full(7,-.1),0],np.r_[np.full(7,.1),1])
d={'rgb':np.stack([np.full((4,3,84,84),i,np.uint8) for i in range(4)]),
 'state':np.stack([np.full(15,i,np.float32) for i in range(4)]),
 'raw_actions':np.zeros((3,8),np.float32),'rewards':np.array([0,0,1],np.float32),'terminated':np.array([False,False,True])}
r=GPUReplay(c,'cpu',capacity=32);r.add(d,True,'fixture');b=r.sample(16);cpu=stack_episode(d)
for i in range(16):
 t=int(b[1][i,-1]);assert torch.equal(b[0][i],torch.from_numpy(cpu[t][0]));assert torch.equal(b[9][i],torch.from_numpy(cpu[t+1][0]))
class Encoder(torch.nn.Module):
 def __init__(self):super().__init__();self.linear=torch.nn.Linear(96,16)
 def forward(self,x):return self.linear(x.float().mean((-1,-2)).flatten(1)/255.)
e=Encoder();q=C2FCritic((8,),16,120,8,16,3,5,51,-2.,2.)
a=SimpleNamespace(encoder=e,critic=q,critic_target=copy.deepcopy(q),aug=torch.nn.Identity(),
 encoder_opt=torch.optim.AdamW(e.parameters(),lr=5e-5),critic_opt=torch.optim.AdamW(q.parameters(),lr=5e-5))
for i in range(3):m=update(a,r.sample(16),rl=i>0)
assert m['encoder_grad_norm']>0 and m['soft_action_requires_grad'] and np.isfinite(m['loss'])
print('ALIGNED_FRAME_STACK_AND_PAPER_LOSS_GRADIENT_PASS',m)
