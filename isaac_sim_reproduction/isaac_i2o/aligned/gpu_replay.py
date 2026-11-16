"""Source-style episode sampling with raw frames cached on the selected GPU.

No repeated stacked-image host-to-device copy. Sampling semantics stay episode
uniform; ordinary and successful pools are each half of the effective batch.
"""
import numpy as np
import torch

class GPUReplay:
    def __init__(self,codec,device,capacity=72000,seed=1):
        self.codec=codec;self.device=device;self.capacity=capacity
        self.rng=np.random.default_rng(seed);self.episodes=[];self.success_ids=[];self.used=0
        self.rgb=torch.empty((capacity,4,3,84,84),dtype=torch.uint8,device=device)
        self.state=torch.empty((capacity,15),device=device)
        self.actions=torch.empty((capacity,8),device=device)
        self.rewards=torch.empty((capacity,1),device=device)
        self.discounts=torch.empty((capacity,1),device=device)
        self.demo=torch.empty((capacity,1),device=device)
    def add(self,data,successful,path):
        n=len(data['raw_actions']);start=self.used;end=start+n+1
        assert end<=self.capacity,'raw-frame cache exhausted; refusing to evict differently from source'
        self.rgb[start:end].copy_(torch.as_tensor(data['rgb'],device=self.device))
        self.state[start:end].copy_(torch.as_tensor(data['state'],device=self.device))
        self.actions[start:start+n].copy_(torch.as_tensor(self.codec.encode(data['raw_actions']),device=self.device))
        self.rewards[start:start+n,0].copy_(torch.as_tensor(data['rewards'],device=self.device))
        discounts=np.where(data['terminated'],0.,.99).astype(np.float32)
        self.discounts[start:start+n,0].copy_(torch.as_tensor(discounts,device=self.device))
        self.demo[start:start+n]=float(successful)
        self.episodes.append({'offset':start,'length':n,'successful':bool(successful),'file':str(path)})
        if successful:self.success_ids.append(len(self.episodes)-1)
        self.used=end
    def _sample_ids(self,pool,count):
        chosen=np.asarray(pool)[self.rng.integers(len(pool),size=count)]
        lengths=np.array([self.episodes[i]['length'] for i in chosen])
        times=self.rng.integers(lengths)
        offsets=np.array([self.episodes[i]['offset'] for i in chosen])
        return offsets+times,offsets
    def sample(self,count=256,demonstrations_only=False):
        if demonstrations_only:ids,starts=self._sample_ids(self.success_ids,count)
        else:
            a,b=self._sample_ids(range(len(self.episodes)),count//2)
            c,d=self._sample_ids(self.success_ids,count-count//2)
            ids=np.r_[a,c];starts=np.r_[b,d]
        current=np.maximum(starts[:,None],ids[:,None]-np.arange(7,-1,-1))
        nxt=np.maximum(starts[:,None],ids[:,None]+1-np.arange(7,-1,-1))
        current=torch.as_tensor(current,device=self.device);nxt=torch.as_tensor(nxt,device=self.device)
        ids=torch.as_tensor(ids,device=self.device)
        def frames(index):
            return self.rgb[index].permute(0,2,1,3,4,5).reshape(count,4,24,84,84)
        unused=torch.zeros((count,7),device=self.device)
        return (frames(current),self.state[current].reshape(count,120),unused,unused,unused,unused,
          self.actions[ids],self.rewards[ids],self.discounts[ids],frames(nxt),self.state[nxt].reshape(count,120),
          self.demo[ids],unused,unused,unused,unused)
    def batches(self,count=256):
        while True:yield self.sample(count)
    def files(self):return [{k:e[k] for k in ['file','successful']} for e in self.episodes]
