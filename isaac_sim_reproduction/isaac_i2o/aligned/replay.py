"""Episode-uniform source-style prefill, mixed batches and success relabeling."""
import numpy as np
from .contract import stack_episode

class SourceReplay:
    def __init__(self,codec,seed=1):
        self.codec=codec;self.rng=np.random.default_rng(seed)
        self.all=[];self.successful=[]
    def add_episode(self,data,successful):
        obs=stack_episode(data,8);episode=[]
        for i,raw in enumerate(data['raw_actions']):
            terminal=bool(data['terminated'][i])
            episode.append((obs[i],self.codec.encode(raw),float(data['rewards'][i]),0. if terminal else .99,obs[i+1],float(successful)))
        self.all.append(episode)
        if successful:self.successful.append(episode)
    def _rows(self,episodes,count):
        out=[]
        for index in self.rng.integers(len(episodes),size=count):
            episode=episodes[index];out.append(episode[self.rng.integers(len(episode))])
        return out
    def sample(self,count=256,demonstrations_only=False):
        rows=self._rows(self.successful,count) if demonstrations_only else self._rows(self.all,count//2)+self._rows(self.successful,count-count//2)
        rgb=np.stack([x[0][0] for x in rows]);low=np.stack([x[0][1] for x in rows])
        nxt_rgb=np.stack([x[4][0] for x in rows]);nxt_low=np.stack([x[4][1] for x in rows])
        unused=np.zeros((count,7),np.float32)
        return (rgb,low,unused,unused,unused,unused,np.stack([x[1] for x in rows]),
            np.array([[x[2]] for x in rows],np.float32),np.array([[x[3]] for x in rows],np.float32),
            nxt_rgb,nxt_low,np.array([[x[5]] for x in rows],np.float32),unused,unused,unused,unused)
    def batches(self,count=256):
        while True:yield self.sample(count)
