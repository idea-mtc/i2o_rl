"""Action normalization copied algebraically from the original RLBench wrapper."""
import numpy as np

class ActionCodec:
    def __init__(self,minimum,maximum):
        self.minimum=np.asarray(minimum,np.float32);self.maximum=np.asarray(maximum,np.float32)
        self.low=self.minimum-.2*np.abs(self.minimum)
        self.high=self.maximum+.2*np.abs(self.maximum)
        assert self.low.shape==self.high.shape==(8,) and np.all(self.high>self.low)
    @classmethod
    def fit(cls,episodes):
        # Original extractor includes the reset timestep's zero action in statistics.
        actions=np.concatenate([np.zeros((1,8),np.float32)]+[x['raw_actions'] for x in episodes])
        low=actions.min(0);high=actions.max(0);low[-1]=0;high[-1]=1
        return cls(low,high)
    def encode(self,raw):return ((np.asarray(raw)-self.low)/(self.high-self.low)*2-1).astype(np.float32)
    def decode(self,normalized):return ((np.asarray(normalized)+1)*.5*(self.high-self.low)+self.low).astype(np.float32)
    def json(self):return {'min':self.minimum.tolist(),'max':self.maximum.tolist(),'expanded_min':self.low.tolist(),'expanded_max':self.high.tolist()}


def stack_episode(data,frames=8):
    out=[]
    for t in range(len(data['rgb'])):
        ids=np.maximum(0,np.arange(t-frames+1,t+1))
        out.append((np.concatenate([data['rgb'][i] for i in ids],axis=1),
                    np.concatenate([data['state'][i] for i in ids]).astype(np.float32)))
    return out
