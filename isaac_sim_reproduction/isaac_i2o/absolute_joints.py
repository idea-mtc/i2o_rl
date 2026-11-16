"""Explicit normalized joint-position action contract, not an expert controller."""
import numpy as np


def encode_joints(q, lower, upper):
    return (2*(q-lower)/(upper-lower)-1).astype(np.float32)


def decode_joints(a, lower, upper):
    return lower+(np.clip(a,-1,1)+1)*.5*(upper-lower)


def absolute_episode_actions(data, env, phase_goals=False):
    """Recover commanded positions from the original incremental action log."""
    target=env.home[env.arm].copy();actions=[]
    for a in data['actions']:
        target=np.clip(target+env.delta*a[:7],env.lower,env.upper)
        actions.append(np.r_[encode_joints(target,env.lower,env.upper),a[7]])
    actions=np.asarray(actions,np.float32)
    if phase_goals:
        for phase in np.unique(data["phase"]):
            ids=np.flatnonzero(data["phase"]==phase)
            actions[ids]=actions[ids[-1]].copy()
    return actions


class AbsoluteJointEnv:
    def __init__(self, base):
        self.base=base
        self.config=dict(base.config,action_contract='normalized absolute joint1..7 targets plus absolute gripper')

    def __getattr__(self,name):
        return getattr(self.base,name)

    @property
    def rng(self):return self.base.rng

    @rng.setter
    def rng(self,value):self.base.rng=value

    def expert_actions(self, *args, **kwargs):
        for delta in self.base.expert_actions(*args, **kwargs):
            goal=np.clip(self.target[self.arm]+self.delta*delta[:7],self.lower,self.upper)
            yield np.r_[encode_joints(goal,self.lower,self.upper),delta[7]].astype(np.float32)

    def step(self, action):
        action=np.asarray(action,np.float32)
        goal=decode_joints(action[:7],self.lower,self.upper)
        delta_action=np.r_[(goal-self.target[self.arm])/self.delta,action[7]]
        delta_action=np.clip(delta_action,-1,1).astype(np.float32)
        obs,r,term,trunc,info=self.base.step(delta_action)
        info.update(policy_absolute_action=action.tolist(),executed_delta_action=delta_action.tolist())
        return obs,r,term,trunc,info
