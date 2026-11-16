"""Normalized absolute TCP XYZ + gripper, with a fixed downward tool orientation."""
import numpy as np
import os
from pathlib import Path
from absolute_joints import absolute_episode_actions,decode_joints

LOW=np.array([.38,-.08,.30]);HIGH=np.array([.54,.08,.58])


def encode_position(xyz):return (2*(xyz-LOW)/(HIGH-LOW)-1).astype(np.float32)


def decode_position(action):return LOW+(np.clip(action,-1,1)+1)*.5*(HIGH-LOW)


def cartesian_episode_actions(data,env,phase_goals=True):
    joint_actions=absolute_episode_actions(data,env,phase_goals)
    cache={};converted=[]
    for a in joint_actions:
        key=tuple(a[:7])
        if key not in cache:
            q=decode_joints(a[:7],env.lower,env.upper)
            cache[key]=encode_position(env.fk(q)[:3,3])
        converted.append(np.r_[cache[key],a[7]])
    actions=np.asarray(converted,np.float32)
    assert np.abs(actions).max()<=1.00001,'expert TCP outside configured policy workspace'
    return actions


class CartesianGoalEnv:
    action_shape=(4,)
    def __init__(self,base):
        self.base=base
        self._last_xyz=None;self._last_joints=None
        self._ik_solver=base.ik
        urdf=Path(os.environ.get('PINGANDOG_ROOT','/home/ubuntu/pingandog'))/'assets/robot/panda/panda.urdf'
        if urdf.exists() and hasattr(base,'robot'):
            from fast_panda_ik import FastPandaIK
            self._ik_solver=FastPandaIK(urdf,base.lower,base.upper,fallback=base.ik)
            np.testing.assert_allclose(self._ik_solver.fk_jacobian(base.home[base.arm])[0],
                                       base.fk(base.home[base.arm]),atol=1e-10)
        self.config=dict(base.config,action_contract='absolute TCP XYZ plus gripper; downward tool orientation',
                         position_workspace_low=LOW.tolist(),position_workspace_high=HIGH.tolist(),
                         ik_solver=type(self._ik_solver).__name__,ik_cache_tolerance_m=.00025)
    def __getattr__(self,name):return getattr(self.base,name)
    @property
    def rng(self):return self.base.rng
    @rng.setter
    def rng(self,value):self.base.rng=value
    def reset(self,*args,**kwargs):
        self._last_xyz=None;self._last_joints=None
        return self.base.reset(*args,**kwargs)
    def step(self,action):
        action=np.asarray(action,np.float32);assert action.shape==(4,)
        xyz=decode_position(action[:3])
        if self._last_xyz is None or np.linalg.norm(xyz-self._last_xyz)>.00025:
            seed=self._last_joints if self._last_joints is not None else self.target[self.arm]
            self._last_joints=self._ik_solver(xyz,seed)
            self._last_xyz=xyz.copy()
        joints=self._last_joints
        delta=np.r_[(joints-self.target[self.arm])/self.delta,action[3]]
        delta=np.clip(delta,-1,1).astype(np.float32)
        obs,r,term,trunc,info=self.base.step(delta)
        info.update(policy_cartesian_action=action.tolist(),tcp_goal=xyz.tolist(),ik_cached_goal=self._last_xyz.tolist(),executed_delta_action=delta.tolist())
        return obs,r,term,trunc,info
