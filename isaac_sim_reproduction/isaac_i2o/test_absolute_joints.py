import unittest
import numpy as np
from absolute_joints import AbsoluteJointEnv,absolute_episode_actions,encode_joints,decode_joints


class Base:
    config={};rng=None;delta=.04;arm=np.arange(7);lower=np.full(7,-2.);upper=np.full(7,2.)
    home=np.zeros(9)
    def __init__(self):self.target=self.home.copy()
    def step(self,a):
        self.target[:7]=np.clip(self.target[:7]+self.delta*a[:7],self.lower,self.upper)
        return None,0,False,False,{}
    def expert_actions(self):
        for _ in range(3):yield np.r_[np.ones(7)*.5,.75]


class AbsoluteTests(unittest.TestCase):
    def test_position_encoding_roundtrip(self):
        q=np.linspace(-2,2,7)
        np.testing.assert_allclose(decode_joints(encode_joints(q,Base.lower,Base.upper),Base.lower,Base.upper),q,atol=1e-6)

    def test_repeated_absolute_action_does_not_accumulate_drift(self):
        b=Base();e=AbsoluteJointEnv(b);a=np.r_[encode_joints(np.full(7,.12),b.lower,b.upper),-.9]
        for _ in range(100):e.step(a)
        np.testing.assert_allclose(b.target[:7],.12,atol=1e-6)

    def test_transformed_demonstrations_replay_identically(self):
        b=Base();rng=np.random.default_rng(4);raw=rng.uniform(-1,1,(100,8)).astype(np.float32)
        converted=absolute_episode_actions({'actions':raw},b)
        e=AbsoluteJointEnv(Base())
        for a,target in zip(raw,converted):
            b.step(a);e.step(target)
            np.testing.assert_allclose(e.target[:7],b.target[:7],atol=2e-6)

    def test_absolute_expert_wrapper_preserves_execution(self):
        b=Base();e=AbsoluteJointEnv(b)
        for action in e.expert_actions():e.step(action)
        np.testing.assert_allclose(b.target[:7],.06,atol=1e-6)


if __name__=='__main__':unittest.main()
