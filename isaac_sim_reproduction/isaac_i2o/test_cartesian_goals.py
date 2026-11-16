import unittest
import numpy as np
from cartesian_goals import CartesianGoalEnv,cartesian_episode_actions,encode_position,decode_position,LOW,HIGH


class Base:
    config={};rng=None;arm=np.arange(7);delta=.04
    lower=np.full(7,-2.);upper=np.full(7,2.);home=np.zeros(9)
    def __init__(self):self.target=self.home.copy()
    def ik(self,xyz,seed):return np.r_[(xyz-np.array([.46,0,.44]))/.1,np.zeros(4)]
    def fk(self,q):
        m=np.eye(4);m[:3,3]=np.array([.46,0,.44])+q[:3]*.1;return m
    def step(self,a):
        self.target[:7]+=self.delta*a[:7]
        return None,0,False,False,{}


class CartesianTests(unittest.TestCase):
    def test_workspace_roundtrip_and_bounds(self):
        for p in [LOW,HIGH,(LOW+HIGH)/2]:
            np.testing.assert_allclose(decode_position(encode_position(p)),p,atol=1e-7)
        np.testing.assert_allclose(decode_position(np.full(3,100)),HIGH)

    def test_goal_tracking_does_not_integrate_fixed_bias(self):
        e=CartesianGoalEnv(Base());goal=np.array([.47,.01,.43])
        a=np.r_[encode_position(goal),-1.]
        for _ in range(100):e.step(a)
        np.testing.assert_allclose(e.fk(e.target[:7])[:3,3],goal,atol=1e-7)

    def test_ik_cache_bounds_and_reuse(self):
        class Counted(Base):
            calls=0
            def ik(self,xyz,seed):
                self.calls+=1
                return super().ik(xyz,seed)
        b=Counted();e=CartesianGoalEnv(b);goal=np.array([.47,.01,.43])
        for _ in range(20):e.step(np.r_[encode_position(goal),-1.])
        self.assertEqual(b.calls,1)
        goal[0]+=.0001;e.step(np.r_[encode_position(goal),-1.]);self.assertEqual(b.calls,1)
        goal[0]+=.0003;e.step(np.r_[encode_position(goal),-1.]);self.assertEqual(b.calls,2)

    def test_phase_goal_labels_use_recorded_phase_endpoint(self):
        actions=np.zeros((6,8),np.float32);actions[:,:3]=.1;actions[:,7]=.75
        d={'actions':actions,'phase':np.array(['approach']*3+['descend']*3)}
        converted=cartesian_episode_actions(d,Base(),True)
        self.assertEqual(converted.shape,(6,4))
        np.testing.assert_array_equal(converted[0],converted[2])
        np.testing.assert_array_equal(converted[3],converted[5])
        self.assertFalse(np.allclose(converted[0],converted[3]))
        xyz=decode_position(converted[2,:3])
        np.testing.assert_allclose(xyz,np.array([.46,0,.44])+.0012,atol=1e-7)


if __name__=='__main__':unittest.main()
