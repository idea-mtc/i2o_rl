import unittest
import numpy as np
from phase_clock import clock_features,condition,ClockConditionedEnv


class PhaseClockTests(unittest.TestCase):
    def test_boundaries_match_expert_action_at_current_step(self):
        for step,phase in [(0,0),(34,0),(35,1),(74,1),(75,2),(99,2),
                           (100,3),(144,3),(145,4),(179,4),(260,4)]:
            features=clock_features(step)
            self.assertEqual(features[:5].argmax(),phase)
            self.assertEqual(features[:5].sum(),1)
            self.assertTrue(0<=features[-1]<=1)
        np.testing.assert_allclose(clock_features(35),[0,1,0,0,0,0])

    def test_condition_preserves_camera_and_robot_observations(self):
        rgb=np.zeros((4,24,84,84),np.uint8);state=np.arange(120,dtype=np.float32)
        conditioned=condition((rgb,state),75)
        self.assertIs(conditioned[0],rgb)
        np.testing.assert_array_equal(conditioned[1][:120],state)
        self.assertEqual(conditioned[1].shape,(126,))

    def test_reset_and_transition_have_correct_clock(self):
        class Base:
            state_shape=(120,);config={};rng=None;t=999
            def reset(self):
                self.t=0
                return np.zeros((4,24,84,84),np.uint8),np.zeros(120,np.float32)
            def step(self,action):
                self.t+=1
                return (np.zeros((4,24,84,84),np.uint8),np.zeros(120,np.float32)),0,False,False,{}
        env=ClockConditionedEnv(Base());obs=env.reset()
        np.testing.assert_array_equal(obs[1][-6:],clock_features(0))
        for step in range(1,37):obs,*_=env.step(np.zeros(8))
        np.testing.assert_array_equal(obs[1][-6:],clock_features(36))
        env.reset();self.assertEqual(env.base.t,0)


if __name__=='__main__':unittest.main()
