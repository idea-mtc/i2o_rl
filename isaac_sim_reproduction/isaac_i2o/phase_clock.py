"""Observable control-clock conditioning for the fixed-duration lid expert.

No object pose, IK target or expert action enters these six policy features.
This explicitly changes the original paper-style observation contract.
"""
import numpy as np

DURATIONS = np.array([35, 40, 25, 45, 35])
BOUNDARIES = np.r_[0, np.cumsum(DURATIONS)]


def clock_features(step):
    phase = min(int(np.searchsorted(BOUNDARIES[1:], step, side='right')), 4)
    local = np.clip((step-BOUNDARIES[phase])/DURATIONS[phase], 0., 1.)
    features = np.zeros(6, np.float32)
    features[phase] = 1.
    features[5] = local
    return features


def condition(obs, step):
    return obs[0], np.r_[obs[1], clock_features(step)].astype(np.float32)


class ClockConditionedEnv:
    def __init__(self, base):
        self.base = base
        self.state_shape = (base.state_shape[0]+6,)
        self.config = dict(base.config, clock_conditioning={'durations':DURATIONS.tolist(),
                           'features':'five phase indicators plus local time; derived from control step'})

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def rng(self):
        return self.base.rng

    @rng.setter
    def rng(self, value):
        self.base.rng = value

    def reset(self, *args, **kwargs):
        return condition(self.base.reset(*args, **kwargs), 0)

    def step(self, action):
        obs, r, terminated, truncated, info = self.base.step(action)
        return condition(obs, self.base.t), r, terminated, truncated, info

    def observe(self):
        return condition(self.base.observe(), self.base.t)
