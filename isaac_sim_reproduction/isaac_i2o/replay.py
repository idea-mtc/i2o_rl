"""Separate successful demonstration/online replay with the original I2O batch API."""
import numpy as np


class Replay:
    def __init__(self, capacity=10000, seed=1):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.demos, self.online = [], []
        self.cursor = 0

    def add(self, obs, action, reward, discount, next_obs, demo=False):
        item = (obs, np.asarray(action).copy(), float(reward), float(discount), next_obs, demo)
        if demo:
            self.demos.append(item)
        elif len(self.online) < self.capacity:
            self.online.append(item)
        else:
            self.online[self.cursor] = item
            self.cursor = (self.cursor + 1) % self.capacity

    def batches(self, batch_size):
        while True:
            assert self.demos, "collect successful demonstrations first"
            n_demo = batch_size // 2 if self.online else batch_size
            rows = [self.demos[i] for i in self.rng.integers(len(self.demos), size=n_demo)]
            rows += [self.online[i] for i in self.rng.integers(len(self.online), size=batch_size-n_demo)]
            rgb = np.stack([x[0][0] for x in rows])
            low = np.stack([x[0][1] for x in rows])
            next_rgb = np.stack([x[4][0] for x in rows])
            next_low = np.stack([x[4][1] for x in rows])
            # These legacy auxiliary fields are unpacked by I2O.update(), but
            # its active loss does not use them. Preserve the upstream API.
            unused = np.zeros((batch_size, 7), np.float32)
            yield (rgb, low, unused, unused, unused, unused,
                   np.stack([x[1] for x in rows]),
                   np.array([[x[2]] for x in rows], np.float32),
                   np.array([[x[3]] for x in rows], np.float32),
                   next_rgb, next_low,
                   np.array([[x[5]] for x in rows], np.float32),
                   unused, unused, unused, unused)

    def save(self, path):
        rows = self.demos
        np.savez_compressed(path,
            rgb=np.stack([x[0][0] for x in rows]),
            state=np.stack([x[0][1] for x in rows]),
            actions=np.stack([x[1] for x in rows]),
            rewards=np.array([x[2] for x in rows]),
            discounts=np.array([x[3] for x in rows]),
            next_rgb=np.stack([x[4][0] for x in rows]),
            next_state=np.stack([x[4][1] for x in rows]))
