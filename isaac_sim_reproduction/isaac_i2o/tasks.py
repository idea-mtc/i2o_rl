"""Lazy task factory shared by training and demonstration collection."""
TASKS = ("cube", "take_lid_off_saucepan")


def make_env(task, frame_stack=2, horizon=None, seed=1, randomization=0.02):
    if task == "cube":
        from panda_env import PandaLiftEnv
        return PandaLiftEnv(frame_stack, horizon or 220, seed, randomization)
    if task == "take_lid_off_saucepan":
        from saucepan_env import SaucepanEnv
        # Same camera mounts/FOV as scene validation; lower native 4:3 image
        # resolution for collection/training. Policy RGB remains exactly 84x84.
        return SaucepanEnv(frame_stack, horizon or 260, seed, randomization,
                           preview_resolution=(168,126))
    raise ValueError(f"unknown task: {task}")
