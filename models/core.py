from abc import ABC, abstractmethod
from typing import Iterator, TypeAlias, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from replay_buffer import ReplayBuffer

from models.utils import (
    extract_many_from_spec,
    extract_from_spec,
    extract_many_from_batch,
)

BatchedActionSequence: TypeAlias = np.ndarray[
    tuple[int, int, int], np.dtype[np.float32]
]
Metrics: TypeAlias = dict[str, np.ndarray]

class IntrinsicRewardModule(ABC):
    """Base class of intrinsic reward module."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        device: torch.device,
        beta: float = 0.05,
        kappa: float = 0.000025,
    ) -> None:
        """Init.

        Args:
            observation_space: The observation space of environment.
            action_space: The action space of environment.
            device: Device to run the model.
            beta: The initial weighting coefficient of the intrinsic rewards.
            kappa: The decay rate.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.beta = beta
        self.kappa = kappa

        self.rgb_spaces = extract_many_from_spec(
            observation_space, r"rgb.*", missing_ok=True
        )
        self.low_dim_space = extract_from_spec(
            observation_space, "low_dim_state", missing_ok=True
        )
        self.use_pixels = len(self.rgb_spaces) > 0

    @abstractmethod
    def compute_irs(
        self, batch: dict[str, torch.Tensor], step: int = 0
    ) -> torch.Tensor:
        """Compute the intrinsic rewards for current samples.

        Args:
            batch: Batch of data.
            step: The global training step.

        Returns:
            The intrinsic rewards.
        """

    @abstractmethod
    def update(
        self,
        batch: dict[str, torch.Tensor],
    ) -> None:
        """Update the intrinsic reward module if necessary.

        Args:
            batch: Batch of data.
        """

    def _extract_obs(self, batch: dict[str, torch.Tensor], name_or_regex: str):
        if self.use_pixels:
            # dict of {"cam_name": (B, T, 3, H, W)}
            obs = extract_many_from_batch(batch, name_or_regex)
            # Fold views into time axis
            obs = torch.cat(list(obs.values()), 1)
            # Fold time into channel axis
            obs = obs.view(obs.shape[0], -1, *obs.shape[3:])
        else:
            # Get last timestep
            obs = batch[name_or_regex][:, -1]
        return obs

class BaseModule(nn.Module, ABC):
    @property
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError()

    def calculate_loss(self, *args, **kwargs) -> Optional[torch.Tensor]:
        return None
    

class FullyConnectedModule(BaseModule, ABC):
    def __init__(self, input_shapes: dict[str, tuple], output_shape: int | Tuple):
        super().__init__()
        self.input_shapes = input_shapes
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        # Hydra cannot take tuple inputs, this has to be done
        self._output_shape = tuple(output_shape)
        self.eval_env_running = False

    @property
    def output_shape(self):
        return self._output_shape

    def reset(self, env_index: int):
        pass

class Method(nn.Module, ABC):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        device: torch.device,
        num_train_envs: int,
        replay_alpha: float,
        replay_beta: float,
        frame_stack_on_channel: bool,
        intrinsic_reward_module: Optional[IntrinsicRewardModule] = None,
        is_rl: bool = False,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.num_train_envs = num_train_envs
        self.replay_alpha = replay_alpha
        self.replay_beta = replay_beta
        self.frame_stack_on_channel = frame_stack_on_channel
        self.num_eval_envs = 1  # Fixed to 1 for now
        self.intrinsic_reward_module = intrinsic_reward_module
        self._eval_env_running = False
        self.logging = False
        self.is_rl = is_rl

    @property
    def random_explore_action(self) -> torch.Tensor:
        # All actions live in -1 to 1, regardless of environment.
        min_action = -1
        max_action = 1
        return (min_action - max_action) * torch.rand(
            size=(self.num_train_envs,) + self.action_space.shape
        ) + max_action

    @abstractmethod
    def act(
        self, observations: dict[str, torch.Tensor], step: int, eval_mode: bool
    ) -> BatchedActionSequence:
        pass

    @abstractmethod
    def update(
        self,
        replay_iter: Iterator[dict[str, torch.Tensor]],
        step: int,
        replay_buffer: ReplayBuffer = None,
    ) -> Metrics:
        pass

    @abstractmethod
    def reset(self, step: int, agents_to_reset: list[int]):
        pass

    @property
    def eval_env_running(self):
        return self._eval_env_running

    def set_eval_env_running(self, value: bool):
        self._eval_env_running = value
