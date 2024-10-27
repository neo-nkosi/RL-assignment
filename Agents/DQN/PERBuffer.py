import numpy as np
import torch as th
from typing import Optional, Union, Tuple
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Replay Buffer implementation.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param alpha: How much prioritization is used (0=none, 1=full)
    :param beta: To what degree to use importance sampling weights (0=none, 1=full)
    :param beta_increment: Increment beta by this amount after each sampling up to 1.0
    :param epsilon: Small positive constant to avoid zero probabilities
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            alpha: float = 0.6,
            beta: float = 0.4,
            beta_increment: float = 0.001,
            epsilon: float = 1e-6,
            **kwargs
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self.priorities = np.zeros((buffer_size,), dtype=np.float32)

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: list
    ) -> None:
        """Add transition to buffer with maximum priority."""
        super().add(obs, next_obs, action, reward, done, infos)

        self.priorities[self.pos - 1] = self.max_priority ** self.alpha

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """Sample a batch of transitions with priorities."""
        if self.full:
            prio_sum = np.sum(self.priorities[:self.buffer_size])
        else:
            prio_sum = np.sum(self.priorities[:self.pos])

        # Get sampling probabilities from priorities
        if self.full:
            P = self.priorities[:self.buffer_size] / prio_sum
        else:
            P = self.priorities[:self.pos] / prio_sum

        # Sample indices based on priorities
        indices = np.random.choice(
            len(P),
            size=batch_size,
            p=P
        )

        self.beta = min(1.0, self.beta + self.beta_increment)

        # Importance sampling weights
        weights = (len(P) * P[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        # Get samples
        samples = self._get_samples(indices, env)

        return (samples, indices, weights)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for given indices."""
        priorities = priorities.squeeze()
        priorities = np.clip(priorities, self.epsilon, None)

        self.priorities[indices] = priorities ** self.alpha
        self.max_priority = max(self.max_priority, priorities.max())