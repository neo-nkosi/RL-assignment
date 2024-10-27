import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, Union, ClassVar

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    create_mlp,
    FlattenExtractor
)
from stable_baselines3.dqn import DQN
from stable_baselines3.common.type_aliases import GymEnv, Schedule


class DuelingNetwork(BasePolicy):
    """
    Dueling Network architecture for DQN.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Discrete,
            features_extractor: BaseFeaturesExtractor,
            features_dim: int,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.features_dim = features_dim
        action_dim = int(self.action_space.n)

        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(features_dim, net_arch[0]),
            activation_fn(),
            nn.Linear(net_arch[0], net_arch[1]),
            activation_fn()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(net_arch[1], net_arch[1]),
            activation_fn(),
            nn.Linear(net_arch[1], 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(net_arch[1], net_arch[1]),
            activation_fn(),
            nn.Linear(net_arch[1], action_dim)
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Forward pass through the network.
        """
        features = self.extract_features(obs, self.features_extractor)
        shared_features = self.shared_net(features)

        values = self.value_stream(shared_features)
        advantages = self.advantage_stream(shared_features)

        # Combine value and advantage streams
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

class DuelingDQNPolicy(DQNPolicy):
    """
    Policy class with Q-Network and target network for DuelingDQN.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Discrete,
            lr_schedule: Schedule,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(BasePolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.features_extractor = features_extractor_class(self.observation_space)
        self.features_dim = self.features_extractor.features_dim

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "features_extractor": self.features_extractor,
            "features_dim": self.features_dim
        }

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.
        """
        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_q_net(self) -> DuelingNetwork:
        """Create the dueling network."""
        return DuelingNetwork(**self.net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic)

class DuelingDQN(DQN):
    """
    Dueling Deep Q-Network (DuelingDQN)
    """
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": DuelingDQNPolicy,
        "CnnPolicy": DuelingDQNPolicy,
        "MultiInputPolicy": DuelingDQNPolicy,
    }

    def __init__(
            self,
            policy: Union[str, Type[DuelingDQNPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 1e-4,
            buffer_size: int = 1_000_000,
            learning_starts: int = 100,
            batch_size: int = 32,
            tau: float = 1.0,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 4,
            gradient_steps: int = 1,
            target_update_interval: int = 10000,
            exploration_fraction: float = 0.1,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.05,
            max_grad_norm: float = 10,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if policy_kwargs is None:
            policy_kwargs = {}

        if "net_arch" not in policy_kwargs:
            policy_kwargs["net_arch"] = [64, 64]

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

class DQNWithPER(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer with priorities
            replay_data, indices, weights = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute TD error
            td_error = (target_q_values - current_q_values).abs().detach().cpu().numpy()

            # Update priorities
            self.replay_buffer.update_priorities(indices, td_error)

            # Compute weighted Huber loss (less sensitive to outliers)
            weights = th.FloatTensor(weights).to(self.device)
            loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
            loss = (loss * weights.reshape(-1, 1)).mean()

            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

class DuelingDQNWithPER(DuelingDQN):
    """
    Dueling DQN with Prioritized Experience Replay
    """
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer with priorities
            replay_data, indices, weights = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute TD error for updating priorities
            td_error = (target_q_values - current_q_values).abs().detach().cpu().numpy()

            # Update priorities in replay buffer
            self.replay_buffer.update_priorities(indices, td_error)

            # Compute Huber loss (less sensitive to outliers) with importance sampling weights
            weights = th.FloatTensor(weights).to(self.device)
            loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
            loss = (loss * weights.reshape(-1, 1)).mean()

            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))