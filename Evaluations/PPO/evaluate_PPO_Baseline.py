import gymnasium as gym
from collections import OrderedDict
import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.gym_compat import GymEnv, BoxGymActSpace, BoxGymObsSpace
from lightsim2grid import LightSimBackend
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import copy
import os
import random
import torch

def set_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Gym environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
        p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep

        # Make grid2op env
        self._g2op_env = grid2op.make(
            self._env_name,
            backend=self._backend,
            test=False,
            action_class=action_class,
            observation_class=observation_class,
            reward_class=reward_class,
            param=p,
        )

        ##########
        # REWARD #
        ##########
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        cr.initialize(self._g2op_env)
        ##########

        self._gym_env = GymEnv(self._g2op_env)

        self.setup_observations()
        self.setup_actions()

    def setup_observations(self):
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = BoxGymObsSpace(
            self._g2op_env.observation_space
        )
        self.observation_space = self._gym_env.observation_space

    def setup_actions(self):
        # BoxGymActSpace includes the continuous actions
        self._gym_env.action_space.close()
        self._gym_env.action_space = BoxGymActSpace(
            self._g2op_env.action_space
        )
        self.action_space = self._gym_env.action_space

    def reset(self, seed=None, options=None):
        obs, info = self._gym_env.reset(seed=seed, options=options)
        if isinstance(obs, tuple):
            obs = obs[0]  
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._gym_env.step(action)
        done = terminated or truncated
        return obs, reward, terminated, truncated, info

    def render(self):
        # Optional: Implement render functionality if needed
        return self._gym_env.render()

def evaluate_agent(env, model, num_episodes=100, max_steps=10000, save_data=False, data_filename=None):
    total_rewards = []
    episode_lengths = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            if done:
                print(f"Episode {episode + 1} terminated at step {step + 1} with reward {episode_reward}")
                break
        total_rewards.append(episode_reward)
        episode_lengths.append(step + 1)

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    print("Agent evaluation completed.")

    if save_data and data_filename is not None:
        np.savez(data_filename, rewards=total_rewards, lengths=episode_lengths)

    return avg_reward, std_reward, avg_length, std_length

def main():
    set_seed()

    env = Gym2OpEnv()
    env = Monitor(env)

    # Load the trained agent
    model = PPO.load("ppo_grid2op_agent_base", env=env)

    # Evaluate the trained agent
    print("Evaluating the baseline trained agent...")
    avg_reward_trained, std_reward_trained, avg_length_trained, std_length_trained = evaluate_agent(
        env, model, num_episodes=100, max_steps=10000, save_data=True, data_filename="trained_agent_data.npz"
    )

    print(f"Trained Agent (baseline) Average Reward: {avg_reward_trained} ± {std_reward_trained}")
    print(f"Trained Agent (baseline) Average Episode Length: {avg_length_trained} ± {std_length_trained}")


if __name__ == "__main__":
    main()
