import gymnasium as gym
from collections import OrderedDict

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, BoxGymActSpace

from lightsim2grid import LightSimBackend

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import numpy as np
import os

# Gym environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Define the backend and environment name
        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"

        # Define action and observation classes
        action_class = PlayableAction
        observation_class = CompleteObservation

        # Set parameters
        p = Parameters()
        p.MAX_SUB_CHANGED = 4
        p.MAX_LINE_STATUS_CHANGED = 4

        # Create the Grid2Op environment with CombinedScaledReward
        self._g2op_env = grid2op.make(
            self._env_name,
            backend=self._backend,
            action_class=action_class,
            observation_class=observation_class,
            reward_class=CombinedScaledReward,
            param=p,
        )

        # Get the combined reward instance
        cr = self._g2op_env.get_reward_instance()
        # Add individual rewards with their weights
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        # Initialize the combined reward
        cr.initialize(self._g2op_env)

        self._gym_env = GymEnv(self._g2op_env)

        # Set up observations and actions
        self.setup_observations()
        self.setup_actions()

    def setup_observations(self):
        # Define the observation space with selected attributes
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = BoxGymObsSpace(
            self._g2op_env.observation_space,
             attr_to_keep=[
                "rho",
                "line_status",
                "topo_vect",
                "prod_p", "prod_q", "prod_v",
                "load_p", "load_q", "load_v",
                "target_dispatch",
                "actual_dispatch",
                "hour_of_day",
                "day_of_week",
                "month",
                "gen_margin_up",
                "gen_margin_down",
            ]
        )
        self.observation_space = self._gym_env.observation_space

    def setup_actions(self):
        # Define the action space
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
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._gym_env.render()

def evaluate_trained_agent():
    # Initialize the environment
    env = Gym2OpEnv()

    # Wrap the environment with Monitor
    env = Monitor(env)

    # Load the trained agent
    model = PPO.load("ppo_agent_l2rpn_lines_reconnected", env=env)

    # Evaluate the agent
    num_episodes = 100  # Number of episodes to run for evaluation
    max_steps_per_episode = 20000  # Maximum steps per episode
    total_rewards = []
    steps_survived = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        step = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            if terminated or truncated:
                print(f"Episode {episode + 1} terminated at step {step}")
                break
        total_rewards.append(total_reward)
        steps_survived.append(step)
        print(f"Total reward for episode {episode + 1}: {total_reward}")

    average_reward = np.mean(total_rewards)
    std_dev_reward = np.std(total_rewards)
    average_steps = np.mean(steps_survived)
    std_dev_steps = np.std(steps_survived)

    print(f"\nAverage reward over {num_episodes} episodes: {average_reward}")
    print(f"\nStandard deviation for reward over {num_episodes} episodes: {std_dev_reward}")

    print(f"Average steps survived over {num_episodes} episodes: {average_steps}")
    print(f"\nStandard deviation for steps over {num_episodes} episodes: {std_dev_steps}")

if __name__ == "__main__":
    evaluate_trained_agent()
