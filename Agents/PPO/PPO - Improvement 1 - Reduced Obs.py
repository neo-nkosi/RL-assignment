import gymnasium as gym
from collections import OrderedDict

import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.gym_compat import GymEnv, DiscreteActSpace, BoxGymObsSpace
from grid2op.gym_compat import BoxGymActSpace

from lightsim2grid import LightSimBackend

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import numpy as np
import os

class TrainingCallback(BaseCallback):
    """
    A custom callback that prints training progress at each step.
    """
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            print(f"Current timestep: {self.num_timesteps}")
        return True

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

        # Debug: List available action attributes
        available_attrs = self._g2op_env.action_space
        print("Available action attributes:", available_attrs)

        self.setup_observations()
        self.setup_actions()

    def setup_observations(self):
        # Uses all observation attributes
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = BoxGymObsSpace(
            self._g2op_env.observation_space,
        )
        self.observation_space = self._gym_env.observation_space
        print("Observation space set up with shape:", self.observation_space.shape)

    def setup_actions(self):
        # BoxGymActSpace includes the continuous actions
        self._gym_env.action_space.close()
        self._gym_env.action_space = BoxGymActSpace(
            self._g2op_env.action_space
        )
        self.action_space = self._gym_env.action_space
        print(f"Action space set up with shape: {self.action_space}")

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


def main():
    env = Gym2OpEnv()

    print("#####################")
    print("# OBSERVATION SPACE #")
    print("#####################")
    print(env.observation_space)
    print("#####################\n")

    print("#####################")
    print("#   ACTION SPACE    #")
    print("#####################")
    print(env.action_space)
    print("#####################\n\n")

    # Tensorboards used for visualising rewards during training
    log_dir = "tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(env, log_dir)

    # Create the PPO agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    callback = TrainingCallback(verbose=1)

    # Train the agent
    total_timesteps = 1000000 
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the trained agent
    model.save("ppo_grid2op_agent_base")


if __name__ == "__main__":
    main()
