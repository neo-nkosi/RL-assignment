import gymnasium as gym
import grid2op
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, LinesReconnectedReward, CombinedScaledReward
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, BoxGymActSpace

from lightsim2grid import LightSimBackend

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import numpy as np
import os

# Custom callback for training progress
class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            print(f"Current timestep: {self.num_timesteps}")
        return True

# Gym environment wrapper
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
        cr.addReward("l2rpn", L2RPNReward(), 1.0)
        cr.addReward("lines_reconnected", LinesReconnectedReward(), 0.5)
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

def main():
    env = Gym2OpEnv()

    # Create directory for TensorBoard logs
    log_dir = "tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap the environment with Monitor
    env = Monitor(env, log_dir)

    # Create the PPO agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # Initialize the custom callback
    callback = TrainingCallback(verbose=1)

    # Train the agent
    total_timesteps = 1000000
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the trained agent
    model.save("ppo_agent_l2rpn_lines_reconnected")

    # Evaluate the trained agent
    obs, _ = env.reset()
    total_reward = 0
    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode terminated at step {step + 1}")
            obs, _ = env.reset()

    print("Evaluation completed. Total reward:", total_reward)

if __name__ == "__main__":
    main()
