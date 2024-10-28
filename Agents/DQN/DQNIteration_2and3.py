#%%
import gymnasium as gym
import grid2op
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.gym_compat import GymEnv, DiscreteActSpace, BoxGymObsSpace, ContinuousToDiscreteConverter
from gymnasium.wrappers import NormalizeObservation
from lightsim2grid import LightSimBackend
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
from Agents.DQN.graphs import GridMonitor
from Agents.DQN.PERBuffer import PrioritizedReplayBuffer
from Agents.DQN.customDQNs import DuelingDQNWithPER, DQNWithPER
from Agents.DQN.evaluate import evaluate_agent

#%%
class TrainingCallback(BaseCallback):
    """Custom callback for training monitoring."""
    def __init__(self, monitor: GridMonitor, verbose=0):
        super().__init__(verbose)
        self.monitor = monitor
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        reward = self.locals['rewards']
        self.current_episode_reward += reward
        
        if self.locals.get('done', False):
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        return True

#%%
class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env, epsilon=1e-8):
        super(NormalizeObservation, self).__init__(env)
        self.epsilon = epsilon
        self.obs_mean = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.obs_var = np.ones(self.observation_space.shape, dtype=np.float32)
        self.obs_count = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.observation_space = env.observation_space  # Maintain the same observation space

    def observation(self, observation):
        # Update running mean and variance using Welford's online algorithm
        self.obs_count += 1
        delta = observation - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = observation - self.obs_mean
        self.obs_var += delta * delta2

        # Compute normalized observation
        obs_std = np.sqrt(self.obs_var / (self.obs_count + self.epsilon))
        normalized_obs = (observation - self.obs_mean) / (obs_std + self.epsilon)

        return normalized_obs

#%%
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

        available_attrs = self._g2op_env.action_space
        print("Available action attributes:", available_attrs)

        self.setup_observations()
        self.setup_actions()

    def setup_observations(self):
        obs_attr_to_keep = ["rho", "gen_p",
                            "load_p", "topo_vect",
                            "thermal_limit",
                            "actual_dispatch",
                            "current_step"]

        self._gym_env.observation_space = BoxGymObsSpace(
            self._g2op_env.observation_space,
             attr_to_keep=obs_attr_to_keep
        )
        self.observation_space = gym.spaces.Box(
            low=self._gym_env.observation_space.low,
            high=self._gym_env.observation_space.high,
            shape=self._gym_env.observation_space.shape,
            dtype=np.float32
        )
        print("Observation space set up with shape:", self.observation_space.shape)


    def setup_actions(self):
        self._gym_env.action_space = self._gym_env.action_space.reencode_space("redispatch", ContinuousToDiscreteConverter(nb_bins=11))
        # Customize action space using DiscreteActSpace
        act_attr_to_keep = ["change_line_status", "change_bus", "redispatch"]
        self._gym_env.action_space = DiscreteActSpace(
            self._g2op_env.action_space,
             attr_to_keep=act_attr_to_keep
        )
        self.action_space = self._gym_env.action_space
        print(f"Action space set up with {self.action_space.n} discrete actions.")

    def reset(self, seed=None, options=None):
        obs, info = self._gym_env.reset(seed=seed, options=options)
        if isinstance(obs, tuple):
            obs = obs[0]  # Extract observation if a tuple is returned
        # print(f"Reset called. Observation: {obs}")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._gym_env.step(action)
        done = terminated or truncated
        # print(f"Step called. Action: {action}, Reward: {reward}, Done: {done}")

        return obs, reward, terminated, truncated, info

    def render(self):
        # Optional: Implement render functionality if needed
        return self._gym_env.render()

#%%
def main():
    # Create environment
    env = Gym2OpEnv()
    env = Monitor(env, "tensorboard_logs/")
    env = NormalizeObservation(env)
    
    total_timesteps = 500000
    initial_beta = 0.4
    final_beta = 1.0
    # Initialize monitor
    monitor = GridMonitor(save_dir="training_plots/Iteration_2")
    
    required_increase = final_beta - initial_beta
    beta_increment = required_increase / total_timesteps
    
    models = {
        "DQNWithPER": DQNWithPER(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log="tensorboard_logs/",
            exploration_fraction=0.30,
            replay_buffer_class=PrioritizedReplayBuffer,
            replay_buffer_kwargs={
                "alpha": 0.6,  # Priority exponent
                "beta": 0.4,  # Initial importance sampling weight
                "beta_increment": beta_increment  # Calculated based on total timesteps
            }
        ),
        "DuelingDQNWithPER": DuelingDQNWithPER(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": [64, 64]},  # Dueling network architecture
            verbose=1,
            tensorboard_log="tensorboard_logs/",
            exploration_fraction=0.30,
            replay_buffer_class=PrioritizedReplayBuffer,
            replay_buffer_kwargs={
                "alpha": 0.6,  # Priority exponent
                "beta": 0.4,  # Initial importance sampling weight
                "beta_increment": 0.0000012  # Calculated based on total timesteps
            }
                )
    }
    models_data = {}
   
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        callback = TrainingCallback(monitor, verbose=1)
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # Save model
        model.save(f"{model_name}_final")
        
        # Evaluate and save plots
        eval_data = evaluate_agent(env, model, n_episodes=100)
        monitor.save_training_plots(callback.episode_rewards, title_prefix=model_name)
        monitor.save_evaluation_plots(eval_data, title_prefix=model_name)
        
        # Store data for comparison
        models_data[model_name] = {
            'eval_rewards': eval_data[1],  # all_episode_rewards
            'episode_lengths': eval_data[2]
        }
    
    # Save comparison plots
    monitor.save_comparison_plots(models_data)

#%%

if __name__ == "__main__":
    main()