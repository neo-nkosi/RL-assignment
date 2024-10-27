import os
from typing import List, Dict, Optional
from matplotlib import pyplot as plt
import numpy as np


class GridMonitor:
    def __init__(self, save_dir: str = "training_plots/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.training_rewards: List[float] = []
        self.evaluation_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.rewards_per_step: List[List[float]] = []

    def save_training_plots(self, callback_rewards: List[float], title_prefix: str = ""):
        """Save training metrics plots."""
        # Learning curve
        plt.figure(figsize=(10, 5))
        plt.plot(callback_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Training Reward')
        plt.title(f'{title_prefix} Training Learning Curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f'{title_prefix}_training_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Training reward variance
        rewards_variance = [np.var(callback_rewards[:i + 1]) for i in range(len(callback_rewards))]
        plt.figure(figsize=(10, 5))
        plt.plot(rewards_variance)
        plt.xlabel('Episode')
        plt.ylabel('Training Reward Variance')
        plt.title(f'{title_prefix} Training Reward Variance')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f'{title_prefix}_training_variance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_evaluation_plots(self, eval_data: Dict, title_prefix: str = ""):
        """Save evaluation metrics plots."""
        avg_rewards_per_step, episode_rewards, episode_lengths = eval_data

        # Average reward per step
        plt.figure(figsize=(10, 5))
        plt.plot(avg_rewards_per_step)
        plt.xlabel('Step')
        plt.ylabel('Average Reward')
        plt.title(f'{title_prefix} Average Reward per Step')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f'{title_prefix}_avg_reward_per_step.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Evaluation learning curve
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Evaluation Reward')
        plt.title(f'{title_prefix} Evaluation Learning Curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f'{title_prefix}_eval_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Episode lengths
        plt.figure(figsize=(10, 5))
        plt.plot(episode_lengths)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title(f'{title_prefix} Episode Lengths')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f'{title_prefix}_episode_lengths.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Reward variance
        rewards_variance = [np.var(episode_rewards[:i + 1]) for i in range(len(episode_rewards))]
        plt.figure(figsize=(10, 5))
        plt.plot(rewards_variance)
        plt.xlabel('Episode')
        plt.ylabel('Evaluation Reward Variance')
        plt.title(f'{title_prefix} Evaluation Reward Variance')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f'{title_prefix}_eval_variance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_comparison_plots(self, models_data: Dict[str, Dict]):
        """Save comparison plots between different models."""
        plt.figure(figsize=(12, 6))
        for model_name, data in models_data.items():
            plt.plot(data['eval_rewards'], label=model_name)
        plt.xlabel('Episode')
        plt.ylabel('Evaluation Reward')
        plt.title('Models Comparison - Evaluation Rewards')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'models_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
