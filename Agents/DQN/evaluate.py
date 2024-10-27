import numpy as np


def evaluate_agent(env, model, n_episodes=100):
    """Evaluate the agent's performance."""
    all_episode_rewards = []
    episode_lengths = []
    rewards_per_step = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        episode_rewards = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            episode_rewards.append(reward)
            episode_length += 1

        all_episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        rewards_per_step.append(episode_rewards)

    max_steps = max(len(episode) for episode in rewards_per_step)
    average_rewards_per_step = np.zeros(max_steps)

    for step in range(max_steps):
        rewards_at_step = [episode[step] for episode in rewards_per_step if step < len(episode)]
        if rewards_at_step:
            average_rewards_per_step[step] = np.mean(rewards_at_step)

    print(f"Average reward over {n_episodes} episodes: {np.mean(all_episode_rewards):.2f}")
    print(f"Average episode length(steps) over {n_episodes} episodes: {np.mean(episode_lengths):.2f}")
    return average_rewards_per_step, all_episode_rewards, episode_lengths