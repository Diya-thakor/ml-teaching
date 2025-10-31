"""
Improved training for Flappy Bird - targeting 10-20+ pipes
Uses better hyperparameters and reward shaping
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import math
from pathlib import Path
import json
from flappy_env import FlappyBirdEnv
from train_agent import DQNAgent, save_checkpoint, save_metrics

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Setup directories
CHECKPOINT_DIR = Path("checkpoints_better")
METRICS_DIR = Path("metrics_better")
CHECKPOINT_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)


def train():
    """Improved training loop with better hyperparameters"""

    # IMPROVED Training configuration
    config = {
        'lr': 5e-5,  # Lower learning rate for stability
        'batch_size': 256,  # Larger batch for better gradients
        'gamma': 0.995,  # Higher discount - value future rewards more
        'eps_start': 1.0,  # Start with full exploration
        'eps_end': 0.01,  # Keep some exploration
        'eps_decay': 10000,  # Slower decay - explore more
        'tau': 0.001,  # Slower target network updates
        'memory_size': 100000  # Larger replay buffer
    }

    # Training parameters
    num_episodes = 3000  # More episodes to learn better
    checkpoint_interval = 50

    # Create environment
    env = FlappyBirdEnv()

    # Initialize agent
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(n_observations, n_actions, config)

    print(f"\n{'='*80}")
    print(f"IMPROVED TRAINING FOR HIGH SCORES (10-20+ PIPES)")
    print(f"{'='*80}")
    print(f"Target: 10-20+ pipes per episode")
    print(f"Episodes: {num_episodes}")
    print(f"Observation space: {n_observations}, Action space: {n_actions}")
    print(f"\nKey improvements:")
    print(f"  - Easier game (bigger gaps, slower speed)")
    print(f"  - Better hyperparameters (lower LR, larger batch)")
    print(f"  - More exploration (slower epsilon decay)")
    print(f"  - Larger replay buffer (100k)")
    print(f"{'='*80}\n")

    best_score = 0
    best_avg_score = 0
    episodes_above_10 = 0

    # Training loop
    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        episode_reward = 0
        episode_length = 0
        max_steps = 5000  # Longer episodes for better scores

        while episode_length < max_steps:
            # Select and perform action
            action = agent.select_action(state, training=True)
            observation, reward, terminated, truncated, info = env.step(action.item())

            episode_reward += reward
            episode_length += 1

            reward_tensor = torch.tensor([reward], device=device)

            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32,
                                         device=device).unsqueeze(0)

            # Store transition in memory
            agent.memory.push(state, action, next_state, reward_tensor)

            # Move to next state
            state = next_state

            # Perform optimization (do more updates per step for better learning)
            if len(agent.memory) >= config['batch_size']:
                agent.optimize_model()
                # Do extra optimization every 4 steps
                if episode_length % 4 == 0:
                    agent.optimize_model()

            # Soft update target network
            agent.update_target_network()

            if done:
                score = info['score']
                agent.episode_rewards.append(episode_reward)
                agent.episode_lengths.append(episode_length)
                agent.episode_scores.append(score)

                if score > best_score:
                    best_score = score
                    print(f"🎉 NEW BEST SCORE: {best_score} pipes at episode {episode}!")

                if score >= 10:
                    episodes_above_10 += 1

                break

        # Print progress with more detail
        if (episode + 1) % 10 == 0:
            recent_scores = agent.episode_scores[-50:] if len(agent.episode_scores) >= 50 else agent.episode_scores
            recent_rewards = agent.episode_rewards[-50:]
            recent_steps = agent.episode_lengths[-50:]

            avg_score = np.mean(recent_scores)
            if avg_score > best_avg_score:
                best_avg_score = avg_score

            print(f"Ep {episode + 1:4d}/{num_episodes} | "
                  f"Score: {avg_score:5.2f} (best: {best_score:2d}) | "
                  f"Steps: {np.mean(recent_steps):6.0f} | "
                  f"Reward: {np.mean(recent_rewards):7.1f} | "
                  f"10+: {episodes_above_10:3d} | "
                  f"Mem: {len(agent.memory):6d}")

        # Save checkpoint at specific episodes and when beating records
        if (episode + 1) % checkpoint_interval == 0 or episode == 0:
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_ep{episode:04d}.pt"
            metrics_path = METRICS_DIR / f"metrics_ep{episode:04d}.json"
            save_checkpoint(agent, episode, config, checkpoint_path)
            save_metrics(agent, episode, metrics_path)

            # Show progress report
            if episode > 0:
                recent_100 = agent.episode_scores[-100:] if len(agent.episode_scores) >= 100 else agent.episode_scores
                print(f"    📊 Checkpoint saved | Avg(last 100): {np.mean(recent_100):.2f} | Best: {best_score}")

    env.close()

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

    if len(agent.episode_scores) >= 100:
        last_100 = agent.episode_scores[-100:]
        print(f"Performance (last 100 episodes):")
        print(f"  Average Score:  {np.mean(last_100):.2f} pipes")
        print(f"  Best Score:     {np.max(last_100):.0f} pipes")
        print(f"  Median Score:   {np.median(last_100):.0f} pipes")
        print(f"  Episodes 10+:   {sum(1 for s in last_100 if s >= 10)}")
        print(f"  Episodes 20+:   {sum(1 for s in last_100 if s >= 20)}")

    print(f"\nOverall Best Score: {best_score} pipes")
    print(f"Total Episodes 10+: {episodes_above_10}")
    print(f"\nCheckpoints saved to: {CHECKPOINT_DIR}")
    print(f"Next steps:")
    print(f"  1. Generate videos: python generate_videos_better.py")
    print(f"  2. Play with agent: python play_game.py (select from checkpoints_better/)")
    print("="*80)


if __name__ == "__main__":
    train()
