"""
RL Training Demo with Checkpoints and Videos
Trains a DQN agent on CartPole and saves checkpoints + videos at intervals
"""
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import math
from pathlib import Path
import json

# Define save directories
CHECKPOINT_DIR = Path("checkpoints")
VIDEO_DIR = Path("videos")
METRICS_DIR = Path("metrics")

for dir in [CHECKPOINT_DIR, VIDEO_DIR, METRICS_DIR]:
    dir.mkdir(exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transition for replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """Experience replay buffer"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """Deep Q-Network"""
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    """DQN Agent with training capabilities"""
    def __init__(self, n_observations, n_actions, config):
        self.n_actions = n_actions
        self.config = config

        # Networks
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer and memory
        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                     lr=config['lr'], amsgrad=True)
        self.memory = ReplayMemory(config['memory_size'])

        # Training state
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training:
            eps_threshold = (self.config['eps_end'] +
                           (self.config['eps_start'] - self.config['eps_end']) *
                           math.exp(-1. * self.steps_done / self.config['eps_decay']))
            self.steps_done += 1

            if random.random() > eps_threshold:
                with torch.no_grad():
                    return self.policy_net(state).max(1).indices.view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.n_actions)]],
                                  device=device, dtype=torch.long)
        else:
            # Greedy action for evaluation
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)

    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < self.config['batch_size']:
            return None

        transitions = self.memory.sample(self.config['batch_size'])
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)),
                                     device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                          if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # V(s_{t+1})
        next_state_values = torch.zeros(self.config['batch_size'], device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Expected Q values
        expected_state_action_values = (next_state_values * self.config['gamma']) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Soft update of target network"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (policy_net_state_dict[key] * self.config['tau'] +
                                         target_net_state_dict[key] * (1 - self.config['tau']))
        self.target_net.load_state_dict(target_net_state_dict)

    def save_checkpoint(self, episode, filename):
        """Save model checkpoint"""
        checkpoint = {
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'config': self.config
        }
        torch.save(checkpoint, filename)
        print(f"✓ Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        print(f"✓ Checkpoint loaded: {filename}")
        return checkpoint['episode']


def evaluate_agent(agent, env, n_episodes=10):
    """Evaluate agent performance"""
    total_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0

        while True:
            action = agent.select_action(state, training=False)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward

            if terminated or truncated:
                break

            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)


def train_dqn(num_episodes=500, checkpoint_intervals=[0, 50, 100, 200, 300, 400, 500]):
    """Main training loop with checkpoints"""

    # Training configuration
    config = {
        'batch_size': 128,
        'gamma': 0.99,
        'eps_start': 0.9,
        'eps_end': 0.05,
        'eps_decay': 1000,
        'tau': 0.005,
        'lr': 1e-4,
        'memory_size': 10000
    }

    # Create environment
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    # Create agent
    agent = DQNAgent(n_observations, n_actions, config)

    # Training metrics
    all_losses = []

    print(f"\n{'='*60}")
    print(f"Training DQN on CartPole-v1")
    print(f"Episodes: {num_episodes}")
    print(f"Checkpoints at: {checkpoint_intervals}")
    print(f"{'='*60}\n")

    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0
        episode_length = 0

        while True:
            # Select and perform action
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward
            episode_length += 1

            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32,
                                        device=device).unsqueeze(0)

            # Store transition
            agent.memory.push(state, action, next_state, reward)
            state = next_state

            # Optimize model
            loss = agent.optimize_model()
            if loss is not None:
                all_losses.append(loss)

            # Update target network
            agent.update_target_network()

            if done:
                agent.episode_rewards.append(episode_reward)
                agent.episode_lengths.append(episode_length)
                break

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(agent.episode_rewards[-50:])
            avg_length = np.mean(agent.episode_lengths[-50:])
            print(f"Episode {episode+1:3d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Length: {avg_length:6.2f} | "
                  f"Steps: {agent.steps_done:5d}")

        # Save checkpoint
        if episode in checkpoint_intervals:
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_ep{episode:04d}.pt"
            agent.save_checkpoint(episode, checkpoint_path)

            # Evaluate and save metrics
            mean_reward, std_reward = evaluate_agent(agent, env)
            metrics = {
                'episode': episode,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'training_rewards': agent.episode_rewards,
                'training_lengths': agent.episode_lengths
            }
            metrics_path = METRICS_DIR / f"metrics_ep{episode:04d}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  → Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()

    # Final checkpoint
    if num_episodes not in checkpoint_intervals:
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_ep{num_episodes:04d}.pt"
        agent.save_checkpoint(num_episodes, checkpoint_path)

    # Save final metrics
    final_metrics = {
        'total_episodes': num_episodes,
        'final_rewards': agent.episode_rewards,
        'final_lengths': agent.episode_lengths,
        'total_steps': agent.steps_done,
        'config': config
    }
    with open(METRICS_DIR / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print(f"Metrics saved in: {METRICS_DIR}")
    print(f"{'='*60}\n")

    return agent


if __name__ == "__main__":
    agent = train_dqn(num_episodes=500)
