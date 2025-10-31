"""
RL Training for Flappy Bird using DQN
Trains agent and saves checkpoints at intervals
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

# Define save directories
CHECKPOINT_DIR = Path("checkpoints")
METRICS_DIR = Path("metrics")

for dir in [CHECKPOINT_DIR, METRICS_DIR]:
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
    """Deep Q-Network for Flappy Bird"""
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


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
        self.episode_scores = []

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
            return

        transitions = self.memory.sample(self.config['batch_size'])
        batch = Transition(*zip(*transitions))

        # Compute mask for non-final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.config['batch_size'], device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.config['gamma']) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_network(self):
        """Update target network with policy network weights"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        # Soft update
        tau = self.config['tau']
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * tau + \
                                        target_net_state_dict[key] * (1 - tau)
        self.target_net.load_state_dict(target_net_state_dict)


def save_checkpoint(agent, episode, config, filename):
    """Save training checkpoint"""
    checkpoint = {
        'episode': episode,
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'config': config,
        'steps_done': agent.steps_done
    }
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint: {filename}")


def save_metrics(agent, episode, filename):
    """Save training metrics"""
    metrics = {
        'training_rewards': agent.episode_rewards,
        'training_lengths': agent.episode_lengths,
        'training_scores': agent.episode_scores,
        'mean_reward': np.mean(agent.episode_rewards[-100:]) if len(agent.episode_rewards) >= 100
                       else np.mean(agent.episode_rewards),
        'mean_score': np.mean(agent.episode_scores[-100:]) if len(agent.episode_scores) >= 100
                     else np.mean(agent.episode_scores),
        'std_reward': np.std(agent.episode_rewards[-100:]) if len(agent.episode_rewards) >= 100
                     else np.std(agent.episode_rewards)
    }

    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {filename}")


def train():
    """Main training loop"""

    # Training configuration
    config = {
        'lr': 1e-4,
        'batch_size': 128,
        'gamma': 0.99,
        'eps_start': 0.9,
        'eps_end': 0.05,
        'eps_decay': 5000,
        'tau': 0.005,
        'memory_size': 50000
    }

    # Training parameters
    num_episodes = 2000
    checkpoint_interval = 50  # Save every 50 episodes

    # Create environment
    env = FlappyBirdEnv()

    # Initialize agent
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(n_observations, n_actions, config)

    print(f"Starting training for {num_episodes} episodes")
    print(f"Observation space: {n_observations}, Action space: {n_actions}")

    # Training loop
    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        episode_reward = 0
        episode_length = 0

        while True:
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

            # Perform optimization
            agent.optimize_model()

            # Soft update target network
            agent.update_target_network()

            if done:
                agent.episode_rewards.append(episode_reward)
                agent.episode_lengths.append(episode_length)
                agent.episode_scores.append(info['score'])
                break

        # Print progress
        if (episode + 1) % 10 == 0:
            recent_rewards = agent.episode_rewards[-10:]
            recent_scores = agent.episode_scores[-10:]
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {np.mean(recent_rewards):.2f} | "
                  f"Avg Score: {np.mean(recent_scores):.2f} | "
                  f"Steps: {agent.steps_done}")

        # Save checkpoint
        if (episode + 1) % checkpoint_interval == 0 or episode == 0:
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_ep{episode:04d}.pt"
            metrics_path = METRICS_DIR / f"metrics_ep{episode:04d}.json"

            save_checkpoint(agent, episode, config, checkpoint_path)
            save_metrics(agent, episode, metrics_path)

    # Save final checkpoint
    final_checkpoint = CHECKPOINT_DIR / f"checkpoint_ep{num_episodes-1:04d}.pt"
    final_metrics = METRICS_DIR / f"metrics_ep{num_episodes-1:04d}.json"
    save_checkpoint(agent, num_episodes-1, config, final_checkpoint)
    save_metrics(agent, num_episodes-1, final_metrics)

    env.close()
    print("\nTraining complete!")
    print(f"Final average reward (last 100): {np.mean(agent.episode_rewards[-100:]):.2f}")
    print(f"Final average score (last 100): {np.mean(agent.episode_scores[-100:]):.2f}")


if __name__ == "__main__":
    train()
