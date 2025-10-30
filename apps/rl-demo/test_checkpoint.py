"""
Quick script to test loading and evaluating a checkpoint
Usage: python test_checkpoint.py <episode_number>
Example: python test_checkpoint.py 100
"""
import sys
import gymnasium as gym
import torch
from pathlib import Path
from train_agent import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_test(episode_num, n_episodes=10, render=False):
    """Load checkpoint and run test episodes"""

    checkpoint_path = Path(f"checkpoints/checkpoint_ep{episode_num:04d}.pt")

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create agent
    config = checkpoint['config']
    n_observations = 4
    n_actions = 2
    agent = DQNAgent(n_observations, n_actions, config)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

    print(f"✓ Loaded agent from episode {checkpoint['episode']}")
    print(f"  Total training steps: {checkpoint['steps_done']}")

    # Create environment
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')

    # Run test episodes
    print(f"\nRunning {n_episodes} test episodes...")
    rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0

        while True:
            action = agent.select_action(state, training=False)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward

            if terminated or truncated:
                break

            state = torch.tensor(observation, dtype=torch.float32,
                               device=device).unsqueeze(0)

        rewards.append(episode_reward)
        print(f"  Episode {ep+1}: Reward = {episode_reward:.0f}")

    env.close()

    # Print summary
    print(f"\n{'='*50}")
    print(f"Results for Episode {episode_num} checkpoint:")
    print(f"  Mean Reward: {sum(rewards)/len(rewards):.2f}")
    print(f"  Std Reward:  {(sum((r - sum(rewards)/len(rewards))**2 for r in rewards)/len(rewards))**0.5:.2f}")
    print(f"  Min Reward:  {min(rewards):.0f}")
    print(f"  Max Reward:  {max(rewards):.0f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_checkpoint.py <episode_number> [--render]")
        print("Example: python test_checkpoint.py 100")
        print("         python test_checkpoint.py 500 --render")
        sys.exit(1)

    episode = int(sys.argv[1])
    render = "--render" in sys.argv

    load_and_test(episode, n_episodes=10, render=render)
