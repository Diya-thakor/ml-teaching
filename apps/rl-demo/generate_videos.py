"""
Generate videos from saved checkpoints to show agent improvement
"""
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import json
from train_agent import DQN, DQNAgent

# Directories
CHECKPOINT_DIR = Path("checkpoints")
VIDEO_DIR = Path("videos")
METRICS_DIR = Path("metrics")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_agent_from_checkpoint(checkpoint_path):
    """Load agent from checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract info
    config = checkpoint['config']
    n_observations = 4  # CartPole has 4 observations
    n_actions = 2  # CartPole has 2 actions

    # Create agent
    agent = DQNAgent(n_observations, n_actions, config)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])

    return agent, checkpoint['episode']


def record_episode(agent, env, max_steps=500):
    """Record a single episode and return frames + info"""
    frames = []
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    episode_reward = 0
    step = 0

    # Capture initial frame
    frames.append(env.render())

    while step < max_steps:
        # Select action (greedy, no exploration)
        action = agent.select_action(state_tensor, training=False)

        # Step environment
        observation, reward, terminated, truncated, _ = env.step(action.item())
        episode_reward += reward
        step += 1

        # Capture frame
        frames.append(env.render())

        if terminated or truncated:
            break

        state_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    return frames, episode_reward, step


def create_video_from_frames(frames, episode_num, episode_reward, steps, output_path):
    """Create video from frames with info overlay"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')

    # Display first frame
    im = ax.imshow(frames[0])

    # Add title
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes,
                   ha='center', fontsize=14, fontweight='bold')

    def init():
        im.set_data(frames[0])
        title.set_text(f'Episode {episode_num} | Reward: {episode_reward:.1f} | Steps: {steps}')
        return [im, title]

    def update(frame_idx):
        im.set_data(frames[frame_idx])
        return [im, title]

    anim = FuncAnimation(fig, update, init_func=init, frames=len(frames),
                        interval=50, blit=True)

    # Save as GIF
    writer = PillowWriter(fps=20)
    anim.save(output_path, writer=writer)
    plt.close(fig)

    print(f"  ✓ Video saved: {output_path}")


def generate_all_videos(checkpoint_episodes=None):
    """Generate videos for all checkpoints"""

    print(f"\n{'='*60}")
    print(f"Generating Videos from Checkpoints")
    print(f"{'='*60}\n")

    # Find all checkpoints
    if checkpoint_episodes is None:
        checkpoint_files = sorted(CHECKPOINT_DIR.glob("checkpoint_ep*.pt"))
    else:
        checkpoint_files = [CHECKPOINT_DIR / f"checkpoint_ep{ep:04d}.pt"
                          for ep in checkpoint_episodes]
        checkpoint_files = [f for f in checkpoint_files if f.exists()]

    if not checkpoint_files:
        print("No checkpoint files found!")
        return

    # Create environment for rendering
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    video_info = []

    for checkpoint_path in checkpoint_files:
        print(f"\nProcessing: {checkpoint_path.name}")

        # Load agent
        agent, episode_num = load_agent_from_checkpoint(checkpoint_path)

        # Record multiple episodes and take the best one
        best_frames = None
        best_reward = -float('inf')
        best_steps = 0

        num_trials = 5
        for trial in range(num_trials):
            frames, reward, steps = record_episode(agent, env)

            if reward > best_reward:
                best_reward = reward
                best_frames = frames
                best_steps = steps

        # Create video
        video_path = VIDEO_DIR / f"agent_ep{episode_num:04d}.gif"
        create_video_from_frames(best_frames, episode_num, best_reward,
                               best_steps, video_path)

        video_info.append({
            'episode': episode_num,
            'reward': float(best_reward),
            'steps': best_steps,
            'video_path': str(video_path)
        })

    env.close()

    # Save video info
    with open(VIDEO_DIR / 'video_info.json', 'w') as f:
        json.dump(video_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Video Generation Complete!")
    print(f"Videos saved in: {VIDEO_DIR}")
    print(f"{'='*60}\n")

    return video_info


def create_comparison_figure():
    """Create a comparison figure showing progression"""
    import json

    # Load video info
    with open(VIDEO_DIR / 'video_info.json', 'r') as f:
        video_info = json.load(f)

    # Create comparison
    fig, axes = plt.subplots(2, len(video_info), figsize=(4*len(video_info), 8))
    if len(video_info) == 1:
        axes = axes.reshape(-1, 1)

    for idx, info in enumerate(video_info):
        ep = info['episode']

        # Load metrics
        metrics_path = METRICS_DIR / f"metrics_ep{ep:04d}.json"
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # Plot training curve
        ax = axes[0, idx]
        rewards = metrics['training_rewards']
        ax.plot(rewards, alpha=0.3, color='blue')
        # Rolling average
        window = 50
        if len(rewards) >= window:
            rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), rolling, color='red', linewidth=2)
        ax.set_title(f"Episode {ep}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(alpha=0.3)
        ax.axhline(y=500, color='green', linestyle='--', alpha=0.5, label='Max')

        # Show sample frame
        ax = axes[1, idx]
        video_path = Path(info['video_path'])
        ax.text(0.5, 0.5, f"See: {video_path.name}\n\n"
                         f"Reward: {info['reward']:.0f}\n"
                         f"Steps: {info['steps']}",
               ha='center', va='center', fontsize=12,
               transform=ax.transAxes)
        ax.set_title(f"Performance")
        ax.axis('off')

    plt.tight_layout()
    comparison_path = VIDEO_DIR / 'training_progression.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Comparison figure saved: {comparison_path}")


if __name__ == "__main__":
    # Generate videos for key checkpoints
    generate_all_videos()
    create_comparison_figure()
