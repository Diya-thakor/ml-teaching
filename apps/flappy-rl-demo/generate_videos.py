"""
Generate animated GIF videos of trained agents playing Flappy Bird
"""
import torch
import numpy as np
from pathlib import Path
import imageio
from PIL import Image, ImageDraw, ImageFont
from flappy_env import FlappyBirdEnv
from train_agent import DQNAgent

# Directories
CHECKPOINT_DIR = Path("checkpoints")
VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def render_frame(env, score, episode_num):
    """Render a single frame as a PIL Image"""
    # Create image
    img = Image.new('RGB', (400, 600), color=(135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)

    # Draw pipes
    for pipe in env.pipes:
        # Top pipe (green)
        top_rect = pipe.get_top_rect()
        draw.rectangle(
            [top_rect.x, top_rect.y, top_rect.x + top_rect.width, top_rect.y + top_rect.height],
            fill=(0, 200, 0), outline=(0, 150, 0), width=2
        )

        # Bottom pipe (green)
        bottom_rect = pipe.get_bottom_rect()
        draw.rectangle(
            [bottom_rect.x, bottom_rect.y, bottom_rect.x + bottom_rect.width, bottom_rect.y + bottom_rect.height],
            fill=(0, 200, 0), outline=(0, 150, 0), width=2
        )

    # Draw ground
    draw.rectangle([0, 500, 400, 600], fill=(222, 216, 149))

    # Draw bird (yellow circle)
    bird_rect = env.bird.get_rect()
    draw.ellipse(
        [bird_rect.x, bird_rect.y, bird_rect.x + bird_rect.width, bird_rect.y + bird_rect.height],
        fill=(255, 255, 0), outline=(200, 200, 0), width=2
    )

    # Draw score
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), f"Score: {score}", fill=(255, 255, 255), font=font)
    draw.text((10, 50), f"Episode: {episode_num}", fill=(255, 255, 255), font=font)

    return np.array(img)


def load_agent_from_checkpoint(checkpoint_path):
    """Load agent from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    n_observations = 6  # Flappy Bird observation space
    n_actions = 2

    agent = DQNAgent(n_observations, n_actions, config)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

    return agent, checkpoint['episode']


def generate_video(checkpoint_path, output_path, max_frames=500):
    """Generate a video of the agent playing"""
    print(f"Generating video from {checkpoint_path}...")

    # Load agent
    agent, episode_num = load_agent_from_checkpoint(checkpoint_path)

    # Create environment
    env = FlappyBirdEnv()

    # Run episode and capture frames
    frames = []
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    frame_count = 0
    score = 0

    while frame_count < max_frames:
        # Capture frame
        frame = render_frame(env, score, episode_num)
        frames.append(frame)

        # Select action
        action = agent.select_action(state, training=False)

        # Step environment
        observation, reward, terminated, truncated, info = env.step(action.item())
        score = info['score']

        frame_count += 1

        if terminated or truncated:
            # Add a few final frames showing the crash
            for _ in range(5):
                frames.append(frame)
            break

        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    env.close()

    # Save as GIF
    imageio.mimsave(output_path, frames, fps=30, loop=0)
    print(f"Saved video: {output_path} (Score: {score}, Frames: {len(frames)})")

    return score, len(frames)


def main():
    """Generate videos for all checkpoints"""
    checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_ep*.pt"))

    if not checkpoints:
        print("No checkpoints found! Please run training first.")
        return

    print(f"Found {len(checkpoints)} checkpoints")

    # Generate videos for key checkpoints
    key_episodes = [0, 50, 100, 200, 500, 1000, 1500, 1950, 1999]  # Added final trained agent

    for checkpoint_path in checkpoints:
        episode_num = int(checkpoint_path.stem.split('ep')[1])

        # Only generate for key episodes
        if episode_num not in key_episodes:
            continue

        output_path = VIDEO_DIR / f"agent_ep{episode_num:04d}.gif"

        if output_path.exists():
            print(f"Video already exists: {output_path}")
            continue

        try:
            score, frames = generate_video(checkpoint_path, output_path, max_frames=1000)
        except Exception as e:
            print(f"Error generating video for episode {episode_num}: {e}")
            continue

    print("\nVideo generation complete!")


if __name__ == "__main__":
    main()
