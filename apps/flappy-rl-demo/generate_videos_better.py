"""
Generate animated GIF videos with IMPROVED GRAPHICS
Shows the 32-pipe trained agent in action!
"""
import torch
import numpy as np
from pathlib import Path
import imageio
from PIL import Image, ImageDraw, ImageFont
from flappy_env import FlappyBirdEnv
from train_agent import DQNAgent

# Directories
CHECKPOINT_DIR = Path("checkpoints_better")
VIDEO_DIR = Path("videos_better")
VIDEO_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Colors (matching play_game.py improved graphics)
SKY_BLUE = (135, 206, 250)
SKY_LIGHT = (180, 220, 255)
GREEN = (34, 177, 76)
DARK_GREEN = (25, 130, 50)
YELLOW = (255, 220, 0)
ORANGE = (255, 140, 0)
GROUND_COLOR = (222, 216, 149)
GROUND_DARK = (180, 160, 100)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GOLD = (255, 215, 0)


def render_frame_improved(env, score, episode_num):
    """Render a single frame with IMPROVED GRAPHICS"""
    img = Image.new('RGB', (400, 600))
    draw = ImageDraw.Draw(img)

    # Gradient sky background
    for y in range(0, 500, 2):
        color_ratio = y / 500
        r = int(SKY_BLUE[0] + (SKY_LIGHT[0] - SKY_BLUE[0]) * color_ratio)
        g = int(SKY_BLUE[1] + (SKY_LIGHT[1] - SKY_BLUE[1]) * color_ratio)
        b = int(SKY_BLUE[2] + (SKY_LIGHT[2] - SKY_BLUE[2]) * color_ratio)
        draw.rectangle([0, y, 400, y+2], fill=(r, g, b))

    # Draw pipes with 3D effect
    for pipe in env.pipes:
        # Top pipe
        top_rect = pipe.get_top_rect()
        # Main body
        draw.rectangle([top_rect.x, top_rect.y, top_rect.x + top_rect.width,
                       top_rect.y + top_rect.height], fill=GREEN)
        # Left highlight
        draw.rectangle([top_rect.x, top_rect.y, top_rect.x + 8,
                       top_rect.y + top_rect.height], fill=(50, 200, 100))
        # Right shadow
        draw.rectangle([top_rect.right - 8, top_rect.y, top_rect.right,
                       top_rect.y + top_rect.height], fill=DARK_GREEN)
        # Border
        draw.rectangle([top_rect.x, top_rect.y, top_rect.x + top_rect.width,
                       top_rect.y + top_rect.height], outline=BLACK, width=2)

        # Top pipe cap
        if top_rect.height > 0:
            cap_y = top_rect.bottom - 35
            draw.rectangle([top_rect.x - 5, cap_y, top_rect.x + top_rect.width + 5,
                          top_rect.bottom], fill=DARK_GREEN, outline=BLACK, width=2)
            draw.rectangle([top_rect.x - 5, cap_y, top_rect.x + 5, top_rect.bottom],
                          fill=(50, 180, 80))

        # Bottom pipe
        bottom_rect = pipe.get_bottom_rect()
        draw.rectangle([bottom_rect.x, bottom_rect.y, bottom_rect.x + bottom_rect.width,
                       bottom_rect.y + bottom_rect.height], fill=GREEN)
        draw.rectangle([bottom_rect.x, bottom_rect.y, bottom_rect.x + 8,
                       bottom_rect.y + bottom_rect.height], fill=(50, 200, 100))
        draw.rectangle([bottom_rect.right - 8, bottom_rect.y, bottom_rect.right,
                       bottom_rect.y + bottom_rect.height], fill=DARK_GREEN)
        draw.rectangle([bottom_rect.x, bottom_rect.y, bottom_rect.x + bottom_rect.width,
                       bottom_rect.y + bottom_rect.height], outline=BLACK, width=2)

        # Bottom pipe cap
        draw.rectangle([bottom_rect.x - 5, bottom_rect.y, bottom_rect.x + bottom_rect.width + 5,
                      bottom_rect.y + 35], fill=DARK_GREEN, outline=BLACK, width=2)
        draw.rectangle([bottom_rect.x - 5, bottom_rect.y, bottom_rect.x + 5,
                      bottom_rect.y + 35], fill=(50, 180, 80))

    # Draw ground with texture
    draw.rectangle([0, 500, 400, 600], fill=GROUND_COLOR)
    for x in range(0, 400, 40):
        draw.rectangle([x, 500, x + 20, 600], fill=GROUND_DARK)
    draw.line([(0, 500), (400, 500)], fill=GROUND_DARK, width=4)

    # Draw bird (improved design)
    bird_rect = env.bird.get_rect()

    # Bird body
    draw.ellipse([bird_rect.x, bird_rect.y, bird_rect.x + bird_rect.width,
                  bird_rect.y + bird_rect.height], fill=YELLOW, outline=ORANGE, width=3)

    # Belly highlight
    belly_y = bird_rect.centery
    draw.ellipse([bird_rect.x + 5, belly_y, bird_rect.right - 5,
                  bird_rect.bottom - 5], fill=(255, 240, 100))

    # Wing
    wing_points = [
        (bird_rect.centerx - 5, bird_rect.centery),
        (bird_rect.left - 8, bird_rect.centery - 5),
        (bird_rect.left - 3, bird_rect.centery + 5)
    ]
    draw.polygon(wing_points, fill=ORANGE, outline=(200, 120, 0))

    # Eye
    eye_x = int(bird_rect.x + bird_rect.width * 0.65)
    eye_y = int(bird_rect.y + bird_rect.height * 0.35)
    draw.ellipse([eye_x - 6, eye_y - 6, eye_x + 6, eye_y + 6], fill=WHITE)
    draw.ellipse([eye_x - 3, eye_y - 3, eye_x + 5, eye_y + 5], fill=BLACK)
    draw.ellipse([eye_x, eye_y - 2, eye_x + 4, eye_y + 2], fill=WHITE)

    # Beak
    beak_points = [
        (bird_rect.right - 2, bird_rect.centery - 5),
        (bird_rect.right + 12, bird_rect.centery),
        (bird_rect.right - 2, bird_rect.centery + 5)
    ]
    draw.polygon(beak_points, fill=ORANGE, outline=(200, 100, 0))

    # Score display with gold background
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 42)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Score background
    draw.rounded_rectangle([5, 5, 155, 55], radius=10, fill=(50, 50, 50, 200),
                          outline=GOLD, width=3)
    draw.text((12, 10), f"Score: {score}", fill=GOLD, font=font)

    # Episode info
    draw.rounded_rectangle([5, 60, 235, 95], radius=8, fill=(50, 50, 50, 200),
                          outline=WHITE, width=2)
    draw.text((10, 65), f"Episode {episode_num} Agent", fill=WHITE, font=small_font)

    # AI indicator
    draw.rounded_rectangle([240, 5, 395, 40], radius=8, fill=(50, 50, 50, 200),
                          outline=(0, 255, 0), width=2)
    draw.text((250, 10), "AI Playing", fill=(0, 255, 0), font=small_font)

    return np.array(img)


def load_agent_from_checkpoint(checkpoint_path):
    """Load agent from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    n_observations = 6
    n_actions = 2

    agent = DQNAgent(n_observations, n_actions, config)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

    return agent, checkpoint['episode']


def generate_video(checkpoint_path, output_path, max_frames=2000):
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
        # Capture frame every 2 frames to reduce file size
        if frame_count % 2 == 0:
            frame = render_frame_improved(env, score, episode_num)
            frames.append(frame)

        # Select action
        action = agent.select_action(state, training=False)

        # Step environment
        observation, reward, terminated, truncated, info = env.step(action.item())
        score = info['score']

        frame_count += 1

        if terminated or truncated:
            # Add final frames
            final_frame = render_frame_improved(env, score, episode_num)
            for _ in range(10):
                frames.append(final_frame)
            break

        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    env.close()

    # Save as GIF
    imageio.mimsave(output_path, frames, fps=15, loop=0)
    print(f"✓ Saved: {output_path} | Score: {score} | Frames: {len(frames)}")

    return score, len(frames)


def main():
    """Generate videos for key checkpoints"""
    checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_ep*.pt"))

    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"\n{'='*70}")
    print(f"GENERATING VIDEOS WITH IMPROVED GRAPHICS")
    print(f"{'='*70}")
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Output directory: {VIDEO_DIR}\n")

    # Generate videos for MANY episodes so Streamlit app shows them
    # Every 100 episodes + key best performing ones
    key_episodes = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899,
                    999, 1099, 1199, 1299, 1399, 1499, 1599, 1699, 1799, 1899,
                    1999, 2099, 2199, 2299, 2399, 2499, 2599, 2699, 2799, 2899, 2999]

    scores = []

    for checkpoint_path in checkpoints:
        episode_num = int(checkpoint_path.stem.split('ep')[1])

        if episode_num not in key_episodes:
            continue

        output_path = VIDEO_DIR / f"agent_ep{episode_num:04d}.gif"

        try:
            score, frames = generate_video(checkpoint_path, output_path, max_frames=2000)
            scores.append((episode_num, score))
        except Exception as e:
            print(f"✗ Error for episode {episode_num}: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"VIDEO GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nGenerated {len(scores)} videos:")
    for ep, score in scores:
        print(f"  Episode {ep:4d}: {score:2d} pipes")

    print(f"\nVideos saved to: {VIDEO_DIR}/")
    print(f"These videos show the IMPROVED GRAPHICS!")
    print(f"Watch the agent progress from 0 pipes to 20+ pipes!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
