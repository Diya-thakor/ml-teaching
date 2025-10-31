"""
Play Flappy Bird with a trained agent using Pygame
Load agent weights and watch it play in real-time!
"""
import pygame
import torch
import numpy as np
from pathlib import Path
import sys
import math
from flappy_env import FlappyBirdEnv
from train_agent import DQNAgent

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Display settings
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SKY_BLUE = (135, 206, 250)
SKY_LIGHT = (180, 220, 255)
GREEN = (34, 177, 76)
DARK_GREEN = (25, 130, 50)
YELLOW = (255, 220, 0)
ORANGE = (255, 140, 0)
GROUND_COLOR = (222, 216, 149)
GROUND_DARK = (180, 160, 100)
RED = (255, 50, 50)
GOLD = (255, 215, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load or create sounds
try:
    # Try to create simple beep sounds
    flap_sound = None
    score_sound = None
    crash_sound = None
except:
    flap_sound = None
    score_sound = None
    crash_sound = None


def load_agent(checkpoint_path):
    """Load agent from checkpoint"""
    print(f"Loading agent from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    n_observations = 6
    n_actions = 2

    agent = DQNAgent(n_observations, n_actions, config)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

    episode = checkpoint['episode']
    print(f"✓ Loaded checkpoint from Episode {episode}")
    return agent, episode


def draw_game(screen, env, score, episode_num, font, small_font, game_over=False):
    """Draw the game state with enhanced visuals"""
    # Gradient background (sky)
    for y in range(0, 500, 2):
        color_ratio = y / 500
        r = int(SKY_BLUE[0] + (SKY_LIGHT[0] - SKY_BLUE[0]) * color_ratio)
        g = int(SKY_BLUE[1] + (SKY_LIGHT[1] - SKY_BLUE[1]) * color_ratio)
        b = int(SKY_BLUE[2] + (SKY_LIGHT[2] - SKY_BLUE[2]) * color_ratio)
        pygame.draw.line(screen, (r, g, b), (0, y), (SCREEN_WIDTH, y), 2)

    # Draw pipes with 3D effect
    for pipe in env.pipes:
        # Top pipe
        top_rect = pipe.get_top_rect()
        # Main pipe body
        pygame.draw.rect(screen, GREEN, top_rect)
        # Left highlight
        highlight_rect = pygame.Rect(top_rect.x, top_rect.y, 8, top_rect.height)
        pygame.draw.rect(screen, (50, 200, 100), highlight_rect)
        # Right shadow
        shadow_rect = pygame.Rect(top_rect.right - 8, top_rect.y, 8, top_rect.height)
        pygame.draw.rect(screen, DARK_GREEN, shadow_rect)
        # Border
        pygame.draw.rect(screen, BLACK, top_rect, 2)

        # Bottom pipe
        bottom_rect = pipe.get_bottom_rect()
        pygame.draw.rect(screen, GREEN, bottom_rect)
        highlight_rect = pygame.Rect(bottom_rect.x, bottom_rect.y, 8, bottom_rect.height)
        pygame.draw.rect(screen, (50, 200, 100), highlight_rect)
        shadow_rect = pygame.Rect(bottom_rect.right - 8, bottom_rect.y, 8, bottom_rect.height)
        pygame.draw.rect(screen, DARK_GREEN, shadow_rect)
        pygame.draw.rect(screen, BLACK, bottom_rect, 2)

        # Draw pipe caps (3D effect)
        cap_height = 35
        cap_width = top_rect.width + 10
        if top_rect.height > 0:
            cap_rect = pygame.Rect(top_rect.x - 5, top_rect.bottom - cap_height,
                                   cap_width, cap_height)
            pygame.draw.rect(screen, DARK_GREEN, cap_rect)
            pygame.draw.rect(screen, BLACK, cap_rect, 2)
            # Cap highlight
            pygame.draw.rect(screen, (50, 180, 80),
                           (cap_rect.x, cap_rect.y, 10, cap_height))

        bottom_cap_rect = pygame.Rect(bottom_rect.x - 5, bottom_rect.top,
                                       cap_width, cap_height)
        pygame.draw.rect(screen, DARK_GREEN, bottom_cap_rect)
        pygame.draw.rect(screen, BLACK, bottom_cap_rect, 2)
        pygame.draw.rect(screen, (50, 180, 80),
                       (bottom_cap_rect.x, bottom_cap_rect.y, 10, cap_height))

    # Draw ground with texture
    pygame.draw.rect(screen, GROUND_COLOR, (0, 500, SCREEN_WIDTH, 100))
    # Add stripes
    for x in range(0, SCREEN_WIDTH, 40):
        pygame.draw.rect(screen, GROUND_DARK, (x, 500, 20, 100))
    pygame.draw.line(screen, GROUND_DARK, (0, 500), (SCREEN_WIDTH, 500), 4)

    # Draw bird with improved design
    bird_rect = env.bird.get_rect()

    # Bird body (main circle)
    pygame.draw.ellipse(screen, YELLOW, bird_rect)

    # Wing (animated flap based on velocity)
    wing_offset = 0 if env.bird.speed < 0 else 5
    wing_points = [
        (bird_rect.centerx - 5, bird_rect.centery + wing_offset),
        (bird_rect.left - 8, bird_rect.centery - 5 + wing_offset),
        (bird_rect.left - 3, bird_rect.centery + 5 + wing_offset)
    ]
    pygame.draw.polygon(screen, ORANGE, wing_points)
    pygame.draw.polygon(screen, DARK_GREEN, wing_points, 2)

    # Bird outline
    pygame.draw.ellipse(screen, ORANGE, bird_rect, 3)

    # Belly highlight
    belly_rect = pygame.Rect(bird_rect.x + 5, bird_rect.centery,
                             bird_rect.width - 10, bird_rect.height // 2 - 5)
    pygame.draw.ellipse(screen, (255, 240, 100), belly_rect)

    # Draw bird eye
    eye_x = bird_rect.x + bird_rect.width * 0.65
    eye_y = bird_rect.y + bird_rect.height * 0.35
    # White of eye
    pygame.draw.circle(screen, WHITE, (int(eye_x), int(eye_y)), 6)
    # Black pupil
    pygame.draw.circle(screen, BLACK, (int(eye_x + 1), int(eye_y)), 4)
    # Highlight
    pygame.draw.circle(screen, WHITE, (int(eye_x + 2), int(eye_y - 1)), 2)

    # Draw beak (more detailed)
    beak_top = [
        (bird_rect.right - 2, bird_rect.centery - 5),
        (bird_rect.right + 12, bird_rect.centery - 2),
        (bird_rect.right, bird_rect.centery)
    ]
    beak_bottom = [
        (bird_rect.right, bird_rect.centery),
        (bird_rect.right + 12, bird_rect.centery + 2),
        (bird_rect.right - 2, bird_rect.centery + 5)
    ]
    pygame.draw.polygon(screen, ORANGE, beak_top)
    pygame.draw.polygon(screen, ORANGE, beak_bottom)
    pygame.draw.polygon(screen, (200, 100, 0), beak_top, 2)
    pygame.draw.polygon(screen, (200, 100, 0), beak_bottom, 2)

    # Score display with gold background
    score_bg = pygame.Rect(5, 5, 150, 50)
    pygame.draw.rect(screen, (50, 50, 50, 180), score_bg, border_radius=10)
    pygame.draw.rect(screen, GOLD, score_bg, 3, border_radius=10)

    score_text = font.render(f"Score: {score}", True, GOLD)
    score_shadow = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_shadow, (12, 12))
    screen.blit(score_text, (10, 10))

    # Episode info
    ep_bg = pygame.Rect(5, 60, 230, 35)
    pygame.draw.rect(screen, (50, 50, 50, 180), ep_bg, border_radius=8)
    pygame.draw.rect(screen, WHITE, ep_bg, 2, border_radius=8)

    ep_text = small_font.render(f"Episode {episode_num} Checkpoint", True, WHITE)
    ep_shadow = small_font.render(f"Episode {episode_num} Checkpoint", True, BLACK)
    screen.blit(ep_shadow, (12, 67))
    screen.blit(ep_text, (10, 65))

    # Agent status
    status_bg = pygame.Rect(SCREEN_WIDTH - 160, 5, 155, 35)
    pygame.draw.rect(screen, (50, 50, 50, 180), status_bg, border_radius=8)
    pygame.draw.rect(screen, (0, 255, 0), status_bg, 2, border_radius=8)

    status_text = small_font.render("AI Agent Playing", True, (0, 255, 0))
    status_shadow = small_font.render("AI Agent Playing", True, BLACK)
    screen.blit(status_shadow, (SCREEN_WIDTH - 152, 12))
    screen.blit(status_text, (SCREEN_WIDTH - 150, 10))

    # Game over overlay
    if game_over:
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))

        # Game over text
        game_over_font = pygame.font.Font(None, 72)
        go_text = game_over_font.render("GAME OVER", True, RED)
        go_rect = go_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        screen.blit(go_text, go_rect)

        # Final score
        final_score_text = font.render(f"Final Score: {score}", True, WHITE)
        score_rect = final_score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
        screen.blit(final_score_text, score_rect)

        # Instructions
        restart_text = small_font.render("Press SPACE to restart or Q to quit", True, WHITE)
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
        screen.blit(restart_text, restart_rect)


def select_checkpoint():
    """Interactive checkpoint selection"""
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("ERROR: No checkpoints directory found!")
        print("Please run 'python train_agent.py' first to train an agent.")
        sys.exit(1)

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_ep*.pt"))
    if not checkpoints:
        print("ERROR: No checkpoints found!")
        print("Please run 'python train_agent.py' first.")
        sys.exit(1)

    print("\n" + "="*60)
    print("Available Checkpoints:")
    print("="*60)

    checkpoint_list = []
    for i, cp in enumerate(checkpoints):
        episode = int(cp.stem.split('ep')[1])
        checkpoint_list.append((episode, cp))

        # Add descriptions
        if episode == 0:
            desc = "Untrained (random)"
        elif episode < 100:
            desc = "Early learning"
        elif episode < 500:
            desc = "Getting better"
        elif episode < 1000:
            desc = "Competent"
        elif episode < 1500:
            desc = "Advanced"
        else:
            desc = "Expert"

        print(f"{i+1}. Episode {episode:4d} - {desc}")

    print("="*60)

    while True:
        try:
            choice = input("\nSelect checkpoint number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                sys.exit(0)

            idx = int(choice) - 1
            if 0 <= idx < len(checkpoint_list):
                return checkpoint_list[idx][1]
            else:
                print(f"Please enter a number between 1 and {len(checkpoint_list)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    """Main game loop"""
    # Select checkpoint
    checkpoint_path = select_checkpoint()

    # Load agent
    agent, episode_num = load_agent(checkpoint_path)

    # Setup display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Flappy Bird RL Agent (Episode {episode_num})")
    clock = pygame.time.Clock()

    # Fonts
    try:
        font = pygame.font.Font(None, 48)
        small_font = pygame.font.Font(None, 28)
    except:
        font = pygame.font.SysFont('arial', 48)
        small_font = pygame.font.SysFont('arial', 28)

    # Create environment
    env = FlappyBirdEnv()

    # Game state
    running = True
    playing = True
    game_over = False
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    score = 0
    total_episodes = 0
    total_score = 0

    print("\n" + "="*60)
    print("CONTROLS:")
    print("  SPACE - Restart game")
    print("  Q     - Quit")
    print("  P     - Pause/Resume")
    print("="*60)
    print(f"\nWatching agent from Episode {episode_num}...\n")

    paused = False

    while running:
        clock.tick(FPS)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Restart
                    if game_over:
                        state, _ = env.reset()
                        state_tensor = torch.tensor(state, dtype=torch.float32,
                                                   device=device).unsqueeze(0)
                        score = 0
                        playing = True
                        game_over = False
                    else:
                        # Manual restart during play
                        state, _ = env.reset()
                        state_tensor = torch.tensor(state, dtype=torch.float32,
                                                   device=device).unsqueeze(0)
                        score = 0
                        playing = True
                        game_over = False
                elif event.key == pygame.K_p:
                    paused = not paused
                    if paused:
                        print("PAUSED")
                    else:
                        print("RESUMED")

        # Game logic
        if playing and not game_over and not paused:
            # Agent selects action
            with torch.no_grad():
                action = agent.select_action(state_tensor, training=False)

            # Step environment
            observation, reward, terminated, truncated, info = env.step(action.item())
            score = info['score']

            if terminated or truncated:
                playing = False
                game_over = True
                total_episodes += 1
                total_score += score
                avg_score = total_score / total_episodes

                print(f"Episode ended: Score = {score}, Avg = {avg_score:.2f}")

            else:
                state_tensor = torch.tensor(observation, dtype=torch.float32,
                                           device=device).unsqueeze(0)

        # Draw everything
        draw_game(screen, env, score, episode_num, font, small_font, game_over)

        # Pause indicator
        if paused:
            pause_font = pygame.font.Font(None, 64)
            pause_text = pause_font.render("PAUSED", True, WHITE)
            pause_rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            pause_shadow = pause_font.render("PAUSED", True, BLACK)
            pause_shadow_rect = pause_shadow.get_rect(center=(SCREEN_WIDTH // 2 + 2,
                                                               SCREEN_HEIGHT // 2 + 2))
            screen.blit(pause_shadow, pause_shadow_rect)
            screen.blit(pause_text, pause_rect)

        pygame.display.flip()

    # Cleanup
    env.close()
    pygame.quit()

    if total_episodes > 0:
        print("\n" + "="*60)
        print("FINAL STATISTICS:")
        print(f"  Total Episodes: {total_episodes}")
        print(f"  Total Score: {total_score}")
        print(f"  Average Score: {total_score / total_episodes:.2f}")
        print("="*60)


if __name__ == "__main__":
    main()
