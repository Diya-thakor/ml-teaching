"""
Play Flappy Bird - Manual Play or Watch AI
Choose to play yourself or watch a trained AI agent
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
BLUE = (50, 150, 255)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_sound(frequency, duration=100):
    """Create a simple beep sound"""
    sample_rate = 22050
    n_samples = int(sample_rate * duration / 1000)
    buf = np.sin(2 * np.pi * np.arange(n_samples) * frequency / sample_rate)
    buf = (buf * 32767).astype(np.int16)
    buf = np.repeat(buf.reshape(n_samples, 1), 2, axis=1)
    sound = pygame.sndarray.make_sound(buf)
    return sound


# Create sound effects
try:
    flap_sound = create_sound(800, 100)  # High beep for flap
    score_sound = create_sound(1200, 150)  # Higher beep for score
    crash_sound = create_sound(200, 300)  # Low beep for crash
    countdown_sound = create_sound(600, 200)  # Medium beep for countdown
except:
    flap_sound = None
    score_sound = None
    crash_sound = None
    countdown_sound = None


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
    print(f"✓ Loaded AI agent from Episode {episode}")
    return agent, episode


def draw_game(screen, env, score, mode, episode_num, font, small_font,
              last_action=None, game_over=False):
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
        pygame.draw.rect(screen, GREEN, top_rect)
        highlight_rect = pygame.Rect(top_rect.x, top_rect.y, 8, top_rect.height)
        pygame.draw.rect(screen, (50, 200, 100), highlight_rect)
        shadow_rect = pygame.Rect(top_rect.right - 8, top_rect.y, 8, top_rect.height)
        pygame.draw.rect(screen, DARK_GREEN, shadow_rect)
        pygame.draw.rect(screen, BLACK, top_rect, 2)

        # Bottom pipe
        bottom_rect = pipe.get_bottom_rect()
        pygame.draw.rect(screen, GREEN, bottom_rect)
        highlight_rect = pygame.Rect(bottom_rect.x, bottom_rect.y, 8, bottom_rect.height)
        pygame.draw.rect(screen, (50, 200, 100), highlight_rect)
        shadow_rect = pygame.Rect(bottom_rect.right - 8, bottom_rect.y, 8, bottom_rect.height)
        pygame.draw.rect(screen, DARK_GREEN, shadow_rect)
        pygame.draw.rect(screen, BLACK, bottom_rect, 2)

        # Pipe caps
        cap_height = 35
        cap_width = top_rect.width + 10
        if top_rect.height > 0:
            cap_rect = pygame.Rect(top_rect.x - 5, top_rect.bottom - cap_height,
                                   cap_width, cap_height)
            pygame.draw.rect(screen, DARK_GREEN, cap_rect)
            pygame.draw.rect(screen, BLACK, cap_rect, 2)
            pygame.draw.rect(screen, (50, 180, 80), (cap_rect.x, cap_rect.y, 10, cap_height))

        bottom_cap_rect = pygame.Rect(bottom_rect.x - 5, bottom_rect.top, cap_width, cap_height)
        pygame.draw.rect(screen, DARK_GREEN, bottom_cap_rect)
        pygame.draw.rect(screen, BLACK, bottom_cap_rect, 2)
        pygame.draw.rect(screen, (50, 180, 80),
                       (bottom_cap_rect.x, bottom_cap_rect.y, 10, cap_height))

    # Draw ground with texture
    pygame.draw.rect(screen, GROUND_COLOR, (0, 500, SCREEN_WIDTH, 100))
    for x in range(0, SCREEN_WIDTH, 40):
        pygame.draw.rect(screen, GROUND_DARK, (x, 500, 20, 100))
    pygame.draw.line(screen, GROUND_DARK, (0, 500), (SCREEN_WIDTH, 500), 4)

    # Draw bird with improved design
    bird_rect = env.bird.get_rect()

    # Bird body
    pygame.draw.ellipse(screen, YELLOW, bird_rect)

    # Wing (animated based on action)
    wing_offset = -3 if last_action == 1 else 5
    wing_points = [
        (bird_rect.centerx - 5, bird_rect.centery + wing_offset),
        (bird_rect.left - 8, bird_rect.centery - 5 + wing_offset),
        (bird_rect.left - 3, bird_rect.centery + 5 + wing_offset)
    ]
    pygame.draw.polygon(screen, ORANGE, wing_points)
    pygame.draw.polygon(screen, (200, 120, 0), wing_points, 2)

    # Bird outline
    pygame.draw.ellipse(screen, ORANGE, bird_rect, 3)

    # Belly highlight
    belly_rect = pygame.Rect(bird_rect.x + 5, bird_rect.centery,
                             bird_rect.width - 10, bird_rect.height // 2 - 5)
    pygame.draw.ellipse(screen, (255, 240, 100), belly_rect)

    # Eye
    eye_x = bird_rect.x + bird_rect.width * 0.65
    eye_y = bird_rect.y + bird_rect.height * 0.35
    pygame.draw.circle(screen, WHITE, (int(eye_x), int(eye_y)), 6)
    pygame.draw.circle(screen, BLACK, (int(eye_x + 1), int(eye_y)), 4)
    pygame.draw.circle(screen, WHITE, (int(eye_x + 2), int(eye_y - 1)), 2)

    # Beak
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

    # Score display
    score_bg = pygame.Rect(5, 5, 150, 50)
    pygame.draw.rect(screen, (50, 50, 50, 200), score_bg, border_radius=10)
    pygame.draw.rect(screen, GOLD, score_bg, 3, border_radius=10)
    score_text = font.render(f"Score: {score}", True, GOLD)
    score_shadow = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_shadow, (12, 12))
    screen.blit(score_text, (10, 10))

    # Mode indicator
    if mode == "human":
        mode_text = "YOU Playing"
        mode_color = BLUE
        control_text = "Press SPACE to Flap!"
    else:
        mode_text = f"AI (Ep {episode_num})"
        mode_color = (0, 255, 0)
        control_text = "Watching AI Agent"

    mode_bg = pygame.Rect(5, 60, 200, 35)
    pygame.draw.rect(screen, (50, 50, 50, 200), mode_bg, border_radius=8)
    pygame.draw.rect(screen, mode_color, mode_bg, 2, border_radius=8)
    mode_display = small_font.render(mode_text, True, mode_color)
    mode_shadow = small_font.render(mode_text, True, BLACK)
    screen.blit(mode_shadow, (12, 67))
    screen.blit(mode_display, (10, 65))

    # Control/Action indicator
    control_bg = pygame.Rect(5, 100, 250, 30)
    pygame.draw.rect(screen, (50, 50, 50, 200), control_bg, border_radius=8)

    if last_action == 1:
        # Flapping!
        action_color = GOLD
        action_text = "🔥 FLAP! 🔥"
        pygame.draw.rect(screen, action_color, control_bg, 2, border_radius=8)
    else:
        action_color = WHITE
        action_text = control_text
        pygame.draw.rect(screen, (100, 100, 100), control_bg, 2, border_radius=8)

    action_display = small_font.render(action_text, True, action_color)
    screen.blit(action_display, (12, 105))

    # Game over overlay
    if game_over:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))

        game_over_font = pygame.font.Font(None, 72)
        go_text = game_over_font.render("GAME OVER", True, RED)
        go_rect = go_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80))
        screen.blit(go_text, go_rect)

        final_score_text = font.render(f"Final Score: {score}", True, GOLD)
        score_rect = final_score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 10))
        screen.blit(final_score_text, score_rect)

        if mode == "human":
            feedback = "Great effort!" if score >= 5 else "Keep trying!" if score >= 1 else "Practice makes perfect!"
            feedback_text = small_font.render(feedback, True, WHITE)
            feedback_rect = feedback_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
            screen.blit(feedback_text, feedback_rect)

        restart_text = small_font.render("Press SPACE to restart or Q to quit", True, WHITE)
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
        screen.blit(restart_text, restart_rect)


def show_countdown(screen, env, font, small_font):
    """Show 3-2-1-GO countdown"""
    countdown_font = pygame.font.Font(None, 120)

    for count in [3, 2, 1, "GO!"]:
        # Draw game state in background
        draw_game(screen, env, 0, "countdown", 0, font, small_font, 0, False)

        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))

        # Countdown number/text
        if isinstance(count, int):
            text = countdown_font.render(str(count), True, GOLD)
            color = GOLD
        else:
            text = countdown_font.render(count, True, GREEN)
            color = GREEN

        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))

        # Animated pulse effect
        for frame in range(30):  # 1 second at 30 FPS
            screen.blit(overlay, (0, 0))

            # Scale effect
            scale = 1.0 + (0.3 * (1 - frame / 30))
            scaled_font_size = int(120 * scale)
            pulse_font = pygame.font.Font(None, scaled_font_size)
            pulse_text = pulse_font.render(str(count) if isinstance(count, int) else count, True, color)
            pulse_rect = pulse_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))

            screen.blit(pulse_text, pulse_rect)
            pygame.display.flip()
            pygame.time.Clock().tick(30)

        # Play sound
        if countdown_sound and count != "GO!":
            countdown_sound.play()
        elif countdown_sound and count == "GO!":
            if flap_sound:
                flap_sound.play()


def select_mode():
    """Choose play mode"""
    print("\n" + "="*60)
    print("FLAPPY BIRD - Play Mode Selection")
    print("="*60)
    print("1. Play Yourself (Human)")
    print("2. Watch AI Agent")
    print("="*60)

    while True:
        choice = input("\nSelect mode (1 or 2): ").strip()
        if choice == "1":
            return "human", None, None
        elif choice == "2":
            # Select AI checkpoint - check directories in order of preference
            for dir_name in ["checkpoints_better", "checkpoints_demo", "checkpoints"]:
                checkpoint_dir = Path(dir_name)
                if checkpoint_dir.exists() and list(checkpoint_dir.glob("checkpoint_ep*.pt")):
                    break
            else:
                print("ERROR: No checkpoints found! Train an agent first.")
                print("To get started quickly, you can use the pre-trained demo checkpoints.")
                sys.exit(1)

            checkpoints = sorted(checkpoint_dir.glob("checkpoint_ep*.pt"))
            if not checkpoints:
                print("ERROR: No checkpoints found!")
                sys.exit(1)

            print("\nAvailable AI Agents:")
            checkpoint_list = []
            for i, cp in enumerate(checkpoints):
                episode = int(cp.stem.split('ep')[1])
                checkpoint_list.append((episode, cp))
                desc = "Untrained" if episode == 0 else "Learning" if episode < 500 else "Advanced" if episode < 1500 else "Expert"
                print(f"{i+1}. Episode {episode:4d} - {desc}")

            while True:
                try:
                    ai_choice = input("\nSelect AI checkpoint number: ").strip()
                    idx = int(ai_choice) - 1
                    if 0 <= idx < len(checkpoint_list):
                        agent, episode = load_agent(checkpoint_list[idx][1])
                        return "ai", agent, episode
                except ValueError:
                    print("Invalid input.")
        else:
            print("Please enter 1 or 2")


def main():
    """Main game loop"""
    # Select mode
    mode, agent, episode_num = select_mode()

    # Setup display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    if mode == "human":
        pygame.display.set_caption("Flappy Bird - YOU Play!")
    else:
        pygame.display.set_caption(f"Flappy Bird - AI Agent (Episode {episode_num})")

    clock = pygame.time.Clock()

    # Fonts
    try:
        font = pygame.font.Font(None, 48)
        small_font = pygame.font.Font(None, 24)
    except:
        font = pygame.font.SysFont('arial', 48)
        small_font = pygame.font.SysFont('arial', 24)

    # Create environment
    env = FlappyBirdEnv()

    # Game state
    running = True
    playing = True
    game_over = False
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) if mode == "ai" else None
    score = 0
    last_action = 0
    action_cooldown = 0

    total_games = 0
    total_score = 0
    best_score = 0

    print("\n" + "="*60)
    print("CONTROLS:")
    if mode == "human":
        print("  SPACE / UP  - Flap (jump)")
    print("  SPACE       - Restart game (when game over)")
    print("  P           - Pause/Resume")
    print("  M           - Return to mode selection")
    print("  Q           - Quit")
    print("="*60)
    if mode == "human":
        print("\nTIP: Timing is everything! Tap space to flap through gaps.")
    else:
        print(f"\nWatching AI agent from Episode {episode_num}...")
    print()

    paused = False
    prev_score = 0  # Track score changes for sound

    # Show countdown before first game
    show_countdown(screen, env, font, small_font)

    while running:
        clock.tick(FPS)

        # Decay action indicator
        if action_cooldown > 0:
            action_cooldown -= 1
        if action_cooldown == 0:
            last_action = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_m:
                    # Return to mode selection
                    mode, agent, episode_num = select_mode()
                    state, _ = env.reset()
                    state_tensor = torch.tensor(state, dtype=torch.float32,
                                               device=device).unsqueeze(0) if mode == "ai" else None
                    score = 0
                    playing = True
                    game_over = False
                elif event.key == pygame.K_SPACE:
                    if game_over:
                        # Restart
                        state, _ = env.reset()
                        state_tensor = torch.tensor(state, dtype=torch.float32,
                                                   device=device).unsqueeze(0) if mode == "ai" else None
                        score = 0
                        prev_score = 0
                        last_action = 0

                        # Show countdown
                        show_countdown(screen, env, font, small_font)

                        playing = True
                        game_over = False
                    elif mode == "human" and playing and not paused:
                        # Human flap
                        last_action = 1
                        action_cooldown = 5
                        if flap_sound:
                            flap_sound.play()
                elif event.key == pygame.K_UP and mode == "human" and playing and not paused:
                    # Alternative flap key
                    last_action = 1
                    action_cooldown = 5
                    if flap_sound:
                        flap_sound.play()
                elif event.key == pygame.K_p:
                    paused = not paused

        # Game logic
        if playing and not game_over and not paused:
            # Determine action
            if mode == "human":
                action = 1 if last_action == 1 and action_cooldown == 5 else 0
            else:
                # AI action
                with torch.no_grad():
                    action_tensor = agent.select_action(state_tensor, training=False)
                    action = action_tensor.item()
                    last_action = action
                    if action == 1:
                        action_cooldown = 5
                        if flap_sound:
                            flap_sound.play()

            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            score = info['score']

            # Play score sound when score increases
            if score > prev_score:
                if score_sound:
                    score_sound.play()
                prev_score = score

            if score > best_score:
                best_score = score

            if terminated or truncated:
                playing = False
                game_over = True
                total_games += 1
                total_score += score

                # Play crash sound
                if crash_sound:
                    crash_sound.play()

                if mode == "human":
                    print(f"Game Over! Score: {score}, Best: {best_score}, Avg: {total_score/total_games:.1f}")
                else:
                    print(f"AI Game Over! Score: {score}")
            else:
                if mode == "ai":
                    state_tensor = torch.tensor(observation, dtype=torch.float32,
                                               device=device).unsqueeze(0)

        # Draw everything
        draw_game(screen, env, score, mode, episode_num if mode == "ai" else 0,
                 font, small_font, last_action, game_over)

        # Pause indicator
        if paused:
            pause_font = pygame.font.Font(None, 64)
            pause_text = pause_font.render("PAUSED", True, WHITE)
            pause_rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            pause_bg = pygame.Rect(pause_rect.x - 20, pause_rect.y - 10,
                                   pause_rect.width + 40, pause_rect.height + 20)
            pygame.draw.rect(screen, BLACK, pause_bg, border_radius=10)
            pygame.draw.rect(screen, WHITE, pause_bg, 3, border_radius=10)
            screen.blit(pause_text, pause_rect)

        pygame.display.flip()

    # Cleanup
    env.close()
    pygame.quit()

    if total_games > 0:
        print("\n" + "="*60)
        print("FINAL STATISTICS:")
        print(f"  Mode: {'Human' if mode == 'human' else f'AI (Episode {episode_num})'}")
        print(f"  Total Games: {total_games}")
        print(f"  Total Score: {total_score}")
        print(f"  Best Score: {best_score}")
        print(f"  Average Score: {total_score / total_games:.2f}")
        print("="*60)


if __name__ == "__main__":
    main()
