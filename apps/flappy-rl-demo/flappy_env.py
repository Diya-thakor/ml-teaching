"""
Flappy Bird Gymnasium Environment
A custom Gym environment for training RL agents on Flappy Bird
"""
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from typing import Optional, Tuple

# Game constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
SPEED = 15  # Reduced from 20 - gentler flap
GRAVITY = 1.5  # Reduced from 2.5 - bird falls slower
GAME_SPEED = 6  # Reduced from 10 - pipes move slower
GROUND_WIDTH = 2 * SCREEN_WIDTH
GROUND_HEIGHT = 100
PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_GAP = 200  # Increased from 150 - MUCH bigger gaps!
FPS = 30


class Bird:
    """The Flappy Bird"""
    def __init__(self):
        self.x = SCREEN_WIDTH / 6
        self.y = SCREEN_HEIGHT / 2
        self.speed = SPEED
        self.width = 34
        self.height = 24

    def update(self):
        """Update bird position"""
        self.speed += GRAVITY
        self.y += self.speed

    def flap(self):
        """Make the bird flap (jump)"""
        self.speed = -SPEED

    def get_rect(self):
        """Get bird bounding box"""
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Pipe:
    """A pipe obstacle"""
    def __init__(self, x, gap_y):
        self.x = x
        self.gap_y = gap_y  # Y position of gap center
        self.width = PIPE_WIDTH
        self.scored = False

    def update(self):
        """Move pipe left"""
        self.x -= GAME_SPEED

    def get_top_rect(self):
        """Get top pipe bounding box"""
        return pygame.Rect(self.x, 0, self.width, self.gap_y - PIPE_GAP // 2)

    def get_bottom_rect(self):
        """Get bottom pipe bounding box"""
        bottom_y = self.gap_y + PIPE_GAP // 2
        return pygame.Rect(self.x, bottom_y, self.width, SCREEN_HEIGHT - bottom_y)

    def is_off_screen(self):
        """Check if pipe is completely off screen"""
        return self.x < -self.width


class FlappyBirdEnv(gym.Env):
    """
    Flappy Bird Environment compatible with Gymnasium

    Observation Space:
        - bird_y: vertical position (normalized 0-1)
        - bird_velocity: vertical velocity (normalized)
        - next_pipe_x: horizontal distance to next pipe (normalized)
        - next_pipe_gap_y: vertical position of next pipe gap (normalized)
        - next_pipe_top: distance to top pipe (normalized)
        - next_pipe_bottom: distance to bottom pipe (normalized)

    Action Space:
        - 0: Do nothing
        - 1: Flap

    Reward:
        - +1 for each frame survived
        - +10 for passing through a pipe
        - -100 for collision (game over)
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        # Action space: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)

        # Observation space: 6 features (all normalized 0-1)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Game state
        self.bird = None
        self.pipes = []
        self.score = 0
        self.frames = 0

        # Load pygame if rendering
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Flappy Bird RL')
            self.clock = pygame.time.Clock()
            self._load_graphics()

    def _load_graphics(self):
        """Load game graphics (only when rendering)"""
        try:
            self.bg_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.bg_surface.fill((135, 206, 235))  # Sky blue

            self.bird_surface = pygame.Surface((34, 24))
            self.bird_surface.fill((255, 255, 0))  # Yellow bird

            self.pipe_surface = pygame.Surface((PIPE_WIDTH, PIPE_HEIGHT))
            self.pipe_surface.fill((0, 200, 0))  # Green pipe

            self.ground_surface = pygame.Surface((SCREEN_WIDTH, GROUND_HEIGHT))
            self.ground_surface.fill((222, 216, 149))  # Ground color
        except:
            pass

    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        # Find the next pipe (first pipe that hasn't been passed)
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + pipe.width > self.bird.x:
                next_pipe = pipe
                break

        if next_pipe is None:
            # No pipe ahead (shouldn't happen in practice)
            next_pipe_x = SCREEN_WIDTH
            next_pipe_gap_y = SCREEN_HEIGHT / 2
        else:
            next_pipe_x = next_pipe.x
            next_pipe_gap_y = next_pipe.gap_y

        # Normalize all values to 0-1 range
        obs = np.array([
            self.bird.y / SCREEN_HEIGHT,  # Bird vertical position
            (self.bird.speed + 20) / 40,  # Bird velocity (normalized from -20 to +20)
            (next_pipe_x - self.bird.x) / SCREEN_WIDTH,  # Horizontal distance to pipe
            next_pipe_gap_y / SCREEN_HEIGHT,  # Pipe gap vertical position
            (next_pipe_gap_y - PIPE_GAP // 2 - self.bird.y) / SCREEN_HEIGHT,  # Distance to top pipe
            (self.bird.y - (next_pipe_gap_y + PIPE_GAP // 2)) / SCREEN_HEIGHT,  # Distance to bottom pipe
        ], dtype=np.float32)

        # Clip to valid range
        obs = np.clip(obs, 0, 1)

        return obs

    def _check_collision(self) -> bool:
        """Check if bird collided with pipe or ground/ceiling"""
        bird_rect = self.bird.get_rect()

        # Check ground/ceiling collision
        if self.bird.y <= 0 or self.bird.y >= SCREEN_HEIGHT - GROUND_HEIGHT:
            return True

        # Check pipe collision
        for pipe in self.pipes:
            if bird_rect.colliderect(pipe.get_top_rect()) or \
               bird_rect.colliderect(pipe.get_bottom_rect()):
                return True

        return False

    def _spawn_pipe(self):
        """Spawn a new pipe"""
        gap_y = random.randint(150, SCREEN_HEIGHT - GROUND_HEIGHT - 150)
        pipe = Pipe(SCREEN_WIDTH + 100, gap_y)
        self.pipes.append(pipe)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment"""
        super().reset(seed=seed)

        # Initialize bird
        self.bird = Bird()

        # Initialize pipes
        self.pipes = []
        for i in range(3):
            gap_y = random.randint(150, SCREEN_HEIGHT - GROUND_HEIGHT - 150)
            pipe = Pipe(SCREEN_WIDTH + i * 250, gap_y)
            self.pipes.append(pipe)

        self.score = 0
        self.frames = 0

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment"""
        self.frames += 1

        # Apply action
        if action == 1:
            self.bird.flap()

        # Update bird
        self.bird.update()

        # Update pipes
        for pipe in self.pipes:
            pipe.update()

            # Check if bird passed pipe (for scoring)
            if not pipe.scored and pipe.x + pipe.width < self.bird.x:
                pipe.scored = True
                self.score += 1

        # Remove off-screen pipes and spawn new ones
        self.pipes = [p for p in self.pipes if not p.is_off_screen()]
        if len(self.pipes) < 3:
            self._spawn_pipe()

        # Check collision
        collision = self._check_collision()

        # Calculate reward
        if collision:
            reward = -100
            terminated = True
        else:
            reward = 0.1  # Small reward for surviving
            terminated = False

            # Bonus reward for passing pipe
            if any(p.scored and p.x + p.width >= self.bird.x - GAME_SPEED for p in self.pipes):
                reward += 10

        truncated = False  # No time limit

        observation = self._get_observation()
        info = {'score': self.score, 'frames': self.frames}

        # Render if needed
        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self._load_graphics()

        # Draw background
        self.screen.blit(self.bg_surface, (0, 0))

        # Draw pipes
        for pipe in self.pipes:
            # Top pipe
            top_rect = pipe.get_top_rect()
            self.screen.blit(
                pygame.transform.scale(self.pipe_surface, (top_rect.width, top_rect.height)),
                (top_rect.x, top_rect.y)
            )
            # Bottom pipe
            bottom_rect = pipe.get_bottom_rect()
            self.screen.blit(
                pygame.transform.scale(self.pipe_surface, (bottom_rect.width, bottom_rect.height)),
                (bottom_rect.x, bottom_rect.y)
            )

        # Draw ground
        self.screen.blit(self.ground_surface, (0, SCREEN_HEIGHT - GROUND_HEIGHT))

        # Draw bird
        self.screen.blit(self.bird_surface, (self.bird.x, self.bird.y))

        # Draw score
        font = pygame.font.Font(None, 48)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()

        if self.clock:
            self.clock.tick(FPS)

    def close(self):
        """Clean up resources"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
