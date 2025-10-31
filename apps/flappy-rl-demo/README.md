# Flappy Bird RL Demo

An interactive demonstration of Deep Q-Network (DQN) reinforcement learning on Flappy Bird. Watch as an agent learns from crashing immediately to achieving high scores!

## Features

- Custom Gymnasium environment for Flappy Bird
- DQN agent with experience replay and target networks
- Training from scratch with checkpoint saving
- Animated GIF generation showing agent behavior
- Interactive Streamlit app to visualize learning progress
- Compare performance across different training stages

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Agent

```bash
python train_agent.py
```

This will train for 2000 episodes and save checkpoints every 50 episodes in the `checkpoints/` directory.

Training takes approximately 2-4 hours on a modern CPU (faster with GPU).

### 3. Generate Videos

```bash
python generate_videos.py
```

This creates animated GIFs showing agent behavior at key checkpoints (episodes 0, 50, 100, 200, 500, 1000, 1500, 1950).

### 4. Launch the Interactive Demo

```bash
streamlit run demo_app.py
```

Open the provided URL in your browser to explore the interactive visualization!

## What You'll See

### Episode 0 (Untrained)
- Random actions
- Crashes immediately (~10 frames)
- Score: 0 pipes

### Episode 100 (Early Learning)
- Starting to understand physics
- Occasionally passes 1 pipe
- Score: 0-1 pipes

### Episode 500 (Getting Competent)
- Passes multiple pipes consistently
- Score: 3-5 pipes

### Episode 1500+ (Expert)
- Near-optimal play
- Score: 15-20+ pipes!

## Project Structure

```
flappy-rl-demo/
├── flappy_env.py         # Gymnasium environment
├── train_agent.py        # DQN training script
├── generate_videos.py    # Video generation
├── demo_app.py           # Streamlit visualization app
├── requirements.txt      # Python dependencies
├── checkpoints/          # Saved model checkpoints
├── videos/               # Generated GIF videos
└── metrics/              # Training metrics (JSON)
```

## Technical Details

### Environment

**Observation Space** (6 features, normalized 0-1):
- Bird vertical position
- Bird vertical velocity
- Horizontal distance to next pipe
- Vertical position of pipe gap
- Distance to top pipe
- Distance to bottom pipe

**Action Space**:
- 0: Do nothing (gravity pulls bird down)
- 1: Flap (jump upward)

**Rewards**:
- +0.1: Each frame survived
- +10: Passing through a pipe
- -100: Collision (game over)

### Network Architecture

- 4-layer fully connected DQN
- Layers: 256 → 256 → 128 → 2
- ReLU activations
- Experience replay buffer (50k transitions)
- Soft target network updates (τ=0.005)

### Hyperparameters

- Learning rate: 1e-4
- Batch size: 128
- Discount factor (γ): 0.99
- Epsilon: 0.9 → 0.05 (decay over 5000 steps)
- Optimizer: AdamW

## Customization

### Change Training Duration

Edit `train_agent.py`:
```python
num_episodes = 2000  # Change this value
checkpoint_interval = 50  # Save frequency
```

### Adjust Difficulty

Edit `flappy_env.py`:
```python
PIPE_GAP = 150  # Increase for easier, decrease for harder
GAME_SPEED = 15  # Pipe movement speed
GRAVITY = 2.5  # Bird fall acceleration
```

### Modify Network Architecture

Edit the `DQN` class in `train_agent.py`:
```python
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)  # Change sizes
        self.layer2 = nn.Linear(256, 256)
        # Add more layers if desired
```

## Troubleshooting

### Training is slow
- Reduce `num_episodes` in `train_agent.py`
- Use a GPU if available (CUDA)
- Decrease batch size

### Agent not learning
- Check that epsilon decay is appropriate
- Verify reward signals are working
- Try increasing learning rate slightly
- Ensure sufficient exploration (higher epsilon)

### Videos not generating
- Make sure pygame is installed
- Check that checkpoints exist
- Verify PIL/Pillow installation

## Future Enhancements

- [ ] Manual play mode with agent assistance
- [ ] Double DQN implementation
- [ ] Dueling DQN architecture
- [ ] Prioritized experience replay
- [ ] Real-time training visualization
- [ ] Pretrained model weights
- [ ] Multi-agent comparison

## References

- [DQN Paper](https://www.nature.com/articles/nature14236) - Mnih et al., 2015
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

## License

MIT License - See LICENSE file for details
