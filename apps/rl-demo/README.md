# RL Training Demo: DQN on CartPole

An interactive demonstration showing how Deep Q-Network (DQN) agents learn over time, with checkpoints, videos, and visualizations.

## 🎯 What This Demo Shows

This project demonstrates:
- **Progressive Learning**: See how an RL agent improves from random actions to optimal behavior
- **Checkpoint System**: Save model weights at different training stages
- **Video Generation**: Create GIFs showing agent performance at each checkpoint
- **Interactive Dashboard**: Compare different training stages with Streamlit

## 📁 Project Structure

```
rl-demo/
├── train_agent.py          # Main training script with checkpoint saving
├── generate_videos.py      # Create videos from saved checkpoints
├── demo_app.py            # Interactive Streamlit dashboard
├── requirements.txt       # Python dependencies
├── checkpoints/          # Saved model weights (created during training)
├── videos/              # Generated GIF videos (created by generate_videos.py)
└── metrics/             # Training metrics JSON files (created during training)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Agent

Train a DQN agent on CartPole for 500 episodes (takes ~5-10 minutes):

```bash
python train_agent.py
```

This will:
- Train the agent for 500 episodes
- Save checkpoints at episodes: 0, 50, 100, 200, 300, 400, 500
- Save training metrics for each checkpoint
- Print progress every 50 episodes

Output:
```
Training DQN on CartPole-v1
Episodes: 500
Checkpoints at: [0, 50, 100, 200, 300, 400, 500]

Episode  50 | Avg Reward:  21.34 | Avg Length:  21.34 | Steps:   957
✓ Checkpoint saved: checkpoints/checkpoint_ep0050.pt
  → Evaluation: 22.10 ± 3.45

Episode 100 | Avg Reward:  45.67 | Avg Length:  45.67 | Steps:  2341
✓ Checkpoint saved: checkpoints/checkpoint_ep0100.pt
  → Evaluation: 48.30 ± 8.12
...
```

### 3. Generate Videos

Create GIF videos showing agent behavior at each checkpoint:

```bash
python generate_videos.py
```

This will:
- Load each checkpoint
- Run 5 test episodes and pick the best one
- Create a GIF for each checkpoint
- Generate a comparison figure

Output:
```
Generating Videos from Checkpoints

Processing: checkpoint_ep0000.pt
  ✓ Video saved: videos/agent_ep0000.gif

Processing: checkpoint_ep0050.pt
  ✓ Video saved: videos/agent_ep0050.gif
...
```

### 4. Launch Interactive Demo

Run the Streamlit dashboard:

```bash
streamlit run demo_app.py
```

Then open your browser to `http://localhost:8501`

## 🎮 Using the Demo App

### Select Checkpoint
Use the sidebar to select different training stages:
- **Episode 0**: Untrained agent (random actions)
- **Episode 100**: Learning basic balance
- **Episode 300**: Competent performance
- **Episode 500**: Near-optimal behavior

### Run Test Episodes
Click "▶️ Run Test Episodes" to:
- Evaluate the agent's current performance
- See statistics (mean, std, best reward)
- View performance distribution

### Compare Checkpoints
Select multiple checkpoints to:
- Visualize improvement over training
- Compare performance metrics
- See percentage improvement

## 📊 Understanding the Results

### Training Curves
- **Episode Rewards**: Total reward per episode (max = 500)
- **Rolling Average**: Smoothed trend over 50 episodes
- **Episode Lengths**: Number of steps before failure

### Performance Metrics
- **Mean Reward**: Average over test episodes
- **Std Reward**: Performance consistency
- **Best Reward**: Maximum achieved reward

### Expected Progress
| Episode | Expected Reward | Performance Level |
|---------|----------------|-------------------|
| 0       | ~20-30         | Random (untrained) |
| 50      | ~40-80         | Basic learning |
| 100     | ~100-200       | Improving |
| 200     | ~250-400       | Good |
| 500     | ~400-500       | Excellent |

## 🔧 Customization

### Training Parameters

Edit `train_agent.py` to modify:

```python
config = {
    'batch_size': 128,      # Replay batch size
    'gamma': 0.99,          # Discount factor
    'eps_start': 0.9,       # Initial exploration rate
    'eps_end': 0.05,        # Final exploration rate
    'eps_decay': 1000,      # Exploration decay rate
    'tau': 0.005,           # Target network update rate
    'lr': 1e-4,             # Learning rate
    'memory_size': 10000    # Replay buffer size
}
```

### Different Environments

Change the environment in `train_agent.py`:

```python
# Instead of CartPole-v1, try:
env = gym.make('LunarLander-v2')  # Lunar landing
env = gym.make('Acrobot-v1')      # Swing-up task
env = gym.make('MountainCar-v0')  # Mountain car
```

Note: Different environments have different observation/action spaces, so you'll need to adjust `n_observations` and `n_actions`.

### Checkpoint Intervals

Modify which episodes to save:

```python
train_dqn(
    num_episodes=1000,
    checkpoint_intervals=[0, 100, 250, 500, 750, 1000]
)
```

## 📚 Technical Details

### DQN Algorithm
- **Network**: 3-layer MLP (4 → 128 → 128 → 2)
- **Experience Replay**: 10,000 transition buffer
- **Target Network**: Soft updates with τ=0.005
- **Loss**: Smooth L1 (Huber) loss
- **Optimizer**: AdamW with gradient clipping

### CartPole Environment
- **Observation**: [cart position, cart velocity, pole angle, pole angular velocity]
- **Actions**: [push left, push right]
- **Reward**: +1 for each timestep pole stays upright
- **Terminal**: Pole angle > 12° or cart position > 2.4
- **Success**: Average reward > 475 over 100 episodes

## 🎓 Educational Use

### For Teaching
This demo is perfect for:
- Demonstrating RL concepts (exploration, exploitation, learning curves)
- Showing the importance of hyperparameters
- Visualizing agent improvement over time
- Comparing different training strategies

### Key Concepts Illustrated
1. **Exploration vs Exploitation**: ε-greedy strategy
2. **Experience Replay**: Learning from past experiences
3. **Target Networks**: Stabilizing training
4. **Reward Shaping**: Understanding sparse vs dense rewards
5. **Convergence**: When and how agents learn

## 🐛 Troubleshooting

### Training is too slow
- Reduce `num_episodes` to 300
- Use fewer checkpoint intervals
- Check if GPU is available: `torch.cuda.is_available()`

### Videos not generating
- Make sure you've run training first
- Check that `checkpoints/` directory exists and has .pt files
- Install Pillow: `pip install Pillow`

### Streamlit app not loading
- Ensure all dependencies are installed
- Check that checkpoints and metrics exist
- Try: `streamlit run demo_app.py --server.port 8502`

## 📖 Further Reading

- [OpenAI Spinning Up - DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html)
- [DeepMind DQN Paper](https://www.nature.com/articles/nature14236)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

## 📝 Citation

If you use this demo in your teaching materials:

```
@software{rl_training_demo,
  title={RL Training Demo: Interactive DQN Visualization},
  author={Nipun Batra},
  year={2024},
  url={https://github.com/nipunbatra/ml-teaching}
}
```

## 📄 License

Part of the ML Teaching repository. See main repository for license details.
