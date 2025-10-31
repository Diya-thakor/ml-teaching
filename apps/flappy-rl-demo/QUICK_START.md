# Flappy Bird RL Demo - Quick Start

## What This Demo Shows

Watch a Deep Q-Network (DQN) agent learn to play Flappy Bird from scratch! The demo includes:

1. **Training visualization** - See how the agent improves over time
2. **Video playback** - Watch GIFs of the agent at different training stages
3. **Interactive Streamlit app** - Compare performance and test agents
4. **Live gameplay** - Watch agents play in real-time with Pygame

## Setup (5 minutes)

### 1. Install Dependencies

```bash
cd apps/flappy-rl-demo
pip install -r requirements.txt
```

### 2. Quick Training (Already Done!)

We've already trained an agent for 300 episodes. Checkpoints are saved in `checkpoints/`.

To train more:
```bash
python quick_train.py  # 300 episodes (~5-10 minutes)
# OR
python train_agent.py  # 2000 episodes (~2-4 hours)
```

### 3. Generate Videos (Already Done!)

```bash
python generate_videos.py
```

Videos are saved in `videos/` as GIF files.

## Running the Demos

### Option 1: Interactive Streamlit App (Recommended)

```bash
streamlit run demo_app.py
```

Open browser at http://localhost:8501

**Features:**
- Select different training checkpoints
- Watch animated videos of agent behavior
- Run test episodes and see statistics
- Compare multiple checkpoints side-by-side
- View training curves (scores, rewards, episode lengths)

### Option 2: Live Pygame Playback

```bash
python play_with_agent.py
```

**Features:**
- Select which checkpoint to load
- Watch the agent play in real-time with nice graphics
- See score and episode number
- Press SPACE to restart, P to pause, Q to quit
- View statistics at the end

### Option 3: Manual Testing

```bash
python test_checkpoint.py
```

Test a specific checkpoint and see detailed statistics.

## What to Look For

### Episode 0 (Untrained)
- **Behavior**: Random actions, chaotic flapping
- **Performance**: Crashes almost immediately (~20 frames)
- **Score**: 0 pipes
- **Learning**: None - completely random policy

### Episode 50-100 (Early Learning)
- **Behavior**: Still mostly random, but slight pattern emerging
- **Performance**: ~25-30 frames survival
- **Score**: 0-1 pipes occasionally
- **Learning**: Starting to correlate actions with rewards

### Episode 150-200 (Breakthrough)
- **Behavior**: More controlled flapping, timing improving
- **Performance**: ~40-60 frames
- **Score**: 1-2 pipes regularly
- **Learning**: Understanding basic physics and timing

### Episode 250-299 (Competent)
- **Behavior**: Smoother play, anticipating gaps
- **Performance**: Variable but improving
- **Score**: Best scores of 2-3 pipes
- **Learning**: Decent policy but needs more training for consistency

## Training Longer for Better Results

For really impressive performance (10-20+ pipes), train longer:

```bash
python train_agent.py  # 2000 episodes
```

Expected progression:
- **500 episodes**: 5-8 pipes regularly
- **1000 episodes**: 10-15 pipes
- **1500+ episodes**: 20+ pipes, expert-level play

## Understanding the Visualizations

### Training Curves in Streamlit

1. **Scores Plot** (Most Important!)
   - Blue line = raw episode scores
   - Red line = 50-episode moving average
   - Shows how many pipes the agent passes

2. **Rewards Plot**
   - Shows total reward per episode
   - Combines survival time + pipe bonuses
   - Should trend upward as learning progresses

3. **Episode Lengths Plot**
   - How long the agent survives (in steps)
   - Longer = better (more careful play)
   - Correlates with score but not perfectly

### Comparison Charts

Select multiple checkpoints to see:
- Bar charts comparing mean scores and rewards
- Percentage improvement over training
- How quickly learning happened

## Troubleshooting

### "No checkpoints found!"
Run training first: `python quick_train.py`

### "Videos not available"
Run: `python generate_videos.py`

### Training seems stuck at low scores
This is normal early on! The agent needs:
- Sufficient exploration (epsilon decay)
- Enough replay memory
- Time to learn the complex timing

After 300 episodes, expect 0-2 pipes. After 1000+, expect 5-15 pipes.

### Pygame window not responding
Close and rerun `play_with_agent.py`. Make sure pygame is installed.

## Next Steps

1. **Train longer**: `python train_agent.py` for 2000 episodes
2. **Experiment with hyperparameters**: Edit configs in `train_agent.py`
3. **Try different rewards**: Modify reward structure in `flappy_env.py`
4. **Implement improvements**:
   - Double DQN
   - Dueling DQN
   - Prioritized replay
   - Different network architectures

## File Overview

| File | Purpose |
|------|---------|
| `flappy_env.py` | Gymnasium environment (game logic + RL interface) |
| `train_agent.py` | Full DQN training (2000 episodes) |
| `quick_train.py` | Quick training (300 episodes for demo) |
| `generate_videos.py` | Create GIF videos from checkpoints |
| `demo_app.py` | Streamlit interactive visualization |
| `play_with_agent.py` | Pygame live playback with agent |
| `test_checkpoint.py` | Test individual checkpoints |
| `checkpoints/` | Saved model weights |
| `videos/` | Generated GIF animations |
| `metrics/` | Training statistics (JSON) |

Enjoy watching the agent learn! 🐦
