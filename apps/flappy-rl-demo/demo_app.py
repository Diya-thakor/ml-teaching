"""
Interactive Flappy Bird RL Training Visualization App
Shows agent improvement from crashing early to achieving high scores
"""
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import base64
from train_agent import DQNAgent
from flappy_env import FlappyBirdEnv

st.set_page_config(page_title="Flappy Bird RL Demo", layout="wide", page_icon="🐦")

# Directories - check in order of preference
def get_checkpoint_dir():
    for name in ["checkpoints_better", "checkpoints_demo", "checkpoints"]:
        p = Path(name)
        if p.exists() and list(p.glob("checkpoint_ep*.pt")):
            return p
    return Path("checkpoints_demo")  # Default to demo

CHECKPOINT_DIR = get_checkpoint_dir()
VIDEO_DIR = Path("videos_better") if Path("videos_better").exists() else Path("videos")
METRICS_DIR = Path("metrics_better") if Path("metrics_better").exists() else Path("metrics")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_agent_from_checkpoint(checkpoint_path):
    """Load agent from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    n_observations = 6
    n_actions = 2

    agent = DQNAgent(n_observations, n_actions, config)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

    return agent, checkpoint['episode'], checkpoint


def get_available_checkpoints():
    """Get list of available checkpoints"""
    if not CHECKPOINT_DIR.exists():
        return []
    checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_ep*.pt"))
    return [(int(cp.stem.split('ep')[1]), cp) for cp in checkpoints]


def run_episode(agent, env, max_steps=2000):
    """Run one episode and return metrics"""
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    episode_reward = 0
    step = 0

    while step < max_steps:
        action = agent.select_action(state_tensor, training=False)
        observation, reward, terminated, truncated, info = env.step(action.item())
        episode_reward += reward
        step += 1

        if terminated or truncated:
            break

        state_tensor = torch.tensor(observation, dtype=torch.float32,
                                   device=device).unsqueeze(0)

    return episode_reward, step, info['score']


# ============================================================================
# Header
# ============================================================================
st.title("🐦 RL Training Visualization: DQN on Flappy Bird")

# Introduction
with st.expander("📖 What is this demo?", expanded=True):
    st.markdown("""
    ### The Challenge: Flappy Bird

    An agent must navigate a bird through gaps between pipes by choosing when to flap.
    - **Action**: Flap (jump) or do nothing (fall)
    - **Reward**: +10 for passing through each pipe, -100 for collision
    - **Challenge**: Precise timing required - pipes come continuously!

    ### Watch the Learning Journey

    This demo shows how a Deep Q-Network (DQN) agent learns through **reinforcement learning**:

    1. **Episode 0**: Untrained agent (random actions) - crashes immediately in ~10 frames
    2. **Episode 50**: Still struggling - learning basic physics (~20-30 frames)
    3. **Episode 100**: Minor breakthrough - passes 1 pipe occasionally
    4. **Episode 500+**: Getting competent - passes 3-5 pipes consistently
    5. **Episode 1500+**: Expert behavior - achieves scores of 10-20+ pipes!

    ### How to Use This Demo

    1. **Select a checkpoint** from the sidebar (different training stages)
    2. **Watch the animated video** showing agent behavior
    3. **Run test episodes** to evaluate current performance
    4. **Compare checkpoints** at the bottom to see improvement over time
    """)

st.info("👈 **Start by selecting a training checkpoint from the sidebar!**")

# ============================================================================
# Sidebar
# ============================================================================
st.sidebar.header("⚙️ Control Panel")

checkpoints = get_available_checkpoints()

if not checkpoints:
    st.error("No checkpoints found! Please run training first: `python train_agent.py`")
    st.stop()

# Checkpoint selection
st.sidebar.markdown("### 🎯 Select Training Stage")

checkpoint_options = {ep: path for ep, path in checkpoints}

# Create helpful descriptions
checkpoint_descriptions = {
    0: "🎲 Untrained - Random flapping (crashes immediately)",
    50: "🌱 Early Learning - Still crashing fast (~score 0)",
    100: "📈 Starting to Learn - Occasionally passes 1 pipe",
    200: "🚀 Getting Better - Passes 2-3 pipes",
    500: "⭐ Competent - Scores 3-5 pipes regularly",
    1000: "🏆 Advanced - Scores 8-12 pipes",
    1500: "💎 Expert - Scores 15-20+ pipes!",
    1950: "🎯 Master - Near-optimal play"
}

# Display checkpoint selector
selected_episode = st.sidebar.selectbox(
    "Choose a checkpoint:",
    options=list(checkpoint_options.keys()),
    format_func=lambda x: checkpoint_descriptions.get(x, f"Episode {x}"),
    help="Select different training stages to see improvement"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Test Settings")
num_test_episodes = st.sidebar.slider(
    "Number of test episodes",
    min_value=1,
    max_value=10,
    value=5,
    help="How many episodes to run for testing"
)

# ============================================================================
# Main Content
# ============================================================================

# Load selected checkpoint
checkpoint_path = checkpoint_options[selected_episode]
agent, episode_num, checkpoint = load_agent_from_checkpoint(checkpoint_path)

st.sidebar.success(f"✓ Loaded checkpoint from Episode {episode_num}")

# Show training configuration
with st.sidebar.expander("📋 Training Config"):
    config = checkpoint['config']
    st.json(config)

# Create columns
col1, col2 = st.columns([1, 1])

# ============================================================================
# Column 1: Agent Video & Performance Testing
# ============================================================================
with col1:
    st.subheader(f"🎮 Agent at Episode {episode_num}")

    # Show video
    video_path = VIDEO_DIR / f"agent_ep{episode_num:04d}.gif"
    if video_path.exists():
        st.markdown("**📹 Watch the agent play:**")

        # Read and display GIF
        with open(video_path, "rb") as f:
            gif_bytes = f.read()
        gif_base64 = base64.b64encode(gif_bytes).decode()

        html_code = f"""
        <div style="display: flex; flex-direction: column; align-items: center; margin: 20px 0;">
            <img src="data:image/gif;base64,{gif_base64}"
                 alt="Agent at Episode {episode_num}"
                 style="max-width: 100%; border: 2px solid #ddd; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <p style="text-align: center; margin-top: 10px; color: #666; font-style: italic;">
                Agent behavior at Episode {episode_num}
            </p>
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)
    else:
        st.warning("💡 Video not available. Generate videos using: `python generate_videos.py`")

    # Testing section
    st.markdown("---")
    st.markdown("**🧪 Test Current Agent**")
    st.markdown("Run multiple episodes to evaluate performance:")

    if st.button("▶️ Run Test Episodes", type="primary", use_container_width=True):
        env = FlappyBirdEnv()

        test_rewards = []
        test_steps = []
        test_scores = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(num_test_episodes):
            reward, steps, score = run_episode(agent, env)
            test_rewards.append(reward)
            test_steps.append(steps)
            test_scores.append(score)

            progress_bar.progress((i + 1) / num_test_episodes)
            status_text.text(f"Episode {i+1}/{num_test_episodes}: "
                           f"Score = {score}, Steps = {steps}")

        env.close()

        # Display results
        st.markdown("---")
        st.markdown("### 📊 Test Results")

        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            st.metric("Mean Score", f"{np.mean(test_scores):.1f}")
        with metrics_cols[1]:
            st.metric("Best Score", f"{np.max(test_scores):.0f}")
        with metrics_cols[2]:
            st.metric("Mean Steps", f"{np.mean(test_steps):.0f}")
        with metrics_cols[3]:
            st.metric("Mean Reward", f"{np.mean(test_rewards):.1f}")

        # Plot test results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Scores
        ax1.bar(range(len(test_scores)), test_scores, color='#2E86AB', alpha=0.7)
        ax1.axhline(y=np.mean(test_scores), color='red', linestyle='--',
                   label=f'Mean: {np.mean(test_scores):.1f}')
        ax1.set_xlabel("Test Episode")
        ax1.set_ylabel("Score (Pipes Passed)")
        ax1.set_title("Test Scores")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Steps survived
        ax2.bar(range(len(test_steps)), test_steps, color='#A23B72', alpha=0.7)
        ax2.axhline(y=np.mean(test_steps), color='red', linestyle='--',
                   label=f'Mean: {np.mean(test_steps):.0f}')
        ax2.set_xlabel("Test Episode")
        ax2.set_ylabel("Steps Survived")
        ax2.set_title("Survival Time")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

# ============================================================================
# Column 2: Training Progress
# ============================================================================
with col2:
    st.subheader("📈 Training Progress")
    st.markdown("**See how the agent learned over time:**")

    # Load metrics
    metrics_path = METRICS_DIR / f"metrics_ep{episode_num:04d}.json"

    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # Plot training curves
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

        # Scores (most important!)
        scores = metrics['training_scores']
        ax1.plot(scores, alpha=0.3, color='blue', label='Episode Score')

        window = 50
        if len(scores) >= window:
            rolling = np.convolve(scores, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(scores)), rolling, color='red',
                    linewidth=2, label=f'{window}-Episode Average')

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Score (Pipes Passed)")
        ax1.set_title(f"Training Scores (up to Episode {episode_num})")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Rewards
        rewards = metrics['training_rewards']
        ax2.plot(rewards, alpha=0.3, color='green', label='Episode Reward')

        if len(rewards) >= window:
            rolling_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(rewards)), rolling_rewards, color='orange',
                    linewidth=2, label=f'{window}-Episode Average')

        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Total Reward")
        ax2.set_title("Training Rewards")
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Episode lengths
        lengths = metrics['training_lengths']
        ax3.plot(lengths, alpha=0.3, color='purple', label='Episode Length')

        if len(lengths) >= window:
            rolling_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(lengths)), rolling_len, color='brown',
                    linewidth=2, label=f'{window}-Episode Average')

        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Steps")
        ax3.set_title("Episode Lengths (Survival Time)")
        ax3.legend()
        ax3.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Summary statistics
        st.markdown("### 📊 Training Statistics")
        stat_cols = st.columns(3)

        with stat_cols[0]:
            st.metric("Total Episodes", len(scores))
        with stat_cols[1]:
            last_100 = scores[-100:] if len(scores) >= 100 else scores
            st.metric("Avg Score (last 100)", f"{np.mean(last_100):.1f}")
        with stat_cols[2]:
            st.metric("Best Score", f"{np.max(scores):.0f}")

    else:
        st.warning("Metrics file not found for this checkpoint.")

# ============================================================================
# Comparison Section
# ============================================================================
st.markdown("---")
st.header("🔄 Compare Training Progress")
st.markdown("""
**Visualize the Learning Journey:** Select multiple checkpoints to see how performance improved.
Great for understanding breakthrough moments!
""")

compare_episodes = st.multiselect(
    "Select 2 or more checkpoints to compare:",
    options=list(checkpoint_options.keys()),
    default=list(checkpoint_options.keys())[:min(4, len(checkpoint_options))],
    format_func=lambda x: checkpoint_descriptions.get(x, f"Episode {x}"),
    help="Choose different training stages to compare"
)

if len(compare_episodes) >= 2:
    if st.button("📊 Generate Comparison", type="primary", use_container_width=True):
        comparison_data = []

        for ep in compare_episodes:
            metrics_path = METRICS_DIR / f"metrics_ep{ep:04d}.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    m = json.load(f)
                    comparison_data.append({
                        'episode': ep,
                        'mean_score': m.get('mean_score', 0),
                        'mean_reward': m.get('mean_reward', 0)
                    })

        if comparison_data:
            # Plot comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            episodes = [d['episode'] for d in comparison_data]
            scores = [d['mean_score'] for d in comparison_data]
            rewards = [d['mean_reward'] for d in comparison_data]

            # Scores comparison
            ax1.bar(range(len(episodes)), scores, color='#2E86AB', alpha=0.7, edgecolor='black')
            ax1.set_xticks(range(len(episodes)))
            ax1.set_xticklabels([f"Ep {e}" for e in episodes])
            ax1.set_ylabel("Mean Score (Pipes Passed)")
            ax1.set_title("Score Comparison Across Training")
            ax1.grid(alpha=0.3, axis='y')

            # Rewards comparison
            ax2.bar(range(len(episodes)), rewards, color='#F18F01', alpha=0.7, edgecolor='black')
            ax2.set_xticks(range(len(episodes)))
            ax2.set_xticklabels([f"Ep {e}" for e in episodes])
            ax2.set_ylabel("Mean Reward")
            ax2.set_title("Reward Comparison Across Training")
            ax2.grid(alpha=0.3, axis='y')

            plt.tight_layout()
            st.pyplot(fig)

            # Show improvement
            if len(comparison_data) >= 2:
                initial_score = comparison_data[0]['mean_score']
                final_score = comparison_data[-1]['mean_score']

                if initial_score > 0:
                    improvement = ((final_score - initial_score) / initial_score) * 100
                    st.success(f"📈 **Score Improvement: {improvement:.0f}%** "
                              f"(from {initial_score:.1f} to {final_score:.1f} pipes)")
                else:
                    st.success(f"📈 **Score Improvement: {final_score:.1f} pipes!** "
                              f"(from 0 to {final_score:.1f})")
else:
    st.info("👆 Select at least 2 checkpoints above to compare!")

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
with st.expander("🚀 Advanced Options & Next Steps"):
    st.markdown("""
    ### Experiment with Training

    **Retrain with different settings:**
    ```bash
    # Edit config in train_agent.py, then:
    python train_agent.py
    python generate_videos.py
    ```

    **Manual Play (Coming Soon):**
    Load trained agent weights and play Flappy Bird yourself with AI assistance!

    ### Understanding the Results

    **Good performance indicators:**
    - Scores consistently above 10 pipes
    - Survival time > 300 steps
    - Low variance between test episodes

    **Signs agent needs more training:**
    - Scores stuck at 0-2 pipes
    - High variance in performance
    - No clear upward trend in learning curves

    ### Technical Details

    **Network Architecture:**
    - 4-layer DQN (256-256-128 neurons)
    - Input: 6 features (bird position, velocity, pipe distances)
    - Output: 2 actions (flap or not)

    **Training Hyperparameters:**
    - Learning rate: 1e-4
    - Batch size: 128
    - Replay buffer: 50,000 transitions
    - Epsilon decay: 5,000 steps
    """)
