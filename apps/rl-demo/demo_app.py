"""
Interactive RL Training Visualization App
Shows agent improvement across training checkpoints
"""
import streamlit as st
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import base64
from train_agent import DQNAgent

st.set_page_config(page_title="RL Training Demo", layout="wide", page_icon="🤖")

# Directories
CHECKPOINT_DIR = Path("checkpoints")
VIDEO_DIR = Path("videos")
METRICS_DIR = Path("metrics")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_agent_from_checkpoint(checkpoint_path):
    """Load agent from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    n_observations = 4
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


def run_episode(agent, env, max_steps=500):
    """Run one episode and return metrics"""
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    episode_reward = 0
    step = 0

    while step < max_steps:
        action = agent.select_action(state_tensor, training=False)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        episode_reward += reward
        step += 1

        if terminated or truncated:
            break

        state_tensor = torch.tensor(observation, dtype=torch.float32,
                                   device=device).unsqueeze(0)

    return episode_reward, step


# ============================================================================
# Header
# ============================================================================
st.title("🤖 RL Training Visualization: DQN on CartPole")

# Introduction with expander
with st.expander("📖 What is this demo?", expanded=True):
    st.markdown("""
    ### The Problem: CartPole Balance

    An agent must balance a pole on a moving cart by pushing left or right.
    - **Reward**: +1 for each timestep the pole stays upright
    - **Maximum score**: 500 steps
    - **Challenge**: The pole falls if it tilts >12° or the cart moves >2.4 units from center

    ### What You'll See

    This demo shows how a Deep Q-Network (DQN) agent learns through **reinforcement learning**:

    1. **Episode 0**: Untrained agent (random actions) - pole falls in ~10 steps
    2. **Episode 100**: Starting to learn - slight improvement (~15-20 steps)
    3. **Episode 200**: Major breakthrough - balances for ~130 steps!
    4. **Episode 300-400**: Stable expert behavior - 120-250 steps

    ### How to Use This Demo

    1. **Select a checkpoint** from the sidebar (different training stages)
    2. **View the animated video** showing agent behavior
    3. **Run test episodes** to evaluate current performance
    4. **Compare checkpoints** at the bottom to see improvement
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

# Checkpoint selection with descriptions
st.sidebar.markdown("### 🎯 Select Training Stage")

checkpoint_options = {ep: path for ep, path in checkpoints}

# Create helpful descriptions for each checkpoint
checkpoint_descriptions = {
    0: "🎲 Untrained - Random actions (~10 steps)",
    50: "🌱 Early Learning - Still mostly random",
    100: "📈 Starting to Learn - Minor improvements (~15-20 steps)",
    200: "🚀 Breakthrough! - Learning kicks in (~130 steps)",
    300: "⭐ Competent - Stable performance (~125 steps)",
    400: "🏆 Expert - Near-optimal behavior (~200+ steps)"
}

# Display checkpoint info
selected_episode = st.sidebar.selectbox(
    "Choose a checkpoint:",
    options=list(checkpoint_options.keys()),
    format_func=lambda x: checkpoint_descriptions.get(x, f"Episode {x}"),
    help="Select different training stages to see how the agent improves over time"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Test Settings")
num_test_episodes = st.sidebar.slider(
    "Number of test episodes",
    min_value=1,
    max_value=20,
    value=5,
    help="How many episodes to run when testing the agent"
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

# Create columns for layout
col1, col2 = st.columns([1, 1])

# ============================================================================
# Column 1: Agent Video & Performance Testing
# ============================================================================
with col1:
    st.subheader(f"🎮 Agent at Episode {episode_num}")

    # Show video first (most important)
    video_path = VIDEO_DIR / f"agent_ep{episode_num:04d}.gif"
    if video_path.exists():
        st.markdown("**📹 Watch the agent in action:**")

        # Read GIF file as bytes and encode to base64 for HTML embedding
        with open(video_path, "rb") as f:
            gif_bytes = f.read()
        gif_base64 = base64.b64encode(gif_bytes).decode()

        # Display using HTML to ensure animation works
        html_code = f"""
        <div style="display: flex; flex-direction: column; align-items: center; margin: 20px 0;">
            <img src="data:image/gif;base64,{gif_base64}"
                 alt="Agent behavior at Episode {episode_num}"
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
    st.markdown("Run multiple episodes to evaluate how well the agent performs:")

    if st.button("▶️ Run Test Episodes", type="primary", use_container_width=True):
        env = gym.make('CartPole-v1')

        test_rewards = []
        test_steps = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(num_test_episodes):
            reward, steps = run_episode(agent, env)
            test_rewards.append(reward)
            test_steps.append(steps)

            progress_bar.progress((i + 1) / num_test_episodes)
            status_text.text(f"Episode {i+1}/{num_test_episodes}: "
                           f"Reward = {reward:.0f}, Steps = {steps}")

        env.close()

        # Display results
        st.markdown("---")
        st.markdown("### 📊 Test Results")

        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric("Mean Reward", f"{np.mean(test_rewards):.1f}")
        with metrics_cols[1]:
            st.metric("Std Reward", f"{np.std(test_rewards):.1f}")
        with metrics_cols[2]:
            st.metric("Best Reward", f"{np.max(test_rewards):.0f}")

        # Plot test results
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(test_rewards)), test_rewards, color='#2E86AB', alpha=0.7)
        ax.axhline(y=np.mean(test_rewards), color='red', linestyle='--',
                  label=f'Mean: {np.mean(test_rewards):.1f}')
        ax.axhline(y=500, color='green', linestyle='--', alpha=0.5,
                  label='Maximum: 500')
        ax.set_xlabel("Test Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title(f"Test Performance (Episode {episode_num})")
        ax.legend()
        ax.grid(alpha=0.3)
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

        # Plot training curve
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        # Rewards
        rewards = metrics['training_rewards']
        ax1.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')

        # Rolling average
        window = 50
        if len(rewards) >= window:
            rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), rolling, color='red',
                    linewidth=2, label=f'{window}-Episode Average')

        ax1.axhline(y=500, color='green', linestyle='--', alpha=0.5,
                   label='Max Possible (500)')
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title(f"Training Rewards (up to Episode {episode_num})")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Episode lengths
        lengths = metrics['training_lengths']
        ax2.plot(lengths, alpha=0.3, color='purple', label='Episode Length')

        if len(lengths) >= window:
            rolling_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(lengths)), rolling_len, color='orange',
                    linewidth=2, label=f'{window}-Episode Average')

        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.set_title(f"Episode Lengths (up to Episode {episode_num})")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Show summary statistics
        st.markdown("### 📊 Training Statistics")
        stat_cols = st.columns(3)

        with stat_cols[0]:
            st.metric("Total Episodes", len(rewards))
        with stat_cols[1]:
            last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
            st.metric("Avg Reward (last 100)", f"{np.mean(last_100):.1f}")
        with stat_cols[2]:
            st.metric("Best Episode", f"{np.max(rewards):.0f}")

    else:
        st.warning("Metrics file not found for this checkpoint.")

# ============================================================================
# Comparison Section
# ============================================================================
st.markdown("---")
st.header("🔄 Compare Multiple Checkpoints")
st.markdown("""
**Visualize Learning Progress:** Select multiple checkpoints to see how performance improved during training.
Great for understanding the learning curve and identifying breakthrough moments!
""")

compare_episodes = st.multiselect(
    "Select 2 or more checkpoints to compare:",
    options=list(checkpoint_options.keys()),
    default=list(checkpoint_options.keys())[:min(4, len(checkpoint_options))],
    format_func=lambda x: checkpoint_descriptions.get(x, f"Episode {x}"),
    help="Choose different training stages to compare side-by-side"
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
                        'mean_reward': m.get('mean_reward', np.mean(m['training_rewards'][-10:])),
                        'std_reward': m.get('std_reward', 0)
                    })

        if comparison_data:
            # Plot comparison
            fig, ax = plt.subplots(figsize=(10, 5))

            episodes = [d['episode'] for d in comparison_data]
            means = [d['mean_reward'] for d in comparison_data]
            stds = [d['std_reward'] for d in comparison_data]

            ax.bar(range(len(episodes)), means, yerr=stds, capsize=5,
                  color='#2E86AB', alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(episodes)))
            ax.set_xticklabels([f"Ep {e}" for e in episodes])
            ax.set_ylabel("Mean Reward")
            ax.set_title("Performance Comparison Across Training")
            ax.axhline(y=500, color='green', linestyle='--', alpha=0.5,
                      label='Maximum (500)')
            ax.grid(alpha=0.3, axis='y')
            ax.legend()

            st.pyplot(fig)

            # Show improvement
            if len(comparison_data) >= 2:
                initial = comparison_data[0]['mean_reward']
                final = comparison_data[-1]['mean_reward']
                improvement = ((final - initial) / max(initial, 1)) * 100

                st.success(f"📈 **Improvement: {improvement:.1f}%** "
                          f"(from {initial:.1f} to {final:.1f})")
else:
    st.info("👆 Select at least 2 checkpoints above to compare their performance!")

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
with st.expander("🚀 Advanced Options & Next Steps"):
    st.markdown("""
    ### Experiment with the Training

    **Retrain with different settings:**
    ```bash
    # Edit config in train_agent.py, then:
    python train_agent.py
    python generate_videos.py
    ```

    **Try other environments:**
    - LunarLander-v2 (rocket landing)
    - MountainCar-v0 (hill climbing)
    - Acrobot-v1 (swing-up task)

    **Implement improvements:**
    - Double DQN (reduces overestimation)
    - Dueling DQN (separate value/advantage streams)
    - Prioritized Experience Replay (learn from important transitions)

    ### Understanding the Results

    **Good performance indicators:**
    - Rewards consistently above 400-475
    - Low variance between episodes
    - Quick recovery from random fluctuations

    **Signs the agent needs more training:**
    - High variance in rewards
    - Frequent episodes below 100 steps
    - No clear upward trend in learning curve
    """)
