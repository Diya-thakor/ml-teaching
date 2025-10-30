# Quick Start Guide

## ✅ You're Ready to Demo!

Everything is set up and ready to go. Here's how to use it:

### 🎬 Option 1: Interactive App (Recommended)

```bash
cd /Users/nipun/git/ml-teaching/apps/rl-demo
streamlit run demo_app.py
```

The app now includes:
- ✨ **Clear instructions** - Expandable intro explaining the demo
- 🎯 **Descriptive checkpoints** - Each labeled with performance level
- 📹 **Animated videos** - GIFs now display and animate properly
- 🧪 **Easy testing** - One-click agent evaluation
- 📊 **Visual comparisons** - Side-by-side performance charts

**How to use:**
1. Open the app (it starts with instructions visible)
2. Select a checkpoint from sidebar (e.g., "🚀 Breakthrough! - Learning kicks in")
3. Watch the animated video showing agent behavior
4. Click "Run Test Episodes" to evaluate performance
5. Scroll down to compare multiple checkpoints

### 🎥 Option 2: Quick Video Demo

Just show the GIFs in order:

```bash
cd /Users/nipun/git/ml-teaching/apps/rl-demo/videos

# Method 1: Open in browser/viewer
open agent_ep0000.gif    # Random (fails immediately)
open agent_ep0100.gif    # Learning (slight improvement)
open agent_ep0200.gif    # Breakthrough! (balances ~150 steps)
open agent_ep0400.gif    # Expert (balances ~255 steps)

# Method 2: Show the comparison figure
open training_progression.png
```

### 📊 Option 3: Command Line Testing

```bash
# Test specific checkpoint
python test_checkpoint.py 200

# Test with live rendering
python test_checkpoint.py 400 --render
```

## 🎯 What Changed

### Fixed Issues:
1. ✅ **Video now animates** - Uses HTML embedding with base64 encoding
2. ✅ **Clear instructions** - Expandable intro with problem description
3. ✅ **Better UX** - Descriptive labels, tooltips, and help text

### New Features:
- **Checkpoint descriptions**: "🎲 Untrained", "🚀 Breakthrough!", "🏆 Expert"
- **Video moved to top**: Most important content first
- **Step-by-step guide**: Clear workflow from selection to comparison
- **Better layout**: Videos, testing, and progress charts organized logically

## 📝 For Presentations

### 5-Minute Demo Script:

**1. Problem Introduction (30 sec)**
- Open app, show the intro expander
- "CartPole: Balance a pole on a moving cart"
- "Max reward: 500 steps"

**2. Untrained Agent (30 sec)**
- Select "🎲 Untrained" from sidebar
- Show video: "Random actions, fails in ~10 steps"

**3. Learning Process (1 min)**
- Switch to "📈 Starting to Learn" (Ep 100)
- Show video: "Still struggling, minimal improvement"
- Switch to "🚀 Breakthrough!" (Ep 200)
- Show video: "Aha moment! Balances for ~150 steps"

**4. Expert Performance (1 min)**
- Switch to "🏆 Expert" (Ep 400)
- Show video: "Near-perfect, balances for ~255 steps"
- Click "Run Test Episodes" to show live evaluation

**5. Compare & Discuss (2 min)**
- Scroll to comparison section
- Select episodes 0, 100, 200, 400
- Click "Generate Comparison"
- Point out: dramatic improvement curve
- Discuss: exploration → breakthrough → optimization

## 🐛 Troubleshooting

### Video not animating?
- Should be fixed now (using HTML embedding)
- If issues persist, check browser console for errors

### Streamlit errors?
```bash
# Reinstall if needed
uv pip install streamlit --upgrade
```

### Missing checkpoints?
```bash
# Retrain (takes ~2-3 minutes)
python train_agent.py
python generate_videos.py
```

## 📊 Expected Performance

Your trained agent achieved:
- **Episode 0**: 8.8 avg (random)
- **Episode 100**: 15.7 avg (learning)
- **Episode 200**: 131.8 avg (breakthrough!)
- **Episode 400**: 210.2 avg (expert)
- **Episode 500**: 500.0 avg (perfect!)

## 🎓 Teaching Points

Use this demo to illustrate:

1. **Exploration vs Exploitation**
   - Early: High epsilon = random exploration
   - Late: Low epsilon = exploit learned policy

2. **Learning Curve Shape**
   - Flat period (random behavior)
   - Sudden jump (breakthrough moment ~Ep 130)
   - Plateau (stable performance)
   - Final optimization (near-perfect)

3. **Experience Replay**
   - Agent learns from past experiences
   - More sample efficient than online learning

4. **Target Networks**
   - Stabilizes training
   - Prevents oscillations and divergence

## 📁 File Locations

- **Checkpoints**: `checkpoints/` (6 files, ~140-360KB each)
- **Videos**: `videos/` (6 GIFs, 72KB to 1.4MB)
- **Metrics**: `metrics/` (JSON files with training data)
- **App**: `demo_app.py` (interactive Streamlit dashboard)

All files are in:
```
/Users/nipun/git/ml-teaching/apps/rl-demo/
```

## 🚀 You're All Set!

Just run:
```bash
streamlit run demo_app.py
```

And start demoing! The app is now much clearer and videos work properly.
