# Pre-trained Demo Checkpoints

These are pre-trained model checkpoints ready for immediate demos.

## Checkpoints Included

| Checkpoint | Episode | Performance | Purpose |
|------------|---------|-------------|---------|
| `checkpoint_ep0000.pt` | 0 | Untrained (random) | Show baseline |
| `checkpoint_ep1699.pt` | 1699 | ~5 pipes average | Show learning progress |
| `checkpoint_ep2099.pt` | 2099 | ~15 pipes | Show high performance |
| `checkpoint_ep2199.pt` | 2199 | ~13 pipes | Show consistency |
| `checkpoint_ep2999.pt` | 2999 | Final agent (~5 pipes avg) | Show final trained model |

## Best Performance

- **Best single run**: 32 pipes (achieved during training)
- **Average (last 100 episodes)**: 4.49 pipes
- **Episodes with 10+ pipes**: 136 out of 3000

## Quick Start

```bash
# Play with the AI
python play_game.py
# Select "2" for AI mode
# Choose checkpoint 3 (ep 2099) or 4 (ep 2199) for best performance!

# Or use Streamlit
streamlit run demo_app.py
```

## Size

Total: 8.5 MB (5 checkpoints)

## Training Details

- Algorithm: DQN (Deep Q-Network)
- Network: 4-layer (256-256-128-2)
- Training: 3000 episodes
- Game settings: Easy mode (bigger gaps, slower speed)
- Learning rate: 5e-5
- Replay buffer: 100k transitions

## Full Checkpoints

For all 61 checkpoints from the full training run, you can:
1. Clone the repo
2. Run `python train_better.py` to train from scratch
3. Or contact for the full checkpoint archive (~122 MB)
