# BBF Implementation Notes

This document details the PyTorch implementation of BBF and how it maps to the official JAX implementation.

## Architecture

### Network Components

1. **Impala CNN Encoder** (`ImpalaEncoder`)
   - 3 stages with residual blocks
   - Width scaling: 4× (base channels: [16, 32, 32] → [64, 128, 128])
   - Num blocks: 2 per stage
   - Max pooling after each stage's conv
   - Output: Flattened 128×11×11 = 15,488-dim vector

2. **Transition Model** (`TransitionModel`)
   - For SPR: predicts next latent state from current latent + action
   - Action embedding layer
   - 2-layer MLP with residual connection
   - Layer norm for stability

3. **Projection Heads** (`ProjectionHead`)
   - For SPR self-supervised learning
   - Projects latent to 256-dim space
   - Used in both online and target networks

4. **Q-value Head**
   - Dueling architecture (separate value and advantage streams)
   - C51 distributional RL (51 atoms)
   - Hidden layer: 2048 dimensions

### Key Hyperparameters (from `BBF.gin`)

```python
# Network
width_scale = 4
hidden_dim = 2048
num_atoms = 51
v_min, v_max = -10.0, 10.0

# Learning
learning_rate = 1e-4
weight_decay = 0.1 (AdamW)
gamma: 0.97 → 0.997 (annealing)
n_step: 10 → 3 (annealing)

# Replay
buffer_capacity = 200,000
batch_size = 32
replay_ratio = 2 or 8 (gradient steps per env step)
learning_starts = 2,000

# Target Network
target_update_tau = 0.005 (EMA coefficient)
target_action_selection = True

# Resets (SR-SPR)
reset_every = 20,000 gradient steps
shrink_factor = 0.5
perturb_factor = 0.5
cycle_steps = 10,000 (annealing period)

# SPR
spr_weight = 5.0
jumps = 5 (future steps to predict)

# Architecture flags
dueling = True
double_dqn = True
distributional = True
noisy = False
```

## Annealing Schedules

### Gamma Annealing

Exponential annealing over `cycle_steps` after each reset:

```python
progress = steps_since_reset / cycle_steps
log_min = log(1 - gamma_min)  # log(1 - 0.97) = log(0.03)
log_max = log(1 - gamma_max)  # log(1 - 0.997) = log(0.003)
log_gamma = log_min + progress * (log_max - log_min)
gamma = 1 - exp(log_gamma)
```

Starts at 0.97, anneals to 0.997 over 10,000 steps, then stays at 0.997.

### N-step Annealing

Linear annealing from max to min:

```python
progress = steps_since_reset / cycle_steps
n = max_n - progress * (max_n - min_n)
n = round(n)
```

Starts at 10, anneals to 3 over 10,000 steps, then stays at 3.

**Rationale:** Large n-step converges faster but to worse asymptote. Start with large n for fast learning, anneal to small n for better final performance.

## Periodic Resets

Every 20,000 gradient steps (until 100,000 steps):

1. Create new random network with same architecture
2. Interpolate encoder and transition_model parameters:
   ```python
   new_param = shrink * old_param + perturb * random_param
   ```
   with shrink=0.5, perturb=0.5 (harder reset than SR-SPR's 0.8/0.2)
3. Keep Q-head and projection head parameters unchanged
4. Restart annealing schedule

**Purpose:** Prevents overfitting to early data, maintains plasticity at high replay ratios.

## SPR Loss

Self-predictive representations (SPR) predicts future latent states:

1. Encode current state to latent: `z_t = encoder(s_t)`
2. Rollout future latents: `z_{t+k} = transition^k(z_t, a_t:t+k)`
3. Project and predict: `p_{t+k} = predictor(projector(z_{t+k}))`
4. Target: `τ_{t+k} = projector_target(encoder_target(s_{t+k}))`
5. Loss: `-cosine_similarity(p_{t+k}, τ_{t+k})`

Sum over k=1 to `jumps` (5 future steps).

**Purpose:** Self-supervised learning improves sample efficiency in low-data regime.

## EMA Target Network

Instead of periodic hard updates (DQN), use exponential moving average:

```python
target_param = (1 - tau) * target_param + tau * online_param
```

with tau=0.005, updated every gradient step.

**Purpose:** Smoother target updates reduce Q-value oscillations, especially important for large networks.

## Data Augmentation

DrQ-style augmentation on every batch:

1. Pad image by 4 pixels (replication padding)
2. Random crop back to 84×84
3. Intensity augmentation: multiply by `1 + 0.05 * clip(N(0,1), -2, 2)`

Applied to both states and next_states before network forward pass.

## Prioritized Replay

Subsequence sampling for SPR:

- Sample start index such that `jumps+1` consecutive transitions are valid (no episode boundaries)
- Priority = TD error + ε
- Importance sampling weights: `(N * p_i)^(-β)`, normalized by max

## C51 Distributional RL

Instead of scalar Q-values, predict distribution over 51 atoms:

1. Support: [-10, 10] with 51 atoms
2. Network outputs logits for each (action, atom) pair
3. Target distribution projected onto support (categorical projection)
4. Loss: Cross-entropy between predicted and target distributions
5. Q-value for action selection: `Q(s,a) = Σ_i p_i(s,a) * z_i`

## Training Loop

For each environment step:
1. Store transition in replay buffer
2. For `replay_ratio` times:
   - Sample batch of subsequences
   - Compute TD loss (C51)
   - Compute SPR loss
   - Combined loss: `td_loss + spr_weight * spr_loss`
   - Weighted by importance sampling weights
   - Backprop and update
   - Update target network (EMA)
   - Update replay priorities
3. Check if reset needed (every 20K gradient steps)

## Known Limitations vs Official Implementation

1. **No mixed precision training:** Official uses optional half-precision, we use float32
2. **No multi-GPU support:** Official uses JAX pmap, we use single GPU
3. **Simpler optimizer state reset:** Official resets optimizer momentum for reset layers
4. **Different RNG:** PyTorch vs JAX random number generation may differ
5. **Replay buffer implementation:** Simplified subsequence sampling

## Validation Strategy

To validate correctness:

1. **Architecture check:** Print network and count parameters (~21.8M)
2. **Single seed reproduction:** Run seed 7779 on Atlantis, expect ~80K score (RR=2)
3. **Training curves:** Monitor TD loss, SPR loss, episode returns
4. **Ablations:** Remove components one-by-one, verify performance drops match paper Figure 5

## Future Improvements

To match official implementation more closely:

1. Implement mixed precision training (torch.cuda.amp)
2. Add proper optimizer state reset during shrink-and-perturb
3. Multi-GPU support with distributed training
4. More sophisticated replay buffer (exact match to official)
5. Add evaluation episodes during training (separate from training env)

