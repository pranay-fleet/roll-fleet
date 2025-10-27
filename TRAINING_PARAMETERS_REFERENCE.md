# ROLL Agentic Training: Complete Parameter Reference

**Configuration File:** `examples/qwen2.5-vl-3B-agentic/agentic_click_and_read.yaml`  
**Last Updated:** October 26, 2025

---

## Table of Contents
1. [Quick Reference](#quick-reference)
2. [System Architecture](#system-architecture)
3. [Environment Configuration](#environment-configuration)
4. [Batch Size Calculations](#batch-size-calculations)
5. [Training Loop Breakdown](#training-loop-breakdown)
6. [Memory Distribution](#memory-distribution)
7. [GRPO Baseline Computation](#grpo-baseline-computation)
8. [Parameter Relationships](#parameter-relationships)
9. [Modification Guide](#modification-guide)

---

## Quick Reference

### Key Numbers at a Glance

```yaml
# Training Scale
Total GPUs:                8
Training Steps:            1,024
Gradient Updates:          2,048 (2 per step)
Training Trajectories:     524,288 total

# Batch Sizes
Rollout Batch:            512 trajectories/step
Per-GPU Micro-batch:      2 trajectories
Mini-batch (all GPUs):    16 trajectories
Effective Batch:          256 trajectories
Gradient Accumulation:    16 steps

# Environment Groups (GRPO)
Training Groups:          64 groups
Samples per Group:        8 trajectories
Total per Step:           512 trajectories

# Evaluation
Eval Frequency:           Every 10 steps
Eval Trajectories:        512 per eval
Total Evals:              ~102
```

---

## System Architecture

### Hardware Layout

```
8 x GPUs (num_gpus_per_node: 8)
├─ Actor Training:    GPUs 0-7 (DeepSpeed ZeRO-3)
├─ Actor Inference:   GPUs 0-7 (vLLM, shared)
└─ Reference Model:   GPUs 0-7 (HuggingFace, shared)
```

**Note:** Models share GPUs but run at different times in the pipeline.

### Model Configuration

```yaml
Model: Qwen/Qwen2.5-VL-3B-Instruct
Parameters: ~3 billion
Precision: bf16
Context Length: 8,192 tokens
Max Actions/Trajectory: 5
Max Tokens/Action: 128
```

---

## Environment Configuration

### Training Environments

```yaml
train_env_manager:
  num_env_groups: 64        # Unique environment configurations
  group_size: 8             # Trajectories per group (for GRPO baseline)
  max_env_num_per_worker: 8 # Max environments per CPU worker
  tags: [ClickAndRead]
```

**Structure:**
- **64 unique scenarios** (e.g., different webpages, different questions)
- **8 attempts per scenario** (same initial state, stochastic policy)
- **Total: 512 trajectories per rollout**

```
Group 0:  [Traj 0, Traj 1, ..., Traj 7]   → Same env config/seed
Group 1:  [Traj 8, Traj 9, ..., Traj 15]  → Different env config/seed
...
Group 63: [Traj 504, ..., Traj 511]       → Different env config/seed
```

### Validation Environments

```yaml
val_env_manager:
  num_env_groups: 512       # More diverse scenarios
  group_size: 1             # One attempt per scenario
  max_env_num_per_worker: 16
```

**Why group_size=1 for validation?**
- Validation uses deterministic sampling (temperature ≈ 0)
- Same prompt + deterministic = identical outputs
- So we test on 512 unique scenarios instead

---

## Batch Size Calculations

### Formula Chain

```
┌─────────────────────────────────────────────────────────┐
│ ROLLOUT BATCH SIZE                                      │
└─────────────────────────────────────────────────────────┘

rollout_batch_size = num_env_groups × group_size
                   = 64 × 8
                   = 512 trajectories

┌─────────────────────────────────────────────────────────┐
│ MICRO-BATCH (Per GPU, per forward pass)                │
└─────────────────────────────────────────────────────────┘

micro_batch = per_device_train_batch_size
            = 2 trajectories

┌─────────────────────────────────────────────────────────┐
│ MINI-BATCH (All GPUs, per forward pass)                │
└─────────────────────────────────────────────────────────┘

mini_batch = micro_batch × num_data_parallel_gpus
           = 2 × 8
           = 16 trajectories

┌─────────────────────────────────────────────────────────┐
│ EFFECTIVE TRAINING BATCH (After gradient accumulation) │
└─────────────────────────────────────────────────────────┘

effective_batch = mini_batch × gradient_accumulation_steps
                = 16 × 16
                = 256 trajectories

┌─────────────────────────────────────────────────────────┐
│ GRADIENT UPDATES PER STEP                               │
└─────────────────────────────────────────────────────────┘

updates_per_step = (rollout_batch_size / effective_batch) × ppo_epochs
                 = (512 / 256) × 1
                 = 2 gradient updates
```

### Why This Matters

- **Micro-batch (2):** Small enough to fit in GPU memory during training
- **Gradient accumulation (16):** Simulates larger batch without memory overhead
- **Effective batch (256):** Large enough for stable policy updates
- **2 updates/step:** Processes all 512 rollout samples

---

## Training Loop Breakdown

### Complete Training Step (Step N)

```
═══════════════════════════════════════════════════════════════
PHASE 1: ROLLOUT (Generate 512 Trajectories)
═══════════════════════════════════════════════════════════════

Hardware: vLLM on GPUs 0-7
Time: ~2-5 minutes
Memory: 20-30 GB per GPU

Process:
├─ Initialize 64 groups × 8 parallel environments = 512 envs
├─ For each action (up to 5 per trajectory):
│   ├─ All 512 envs send observation → policy
│   ├─ vLLM generates actions (continuous batching)
│   ├─ Envs execute actions and return rewards
│   └─ Check if done, else continue
│
└─ Output to CPU RAM (~10 GB):
    ├─ input_ids: [512, ~8192] tokens
    ├─ attention_mask: [512, ~8192]
    ├─ response_mask: [512, ~8192] (which tokens are generated)
    ├─ prompt_mask: [512, ~8192] (which tokens are prompt)
    ├─ old_log_probs: [512, ~8192] (from actor during generation)
    └─ scores: [512] (episode rewards from environment)

═══════════════════════════════════════════════════════════════
PHASE 2: REFERENCE LOG PROBS (512 Trajectories)
═══════════════════════════════════════════════════════════════

Hardware: HuggingFace Inference on GPUs 0-7
Time: ~1-2 minutes
Memory: 15-20 GB per GPU

Process:
├─ Load frozen reference model (initial policy checkpoint)
├─ Process in batches: 2 per GPU × 8 GPUs = 16 at a time
├─ Total forward passes: 512 ÷ 16 = 32 batches
│
└─ For each batch:
    ├─ Load input_ids from CPU → GPU
    ├─ Forward pass through reference model
    ├─ Extract log_probs for response tokens only
    └─ Save ref_log_probs to CPU RAM

Output to CPU RAM (+4 GB):
└─ ref_log_probs: [512, ~8192]

Purpose: Compute KL divergence penalty (if init_kl_coef > 0)

═══════════════════════════════════════════════════════════════
PHASE 3: ADVANTAGE COMPUTATION (CPU)
═══════════════════════════════════════════════════════════════

Hardware: CPU only
Time: ~10-30 seconds
Memory: ~2 GB CPU RAM

Process:
├─ Step 3a: Group-based Reward Normalization (GRPO Baseline)
│   ├─ Group 512 trajectories by traj_group_id (64 groups)
│   ├─ For each group of 8 trajectories:
│   │   ├─ group_mean = mean(8 rewards)
│   │   ├─ group_std = std(8 rewards)
│   │   └─ normalized_reward = (reward - group_mean) / (group_std + ε)
│   └─ This creates relative advantages within each group
│
├─ Step 3b: Apply KL Penalty (Optional)
│   └─ If init_kl_coef > 0:
│       └─ reward = reward - kl_coef × (old_log_prob - ref_log_prob)
│
├─ Step 3c: Expand to Token Level
│   └─ response_level_rewards → token_level_rewards
│       (Assign episode reward to final token or distribute)
│
├─ Step 3d: Compute Advantages (REINFORCE/GRPO)
│   ├─ advantages = discounted_returns(token_level_rewards)
│   └─ For GRPO/REINFORCE with gamma=1: advantages ≈ token_level_rewards
│
└─ Step 3e: Whiten Advantages (Optional)
    └─ If whiten_advantages=True:
        └─ advantages = (advantages - mean) / (std + ε)

Output to CPU RAM (+2 GB):
├─ response_level_rewards: [512]
├─ token_level_rewards: [512, ~8192]
├─ advantages: [512, ~8192]
└─ returns: [512, ~8192]

═══════════════════════════════════════════════════════════════
PHASE 4: TRAINING (2 Gradient Updates)
═══════════════════════════════════════════════════════════════

Hardware: DeepSpeed ZeRO-3 on GPUs 0-7
Time: ~30-60 seconds per update
Memory: 35-45 GB per GPU

─────────────────────────────────────────────────────────────
GRADIENT UPDATE 1: Process First 256 Trajectories
─────────────────────────────────────────────────────────────

For micro_step in range(16):  # gradient_accumulation_steps
    
    ┌─ Micro-step i (i=0..15) ─────────────────────────────┐
    │                                                        │
    │ 1. Select Data:                                        │
    │    └─ 16 trajectories (2 per GPU × 8 GPUs)           │
    │                                                        │
    │ 2. Load to GPU:                                        │
    │    ├─ input_ids, attention_mask, position_ids         │
    │    ├─ advantages (pre-computed)                       │
    │    ├─ old_log_probs (from rollout)                    │
    │    └─ response_mask (which tokens to compute loss on) │
    │                                                        │
    │ 3. Forward Pass (Current Policy):                     │
    │    ├─ logits = actor_model(input_ids)                 │
    │    └─ new_log_probs = log_softmax(logits)[tokens]     │
    │                                                        │
    │ 4. Compute PPO Loss:                                   │
    │    ├─ ratio = exp(new_log_probs - old_log_probs)      │
    │    ├─ clipped_ratio = clamp(ratio, 0.8, 1.2)          │
    │    ├─ policy_loss = -min(ratio × adv, clip × adv)     │
    │    ├─ entropy_loss = -mean(entropy) [if coef > 0]     │
    │    └─ total_loss = policy_loss + entropy_coef × ent   │
    │                                                        │
    │ 5. Backward Pass:                                      │
    │    ├─ total_loss.backward()                           │
    │    └─ Gradients accumulated (not applied yet)         │
    │                                                        │
    │ 6. Clear Activation Memory                             │
    └────────────────────────────────────────────────────────┘

After 16 micro-steps:
├─ All-reduce gradients across 8 GPUs (DeepSpeed handles)
├─ Clip gradients: norm = min(norm, max_grad_norm=1.0)
├─ Optimizer step: Update model parameters
├─ Zero gradients
└─ LR scheduler step (cosine schedule)

Progress: Processed 256/512 trajectories

─────────────────────────────────────────────────────────────
GRADIENT UPDATE 2: Process Remaining 256 Trajectories
─────────────────────────────────────────────────────────────

[Identical process to Update 1]

Progress: Processed 512/512 trajectories ✓

═══════════════════════════════════════════════════════════════
PHASE 5: LOGGING & CLEANUP
═══════════════════════════════════════════════════════════════

Time: ~1-5 seconds

Compute Metrics:
├─ critic/score/mean: Average episode reward
├─ critic/reward/mean: Average normalized reward
├─ critic/advantage/mean: Average advantage
├─ policy/loss: PPO policy loss
├─ policy/clip_frac: Fraction of ratios clipped
├─ policy/entropy: Policy entropy
├─ policy/kl: KL divergence from reference
├─ policy/approx_kl: KL divergence from old policy
└─ time/*: Time spent in each phase

Log to W&B/TensorBoard

Clear CPU RAM of step data (keep only model states)

════════════════════════════════════════════════════════════════
TOTAL TIME PER STEP: ~5-10 minutes
MEMORY USAGE:
├─ GPU: 35-45 GB per GPU (during training)
└─ CPU: ~20-30 GB (storing rollout data)
════════════════════════════════════════════════════════════════
```

### Evaluation Step (Every 10 Training Steps)

```
═══════════════════════════════════════════════════════════════
EVALUATION ROLLOUT
═══════════════════════════════════════════════════════════════

Frequency: Every 10 training steps (eval_steps: 10)
Trajectories: 512 unique scenarios (group_size: 1)
Sampling: Deterministic (temperature ≈ 0)
Purpose: Measure generalization

Process:
├─ Load current actor model
├─ Run 512 trajectories on validation environments
├─ Collect metrics (success rate, avg reward, etc.)
└─ Log to W&B/TensorBoard

NO TRAINING - Evaluation only!

Time: ~2-5 minutes
Memory: ~20-30 GB per GPU
```

---

## Memory Distribution

### Memory by Phase

```
┌─────────────────────────────────────────────────────────┐
│ PHASE 1: ROLLOUT (vLLM Inference)                      │
├─────────────────────────────────────────────────────────┤
│ GPU Memory (per GPU): 20-30 GB                         │
│ ├─ Model weights (shared): ~6 GB                       │
│ ├─ KV cache (PagedAttention): ~10-15 GB                │
│ └─ Activations: ~5-10 GB                               │
│                                                          │
│ CPU RAM: ~10 GB                                         │
│ └─ Rollout data: 512 trajectories                      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PHASE 2: REFERENCE MODEL (HF Inference)                │
├─────────────────────────────────────────────────────────┤
│ GPU Memory (per GPU): 15-20 GB                         │
│ ├─ Model weights: ~6 GB                                │
│ └─ Activations (batch=2): ~8-12 GB                     │
│                                                          │
│ CPU RAM: +4 GB                                          │
│ └─ ref_log_probs: [512, ~8192]                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PHASE 3: ADVANTAGE COMPUTATION (CPU)                   │
├─────────────────────────────────────────────────────────┤
│ CPU RAM: +2 GB                                          │
│ └─ Advantages, returns, normalized rewards             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PHASE 4: TRAINING (DeepSpeed ZeRO-3)                   │
├─────────────────────────────────────────────────────────┤
│ GPU Memory (per GPU): 35-45 GB                         │
│ ├─ Model parameters (sharded): ~6 GB                   │
│ ├─ Optimizer states (sharded): ~12 GB                  │
│ ├─ Gradients (accumulated): ~6 GB                      │
│ ├─ Activations (micro-batch=2): ~8-12 GB               │
│ └─ Input data: ~2-4 GB                                 │
│                                                          │
│ DeepSpeed ZeRO-3 Sharding:                             │
│ ├─ Each GPU holds 1/8 of parameters                    │
│ ├─ All-gather during forward/backward                  │
│ └─ Saves ~50% memory vs. DDP                           │
└─────────────────────────────────────────────────────────┘
```

### Total System Memory

```
Peak Memory Usage:
├─ GPU: 35-45 GB per GPU × 8 GPUs = 280-360 GB total
└─ CPU: ~20-30 GB

Storage (Disk):
├─ Model checkpoints: ~6 GB each
├─ Logs/metrics: ~100 MB per 100 steps
└─ Optional trajectory saves: ~10 GB per step
```

---

## GRPO Baseline Computation

### How GRPO Works

**GRPO = Group Relative Policy Optimization**

Unlike PPO with GAE (which uses a learned critic/value function), GRPO computes advantages using **group statistics as the baseline**.

### Step-by-Step Process

```python
# Step 1: Collect 512 trajectories in 64 groups of 8
rollout_data = {
    "traj_group_id": [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, ..., 63,63,...],
    "scores": [0.8, 0.6, 0.9, 0.5, 0.7, 0.4, 0.85, 0.75, ...]
}

# Step 2: Group by traj_group_id
for group_id in range(64):
    group_scores = scores[group_id * 8 : (group_id + 1) * 8]
    
    # Step 3: Compute group baseline (mean and std)
    group_mean = mean(group_scores)  # This is the "baseline"
    group_std = std(group_scores)
    
    # Step 4: Normalize within group
    for i in range(8):
        normalized_scores[group_id * 8 + i] = (
            (group_scores[i] - group_mean) / (group_std + 1e-6)
        )
```

### Concrete Example

**Group 5 (8 trajectories on same webpage task):**

```
Raw Rewards:
Traj 40: 0.8  (clicked correct link, high reward)
Traj 41: 0.6  (correct but slow)
Traj 42: 0.9  (perfect!)
Traj 43: 0.5  (partially correct)
Traj 44: 0.7  (good)
Traj 45: 0.4  (mostly wrong)
Traj 46: 0.85 (very good)
Traj 47: 0.75 (good)

Group Statistics:
├─ Mean (baseline): 0.6875
└─ Std: 0.175

Normalized Rewards (advantages):
Traj 40: (0.8 - 0.6875) / 0.175 = +0.64  ← Above average
Traj 41: (0.6 - 0.6875) / 0.175 = -0.50  ← Below average
Traj 42: (0.9 - 0.6875) / 0.175 = +1.21  ← Best!
Traj 43: (0.5 - 0.6875) / 0.175 = -1.07  ← Worst
Traj 44: (0.7 - 0.6875) / 0.175 = +0.07  ← Slightly above
Traj 45: (0.4 - 0.6875) / 0.175 = -1.64  ← Very bad
Traj 46: (0.85 - 0.6875) / 0.175 = +0.93 ← Great
Traj 47: (0.75 - 0.6875) / 0.175 = +0.36 ← Good
```

### Policy Gradient Impact

```python
# During training, the gradient for each trajectory is:
gradient = advantage × ∇log_π(action|state)

# Positive advantage → Increase probability of this trajectory
# Negative advantage → Decrease probability of this trajectory
# Zero advantage → No update (neutral)

# This means:
# - Better-than-average trajectories get reinforced
# - Worse-than-average trajectories get discouraged
# - All relative to the group's performance
```

### Why Group-Based?

1. **Same initial conditions:** All 8 trajectories in a group start from the same environment state
2. **Fair comparison:** Compare apples to apples (same task difficulty)
3. **Variance reduction:** Normalizing within group reduces noise
4. **No critic needed:** Group mean serves as baseline (simpler than learning value function)

### Configuration Impact

```yaml
reward_normalization:
  grouping: traj_group_id  # Group by trajectory group
  method: mean_std         # Normalize using (x - mean) / std

# Alternative options:
# grouping: batch          # Normalize across entire batch (512 trajectories)
# grouping: tags           # Group by environment type
# method: mean             # Just subtract mean, don't divide by std
# method: identity         # No normalization
```

---

## Parameter Relationships

### Critical Constraints

```python
# Constraint 1: Rollout batch must match environment groups
rollout_batch_size == num_env_groups × group_size
512 == 64 × 8 ✓

# Constraint 2: Effective batch should divide rollout batch evenly
rollout_batch_size % effective_batch_size == 0
512 % 256 == 0 ✓

# Constraint 3: GPU memory constraint
per_device_batch × gradient_accum × sequence_length × hidden_size < GPU_memory
2 × 16 × 8192 × 3584 < 40GB ✓ (with ZeRO-3)

# Constraint 4: Validation group_size should be 1 if deterministic
if temperature == 0: val_group_size == 1
```

### Formulas

```python
# Basic relationships
mini_batch = per_device_train_batch_size × num_gpus
effective_batch = mini_batch × gradient_accumulation_steps
gradient_updates_per_step = (rollout_batch_size / effective_batch) × ppo_epochs

# Total training
total_gradient_updates = max_steps × gradient_updates_per_step
total_train_trajectories = max_steps × rollout_batch_size
total_eval_trajectories = (max_steps / eval_steps) × val_batch_size
total_trajectories = total_train_trajectories + total_eval_trajectories

# Time estimates (rough)
time_per_rollout = rollout_batch_size / (vllm_throughput_tokens_per_sec)
time_per_gradient_update = 30-60 seconds
time_per_step = time_per_rollout + time_per_ref + time_per_train
total_training_time = max_steps × time_per_step

# Memory
gpu_memory_per_batch = (
    model_size + 
    optimizer_states + 
    gradients + 
    activations(per_device_batch, sequence_length)
)

# With ZeRO-3, each GPU holds:
gpu_memory = (
    model_size / num_gpus +           # Sharded parameters
    optimizer_states / num_gpus +      # Sharded optimizer
    gradients / num_gpus +             # Sharded gradients
    activations(per_device_batch)      # Not sharded
)
```

### Your Configuration Values

```python
# Environment
num_env_groups = 64
group_size = 8
rollout_batch_size = 512

# Training
num_gpus = 8
per_device_train_batch_size = 2
gradient_accumulation_steps = 16
ppo_epochs = 1

# Derived
mini_batch = 2 × 8 = 16
effective_batch = 16 × 16 = 256
updates_per_step = (512 / 256) × 1 = 2

# Scale
max_steps = 1024
total_updates = 1024 × 2 = 2048
total_trajectories = 1024 × 512 = 524,288

# Evaluation
eval_steps = 10
num_evals = ceil(1024 / 10) = 103
eval_trajectories = 103 × 512 = 52,736
```

---

## Modification Guide

### Scaling Up Training

#### Option 1: Increase Effective Batch Size

**Goal:** More stable updates, better convergence

```yaml
# Current: effective_batch = 256
# Option A: Double gradient accumulation
gradient_accumulation_steps: 32  # Was 16
# Result: effective_batch = 512, updates_per_step = 1

# Option B: Double per-device batch
per_device_train_batch_size: 4  # Was 2
gradient_accumulation_steps: 8  # Adjust to compensate
# Result: effective_batch = 512, FASTER but needs more GPU memory

# Memory impact:
# Option A: No change (same micro-batch)
# Option B: ~1.5-2× more GPU memory needed
```

#### Option 2: More Data per Step

**Goal:** More diverse training data

```yaml
# Option A: More groups, same group size
num_env_groups: 128     # Was 64
group_size: 8           # Same
rollout_batch_size: 1024  # Was 512
# Result: 128 unique scenarios, 8 samples each

# Option B: Larger groups
num_env_groups: 64      # Same
group_size: 16          # Was 8
rollout_batch_size: 1024  # Was 512
# Result: 64 scenarios, 16 samples each (better GRPO baseline)

# Training time impact: ~2× longer per step
# Memory impact: Need to adjust effective_batch or ppo_epochs
```

#### Option 3: More PPO Epochs

**Goal:** Better utilize each rollout

```yaml
ppo_epochs: 4  # Was 1
# Result: 4× more gradient updates per step (8 instead of 2)
# Time per step: ~3-4× longer
# Risk: Overfitting to each batch
```

### Reducing Memory Usage

#### If Getting OOM (Out of Memory)

```yaml
# Option 1: Reduce per-device batch (safest)
per_device_train_batch_size: 1  # Was 2
gradient_accumulation_steps: 32  # Was 16
# Result: Same effective batch, but slower

# Option 2: Reduce sequence length
sequence_length: 4096  # Was 8192
# Result: Less memory, but truncates longer trajectories

# Option 3: Enable CPU offloading
strategy_config: ${deepspeed_zero3_cpuoffload}
# Result: Offloads optimizer states to CPU RAM, slower but uses less GPU

# Option 4: Reduce rollout batch
num_env_groups: 32      # Was 64
rollout_batch_size: 256  # Was 512
# Adjust effective batch accordingly
```

### Improving Sample Efficiency

#### Option 1: Better GRPO Baseline

```yaml
# Increase group size for more stable baseline
group_size: 16  # Was 8
num_env_groups: 32  # Was 64 (to keep rollout_batch_size same)
# Result: More samples per baseline = lower variance
```

#### Option 2: Add KL Penalty

```yaml
# Prevent policy from diverging too much from initial
init_kl_coef: 0.02  # Was 0.0
# Result: More conservative updates, may need more steps
```

#### Option 3: Adjust PPO Clipping

```yaml
# Current: clips ratio to [0.8, 1.2]
advantage_clip: 0.1  # Was 0.2 (more conservative)
# Or uncomment:
pg_clip: 0.1          # Policy gradient ratio clipping
dual_clip_loss: True  # Dual clip (helps with large advantages)
```

### Faster Training

```yaml
# Option 1: Larger micro-batches (if memory allows)
per_device_train_batch_size: 4
gradient_accumulation_steps: 8

# Option 2: Fewer evaluation steps
eval_steps: 20  # Was 10 (eval less frequently)

# Option 3: Shorter trajectories
max_actions_per_traj: 3  # Was 5

# Option 4: Use Megatron (tensor parallelism)
strategy_name: megatron_train
strategy_config:
  tensor_model_parallel_size: 2  # Split model across 2 GPUs
  pipeline_model_parallel_size: 1
# Result: Can fit larger models, but needs careful tuning
```

### Common Pitfalls

```yaml
# ❌ BAD: group_size > 1 with deterministic validation
val_env_manager:
  group_size: 8  # With temperature=0, all 8 will be identical!

# ✓ GOOD: group_size = 1 for deterministic sampling
val_env_manager:
  group_size: 1

# ❌ BAD: effective_batch doesn't divide rollout_batch
rollout_batch_size: 512
effective_batch: 300  # 512 / 300 = 1.7 updates (fractional!)

# ✓ GOOD: evenly divides
rollout_batch_size: 512
effective_batch: 256  # 512 / 256 = 2 updates ✓

# ❌ BAD: Too large per_device_batch
per_device_train_batch_size: 16  # Likely OOM on 40GB GPU

# ✓ GOOD: Conservative batch size
per_device_train_batch_size: 2
gradient_accumulation_steps: 16  # Achieve large effective batch safely
```

---

## Debugging & Monitoring

### Key Metrics to Watch

```python
# Training health
"policy/loss"                  # Should decrease
"policy/approx_kl"             # Should be small (< 0.05)
"policy/clip_frac"             # 0.1-0.3 is good range
"policy/entropy"               # Should stay reasonable (> 0.1)

# Reward progression
"critic/score/mean"            # Should increase over time
"critic/reward/mean"           # After normalization (mean ≈ 0)
"critic/advantage/mean"        # Should be near 0 after whitening

# GRPO-specific
"group/advantage_diff/mean"    # Difference between max and min in group
"group/all_correct_ratio"      # Fraction of groups where all succeeded

# System performance
"time/rollout"                 # Rollout time per step
"time/actor_train"             # Training time per step
"tps/actor_train"              # Tokens per second during training
```

### Common Issues

**1. Loss not decreasing:**
- Check learning rate (try 5e-7 to 5e-6)
- Check if advantages are too noisy (increase group_size)
- Check if clipping is too aggressive (increase advantage_clip)

**2. Policy collapse (entropy → 0):**
- Add entropy bonus: `entropy_loss_coef: 0.01`
- Reduce learning rate
- Check if rewards are too sparse

**3. OOM during training:**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps` proportionally
- Enable `deepspeed_zero3_cpuoffload`

**4. Slow rollouts:**
- Check vLLM GPU utilization
- Reduce `max_new_tokens` if possible
- Increase `max_env_num_per_worker`

---

## Quick Reference Card

Print this for quick lookups!

```
╔═══════════════════════════════════════════════════════════╗
║        ROLL AGENTIC TRAINING QUICK REFERENCE              ║
╠═══════════════════════════════════════════════════════════╣
║ BATCH SIZES                                               ║
║ • Rollout:           512 trajectories                     ║
║ • Micro-batch:       2 per GPU                            ║
║ • Mini-batch:        16 (all GPUs)                        ║
║ • Effective:         256 (after grad accum)               ║
║ • Updates/step:      2                                    ║
╠═══════════════════════════════════════════════════════════╣
║ GRPO GROUPS                                               ║
║ • Training groups:   64 unique scenarios                  ║
║ • Samples/group:     8 (for baseline)                     ║
║ • Baseline:          Mean of 8 samples                    ║
║ • Normalization:     (reward - mean) / std                ║
╠═══════════════════════════════════════════════════════════╣
║ TRAINING SCALE                                            ║
║ • Total steps:       1,024                                ║
║ • Gradient updates:  2,048                                ║
║ • Train trajs:       524,288                              ║
║ • Eval frequency:    Every 10 steps                       ║
║ • Eval trajs:        512 per eval                         ║
╠═══════════════════════════════════════════════════════════╣
║ TIME ESTIMATES                                            ║
║ • Per step:          5-10 minutes                         ║
║ • Per eval:          2-5 minutes                          ║
║ • Total training:    ~85-170 hours                        ║
╠═══════════════════════════════════════════════════════════╣
║ MEMORY                                                    ║
║ • GPU (training):    35-45 GB per GPU                     ║
║ • GPU (inference):   20-30 GB per GPU                     ║
║ • CPU RAM:           20-30 GB                             ║
║ • Checkpoint size:   ~6 GB                                ║
╠═══════════════════════════════════════════════════════════╣
║ KEY FORMULAS                                              ║
║ • rollout = groups × group_size                           ║
║ • mini_batch = per_device × num_gpus                      ║
║ • effective = mini_batch × grad_accum                     ║
║ • updates/step = (rollout / effective) × ppo_epochs       ║
╚═══════════════════════════════════════════════════════════╝
```

---

## Additional Resources

- **Original Config:** `examples/qwen2.5-vl-3B-agentic/agentic_click_and_read.yaml`
- **Pipeline Code:** `roll/pipeline/agentic/agentic_pipeline.py`
- **GRPO Utils:** `roll/pipeline/agentic/utils.py`
- **Environment:** `roll/pipeline/agentic/env/click_and_read/env.py`

---

**Document Version:** 1.0  
**Created:** October 26, 2025  
**Config Hash:** Based on default `agentic_click_and_read.yaml`

