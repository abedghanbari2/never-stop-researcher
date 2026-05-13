"""Hyperparameters for the DreamerV3 sketch.

Sizes are deliberately small (vs. the paper's defaults) so the whole
thing trains on CPU in minutes on CartPole-v1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- env ---
    env_id: str = "CartPole-v1"
    seed: int = 0

    # --- world model dims ---
    obs_dim: int = 4               # set by env at runtime
    num_actions: int = 2           # set by env at runtime
    hidden: int = 128              # MLP / GRU hidden width
    deter_dim: int = 128           # GRU recurrent state size
    num_categoricals: int = 8      # N_cat
    classes_per_cat: int = 8       # K  ->  stoch state has 8*8 = 64 dims
    num_bins: int = 41             # two-hot grid resolution
    symlog_low: float = -20.0
    symlog_high: float = 20.0

    # --- losses ---
    kl_alpha: float = 0.8
    kl_free_bits: float = 1.0
    recon_weight: float = 1.0
    reward_weight: float = 1.0
    cont_weight: float = 1.0
    kl_weight: float = 1.0

    # --- actor / critic ---
    horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95
    actor_entropy: float = 1e-2
    target_ema: float = 0.98

    # --- optim ---
    wm_lr: float = 1e-3
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    grad_clip: float = 100.0

    # --- training schedule ---
    total_steps: int = 20_000
    warmup_steps: int = 500
    train_every: int = 4           # gradient step every N env steps
    batch_size: int = 16
    seq_len: int = 16
    log_every: int = 500

    # --- replay ---
    buffer_capacity: int = 100_000
