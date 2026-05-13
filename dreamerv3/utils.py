"""Numerical helpers shared across the DreamerV3 sketch.

Implements the small, distinctively-v3 tricks:

* ``symlog`` / ``symexp`` — robust scale handling for vector observations
  and reward targets (DreamerV3 paper, eq. 1).
* Two-hot encoding/decoding — represent scalar regression targets as a
  categorical over a fixed symlog-spaced grid (eq. 4).
* KL with free bits — clip per-batch KL below a floor before averaging.
* Straight-through one-hot sampler — gradients flow through a discrete
  latent sample as if it were the categorical's probabilities.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Symlog                                                                      #
# --------------------------------------------------------------------------- #

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


# --------------------------------------------------------------------------- #
# Two-hot                                                                     #
# --------------------------------------------------------------------------- #

def make_two_hot_bins(num_bins: int, low: float = -20.0, high: float = 20.0,
                      device=None) -> torch.Tensor:
    """Symlog-spaced bin centers used by the two-hot heads."""
    return torch.linspace(low, high, num_bins, device=device)


def two_hot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Encode scalar values ``x`` (in symlog space) as a two-hot distribution.

    Args:
        x:    Tensor of shape ``(...,)``.
        bins: 1D tensor of bin centers, length ``B``.
    Returns:
        Tensor of shape ``(..., B)`` summing to 1 along the last axis.
    """
    x = x.clamp(bins[0].item(), bins[-1].item())
    # Find the right-hand bin index for each value.
    idx_upper = torch.bucketize(x.contiguous(), bins)
    idx_upper = idx_upper.clamp(1, bins.numel() - 1)
    idx_lower = idx_upper - 1
    lower = bins[idx_lower]
    upper = bins[idx_upper]
    weight_upper = (x - lower) / (upper - lower + 1e-8)
    weight_lower = 1.0 - weight_upper
    out = torch.zeros(*x.shape, bins.numel(), device=x.device, dtype=x.dtype)
    out.scatter_(-1, idx_lower.unsqueeze(-1), weight_lower.unsqueeze(-1))
    out.scatter_add_(-1, idx_upper.unsqueeze(-1), weight_upper.unsqueeze(-1))
    return out


def two_hot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Decode logits over the bin grid back into a scalar in symlog space."""
    probs = F.softmax(logits, dim=-1)
    return (probs * bins).sum(dim=-1)


def two_hot_loss(logits: torch.Tensor, target: torch.Tensor,
                 bins: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between the two-hot target distribution and ``logits``."""
    target_dist = two_hot_encode(target, bins).detach()
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1)


# --------------------------------------------------------------------------- #
# KL with free bits                                                           #
# --------------------------------------------------------------------------- #

def categorical_kl(p_logits: torch.Tensor, q_logits: torch.Tensor
                   ) -> torch.Tensor:
    """KL(p || q) for independent categoricals along the last axis."""
    p_log = F.log_softmax(p_logits, dim=-1)
    q_log = F.log_softmax(q_logits, dim=-1)
    p = p_log.exp()
    return (p * (p_log - q_log)).sum(dim=-1)


def kl_balance_loss(post_logits: torch.Tensor, prior_logits: torch.Tensor,
                    alpha: float = 0.8, free_bits: float = 1.0
                    ) -> torch.Tensor:
    """KL balancing as in DreamerV2/V3.

    ``alpha`` weights the prior-update term (sg on posterior).
    ``free_bits`` is a floor applied per-step before averaging.
    Inputs have shape ``(..., N_cat, K)``; KL is summed over ``N_cat``.
    """
    kl_prior = categorical_kl(post_logits.detach(), prior_logits)
    kl_post = categorical_kl(post_logits, prior_logits.detach())
    # Sum over the N_cat axis -> per-step KL in nats.
    kl_prior = kl_prior.sum(dim=-1)
    kl_post = kl_post.sum(dim=-1)
    kl_prior = kl_prior.clamp(min=free_bits)
    kl_post = kl_post.clamp(min=free_bits)
    return alpha * kl_prior + (1.0 - alpha) * kl_post


# --------------------------------------------------------------------------- #
# Straight-through one-hot                                                    #
# --------------------------------------------------------------------------- #

def sample_one_hot_st(logits: torch.Tensor) -> torch.Tensor:
    """Sample a one-hot from ``logits`` with a straight-through gradient.

    ``logits`` has shape ``(..., K)``. Returns the same shape; gradient
    flows back through ``softmax(logits)``.
    """
    probs = F.softmax(logits, dim=-1)
    index = torch.distributions.Categorical(probs=probs).sample()
    one_hot = F.one_hot(index, num_classes=logits.shape[-1]).to(probs.dtype)
    return one_hot + probs - probs.detach()


# --------------------------------------------------------------------------- #
# Running percentile range (for return normalization)                         #
# --------------------------------------------------------------------------- #

class PercentileReturnNorm:
    """Tracks an EMA over the 5th/95th percentiles of returns.

    DreamerV3 divides advantages by ``max(1, p95 - p5)`` to keep scales
    consistent across tasks.
    """

    def __init__(self, decay: float = 0.99, low: float = 0.05,
                 high: float = 0.95):
        self.decay = decay
        self.low = low
        self.high = high
        self.lo = 0.0
        self.hi = 0.0
        self.initialized = False

    def update(self, returns: torch.Tensor) -> None:
        flat = returns.detach().flatten()
        if flat.numel() == 0:
            return
        lo = torch.quantile(flat, self.low).item()
        hi = torch.quantile(flat, self.high).item()
        if not self.initialized:
            self.lo, self.hi = lo, hi
            self.initialized = True
        else:
            d = self.decay
            self.lo = d * self.lo + (1 - d) * lo
            self.hi = d * self.hi + (1 - d) * hi

    def scale(self) -> float:
        return max(1.0, self.hi - self.lo)


# --------------------------------------------------------------------------- #
# Sanity checks (called from train.py at startup)                             #
# --------------------------------------------------------------------------- #

def sanity_check() -> None:
    x = torch.tensor([-12.3, -0.5, 0.0, 0.5, 17.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-5), "symlog inverse"
    bins = make_two_hot_bins(41)
    enc = two_hot_encode(symlog(x), bins)
    dec_symlog = (enc * bins).sum(dim=-1)
    assert torch.allclose(symexp(dec_symlog), x, atol=1e-3), "two-hot round-trip"
    assert math.isclose(enc.sum(-1).mean().item(), 1.0, abs_tol=1e-5)
