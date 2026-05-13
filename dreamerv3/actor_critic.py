"""Actor and critic trained on imagined rollouts from the world model.

Critic uses a two-hot head in symlog space (matching DreamerV3 eq. 4).
Actor is a categorical policy (CartPole has discrete actions) trained
with REINFORCE on percentile-normalized advantages.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .utils import (
    PercentileReturnNorm,
    make_two_hot_bins,
    sample_one_hot_st,
    symexp,
    symlog,
    two_hot_decode,
    two_hot_loss,
)
from .world_model import WorldModel, mlp


@dataclass
class Sampled:
    actions: torch.Tensor      # (B, A) one-hot with ST grad
    log_prob: torch.Tensor     # (B,)
    entropy: torch.Tensor      # (B,)


class Actor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        feat_dim = cfg.deter_dim + cfg.num_categoricals * cfg.classes_per_cat
        self.net = mlp(feat_dim, cfg.hidden, cfg.num_actions)

    def forward(self, features: torch.Tensor) -> Sampled:
        logits = self.net(features)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        idx = torch.distributions.Categorical(probs=probs).sample()
        one_hot = F.one_hot(idx, num_classes=logits.shape[-1]).to(probs.dtype)
        # ST grad through probs (lets actor entropy bonus flow normally)
        actions = one_hot + probs - probs.detach()
        log_prob = (one_hot * log_probs).sum(-1)
        entropy = -(probs * log_probs).sum(-1)
        return Sampled(actions=actions, log_prob=log_prob, entropy=entropy)

    @torch.no_grad()
    def act(self, features: torch.Tensor) -> torch.Tensor:
        """Pick a single action (returns index, not one-hot)."""
        logits = self.net(features)
        probs = F.softmax(logits, dim=-1)
        return torch.distributions.Categorical(probs=probs).sample()


class Critic(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        feat_dim = cfg.deter_dim + cfg.num_categoricals * cfg.classes_per_cat
        self.net = mlp(feat_dim, cfg.hidden, cfg.num_bins)
        self.register_buffer(
            "bins",
            make_two_hot_bins(cfg.num_bins, cfg.symlog_low, cfg.symlog_high),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns logits over the two-hot bin grid."""
        return self.net(features)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        """Decoded value, back in original (non-symlog) scale."""
        logits = self.forward(features)
        return symexp(two_hot_decode(logits, self.bins))


def lambda_returns(rewards: torch.Tensor, cont: torch.Tensor,
                   values: torch.Tensor, gamma: float, lam: float
                   ) -> torch.Tensor:
    """Compute Dreamer-style λ-returns over imagined rollouts.

    Args (all in original reward scale):
        rewards: (H,   B)   reward at step t (after taking action_t)
        cont:    (H+1, B)   episode-continue probability at step t
        values:  (H+1, B)   value(state_t)
    Returns:
        returns: (H, B)
    """
    H = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    next_ret = values[-1]
    for t in reversed(range(H)):
        # bootstrap: r_t + γ·cont_{t+1} · [(1-λ)·V_{t+1} + λ·next_ret]
        boot = (1.0 - lam) * values[t + 1] + lam * next_ret
        next_ret = rewards[t] + gamma * cont[t + 1] * boot
        returns[t] = next_ret
    return returns


class ActorCriticTrainer:
    """Bundles actor + critic + their optimizers + target critic EMA."""

    def __init__(self, cfg: Config, device):
        self.cfg = cfg
        self.device = device
        self.actor = Actor(cfg).to(device)
        self.critic = Critic(cfg).to(device)
        self.target_critic = Critic(cfg).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr
        )
        self.return_norm = PercentileReturnNorm()

    def update(self, world_model: WorldModel, init_state) -> dict:
        cfg = self.cfg
        traj = world_model.imagine(init_state, self.actor, cfg.horizon)
        feats = traj["feats"]           # (H+1, B, F)
        rewards = traj["rewards"][:-1]  # (H,   B)
        cont = traj["cont"]             # (H+1, B)

        with torch.no_grad():
            values_target = self.target_critic.value(feats)
        returns = lambda_returns(
            rewards, cont, values_target, cfg.gamma, cfg.lam
        )  # (H, B)

        self.return_norm.update(returns)
        scale = self.return_norm.scale()

        # --- actor loss (REINFORCE on normalized advantage) ---
        with torch.no_grad():
            online_values = self.critic.value(feats[:-1])   # (H, B)
        adv = (returns - online_values) / scale
        # weight by predicted continuation prob so dead branches stop counting
        with torch.no_grad():
            disc = torch.cumprod(
                cfg.gamma * cont[:-1].clamp(0.0, 1.0), dim=0
            ) / cfg.gamma  # starts at 1.0
        actor_loss = -(disc * adv.detach() * traj["log_probs"]).mean()
        actor_loss = actor_loss - cfg.actor_entropy * traj["entropies"].mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
        self.actor_opt.step()

        # --- critic loss (two-hot in symlog space) ---
        target_symlog = symlog(returns.detach())
        v_logits = self.critic(feats[:-1].detach())
        critic_loss = two_hot_loss(
            v_logits, target_symlog, self.critic.bins
        ).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_opt.step()

        # --- EMA update for target critic ---
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(),
                             self.target_critic.parameters()):
                tp.mul_(cfg.target_ema).add_(p.detach(), alpha=1 - cfg.target_ema)

        return {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "return_mean": returns.mean().detach(),
            "return_scale": torch.tensor(scale),
            "entropy": traj["entropies"].mean().detach(),
        }
