"""Recurrent state-space world model with discrete stochastic latents.

The model carries state ``(h, z)`` where:

* ``h`` is a deterministic recurrent state produced by a GRU.
* ``z`` is a stochastic state: ``N_cat`` categorical variables of ``K``
  classes each, flattened to ``(N_cat * K,)``. We sample ``z`` with a
  straight-through one-hot trick so gradients flow.

For each step in a sequence, the model produces a *prior* over ``z``
from ``(h,)`` and a *posterior* over ``z`` from ``(h, encoded_obs)``.
KL between the two trains the prior to match what posterior actually
extracts from observations.

The decoder, reward head, and continue head all consume ``(h, z)`` and
together provide the reconstruction loss that grounds the latent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .utils import (
    kl_balance_loss,
    make_two_hot_bins,
    sample_one_hot_st,
    symlog,
    two_hot_loss,
)


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Module:
    mods = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.SiLU()]
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)


@dataclass
class State:
    h: torch.Tensor                # (..., deter_dim)
    z: torch.Tensor                # (..., N_cat * K)  one-hot flattened
    z_logits: torch.Tensor         # (..., N_cat, K)

    def features(self) -> torch.Tensor:
        return torch.cat([self.h, self.z], dim=-1)

    def detach(self) -> "State":
        return State(self.h.detach(), self.z.detach(), self.z_logits.detach())


class RSSM(nn.Module):
    """Recurrent state-space model (deterministic GRU + discrete stoch state)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        stoch_dim = cfg.num_categoricals * cfg.classes_per_cat

        # action + previous stoch state -> GRU input
        self.pre_gru = nn.Sequential(
            nn.Linear(stoch_dim + cfg.num_actions, cfg.hidden), nn.SiLU(),
        )
        self.gru = nn.GRUCell(cfg.hidden, cfg.deter_dim)

        # prior: h -> logits over N_cat * K
        self.prior_head = mlp(cfg.deter_dim, cfg.hidden, stoch_dim)
        # posterior: (h, encoded_obs) -> logits
        self.post_head = mlp(cfg.deter_dim + cfg.hidden, cfg.hidden, stoch_dim)

    @property
    def stoch_shape(self):
        return (self.cfg.num_categoricals, self.cfg.classes_per_cat)

    def initial_state(self, batch_size: int, device) -> State:
        h = torch.zeros(batch_size, self.cfg.deter_dim, device=device)
        z_logits = torch.zeros(batch_size, *self.stoch_shape, device=device)
        z = F.softmax(z_logits, dim=-1).reshape(batch_size, -1)
        return State(h, z, z_logits)

    # ------------------------------------------------------------------ #
    # Single step                                                        #
    # ------------------------------------------------------------------ #

    def img_step(self, prev: State, action_onehot: torch.Tensor) -> State:
        """Advance the deterministic state and sample from the prior."""
        x = torch.cat([prev.z, action_onehot], dim=-1)
        x = self.pre_gru(x)
        h = self.gru(x, prev.h)
        logits = self.prior_head(h).reshape(-1, *self.stoch_shape)
        z = sample_one_hot_st(logits).reshape(h.shape[0], -1)
        return State(h, z, logits)

    def obs_step(self, prev: State, action_onehot: torch.Tensor,
                 enc_obs: torch.Tensor) -> tuple[State, torch.Tensor]:
        """Advance to ``h``, then form posterior from ``(h, enc_obs)``."""
        x = torch.cat([prev.z, action_onehot], dim=-1)
        x = self.pre_gru(x)
        h = self.gru(x, prev.h)
        prior_logits = self.prior_head(h).reshape(-1, *self.stoch_shape)
        post_logits = self.post_head(torch.cat([h, enc_obs], dim=-1)
                                     ).reshape(-1, *self.stoch_shape)
        z = sample_one_hot_st(post_logits).reshape(h.shape[0], -1)
        return State(h, z, post_logits), prior_logits


class WorldModel(nn.Module):
    """Encoder + RSSM + decoder/reward/continue heads."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = mlp(cfg.obs_dim, cfg.hidden, cfg.hidden)
        self.rssm = RSSM(cfg)
        feat_dim = cfg.deter_dim + cfg.num_categoricals * cfg.classes_per_cat
        self.decoder = mlp(feat_dim, cfg.hidden, cfg.obs_dim)
        self.reward_head = mlp(feat_dim, cfg.hidden, cfg.num_bins)
        self.continue_head = mlp(feat_dim, cfg.hidden, 1)
        self.register_buffer(
            "bins",
            make_two_hot_bins(cfg.num_bins, cfg.symlog_low, cfg.symlog_high),
        )

    # ------------------------------------------------------------------ #
    # Single-step API (used for env interaction in train.py)             #
    # ------------------------------------------------------------------ #

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(symlog(obs))

    def obs_step(self, prev: State, action_onehot: torch.Tensor,
                 obs: torch.Tensor) -> State:
        enc = self.encode(obs)
        post, _ = self.rssm.obs_step(prev, action_onehot, enc)
        return post

    # ------------------------------------------------------------------ #
    # Sequence training                                                  #
    # ------------------------------------------------------------------ #

    def observe(self, obs_seq: torch.Tensor, action_seq: torch.Tensor,
                reward_seq: torch.Tensor, cont_seq: torch.Tensor
                ) -> dict[str, torch.Tensor]:
        """Run the model over a batch of sequences and compute training losses.

        Shapes (B = batch, T = seq_len):
            obs_seq:    (B, T, obs_dim)
            action_seq: (B, T, num_actions)   one-hot
            reward_seq: (B, T)                raw rewards
            cont_seq:   (B, T)                1.0 if episode continues
        """
        B, T = obs_seq.shape[:2]
        device = obs_seq.device
        enc = self.encode(obs_seq.reshape(B * T, -1)).reshape(B, T, -1)

        prev = self.rssm.initial_state(B, device)
        feats = []
        post_logits_seq = []
        prior_logits_seq = []
        for t in range(T):
            post, prior_logits = self.rssm.obs_step(
                prev, action_seq[:, t], enc[:, t]
            )
            feats.append(post.features())
            post_logits_seq.append(post.z_logits)
            prior_logits_seq.append(prior_logits)
            prev = post

        feats = torch.stack(feats, dim=1)                      # (B,T,F)
        post_logits = torch.stack(post_logits_seq, dim=1)      # (B,T,N,K)
        prior_logits = torch.stack(prior_logits_seq, dim=1)

        # --- reconstruction loss (symlog MSE) ---
        recon = self.decoder(feats)
        recon_loss = F.mse_loss(recon, symlog(obs_seq), reduction="none"
                                ).sum(-1).mean()

        # --- reward loss (two-hot in symlog space) ---
        r_logits = self.reward_head(feats)
        reward_loss = two_hot_loss(
            r_logits, symlog(reward_seq), self.bins
        ).mean()

        # --- continue loss (BCE with logits) ---
        c_logits = self.continue_head(feats).squeeze(-1)
        cont_loss = F.binary_cross_entropy_with_logits(
            c_logits, cont_seq, reduction="mean"
        )

        # --- KL with balancing + free bits ---
        kl = kl_balance_loss(
            post_logits, prior_logits,
            alpha=self.cfg.kl_alpha, free_bits=self.cfg.kl_free_bits,
        ).mean()

        loss = (self.cfg.recon_weight * recon_loss
                + self.cfg.reward_weight * reward_loss
                + self.cfg.cont_weight * cont_loss
                + self.cfg.kl_weight * kl)

        # Posterior trajectory, detached, used as imagination starting points.
        post_states = State(
            h=feats[..., :self.cfg.deter_dim],
            z=feats[..., self.cfg.deter_dim:],
            z_logits=post_logits,
        )

        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "reward_loss": reward_loss.detach(),
            "cont_loss": cont_loss.detach(),
            "kl": kl.detach(),
            "post_states": post_states,
        }

    # ------------------------------------------------------------------ #
    # Imagination                                                        #
    # ------------------------------------------------------------------ #

    def imagine(self, init: State, actor: Callable[[torch.Tensor], "Sampled"],
                horizon: int) -> dict[str, torch.Tensor]:
        """Roll the prior forward for ``horizon`` steps using ``actor``.

        We use REINFORCE for the actor (no dynamics-backprop), so the
        rollout itself runs under ``no_grad``; gradients only flow
        through the actor's ``log_prob`` and ``entropy`` outputs.
        """
        from .utils import symexp, two_hot_decode

        feats = []
        log_probs = []
        entropies = []

        state = init.detach()
        for _ in range(horizon):
            f = state.features()                              # no grad
            sampled = actor(f)                                # actor grads only
            feats.append(f)
            log_probs.append(sampled.log_prob)
            entropies.append(sampled.entropy)
            with torch.no_grad():
                state = self.rssm.img_step(state, sampled.actions.detach())

        with torch.no_grad():
            feats.append(state.features())

        feats = torch.stack(feats, dim=0)                     # (H+1, B, F)
        log_probs = torch.stack(log_probs, dim=0)             # (H,   B)
        entropies = torch.stack(entropies, dim=0)             # (H,   B)

        with torch.no_grad():
            r_logits = self.reward_head(feats)
            rewards = symexp(two_hot_decode(r_logits, self.bins))
            cont = torch.sigmoid(self.continue_head(feats).squeeze(-1))

        return {
            "feats": feats,           # (H+1, B, F)  detached
            "log_probs": log_probs,   # (H,   B)     grad to actor
            "entropies": entropies,   # (H,   B)     grad to actor
            "rewards": rewards,       # (H+1, B)     detached
            "cont": cont,             # (H+1, B)     detached
        }
