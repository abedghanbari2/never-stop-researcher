"""End-to-end training script for the DreamerV3 sketch on CartPole-v1.

Usage:
    python -m dreamerv3.train
    python -m dreamerv3.train --total-steps 50000 --log-every 1000
"""

from __future__ import annotations

import argparse
import collections
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from .actor_critic import ActorCriticTrainer
from .buffer import ReplayBuffer
from .config import Config
from .utils import sanity_check
from .world_model import WorldModel


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=cfg.total_steps)
    p.add_argument("--warmup-steps", type=int, default=cfg.warmup_steps)
    p.add_argument("--log-every", type=int, default=cfg.log_every)
    p.add_argument("--train-every", type=int, default=cfg.train_every)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--seq-len", type=int, default=cfg.seq_len)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--env-id", type=str, default=cfg.env_id)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    cfg.total_steps = args.total_steps
    cfg.warmup_steps = args.warmup_steps
    cfg.log_every = args.log_every
    cfg.train_every = args.train_every
    cfg.batch_size = args.batch_size
    cfg.seq_len = args.seq_len
    cfg.seed = args.seed
    cfg.env_id = args.env_id
    cfg._device = args.device  # attach for caller convenience
    return cfg


def main() -> None:
    cfg = parse_args()
    sanity_check()

    device = torch.device(getattr(cfg, "_device", "cpu"))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    env = gym.make(cfg.env_id)
    cfg.obs_dim = int(np.prod(env.observation_space.shape))
    cfg.num_actions = int(env.action_space.n)

    print(f"[init] env={cfg.env_id} obs_dim={cfg.obs_dim} "
          f"num_actions={cfg.num_actions} device={device}")

    world_model = WorldModel(cfg).to(device)
    wm_opt = torch.optim.Adam(world_model.parameters(), lr=cfg.wm_lr)
    ac = ActorCriticTrainer(cfg, device)
    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim, cfg.num_actions)

    obs, _ = env.reset(seed=cfg.seed)
    obs = np.asarray(obs, dtype=np.float32)
    ep_return = 0.0
    ep_len = 0
    recent_returns: collections.deque[float] = collections.deque(maxlen=20)

    rssm_state = world_model.rssm.initial_state(1, device)
    train_step = 0
    last_log = time.time()

    for step in range(1, cfg.total_steps + 1):
        # --- act ---
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
            if buffer.size < cfg.warmup_steps:
                action_idx = int(env.action_space.sample())
            else:
                action_idx = int(ac.actor.act(rssm_state.features()).item())
            action_onehot_np = np.zeros(cfg.num_actions, dtype=np.float32)
            action_onehot_np[action_idx] = 1.0
            action_onehot = torch.from_numpy(action_onehot_np
                                              ).to(device).unsqueeze(0)
            rssm_state = world_model.obs_step(rssm_state, action_onehot, obs_t)

        next_obs, reward, terminated, truncated, _ = env.step(action_idx)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        done = bool(terminated or truncated)
        buffer.add_step(obs, action_onehot_np, float(reward), done)
        ep_return += float(reward)
        ep_len += 1
        obs = next_obs

        if done:
            recent_returns.append(ep_return)
            ep_return = 0.0
            ep_len = 0
            obs, _ = env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            rssm_state = world_model.rssm.initial_state(1, device)

        # --- learn ---
        if (buffer.size >= cfg.warmup_steps
                and buffer.can_sample(cfg.seq_len)
                and step % cfg.train_every == 0):
            train_step += 1
            batch = buffer.sample(cfg.batch_size, cfg.seq_len, rng)
            obs_seq = torch.from_numpy(batch["obs"]).to(device)
            act_seq = torch.from_numpy(batch["action"]).to(device)
            rew_seq = torch.from_numpy(batch["reward"]).to(device)
            cont_seq = torch.from_numpy(batch["cont"]).to(device)

            wm_out = world_model.observe(obs_seq, act_seq, rew_seq, cont_seq)
            wm_opt.zero_grad(set_to_none=True)
            wm_out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                world_model.parameters(), cfg.grad_clip
            )
            wm_opt.step()

            # actor + critic on imagined rollouts from detached posteriors
            init = wm_out["post_states"].detach()
            # flatten (B, T, ...) -> (B*T, ...)
            B, T = init.h.shape[:2]
            from .world_model import State
            flat_init = State(
                h=init.h.reshape(B * T, -1),
                z=init.z.reshape(B * T, -1),
                z_logits=init.z_logits.reshape(
                    B * T, cfg.num_categoricals, cfg.classes_per_cat
                ),
            )
            ac_out = ac.update(world_model, flat_init)

            if train_step % max(1, cfg.log_every // cfg.train_every) == 0:
                now = time.time()
                fps = cfg.log_every / (now - last_log + 1e-9)
                last_log = now
                mean_ret = (sum(recent_returns) / len(recent_returns)
                            if recent_returns else float("nan"))
                print(
                    f"[step {step:>6d}] "
                    f"ret(avg20)={mean_ret:6.1f}  "
                    f"wm={wm_out['loss'].item():6.3f} "
                    f"recon={wm_out['recon_loss'].item():5.2f} "
                    f"rew={wm_out['reward_loss'].item():5.2f} "
                    f"cont={wm_out['cont_loss'].item():5.2f} "
                    f"kl={wm_out['kl'].item():5.2f}  "
                    f"a={ac_out['actor_loss'].item():+.3f} "
                    f"c={ac_out['critic_loss'].item():5.2f} "
                    f"H={ac_out['entropy'].item():.3f}  "
                    f"fps={fps:5.0f}"
                )

    env.close()
    final_mean = (sum(recent_returns) / len(recent_returns)
                  if recent_returns else float("nan"))
    print(f"\n[done] final mean return (last 20 eps) = {final_mean:.1f}")


if __name__ == "__main__":
    main()
