"""Episodic replay buffer.

Stores complete episodes as numpy arrays in a ring of capacity ``capacity``
(in env steps, not episodes). Sampling draws ``batch_size`` sequences of
length ``seq_len`` uniformly from the stored steps — sequences are
guaranteed to lie within a single episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    obs: np.ndarray            # (T, obs_dim)   obs *before* action
    action: np.ndarray         # (T, num_actions) one-hot
    reward: np.ndarray         # (T,)           reward received after action
    cont: np.ndarray           # (T,)           1.0 unless episode ended


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, num_actions: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.episodes: list[Episode] = []
        self._current_obs: list[np.ndarray] = []
        self._current_action: list[np.ndarray] = []
        self._current_reward: list[float] = []
        self._current_cont: list[float] = []
        self._total_steps = 0

    @property
    def size(self) -> int:
        return self._total_steps

    def add_step(self, obs: np.ndarray, action_onehot: np.ndarray,
                 reward: float, done: bool) -> None:
        self._current_obs.append(obs.astype(np.float32))
        self._current_action.append(action_onehot.astype(np.float32))
        self._current_reward.append(float(reward))
        self._current_cont.append(0.0 if done else 1.0)
        self._total_steps += 1
        if done:
            self._flush_episode()
        self._evict_if_needed()

    def _flush_episode(self) -> None:
        if not self._current_obs:
            return
        ep = Episode(
            obs=np.asarray(self._current_obs, dtype=np.float32),
            action=np.asarray(self._current_action, dtype=np.float32),
            reward=np.asarray(self._current_reward, dtype=np.float32),
            cont=np.asarray(self._current_cont, dtype=np.float32),
        )
        self.episodes.append(ep)
        self._current_obs.clear()
        self._current_action.clear()
        self._current_reward.clear()
        self._current_cont.clear()

    def _evict_if_needed(self) -> None:
        while self._total_steps > self.capacity and self.episodes:
            dropped = self.episodes.pop(0)
            self._total_steps -= dropped.obs.shape[0]

    def can_sample(self, seq_len: int) -> bool:
        return any(ep.obs.shape[0] >= seq_len for ep in self.episodes)

    def sample(self, batch_size: int, seq_len: int,
               rng: np.random.Generator) -> dict:
        eligible = [ep for ep in self.episodes if ep.obs.shape[0] >= seq_len]
        if not eligible:
            raise RuntimeError("No episodes long enough to sample")
        obs_batch = np.empty((batch_size, seq_len, self.obs_dim),
                             dtype=np.float32)
        act_batch = np.empty((batch_size, seq_len, self.num_actions),
                             dtype=np.float32)
        rew_batch = np.empty((batch_size, seq_len), dtype=np.float32)
        cont_batch = np.empty((batch_size, seq_len), dtype=np.float32)
        for i in range(batch_size):
            ep = eligible[rng.integers(len(eligible))]
            start = rng.integers(0, ep.obs.shape[0] - seq_len + 1)
            sl = slice(start, start + seq_len)
            obs_batch[i] = ep.obs[sl]
            act_batch[i] = ep.action[sl]
            rew_batch[i] = ep.reward[sl]
            cont_batch[i] = ep.cont[sl]
        return {
            "obs": obs_batch,
            "action": act_batch,
            "reward": rew_batch,
            "cont": cont_batch,
        }
