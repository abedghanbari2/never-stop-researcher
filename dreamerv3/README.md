# DreamerV3 (minimal pedagogical sketch)

A ~1k-LOC PyTorch recreation of the DreamerV3 algorithm
([Hafner et al., 2023](https://arxiv.org/abs/2301.04104)),
trimmed down for clarity and trainable on `CartPole-v1` in a few minutes
on CPU.

The point of this sketch is to make the **DreamerV3-distinctive tricks**
legible in code:

| File | Contents |
|---|---|
| `utils.py` | `symlog/symexp`, two-hot encode/decode, KL-balanced loss with free bits, straight-through one-hot sampler, percentile return normalizer |
| `world_model.py` | Encoder, RSSM (GRU + discrete categorical latents), decoder, two-hot reward head, Bernoulli continue head, sequence-`observe` + `imagine` |
| `actor_critic.py` | Categorical actor, two-hot critic with EMA target, λ-returns, REINFORCE-style update on percentile-normalized advantages |
| `buffer.py` | Episodic replay buffer; uniformly samples sub-sequences within a single episode |
| `train.py` | CartPole training loop: collect → train world model → train actor/critic on imagined rollouts |
| `config.py` | All hyperparameters in one dataclass |

## Quick start

```bash
pip install -r dreamerv3/requirements.txt
python -m dreamerv3.train
```

By default trains for 20 000 env steps. Expect the mean return (averaged over
the last 20 episodes) to rise from the random baseline (~22) past 100 within
a few minutes on CPU; it typically peaks in the 200–350 range mid-training.
The actor is plain REINFORCE on imagined rollouts, which is *not* the most
stable choice — late-training entropy collapses are common and returns can
crash. A faithful DreamerV3 uses a stronger actor (entropy schedule, unimix,
return-percentile clamp) to avoid this. This is a sketch, not a benchmark
reproduction.

## What this does NOT include

To stay small and readable the sketch leaves out:

- pixel observations (CNN encoder/decoder, transposed CNN reconstruction)
- continuous-action support (Gaussian/tanh-normal actor with reparam grads)
- multi-GPU training, mixed precision, jit/compile, learning-rate schedules
- the paper's larger `(32, 32)` latent grid (we use `(8, 8)`)
- careful initialization, layer-norm, output unimix mixing
- evaluation harness, deterministic eval, video logging

If you want a faithful research reimplementation, see
[`danijar/dreamerv3`](https://github.com/danijar/dreamerv3) or
[`NM512/dreamerv3-torch`](https://github.com/NM512/dreamerv3-torch).
