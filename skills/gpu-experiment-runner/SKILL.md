---
name: gpu-experiment-runner
description: >-
  Launch, monitor, and compare GPU training experiments on the local machine.
  Handles nvidia-smi checks, CUDA_VISIBLE_DEVICES, PYTHONPATH, __pycache__
  cleanup, Hydra override syntax, log tailing, TensorBoard metric extraction,
  and crash recovery. Use when the user asks to run training, launch an
  experiment, monitor a GPU job, compare training metrics, or do a bisect
  across commits.
---

# GPU Experiment Runner

## Pre-flight Checklist

Before launching any training run:

1. **GPU availability** — run `nvidia-smi` and pick an idle GPU.
2. **Environment** — use the `.mlenv` venv at `$REPO/.mlenv/bin/python`.
3. **Clean pycache** — always `find ml/ -name "__pycache__" -type d -exec rm -rf {} +` after a `git checkout`.
4. **PYTHONPATH** — must include the repo root.
5. **Output directory** — create it before launch: `mkdir -p $OUTPUT_DIR`.

## Launch Template

```bash
cd $REPO_ROOT
find ml/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MAPS_DB_ROOT=/data/sets/maps
export DATA_SET_ROOT=/data/sets
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

$REPO_ROOT/.mlenv/bin/python -B -m $MODULE \
    --config-dir=$CONFIG_DIR \
    --config-name=$CONFIG_NAME \
    $OVERRIDES \
    2>&1 | tee $LOG_FILE
```

### Critical Hydra Override Rules

| Situation | Syntax | Why |
|-----------|--------|-----|
| Key exists in config | `key=value` | Normal override |
| Key does NOT exist in config | `+key=value` | The `+` prefix **creates** the key |
| Local (no Ray) | `use_ray=False ray_config.scaling_config.num_workers=1` | **Must** set both |
| S3 paths | Quote the whole arg if it has special chars | Avoid shell expansion |

### GPU Memory Sizing Guide

For a single consumer GPU (RTX 2080 SUPER 8 GB):

| Parameter | Safe starting value |
|-----------|-------------------|
| `num_worlds` | 10 |
| `num_agents` | 50 |
| `batch_size` | 4000 |
| `num_timesteps` | 128 |

Scale linearly for larger GPUs. If OOM, halve `num_worlds` first.

## Monitoring a Running Experiment

### Progress Polling

Use Shell with `block_until_ms: 0` to launch, then poll:

```bash
# Quick check: last progress line
tail -1 $LOG_FILE | grep -oP '\d+/\d+' | tail -1

# Cycle rate
grep -oP '\d+\.\d+s/it' $LOG_FILE | tail -1
```

### Crash Detection

After launch, check within 90 seconds for common failures:

| Error pattern | Root cause | Fix |
|---------------|-----------|-----|
| `UnsupportedInterpolationType` | Missing OmegaConf resolver | Register resolvers before Hydra init |
| `InstantiationException` | Wrong Hydra override syntax | Use `+` prefix for missing keys |
| `missing.*argument` | Config-code mismatch | Check YAML against `__init__` signature |
| `Failed to initialize valid world` | Too many agents for map | Lower `num_agents` or `num_worlds` |
| `CUDA out of memory` | GPU OOM | Halve `num_worlds`, then `batch_size` |
| `size mismatch` in `load_state_dict` | Checkpoint architecture mismatch | Skip this commit / use compatible ckpt |

If training crashes, do NOT silently skip — log the failure in the experiment tracker.

## Metric Extraction (TensorBoard)

Use this script pattern to compare runs:

```python
import sys
sys.path.insert(0, '$MLENV/lib/python3.9/site-packages')
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator('$TB_DIR')
ea.Reload()
events = ea.Scalars('$TAG')
last_n = events[-50:]
avg = sum(e.value for e in last_n) / len(last_n)
final = events[-1].value
```

### Key Metric Tags

| Tag | What it measures |
|-----|-----------------|
| `filtered_rollout_stats/rewards` | Overall reward signal |
| `train_metrics/percent_success_out_of_finished_agents` | Success rate |
| `train_metrics/total_construction_zone_violations` | CZ violations per cycle |
| `train_metrics/total_offroad_violations` | Offroad violations per cycle |
| `model_info/collisions_per_agent` | Collision rate |
| `model_info/distance_per_agent` | How far agents travel |
| `loss/overall_loss` | Training loss (should decrease) |
| `loss/kl_divergence` | Policy change rate |

### Interpreting Results

- **Loss diverging** (increasing or going negative): training is unstable, revert last change.
- **Low distance_per_agent**: agents are stuck or stopping early.
- **High violations but low loss**: environment is too hard; relax parameters.
- **Success rate dropping while violations drop**: agent is learning to stop early (not ideal).

## Bisect Workflow

When bisecting commits to find a regression:

1. List commits: `git log --oneline BASE..TIP`
2. Probe checkpoint compatibility for each commit (load `actor.pt` with `strict=True`)
3. Skip incompatible commits, document why
4. Run each compatible commit with identical parameters
5. Extract metrics, build comparison table
6. Identify the regression point by metric deltas

### Experiment Naming

Use consistent naming: `commit_NN_SHORTHASH` for bisects, `fix_vN` for iterations.

## Additional Resources

- For PlanZero-specific config hierarchy, see the planzero-dev guide
- For the autonomous research loop, see the never-stop-researcher skill
