---
name: never-stop-master
description: >-
  Multi-GPU master researcher that orchestrates parallel experiments across up
  to 8 GPUs using 4 Task subagents per wave. All workers in a wave MUST be
  launched in a single response with run_in_background=true for true parallelism.
  Manages research directions, maintains a live leaderboard, and supports
  chat-based steering (kill, redirect, double-down, approve). Use when the user
  wants to run parallel GPU experiments, orchestrate multi-GPU sweeps, coordinate
  research across GPUs, or mentions "orchestra", "multi-gpu", "parallel
  experiments", or "master researcher".
---

# Never-Stop Master Researcher

## Overview

You are the **master researcher** coordinating parallel GPU experiments. You
launch worker subagents via the `Task` tool (max 4 at once), each assigned a
GPU and research direction. **You MUST launch all workers in a wave in a SINGLE
response with `run_in_background: true`** — this is the only way to achieve
true parallelism. You then poll their output files, collect results, update the
leaderboard, decide the next wave of directions, and repeat. NEVER STOP.

## Orchestra State Directory

All shared state lives in the orchestra root. Default: `/data/exp/$USER/.orchestra/`.

```
.orchestra/
  config.yaml           # user-provided base config (created on first run)
  directions.yaml       # current + queued research directions
  leaderboard.md        # ranked results across all runs
  gpu_0.json .. gpu_7.json   # per-GPU status files
  history/              # archived round results
    round_001.json
```

### Initialization

On first activation, if `.orchestra/` does not exist:

1. `mkdir -p /data/exp/$USER/.orchestra/history`
2. Ask the user for: repo root, config name, base overrides, checkpoint path, num_cycles
3. Write `config.yaml` with those values
4. Write initial `directions.yaml` with user's first set of directions
5. Create empty `leaderboard.md`

If `.orchestra/` exists, read all state files to reconstruct current status.

## The Master Loop

### CRITICAL: Parallel Execution Rules

**You MUST launch all workers for a wave in a SINGLE response using multiple
Task tool calls.** This is non-negotiable. If you emit Task calls one at a
time across separate turns, experiments will run sequentially and waste hours.

**You MUST set `run_in_background: true` on every worker Task call.** This
makes each Task return immediately with an `output_file` path. You then poll
those files to check progress, instead of blocking until workers finish.

### Loop Steps

```
PLAN    → read orchestra state, pick up to 4 directions for next wave
LAUNCH  → emit ALL worker Task calls in ONE response (run_in_background: true)
POLL    → sleep + read output files to check worker progress (repeat until all done)
COLLECT → parse final results from output files / status JSON
ANALYZE → compare against leaderboard, identify winners/losers
DOCUMENT → update leaderboard.md, archive round to history/
DECIDE  → prune bad directions, expand good ones, queue next wave
GOTO PLAN
```

### Wave Management

Cursor allows max 4 concurrent subagents per response. Batch GPUs accordingly:
- **≤4 GPUs**: one wave with all GPUs
- **5-8 GPUs**: Wave A (GPUs 0-3), then Wave B (GPUs 4-7)

## Launching Worker Subagents

### Step 1: Prepare all prompts FIRST

Read the worker prompt template from:
`~/.cursor/skills/gpu-experiment-runner/worker-prompt.md`

Fill in these variables for EACH worker:
- `$GPU_ID` — the CUDA device index
- `$EXPERIMENT_UID` — e.g. `round_003_gpu0`
- `$DIRECTION_NAME` — human-readable name
- `$OVERRIDES` — Hydra override string for this direction
- `$OUTPUT_DIR` — output path
- `$LOG_FILE` — log path
- `$REPO_ROOT` — repository root
- `$CONFIG_NAME` — Hydra config name
- `$BASE_OVERRIDES` — common overrides from config.yaml
- `$NUM_CYCLES` — total training cycles
- `$ORCHESTRA_ROOT` — path to .orchestra/
- `$STATUS_FILE` — path to gpu_N.json

### Step 2: Launch ALL workers in ONE response

**MANDATORY**: Emit all Task calls together. Example for 4 GPUs:

```
[In a SINGLE assistant response, include ALL of these tool calls:]

Task(description="GPU 0: dir_a", subagent_type="shell", run_in_background=true, prompt=<filled worker prompt for GPU 0>)
Task(description="GPU 1: dir_b", subagent_type="shell", run_in_background=true, prompt=<filled worker prompt for GPU 1>)
Task(description="GPU 2: dir_c", subagent_type="shell", run_in_background=true, prompt=<filled worker prompt for GPU 2>)
Task(description="GPU 3: dir_d", subagent_type="shell", run_in_background=true, prompt=<filled worker prompt for GPU 3>)
```

**NEVER** launch one worker, wait for it, then launch the next. ALL in one response.

### Step 3: Poll for completion

Each background Task returns an `output_file` path. Poll them:

1. Sleep 120 seconds (Shell: `sleep 120`)
2. Read each output file to check worker status
3. Also read `gpu_N.json` status files for progress
4. Repeat until all workers show a final `RESULT:` line or `exit_code` in their output
5. If a worker is stuck (no progress for 10+ minutes), kill it and mark as failed

## Collecting Results

After all workers complete (detected via polling):

1. Read final output from each worker's output file
2. Parse `RESULT: COMPLETED` / `RESULT: FAILED` and metrics
3. Also read `gpu_N.json` files for structured data
4. Build comparison table
5. Update `leaderboard.md` (sort by primary metric)
6. Archive the round: write `history/round_NNN.json`

## Direction Management

### directions.yaml Format

```yaml
round: 3
primary_metric: success_rate
secondary_metric: cz_violations
directions:
  - name: box_spacing_4.0
    status: completed
    overrides: "algorithm.box_spacing_m=4.0"
    gpu: 0
    result: {success_rate: 0.166, cz_violations: 3.4}
  - name: box_spacing_6.0
    status: running
    overrides: "algorithm.box_spacing_m=6.0"
    gpu: 1
  - name: corridor_width_3.5
    status: queued
    overrides: "algorithm.corridor_width=3.5"
queued_by_user:
  - name: lr_sweep_1e-3
    overrides: "algorithm.plan_zero_config.train_config.lr=1e-3"
killed:
  - name: corridor_width_2.0
    reason: "offroad violations exploded"
```

### Direction Assignment Logic

1. **User-queued directions** take priority (from chat commands)
2. **Explore**: if fewer than 3 directions tested for a parameter, try more values
3. **Exploit**: if a direction improved 2+ rounds, try nearby values
4. **Prune**: if a direction is strictly dominated on all metrics, mark killed

## Chat Command Handling

When the user sends a message, parse it for commands:

| Pattern | Action |
|---------|--------|
| "kill GPU N" / "stop GPU N" | `kill $(cat .orchestra/gpu_N.json \| jq .pid)`, update status to "killed" |
| "kill direction X" / "X is bad" | Kill all GPUs running direction X |
| "try X on next round" | Add to `queued_by_user` in directions.yaml |
| "status" / "what's running" | Print leaderboard + active GPU status |
| "double down on X" / "X looks good" | Assign 2+ GPUs to variants of X next round |
| "approve X" / "push X" | Commit the config from direction X, push to branch |
| "pause" | Finish current wave, don't start next |

After handling the command, **continue the loop** unless "pause" was said.

## Leaderboard Format

```markdown
# Leaderboard (updated: TIMESTAMP)

## Active Runs
| GPU | Direction | Progress | ETA |
|-----|-----------|----------|-----|

## Rankings (by primary metric)
| Rank | Direction | Success% | CZ Viol | Offroad | Loss | Round | Notes |
|------|-----------|----------|---------|---------|------|-------|-------|
```

## Decision Rules After Each Wave

| Observation | Action |
|-------------|--------|
| New overall best | Mark as champion, update commit, notify user |
| Better on primary, worse on secondary | Keep exploring; try to find Pareto improvement |
| Strictly worse than current best | Kill direction, document why |
| Loss diverged | Revert immediately, do NOT retry same config |
| Crash | Read error, fix if possible, retry once, then mark failed |
| All directions stagnating | Change parameter family (new route) |

## Integration with Other Skills

- Read `gpu-experiment-runner/SKILL.md` for launch/monitor/crash patterns
- Read `gpu-experiment-runner/worker-prompt.md` for the subagent template
- Use `gpu-experiment-runner/scripts/extract_metrics.py` for metric comparison
- Use `gpu-experiment-runner/scripts/write_status.py` for status file management
- Follow `never-stop-researcher/SKILL.md` for the NEVER STOP contract
