---
name: never-stop-researcher
description: >-
  Hook-powered autonomous research loop. Uses Cursor's stop hook with
  followup_message to achieve truly infinite execution across context windows.
  Each turn: read checkpoint → run batch → update checkpoint → turn ends →
  stop hook auto-continues. Use when the user says "never stop", "don't stop",
  "keep going", requests autonomous experimentation, bisecting, hyperparameter
  sweeps, or iterative debugging across commits.
---

# Never-Stop Researcher (Hook-Powered)

## How It Works

This skill uses **Cursor hooks** to achieve truly infinite execution. Instead of
trying to loop forever within one conversation (which hits context limits after
~1 hour), it runs in **discrete turns** connected by the `stop` hook:

```
Turn 1: Read checkpoint → Run batch → Write checkpoint → Turn ends
                                                              │
         ┌────────────────────────────────────────────────────┘
         │  stop hook fires → reads ~/.cursor/research-active.json
         │  returns followup_message → Cursor auto-submits it
         ▼
Turn 2: Read checkpoint → Run batch → Write checkpoint → Turn ends
                                                              │
         ┌────────────────────────────────────────────────────┘
         ▼
Turn 3: ...and so on forever (loop_limit: null)
```

Each turn gets a fresh context window. All persistent state is on disk.

## The Contract

Once activated, you are an **autonomous research loop**. You do NOT pause to ask
"should I continue?" or "is this a good stopping point?". The stop hook handles
continuation automatically. The user may be asleep.

## Phase 0: First Activation (no checkpoint exists)

If there is no checkpoint file, this is a fresh start. Run discovery:

### 0.1 Hardware Discovery

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
nproc && free -h | head -2
```

Record: `NUM_GPUS`, `GPU_MODEL`, `GPU_MEMORY_GB`

### 0.2 Codebase Discovery

- Identify ML framework, training entrypoint, config system
- Find existing experiment artifacts or logs
- Check for virtual environment

Record: `REPO_ROOT`, `VENV_ACTIVATE`, `TRAIN_CMD`, `CONFIG_SYSTEM`, `ARTIFACT_DIR`

### 0.3 User Confirmation

Present a summary and ask for:
- **Metric(s) to optimize** (and direction: lower/higher is better)
- **Time budget per experiment** (default: 20 minutes)
- **Starting experiments** or free exploration
- **Any constraints**

Record: `PRIMARY_METRIC`, `METRIC_DIRECTION`, `TIME_BUDGET`, `RESEARCH_GOAL`

### 0.4 Activate the Research Session

After user confirms, write the state signal file and initial checkpoint:

```python
# Write ~/.cursor/research-active.json
import json
state = {
    "active": True,
    "checkpoint": "{ARTIFACT_DIR}/checkpoint.md",
    "skill": "never-stop-researcher",
    "research_goal": "{RESEARCH_GOAL}"
}
with open(os.path.expanduser("~/.cursor/research-active.json"), "w") as f:
    json.dump(state, f, indent=2)
```

This file tells the stop hook to auto-continue after each turn.

Create the initial checkpoint at `{ARTIFACT_DIR}/checkpoint.md`:

```markdown
# Research Checkpoint

## Session Info
- **Goal**: {RESEARCH_GOAL}
- **Primary metric**: {PRIMARY_METRIC} ({METRIC_DIRECTION} is better)
- **Time budget**: {TIME_BUDGET}s per experiment
- **Hardware**: {NUM_GPUS}× {GPU_MODEL}
- **Repo**: {REPO_ROOT} @ {COMMIT}
- **Branch**: {BRANCH}
- **Venv**: {VENV_ACTIVATE}
- **Train cmd**: {TRAIN_CMD}
- **Artifact dir**: {ARTIFACT_DIR}
- **Started**: {timestamp}

## State
- **Batch**: 0
- **Total experiments**: 0
- **Status**: STARTING

## Leaderboard
(empty)

## Next Batch Plan
1. {first experiment plan}
2. {second experiment plan}
...

## Analysis Notes
(none yet)
```

Also create `{ARTIFACT_DIR}/leaderboard.md` and `{ARTIFACT_DIR}/journal.md`.

**Do NOT proceed to the research loop until the user confirms.**
After confirmation, run the first batch in this same turn.

---

## Phase 1: Research Turn (runs every turn via stop hook)

Every turn follows this sequence. The stop hook ensures another turn starts
after this one ends.

### Step 1: Read Checkpoint

Read `{ARTIFACT_DIR}/checkpoint.md` to restore state. Extract:
- Current batch number
- Leaderboard standings
- Last batch results
- Next batch plan
- All environment config

If the checkpoint says `Status: PAUSED` or `Status: COMPLETED`, deactivate:

```python
# Deactivate the stop hook
import json, os
state_file = os.path.expanduser("~/.cursor/research-active.json")
with open(state_file) as f:
    state = json.load(f)
state["active"] = False
with open(state_file, "w") as f:
    json.dump(state, f, indent=2)
```

### Step 2: Plan Batch

Decide what experiments to run based on:
- The research goal
- Results from previous batches (from checkpoint + journal)
- Gaps in coverage
- Strategy rules (see Research Strategy below)

For each experiment, define: `EXP_ID`, `GPU_ID`, `TRAIN_CMD`, `TIME_BUDGET`,
optional `CODE_CHANGE` and `NEEDS_ISOLATION`.

Write the plan to the journal.

### Step 3: Launch Experiments

Use **Task subagents** — one per GPU, all launched in parallel in a single message.

```
Task(
  subagent_type: "generalPurpose",
  description: "Experimenter GPU {GPU_ID}",
  prompt: <experimenter prompt with all context>
)
```

For monitoring, also launch an auditor subagent alongside the experimenters.

### Step 4: Collect Results

After subagents return:
1. Read each experimenter's summary
2. Read the auditor's report
3. Update the leaderboard

### Step 5: Analyze and Journal

Append a batch entry to `{ARTIFACT_DIR}/journal.md`:

```markdown
## Batch {N} — {timestamp}

### Results
| Experiment | Status | {PRIMARY_METRIC} | Wall Time |
|-----------|--------|-------------------|-----------|
| ... | ... | ... | ... |

### Analysis
{What improved? What regressed? Why?}

### Next Batch Plan
{What to try next}
```

### Step 6: Write Checkpoint (CRITICAL)

**You MUST update the checkpoint before your turn ends.** The stop hook will
use this to continue the next turn.

Update `{ARTIFACT_DIR}/checkpoint.md` with:
- Incremented batch number
- Updated leaderboard (top 10)
- This batch's results summary
- Next batch plan
- Any updated analysis notes

```markdown
# Research Checkpoint

## Session Info
(same as before)

## State
- **Batch**: {N}
- **Total experiments**: {total}
- **Status**: RUNNING
- **Current best**: {EXP_ID} ({PRIMARY_METRIC}={value}, batch {N})

## Leaderboard (top 10)
| Rank | Experiment | {PRIMARY_METRIC} | Batch | Notes |
|------|-----------|-------------------|-------|-------|
| 1 | ... | ... | ... | ... |

## Last Batch Results
{Brief summary of batch N results}

## Next Batch Plan
1. {next_exp_1}: {rationale}
2. {next_exp_2}: {rationale}
...

## Analysis Notes
{Cumulative insights: what works, what doesn't, routes exhausted}
```

### Step 7: Continue or End Turn

After writing the checkpoint, you have two options:

- **Run another batch** if context usage is low and you have capacity.
  Go back to Step 2.
- **End the turn naturally** by completing your response. The stop hook
  will auto-continue with a fresh turn.

**Prefer running 2-3 batches per turn** if experiments are short, to minimize
overhead. But always checkpoint between batches.

---

## Stopping the Research

The user can stop the research by:
1. Saying "stop" or "pause" in chat
2. Manually editing `~/.cursor/research-active.json` to set `"active": false`

When you receive a stop command:
1. Write a final checkpoint with `Status: PAUSED`
2. Deactivate the state file
3. Present a final summary

---

## Research Strategy

### Route-Based Exploration

Treat each parameter family as a **route**:
- If a route improves for 2+ batches → keep pushing it
- If a route stagnates or regresses → switch to another
- Track which routes you've exhausted in the analysis notes

### When Stuck

1. Re-read the code diff between the best run and the baseline
2. Check for hidden defaults in the code
3. Combine the best aspects of two near-misses
4. Try a more radical change
5. Read comments/docstrings in the source for hints

### Decision Rules

| Situation | Action |
|-----------|--------|
| New best on primary metric | Update checkpoint, document in journal |
| Better on one metric, worse on another | Keep running; seek Pareto improvement |
| Strictly worse | Revert, document why, try different route |
| Training crashed | Log failure, fix root cause, relaunch |
| Loss diverging | Revert last change immediately |

---

## Experimenter Protocol

Each experimenter subagent runs **one** experiment on **one** GPU:

1. **Setup**: activate venv, set `CUDA_VISIBLE_DEVICES={GPU_ID}`, create artifact dir
2. **Launch**: run training with `timeout {TIME_BUDGET}s`, background with `block_until_ms: 0`
3. **Monitor**: poll terminal file every 30-60s until exit
4. **Extract**: search log for metric keywords, take last values
5. **Report**: write `{ARTIFACT_DIR}/{EXP_ID}_summary.md`
6. **Return**: status, wall time, key metrics, anomalies

### Failure Handling

| Failure | Action |
|---------|--------|
| CUDA OOM | Record, suggest smaller batch/model, return FAILED |
| Config error | Record error message, return FAILED |
| Timeout (exit 124) | Expected — extract partial metrics, return TIMEOUT |
| NaN loss | Note in summary, let it finish |
| Other crash | Record exit code + last 50 log lines, return FAILED |

---

## Auditor Protocol

One auditor subagent runs alongside experimenters on CPU:

1. **Monitor** every 60s: `nvidia-smi`, process liveness, tail logs
2. **Flag** anomalies: OOM, NaN, dead processes, idle GPUs
3. **Kill** experiments wasting GPU time (NaN confirmed 2+ cycles, loss diverging 10+ iters)
4. **Report**: write `{ARTIFACT_DIR}/audit_batch_{N}.md`

---

## Git Discipline

After each batch, use `git commit-tree` to avoid hook trailers:

```bash
TREE=$(git write-tree)
PARENT=$(git rev-parse HEAD)
NEW=$(echo "batch N: description" | git commit-tree "$TREE" -p "$PARENT")
git reset --hard "$NEW"
```

When a new best is found, push to the experiment branch.

---

## Communication Rules

- **Be concise** — report metrics, not essays
- **Always show**: current run, best run, next planned experiment
- **If the user asks a question**: answer briefly, then continue working
- **Never ask for permission to continue** — the hook handles continuation

## Integration

This skill works with:
- **Cursor hooks** (`~/.cursor/hooks.json`) for auto-continuation via `stop` hook
- **gpu-experiment-runner** for launch/monitor/metrics patterns
- **~/.cursor/research-active.json** as the signal file for the stop hook
