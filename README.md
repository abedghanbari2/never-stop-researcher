# Never-Stop Researcher

Autonomous ML experiment runner for [Cursor](https://cursor.com) that runs **indefinitely** using lifecycle hooks.

## The Problem

Cursor's agent hits context window limits after ~1 hour of continuous operation. Skills that try to loop forever within a single conversation inevitably stop.

## The Solution

Uses Cursor's `stop` hook with `followup_message` to chain turns together:

```
Turn 1: Run experiments → Write checkpoint to disk → Turn ends
           stop hook fires → reads research-active.json → returns followup_message
Turn 2: Read checkpoint → Run next batch → Write checkpoint → Turn ends
           stop hook fires → ...forever
```

Each turn gets a fresh context window. All persistent state lives on disk in checkpoint files.

## Install

```bash
git clone https://github.com/<you>/never-stop-researcher.git
cd never-stop-researcher
./install.sh
```

This installs:

| What | Where |
|------|-------|
| Hook scripts | `~/.cursor/hooks/` |
| Hook config | `~/.cursor/hooks.json` (merged with existing) |
| Skill | `~/.cursor/skills/never-stop-researcher/SKILL.md` |
| State file | `~/.cursor/research-active.json` |

## Usage

1. Open Cursor in your ML project
2. Start a conversation:
   > Run autonomous experiments on my training pipeline. Never stop. Optimize for lowest validation loss.
3. The agent discovers your environment, asks for confirmation, then runs indefinitely
4. To stop: say **"stop"** in chat, or:
   ```bash
   echo '{"active": false}' > ~/.cursor/research-active.json
   ```

## How It Works

### Hooks (`~/.cursor/hooks.json`)

| Hook | Purpose |
|------|---------|
| `stop` | When the agent finishes a turn, checks if research is active. If yes, returns a `followup_message` that Cursor auto-submits, starting a new turn. `loop_limit: null` = no cap. |
| `sessionStart` | When a new conversation opens, injects checkpoint context so the agent knows about ongoing research. |
| `preCompact` | Notifies when context window compaction occurs. |

### State File (`~/.cursor/research-active.json`)

Tiny signal file the stop hook reads:

```json
{
  "active": true,
  "checkpoint": "/path/to/artifacts/checkpoint.md",
  "skill": "never-stop-researcher",
  "research_goal": "minimize validation loss"
}
```

Set `"active": false` to stop the loop.

### Checkpoint (`checkpoint.md`)

The agent writes a full state checkpoint to disk after every batch of experiments. Contains: batch number, leaderboard, last results, next plan, environment config. The stop hook's followup message tells the next turn where to find it.

### Skill (`SKILL.md`)

The agent skill that defines the research protocol: discovery, planning, launching parallel experimenters via Task subagents, collecting results, analyzing, checkpointing, and continuing.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Cursor Agent Turn                              │
│                                                 │
│  1. Read checkpoint.md from disk                │
│  2. Plan next batch of experiments              │
│  3. Launch N experimenter subagents (1 per GPU) │
│  4. Launch 1 auditor subagent (CPU)             │
│  5. Collect results                             │
│  6. Update leaderboard + journal                │
│  7. Write updated checkpoint.md                 │
│  8. Turn ends naturally                         │
└──────────────────┬──────────────────────────────┘
                   │
    stop hook fires│
    reads research-active.json
    returns {"followup_message": "Continue..."}
    Cursor auto-submits as next user message
                   │
┌──────────────────▼──────────────────────────────┐
│  Next Cursor Agent Turn                         │
│  (fresh context window)                         │
│                                                 │
│  1. Read checkpoint.md → full state restored    │
│  2. Plan next batch...                          │
│  ...                                            │
└─────────────────────────────────────────────────┘
```

## Uninstall

```bash
cd never-stop-researcher
./uninstall.sh
```

## Requirements

- [Cursor](https://cursor.com) with hooks support (v1.7+)
- Python 3.6+
- GPUs (for ML experiments — the skill discovers available hardware)

## License

Public domain. Do whatever you want with it.
