#!/usr/bin/env bash
set -euo pipefail

# never-stop-researcher installer
# Usage: git clone <repo> && cd never-stop-researcher && ./install.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CURSOR_DIR="$HOME/.cursor"
HOOKS_DIR="$CURSOR_DIR/hooks"
SKILL_DIR="$CURSOR_DIR/skills/never-stop-researcher"
STATE_FILE="$CURSOR_DIR/research-active.json"

echo "=== Never-Stop Researcher Installer ==="
echo ""

# 1. Install hook scripts
echo "[1/4] Installing hook scripts to $HOOKS_DIR/"
mkdir -p "$HOOKS_DIR"
cp "$SCRIPT_DIR/hooks/never-stop-continue.py" "$HOOKS_DIR/"
cp "$SCRIPT_DIR/hooks/research-session-init.py" "$HOOKS_DIR/"
cp "$SCRIPT_DIR/hooks/checkpoint-reminder.py" "$HOOKS_DIR/"
chmod +x "$HOOKS_DIR/never-stop-continue.py"
chmod +x "$HOOKS_DIR/research-session-init.py"
chmod +x "$HOOKS_DIR/checkpoint-reminder.py"
echo "  -> Copied 3 hook scripts"

# 2. Install/merge hooks.json
echo "[2/4] Configuring $CURSOR_DIR/hooks.json"
if [ -f "$CURSOR_DIR/hooks.json" ]; then
    echo "  -> Existing hooks.json found — merging..."
    python3 "$SCRIPT_DIR/merge_hooks.py" "$CURSOR_DIR/hooks.json" "$SCRIPT_DIR/hooks/hooks.json"
    echo "  -> Merged (backup at hooks.json.bak)"
else
    cp "$SCRIPT_DIR/hooks/hooks.json" "$CURSOR_DIR/hooks.json"
    echo "  -> Created new hooks.json"
fi

# 3. Install skill
echo "[3/4] Installing skill to $SKILL_DIR/"
mkdir -p "$SKILL_DIR"
cp "$SCRIPT_DIR/skill/SKILL.md" "$SKILL_DIR/SKILL.md"
echo "  -> Installed SKILL.md"

# 4. Initialize state file (inactive by default)
echo "[4/4] Initializing state file"
if [ ! -f "$STATE_FILE" ]; then
    echo '{"active": false}' > "$STATE_FILE"
    echo "  -> Created $STATE_FILE (inactive)"
else
    echo "  -> $STATE_FILE already exists, skipping"
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "To use:"
echo "  1. Open Cursor"
echo "  2. Start a conversation and say: 'Run autonomous experiments, never stop'"
echo "  3. The agent reads the skill, sets up checkpoints, and runs indefinitely"
echo "  4. To stop: say 'stop' in chat, or run:"
echo "     echo '{\"active\": false}' > ~/.cursor/research-active.json"
echo ""
echo "Hooks installed:"
echo "  stop        -> auto-continues research when agent turn ends (loop_limit: null)"
echo "  sessionStart -> injects checkpoint context into new conversations"
echo "  preCompact  -> notifies when context window is being compacted"
