#!/usr/bin/env bash
set -euo pipefail

# never-stop-researcher uninstaller

CURSOR_DIR="$HOME/.cursor"

echo "=== Never-Stop Researcher Uninstaller ==="
echo ""

# Deactivate any running research
if [ -f "$CURSOR_DIR/research-active.json" ]; then
    echo '{"active": false}' > "$CURSOR_DIR/research-active.json"
    echo "[1] Deactivated research session"
fi

# Remove hook scripts
for script in never-stop-continue.py research-session-init.py checkpoint-reminder.py; do
    if [ -f "$CURSOR_DIR/hooks/$script" ]; then
        rm "$CURSOR_DIR/hooks/$script"
        echo "[2] Removed $script"
    fi
done

# Remove skill
if [ -d "$CURSOR_DIR/skills/never-stop-researcher" ]; then
    rm -rf "$CURSOR_DIR/skills/never-stop-researcher"
    echo "[3] Removed skill directory"
fi

echo ""
echo "NOTE: hooks.json was not modified. Remove the never-stop entries manually:"
echo "  $CURSOR_DIR/hooks.json"
echo ""
echo "=== Uninstall complete ==="
