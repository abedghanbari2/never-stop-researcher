#!/usr/bin/env python3
"""
sessionStart hook that injects research checkpoint context.

When a new conversation starts and there's an active research session,
this hook reads the checkpoint file and injects a summary as additional_context.
This way the agent immediately knows about the ongoing research.
"""
import json
import os
import sys


STATE_FILE = os.path.expanduser("~/.cursor/research-active.json")


def main():
    sys.stdin.read()  # drain stdin

    if not os.path.exists(STATE_FILE):
        print(json.dumps({}))
        return

    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(json.dumps({}))
        return

    if not state.get("active"):
        print(json.dumps({}))
        return

    checkpoint = state.get("checkpoint", "")
    if not checkpoint or not os.path.exists(checkpoint):
        print(json.dumps({}))
        return

    try:
        with open(checkpoint) as f:
            content = f.read()[:4000]
    except OSError:
        print(json.dumps({}))
        return

    context = (
        "ACTIVE RESEARCH SESSION DETECTED.\n"
        f"Checkpoint file: {checkpoint}\n"
        f"Skill: {state.get('skill', 'never-stop-researcher')}\n\n"
        "To continue the research loop, read the full checkpoint file "
        "and execute the next planned batch. "
        "Do NOT ask for permission — the NEVER STOP contract is active.\n\n"
        f"Checkpoint preview:\n{content}"
    )

    print(json.dumps({"additional_context": context}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({}))
