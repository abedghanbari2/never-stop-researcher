#!/usr/bin/env python3
"""
Stop hook that auto-continues research sessions.

When the agent finishes a turn, this hook checks for an active research session
(signaled by ~/.cursor/research-active.json). If active, it returns a
followup_message that Cursor auto-submits as the next user message, creating
a self-sustaining loop.

The agent checkpoints its state to disk after every batch, so each new turn
can reconstruct full context from the checkpoint file alone.
"""
import json
import os
import sys


STATE_FILE = os.path.expanduser("~/.cursor/research-active.json")


def main():
    payload = json.load(sys.stdin)
    status = payload.get("status", "")

    if status != "completed":
        print(json.dumps({}))
        return

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
    skill = state.get("skill", "never-stop-researcher")
    goal = state.get("research_goal", "")

    msg_parts = [
        "Continue autonomous research.",
        f"Read your checkpoint at: {checkpoint}" if checkpoint else "",
        f"Research goal: {goal}" if goal else "",
        f"Read the skill at ~/.cursor/skills/{skill}/SKILL.md for the full protocol.",
        "Execute the next planned batch. Do NOT pause or ask for permission.",
    ]

    followup = " ".join(p for p in msg_parts if p)

    print(json.dumps({"followup_message": followup}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({}))
