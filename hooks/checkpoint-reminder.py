#!/usr/bin/env python3
"""
preCompact hook that reminds the user when context is being compacted.

This is observational — it logs the compaction event and shows a message.
The agent should already be checkpointing after each batch, so this is
just a safety net notification.
"""
import json
import sys


def main():
    payload = json.load(sys.stdin)
    usage = payload.get("context_usage_percent", 0)
    tokens = payload.get("context_tokens", 0)
    window = payload.get("context_window_size", 0)
    is_first = payload.get("is_first_compaction", False)

    msg = (
        f"Context compaction triggered ({usage}% used, "
        f"{tokens}/{window} tokens). "
    )
    if is_first:
        msg += "First compaction for this session. "
    msg += "Research state is preserved in checkpoint files on disk."

    print(json.dumps({"user_message": msg}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({}))
