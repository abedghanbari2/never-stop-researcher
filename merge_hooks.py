#!/usr/bin/env python3
"""
Merge new hooks into an existing hooks.json without clobbering user's hooks.
Creates a .bak backup before modifying.

Usage: python3 merge_hooks.py <existing_hooks.json> <new_hooks.json>
"""
import json
import shutil
import sys


def merge(existing: dict, new: dict) -> dict:
    existing.setdefault("version", 1)
    existing.setdefault("hooks", {})

    for event, new_entries in new.get("hooks", {}).items():
        existing_entries = existing["hooks"].setdefault(event, [])
        existing_cmds = {e.get("command", "") for e in existing_entries}

        for entry in new_entries:
            if entry.get("command", "") not in existing_cmds:
                existing_entries.append(entry)

    return existing


def main():
    existing_path = sys.argv[1]
    new_path = sys.argv[2]

    with open(existing_path) as f:
        existing = json.load(f)
    with open(new_path) as f:
        new = json.load(f)

    shutil.copy2(existing_path, existing_path + ".bak")

    merged = merge(existing, new)

    with open(existing_path, "w") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
