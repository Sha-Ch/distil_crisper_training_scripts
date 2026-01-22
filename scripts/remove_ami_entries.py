#!/usr/bin/env python3
"""
Remove AMI dataset entries from all_pseudo_labels.jsonl

- Creates a backup before modifying
- Removes entries where "dataset": "ami"
- Keeps all other entries intact
- Reports how many entries were removed

Usage:
    python remove_ami_entries.py
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

def main():
    # Target file
    jsonl_path = Path("/workspace/pseudo_labels/all_pseudo_labels.jsonl")

    if not jsonl_path.exists():
        print(f"ERROR: File not found: {jsonl_path}")
        return 1

    # Create backup with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = jsonl_path.parent / f"all_pseudo_labels_backup_{timestamp}.jsonl"

    print(f"Creating backup: {backup_path}")
    shutil.copy2(jsonl_path, backup_path)
    print(f"Backup created successfully")

    # Read and filter entries
    kept_entries = []
    removed_count = 0
    total_count = 0

    print(f"Reading {jsonl_path}...")
    with open(jsonl_path, 'r') as f:
        for line in f:
            total_count += 1
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                if entry.get("dataset") == "ami":
                    removed_count += 1
                else:
                    kept_entries.append(line)
            except json.JSONDecodeError:
                # Keep malformed lines to avoid data loss
                kept_entries.append(line)

    print(f"Total entries: {total_count:,}")
    print(f"AMI entries to remove: {removed_count:,}")
    print(f"Entries to keep: {len(kept_entries):,}")

    if removed_count == 0:
        print("No AMI entries found. File unchanged.")
        return 0

    # Write filtered entries back
    print(f"Writing filtered data to {jsonl_path}...")
    with open(jsonl_path, 'w') as f:
        for line in kept_entries:
            f.write(line + '\n')

    print(f"Done. Removed {removed_count:,} AMI entries.")
    print(f"Backup saved at: {backup_path}")

    return 0

if __name__ == "__main__":
    exit(main())
