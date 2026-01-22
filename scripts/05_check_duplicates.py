#!/usr/bin/env python3
"""
=============================================================================
Duplicate Checker for Pseudo-Label Files
=============================================================================
Scans pseudo-label JSONL files to find duplicate sample IDs.

Usage:
  python 05_check_duplicates.py --dir /workspace/pseudo_labels
  python 05_check_duplicates.py --dir /workspace/pseudo_labels --fix  # Remove duplicates
  python 05_check_duplicates.py --dir /workspace/pseudo_labels --dataset librispeech

=============================================================================
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from tqdm import tqdm

console = Console()


def find_duplicates(
    input_dir: Path,
    dataset_filter: str = None,
) -> Dict[str, Dict[str, List[Tuple[Path, int]]]]:
    """
    Find duplicate sample IDs across all pseudo-label files.

    Args:
        input_dir: Directory containing pseudo-label JSONL files
        dataset_filter: Optional dataset name to filter (e.g., 'librispeech')

    Returns:
        Dict mapping dataset -> sample_id -> list of (filepath, line_number) where it appears
    """
    # Track where each sample_id appears: dataset -> sample_id -> [(file, line_num), ...]
    sample_locations: Dict[str, Dict[str, List[Tuple[Path, int]]]] = defaultdict(lambda: defaultdict(list))

    # Find all JSONL files
    patterns = ['*_gpu*_accepted.jsonl', '*_gpu*_rejected.jsonl']
    all_files = []
    for pattern in patterns:
        all_files.extend(input_dir.glob(pattern))

    if dataset_filter:
        all_files = [f for f in all_files if f.name.startswith(dataset_filter + '_')]

    if not all_files:
        console.print(f"[yellow]No pseudo-label files found in {input_dir}[/yellow]")
        return {}

    console.print(f"[cyan]Scanning {len(all_files)} files...[/cyan]")

    for filepath in tqdm(all_files, desc="Scanning files"):
        # Extract dataset name from filename
        name = filepath.name.rsplit('_gpu', 1)[0]

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line)
                        sample_id = entry.get('sample_id', '')
                        if sample_id:
                            sample_locations[name][sample_id].append((filepath, line_num))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            console.print(f"[red]Error reading {filepath}: {e}[/red]")

    return dict(sample_locations)


def analyze_duplicates(sample_locations: Dict[str, Dict[str, List[Tuple[Path, int]]]]) -> Dict[str, dict]:
    """
    Analyze duplicate statistics.

    Returns:
        Dict mapping dataset -> stats dict
    """
    stats = {}

    for dataset, samples in sample_locations.items():
        total_samples = len(samples)
        total_entries = sum(len(locs) for locs in samples.values())
        duplicates = {sid: locs for sid, locs in samples.items() if len(locs) > 1}
        duplicate_count = len(duplicates)
        duplicate_entries = sum(len(locs) for locs in duplicates.values())

        # Categorize duplicates
        same_file_dups = 0  # Duplicates within same file
        cross_file_dups = 0  # Duplicates across different files
        cross_type_dups = 0  # Same sample in both accepted AND rejected

        for sample_id, locations in duplicates.items():
            files = set(loc[0] for loc in locations)
            if len(files) == 1:
                same_file_dups += 1
            else:
                cross_file_dups += 1
                # Check if it's in both accepted and rejected
                has_accepted = any('accepted' in str(loc[0]) for loc in locations)
                has_rejected = any('rejected' in str(loc[0]) for loc in locations)
                if has_accepted and has_rejected:
                    cross_type_dups += 1

        stats[dataset] = {
            'total_unique_samples': total_samples,
            'total_entries': total_entries,
            'duplicate_samples': duplicate_count,
            'duplicate_entries': duplicate_entries,
            'extra_entries': duplicate_entries - duplicate_count,  # How many to remove
            'same_file_dups': same_file_dups,
            'cross_file_dups': cross_file_dups,
            'cross_type_dups': cross_type_dups,  # In both accepted AND rejected
        }

    return stats


def print_duplicate_details(
    sample_locations: Dict[str, Dict[str, List[Tuple[Path, int]]]],
    max_show: int = 10,
):
    """Print detailed information about duplicates."""
    for dataset, samples in sample_locations.items():
        duplicates = {sid: locs for sid, locs in samples.items() if len(locs) > 1}

        if not duplicates:
            continue

        console.print(f"\n[bold yellow]Duplicate details for {dataset}:[/bold yellow]")

        shown = 0
        for sample_id, locations in sorted(duplicates.items(), key=lambda x: -len(x[1])):
            if shown >= max_show:
                remaining = len(duplicates) - shown
                console.print(f"  [dim]... and {remaining} more duplicates[/dim]")
                break

            console.print(f"\n  [cyan]{sample_id}[/cyan] appears {len(locations)} times:")
            for filepath, line_num in locations:
                file_type = "accepted" if "accepted" in filepath.name else "rejected"
                console.print(f"    - {filepath.name}:{line_num} [{file_type}]")

            shown += 1


def remove_duplicates(
    input_dir: Path,
    sample_locations: Dict[str, Dict[str, List[Tuple[Path, int]]]],
    dry_run: bool = True,
) -> Dict[str, int]:
    """
    Remove duplicate entries, keeping only the first occurrence.

    Strategy:
    - For same sample in both accepted AND rejected: keep accepted, remove rejected
    - For same sample in multiple accepted files: keep first occurrence
    - For same sample in multiple rejected files: keep first occurrence

    Returns:
        Dict mapping dataset -> number of entries removed
    """
    removed_counts = {}

    for dataset, samples in sample_locations.items():
        duplicates = {sid: locs for sid, locs in samples.items() if len(locs) > 1}

        if not duplicates:
            removed_counts[dataset] = 0
            continue

        # Track which (file, line) to remove
        lines_to_remove: Dict[Path, Set[int]] = defaultdict(set)

        for sample_id, locations in duplicates.items():
            # Sort locations: accepted files first, then by filename, then by line number
            sorted_locs = sorted(locations, key=lambda x: (
                0 if 'accepted' in str(x[0]) else 1,  # Accepted first
                str(x[0]),  # Then by filename
                x[1]  # Then by line number
            ))

            # Keep first, mark rest for removal
            for filepath, line_num in sorted_locs[1:]:
                lines_to_remove[filepath].add(line_num)

        total_to_remove = sum(len(lines) for lines in lines_to_remove.values())
        removed_counts[dataset] = total_to_remove

        if dry_run:
            console.print(f"[yellow]DRY RUN: Would remove {total_to_remove} duplicate entries from {dataset}[/yellow]")
            continue

        # Actually remove duplicates by rewriting files
        console.print(f"[cyan]Removing {total_to_remove} duplicates from {dataset}...[/cyan]")

        for filepath, lines_to_skip in lines_to_remove.items():
            if not lines_to_skip:
                continue

            # Read all lines except duplicates
            new_lines = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num not in lines_to_skip:
                        new_lines.append(line)

            # Write back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

            console.print(f"  [green]Removed {len(lines_to_skip)} duplicates from {filepath.name}[/green]")

    return removed_counts


def main():
    parser = argparse.ArgumentParser(
        description='Check for duplicate samples in pseudo-label files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check for duplicates
  python 05_check_duplicates.py --dir /workspace/pseudo_labels

  # Check specific dataset
  python 05_check_duplicates.py --dir /workspace/pseudo_labels --dataset librispeech

  # Show detailed duplicate info
  python 05_check_duplicates.py --dir /workspace/pseudo_labels --details

  # Remove duplicates (dry run first)
  python 05_check_duplicates.py --dir /workspace/pseudo_labels --fix --dry-run

  # Actually remove duplicates
  python 05_check_duplicates.py --dir /workspace/pseudo_labels --fix
        """
    )

    parser.add_argument(
        '--dir', '-d',
        type=str,
        required=True,
        help='Directory containing pseudo-label JSONL files'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Filter to specific dataset (e.g., librispeech)'
    )

    parser.add_argument(
        '--details',
        action='store_true',
        help='Show detailed duplicate information'
    )

    parser.add_argument(
        '--fix',
        action='store_true',
        help='Remove duplicate entries (keeps first occurrence)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='With --fix, show what would be removed without actually removing'
    )

    args = parser.parse_args()

    input_dir = Path(args.dir)
    if not input_dir.exists():
        console.print(f"[red]Directory not found: {input_dir}[/red]")
        sys.exit(1)

    console.print(Panel(
        f"[bold]Pseudo-Label Duplicate Checker[/bold]\n\n"
        f"Directory: {input_dir}\n"
        f"Dataset Filter: {args.dataset or 'All'}\n"
        f"Fix Mode: {args.fix}\n"
        f"Dry Run: {args.dry_run}",
        title="Configuration"
    ))

    # Find all samples and their locations
    sample_locations = find_duplicates(input_dir, args.dataset)

    if not sample_locations:
        console.print("[yellow]No data found to analyze[/yellow]")
        sys.exit(0)

    # Analyze duplicates
    stats = analyze_duplicates(sample_locations)

    # Print summary table
    table = Table(title="[bold]Duplicate Analysis Summary[/bold]")
    table.add_column("Dataset", style="cyan")
    table.add_column("Unique Samples", justify="right")
    table.add_column("Total Entries", justify="right")
    table.add_column("Duplicates", justify="right")
    table.add_column("Extra Entries", justify="right", style="red")
    table.add_column("Same File", justify="right")
    table.add_column("Cross File", justify="right")
    table.add_column("Accepted+Rejected", justify="right", style="yellow")

    total_unique = 0
    total_entries = 0
    total_dups = 0
    total_extra = 0

    for dataset, s in sorted(stats.items()):
        dup_style = "red" if s['duplicate_samples'] > 0 else "green"
        table.add_row(
            dataset,
            f"{s['total_unique_samples']:,}",
            f"{s['total_entries']:,}",
            f"[{dup_style}]{s['duplicate_samples']:,}[/{dup_style}]",
            f"{s['extra_entries']:,}" if s['extra_entries'] > 0 else "[green]0[/green]",
            f"{s['same_file_dups']:,}",
            f"{s['cross_file_dups']:,}",
            f"{s['cross_type_dups']:,}" if s['cross_type_dups'] > 0 else "0",
        )
        total_unique += s['total_unique_samples']
        total_entries += s['total_entries']
        total_dups += s['duplicate_samples']
        total_extra += s['extra_entries']

    table.add_row("", "", "", "", "", "", "", "")
    dup_style = "red" if total_dups > 0 else "green"
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_unique:,}[/bold]",
        f"[bold]{total_entries:,}[/bold]",
        f"[bold {dup_style}]{total_dups:,}[/bold {dup_style}]",
        f"[bold red]{total_extra:,}[/bold red]" if total_extra > 0 else "[bold green]0[/bold green]",
        "", "", ""
    )

    console.print(table)

    # Print verdict
    if total_dups == 0:
        console.print("\n[bold green]No duplicates found! All samples are unique.[/bold green]")
    else:
        console.print(f"\n[bold yellow]Found {total_dups:,} duplicate sample IDs ({total_extra:,} extra entries to remove)[/bold yellow]")

        # Show details if requested
        if args.details:
            print_duplicate_details(sample_locations)

        # Fix if requested
        if args.fix:
            console.print("")
            removed = remove_duplicates(input_dir, sample_locations, dry_run=args.dry_run)

            if not args.dry_run:
                total_removed = sum(removed.values())
                console.print(f"\n[bold green]Removed {total_removed:,} duplicate entries[/bold green]")
        else:
            console.print("\n[dim]Use --fix to remove duplicates (--dry-run to preview)[/dim]")


if __name__ == '__main__':
    main()
