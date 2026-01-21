#!/usr/bin/env python3
"""
=============================================================================
Real-Time Monitoring Dashboard for Pseudo-Label Generation
=============================================================================
Displays live GPU utilization, processing progress, and statistics.
Updates every second with a clean CLI interface.

Usage:
  python3 monitor_progress.py                    # Monitor default location
  python3 monitor_progress.py --dir /workspace/pseudo_labels
  python3 monitor_progress.py --refresh 0.5      # Faster refresh (0.5s)

Run this in a separate terminal while pseudo-label generation runs.
=============================================================================
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

try:
    from rich.console import Console
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
    from rich import box
except ImportError:
    print("Installing rich library...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
    from rich import box

console = Console()


def get_gpu_stats():
    """Get GPU statistics using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'utilization': float(parts[2]) if parts[2] != '[N/A]' else 0,
                        'memory_used': float(parts[3]) if parts[3] != '[N/A]' else 0,
                        'memory_total': float(parts[4]) if parts[4] != '[N/A]' else 0,
                        'temperature': float(parts[5]) if parts[5] != '[N/A]' else 0,
                        'power': float(parts[6]) if parts[6] != '[N/A]' else 0,
                    })
        return gpus
    except Exception as e:
        return []


def get_progress_data(pseudo_labels_dir: Path):
    """Load progress data from the progress file."""
    progress_file = pseudo_labels_dir / 'generation_progress.json'
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def count_samples_from_files(pseudo_labels_dir: Path):
    """Count samples from output files for each dataset and GPU."""
    stats = defaultdict(lambda: {
        'accepted': 0,
        'rejected': 0,
        'hours': 0.0,
        'wer_sum': 0.0,
        'gpu_counts': defaultdict(int)
    })

    # Scan all output files
    for filepath in pseudo_labels_dir.glob('*_gpu*_accepted.jsonl'):
        try:
            # Parse dataset name and GPU from filename
            name = filepath.stem.split('_gpu')[0]
            gpu_part = filepath.stem.split('_gpu')[1].split('_')[0]
            gpu_id = int(gpu_part)

            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        stats[name]['accepted'] += 1
                        stats[name]['hours'] += entry.get('duration_seconds', 0) / 3600
                        stats[name]['wer_sum'] += entry.get('wer', 0)
                        stats[name]['gpu_counts'][gpu_id] += 1
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    for filepath in pseudo_labels_dir.glob('*_gpu*_rejected.jsonl'):
        try:
            name = filepath.stem.split('_gpu')[0]
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        json.loads(line)
                        stats[name]['rejected'] += 1
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    return dict(stats)


def create_gpu_table(gpus):
    """Create a table showing GPU statistics."""
    table = Table(
        title="[bold cyan]GPU Status[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("GPU", justify="center", style="cyan", width=5)
    table.add_column("Name", justify="left", width=20)
    table.add_column("Util", justify="right", width=7)
    table.add_column("Memory", justify="right", width=18)
    table.add_column("Temp", justify="right", width=6)
    table.add_column("Power", justify="right", width=8)
    table.add_column("Status", justify="center", width=10)

    for gpu in gpus:
        # Color coding for utilization
        util = gpu['utilization']
        if util >= 80:
            util_style = "bold green"
            status = "[green]ACTIVE[/green]"
        elif util >= 30:
            util_style = "yellow"
            status = "[yellow]WORKING[/yellow]"
        elif util > 0:
            util_style = "dim yellow"
            status = "[dim]IDLE[/dim]"
        else:
            util_style = "dim red"
            status = "[red]IDLE[/red]"

        # Memory bar
        mem_pct = (gpu['memory_used'] / gpu['memory_total'] * 100) if gpu['memory_total'] > 0 else 0
        mem_str = f"{gpu['memory_used']/1024:.1f}/{gpu['memory_total']/1024:.0f}GB"

        # Temperature color
        temp = gpu['temperature']
        if temp >= 80:
            temp_style = "bold red"
        elif temp >= 70:
            temp_style = "yellow"
        else:
            temp_style = "green"

        table.add_row(
            str(gpu['index']),
            gpu['name'][:20],
            f"[{util_style}]{util:.0f}%[/{util_style}]",
            f"{mem_str} ({mem_pct:.0f}%)",
            f"[{temp_style}]{temp:.0f}C[/{temp_style}]",
            f"{gpu['power']:.0f}W",
            status
        )

    return table


def create_dataset_table(file_stats, progress_data):
    """Create a table showing per-dataset progress."""
    table = Table(
        title="[bold cyan]Dataset Progress[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("Dataset", justify="left", width=15)
    table.add_column("Accepted", justify="right", width=10)
    table.add_column("Rejected", justify="right", width=10)
    table.add_column("Acc Rate", justify="right", width=8)
    table.add_column("Hours", justify="right", width=8)
    table.add_column("Avg WER", justify="right", width=8)
    table.add_column("Status", justify="center", width=12)

    # Dataset order by priority
    dataset_order = ['librispeech', 'gigaspeech', 'voxpopuli', 'common_voice',
                     'tedlium', 'ami', 'peoples_speech', 'yodas']

    total_accepted = 0
    total_rejected = 0
    total_hours = 0.0

    datasets_info = progress_data.get('datasets', {})

    for name in dataset_order:
        stats = file_stats.get(name, {'accepted': 0, 'rejected': 0, 'hours': 0, 'wer_sum': 0})
        ds_info = datasets_info.get(name, {})

        accepted = stats['accepted']
        rejected = stats['rejected']
        hours = stats['hours']
        total = accepted + rejected

        total_accepted += accepted
        total_rejected += rejected
        total_hours += hours

        # Calculate rates
        acc_rate = (accepted / total * 100) if total > 0 else 0
        avg_wer = (stats['wer_sum'] / accepted * 100) if accepted > 0 else 0

        # Status from progress data
        status = ds_info.get('status', 'pending')
        if status == 'completed':
            status_str = "[green]DONE[/green]"
        elif status == 'processing':
            status_str = "[yellow]RUNNING[/yellow]"
        elif status == 'error':
            status_str = "[red]ERROR[/red]"
        elif total > 0:
            status_str = "[yellow]PARTIAL[/yellow]"
        else:
            status_str = "[dim]PENDING[/dim]"

        if total > 0:
            table.add_row(
                name,
                f"[green]{accepted:,}[/green]",
                f"[red]{rejected:,}[/red]",
                f"{acc_rate:.1f}%",
                f"{hours:.1f}",
                f"{avg_wer:.1f}%",
                status_str
            )
        else:
            table.add_row(
                f"[dim]{name}[/dim]",
                "[dim]0[/dim]",
                "[dim]0[/dim]",
                "[dim]-[/dim]",
                "[dim]0[/dim]",
                "[dim]-[/dim]",
                status_str
            )

    # Total row
    total_all = total_accepted + total_rejected
    total_acc_rate = (total_accepted / total_all * 100) if total_all > 0 else 0

    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold green]{total_accepted:,}[/bold green]",
        f"[bold red]{total_rejected:,}[/bold red]",
        f"[bold]{total_acc_rate:.1f}%[/bold]",
        f"[bold]{total_hours:.1f}[/bold]",
        "",
        ""
    )

    return table, total_accepted, total_hours


def create_gpu_distribution_table(file_stats, num_gpus):
    """Create a table showing sample distribution across GPUs."""
    table = Table(
        title="[bold cyan]GPU Work Distribution[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("GPU", justify="center", width=6)

    # Add column for each active dataset
    active_datasets = [name for name, stats in file_stats.items() if stats['accepted'] > 0]
    for name in active_datasets[:6]:  # Limit to 6 columns
        table.add_column(name[:10], justify="right", width=10)
    table.add_column("Total", justify="right", width=10)

    for gpu_id in range(num_gpus):
        row = [f"GPU {gpu_id}"]
        total = 0
        for name in active_datasets[:6]:
            count = file_stats[name]['gpu_counts'].get(gpu_id, 0)
            total += count
            row.append(f"{count:,}" if count > 0 else "[dim]0[/dim]")
        row.append(f"[bold]{total:,}[/bold]")
        table.add_row(*row)

    return table


def create_summary_panel(total_accepted, total_hours, start_time, gpus):
    """Create a summary panel with key metrics."""
    elapsed = datetime.now() - start_time
    elapsed_hours = elapsed.total_seconds() / 3600

    # Calculate rates
    samples_per_hour = total_accepted / elapsed_hours if elapsed_hours > 0 else 0
    hours_per_hour = total_hours / elapsed_hours if elapsed_hours > 0 else 0

    # GPU averages
    avg_util = sum(g['utilization'] for g in gpus) / len(gpus) if gpus else 0
    avg_mem = sum(g['memory_used'] for g in gpus) / len(gpus) / 1024 if gpus else 0
    total_power = sum(g['power'] for g in gpus)

    # Estimate remaining time (assuming 98,000 hours target)
    target_hours = 98000
    remaining_hours = max(0, target_hours - total_hours)
    eta_hours = remaining_hours / hours_per_hour if hours_per_hour > 0 else float('inf')

    if eta_hours < float('inf'):
        eta_str = f"{eta_hours:.1f} hours"
        if eta_hours > 24:
            eta_str = f"{eta_hours/24:.1f} days"
    else:
        eta_str = "calculating..."

    summary = f"""[bold cyan]Processing Summary[/bold cyan]

[green]Samples Accepted:[/green] {total_accepted:,}
[green]Audio Hours:[/green] {total_hours:.1f} / ~98,000 target

[yellow]Processing Rate:[/yellow]
  {samples_per_hour:,.0f} samples/hour
  {hours_per_hour:.2f} audio hours/hour

[cyan]GPU Performance:[/cyan]
  Avg Utilization: {avg_util:.0f}%
  Avg Memory: {avg_mem:.1f} GB
  Total Power: {total_power:.0f}W

[magenta]Time:[/magenta]
  Elapsed: {str(elapsed).split('.')[0]}
  ETA: {eta_str}
"""

    return Panel(summary, title="[bold]Summary[/bold]", border_style="cyan")


def create_dashboard(pseudo_labels_dir: Path, start_time: datetime):
    """Create the full dashboard layout."""
    # Get data
    gpus = get_gpu_stats()
    progress_data = get_progress_data(pseudo_labels_dir)
    file_stats = count_samples_from_files(pseudo_labels_dir)

    # Create tables
    gpu_table = create_gpu_table(gpus)
    dataset_table, total_accepted, total_hours = create_dataset_table(file_stats, progress_data)
    summary_panel = create_summary_panel(total_accepted, total_hours, start_time, gpus)

    # GPU distribution (if we have data)
    if any(stats['accepted'] > 0 for stats in file_stats.values()):
        gpu_dist_table = create_gpu_distribution_table(file_stats, len(gpus))
    else:
        gpu_dist_table = Panel("[dim]Waiting for data...[/dim]", title="GPU Distribution")

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )

    # Header
    header_text = Text()
    header_text.append("  DISTIL-CRISPERWHISPER PSEUDO-LABEL GENERATION MONITOR  ", style="bold white on blue")
    header_text.append(f"\n  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ", style="dim")
    header_text.append(f"  Output: {pseudo_labels_dir}", style="dim")
    layout["header"].update(Panel(header_text, style="blue"))

    # Main content
    layout["main"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1)
    )

    layout["left"].split_column(
        Layout(gpu_table, name="gpus"),
        Layout(dataset_table, name="datasets")
    )

    layout["right"].split_column(
        Layout(summary_panel, name="summary"),
        Layout(gpu_dist_table, name="distribution")
    )

    # Footer
    footer_text = Text()
    footer_text.append("  Press ", style="dim")
    footer_text.append("Ctrl+C", style="bold yellow")
    footer_text.append(" to exit  |  ", style="dim")
    footer_text.append("Auto-refresh: 1s", style="dim green")
    layout["footer"].update(Panel(footer_text, style="dim"))

    return layout


def main():
    parser = argparse.ArgumentParser(description='Monitor pseudo-label generation progress')
    parser.add_argument('--dir', type=str, default='/workspace/pseudo_labels',
                        help='Pseudo-labels directory to monitor')
    parser.add_argument('--refresh', type=float, default=1.0,
                        help='Refresh interval in seconds')
    args = parser.parse_args()

    pseudo_labels_dir = Path(args.dir)

    if not pseudo_labels_dir.exists():
        console.print(f"[yellow]Directory {pseudo_labels_dir} does not exist yet.[/yellow]")
        console.print("[yellow]Will start monitoring when it's created...[/yellow]")
        while not pseudo_labels_dir.exists():
            time.sleep(1)

    start_time = datetime.now()

    console.print(f"[bold green]Starting monitor for {pseudo_labels_dir}[/bold green]")
    console.print(f"[dim]Refresh rate: {args.refresh}s | Press Ctrl+C to exit[/dim]\n")

    try:
        # Use Live with screen=True for smooth updates without flashing
        # vertical_overflow="visible" prevents content from being cut off
        with Live(
            create_dashboard(pseudo_labels_dir, start_time),
            console=console,
            screen=True,  # Use alternate screen buffer - no flashing!
            refresh_per_second=4,  # Internal refresh rate (smooth)
            vertical_overflow="visible"
        ) as live:
            while True:
                time.sleep(args.refresh)
                live.update(create_dashboard(pseudo_labels_dir, start_time))
    except KeyboardInterrupt:
        pass  # Clean exit, screen buffer restored automatically

    console.print("\n[yellow]Monitor stopped.[/yellow]")


if __name__ == '__main__':
    main()
