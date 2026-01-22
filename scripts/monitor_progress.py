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
    from rich.console import Console, Group
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn, TaskProgressColumn
    from rich import box
except ImportError:
    print("Installing rich library...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console, Group
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn, TaskProgressColumn
    from rich import box

console = Console()

# Dataset sample counts - RESEARCHED ACTUAL values from HuggingFace documentation
# Sources:
# - LibriSpeech: https://huggingface.co/datasets/openslr/librispeech_asr
# - AMI: https://huggingface.co/datasets/edinburghcstr/ami
# - PodcastFillers: https://huggingface.co/datasets/ylacombe/podcast_fillers_by_license
# - VoxPopuli: https://huggingface.co/datasets/facebook/voxpopuli
# - TED-LIUM: https://www.tensorflow.org/datasets/catalog/tedlium (release3)
# - GigaSpeech: https://huggingface.co/datasets/speechcolab/gigaspeech
#
# The processing script will also create {dataset}_metadata.json with exact counts
# when downloading - those override these values if present.
#
# ORDER: Matches priority order in 02_generate_pseudo_labels_multi_gpu.py
DATASET_ESTIMATES = {
    # Priority 1: LibriSpeech - High quality, clean audio
    # train.clean.100 (28,539) + train.clean.360 (104,014) + train.other.500 (148,688)
    'librispeech': {'hours': 960, 'samples': 281241},

    # Priority 2: AMI IHM - Critical for filler preservation
    # 108,502 samples (from dataset card example output)
    'ami': {'hours': 100, 'samples': 108502},

    # Priority 3: PodcastFillers - Explicit filler annotations
    # 199 full episodes split by license (CC_BY_3.0: 100, CC_BY_SA_3.0: 79, CC_BY_ND_3.0: 20)
    'podcast_fillers': {'hours': 145, 'samples': 199},

    # Priority 4: VoxPopuli EN - European Parliament formal speech
    # 543 hours transcribed, ~182k samples estimated
    'voxpopuli': {'hours': 543, 'samples': 182000},

    # Priority 5: Common Voice 17 EN - Crowdsourced diverse speakers
    # Requires Mozilla Data Collective access, estimate ~1.2M
    'common_voice': {'hours': 3000, 'samples': 1200000},

    # Priority 6: TED-LIUM release3 - Professional presentations
    # 268,263 samples (from TensorFlow Datasets catalog)
    'tedlium': {'hours': 450, 'samples': 268263},

    # Priority 7: GigaSpeech XL - YouTube/podcasts/audiobooks
    # 10,000 hours - XS has 9,389 samples for 10 hours, so XL ~2.5-3M
    'gigaspeech': {'hours': 10000, 'samples': 2500000},

    # Priority 8: People's Speech - Large scale diverse
    # Streaming only - large estimates
    'peoples_speech': {'hours': 30000, 'samples': 9000000},

    # Priority 9: YODAS - Massive YouTube dataset
    # Streaming only - very large estimates
    'yodas': {'hours': 150000, 'samples': 50000000},
}


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


def load_actual_sample_counts(pseudo_labels_dir: Path) -> dict:
    """
    Load ACTUAL sample counts from metadata files created by processing script.

    The processing script saves {dataset}_metadata.json with counts:
    - 'downloaded_exact': Exact count from len(dataset) when downloaded
    - 'actual_processed': Verified count from completed processing
    - 'estimated': Rough estimate for streaming datasets (based on hours)
    - 'previous_completion': Using count from a previous completed run

    The 'count_verified' field indicates if the count is exact (True) or estimated (False).
    """
    actual_counts = {}

    # Look for all metadata files
    metadata_files = list(pseudo_labels_dir.glob('*_metadata.json'))

    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                name = metadata.get('dataset')
                total = metadata.get('total_samples')
                if name and total:
                    source = metadata.get('source', 'unknown')
                    is_streaming = metadata.get('is_streaming', False)
                    count_verified = metadata.get('count_verified', False)

                    # Determine display source
                    if source == 'actual_processed' or metadata.get('completed', False):
                        display_source = 'verified'
                    elif source == 'downloaded_exact':
                        display_source = 'exact'
                    elif source == 'estimated':
                        display_source = 'estimate'
                    elif source == 'previous_completion':
                        display_source = 'previous'
                    else:
                        display_source = source

                    # actual_iterated_count is the TRUE count from dataset iterator
                    # This is more accurate than len(dataset) for concatenated datasets
                    actual_iterated_count = metadata.get('actual_iterated_count')

                    # For progress tracking, prefer actual_iterated_count (verified iterator total)
                    # over total_samples (which may be from len(dataset) or processed count)
                    best_total = actual_iterated_count if actual_iterated_count else total

                    actual_counts[name] = {
                        'samples': best_total,  # Use verified iterator count when available
                        'source': display_source,
                        'is_streaming': is_streaming,
                        'count_verified': count_verified or (actual_iterated_count is not None),
                        'hours': metadata.get('estimated_hours', DATASET_ESTIMATES.get(name, {}).get('hours', 0)),
                        'completed': metadata.get('completed', False),
                        'verified': metadata.get('verified', False),
                        'verification_status': metadata.get('verification_status', ''),
                        'unique_samples': metadata.get('unique_samples', total),
                        'original_estimate': metadata.get('original_estimate', total),
                        'actual_iterated_count': actual_iterated_count,
                        'processed_count': total,  # Keep original processed count too
                    }
        except Exception:
            continue

    return actual_counts


def get_dataset_estimates(pseudo_labels_dir: Path, file_stats: dict, progress_data: dict) -> dict:
    """
    Get the best available sample estimates for each dataset.

    Priority:
    1. Actual counts from metadata files (created when datasets are downloaded)
    2. Actual counts from completed datasets (all samples processed)
    3. Fallback to hardcoded estimates

    Returns dict of {dataset_name: {'samples': count, 'hours': hours, 'source': source}}
    """
    # Start with hardcoded estimates as fallback
    estimates = {}
    for name, est in DATASET_ESTIMATES.items():
        estimates[name] = {
            'samples': est['samples'],
            'hours': est['hours'],
            'source': 'estimate'
        }

    # Override with actual counts from metadata files (highest priority)
    # These are created by the processing script when it downloads datasets
    actual_counts = load_actual_sample_counts(pseudo_labels_dir)
    for name, data in actual_counts.items():
        estimates[name] = data

    # NOTE: We do NOT use "completed" status to determine total sample count
    # because a dataset can be marked "completed" even if it was interrupted
    # or if only a subset was processed. The metadata files and hardcoded
    # values are the source of truth for actual dataset sizes.

    return estimates


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
    table.add_column("Name", justify="left", width=18)
    table.add_column("Util", justify="right", width=6)
    table.add_column("Memory", justify="right", width=16)
    table.add_column("Temp", justify="right", width=5)
    table.add_column("Power", justify="right", width=6)
    table.add_column("Status", justify="center", width=8)

    for gpu in gpus:
        # Color coding for utilization
        util = gpu['utilization']
        if util >= 80:
            util_style = "bold green"
            status = "[green]ACTIVE[/green]"
        elif util >= 30:
            util_style = "yellow"
            status = "[yellow]WORK[/yellow]"
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
            gpu['name'][:18],
            f"[{util_style}]{util:.0f}%[/{util_style}]",
            f"{mem_str}",
            f"[{temp_style}]{temp:.0f}C[/{temp_style}]",
            f"{gpu['power']:.0f}W",
            status
        )

    return table


def make_progress_bar(current, total, width=20):
    """Create a text-based progress bar.

    Handles cases where current > total (more samples processed than metadata predicted).
    This can happen when dataset iteration yields slightly more samples than len(dataset).
    """
    if total == 0:
        return "[dim]" + "-" * width + "[/dim]"

    # Calculate actual ratio (can be > 1.0 if processed more than expected)
    ratio = current / total
    # Cap at 1.0 for display purposes
    pct = min(ratio, 1.0)
    filled = int(pct * width)
    empty = width - filled

    if ratio >= 1.0:
        # Complete - show full bar (green for exactly done, cyan for exceeded)
        if ratio > 1.0:
            return f"[cyan]{'█' * width}[/cyan]"  # Cyan = exceeded expected
        return f"[green]{'█' * width}[/green]"
    elif pct > 0:
        return f"[green]{'█' * filled}[/green][dim]{'░' * empty}[/dim]"
    else:
        return f"[dim]{'░' * width}[/dim]"


def create_dataset_table(file_stats, progress_data, pseudo_labels_dir: Path):
    """Create a table showing per-dataset progress with progress bars.

    IMPORTANT: We use file_stats (from actual JSONL files) as the source of truth
    for sample counts, NOT progress_data (which only tracks GPU 0's view).
    """
    table = Table(
        title="[bold cyan]Dataset Progress[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("Dataset", justify="left", width=13)
    table.add_column("Progress", justify="left", width=22)
    table.add_column("Processed", justify="right", width=12)
    table.add_column("Remaining", justify="right", width=12)
    table.add_column("Acc%", justify="right", width=5)
    table.add_column("WER", justify="right", width=5)
    table.add_column("Status", justify="center", width=9)

    # Dataset order by priority (matches 02_generate_pseudo_labels_multi_gpu.py)
    # 1=librispeech, 2=ami, 3=podcast_fillers, 4=voxpopuli, 5=common_voice,
    # 6=tedlium, 7=gigaspeech, 8=peoples_speech, 9=yodas
    dataset_order = ['librispeech', 'ami', 'podcast_fillers', 'voxpopuli',
                     'common_voice', 'tedlium', 'gigaspeech', 'peoples_speech', 'yodas']

    total_processed = 0
    total_estimated = 0
    total_accepted = 0
    total_hours = 0.0

    datasets_info = progress_data.get('datasets', {})
    current_dataset = None

    # Get best available estimates (actual counts from metadata when available)
    all_estimates = get_dataset_estimates(pseudo_labels_dir, file_stats, progress_data)

    for name in dataset_order:
        # Use file_stats as source of truth (actual JSONL file contents)
        stats = file_stats.get(name, {'accepted': 0, 'rejected': 0, 'hours': 0, 'wer_sum': 0})
        ds_info = datasets_info.get(name, {})
        estimates = all_estimates.get(name, {'samples': 0, 'source': 'estimate'})

        # Get counts from actual files (not progress tracker which only has GPU 0 stats)
        accepted = stats['accepted']
        rejected = stats['rejected']
        hours = stats['hours']
        processed = accepted + rejected
        estimated_total = estimates['samples']

        total_processed += processed
        total_estimated += estimated_total
        total_accepted += accepted
        total_hours += hours

        remaining = max(0, estimated_total - processed)

        # Calculate rates
        acc_rate = (accepted / processed * 100) if processed > 0 else 0
        avg_wer = (stats['wer_sum'] / accepted * 100) if accepted > 0 else 0

        # Status from progress data
        status = ds_info.get('status', 'pending')
        verification_status = estimates.get('verification_status', '')
        is_verified = estimates.get('verified', False)

        if status == 'completed':
            # Show verification status for completed datasets
            if is_verified:
                if verification_status == 'verified':
                    status_str = "[bold green]VERIFIED[/bold green]"
                elif verification_status == 'exceeded':
                    status_str = "[cyan]DONE+[/cyan]"  # Exceeded expected
                elif verification_status == 'missing_samples':
                    status_str = "[yellow]DONE*[/yellow]"  # Some samples missing (likely deduped)
                else:
                    status_str = "[green]DONE[/green]"
            else:
                status_str = "[green]DONE[/green]"
        elif status == 'processing':
            status_str = "[bold yellow]RUNNING[/bold yellow]"
            current_dataset = name
        elif status == 'error':
            status_str = "[red]ERROR[/red]"
        elif processed > 0:
            status_str = "[yellow]PARTIAL[/yellow]"
        else:
            status_str = "[dim]PENDING[/dim]"

        # Progress bar
        progress_bar = make_progress_bar(processed, estimated_total, width=15)
        pct = (processed / estimated_total * 100) if estimated_total > 0 else 0
        # Show actual percentage (can exceed 100% if more samples than metadata predicted)
        # This is informative - shows dataset iteration yielded more than len(dataset)
        if pct > 100:
            progress_str = f"{progress_bar} [cyan]{pct:4.1f}%[/cyan]"
        else:
            progress_str = f"{progress_bar} {pct:4.1f}%"

        if processed > 0 or status == 'processing':
            # If processed > estimated, show 0 remaining but indicate completion
            remaining_display = f"[yellow]{remaining:,}[/yellow]" if remaining > 0 else "[green]0[/green]"
            table.add_row(
                f"[bold]{name}[/bold]" if status == 'processing' else name,
                progress_str,
                f"[cyan]{processed:,}[/cyan]",
                remaining_display,
                f"{acc_rate:.0f}%",
                f"{avg_wer:.1f}%" if accepted > 0 else "-",
                status_str
            )
        else:
            table.add_row(
                f"[dim]{name}[/dim]",
                f"[dim]{progress_bar}   0%[/dim]",
                "[dim]0[/dim]",
                f"[dim]{estimated_total:,}[/dim]",
                "[dim]-[/dim]",
                "[dim]-[/dim]",
                status_str
            )

    # Total row
    total_pct = (total_processed / total_estimated * 100) if total_estimated > 0 else 0
    total_remaining = max(0, total_estimated - total_processed)  # Don't show negative remaining
    total_acc_rate = (total_accepted / total_processed * 100) if total_processed > 0 else 0

    # Format percentage - show cyan if exceeded expected
    if total_pct > 100:
        pct_str = f"[cyan]{total_pct:4.1f}%[/cyan]"
    else:
        pct_str = f"{total_pct:4.1f}%"

    table.add_row(
        "[bold]TOTAL[/bold]",
        f"{make_progress_bar(total_processed, total_estimated, width=15)} {pct_str}",
        f"[bold cyan]{total_processed:,}[/bold cyan]",
        f"[bold yellow]{total_remaining:,}[/bold yellow]" if total_remaining > 0 else "[bold green]0[/bold green]",
        f"[bold]{total_acc_rate:.0f}%[/bold]",
        "",
        ""
    )

    return table, total_accepted, total_hours, current_dataset, total_processed, total_estimated


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
    for name in active_datasets[:5]:  # Limit to 5 columns
        table.add_column(name[:8], justify="right", width=9)
    table.add_column("Total", justify="right", width=10)

    for gpu_id in range(num_gpus):
        row = [f"GPU {gpu_id}"]
        total = 0
        for name in active_datasets[:5]:
            count = file_stats[name]['gpu_counts'].get(gpu_id, 0)
            total += count
            row.append(f"{count:,}" if count > 0 else "[dim]0[/dim]")
        row.append(f"[bold]{total:,}[/bold]")
        table.add_row(*row)

    return table


def create_summary_panel(total_accepted, total_hours, start_time, gpus, current_dataset,
                         total_processed, total_estimated):
    """Create a summary panel with key metrics."""
    elapsed = datetime.now() - start_time
    elapsed_hours = elapsed.total_seconds() / 3600
    elapsed_seconds = elapsed.total_seconds()

    # Calculate rates
    samples_per_hour = total_processed / elapsed_hours if elapsed_hours > 0 else 0
    samples_per_second = total_processed / elapsed_seconds if elapsed_seconds > 0 else 0

    # GPU averages
    avg_util = sum(g['utilization'] for g in gpus) / len(gpus) if gpus else 0
    avg_mem = sum(g['memory_used'] for g in gpus) / len(gpus) / 1024 if gpus else 0
    total_power = sum(g['power'] for g in gpus)

    # ETA based on actual samples remaining
    remaining_samples = total_estimated - total_processed
    if samples_per_second > 0 and remaining_samples > 0:
        eta_seconds = remaining_samples / samples_per_second
        eta = timedelta(seconds=int(eta_seconds))
        if eta_seconds > 86400:  # More than 1 day
            eta_str = f"{eta_seconds/86400:.1f} days"
        elif eta_seconds > 3600:  # More than 1 hour
            eta_str = f"{eta_seconds/3600:.1f} hours"
        else:
            eta_str = str(eta)
    else:
        eta_str = "calculating..."

    # Current dataset indicator
    current_str = f"[bold yellow]{current_dataset}[/bold yellow]" if current_dataset else "[dim]None[/dim]"

    summary = f"""[bold cyan]Processing Summary[/bold cyan]

[green]Currently Processing:[/green] {current_str}

[green]Progress:[/green]
  Samples: [cyan]{total_processed:,}[/cyan] / {total_estimated:,}
  Accepted: [green]{total_accepted:,}[/green]
  Hours of Audio: [cyan]{total_hours:.1f}[/cyan]

[yellow]Processing Rate:[/yellow]
  [cyan]{samples_per_second:.1f}[/cyan] samples/sec
  [cyan]{samples_per_hour:,.0f}[/cyan] samples/hour

[cyan]GPU Performance:[/cyan]
  Avg Util: [green]{avg_util:.0f}%[/green]
  Avg Mem: {avg_mem:.1f} GB
  Power: {total_power:.0f}W

[magenta]Time:[/magenta]
  Elapsed: {str(elapsed).split('.')[0]}
  ETA: [yellow]{eta_str}[/yellow]
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
    dataset_table, total_accepted, total_hours, current_dataset, total_processed, total_estimated = \
        create_dataset_table(file_stats, progress_data, pseudo_labels_dir)
    summary_panel = create_summary_panel(
        total_accepted, total_hours, start_time, gpus,
        current_dataset, total_processed, total_estimated
    )

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

    # Footer - check if we have actual counts from metadata files
    all_estimates = get_dataset_estimates(pseudo_labels_dir, file_stats, progress_data)
    # Count datasets with verified (completed) or actual (from metadata) counts
    actual_count = sum(1 for e in all_estimates.values()
                       if e.get('source') in ('actual', 'verified'))
    total_datasets = len(all_estimates)

    footer_text = Text()
    footer_text.append("  Press ", style="dim")
    footer_text.append("Ctrl+C", style="bold yellow")
    footer_text.append(" to exit  |  ", style="dim")
    footer_text.append("Auto-refresh: 1s", style="dim green")
    footer_text.append("  |  ", style="dim")
    if actual_count > 0:
        footer_text.append(f"Using ACTUAL counts for {actual_count}/{total_datasets} datasets", style="green")
    else:
        footer_text.append(f"Using estimates (waiting for metadata)", style="yellow")
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
