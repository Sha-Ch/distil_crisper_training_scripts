#!/usr/bin/env python3
"""
=============================================================================
Reprocess Existing Pseudo-Labels with Updated WER Logic
=============================================================================
Reprocesses existing pseudo-label JSONL files using the new spelling-tolerant
WER calculation and filler word handling. Can move samples between accepted
and rejected based on the new logic.

Features:
- Multithreaded processing for maximum throughput
- Spelling-tolerant WER (British/American variants accepted)
- Filler word stripping for fair WER comparison
- Bidirectional movement: rejected->accepted AND accepted->rejected
- Preserves original files with .backup suffix
- Progress tracking and resumable

Usage:
  python3 04_reprocess_pseudo_labels.py --input-dir /path/to/pseudo_labels --wer-threshold 0.10
  python3 04_reprocess_pseudo_labels.py --input-dir /path/to/pseudo_labels --dry-run  # Preview changes
  python3 04_reprocess_pseudo_labels.py --input-dir /path/to/pseudo_labels --threads 16

=============================================================================
"""

import os
import sys
import json
import argparse
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from difflib import SequenceMatcher
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
except ImportError:
    print("ERROR: transformers library required. Install with: pip install transformers")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    # Fallback if rich not installed
    class FallbackConsole:
        def print(self, *args, **kwargs):
            # Strip rich markup for fallback
            text = str(args[0]) if args else ""
            text = re.sub(r'\[.*?\]', '', text)
            print(text)
    console = FallbackConsole()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ReprocessStats:
    """Track reprocessing statistics."""
    total_processed: int = 0
    accepted_to_rejected: int = 0
    rejected_to_accepted: int = 0
    unchanged_accepted: int = 0
    unchanged_rejected: int = 0
    errors: int = 0

    # Per-dataset stats
    dataset_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add_dataset(self, name: str):
        if name not in self.dataset_stats:
            self.dataset_stats[name] = {
                'total': 0,
                'accepted_to_rejected': 0,
                'rejected_to_accepted': 0,
                'unchanged_accepted': 0,
                'unchanged_rejected': 0,
                'errors': 0,
            }


@dataclass
class PseudoLabelEntry:
    """Represents a single pseudo-label entry."""
    sample_id: str
    ground_truth: str
    pseudo_label: str
    word_timestamps: List[Dict[str, Any]]
    wer: float
    duration_seconds: float
    audio_path: Optional[str] = None
    accepted: bool = True
    rejection_reason: Optional[str] = None
    original_wer: Optional[float] = None  # Track original WER for comparison

    def to_dict(self) -> dict:
        return {
            'sample_id': self.sample_id,
            'ground_truth': self.ground_truth,
            'pseudo_label': self.pseudo_label,
            'word_timestamps': self.word_timestamps,
            'wer': self.wer,
            'duration_seconds': self.duration_seconds,
            'audio_path': self.audio_path,
            'accepted': self.accepted,
            'rejection_reason': self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PseudoLabelEntry':
        return cls(
            sample_id=data.get('sample_id', ''),
            ground_truth=data.get('ground_truth', ''),
            pseudo_label=data.get('pseudo_label', ''),
            word_timestamps=data.get('word_timestamps', []),
            wer=data.get('wer', 0.0),
            duration_seconds=data.get('duration_seconds', 0.0),
            audio_path=data.get('audio_path'),
            accepted=data.get('accepted', True),
            rejection_reason=data.get('rejection_reason'),
            original_wer=data.get('wer', 0.0),  # Store original
        )


# =============================================================================
# WER Calculation (matching 02_generate_pseudo_labels_multi_gpu.py)
# =============================================================================

class WERCalculator:
    """
    Calculates WER with spelling tolerance and filler word handling.
    Matches the logic in 02_generate_pseudo_labels_multi_gpu.py.
    """

    # Filler words to strip (CrisperWhisper may include these)
    FILLER_WORDS = {'um', 'uh', 'er', 'ah', 'uhm', 'erm', 'hmm', 'hm', 'mm', 'mhm', 'uh huh', 'mm hmm'}

    def __init__(self, spelling_threshold: float = 0.85):
        """
        Args:
            spelling_threshold: Similarity threshold for spelling variants (default 0.85)
        """
        self.spelling_threshold = spelling_threshold
        self._english_normalizer = EnglishTextNormalizer({})

    def _are_spelling_variants(self, word1: str, word2: str) -> bool:
        """
        Check if two words are spelling variants (e.g., British vs American).

        Uses difflib.SequenceMatcher to check similarity ratio.
        Examples: colour/color (0.91), realise/realize (0.86), behaviour/behavior (0.94)
        """
        if word1 == word2:
            return True
        # Must start with same letter to be a spelling variant
        if not word1 or not word2 or word1[0] != word2[0]:
            return False
        # Check similarity ratio
        return SequenceMatcher(None, word1, word2).ratio() >= self.spelling_threshold

    def _calculate_wer_spelling_tolerant(self, ref_words: list, hyp_words: list) -> float:
        """
        Calculate WER with tolerance for British/American spelling variants.

        Uses dynamic programming (Levenshtein distance) but treats spelling variants
        as matches rather than substitutions.
        """
        n = len(ref_words)
        m = len(hyp_words)

        # Edge cases
        if n == 0:
            return 1.0 if m > 0 else 0.0
        if m == 0:
            return 1.0

        # DP table
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            dp[i][0] = i  # Deletions
        for j in range(m + 1):
            dp[0][j] = j  # Insertions

        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Check if words match (exact or spelling variant)
                if self._are_spelling_variants(ref_words[i-1], hyp_words[j-1]):
                    dp[i][j] = dp[i-1][j-1]  # No cost - words are equivalent
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,      # Deletion
                        dp[i][j-1] + 1,      # Insertion
                        dp[i-1][j-1] + 1     # Substitution
                    )

        # WER = edit_distance / reference_length
        return dp[n][m] / n

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for WER calculation using official Whisper EnglishTextNormalizer.

        Additionally strips filler words so CrisperWhisper's verbatim transcription
        (with um, uh, etc.) doesn't get penalized when ground truth doesn't include them.
        """
        if not text:
            return ""

        # Remove bracketed fillers first (CrisperWhisper format: [Um], [Uh], etc.)
        text = re.sub(r'\[(?:um|uh|er|ah|uhm|erm|hmm|hm|mm|mhm)\]', '', text, flags=re.IGNORECASE)

        # Apply official Whisper English normalizer
        text = self._english_normalizer(text)

        # Strip standalone filler words
        words = text.split()
        words = [w for w in words if w not in self.FILLER_WORDS]

        return ' '.join(words)

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate with tolerance for spelling variants and fillers.

        Returns:
            WER as float between 0.0 and 1.0 (or higher for very bad transcriptions)
        """
        # Check for all-uppercase transcription (erroneous generation from Whisper)
        if hypothesis is not None and hypothesis.upper() == hypothesis and len(hypothesis) > 0:
            return 1.0  # Reject all-caps as error

        # Normalize both texts
        ref = self._normalize_text(reference)
        hyp = self._normalize_text(hypothesis)

        # Handle edge cases
        if not ref:
            return 1.0  # Can't compute WER with empty reference, reject
        if not hyp:
            return 1.0  # Empty hypothesis is 100% error

        try:
            ref_words = ref.split()
            hyp_words = hyp.split()
            return self._calculate_wer_spelling_tolerant(ref_words, hyp_words)
        except Exception:
            return 1.0


# =============================================================================
# Reprocessor Class
# =============================================================================

class PseudoLabelReprocessor:
    """
    Reprocesses existing pseudo-label files with updated WER logic.
    """

    def __init__(
        self,
        input_dir: Path,
        wer_threshold: float = 0.10,
        num_threads: int = 8,
        dry_run: bool = False,
        backup: bool = True,
    ):
        """
        Args:
            input_dir: Directory containing pseudo-label JSONL files
            wer_threshold: WER threshold for acceptance (default 0.10 = 10%)
            num_threads: Number of threads for parallel processing
            dry_run: If True, don't write changes, just report what would happen
            backup: If True, create .backup files before modifying
        """
        self.input_dir = Path(input_dir)
        self.wer_threshold = wer_threshold
        self.num_threads = num_threads
        self.dry_run = dry_run
        self.backup = backup

        # WER calculator with new logic
        self.wer_calculator = WERCalculator()

        # Thread-safe stats
        self.stats = ReprocessStats()
        self._stats_lock = Lock()

        # Validate input directory
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")

    def _find_dataset_files(self) -> Dict[str, Dict[str, List[Path]]]:
        """
        Find all accepted and rejected JSONL files grouped by dataset.

        Returns:
            Dict mapping dataset name -> {'accepted': [paths], 'rejected': [paths]}
        """
        datasets = {}

        # Find all JSONL files
        for filepath in self.input_dir.glob('*_gpu*_accepted.jsonl'):
            # Extract dataset name (everything before _gpu)
            name = filepath.name.rsplit('_gpu', 1)[0]
            if name not in datasets:
                datasets[name] = {'accepted': [], 'rejected': []}
            datasets[name]['accepted'].append(filepath)

        for filepath in self.input_dir.glob('*_gpu*_rejected.jsonl'):
            name = filepath.name.rsplit('_gpu', 1)[0]
            if name not in datasets:
                datasets[name] = {'accepted': [], 'rejected': []}
            datasets[name]['rejected'].append(filepath)

        return datasets

    def _process_entry(self, entry: PseudoLabelEntry, was_accepted: bool) -> Tuple[PseudoLabelEntry, str]:
        """
        Process a single entry with new WER logic.

        Returns:
            Tuple of (updated_entry, status) where status is one of:
            - 'accepted_to_rejected': Was accepted, now rejected
            - 'rejected_to_accepted': Was rejected, now accepted
            - 'unchanged_accepted': Was and still is accepted
            - 'unchanged_rejected': Was and still is rejected
            - 'error': Processing error
        """
        try:
            # Recalculate WER with new logic
            new_wer = self.wer_calculator.calculate_wer(
                entry.ground_truth,
                entry.pseudo_label
            )

            # Determine new acceptance status
            new_accepted = new_wer <= self.wer_threshold

            # Update entry
            entry.original_wer = entry.wer
            entry.wer = new_wer
            entry.accepted = new_accepted

            if new_accepted:
                entry.rejection_reason = None
            else:
                entry.rejection_reason = f"WER {new_wer:.2%} > {self.wer_threshold:.0%}"

            # Determine status
            if was_accepted and not new_accepted:
                return entry, 'accepted_to_rejected'
            elif not was_accepted and new_accepted:
                return entry, 'rejected_to_accepted'
            elif was_accepted and new_accepted:
                return entry, 'unchanged_accepted'
            else:
                return entry, 'unchanged_rejected'

        except Exception as e:
            return entry, 'error'

    def _process_file_batch(
        self,
        entries: List[Tuple[PseudoLabelEntry, bool]],
    ) -> List[Tuple[PseudoLabelEntry, str]]:
        """
        Process a batch of entries in parallel.

        Args:
            entries: List of (entry, was_accepted) tuples

        Returns:
            List of (updated_entry, status) tuples
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self._process_entry, entry, was_accepted): (entry, was_accepted)
                for entry, was_accepted in entries
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception:
                    entry, was_accepted = futures[future]
                    results.append((entry, 'error'))

        return results

    def _load_jsonl_file(self, filepath: Path) -> List[PseudoLabelEntry]:
        """Load entries from a JSONL file."""
        entries = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            entries.append(PseudoLabelEntry.from_dict(data))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            console.print(f"[red]Error loading {filepath}: {e}[/red]")
        return entries

    def _write_jsonl_file(self, filepath: Path, entries: List[PseudoLabelEntry]):
        """Write entries to a JSONL file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict()) + '\n')

    def process_dataset(self, name: str, files: Dict[str, List[Path]]) -> Dict[str, int]:
        """
        Process all files for a single dataset.

        Args:
            name: Dataset name
            files: Dict with 'accepted' and 'rejected' file lists

        Returns:
            Stats dict for this dataset
        """
        self.stats.add_dataset(name)
        dataset_stats = self.stats.dataset_stats[name]

        # Load all entries from accepted files
        accepted_entries = []
        for filepath in files['accepted']:
            entries = self._load_jsonl_file(filepath)
            for entry in entries:
                accepted_entries.append((entry, True))

        # Load all entries from rejected files
        rejected_entries = []
        for filepath in files['rejected']:
            entries = self._load_jsonl_file(filepath)
            for entry in entries:
                rejected_entries.append((entry, False))

        all_entries = accepted_entries + rejected_entries
        total_entries = len(all_entries)

        if total_entries == 0:
            console.print(f"[yellow]No entries found for {name}[/yellow]")
            return dataset_stats

        console.print(f"\n[bold]Processing {name}:[/bold] {total_entries:,} entries ({len(accepted_entries):,} accepted, {len(rejected_entries):,} rejected)")

        # Process all entries with progress bar
        results = []
        batch_size = 1000

        with tqdm(total=total_entries, desc=f"  Reprocessing {name}", unit="samples") as pbar:
            for i in range(0, total_entries, batch_size):
                batch = all_entries[i:i + batch_size]
                batch_results = self._process_file_batch(batch)
                results.extend(batch_results)
                pbar.update(len(batch))

        # Separate results into new accepted and rejected
        new_accepted = []
        new_rejected = []

        for entry, status in results:
            dataset_stats['total'] += 1

            if status == 'accepted_to_rejected':
                dataset_stats['accepted_to_rejected'] += 1
                new_rejected.append(entry)
            elif status == 'rejected_to_accepted':
                dataset_stats['rejected_to_accepted'] += 1
                new_accepted.append(entry)
            elif status == 'unchanged_accepted':
                dataset_stats['unchanged_accepted'] += 1
                new_accepted.append(entry)
            elif status == 'unchanged_rejected':
                dataset_stats['unchanged_rejected'] += 1
                new_rejected.append(entry)
            else:  # error
                dataset_stats['errors'] += 1
                # Keep in original location on error
                if entry.accepted:
                    new_accepted.append(entry)
                else:
                    new_rejected.append(entry)

        # Update global stats
        with self._stats_lock:
            self.stats.total_processed += dataset_stats['total']
            self.stats.accepted_to_rejected += dataset_stats['accepted_to_rejected']
            self.stats.rejected_to_accepted += dataset_stats['rejected_to_accepted']
            self.stats.unchanged_accepted += dataset_stats['unchanged_accepted']
            self.stats.unchanged_rejected += dataset_stats['unchanged_rejected']
            self.stats.errors += dataset_stats['errors']

        # Print dataset results
        console.print(f"  [green]Rescued (rejected->accepted): {dataset_stats['rejected_to_accepted']:,}[/green]")
        console.print(f"  [red]Demoted (accepted->rejected): {dataset_stats['accepted_to_rejected']:,}[/red]")
        console.print(f"  [dim]Unchanged accepted: {dataset_stats['unchanged_accepted']:,}[/dim]")
        console.print(f"  [dim]Unchanged rejected: {dataset_stats['unchanged_rejected']:,}[/dim]")

        if self.dry_run:
            console.print(f"  [yellow]DRY RUN - No changes written[/yellow]")
            return dataset_stats

        # Write new files (grouped by original GPU to maintain structure)
        # First, backup original files
        if self.backup:
            for filepath in files['accepted'] + files['rejected']:
                backup_path = filepath.with_suffix('.jsonl.backup')
                if not backup_path.exists():
                    shutil.copy2(filepath, backup_path)

        # Distribute entries back to GPU files maintaining original distribution
        # For simplicity, we'll write all accepted to gpu0 and rejected to gpu0
        # (or we could distribute evenly across original GPU files)

        if files['accepted']:
            # Use first accepted file as template for output
            main_accepted = files['accepted'][0]
            # Replace gpu number with 'reprocessed'
            output_accepted = self.input_dir / f'{name}_reprocessed_accepted.jsonl'
            self._write_jsonl_file(output_accepted, new_accepted)
            console.print(f"  [green]Wrote {len(new_accepted):,} accepted samples to {output_accepted.name}[/green]")

        if files['rejected'] or new_rejected:
            output_rejected = self.input_dir / f'{name}_reprocessed_rejected.jsonl'
            self._write_jsonl_file(output_rejected, new_rejected)
            console.print(f"  [yellow]Wrote {len(new_rejected):,} rejected samples to {output_rejected.name}[/yellow]")

        return dataset_stats

    def run(self):
        """Run the reprocessing pipeline."""
        console.print(Panel(
            f"[bold]Pseudo-Label Reprocessor[/bold]\n\n"
            f"Input Directory: {self.input_dir}\n"
            f"WER Threshold: {self.wer_threshold * 100:.0f}%\n"
            f"Threads: {self.num_threads}\n"
            f"Dry Run: {self.dry_run}\n"
            f"Backup: {self.backup}",
            title="Configuration"
        ))

        # Find all dataset files
        datasets = self._find_dataset_files()

        if not datasets:
            console.print("[red]No pseudo-label files found in input directory[/red]")
            return

        console.print(f"\n[bold]Found {len(datasets)} datasets:[/bold]")
        for name, files in datasets.items():
            accepted_count = sum(1 for f in files['accepted'] for _ in open(f))
            rejected_count = sum(1 for f in files['rejected'] for _ in open(f) if files['rejected'])
            console.print(f"  - {name}: {len(files['accepted'])} accepted files, {len(files['rejected'])} rejected files")

        # Process each dataset
        start_time = datetime.now()

        for name, files in datasets.items():
            self.process_dataset(name, files)

        elapsed = datetime.now() - start_time

        # Print final summary
        console.print("\n" + "=" * 60)
        console.print(Panel(
            f"[bold]Reprocessing Complete[/bold]\n\n"
            f"Total Processed: {self.stats.total_processed:,}\n"
            f"[green]Rescued (rejected->accepted): {self.stats.rejected_to_accepted:,}[/green]\n"
            f"[red]Demoted (accepted->rejected): {self.stats.accepted_to_rejected:,}[/red]\n"
            f"Unchanged Accepted: {self.stats.unchanged_accepted:,}\n"
            f"Unchanged Rejected: {self.stats.unchanged_rejected:,}\n"
            f"Errors: {self.stats.errors:,}\n\n"
            f"Time Elapsed: {elapsed}",
            title="Summary"
        ))

        # Net change
        net_change = self.stats.rejected_to_accepted - self.stats.accepted_to_rejected
        if net_change > 0:
            console.print(f"\n[bold green]Net gain: +{net_change:,} samples now accepted[/bold green]")
        elif net_change < 0:
            console.print(f"\n[bold red]Net loss: {net_change:,} samples now rejected[/bold red]")
        else:
            console.print(f"\n[bold]No net change in accepted samples[/bold]")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Reprocess pseudo-labels with updated WER logic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reprocess with default settings
  python 04_reprocess_pseudo_labels.py --input-dir ./pseudo_labels

  # Preview changes without writing (dry run)
  python 04_reprocess_pseudo_labels.py --input-dir ./pseudo_labels --dry-run

  # Use stricter WER threshold (5%)
  python 04_reprocess_pseudo_labels.py --input-dir ./pseudo_labels --wer-threshold 0.05

  # Use more threads for faster processing
  python 04_reprocess_pseudo_labels.py --input-dir ./pseudo_labels --threads 16

  # Skip backup creation
  python 04_reprocess_pseudo_labels.py --input-dir ./pseudo_labels --no-backup
        """
    )

    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='Directory containing pseudo-label JSONL files'
    )

    parser.add_argument(
        '--wer-threshold', '-w',
        type=float,
        default=0.10,
        help='WER threshold for acceptance (default: 0.10 = 10%%)'
    )

    parser.add_argument(
        '--threads', '-t',
        type=int,
        default=8,
        help='Number of threads for parallel processing (default: 8)'
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Preview changes without writing files'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup files'
    )

    args = parser.parse_args()

    try:
        reprocessor = PseudoLabelReprocessor(
            input_dir=Path(args.input_dir),
            wer_threshold=args.wer_threshold,
            num_threads=args.threads,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
        reprocessor.run()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
