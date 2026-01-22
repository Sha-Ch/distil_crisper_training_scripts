#!/usr/bin/env python3
"""
=============================================================================
Pseudo-Label Generation Script for Distil-CrisperWhisper
=============================================================================
Uses CrisperWhisper teacher model to generate high-quality pseudo-labels
with WER-based filtering (following distil-whisper v3.5 methodology).

Key Features (matching official distil-whisper):
1. Generate pseudo-labels using teacher model
2. Compare pseudo-labels to ground truth transcriptions
3. Filter out samples with WER > 10% (removes hallucinations)
4. Save high-quality pseudo-labels for training

Usage: python3 03_generate_pseudo_labels.py [--config config.yaml] [--resume]
=============================================================================
"""

import os
import sys
import json
import yaml
import argparse
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
import soundfile as sf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn

# For WER calculation
try:
    from jiwer import wer as calculate_wer
    from jiwer import transforms as tr
except ImportError:
    print("Installing jiwer for WER calculation...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jiwer"])
    from jiwer import wer as calculate_wer
    from jiwer import transforms as tr

console = Console()


# =============================================================================
# Text Normalization (following distil-whisper methodology)
# =============================================================================

class WhisperTextNormalizer:
    """
    Normalizes text for WER comparison.
    Following the same normalization as distil-whisper.
    """

    def __init__(self):
        # Define transformation pipeline
        self.transform = tr.Compose([
            tr.ToLowerCase(),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.RemovePunctuation(),
            tr.ReduceToListOfListOfWords(),
        ])

    def __call__(self, text: str) -> str:
        """Normalize text for WER comparison."""
        if not text:
            return ""

        # Basic normalization
        text = text.lower()

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text


def are_spelling_variants(word1: str, word2: str, threshold: float = 0.85) -> bool:
    """
    Check if two words are spelling variants (e.g., British vs American).

    Uses difflib.SequenceMatcher to check similarity ratio.
    Examples: colour/color (0.91), realise/realize (0.86), behaviour/behavior (0.94)

    Args:
        word1: First word
        word2: Second word
        threshold: Similarity threshold (default 0.85 catches most UK/US variants)

    Returns:
        True if words are similar enough to be considered equivalent
    """
    from difflib import SequenceMatcher

    if word1 == word2:
        return True
    # Must start with same letter to be a spelling variant
    if not word1 or not word2 or word1[0] != word2[0]:
        return False
    # Check similarity ratio
    return SequenceMatcher(None, word1, word2).ratio() >= threshold


def calculate_wer_spelling_tolerant(reference: str, hypothesis: str) -> float:
    """
    Calculate WER with tolerance for British/American spelling variants.

    Uses dynamic programming (Levenshtein distance) but treats spelling variants
    as matches rather than substitutions.

    Args:
        reference: Reference text (normalized)
        hypothesis: Hypothesis text (normalized)

    Returns:
        WER as float between 0.0 and 1.0 (or higher)
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    n = len(ref_words)
    m = len(hyp_words)

    # Edge cases
    if n == 0:
        return 1.0 if m > 0 else 0.0
    if m == 0:
        return 1.0

    # DP table for edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize base cases
    for i in range(n + 1):
        dp[i][0] = i  # Deletions
    for j in range(m + 1):
        dp[0][j] = j  # Insertions

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Check if words match (exact or spelling variant)
            if are_spelling_variants(ref_words[i-1], hyp_words[j-1]):
                dp[i][j] = dp[i-1][j-1]  # No cost - words are equivalent
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # Deletion
                    dp[i][j-1] + 1,      # Insertion
                    dp[i-1][j-1] + 1     # Substitution
                )

    # WER = edit_distance / reference_length
    return dp[n][m] / n


@dataclass
class PseudoLabel:
    """Represents pseudo-labels for a single audio sample."""
    sample_id: str
    audio_path: str
    pseudo_transcription: str  # Teacher's transcription
    ground_truth: str          # Original ground truth
    wer: float                 # WER between pseudo and ground truth
    duration: float
    source_dataset: str
    generated_at: str
    accepted: bool             # Whether WER <= threshold


class SpotInstanceHandler:
    """Handles spot instance preemption signals."""

    def __init__(self):
        self.should_stop = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        console.print(f"\n[yellow]Received signal {signum}. Saving progress...[/yellow]")
        self.should_stop = True


class PseudoLabelGenerator:
    """
    Generates pseudo-labels using CrisperWhisper teacher model.
    Implements WER-based filtering following distil-whisper v3.5.
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.spot_handler = SpotInstanceHandler()

        # Paths
        self.data_dir = Path(self.config['paths']['data_dir'])
        self.pseudo_labels_dir = self.data_dir / 'pseudo_labels'
        self.pseudo_labels_dir.mkdir(exist_ok=True)

        # WER threshold (distil-whisper uses 10%)
        self.wer_threshold = self.config.get('distillation', {}).get('pseudo_labels', {}).get('wer_threshold', 0.10)

        # Progress tracking
        self.progress_file = self.pseudo_labels_dir / 'progress.json'

        # Text normalizer
        self.normalizer = WhisperTextNormalizer()

        # Model (loaded lazily)
        self.model = None
        self.processor = None

        # Batch processing settings
        batch_config = self.config.get('batch_processing', {})
        self.batch_size = self.config['teacher'].get('pseudo_label_batch_size', 48)
        self.audio_loader_workers = batch_config.get('audio_loader_workers', 8)

        # Statistics
        self.stats = {
            'total_processed': 0,
            'accepted': 0,
            'rejected': 0,
            'avg_wer_accepted': 0.0,
            'avg_wer_rejected': 0.0
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_model(self):
        """Load CrisperWhisper teacher model."""
        if self.model is not None:
            return

        console.print("[bold blue]Loading CrisperWhisper teacher model...[/bold blue]")

        model_id = self.config['teacher']['model_id']
        cache_dir = self.config['paths'].get('hf_cache', None)
        dtype = getattr(torch, self.config['teacher'].get('dtype', 'float16'))

        self.processor = WhisperProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )

        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            device_map="auto"
        )

        self.model.eval()
        console.print(f"[green]✓ Model loaded on {self.device}[/green]")
        console.print(f"[green]✓ WER threshold: {self.wer_threshold * 100:.0f}%[/green]")
        console.print(f"[green]✓ Batch size: {self.batch_size} (GPU batched inference)[/green]")
        console.print(f"[green]✓ Audio loader workers: {self.audio_loader_workers}[/green]")

    def _load_progress(self) -> Dict:
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'processed': [], 'last_dataset': None, 'last_index': 0}

    def _save_progress(self, progress: Dict):
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)

    def generate_all(self, resume: bool = False) -> Dict[str, Dict]:
        """Generate pseudo-labels for all datasets with WER filtering."""

        self._load_model()

        progress = self._load_progress() if resume else {'processed': [], 'last_dataset': None, 'last_index': 0}
        processed_set = set(progress['processed'])

        stats = {}

        for dataset_name in ['gigaspeech', 'voxpopuli', 'librispeech']:
            manifest_path = self.data_dir / dataset_name / 'manifest.jsonl'

            if not manifest_path.exists():
                console.print(f"[yellow]Skipping {dataset_name}: manifest not found[/yellow]")
                continue

            console.print(f"\n[bold blue]Generating pseudo-labels for {dataset_name}...[/bold blue]")

            dataset_stats = self._process_dataset(
                dataset_name=dataset_name,
                manifest_path=manifest_path,
                processed_set=processed_set,
                progress=progress
            )

            stats[dataset_name] = dataset_stats

            if self.spot_handler.should_stop:
                break

        return stats

    def _process_dataset(
        self,
        dataset_name: str,
        manifest_path: Path,
        processed_set: set,
        progress: Dict
    ) -> Dict:
        """Process a single dataset with WER-based filtering."""

        output_path = self.pseudo_labels_dir / f'{dataset_name}_labels.jsonl'
        rejected_path = self.pseudo_labels_dir / f'{dataset_name}_rejected.jsonl'

        # Count total samples
        with open(manifest_path, 'r') as f:
            total_samples = sum(1 for _ in f)

        mode = 'a' if processed_set else 'w'

        accepted_count = 0
        rejected_count = 0
        wer_sum_accepted = 0.0
        wer_sum_rejected = 0.0

        batch_size = self.batch_size

        with open(manifest_path, 'r') as manifest_file, \
             open(output_path, mode) as output_file, \
             open(rejected_path, mode) as rejected_file:

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console
            ) as pbar:

                task = pbar.add_task(f"Processing {dataset_name}", total=total_samples)

                batch = []

                for line in manifest_file:
                    if self.spot_handler.should_stop:
                        self._save_progress(progress)
                        break

                    entry = json.loads(line)
                    sample_id = entry['sample_id']

                    if sample_id in processed_set:
                        pbar.update(task, advance=1)
                        continue

                    batch.append(entry)

                    if len(batch) >= batch_size:
                        labels = self._process_batch(batch)

                        for label in labels:
                            if label.accepted:
                                output_file.write(json.dumps(asdict(label)) + '\n')
                                accepted_count += 1
                                wer_sum_accepted += label.wer
                            else:
                                rejected_file.write(json.dumps(asdict(label)) + '\n')
                                rejected_count += 1
                                wer_sum_rejected += label.wer

                            progress['processed'].append(label.sample_id)
                            processed_set.add(label.sample_id)

                        batch = []
                        output_file.flush()
                        rejected_file.flush()

                        # Save progress periodically
                        if (accepted_count + rejected_count) % 500 == 0:
                            self._save_progress(progress)
                            acceptance_rate = accepted_count / (accepted_count + rejected_count) * 100
                            pbar.update(
                                task,
                                description=f"Processing {dataset_name} (acceptance: {acceptance_rate:.1f}%)"
                            )

                    pbar.update(task, advance=1)

                # Process remaining batch
                if batch and not self.spot_handler.should_stop:
                    labels = self._process_batch(batch)
                    for label in labels:
                        if label.accepted:
                            output_file.write(json.dumps(asdict(label)) + '\n')
                            accepted_count += 1
                            wer_sum_accepted += label.wer
                        else:
                            rejected_file.write(json.dumps(asdict(label)) + '\n')
                            rejected_count += 1
                            wer_sum_rejected += label.wer
                        progress['processed'].append(label.sample_id)

        self._save_progress(progress)

        # Calculate statistics
        total = accepted_count + rejected_count
        acceptance_rate = (accepted_count / total * 100) if total > 0 else 0
        avg_wer_accepted = (wer_sum_accepted / accepted_count) if accepted_count > 0 else 0
        avg_wer_rejected = (wer_sum_rejected / rejected_count) if rejected_count > 0 else 0

        console.print(f"\n[bold]Results for {dataset_name}:[/bold]")
        console.print(f"  Total processed: {total}")
        console.print(f"  Accepted (WER ≤ {self.wer_threshold*100:.0f}%): [green]{accepted_count}[/green] ({acceptance_rate:.1f}%)")
        console.print(f"  Rejected (WER > {self.wer_threshold*100:.0f}%): [red]{rejected_count}[/red]")
        console.print(f"  Avg WER (accepted): {avg_wer_accepted*100:.1f}%")
        console.print(f"  Avg WER (rejected): {avg_wer_rejected*100:.1f}%")

        return {
            'total': total,
            'accepted': accepted_count,
            'rejected': rejected_count,
            'acceptance_rate': acceptance_rate,
            'avg_wer_accepted': avg_wer_accepted,
            'avg_wer_rejected': avg_wer_rejected
        }

    def _load_single_audio(self, entry: Dict) -> Tuple[Optional[np.ndarray], Dict, Optional[str]]:
        """Load a single audio file. Used by parallel loader."""
        try:
            audio, sr = sf.read(entry['audio_path'])

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Resample to 16kHz if needed
            if sr != 16000:
                audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio_tensor).squeeze(0).numpy()

            # Ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            return audio, entry, None
        except Exception as e:
            return None, entry, str(e)

    def _load_audio_parallel(self, entries: List[Dict]) -> Tuple[List[np.ndarray], List[Dict], List[str]]:
        """
        Load audio files in parallel using thread pool.
        Returns: (audios, valid_entries, errors)
        """
        audios = []
        valid_entries = []
        errors = []

        with ThreadPoolExecutor(max_workers=self.audio_loader_workers) as executor:
            futures = {executor.submit(self._load_single_audio, entry): entry for entry in entries}

            for future in as_completed(futures):
                audio, entry, error = future.result()
                if audio is not None:
                    audios.append(audio)
                    valid_entries.append(entry)
                else:
                    errors.append(f"{entry['sample_id']}: {error}")

        return audios, valid_entries, errors

    def _process_batch(self, batch: List[Dict]) -> List[PseudoLabel]:
        """
        Process a batch of audio samples with TRUE GPU batching.

        This method:
        1. Loads audio files in parallel (I/O bound - uses threads)
        2. Batches feature extraction via processor
        3. Runs batched model inference (single GPU call for entire batch!)
        4. Calculates WER for each sample

        This is ~10-50x faster than processing samples one-by-one.
        """
        # 1. Load audio files in parallel
        audios, valid_entries, errors = self._load_audio_parallel(batch)

        if errors:
            for err in errors[:3]:  # Only show first 3 errors
                console.print(f"[dim red]Audio load error: {err}[/dim red]")

        if not audios:
            return []

        # 2. Batch feature extraction - processor handles padding automatically
        try:
            input_features = self.processor(
                audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True  # Pad shorter audios to match longest in batch
            ).input_features.to(self.device)
        except Exception as e:
            console.print(f"[dim red]Feature extraction error: {e}[/dim red]")
            return []

        # 3. Batched model inference - SINGLE GPU call for entire batch!
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    input_features,
                    language="en",
                    task="transcribe",
                    return_timestamps=False
                )

                # 4. Batch decode all transcriptions at once
                transcriptions = self.processor.batch_decode(
                    generated,
                    skip_special_tokens=True
                )
        except Exception as e:
            console.print(f"[dim red]Model inference error: {e}[/dim red]")
            return []

        # 5. Create PseudoLabel objects with WER calculation
        labels = []
        timestamp = datetime.now().isoformat()

        for entry, transcription in zip(valid_entries, transcriptions):
            ground_truth = entry.get('text', '')

            # Calculate WER with British/American spelling tolerance
            normalized_pseudo = self.normalizer(transcription)
            normalized_ground_truth = self.normalizer(ground_truth)

            if not normalized_ground_truth:
                wer_score = 0.0
            elif not normalized_pseudo:
                wer_score = 1.0
            else:
                try:
                    wer_score = calculate_wer_spelling_tolerant(normalized_ground_truth, normalized_pseudo)
                except Exception:
                    wer_score = 1.0

            accepted = wer_score <= self.wer_threshold

            labels.append(PseudoLabel(
                sample_id=entry['sample_id'],
                audio_path=entry['audio_path'],
                pseudo_transcription=transcription,
                ground_truth=ground_truth,
                wer=wer_score,
                duration=entry['duration'],
                source_dataset=entry['source'],
                generated_at=timestamp,
                accepted=accepted
            ))

        return labels


def merge_pseudo_labels(pseudo_labels_dir: Path) -> Tuple[Path, Dict]:
    """Merge all accepted pseudo-label files into a single training file."""

    merged_path = pseudo_labels_dir / 'all_labels.jsonl'

    console.print("\n[bold blue]Merging accepted pseudo-label files...[/bold blue]")

    total_count = 0
    total_duration = 0.0

    with open(merged_path, 'w') as merged_file:
        for label_file in pseudo_labels_dir.glob('*_labels.jsonl'):
            if label_file.name in ['all_labels.jsonl']:
                continue
            if '_rejected' in label_file.name:
                continue

            console.print(f"  Adding {label_file.name}...")

            with open(label_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    # Only include accepted samples
                    if entry.get('accepted', True):
                        merged_file.write(line)
                        total_count += 1
                        total_duration += entry.get('duration', 0)

    hours = total_duration / 3600
    console.print(f"[green]✓ Merged {total_count} samples ({hours:.1f} hours) into {merged_path}[/green]")

    return merged_path, {'count': total_count, 'hours': hours}


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo-labels with WER filtering')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--merge-only', action='store_true', help='Only merge existing label files')
    parser.add_argument('--wer-threshold', type=float, help='Override WER threshold (default: 0.10)')
    args = parser.parse_args()

    # Find config file
    config_path = Path(args.config)
    if not config_path.exists():
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / 'config.yaml'

    if not config_path.exists():
        console.print(f"[red]Config file not found: {args.config}[/red]")
        sys.exit(1)

    console.print(f"[bold]Using config: {config_path}[/bold]")

    # Load config to get paths
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    pseudo_labels_dir = Path(config['paths']['data_dir']) / 'pseudo_labels'

    if args.merge_only:
        merge_pseudo_labels(pseudo_labels_dir)
        return

    # Override WER threshold if specified
    if args.wer_threshold:
        console.print(f"[yellow]Overriding WER threshold to {args.wer_threshold * 100:.0f}%[/yellow]")

    # Generate pseudo-labels
    generator = PseudoLabelGenerator(str(config_path))
    if args.wer_threshold:
        generator.wer_threshold = args.wer_threshold

    stats = generator.generate_all(resume=args.resume)

    console.print("\n[bold green]Pseudo-Label Generation Complete![/bold green]")
    console.print("\n[bold]Summary:[/bold]")
    for name, dataset_stats in stats.items():
        console.print(f"  {name}:")
        console.print(f"    Accepted: {dataset_stats['accepted']} ({dataset_stats['acceptance_rate']:.1f}%)")
        console.print(f"    Avg WER: {dataset_stats['avg_wer_accepted']*100:.1f}%")

    # Merge all accepted files
    merge_pseudo_labels(pseudo_labels_dir)


if __name__ == '__main__':
    main()
