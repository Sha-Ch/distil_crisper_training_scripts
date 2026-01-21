#!/usr/bin/env python3
"""
=============================================================================
Dataset Preparation Script for Distil-CrisperWhisper Training
=============================================================================
Prepares GigaSpeech, VoxPopuli, and LibriSpeech datasets for distillation.
Uses streaming to avoid downloading entire datasets at once.

Usage: python3 02_prepare_data.py [--config config.yaml]
=============================================================================
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Iterator, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Audio, IterableDataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


@dataclass
class AudioSample:
    """Represents a single audio sample."""
    audio_path: str
    audio_array: np.ndarray
    sampling_rate: int
    text: str
    duration: float
    dataset_source: str
    sample_id: str


class DatasetPreparer:
    """Handles dataset loading and preparation for distillation training."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config['paths']['data_dir'])
        self.target_sr = 16000  # Whisper expects 16kHz

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def prepare_all_datasets(self) -> Dict[str, int]:
        """Prepare all enabled datasets and return sample counts."""
        stats = {}

        datasets_config = self.config['datasets']

        if datasets_config.get('gigaspeech', {}).get('enabled', False):
            console.print("\n[bold blue]Preparing GigaSpeech dataset...[/bold blue]")
            stats['gigaspeech'] = self.prepare_gigaspeech()

        if datasets_config.get('voxpopuli', {}).get('enabled', False):
            console.print("\n[bold blue]Preparing VoxPopuli dataset...[/bold blue]")
            stats['voxpopuli'] = self.prepare_voxpopuli()

        if datasets_config.get('librispeech', {}).get('enabled', False):
            console.print("\n[bold blue]Preparing LibriSpeech dataset (validation)...[/bold blue]")
            stats['librispeech'] = self.prepare_librispeech()

        return stats

    def prepare_gigaspeech(self) -> int:
        """
        Prepare GigaSpeech dataset.
        GigaSpeech is large, so we use streaming and save processed shards.
        """
        cfg = self.config['datasets']['gigaspeech']
        output_dir = self.data_dir / 'gigaspeech'
        output_dir.mkdir(exist_ok=True)

        manifest_path = output_dir / 'manifest.jsonl'

        # Check if already prepared
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                count = sum(1 for _ in f)
            console.print(f"[green]✓ GigaSpeech already prepared: {count} samples[/green]")
            return count

        console.print(f"[yellow]Loading GigaSpeech ({cfg['subset']} subset) with streaming...[/yellow]")

        try:
            # Load dataset with streaming
            dataset = load_dataset(
                cfg['name'],
                cfg['subset'],
                split=cfg['split'],
                streaming=cfg['streaming'],
                trust_remote_code=True
            )

            # Process and save samples
            count = self._process_streaming_dataset(
                dataset=dataset,
                output_dir=output_dir,
                manifest_path=manifest_path,
                source_name='gigaspeech',
                max_samples=None  # Process all
            )

            console.print(f"[green]✓ GigaSpeech prepared: {count} samples[/green]")
            return count

        except Exception as e:
            console.print(f"[red]Error preparing GigaSpeech: {e}[/red]")
            console.print("[yellow]GigaSpeech requires authentication. See: https://huggingface.co/datasets/speechcolab/gigaspeech[/yellow]")
            return 0

    def prepare_voxpopuli(self) -> int:
        """
        Prepare VoxPopuli dataset.
        """
        cfg = self.config['datasets']['voxpopuli']
        output_dir = self.data_dir / 'voxpopuli'
        output_dir.mkdir(exist_ok=True)

        manifest_path = output_dir / 'manifest.jsonl'

        # Check if already prepared
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                count = sum(1 for _ in f)
            console.print(f"[green]✓ VoxPopuli already prepared: {count} samples[/green]")
            return count

        console.print(f"[yellow]Loading VoxPopuli ({cfg['subset']}) with streaming...[/yellow]")

        try:
            dataset = load_dataset(
                cfg['name'],
                cfg['subset'],
                split=cfg['split'],
                streaming=cfg['streaming'],
                trust_remote_code=True
            )

            count = self._process_streaming_dataset(
                dataset=dataset,
                output_dir=output_dir,
                manifest_path=manifest_path,
                source_name='voxpopuli',
                max_samples=None
            )

            console.print(f"[green]✓ VoxPopuli prepared: {count} samples[/green]")
            return count

        except Exception as e:
            console.print(f"[red]Error preparing VoxPopuli: {e}[/red]")
            return 0

    def prepare_librispeech(self) -> int:
        """
        Prepare LibriSpeech dataset for validation.
        This is small enough to download fully.
        """
        cfg = self.config['datasets']['librispeech']
        output_dir = self.data_dir / 'librispeech'
        output_dir.mkdir(exist_ok=True)

        manifest_path = output_dir / 'manifest.jsonl'

        # Check if already prepared
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                count = sum(1 for _ in f)
            console.print(f"[green]✓ LibriSpeech already prepared: {count} samples[/green]")
            return count

        console.print(f"[yellow]Loading LibriSpeech ({cfg['subset']})...[/yellow]")

        try:
            dataset = load_dataset(
                cfg['name'],
                cfg['subset'],
                split=cfg['split'],
                trust_remote_code=True
            )

            # Cast audio column to proper format
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.target_sr))

            count = self._process_dataset(
                dataset=dataset,
                output_dir=output_dir,
                manifest_path=manifest_path,
                source_name='librispeech'
            )

            console.print(f"[green]✓ LibriSpeech prepared: {count} samples[/green]")
            return count

        except Exception as e:
            console.print(f"[red]Error preparing LibriSpeech: {e}[/red]")
            return 0

    def _process_streaming_dataset(
        self,
        dataset: IterableDataset,
        output_dir: Path,
        manifest_path: Path,
        source_name: str,
        max_samples: Optional[int] = None
    ) -> int:
        """Process a streaming dataset and save to disk."""

        count = 0
        shard_size = 10000  # Samples per shard
        current_shard = []
        shard_idx = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            task = progress.add_task(f"Processing {source_name}...", total=max_samples or 100000)

            with open(manifest_path, 'w') as manifest_file:
                for sample in dataset:
                    try:
                        # Extract audio and text based on dataset format
                        audio_data = self._extract_audio(sample, source_name)
                        text = self._extract_text(sample, source_name)

                        if audio_data is None or text is None:
                            continue

                        audio_array, sr = audio_data

                        # Resample if necessary
                        if sr != self.target_sr:
                            audio_array = self._resample(audio_array, sr, self.target_sr)

                        # Calculate duration
                        duration = len(audio_array) / self.target_sr

                        # Filter by duration (skip very short or very long samples)
                        if duration < 1.0 or duration > 30.0:
                            continue

                        # Generate sample ID
                        sample_id = f"{source_name}_{count:08d}"

                        # Save audio file
                        audio_path = output_dir / f"{sample_id}.wav"
                        self._save_audio(audio_array, audio_path)

                        # Write to manifest
                        manifest_entry = {
                            'audio_path': str(audio_path),
                            'text': text,
                            'duration': duration,
                            'source': source_name,
                            'sample_id': sample_id
                        }
                        manifest_file.write(json.dumps(manifest_entry) + '\n')

                        count += 1
                        progress.update(task, advance=1)

                        if max_samples and count >= max_samples:
                            break

                    except Exception as e:
                        # Skip problematic samples
                        continue

        return count

    def _process_dataset(
        self,
        dataset,
        output_dir: Path,
        manifest_path: Path,
        source_name: str
    ) -> int:
        """Process a regular (non-streaming) dataset."""

        count = 0

        with open(manifest_path, 'w') as manifest_file:
            for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {source_name}")):
                try:
                    # Extract audio
                    if 'audio' in sample:
                        audio_array = np.array(sample['audio']['array'], dtype=np.float32)
                        sr = sample['audio']['sampling_rate']
                    else:
                        continue

                    # Extract text
                    text = sample.get('text', sample.get('sentence', sample.get('transcription', '')))
                    if not text:
                        continue

                    # Resample if necessary
                    if sr != self.target_sr:
                        audio_array = self._resample(audio_array, sr, self.target_sr)

                    # Calculate duration
                    duration = len(audio_array) / self.target_sr

                    # Filter by duration
                    if duration < 1.0 or duration > 30.0:
                        continue

                    # Generate sample ID
                    sample_id = f"{source_name}_{count:08d}"

                    # Save audio file
                    audio_path = output_dir / f"{sample_id}.wav"
                    self._save_audio(audio_array, audio_path)

                    # Write to manifest
                    manifest_entry = {
                        'audio_path': str(audio_path),
                        'text': text,
                        'duration': duration,
                        'source': source_name,
                        'sample_id': sample_id
                    }
                    manifest_file.write(json.dumps(manifest_entry) + '\n')

                    count += 1

                except Exception as e:
                    continue

        return count

    def _extract_audio(self, sample: Dict, source: str) -> Optional[tuple]:
        """Extract audio array and sample rate from a sample."""
        try:
            if 'audio' in sample:
                audio_info = sample['audio']
                if isinstance(audio_info, dict):
                    return np.array(audio_info['array'], dtype=np.float32), audio_info['sampling_rate']
                elif hasattr(audio_info, 'array'):
                    return np.array(audio_info.array, dtype=np.float32), audio_info.sampling_rate
            return None
        except:
            return None

    def _extract_text(self, sample: Dict, source: str) -> Optional[str]:
        """Extract transcription text from a sample."""
        # Try common field names
        for field in ['text', 'sentence', 'transcription', 'normalized_text']:
            if field in sample and sample[field]:
                return sample[field].strip()
        return None

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio

        # Use torchaudio for high-quality resampling
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        resampled = resampler(audio_tensor)
        return resampled.squeeze(0).numpy()

    def _save_audio(self, audio: np.ndarray, path: Path):
        """Save audio array to WAV file."""
        import soundfile as sf
        sf.write(str(path), audio, self.target_sr)


class DatasetValidator:
    """Validates prepared datasets."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def validate_all(self) -> Dict[str, Dict]:
        """Validate all prepared datasets."""
        results = {}

        for dataset_name in ['gigaspeech', 'voxpopuli', 'librispeech']:
            manifest_path = self.data_dir / dataset_name / 'manifest.jsonl'
            if manifest_path.exists():
                results[dataset_name] = self.validate_dataset(manifest_path)

        return results

    def validate_dataset(self, manifest_path: Path) -> Dict:
        """Validate a single dataset."""
        total_samples = 0
        valid_samples = 0
        total_duration = 0.0
        missing_files = 0

        with open(manifest_path, 'r') as f:
            for line in f:
                total_samples += 1
                entry = json.loads(line)

                if Path(entry['audio_path']).exists():
                    valid_samples += 1
                    total_duration += entry['duration']
                else:
                    missing_files += 1

        return {
            'total_samples': total_samples,
            'valid_samples': valid_samples,
            'missing_files': missing_files,
            'total_duration_hours': total_duration / 3600,
            'valid': missing_files == 0
        }


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for distillation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing datasets')
    args = parser.parse_args()

    # Find config file
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / 'config.yaml'

    if not config_path.exists():
        console.print(f"[red]Config file not found: {args.config}[/red]")
        sys.exit(1)

    console.print(f"[bold]Using config: {config_path}[/bold]")

    if args.validate_only:
        # Load config to get data_dir
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        validator = DatasetValidator(Path(config['paths']['data_dir']))
        results = validator.validate_all()

        console.print("\n[bold]Dataset Validation Results:[/bold]")
        for name, stats in results.items():
            status = "[green]✓[/green]" if stats['valid'] else "[red]✗[/red]"
            console.print(f"  {status} {name}:")
            console.print(f"      Samples: {stats['valid_samples']}/{stats['total_samples']}")
            console.print(f"      Duration: {stats['total_duration_hours']:.1f} hours")
            if stats['missing_files'] > 0:
                console.print(f"      [red]Missing files: {stats['missing_files']}[/red]")
    else:
        # Prepare datasets
        preparer = DatasetPreparer(str(config_path))
        stats = preparer.prepare_all_datasets()

        console.print("\n[bold green]Dataset Preparation Complete![/bold green]")
        console.print("\nSummary:")
        for name, count in stats.items():
            console.print(f"  {name}: {count} samples")


if __name__ == '__main__':
    main()
