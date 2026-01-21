#!/usr/bin/env python3
"""
=============================================================================
Streaming Dataset Preparation for Distil-CrisperWhisper
=============================================================================
Prepares all 8 datasets from official distil-whisper v3.5 methodology using
STREAMING to avoid storage limitations.

Datasets (total ~196,000 hours raw):
1. LibriSpeech (960 hrs)
2. Common Voice EN (3,000 hrs)
3. VoxPopuli EN (1,800 hrs)
4. AMI (100 hrs)
5. People's Speech (30,000 hrs)
6. TED-LIUM (450 hrs)
7. GigaSpeech (10,000 hrs)
8. YODAS (150,000 hrs)

Key Features:
- Full streaming mode - no large downloads required
- On-the-fly pseudo-label generation
- WER-based filtering (discard >10% WER)
- Automatic resume support
- Multi-GPU parallel processing

Usage: python3 02_prepare_data_streaming.py --config ../config.yaml
=============================================================================
"""

import os
import sys
import json
import yaml
import argparse
import signal
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import re

import torch
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
from rich.table import Table

console = Console()


# =============================================================================
# Dataset Configurations
# =============================================================================

DATASET_CONFIGS = {
    'librispeech': {
        'hf_name': 'librispeech_asr',
        'subset': None,  # Will be handled specially
        'splits': ['train.clean.100', 'train.clean.360', 'train.other.500'],
        'text_column': 'text',
        'audio_column': 'audio',
        'hours': 960,
        'requires_auth': False,
        'streaming': True,
    },
    'common_voice': {
        'hf_name': 'mozilla-foundation/common_voice_17_0',
        'subset': 'en',
        'splits': ['train'],
        'text_column': 'sentence',
        'audio_column': 'audio',
        'hours': 3000,
        'requires_auth': True,
        'streaming': True,
    },
    'voxpopuli': {
        'hf_name': 'facebook/voxpopuli',
        'subset': 'en',
        'splits': ['train'],
        'text_column': 'normalized_text',
        'audio_column': 'audio',
        'hours': 1800,
        'requires_auth': False,
        'streaming': True,
    },
    'ami': {
        'hf_name': 'edinburghcstr/ami',
        'subset': 'ihm',
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'hours': 100,
        'requires_auth': False,
        'streaming': True,
    },
    'peoples_speech': {
        'hf_name': 'MLCommons/peoples_speech',
        'subset': 'clean',
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'hours': 30000,
        'requires_auth': True,
        'streaming': True,
    },
    'tedlium': {
        'hf_name': 'LIUM/tedlium',
        'subset': 'release3',
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'hours': 450,
        'requires_auth': False,
        'streaming': True,
    },
    'gigaspeech': {
        'hf_name': 'speechcolab/gigaspeech',
        'subset': 'l',  # Use 'l' (2500 hrs) or 'xl' (10000 hrs)
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'hours': 10000,
        'requires_auth': True,
        'streaming': True,
    },
    'yodas': {
        'hf_name': 'espnet/yodas',
        'subset': 'en000',
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'hours': 150000,
        'requires_auth': False,
        'streaming': True,
    },
}


@dataclass
class DatasetStats:
    """Statistics for a processed dataset."""
    name: str
    samples_processed: int
    samples_accepted: int
    samples_rejected: int
    total_hours: float
    avg_wer: float
    status: str  # 'pending', 'processing', 'completed', 'error'


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


class StreamingDataProcessor:
    """
    Processes datasets in streaming mode with pseudo-label generation.
    Follows official distil-whisper methodology.
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.spot_handler = SpotInstanceHandler()

        # Paths
        self.workspace = Path(self.config['paths']['workspace'])
        self.pseudo_labels_dir = Path(self.config['paths'].get('pseudo_labels_dir',
                                                                self.workspace / 'pseudo_labels'))
        self.pseudo_labels_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.progress_file = self.pseudo_labels_dir / 'progress.json'

        # WER threshold
        distil_config = self.config.get('distillation', {})
        pseudo_config = distil_config.get('pseudo_labels', {})
        self.wer_threshold = pseudo_config.get('wer_threshold', 0.10)

        # Model (loaded lazily)
        self.model = None
        self.processor = None

        console.print(f"[bold]Streaming Data Processor initialized[/bold]")
        console.print(f"  Pseudo-labels dir: {self.pseudo_labels_dir}")
        console.print(f"  WER threshold: {self.wer_threshold * 100:.0f}%")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_model(self):
        """Load CrisperWhisper teacher model for pseudo-label generation."""
        if self.model is not None:
            return

        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        console.print("\n[bold blue]Loading CrisperWhisper teacher model...[/bold blue]")

        teacher_config = self.config['teacher']
        model_id = teacher_config['model_id']
        cache_dir = self.config['paths'].get('hf_cache', None)
        dtype = getattr(torch, teacher_config.get('dtype', 'float16'))
        attn_impl = teacher_config.get('attn_implementation', 'sdpa')

        self.processor = WhisperProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )

        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            device_map="auto"
        )

        self.model.eval()
        console.print(f"[green]Teacher model loaded on {self.device}[/green]")

    def _load_progress(self) -> Dict:
        """Load processing progress for resume support."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'datasets': {},
            'total_samples': 0,
            'total_hours': 0.0,
            'last_update': None
        }

    def _save_progress(self, progress: Dict):
        """Save processing progress."""
        progress['last_update'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate between reference and hypothesis."""
        from jiwer import wer as calculate_wer

        # Normalize texts
        ref = self._normalize_text(reference)
        hyp = self._normalize_text(hypothesis)

        if not ref:
            return 0.0 if not hyp else 1.0
        if not hyp:
            return 1.0

        try:
            return calculate_wer(ref, hyp)
        except Exception:
            return 1.0

    def _normalize_text(self, text: str) -> str:
        """Normalize text for WER comparison."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def _generate_pseudo_label(self, audio_array: np.ndarray, sampling_rate: int) -> str:
        """Generate pseudo-label using teacher model."""
        # Resample if necessary
        if sampling_rate != 16000:
            import torchaudio
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float()
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            audio_array = resampler(audio_tensor).squeeze(0).numpy()

        # Process audio
        input_features = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device)

        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                input_features,
                language="en",
                task="transcribe",
                return_timestamps=False,
                max_new_tokens=256
            )

        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]

    def _load_dataset_streaming(self, name: str) -> Optional[Iterator]:
        """Load a dataset in streaming mode."""
        from datasets import load_dataset

        config = DATASET_CONFIGS.get(name)
        if not config:
            console.print(f"[red]Unknown dataset: {name}[/red]")
            return None

        try:
            console.print(f"[yellow]Loading {name} in streaming mode...[/yellow]")

            # Handle LibriSpeech specially (multiple splits)
            if name == 'librispeech':
                from datasets import interleave_datasets
                datasets_list = []
                for split in config['splits']:
                    ds = load_dataset(
                        config['hf_name'],
                        split=split,
                        streaming=True,
                        trust_remote_code=True
                    )
                    datasets_list.append(ds)

                if len(datasets_list) > 1:
                    dataset = interleave_datasets(datasets_list)
                else:
                    dataset = datasets_list[0]
            else:
                kwargs = {
                    'split': config['splits'][0],
                    'streaming': True,
                    'trust_remote_code': True
                }
                if config['subset']:
                    kwargs['name'] = config['subset']

                dataset = load_dataset(config['hf_name'], **kwargs)

            console.print(f"[green]Dataset {name} loaded successfully[/green]")
            return iter(dataset)

        except Exception as e:
            console.print(f"[red]Error loading {name}: {e}[/red]")
            if config['requires_auth']:
                console.print(f"[yellow]This dataset requires authentication/agreement on HuggingFace[/yellow]")
            return None

    def process_dataset(
        self,
        name: str,
        max_samples: Optional[int] = None,
        resume: bool = True
    ) -> DatasetStats:
        """Process a single dataset with pseudo-label generation and WER filtering."""

        self._load_model()

        config = DATASET_CONFIGS.get(name)
        if not config:
            return DatasetStats(
                name=name, samples_processed=0, samples_accepted=0,
                samples_rejected=0, total_hours=0.0, avg_wer=0.0, status='error'
            )

        # Output file
        output_file = self.pseudo_labels_dir / f'{name}_pseudo_labels.jsonl'
        rejected_file = self.pseudo_labels_dir / f'{name}_rejected.jsonl'

        # Load progress
        progress = self._load_progress()
        dataset_progress = progress['datasets'].get(name, {
            'samples_processed': 0,
            'samples_accepted': 0,
            'samples_rejected': 0,
            'total_hours': 0.0,
            'wer_sum': 0.0
        })

        if resume and dataset_progress['samples_processed'] > 0:
            console.print(f"[yellow]Resuming {name} from {dataset_progress['samples_processed']} samples[/yellow]")
            start_idx = dataset_progress['samples_processed']
            mode = 'a'
        else:
            start_idx = 0
            mode = 'w'
            dataset_progress = {
                'samples_processed': 0,
                'samples_accepted': 0,
                'samples_rejected': 0,
                'total_hours': 0.0,
                'wer_sum': 0.0
            }

        # Load dataset
        dataset_iter = self._load_dataset_streaming(name)
        if dataset_iter is None:
            return DatasetStats(
                name=name, samples_processed=0, samples_accepted=0,
                samples_rejected=0, total_hours=0.0, avg_wer=0.0, status='error'
            )

        text_col = config['text_column']
        audio_col = config['audio_column']

        # Process samples
        with open(output_file, mode) as out_f, open(rejected_file, mode) as rej_f:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console
            ) as pbar:

                total = max_samples or config['hours'] * 100  # Rough estimate
                task = pbar.add_task(f"Processing {name}", total=total)

                idx = 0
                for sample in dataset_iter:
                    if self.spot_handler.should_stop:
                        console.print("[yellow]Stopping due to signal...[/yellow]")
                        break

                    # Skip to resume point
                    if idx < start_idx:
                        idx += 1
                        pbar.update(task, advance=1)
                        continue

                    if max_samples and idx >= max_samples:
                        break

                    try:
                        # Extract audio and text
                        audio_data = sample.get(audio_col, {})
                        if isinstance(audio_data, dict):
                            audio_array = np.array(audio_data['array'], dtype=np.float32)
                            sr = audio_data['sampling_rate']
                        else:
                            idx += 1
                            continue

                        ground_truth = sample.get(text_col, '')
                        if not ground_truth:
                            idx += 1
                            continue

                        # Calculate duration
                        duration = len(audio_array) / sr

                        # Filter by duration (skip very short or very long)
                        if duration < 1.0 or duration > 30.0:
                            idx += 1
                            continue

                        # Generate pseudo-label
                        pseudo_label = self._generate_pseudo_label(audio_array, sr)

                        # Calculate WER
                        wer = self._calculate_wer(ground_truth, pseudo_label)

                        # Create entry
                        entry = {
                            'sample_id': f'{name}_{idx:08d}',
                            'dataset': name,
                            'ground_truth': ground_truth,
                            'pseudo_label': pseudo_label,
                            'wer': wer,
                            'duration': duration,
                            'accepted': wer <= self.wer_threshold
                        }

                        # Write to appropriate file
                        if entry['accepted']:
                            out_f.write(json.dumps(entry) + '\n')
                            dataset_progress['samples_accepted'] += 1
                        else:
                            rej_f.write(json.dumps(entry) + '\n')
                            dataset_progress['samples_rejected'] += 1

                        dataset_progress['samples_processed'] += 1
                        dataset_progress['total_hours'] += duration / 3600
                        dataset_progress['wer_sum'] += wer

                        # Update progress periodically
                        if dataset_progress['samples_processed'] % 100 == 0:
                            progress['datasets'][name] = dataset_progress
                            self._save_progress(progress)

                            acceptance_rate = (dataset_progress['samples_accepted'] /
                                             dataset_progress['samples_processed'] * 100)
                            pbar.update(
                                task,
                                description=f"Processing {name} (acc: {acceptance_rate:.1f}%)"
                            )

                        idx += 1
                        pbar.update(task, advance=1)

                    except Exception as e:
                        # Log error but continue
                        idx += 1
                        continue

        # Final save
        progress['datasets'][name] = dataset_progress
        self._save_progress(progress)

        # Calculate stats
        avg_wer = (dataset_progress['wer_sum'] / dataset_progress['samples_processed']
                   if dataset_progress['samples_processed'] > 0 else 0.0)

        return DatasetStats(
            name=name,
            samples_processed=dataset_progress['samples_processed'],
            samples_accepted=dataset_progress['samples_accepted'],
            samples_rejected=dataset_progress['samples_rejected'],
            total_hours=dataset_progress['total_hours'],
            avg_wer=avg_wer,
            status='completed'
        )

    def process_all_datasets(
        self,
        max_samples_per_dataset: Optional[int] = None,
        datasets_to_process: Optional[List[str]] = None,
        resume: bool = True
    ) -> Dict[str, DatasetStats]:
        """Process all enabled datasets."""

        # Get list of datasets to process
        if datasets_to_process:
            dataset_names = datasets_to_process
        else:
            dataset_names = list(DATASET_CONFIGS.keys())

        # Check which are enabled in config
        enabled_datasets = []
        datasets_config = self.config.get('datasets', {})

        for name in dataset_names:
            ds_config = datasets_config.get(name, {})
            if ds_config.get('enabled', True):
                enabled_datasets.append(name)

        console.print(f"\n[bold blue]Processing {len(enabled_datasets)} datasets[/bold blue]")

        # Process each dataset
        all_stats = {}
        for name in enabled_datasets:
            if self.spot_handler.should_stop:
                break

            console.print(f"\n[bold cyan]═══ Processing: {name} ═══[/bold cyan]")

            stats = self.process_dataset(
                name=name,
                max_samples=max_samples_per_dataset,
                resume=resume
            )
            all_stats[name] = stats

            # Print summary
            console.print(f"\n  Results for {name}:")
            console.print(f"    Processed: {stats.samples_processed}")
            console.print(f"    Accepted: [green]{stats.samples_accepted}[/green]")
            console.print(f"    Rejected: [red]{stats.samples_rejected}[/red]")
            console.print(f"    Hours: {stats.total_hours:.1f}")
            console.print(f"    Avg WER: {stats.avg_wer * 100:.1f}%")

        return all_stats

    def merge_all_labels(self) -> Tuple[Path, Dict]:
        """Merge all pseudo-label files into a single training file."""

        merged_path = self.pseudo_labels_dir / 'all_pseudo_labels.jsonl'

        console.print("\n[bold blue]Merging all pseudo-label files...[/bold blue]")

        total_count = 0
        total_hours = 0.0

        with open(merged_path, 'w') as merged_file:
            for label_file in sorted(self.pseudo_labels_dir.glob('*_pseudo_labels.jsonl')):
                if label_file.name == 'all_pseudo_labels.jsonl':
                    continue

                console.print(f"  Adding {label_file.name}...")
                count = 0

                with open(label_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        if entry.get('accepted', True):
                            merged_file.write(line)
                            count += 1
                            total_hours += entry.get('duration', 0) / 3600

                total_count += count
                console.print(f"    Added {count} samples")

        console.print(f"\n[green]Merged {total_count} samples ({total_hours:.1f} hours)[/green]")
        console.print(f"Output: {merged_path}")

        return merged_path, {'count': total_count, 'hours': total_hours}


def print_dataset_table():
    """Print table of available datasets."""
    table = Table(title="Available Datasets (Official Distil-Whisper v3.5)")

    table.add_column("Dataset", style="cyan")
    table.add_column("Hours", justify="right")
    table.add_column("Auth Required", justify="center")
    table.add_column("HuggingFace Name", style="dim")

    total_hours = 0
    for name, config in DATASET_CONFIGS.items():
        hours = config['hours']
        total_hours += hours
        auth = "Yes" if config['requires_auth'] else "No"
        table.add_row(name, f"{hours:,}", auth, config['hf_name'])

    table.add_row("", "", "", "", style="dim")
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_hours:,}[/bold]", "", "")

    console.print(table)
    console.print(f"\n[yellow]Note: After WER filtering (~50% acceptance), expect ~{total_hours//2:,} hours[/yellow]")


def main():
    parser = argparse.ArgumentParser(description='Stream and process datasets for distillation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to process')
    parser.add_argument('--max-samples', type=int, help='Max samples per dataset (for testing)')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh, ignore previous progress')
    parser.add_argument('--merge-only', action='store_true', help='Only merge existing label files')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets')
    args = parser.parse_args()

    if args.list_datasets:
        print_dataset_table()
        return

    # Find config file
    config_path = Path(args.config)
    if not config_path.exists():
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / 'config.yaml'

    if not config_path.exists():
        console.print(f"[red]Config file not found: {args.config}[/red]")
        sys.exit(1)

    console.print(f"[bold]Using config: {config_path}[/bold]")

    # Create processor
    processor = StreamingDataProcessor(str(config_path))

    if args.merge_only:
        processor.merge_all_labels()
        return

    # Process datasets
    stats = processor.process_all_datasets(
        max_samples_per_dataset=args.max_samples,
        datasets_to_process=args.datasets,
        resume=not args.no_resume
    )

    # Print final summary
    console.print("\n[bold green]═══════════════════════════════════════[/bold green]")
    console.print("[bold green]        Processing Complete!           [/bold green]")
    console.print("[bold green]═══════════════════════════════════════[/bold green]")

    total_accepted = sum(s.samples_accepted for s in stats.values())
    total_hours = sum(s.total_hours for s in stats.values())

    console.print(f"\nTotal samples accepted: [green]{total_accepted:,}[/green]")
    console.print(f"Total hours: [green]{total_hours:.1f}[/green]")

    # Merge all labels
    processor.merge_all_labels()

    console.print("\n[bold]Next step:[/bold]")
    console.print("  python3 04_train_distillation.py --config ../config.yaml")


if __name__ == '__main__':
    main()
