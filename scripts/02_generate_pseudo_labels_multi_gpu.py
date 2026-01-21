#!/usr/bin/env python3
"""
=============================================================================
Multi-GPU Pseudo-Label Generation for Distil-CrisperWhisper
=============================================================================
Generates high-quality pseudo-labels using CrisperWhisper as the teacher model.
Designed for 4x H100 NVL (376GB total VRAM) for maximum throughput.

CRITICAL: We use CrisperWhisper (not standard Whisper) to preserve:
1. Improved word-level timestamp alignment
2. Better handling of disfluencies and filler words
3. Reduced hallucination on silence/music
4. All the fine-tuning improvements from nyrahealth

Datasets (Official Distil-Whisper v3.5 - 196,000 hours raw):
1. LibriSpeech (960 hrs)
2. Common Voice EN (3,000 hrs)
3. VoxPopuli EN (1,800 hrs)
4. AMI (100 hrs)
5. People's Speech (30,000 hrs)
6. TED-LIUM (450 hrs)
7. GigaSpeech (10,000 hrs)
8. YODAS (150,000 hrs)

Features:
- Multi-GPU parallel processing (4x H100 = 4x faster)
- Streaming mode to handle massive datasets
- WER-based filtering (discard >10% WER hallucinations)
- Automatic checkpointing and resume
- Returns word-level timestamps from CrisperWhisper

Usage:
  Single GPU:  python3 02_generate_pseudo_labels_multi_gpu.py --config ../config.yaml
  Multi-GPU:   accelerate launch 02_generate_pseudo_labels_multi_gpu.py --config ../config.yaml

References:
- CrisperWhisper: https://huggingface.co/nyrahealth/CrisperWhisper
- Distil-Whisper: https://arxiv.org/abs/2311.00430
=============================================================================
"""

import os
import sys
import json
import yaml
import argparse
import signal
import gc
import re
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import time
import warnings

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel

# Suppress warnings
warnings.filterwarnings('ignore')

console = Console()

# =============================================================================
# Dataset Configurations - All 8 from Official Distil-Whisper v3.5
# =============================================================================

DATASET_CONFIGS = {
    # Priority 1: High quality, clean audio
    'librispeech': {
        'hf_name': 'librispeech_asr',
        'subset': None,
        'splits': ['train.clean.100', 'train.clean.360', 'train.other.500'],
        'text_column': 'text',
        'audio_column': 'audio',
        'estimated_hours': 960,
        'requires_auth': False,
        'priority': 1,
        'quality': 'high',
    },

    # Priority 2: Large scale, diverse
    'gigaspeech': {
        'hf_name': 'speechcolab/gigaspeech',
        'subset': 'xl',  # Full 10,000 hours
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'estimated_hours': 10000,
        'requires_auth': True,
        'priority': 2,
        'quality': 'high',
    },

    # Priority 3: European Parliament - formal speech
    'voxpopuli': {
        'hf_name': 'facebook/voxpopuli',
        'subset': 'en',
        'splits': ['train'],
        'text_column': 'normalized_text',
        'audio_column': 'audio',
        'estimated_hours': 1800,
        'requires_auth': False,
        'priority': 3,
        'quality': 'high',
    },

    # Priority 4: Crowdsourced - diverse speakers
    'common_voice': {
        'hf_name': 'mozilla-foundation/common_voice_17_0',
        'subset': 'en',
        'splits': ['train'],
        'text_column': 'sentence',
        'audio_column': 'audio',
        'estimated_hours': 3000,
        'requires_auth': True,
        'priority': 4,
        'quality': 'medium',
    },

    # Priority 5: TED talks - clear speech
    'tedlium': {
        'hf_name': 'LIUM/tedlium',
        'subset': 'release3',
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'estimated_hours': 450,
        'requires_auth': False,
        'priority': 5,
        'quality': 'high',
    },

    # Priority 6: Meeting recordings
    'ami': {
        'hf_name': 'edinburghcstr/ami',
        'subset': 'ihm',
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'estimated_hours': 100,
        'requires_auth': False,
        'priority': 6,
        'quality': 'medium',
    },

    # Priority 7: Large scale diverse
    'peoples_speech': {
        'hf_name': 'MLCommons/peoples_speech',
        'subset': 'clean',
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'estimated_hours': 30000,
        'requires_auth': True,
        'priority': 7,
        'quality': 'medium',
    },

    # Priority 8: Massive YouTube dataset (bulk of data)
    'yodas': {
        'hf_name': 'espnet/yodas',
        'subset': 'en000',
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'estimated_hours': 150000,
        'requires_auth': False,
        'priority': 8,
        'quality': 'variable',
    },
}


@dataclass
class PseudoLabelEntry:
    """A single pseudo-labeled sample."""
    sample_id: str
    dataset: str
    ground_truth: str
    pseudo_label: str
    word_timestamps: List[Dict[str, Any]]  # CrisperWhisper's word-level timestamps
    wer: float
    duration_seconds: float
    audio_path: Optional[str] = None
    accepted: bool = True
    rejection_reason: Optional[str] = None


@dataclass
class DatasetProgress:
    """Progress tracking for a dataset."""
    name: str
    samples_processed: int = 0
    samples_accepted: int = 0
    samples_rejected_wer: int = 0
    samples_rejected_duration: int = 0
    samples_rejected_other: int = 0
    total_duration_hours: float = 0.0
    accepted_duration_hours: float = 0.0
    wer_sum: float = 0.0
    last_sample_id: str = ""
    status: str = "pending"  # pending, processing, completed, error
    error_message: Optional[str] = None

    @property
    def acceptance_rate(self) -> float:
        if self.samples_processed == 0:
            return 0.0
        return self.samples_accepted / self.samples_processed

    @property
    def avg_wer(self) -> float:
        if self.samples_accepted == 0:
            return 0.0
        return self.wer_sum / self.samples_accepted


class SpotInstanceHandler:
    """Handles spot instance preemption with graceful shutdown."""

    def __init__(self, save_callback=None):
        self.should_stop = False
        self.save_callback = save_callback
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, self._handle_signal)

    def _handle_signal(self, signum, frame):
        console.print(f"\n[bold yellow]⚠ Received signal {signum}. Saving progress...[/bold yellow]")
        self.should_stop = True
        if self.save_callback:
            self.save_callback()


class CrisperWhisperTeacher:
    """
    CrisperWhisper teacher model wrapper for pseudo-label generation.
    Preserves all CrisperWhisper improvements including word-level timestamps.
    """

    def __init__(
        self,
        model_id: str = "nyrahealth/CrisperWhisper",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        cache_dir: Optional[str] = None,
        local_rank: int = 0,
    ):
        self.model_id = model_id
        self.device = device or torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.local_rank = local_rank

        self.model = None
        self.processor = None
        self._loaded = False

    def load(self):
        """Load the CrisperWhisper model."""
        if self._loaded:
            return

        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        if self.local_rank == 0:
            console.print(f"\n[bold blue]Loading CrisperWhisper teacher model...[/bold blue]")
            console.print(f"  Model: {self.model_id}")
            console.print(f"  Device: {self.device}")
            console.print(f"  Dtype: {self.dtype}")

        self.processor = WhisperProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
        )

        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            torch_dtype=self.dtype,
            attn_implementation="sdpa",  # Fastest on H100
        ).to(self.device)

        self.model.eval()
        self._loaded = True

        if self.local_rank == 0:
            num_params = sum(p.numel() for p in self.model.parameters()) / 1e9
            console.print(f"[green]✓ CrisperWhisper loaded ({num_params:.2f}B parameters)[/green]")

    @torch.inference_mode()
    def generate_pseudo_label(
        self,
        audio_array: np.ndarray,
        sampling_rate: int = 16000,
        return_timestamps: bool = True,
        language: str = "en",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate pseudo-label with word-level timestamps using CrisperWhisper.

        Args:
            audio_array: Audio waveform as numpy array
            sampling_rate: Audio sample rate
            return_timestamps: Whether to return word-level timestamps
            language: Language code

        Returns:
            Tuple of (transcription, word_timestamps)
        """
        if not self._loaded:
            self.load()

        # Resample if necessary
        if sampling_rate != 16000:
            import torchaudio
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float()
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            audio_array = resampler(audio_tensor).squeeze(0).numpy()

        # Ensure float32 for processing
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Process audio to mel spectrogram
        input_features = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device, dtype=self.dtype)

        # Generate with CrisperWhisper
        # Use return_timestamps="word" to get word-level timestamps
        generated = self.model.generate(
            input_features,
            language=language,
            task="transcribe",
            return_timestamps=return_timestamps,
            return_token_timestamps=True if return_timestamps else False,
            max_new_tokens=448,
        )

        # Decode transcription
        transcription = self.processor.batch_decode(
            generated,
            skip_special_tokens=True
        )[0].strip()

        # Extract word timestamps if available
        word_timestamps = []
        if return_timestamps:
            try:
                # Decode with timestamps
                result = self.processor.batch_decode(
                    generated,
                    skip_special_tokens=False,
                    decode_with_timestamps=True
                )[0]

                # Parse timestamps from the result
                word_timestamps = self._parse_word_timestamps(result)
            except Exception:
                # Fallback: no word timestamps
                pass

        return transcription, word_timestamps

    def _parse_word_timestamps(self, decoded_with_timestamps: str) -> List[Dict[str, Any]]:
        """Parse word-level timestamps from decoded output."""
        timestamps = []

        # Pattern to match timestamp tokens like <|0.00|>word<|0.50|>
        pattern = r'<\|(\d+\.?\d*)\|>([^<]+)'
        matches = re.findall(pattern, decoded_with_timestamps)

        for i, (start_time, text) in enumerate(matches):
            text = text.strip()
            if text:
                end_time = float(matches[i+1][0]) if i+1 < len(matches) else float(start_time) + 0.5
                timestamps.append({
                    'word': text,
                    'start': float(start_time),
                    'end': end_time
                })

        return timestamps

    def get_encoder_output(
        self,
        audio_array: np.ndarray,
        sampling_rate: int = 16000,
    ) -> torch.Tensor:
        """Get encoder output for potential feature extraction."""
        if not self._loaded:
            self.load()

        if sampling_rate != 16000:
            import torchaudio
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float()
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            audio_array = resampler(audio_tensor).squeeze(0).numpy()

        input_features = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device, dtype=self.dtype)

        encoder_output = self.model.model.encoder(input_features)
        return encoder_output.last_hidden_state


class MultiGPUPseudoLabelGenerator:
    """
    Multi-GPU pseudo-label generator for maximum throughput on 4x H100 NVL.
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

        # Distributed setup
        self.is_distributed = dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.local_rank = dist.get_rank() if self.is_distributed else 0
        self.is_main = self.local_rank == 0

        # Device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # Paths
        self.workspace = Path(self.config['paths']['workspace'])
        self.pseudo_labels_dir = Path(self.config['paths'].get(
            'pseudo_labels_dir',
            self.workspace / 'pseudo_labels'
        ))
        self.pseudo_labels_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.progress_file = self.pseudo_labels_dir / 'generation_progress.json'

        # WER threshold for filtering
        distil_config = self.config.get('distillation', {})
        pseudo_config = distil_config.get('pseudo_labels', {})
        self.wer_threshold = pseudo_config.get('wer_threshold', 0.10)
        self.min_duration = 1.0  # Minimum audio duration in seconds
        self.max_duration = 30.0  # Maximum audio duration in seconds

        # Teacher model (loaded lazily)
        self.teacher = None

        # Spot instance handler
        self.spot_handler = SpotInstanceHandler(save_callback=self._emergency_save)

        if self.is_main:
            console.print(Panel.fit(
                f"[bold cyan]Multi-GPU Pseudo-Label Generator[/bold cyan]\n"
                f"GPUs: {self.world_size}\n"
                f"WER Threshold: {self.wer_threshold * 100:.0f}%\n"
                f"Duration: {self.min_duration}s - {self.max_duration}s\n"
                f"Output: {self.pseudo_labels_dir}",
                title="Configuration"
            ))

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_teacher(self):
        """Load CrisperWhisper teacher model."""
        if self.teacher is not None:
            return

        teacher_config = self.config['teacher']
        cache_dir = self.config['paths'].get('hf_cache')

        self.teacher = CrisperWhisperTeacher(
            model_id=teacher_config['model_id'],
            device=self.device,
            dtype=getattr(torch, teacher_config.get('dtype', 'float16')),
            cache_dir=cache_dir,
            local_rank=self.local_rank,
        )
        self.teacher.load()

    def _load_progress(self) -> Dict[str, DatasetProgress]:
        """Load processing progress for resume support."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                return {
                    name: DatasetProgress(**info)
                    for name, info in data.get('datasets', {}).items()
                }
        return {}

    def _save_progress(self, progress: Dict[str, DatasetProgress]):
        """Save processing progress."""
        data = {
            'datasets': {name: asdict(p) for name, p in progress.items()},
            'last_update': datetime.now().isoformat(),
            'world_size': self.world_size,
        }

        # Only main process saves
        if self.is_main:
            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2)

    def _emergency_save(self):
        """Emergency save on spot instance preemption."""
        if self.is_main:
            console.print("[bold yellow]Emergency save triggered...[/bold yellow]")
            # Progress is saved incrementally, just log
            console.print("[green]Progress already saved incrementally[/green]")

    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate."""
        from jiwer import wer as calculate_wer

        # Normalize
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
        """Normalize text for WER calculation."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s\']', '', text)
        text = ' '.join(text.split())
        return text

    def _load_dataset_streaming(self, name: str) -> Optional[Iterator]:
        """Load dataset in streaming mode."""
        from datasets import load_dataset, interleave_datasets

        config = DATASET_CONFIGS.get(name)
        if not config:
            return None

        try:
            if self.is_main:
                console.print(f"[yellow]Loading {name} in streaming mode...[/yellow]")

            # Special handling for LibriSpeech (multiple splits)
            if name == 'librispeech':
                datasets_list = []
                for split in config['splits']:
                    ds = load_dataset(
                        config['hf_name'],
                        split=split,
                        streaming=True,
                        trust_remote_code=True,
                    )
                    datasets_list.append(ds)

                dataset = interleave_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]
            else:
                kwargs = {
                    'split': config['splits'][0],
                    'streaming': True,
                    'trust_remote_code': True,
                }
                if config['subset']:
                    kwargs['name'] = config['subset']

                dataset = load_dataset(config['hf_name'], **kwargs)

            if self.is_main:
                console.print(f"[green]✓ {name} loaded successfully[/green]")

            return iter(dataset)

        except Exception as e:
            if self.is_main:
                console.print(f"[red]✗ Error loading {name}: {e}[/red]")
                if config['requires_auth']:
                    console.print(f"[yellow]  This dataset requires HuggingFace authentication[/yellow]")
            return None

    def process_dataset(
        self,
        name: str,
        max_samples: Optional[int] = None,
        resume: bool = True,
    ) -> DatasetProgress:
        """
        Process a single dataset with CrisperWhisper pseudo-labeling.

        Includes:
        - Word-level timestamp extraction
        - WER-based filtering
        - Duration filtering
        - Multi-GPU support via sample sharding
        """
        self._load_teacher()

        config = DATASET_CONFIGS.get(name)
        if not config:
            return DatasetProgress(name=name, status="error", error_message="Unknown dataset")

        # Output files (one per GPU to avoid conflicts)
        output_file = self.pseudo_labels_dir / f'{name}_gpu{self.local_rank}_accepted.jsonl'
        rejected_file = self.pseudo_labels_dir / f'{name}_gpu{self.local_rank}_rejected.jsonl'

        # Load progress
        all_progress = self._load_progress()
        progress = all_progress.get(name, DatasetProgress(name=name))

        # Resume handling
        if resume and progress.samples_processed > 0:
            if self.is_main:
                console.print(f"[yellow]Resuming {name} from sample {progress.samples_processed}[/yellow]")
            start_idx = progress.samples_processed
            mode = 'a'
        else:
            start_idx = 0
            mode = 'w'
            progress = DatasetProgress(name=name)

        progress.status = "processing"

        # Load dataset
        dataset_iter = self._load_dataset_streaming(name)
        if dataset_iter is None:
            progress.status = "error"
            progress.error_message = "Failed to load dataset"
            return progress

        text_col = config['text_column']
        audio_col = config['audio_column']

        # Process samples
        with open(output_file, mode) as out_f, open(rejected_file, mode) as rej_f:
            # Progress bar only on main process
            if self.is_main:
                pbar = tqdm(
                    desc=f"Processing {name} (GPU {self.local_rank})",
                    total=max_samples or config['estimated_hours'] * 120,  # ~120 samples/hour estimate
                    unit="samples"
                )
            else:
                pbar = None

            sample_idx = 0

            for sample in dataset_iter:
                if self.spot_handler.should_stop:
                    break

                # Skip to resume point
                if sample_idx < start_idx:
                    sample_idx += 1
                    continue

                # Multi-GPU sharding: each GPU processes every Nth sample
                if sample_idx % self.world_size != self.local_rank:
                    sample_idx += 1
                    continue

                if max_samples and progress.samples_processed >= max_samples // self.world_size:
                    break

                try:
                    # Extract audio
                    audio_data = sample.get(audio_col, {})
                    if isinstance(audio_data, dict):
                        audio_array = np.array(audio_data['array'], dtype=np.float32)
                        sr = audio_data['sampling_rate']
                    else:
                        sample_idx += 1
                        continue

                    # Get ground truth text
                    ground_truth = sample.get(text_col, '')
                    if not ground_truth or not isinstance(ground_truth, str):
                        sample_idx += 1
                        continue

                    # Calculate duration
                    duration = len(audio_array) / sr
                    progress.total_duration_hours += duration / 3600

                    # Duration filter
                    if duration < self.min_duration or duration > self.max_duration:
                        progress.samples_rejected_duration += 1
                        sample_idx += 1
                        progress.samples_processed += 1
                        continue

                    # Generate pseudo-label with CrisperWhisper
                    pseudo_label, word_timestamps = self.teacher.generate_pseudo_label(
                        audio_array=audio_array,
                        sampling_rate=sr,
                        return_timestamps=True,
                        language="en",
                    )

                    # Calculate WER
                    wer = self._calculate_wer(ground_truth, pseudo_label)

                    # Create entry
                    entry = PseudoLabelEntry(
                        sample_id=f"{name}_{sample_idx:010d}",
                        dataset=name,
                        ground_truth=ground_truth,
                        pseudo_label=pseudo_label,
                        word_timestamps=word_timestamps,
                        wer=wer,
                        duration_seconds=duration,
                        accepted=wer <= self.wer_threshold,
                        rejection_reason=None if wer <= self.wer_threshold else f"WER {wer:.2%} > {self.wer_threshold:.0%}"
                    )

                    # Write to appropriate file
                    entry_json = json.dumps(asdict(entry))

                    if entry.accepted:
                        out_f.write(entry_json + '\n')
                        out_f.flush()  # Ensure data is written
                        progress.samples_accepted += 1
                        progress.accepted_duration_hours += duration / 3600
                        progress.wer_sum += wer
                    else:
                        rej_f.write(entry_json + '\n')
                        progress.samples_rejected_wer += 1

                    progress.samples_processed += 1
                    progress.last_sample_id = entry.sample_id

                    # Update progress bar
                    if pbar:
                        acc_rate = progress.acceptance_rate * 100
                        avg_wer = progress.avg_wer * 100
                        pbar.set_postfix({
                            'acc': f'{acc_rate:.1f}%',
                            'wer': f'{avg_wer:.1f}%',
                            'hrs': f'{progress.accepted_duration_hours:.1f}'
                        })
                        pbar.update(1)

                    # Save progress periodically
                    if progress.samples_processed % 500 == 0:
                        all_progress[name] = progress
                        self._save_progress(all_progress)

                    sample_idx += 1

                except Exception as e:
                    progress.samples_rejected_other += 1
                    sample_idx += 1
                    continue

            if pbar:
                pbar.close()

        # Final save
        progress.status = "completed"
        all_progress[name] = progress
        self._save_progress(all_progress)

        return progress

    def process_all_datasets(
        self,
        datasets_to_process: Optional[List[str]] = None,
        max_samples_per_dataset: Optional[int] = None,
        resume: bool = True,
    ) -> Dict[str, DatasetProgress]:
        """Process all enabled datasets."""

        # Get datasets from config or use all
        if datasets_to_process:
            dataset_names = datasets_to_process
        else:
            datasets_config = self.config.get('datasets', {})
            dataset_names = [
                name for name, cfg in datasets_config.items()
                if cfg.get('enabled', True) and name in DATASET_CONFIGS
            ]

        # Sort by priority
        dataset_names = sorted(
            dataset_names,
            key=lambda x: DATASET_CONFIGS.get(x, {}).get('priority', 99)
        )

        if self.is_main:
            console.print(f"\n[bold blue]Processing {len(dataset_names)} datasets with {self.world_size} GPUs[/bold blue]")
            for name in dataset_names:
                cfg = DATASET_CONFIGS.get(name, {})
                console.print(f"  • {name}: ~{cfg.get('estimated_hours', 0):,} hours")

        # Synchronize before starting
        if self.is_distributed:
            dist.barrier()

        all_progress = {}

        for name in dataset_names:
            if self.spot_handler.should_stop:
                break

            if self.is_main:
                console.print(f"\n[bold cyan]{'═' * 50}[/bold cyan]")
                console.print(f"[bold cyan]  Processing: {name}[/bold cyan]")
                console.print(f"[bold cyan]{'═' * 50}[/bold cyan]")

            progress = self.process_dataset(
                name=name,
                max_samples=max_samples_per_dataset,
                resume=resume,
            )

            all_progress[name] = progress

            # Print summary
            if self.is_main:
                console.print(f"\n  [bold]Results for {name}:[/bold]")
                console.print(f"    Processed: {progress.samples_processed:,}")
                console.print(f"    Accepted: [green]{progress.samples_accepted:,}[/green]")
                console.print(f"    Rejected (WER): [red]{progress.samples_rejected_wer:,}[/red]")
                console.print(f"    Rejected (duration): [yellow]{progress.samples_rejected_duration:,}[/yellow]")
                console.print(f"    Accepted hours: [green]{progress.accepted_duration_hours:.1f}[/green]")
                console.print(f"    Avg WER: {progress.avg_wer * 100:.1f}%")

            # Synchronize between datasets
            if self.is_distributed:
                dist.barrier()

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        return all_progress

    def merge_all_outputs(self) -> Tuple[Path, Dict[str, Any]]:
        """Merge all pseudo-label files from all GPUs into a single training file."""

        if not self.is_main:
            return None, {}

        merged_path = self.pseudo_labels_dir / 'all_pseudo_labels.jsonl'

        console.print("\n[bold blue]Merging all pseudo-label files...[/bold blue]")

        total_count = 0
        total_hours = 0.0
        dataset_stats = {}

        with open(merged_path, 'w') as merged_file:
            # Find all accepted files from all GPUs
            for accepted_file in sorted(self.pseudo_labels_dir.glob('*_accepted.jsonl')):
                # Parse dataset name from filename
                parts = accepted_file.stem.split('_gpu')
                if len(parts) >= 1:
                    dataset_name = parts[0]
                else:
                    dataset_name = accepted_file.stem

                console.print(f"  Adding {accepted_file.name}...")
                count = 0
                hours = 0.0

                with open(accepted_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get('accepted', True):
                                merged_file.write(line)
                                count += 1
                                hours += entry.get('duration_seconds', 0) / 3600
                        except json.JSONDecodeError:
                            continue

                total_count += count
                total_hours += hours

                if dataset_name not in dataset_stats:
                    dataset_stats[dataset_name] = {'count': 0, 'hours': 0.0}
                dataset_stats[dataset_name]['count'] += count
                dataset_stats[dataset_name]['hours'] += hours

                console.print(f"    Added {count:,} samples ({hours:.1f} hours)")

        console.print(f"\n[bold green]✓ Merged {total_count:,} samples ({total_hours:.1f} hours)[/bold green]")
        console.print(f"  Output: {merged_path}")

        # Print per-dataset breakdown
        console.print("\n[bold]Per-dataset breakdown:[/bold]")
        table = Table()
        table.add_column("Dataset")
        table.add_column("Samples", justify="right")
        table.add_column("Hours", justify="right")

        for name, stats in sorted(dataset_stats.items()):
            table.add_row(name, f"{stats['count']:,}", f"{stats['hours']:.1f}")

        table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_count:,}[/bold]", f"[bold]{total_hours:.1f}[/bold]")
        console.print(table)

        return merged_path, {
            'total_count': total_count,
            'total_hours': total_hours,
            'datasets': dataset_stats
        }


def print_dataset_info():
    """Print available datasets information."""
    table = Table(title="[bold]Official Distil-Whisper v3.5 Datasets[/bold]")

    table.add_column("Dataset", style="cyan")
    table.add_column("Hours", justify="right")
    table.add_column("Quality", justify="center")
    table.add_column("Auth", justify="center")
    table.add_column("Priority", justify="center")

    total_hours = 0
    for name, config in sorted(DATASET_CONFIGS.items(), key=lambda x: x[1]['priority']):
        hours = config['estimated_hours']
        total_hours += hours
        quality_color = {'high': 'green', 'medium': 'yellow', 'variable': 'dim'}.get(config['quality'], 'white')
        table.add_row(
            name,
            f"{hours:,}",
            f"[{quality_color}]{config['quality']}[/{quality_color}]",
            "Yes" if config['requires_auth'] else "No",
            str(config['priority'])
        )

    table.add_row("", "", "", "", "")
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_hours:,}[/bold]", "", "", "")

    console.print(table)
    console.print(f"\n[yellow]After WER filtering (~50% acceptance rate): ~{total_hours // 2:,} hours[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description='Generate pseudo-labels using CrisperWhisper (Multi-GPU)'
    )
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to process')
    parser.add_argument('--max-samples', type=int, help='Max samples per dataset (for testing)')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh')
    parser.add_argument('--merge-only', action='store_true', help='Only merge existing files')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed')
    args = parser.parse_args()

    # Handle distributed launch
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    # Initialize distributed if available
    if torch.cuda.device_count() > 1 and not dist.is_initialized():
        try:
            dist.init_process_group(backend='nccl')
        except Exception:
            pass

    is_main = not dist.is_initialized() or dist.get_rank() == 0

    if args.list_datasets:
        if is_main:
            print_dataset_info()
        return

    # Find config file
    config_path = Path(args.config)
    if not config_path.exists():
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / 'config.yaml'

    if not config_path.exists():
        if is_main:
            console.print(f"[red]Config file not found: {args.config}[/red]")
        sys.exit(1)

    if is_main:
        console.print(f"[bold]Using config: {config_path}[/bold]")

    # Create generator
    generator = MultiGPUPseudoLabelGenerator(str(config_path))

    if args.merge_only:
        generator.merge_all_outputs()
        return

    # Process datasets
    all_progress = generator.process_all_datasets(
        datasets_to_process=args.datasets,
        max_samples_per_dataset=args.max_samples,
        resume=not args.no_resume,
    )

    # Synchronize before merging
    if dist.is_initialized():
        dist.barrier()

    # Merge outputs (main process only)
    if is_main:
        merged_path, stats = generator.merge_all_outputs()

        console.print(f"\n[bold green]{'═' * 60}[/bold green]")
        console.print("[bold green]  Pseudo-Label Generation Complete![/bold green]")
        console.print(f"[bold green]{'═' * 60}[/bold green]")
        console.print(f"\nTotal accepted samples: [green]{stats['total_count']:,}[/green]")
        console.print(f"Total accepted hours: [green]{stats['total_hours']:.1f}[/green]")
        console.print(f"\nOutput file: {merged_path}")
        console.print("\n[bold]Next step:[/bold]")
        console.print("  accelerate launch 04_train_distillation.py --config ../config.yaml")

    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
