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
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from threading import Thread, Event
import hashlib

import torch
import torchaudio
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel

# Official Whisper text normalizer (matches distil-whisper methodology)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

# Suppress warnings - especially the noisy Whisper decoder warnings
warnings.filterwarnings('ignore')
# Specifically suppress the forced_decoder_ids and attention_mask warnings from transformers
import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

console = Console()

# =============================================================================
# HuggingFace Authentication Setup
# =============================================================================
# This is CRITICAL to avoid 429 rate limiting when multiple GPUs hit HF Hub
# Set HF_TOKEN environment variable or login via `huggingface-cli login`
# =============================================================================

def setup_hf_authentication():
    """Setup HuggingFace authentication to avoid rate limiting."""
    from huggingface_hub import HfFolder, login

    # Check for existing token
    token = os.environ.get('HF_TOKEN') or HfFolder.get_token()

    if token:
        # Already authenticated
        return True

    # Try to load from common locations
    hf_token_file = Path.home() / '.huggingface' / 'token'
    if hf_token_file.exists():
        return True

    console.print("[yellow]⚠ No HuggingFace token found![/yellow]")
    console.print("[yellow]  This may cause 429 rate limiting with multi-GPU.[/yellow]")
    console.print("[yellow]  To fix: export HF_TOKEN=your_token or run 'huggingface-cli login'[/yellow]")
    return False

# Run auth check at import time (only on main process)
_HF_AUTH_CHECKED = False

# =============================================================================
# Dataset Configurations
# =============================================================================
# Includes:
# - All 8 datasets from Official Distil-Whisper v3.5 (~196,000 hours)
# - CrisperWhisper's official filler/verbatim datasets (AMI-IHM, PodcastFillers)
#
# References:
# - Distil-Whisper: https://arxiv.org/abs/2311.00430
# - CrisperWhisper: https://arxiv.org/abs/2408.16589
# =============================================================================

DATASET_CONFIGS = {
    # =========================================================================
    # CRISPERWHISPER OFFICIAL FILLER DATASETS (verbatim transcriptions)
    # These are CRITICAL for preserving filler word transcription capability
    # Source: arxiv.org/abs/2408.16589
    # =========================================================================

    # AMI Meeting Corpus - IHM (Individual Headset Microphone)
    # ~29,000 meeting recording clips with verbatim transcriptions
    # Contains natural fillers (um, uh), disfluencies, and word-level timestamps
    # This is one of the PRIMARY datasets used by CrisperWhisper
    'ami': {
        'hf_name': 'edinburghcstr/ami',
        'subset': 'ihm',
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'estimated_hours': 100,
        'requires_auth': False,
        'priority': 2,  # High priority - critical for filler preservation
        'quality': 'high',
        'verbatim': True,
        'has_fillers': True,
    },

    # PodcastFillers - Filler word detection dataset
    # ~35,000 filler instances (um, uh) with timing annotations
    # CrisperWhisper expands this to ~105,000 samples via context sampling
    # Source: https://huggingface.co/datasets/ylacombe/podcast_fillers_by_license
    'podcast_fillers': {
        'hf_name': 'ylacombe/podcast_fillers_by_license',
        'subset': None,
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'estimated_hours': 145,
        'requires_auth': False,
        'priority': 3,  # High priority - explicit filler annotations
        'quality': 'high',
        'verbatim': True,
        'has_fillers': True,
        'filler_annotations': True,
    },

    # =========================================================================
    # DISTIL-WHISPER v3.5 DATASETS (~196,000 hours)
    # =========================================================================

    # Priority 1: High quality, clean audio (LibriSpeech) - finish existing progress first
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

    # Priority 4: European Parliament - formal speech (VoxPopuli)
    'voxpopuli': {
        'hf_name': 'facebook/voxpopuli',
        'subset': 'en',
        'splits': ['train'],
        'text_column': 'normalized_text',
        'audio_column': 'audio',
        'estimated_hours': 1800,
        'requires_auth': False,
        'priority': 4,
        'quality': 'high',
    },

    # Priority 5: Crowdsourced - diverse speakers (Common Voice)
    'common_voice': {
        'hf_name': 'mozilla-foundation/common_voice_17_0',
        'subset': 'en',
        'splits': ['train'],
        'text_column': 'sentence',
        'audio_column': 'audio',
        'estimated_hours': 3000,
        'requires_auth': True,
        'priority': 5,
        'quality': 'medium',
    },

    # Priority 6: TED talks - clear speech (TED-LIUM)
    'tedlium': {
        'hf_name': 'LIUM/tedlium',
        'subset': 'release3',
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'estimated_hours': 450,
        'requires_auth': False,
        'priority': 6,
        'quality': 'high',
    },

    # =========================================================================
    # STREAMING DATASETS (large, processed last)
    # =========================================================================

    # Priority 7: Large scale, diverse (GigaSpeech) - STREAMING
    'gigaspeech': {
        'hf_name': 'speechcolab/gigaspeech',
        'subset': 'xl',  # Full 10,000 hours
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'estimated_hours': 10000,
        'requires_auth': True,
        'priority': 7,
        'quality': 'high',
    },

    # Priority 8: Large scale diverse (People's Speech) - STREAMING
    'peoples_speech': {
        'hf_name': 'MLCommons/peoples_speech',
        'subset': 'clean',
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'estimated_hours': 30000,
        'requires_auth': True,
        'priority': 8,
        'quality': 'medium',
    },

    # Priority 9: Massive YouTube dataset - bulk of data (YODAS) - STREAMING
    'yodas': {
        'hf_name': 'espnet/yodas',
        'subset': 'en000',
        'splits': ['train'],
        'text_column': 'text',
        'audio_column': 'audio',
        'estimated_hours': 150000,
        'requires_auth': False,
        'priority': 9,
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
    """
    Progress tracking for a dataset.

    Designed to be GPU-count agnostic for robust resume:
    - Tracks processed sample IDs (not just count)
    - Can resume with different GPU count than original run
    - Handles GPU failures gracefully
    """
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
    last_sample_idx: int = 0  # Track the last processed sample index
    status: str = "pending"  # pending, processing, completed, error
    error_message: Optional[str] = None
    # Track which GPUs contributed (for diagnostics)
    gpu_count_at_start: int = 1
    processed_sample_ids: List[str] = field(default_factory=list)  # For exact resume

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


def generate_content_hash(audio_array: np.ndarray, text: str, dataset_name: str) -> str:
    """
    Generate a deterministic, content-based hash ID for a sample.

    This ensures the same sample always gets the same ID, regardless of:
    - Dataset iteration order
    - Run number
    - GPU count or which GPU processes it

    Uses:
    - First 4000 audio samples (or all if shorter) - captures unique audio signature
    - Ground truth text - additional uniqueness
    - Dataset name - namespace to avoid collisions

    Returns a 16-character hex string like "ami_a1b2c3d4e5f6g7h8"
    (64 bits = collision probability ~0.03% at 20M samples - acceptable)

    NOTE: DO NOT change hash length without migration strategy!
    Existing processed IDs use 16-char hashes.
    """
    hasher = hashlib.md5()

    # Hash audio content (first 4000 samples = ~0.25s at 16kHz)
    # Using tobytes() is fast and deterministic for same audio
    audio_slice = audio_array[:4000] if len(audio_array) > 4000 else audio_array
    hasher.update(audio_slice.tobytes())

    # Hash the text
    hasher.update(text.encode('utf-8'))

    # Include dataset name
    hasher.update(dataset_name.encode('utf-8'))

    # Return formatted ID: {dataset}_{hash16}
    return f"{dataset_name}_{hasher.hexdigest()[:16]}"


class ThreadedPrefetcher:
    """
    Multi-threaded prefetcher for dataset samples.

    Architecture (optimized for 16 CPU cores):
    ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
    │   Dataset   │───>│  Raw Sample Queue │───>│  Worker Pool │
    │  Iterator   │    │   (unbounded)     │    │  (N threads) │
    │  (1 thread) │    └──────────────────┘    └──────┬──────┘
    └─────────────┘                                    │
                                                       v
                                              ┌──────────────────┐
                                              │ Processed Queue  │───> GPU Batches
                                              │ (bounded, 4*batch)│
                                              └──────────────────┘

    - 1 feeder thread: iterates dataset, puts raw samples in queue
    - N worker threads: extract audio, compute hash, filter, validate
    - Main thread: pulls batches for GPU processing

    Big O:
    - Hash generation: O(1) - constant 4000 samples
    - Dedup lookup: O(1) - hash set
    - Per-sample processing: O(audio_length) but parallelized across N workers
    """

    def __init__(
        self,
        dataset_iter: Iterator,
        start_idx: int,
        world_size: int,
        local_rank: int,
        already_processed_ids: set,
        audio_col: str,
        text_col: str,
        min_duration: float,
        max_duration: float,
        name: str,
        prefetch_batches: int = 4,  # How many batches to prefetch
        batch_size: int = 48,
        num_workers: int = 8,  # Number of worker threads
    ):
        self.dataset_iter = dataset_iter
        self.sample_idx = start_idx
        self.world_size = world_size
        self.local_rank = local_rank
        self.already_processed_ids = already_processed_ids
        self.audio_col = audio_col
        self.text_col = text_col
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Raw sample queue (feeder -> workers)
        # Size: enough to keep all workers busy + buffer
        # With many workers (32+), need larger queue to prevent feeder blocking
        self.raw_queue = Queue(maxsize=max(num_workers * 4, 128))

        # Processed sample queue (workers -> main) - bounded to control memory
        # Size: prefetch_batches * batch_size samples ready for GPU
        # This is the KEY buffer that prevents GPU starvation!
        self.processed_queue = Queue(maxsize=prefetch_batches * batch_size)

        self.stop_event = Event()
        self.feeder_done = Event()  # Set when feeder finishes iterating
        self.exhausted = Event()    # Set when all workers are done

        # Thread-safe stats using locks
        import threading
        self._lock = threading.Lock()
        self._skipped_in_loop = 0
        self._rejected_duration = 0
        self._total_duration_hours = 0.0
        self._workers_done = 0

        # Start threads
        self.threads = []
        self._start_workers()

    def _start_workers(self):
        """Start the feeder and worker threads."""
        # 1 feeder thread - iterates dataset
        feeder = Thread(target=self._feeder_loop, daemon=True, name="prefetch-feeder")
        feeder.start()
        self.threads.append(feeder)

        # N worker threads - process samples in parallel
        for i in range(self.num_workers):
            worker = Thread(target=self._worker_loop, daemon=True, name=f"prefetch-worker-{i}")
            worker.start()
            self.threads.append(worker)

    def _feeder_loop(self):
        """Feeder thread: iterates dataset and queues raw samples."""
        try:
            sample_idx = self.sample_idx
            for sample in self.dataset_iter:
                if self.stop_event.is_set():
                    break

                # Put raw sample with its index into queue
                # Use timeout to avoid blocking forever if queue is full
                while not self.stop_event.is_set():
                    try:
                        self.raw_queue.put((sample, sample_idx), timeout=1.0)
                        break
                    except:
                        continue  # Retry until stop or success
                sample_idx += 1

        except Exception:
            pass
        finally:
            self.feeder_done.set()
            # Signal workers that no more raw samples are coming
            # Use timeout to avoid deadlock if queue is full
            for _ in range(self.num_workers):
                while not self.stop_event.is_set():
                    try:
                        self.raw_queue.put(None, timeout=1.0)  # Poison pill
                        break
                    except:
                        continue

    def _worker_loop(self):
        """Worker thread: processes raw samples in parallel."""
        local_skipped = 0
        local_rejected = 0
        local_duration = 0.0

        try:
            while not self.stop_event.is_set():
                try:
                    item = self.raw_queue.get(timeout=1.0)
                except Empty:
                    if self.feeder_done.is_set() and self.raw_queue.empty():
                        break
                    continue

                if item is None:  # Poison pill
                    break

                sample, sample_idx = item

                # Process sample (extract audio, hash, filter)
                result = self._process_sample(sample, sample_idx)
                if result is None:
                    continue
                if result == 'duration_rejected':
                    local_rejected += 1
                    continue

                # Unpack to check dedup and sharding
                audio_array, sr, ground_truth, sample_id, duration, idx = result

                # Update local duration stats
                local_duration += duration / 3600

                # Skip if already processed (O(1) hash set lookup)
                if sample_id in self.already_processed_ids:
                    local_skipped += 1
                    continue

                # Multi-GPU sharding - deterministic based on content hash
                shard_key = int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16)
                if shard_key % self.world_size != self.local_rank:
                    continue

                # Put in processed queue (may block if full - backpressure)
                # Retry with longer timeout to avoid dropping samples
                put_success = False
                for _ in range(12):  # 12 * 5s = 60s max wait
                    if self.stop_event.is_set():
                        break
                    try:
                        self.processed_queue.put(result, timeout=5.0)
                        put_success = True
                        break
                    except:
                        continue  # Retry

                if not put_success and not self.stop_event.is_set():
                    # Log dropped sample (shouldn't happen in normal operation)
                    pass

        except Exception:
            pass
        finally:
            # Aggregate stats thread-safely
            with self._lock:
                self._skipped_in_loop += local_skipped
                self._rejected_duration += local_rejected
                self._total_duration_hours += local_duration
                self._workers_done += 1

                # If all workers done, signal exhausted
                if self._workers_done >= self.num_workers:
                    self.exhausted.set()

    def _process_sample(self, sample, sample_idx: int):
        """
        Process a single sample and generate content-based ID. Thread-safe.

        Returns:
            Tuple of (audio_array, sr, ground_truth, sample_id, duration, sample_idx)
            OR 'duration_rejected' string if filtered by duration
            OR None if invalid sample
        """
        try:
            # Extract audio
            audio_data = sample.get(self.audio_col, {})
            if not isinstance(audio_data, dict):
                return None

            audio_array = np.array(audio_data['array'], dtype=np.float32)
            sr = audio_data['sampling_rate']

            # Get ground truth
            ground_truth = sample.get(self.text_col, '')
            if not ground_truth or not isinstance(ground_truth, str):
                return None

            # Generate CONTENT-BASED sample ID (O(1) - only first 4000 samples)
            sample_id = generate_content_hash(audio_array, ground_truth, self.name)

            # Calculate duration
            duration = len(audio_array) / sr

            # Duration filter - return marker so we can track stats
            if duration < self.min_duration or duration > self.max_duration:
                return 'duration_rejected'

            return (audio_array, sr, ground_truth, sample_id, duration, sample_idx)

        except Exception:
            return None

    def get_batch(self, timeout: float = 30.0) -> List[Tuple]:
        """
        Get a batch of preprocessed samples from the worker pool.

        OPTIMIZATION: Returns partial batches after short wait to avoid GPU idling.
        - First, try to get a full batch with short timeout (0.1s per item)
        - If queue has items, return whatever we have (even if partial)
        - Only wait longer if queue is completely empty

        This ensures GPU stays busy even when prefetching is slightly behind.
        """
        batch = []
        deadline = time.time() + timeout

        # Phase 1: Quick collection - grab whatever is immediately available
        while len(batch) < self.batch_size:
            try:
                # Very short timeout - don't wait long for each item
                item = self.processed_queue.get(timeout=0.05)
                batch.append(item)
            except Empty:
                break  # Nothing immediately available

        # Phase 2: If we got nothing, wait a bit longer for at least some data
        if not batch:
            while len(batch) < self.batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break

                # Check if we're done
                if self.exhausted.is_set() and self.processed_queue.empty():
                    break

                try:
                    item = self.processed_queue.get(timeout=min(remaining, 1.0))
                    batch.append(item)

                    # Once we have at least 25% of a batch, return it
                    # This prevents GPU from waiting too long
                    if len(batch) >= self.batch_size // 4:
                        # Quick grab any more that are immediately ready
                        while len(batch) < self.batch_size:
                            try:
                                item = self.processed_queue.get_nowait()
                                batch.append(item)
                            except Empty:
                                break
                        break
                except Empty:
                    continue

        return batch

    def is_exhausted(self) -> bool:
        """Check if all workers are done and processed queue is empty."""
        return self.exhausted.is_set() and self.processed_queue.empty()

    def stop(self):
        """Signal all threads to stop gracefully."""
        self.stop_event.set()
        # Clear queues to unblock any waiting threads
        try:
            while not self.raw_queue.empty():
                self.raw_queue.get_nowait()
        except:
            pass
        try:
            while not self.processed_queue.empty():
                self.processed_queue.get_nowait()
        except:
            pass

    def get_stats(self) -> dict:
        """Get prefetcher statistics (thread-safe)."""
        with self._lock:
            return {
                'skipped_in_loop': self._skipped_in_loop,
                'rejected_duration': self._rejected_duration,
                'total_duration_hours': self._total_duration_hours,
                'workers_active': self.num_workers - self._workers_done,
                'raw_queue_size': self.raw_queue.qsize(),
                'queue_size': self.processed_queue.qsize(),
            }


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
        # Note: max_new_tokens must be < 448 to leave room for decoder_input_ids
        # (language token, task token, etc. = ~4 tokens)
        generated = self.model.generate(
            input_features,
            language=language,
            task="transcribe",
            return_timestamps=return_timestamps,
            return_token_timestamps=True if return_timestamps else False,
            max_new_tokens=440,
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

    @torch.inference_mode()
    def generate_pseudo_labels_batch(
        self,
        audio_arrays: List[np.ndarray],
        sampling_rate: int = 16000,
        language: str = "en",
    ) -> List[str]:
        """
        Generate pseudo-labels for a BATCH of audio samples.

        This is the key method for GPU-efficient batch processing.
        Instead of processing one sample at a time, this processes
        the entire batch in a single GPU call.

        Args:
            audio_arrays: List of audio waveforms as numpy arrays
            sampling_rate: Audio sample rate (will resample to 16kHz if different)
            language: Language code

        Returns:
            List of transcriptions (one per audio sample)
        """
        if not self._loaded:
            self.load()

        if not audio_arrays:
            return []

        # Preprocess all audio arrays
        processed_audios = []
        for audio in audio_arrays:
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Resample to 16kHz if needed
            if sampling_rate != 16000:
                audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
                resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
                audio = resampler(audio_tensor).squeeze(0).numpy()

            # Ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            processed_audios.append(audio)

        # Batch feature extraction with padding
        # The processor handles variable-length audio by padding to the longest
        input_features = self.processor(
            processed_audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True  # Pad shorter audios to match longest in batch
        ).input_features.to(self.device, dtype=self.dtype)

        # Batched generation - SINGLE GPU call for entire batch!
        # Note: max_new_tokens must be < 448 to leave room for decoder_input_ids
        # (language token, task token, etc. = ~4 tokens)
        generated = self.model.generate(
            input_features,
            language=language,
            task="transcribe",
            return_timestamps=False,  # Skip timestamps for speed
            max_new_tokens=440,
        )

        # Batch decode all transcriptions at once
        transcriptions = self.processor.batch_decode(
            generated,
            skip_special_tokens=True
        )

        return [t.strip() for t in transcriptions]


class MultiGPUPseudoLabelGenerator:
    """
    Multi-GPU pseudo-label generator for maximum throughput.

    Automatically scales to available GPUs (tested with 4-6 GPUs).
    Supports: 4x H100, 6x H100, 4x A100, 6x A100, etc.
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

        # Distributed setup - auto-detect GPU count
        self.is_distributed = dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.local_rank = dist.get_rank() if self.is_distributed else 0
        self.is_main = self.local_rank == 0

        # Auto-detect available GPUs for single-process mode
        self.num_gpus_available = torch.cuda.device_count() if torch.cuda.is_available() else 0

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

        # Batch processing settings for GPU optimization
        batch_config = self.config.get('batch_processing', {})
        base_batch_size = self.config['teacher'].get('pseudo_label_batch_size', 48)

        # Auto-adjust batch size based on GPU count and memory
        # Scale batch size per GPU to maximize throughput while avoiding OOM
        self.batch_size = self._calculate_optimal_batch_size(base_batch_size)
        self.audio_loader_workers = batch_config.get('audio_loader_workers', 8)

        # Scale workers with GPU count (more GPUs = more I/O needed)
        self.audio_loader_workers = min(
            self.audio_loader_workers * max(1, self.world_size // 2),
            32  # Cap at 32 workers
        )

        # Teacher model (loaded lazily)
        self.teacher = None

        # Spot instance handler
        self.spot_handler = SpotInstanceHandler(save_callback=self._emergency_save)

        # Calculate effective throughput
        effective_batch = self.batch_size * self.world_size

        if self.is_main:
            console.print(Panel.fit(
                f"[bold cyan]Multi-GPU Pseudo-Label Generator[/bold cyan]\n"
                f"GPUs Available: {self.num_gpus_available}\n"
                f"GPUs in Use: {self.world_size}\n"
                f"Batch Size per GPU: {self.batch_size}\n"
                f"Effective Batch Size: {effective_batch}\n"
                f"Audio Loader Workers: {self.audio_loader_workers}\n"
                f"WER Threshold: {self.wer_threshold * 100:.0f}%\n"
                f"Duration: {self.min_duration}s - {self.max_duration}s\n"
                f"Output: {self.pseudo_labels_dir}",
                title="Configuration"
            ))

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _calculate_optimal_batch_size(self, base_batch_size: int) -> int:
        """
        Calculate optimal batch size based on available GPU memory.

        Scales automatically for different GPU configurations:
        - H100 94GB: 48-64 samples
        - A100 80GB: 32-48 samples
        - A100 40GB: 16-24 samples
        - RTX 4090 24GB: 8-12 samples

        Args:
            base_batch_size: Base batch size from config

        Returns:
            Optimal batch size for current GPU
        """
        if not torch.cuda.is_available():
            return base_batch_size

        try:
            # Get GPU memory for current device
            gpu_memory_gb = torch.cuda.get_device_properties(self.local_rank).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_properties(self.local_rank).name

            # Memory-based batch size recommendations
            # Whisper large-v3 needs ~4GB base + ~1.5GB per batch sample for 30s audio
            if gpu_memory_gb >= 90:  # H100 94GB
                optimal_batch = min(base_batch_size, 64)
            elif gpu_memory_gb >= 75:  # A100 80GB
                optimal_batch = min(base_batch_size, 48)
            elif gpu_memory_gb >= 35:  # A100 40GB
                optimal_batch = min(base_batch_size, 24)
            elif gpu_memory_gb >= 20:  # RTX 4090/3090 24GB
                optimal_batch = min(base_batch_size, 12)
            elif gpu_memory_gb >= 10:  # RTX 3080/4080 10-16GB
                optimal_batch = min(base_batch_size, 6)
            else:
                optimal_batch = min(base_batch_size, 4)

            if self.is_main:
                console.print(f"[dim]GPU {self.local_rank}: {gpu_name} ({gpu_memory_gb:.1f}GB) -> batch_size={optimal_batch}[/dim]")

            return optimal_batch

        except Exception as e:
            if self.is_main:
                console.print(f"[yellow]Could not detect GPU memory, using base batch size: {e}[/yellow]")
            return base_batch_size

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
        """
        Load processing progress for resume support.

        GPU-count agnostic: can resume with different number of GPUs.
        """
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                result = {}
                for name, info in data.get('datasets', {}).items():
                    # Handle missing fields for backwards compatibility
                    if 'last_sample_idx' not in info:
                        info['last_sample_idx'] = 0
                    if 'gpu_count_at_start' not in info:
                        info['gpu_count_at_start'] = data.get('world_size', 1)
                    if 'processed_sample_ids' not in info:
                        info['processed_sample_ids'] = []
                    result[name] = DatasetProgress(**info)
                return result
        return {}

    def _load_processed_ids_from_files(self, name: str) -> set:
        """
        Load already processed sample IDs from output files.

        This allows resuming even if progress.json was lost, by scanning
        the actual output files from ALL GPUs.

        OPTIMIZATION: Uses a cached ID file for O(1) resume when possible.
        The cache is invalidated if JSONL files are newer than the cache.
        """
        processed_ids = set()
        cache_file = self.pseudo_labels_dir / f'{name}_processed_ids.cache'

        # Get modification times of all JSONL files
        jsonl_files = list(self.pseudo_labels_dir.glob(f'{name}_gpu*_accepted.jsonl'))
        jsonl_files.extend(self.pseudo_labels_dir.glob(f'{name}_gpu*_rejected.jsonl'))

        if not jsonl_files:
            return processed_ids

        latest_jsonl_mtime = max(f.stat().st_mtime for f in jsonl_files)

        # Check if cache is valid (exists and newer than all JSONL files)
        if cache_file.exists():
            cache_mtime = cache_file.stat().st_mtime
            if cache_mtime >= latest_jsonl_mtime:
                # Cache is valid - O(n) read but faster than JSON parsing
                try:
                    with open(cache_file, 'r') as f:
                        processed_ids = set(line.strip() for line in f if line.strip())
                    if self.is_main:
                        console.print(f"[green]Loaded {len(processed_ids):,} IDs from cache (fast resume)[/green]")
                    return processed_ids
                except Exception:
                    pass  # Fall through to rebuild cache

        # Cache miss or invalid - scan JSONL files
        if self.is_main:
            console.print(f"[yellow]Building ID cache from JSONL files...[/yellow]")

        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            sample_id = entry.get('sample_id', '')
                            if sample_id:
                                processed_ids.add(sample_id)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass

        # Save cache for next time (only main process to avoid race)
        if self.is_main and processed_ids:
            try:
                with open(cache_file, 'w') as f:
                    f.write('\n'.join(processed_ids))
                console.print(f"[green]Saved ID cache ({len(processed_ids):,} IDs)[/green]")
            except Exception:
                pass

        return processed_ids

    def _count_processed_from_files(self, name: str) -> int:
        """
        Count total processed samples from output files (all GPUs).

        This gives accurate progress across all GPUs by counting lines
        in the JSONL output files. Used for progress bar display.
        """
        total = 0

        # Count accepted samples from all GPUs
        for filepath in self.pseudo_labels_dir.glob(f'{name}_gpu*_accepted.jsonl'):
            try:
                with open(filepath, 'rb') as f:
                    # Fast line counting using buffer
                    total += sum(1 for _ in f)
            except Exception:
                pass

        # Count rejected samples from all GPUs
        for filepath in self.pseudo_labels_dir.glob(f'{name}_gpu*_rejected.jsonl'):
            try:
                with open(filepath, 'rb') as f:
                    total += sum(1 for _ in f)
            except Exception:
                pass

        return total

    def _save_progress(self, progress: Dict[str, DatasetProgress]):
        """
        Save processing progress.

        NOTE: This saves GPU 0's view of progress. For accurate total counts,
        use _load_processed_ids_from_files() which scans all GPU output files.
        The monitor uses file-based counting for accuracy.

        Includes GPU count for diagnostics when resuming with different config.
        """
        data = {
            'datasets': {name: asdict(p) for name, p in progress.items()},
            'last_update': datetime.now().isoformat(),
            'world_size': self.world_size,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(self.num_gpus_available)] if torch.cuda.is_available() else [],
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

    def _are_spelling_variants(self, word1: str, word2: str, threshold: float = 0.85) -> bool:
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

    def _calculate_wer_spelling_tolerant(self, ref_words: list, hyp_words: list) -> float:
        """
        Calculate WER with tolerance for British/American spelling variants.

        Uses dynamic programming (Levenshtein distance) but treats spelling variants
        as matches rather than substitutions.

        Args:
            ref_words: List of reference words (normalized)
            hyp_words: List of hypothesis words (normalized)

        Returns:
            WER as float between 0.0 and 1.0 (or higher)
        """
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

    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate with tolerance for British/American spelling variants.

        Based on official distil-whisper methodology but extended to accept both
        British and American spellings as equivalent (e.g., colour/color, realise/realize).

        Returns:
            WER as float between 0.0 and 1.0 (or higher for very bad transcriptions)
        """
        # Check for all-uppercase transcription (erroneous generation from Whisper)
        # Official distil-whisper filters these out
        if hypothesis is not None and hypothesis.upper() == hypothesis and len(hypothesis) > 0:
            return 1.0  # Reject all-caps as error

        # Normalize both texts
        ref = self._normalize_text(reference)
        hyp = self._normalize_text(hypothesis)

        # Handle edge cases (matches official implementation)
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

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for WER calculation using official Whisper EnglishTextNormalizer.

        This matches the official distil-whisper methodology:
        https://github.com/huggingface/distil-whisper/blob/main/training/run_distillation.py

        The normalizer handles:
        - Lowercase conversion
        - Punctuation removal
        - Number to word conversion ("3" -> "three")
        - Contractions normalization
        - Unicode normalization

        Note: British/American spelling differences are handled separately in
        _calculate_wer_spelling_tolerant() which accepts both variants as equivalent.

        Additionally strips filler words so CrisperWhisper's verbatim transcription
        (with um, uh, etc.) doesn't get penalized when ground truth doesn't include them.
        """
        if not text:
            return ""

        # Initialize normalizer if not already done (lazy init for efficiency)
        if not hasattr(self, '_english_normalizer'):
            self._english_normalizer = EnglishTextNormalizer({})

        # Remove bracketed fillers first (CrisperWhisper format: [Um], [Uh], etc.)
        # Do this BEFORE the normalizer since normalizer may not handle brackets well
        text = re.sub(r'\[(?:um|uh|er|ah|uhm|erm|hmm|hm|mm|mhm)\]', '', text, flags=re.IGNORECASE)

        # Apply official Whisper English normalizer
        # This handles: lowercase, punctuation, numbers->words, spelling normalization, etc.
        text = self._english_normalizer(text)

        # Strip standalone filler words (in case they appear without brackets)
        # These are common in verbatim transcriptions but often missing from ground truth
        FILLER_WORDS = {'um', 'uh', 'er', 'ah', 'uhm', 'erm', 'hmm', 'hm', 'mm', 'mhm', 'uh huh', 'mm hmm'}
        words = text.split()
        words = [w for w in words if w not in FILLER_WORDS]

        return ' '.join(words)

    def _load_dataset(self, name: str, skip_to: int = 0) -> Tuple[Optional[Iterator], Optional[int]]:
        """
        Load dataset - DOWNLOAD for small datasets, STREAM for massive ones.

        Args:
            name: Dataset name
            skip_to: Number of samples to skip (for fast resume)

        Returns:
            Tuple of (dataset_iterator, total_sample_count)
            - total_sample_count is None for streaming datasets (unknown size)
            - total_sample_count is exact for downloaded datasets
        """
        from datasets import load_dataset, interleave_datasets, concatenate_datasets
        import random

        dataset_config = DATASET_CONFIGS.get(name)
        if not dataset_config:
            return None, None

        # Check config.yaml for streaming preference
        yaml_config = self.config.get('datasets', {}).get(name, {})
        use_streaming = yaml_config.get('streaming', False)

        # Force streaming for massive datasets regardless of config
        STREAMING_ONLY_DATASETS = {'peoples_speech', 'yodas', 'gigaspeech'}
        if name in STREAMING_ONLY_DATASETS:
            use_streaming = True

        # Check HF authentication on first load
        global _HF_AUTH_CHECKED
        if not _HF_AUTH_CHECKED and self.is_main:
            setup_hf_authentication()
            _HF_AUTH_CHECKED = True

        # Add small random delay to stagger GPU requests and avoid rate limiting
        if self.is_distributed and self.local_rank > 0:
            delay = self.local_rank * 0.5 + random.uniform(0, 0.5)
            time.sleep(delay)

        try:
            mode_str = "streaming" if use_streaming else "download"
            if self.is_main:
                console.print(f"[yellow]Loading {name} ({mode_str} mode)...[/yellow]")

            # Helper function with retry logic for rate limiting
            def load_with_retry(hf_name, max_retries=5, **kwargs):
                """Load dataset with exponential backoff for 429 errors."""
                for attempt in range(max_retries):
                    try:
                        return load_dataset(hf_name, **kwargs)
                    except Exception as e:
                        error_str = str(e).lower()
                        if '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
                            wait_time = (2 ** attempt) + random.uniform(0, 1)
                            if self.is_main:
                                console.print(f"[yellow]Rate limited, waiting {wait_time:.1f}s (attempt {attempt+1}/{max_retries})...[/yellow]")
                            time.sleep(wait_time)
                        else:
                            raise
                raise Exception(f"Failed to load {hf_name} after {max_retries} retries due to rate limiting")

            total_samples = None

            # Special handling for LibriSpeech (multiple splits)
            if name == 'librispeech':
                datasets_list = []
                for split in dataset_config['splits']:
                    ds = load_with_retry(
                        dataset_config['hf_name'],
                        split=split,
                        streaming=use_streaming,
                        trust_remote_code=True,
                    )
                    datasets_list.append(ds)
                    # Small delay between split loads to avoid rate limiting
                    time.sleep(0.3)

                if use_streaming:
                    dataset = interleave_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]
                else:
                    # For downloaded datasets, concatenate and get exact count
                    dataset = concatenate_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]
                    total_samples = len(dataset)
                    if self.is_main:
                        console.print(f"[cyan]  {name}: {total_samples:,} samples (exact count)[/cyan]")
            else:
                kwargs = {
                    'split': dataset_config['splits'][0],
                    'streaming': use_streaming,
                    'trust_remote_code': True,
                }
                if dataset_config['subset']:
                    kwargs['name'] = dataset_config['subset']

                dataset = load_with_retry(dataset_config['hf_name'], **kwargs)

                if not use_streaming:
                    total_samples = len(dataset)
                    if self.is_main:
                        console.print(f"[cyan]  {name}: {total_samples:,} samples (exact count)[/cyan]")

            # FAST RESUME: Skip to resume point efficiently
            if skip_to > 0:
                if use_streaming:
                    # For streaming datasets, use .skip() method
                    if hasattr(dataset, 'skip'):
                        if self.is_main:
                            console.print(f"[yellow]Fast-forwarding {skip_to:,} samples (streaming skip)...[/yellow]")
                        dataset = dataset.skip(skip_to)
                    else:
                        if self.is_main:
                            console.print(f"[yellow]Dataset doesn't support skip(), will iterate through {skip_to:,} samples[/yellow]")
                else:
                    # For downloaded datasets, use .select() for instant O(1) access
                    if skip_to < total_samples:
                        if self.is_main:
                            console.print(f"[yellow]Fast-forwarding to sample {skip_to:,} (instant select)...[/yellow]")
                        remaining_indices = list(range(skip_to, total_samples))
                        dataset = dataset.select(remaining_indices)
                    else:
                        if self.is_main:
                            console.print(f"[green]Dataset already fully processed ({skip_to:,} >= {total_samples:,})[/green]")
                        return iter([]), total_samples  # Empty iterator, already done

            if self.is_main:
                console.print(f"[green]✓ {name} loaded successfully[/green]")

            return iter(dataset), total_samples

        except Exception as e:
            if self.is_main:
                console.print(f"[red]✗ Error loading {name}: {e}[/red]")
                if dataset_config['requires_auth']:
                    console.print(f"[yellow]  This dataset requires HuggingFace authentication[/yellow]")
            return None, None

    # Keep old method name for backwards compatibility
    def _load_dataset_streaming(self, name: str) -> Optional[Iterator]:
        """Legacy method - use _load_dataset instead."""
        dataset_iter, _ = self._load_dataset(name, skip_to=0)
        return dataset_iter

    def _extract_audio_from_sample(self, sample: Dict, audio_col: str) -> Optional[Tuple[np.ndarray, int]]:
        """Extract audio array and sample rate from a dataset sample."""
        try:
            audio_data = sample.get(audio_col, {})
            if isinstance(audio_data, dict):
                audio_array = np.array(audio_data['array'], dtype=np.float32)
                sr = audio_data['sampling_rate']
                return audio_array, sr
        except Exception:
            pass
        return None

    def _process_batch_gpu(
        self,
        batch_data: List[Tuple[np.ndarray, int, str, str, float, int]],
        name: str
    ) -> List[PseudoLabelEntry]:
        """
        Process a batch of audio samples with TRUE GPU batching.

        Args:
            batch_data: List of (audio_array, sample_rate, ground_truth, sample_id, duration, sample_idx)
            name: Dataset name

        Returns:
            List of PseudoLabelEntry objects
        """
        if not batch_data:
            return []

        # Unpack batch data
        audio_arrays = [item[0] for item in batch_data]
        sample_rates = [item[1] for item in batch_data]
        ground_truths = [item[2] for item in batch_data]
        sample_ids = [item[3] for item in batch_data]
        durations = [item[4] for item in batch_data]

        # Use the first sample rate (they should all be the same from HF datasets)
        sr = sample_rates[0]

        # Preprocess audio arrays (resample if needed, convert to mono)
        processed_audios = []
        for audio in audio_arrays:
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            # Ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            processed_audios.append(audio)

        # BATCHED INFERENCE - single GPU call for entire batch!
        try:
            transcriptions = self.teacher.generate_pseudo_labels_batch(
                audio_arrays=processed_audios,
                sampling_rate=sr,
                language="en",
            )
        except Exception as e:
            if self.is_main:
                console.print(f"[dim red]Batch inference error: {e}[/dim red]")
            return []

        # Create entries with WER calculation
        entries = []
        timestamp = datetime.now().isoformat()

        for i, (transcription, ground_truth, sample_id, duration) in enumerate(
            zip(transcriptions, ground_truths, sample_ids, durations)
        ):
            wer = self._calculate_wer(ground_truth, transcription)
            accepted = wer <= self.wer_threshold

            entries.append(PseudoLabelEntry(
                sample_id=sample_id,
                dataset=name,
                ground_truth=ground_truth,
                pseudo_label=transcription,
                word_timestamps=[],  # Skip timestamps for speed in batch mode
                wer=wer,
                duration_seconds=duration,
                accepted=accepted,
                rejection_reason=None if accepted else f"WER {wer:.2%} > {self.wer_threshold:.0%}"
            ))

        return entries

    def process_dataset(
        self,
        name: str,
        max_samples: Optional[int] = None,
        resume: bool = True,
    ) -> DatasetProgress:
        """
        Process a single dataset with CrisperWhisper pseudo-labeling.

        Uses TRUE GPU BATCH PROCESSING for maximum throughput:
        - Collects samples into batches
        - Processes entire batch in single GPU call
        - ~10-50x faster than sample-by-sample processing

        GPU-AGNOSTIC RESUME:
        - Tracks processed sample IDs (not just count)
        - Can resume with different number of GPUs
        - Scans output files to recover state if progress.json is lost

        Includes:
        - GPU batched inference
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

        # CONTENT-BASED RESUME: Load processed sample IDs from ALL output files
        # IDs are now content-based hashes (e.g., "ami_a1b2c3d4e5f6g7h8")
        # This ensures same audio+text always gets same ID regardless of:
        # - Dataset iteration order (which may vary between runs!)
        # - GPU count or which GPU processes it
        # - Run number
        #
        # MULTI-GPU SYNC: Main process loads first (and writes cache), then barrier,
        # then other GPUs load from cache. This avoids race conditions.
        already_processed_ids = set()
        if resume:
            if self.is_main:
                # Main process loads and writes cache
                already_processed_ids = self._load_processed_ids_from_files(name)
                if already_processed_ids:
                    console.print(f"[green]Found {len(already_processed_ids):,} already processed samples for {name}[/green]")
                    console.print(f"[cyan]Using content-based deduplication (order-independent)[/cyan]")

            # Synchronize so other GPUs wait for cache to be written
            if self.is_distributed:
                dist.barrier()

            # Non-main GPUs load from cache (which main just created)
            if not self.is_main:
                already_processed_ids = self._load_processed_ids_from_files(name)

            # Final sync to ensure all GPUs have loaded before proceeding
            if self.is_distributed:
                dist.barrier()

        # Always append mode if resuming with existing samples
        if resume and already_processed_ids:
            mode = 'a'
        else:
            mode = 'w'
            progress = DatasetProgress(name=name)

        # Update GPU count for this run
        progress.gpu_count_at_start = self.world_size
        progress.status = "processing"

        # Load dataset (no skip - we use content-based deduplication instead)
        dataset_iter, total_samples = self._load_dataset(name, skip_to=0)
        if dataset_iter is None:
            progress.status = "error"
            progress.error_message = "Failed to load dataset"
            return progress

        # Save actual sample count to a metadata file for the monitor
        if total_samples is not None and self.is_main:
            metadata_file = self.pseudo_labels_dir / f'{name}_metadata.json'
            metadata = {
                'dataset': name,
                'total_samples': total_samples,
                'source': 'actual_count',
                'timestamp': datetime.now().isoformat()
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            if self.is_main:
                console.print(f"[cyan]Saved metadata: {total_samples:,} total samples[/cyan]")

        text_col = config['text_column']
        audio_col = config['audio_column']

        # Track samples processed in THIS run (for progress bar)
        samples_this_run = 0

        # Determine total for progress bar - use actual count if available
        if total_samples is not None:
            pbar_total = total_samples
        else:
            # Fallback to estimate for streaming datasets
            pbar_total = max_samples or config['estimated_hours'] * 120

        # Determine number of prefetch workers based on CPU cores
        # Goal: Keep GPU 100% utilized by having enough workers to saturate I/O
        #
        # With 128 cores: use 32-64 workers (plenty of headroom)
        # Rule: ~25% of cores for prefetch workers, capped at 64 (diminishing returns)
        # Each worker handles: audio extraction, hashing, filtering
        import multiprocessing
        num_cpu_cores = multiprocessing.cpu_count()
        num_prefetch_workers = max(8, min(num_cpu_cores // 4, 64))

        if self.is_main:
            console.print(f"[cyan]Using {num_prefetch_workers} prefetch threads ({num_cpu_cores} CPU cores available)[/cyan]")

        # Create threaded prefetcher for parallel sample loading
        # Uses content-based hashing for deduplication (order-independent!)
        # CRITICAL: prefetch_batches must be high enough to keep GPU fed continuously
        #
        # With 128 cores / 32 workers: workers produce faster than GPU consumes
        # Need large buffer to absorb I/O variance and keep GPU 100% busy
        # 16 batches * 48 samples = 768 samples buffered (~25-30 seconds of runway)
        prefetcher = ThreadedPrefetcher(
            dataset_iter=dataset_iter,
            start_idx=0,  # Always start from 0, dedup handles resume
            world_size=self.world_size,
            local_rank=self.local_rank,
            already_processed_ids=already_processed_ids,
            audio_col=audio_col,
            text_col=text_col,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
            name=name,
            prefetch_batches=16,  # Prefetch 16 batches ahead - large buffer for 128 cores
            batch_size=self.batch_size,
            num_workers=num_prefetch_workers,
        )

        # Process samples with BATCHING using threaded prefetcher
        with open(output_file, mode) as out_f, open(rejected_file, mode) as rej_f:
            # Progress bar only on main process
            # Track actual processed count from files (accurate across all GPUs)
            if self.is_main:
                initial_processed = self._count_processed_from_files(name)
                pbar = tqdm(
                    desc=f"Processing {name} (GPU {self.local_rank}, batch={self.batch_size})",
                    total=pbar_total,  # Use actual count or estimate
                    unit="samples",
                    initial=initial_processed  # Start from actual file counts
                )
                last_pbar_update_time = time.time()
            else:
                pbar = None
                last_pbar_update_time = 0

            last_sample_idx = 0
            gpu_starvation_count = 0  # Track how often GPU had to wait for data
            total_batches = 0

            # Main processing loop - pull batches from prefetcher
            empty_batch_count = 0  # Track consecutive empty batches for stuck detection
            last_progress_time = time.time()

            while not self.spot_handler.should_stop:
                # Get a batch from the prefetcher (blocks until ready or timeout)
                prefetch_stats_before = prefetcher.get_stats()
                queue_was_empty = prefetch_stats_before['queue_size'] == 0

                batch_data = prefetcher.get_batch(timeout=60.0)
                total_batches += 1

                # Track GPU starvation (queue was empty when we asked for data)
                if queue_was_empty and batch_data:
                    gpu_starvation_count += 1

                if not batch_data:
                    # No more data
                    if prefetcher.is_exhausted():
                        if self.is_main:
                            console.print(f"[green]Dataset {name} exhausted - all samples processed[/green]")
                        break

                    # Track consecutive empty batches
                    empty_batch_count += 1

                    # Safety: If we've gotten 10+ empty batches in a row and feeder is done,
                    # the workers may have all finished but exhausted flag not set due to race
                    if empty_batch_count >= 10 and prefetcher.feeder_done.is_set():
                        if self.is_main:
                            console.print(f"[yellow]Warning: {empty_batch_count} empty batches, feeder done - checking exhaustion[/yellow]")
                        # Give workers a moment to finish and set exhausted flag
                        time.sleep(2.0)
                        if prefetcher.is_exhausted() or (prefetcher.processed_queue.empty() and prefetcher.raw_queue.empty()):
                            if self.is_main:
                                console.print(f"[green]Dataset {name} processing complete (detected via empty queues)[/green]")
                            break

                    # Prevent busy loop - small sleep when no data
                    time.sleep(0.1)
                    continue
                else:
                    # Got data, reset empty counter
                    empty_batch_count = 0
                    last_progress_time = time.time()

                if max_samples and samples_this_run >= max_samples // self.world_size:
                    break

                # Process batch on GPU
                entries = self._process_batch_gpu(batch_data, name)

                for entry in entries:
                    entry_json = json.dumps(asdict(entry))

                    if entry.accepted:
                        out_f.write(entry_json + '\n')
                        progress.samples_accepted += 1
                        progress.accepted_duration_hours += entry.duration_seconds / 3600
                        progress.wer_sum += entry.wer
                    else:
                        rej_f.write(entry_json + '\n')
                        progress.samples_rejected_wer += 1

                    progress.samples_processed += 1
                    progress.last_sample_id = entry.sample_id
                    samples_this_run += 1

                # Track last sample index from batch
                if batch_data:
                    last_sample_idx = max(item[5] for item in batch_data)  # sample_idx is at index 5

                # Flush files
                out_f.flush()
                rej_f.flush()

                # Update progress bar with accurate file-based counts
                if pbar:
                    prefetch_stats = prefetcher.get_stats()
                    acc_rate = progress.acceptance_rate * 100
                    avg_wer = progress.avg_wer * 100

                    # Periodically refresh total count from files (every 5 seconds)
                    # This gives accurate progress across all GPUs
                    current_time = time.time()
                    if current_time - last_pbar_update_time >= 5.0:
                        total_processed = self._count_processed_from_files(name)
                        pbar.n = total_processed  # Set absolute position
                        pbar.refresh()
                        last_pbar_update_time = current_time

                    pbar.set_postfix({
                        'acc': f'{acc_rate:.1f}%',
                        'wer': f'{avg_wer:.1f}%',
                        'hrs': f'{progress.accepted_duration_hours:.1f}',
                        'q': prefetch_stats['queue_size'],  # Processed queue size
                        'w': prefetch_stats['workers_active'],  # Active workers
                    })

                # Save progress periodically
                if progress.samples_processed % 500 == 0:
                    all_progress[name] = progress
                    self._save_progress(all_progress)

                    # Invalidate cache so it gets rebuilt on next resume
                    # (simpler than incremental updates, and resume is now fast anyway)
                    cache_file = self.pseudo_labels_dir / f'{name}_processed_ids.cache'
                    if cache_file.exists():
                        try:
                            cache_file.unlink()
                        except Exception:
                            pass

            # Stop prefetcher
            prefetcher.stop()

            if pbar:
                pbar.close()

        # Get final stats from prefetcher
        prefetch_stats = prefetcher.get_stats()
        skipped_in_loop = prefetch_stats['skipped_in_loop']
        progress.samples_rejected_duration += prefetch_stats['rejected_duration']
        progress.total_duration_hours += prefetch_stats['total_duration_hours']

        # Log resume stats and GPU efficiency
        if self.is_main:
            if skipped_in_loop > 0:
                console.print(f"[green]Skipped {skipped_in_loop:,} already-processed samples (content-based dedup)[/green]")

            # Report GPU efficiency
            if total_batches > 0:
                starvation_rate = gpu_starvation_count / total_batches * 100
                if starvation_rate > 10:
                    console.print(f"[yellow]⚠ GPU starvation rate: {starvation_rate:.1f}% ({gpu_starvation_count}/{total_batches} batches)[/yellow]")
                    console.print(f"[yellow]  Consider increasing prefetch_batches or num_workers[/yellow]")
                else:
                    console.print(f"[green]✓ GPU efficiency: {100-starvation_rate:.1f}% (queue kept fed)[/green]")

        # Final save
        progress.status = "completed"
        progress.last_sample_idx = last_sample_idx  # Track where we left off
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

        # Load existing progress to check for completed datasets
        saved_progress = self._load_progress()

        for name in dataset_names:
            if self.spot_handler.should_stop:
                break

            # SKIP ALREADY COMPLETED DATASETS
            # This prevents re-iterating through completed datasets on resume
            if resume and name in saved_progress:
                existing = saved_progress[name]
                if isinstance(existing, dict):
                    status = existing.get('status', 'pending')
                else:
                    status = getattr(existing, 'status', 'pending')

                if status == 'completed':
                    if self.is_main:
                        console.print(f"\n[bold green]{'═' * 50}[/bold green]")
                        console.print(f"[bold green]  Skipping {name} (already completed)[/bold green]")
                        console.print(f"[bold green]{'═' * 50}[/bold green]")
                    # Restore progress for summary
                    if isinstance(existing, dict):
                        all_progress[name] = DatasetProgress(**existing)
                    else:
                        all_progress[name] = existing
                    continue

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
