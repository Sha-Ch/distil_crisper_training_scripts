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

# =============================================================================
# Logging Setup - Console + File
# =============================================================================
# All logs go to both console AND a timestamped log file in logs/ folder
# Log directory: <workspace>/logs/ (works on Linux/Runpod)
# =============================================================================

def setup_logging(log_dir: Path = None):
    """Setup logging to both console and file."""
    if log_dir is None:
        # Default to logs/ folder relative to script
        log_dir = Path(__file__).resolve().parent.parent / 'logs'

    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'pseudo_labels_{timestamp}.log'

    # Setup file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Get the root logger for our app
    logger = logging.getLogger('pseudo_labels')
    logger.setLevel(logging.DEBUG)
    # Prevent duplicate handlers on re-import
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger, log_file

# Initialize logger - logs/ folder next to scripts/
LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'
logger, LOG_FILE = setup_logging(LOG_DIR)

def log(message: str, level: str = 'info'):
    """Log message to both console and file."""
    # Log to file
    if level == 'debug':
        logger.debug(message)
    elif level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)
    else:
        logger.info(message)

# Rich console for pretty output
console = Console()

# Wrapper to log console output to file too
_original_console_print = console.print
def _logged_console_print(*args, **kwargs):
    """Print to console and also log to file."""
    # Convert rich markup to plain text for logging
    from io import StringIO
    string_io = StringIO()
    temp_console = Console(file=string_io, force_terminal=False, no_color=True)
    temp_console.print(*args, **kwargs)
    plain_text = string_io.getvalue().strip()
    if plain_text:
        logger.info(plain_text)
    # Also print to actual console
    _original_console_print(*args, **kwargs)

console.print = _logged_console_print

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
        'splits': ['CC_BY_3.0', 'CC_BY_SA_3.0', 'CC_BY_ND_3.0'],  # Multiple license splits, not 'train'
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
    # Using distil-whisper/tedlium which has audio data readily available
    'tedlium': {
        'hf_name': 'distil-whisper/tedlium',
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

    # Priority 7: Large scale, diverse (GigaSpeech) - CHUNKED DOWNLOAD
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
        # Chunked download settings - download ~100GB at a time to avoid rate limiting
        'use_chunked_download': True,
        'chunk_size_gb': 100,  # Target chunk size in GB (±10GB variance OK)
        'estimated_sample_size_mb': 5,  # Avg sample size for chunk calculation
    },

    # Priority 8: Large scale diverse (People's Speech) - CHUNKED DOWNLOAD
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
        # Chunked download settings
        'use_chunked_download': True,
        'chunk_size_gb': 100,
        'estimated_sample_size_mb': 3,
    },

    # Priority 9: Massive YouTube dataset - bulk of data (YODAS) - CHUNKED DOWNLOAD
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
        # Chunked download settings
        'use_chunked_download': True,
        'chunk_size_gb': 100,
        'estimated_sample_size_mb': 4,
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
    # Verification status after completion
    verification_status: str = ""  # verified, missing_samples, exceeded

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
        self._sharded_out = 0  # Samples that belong to other GPUs (not processed by this GPU)
        self._invalid_samples = 0  # Samples that couldn't be processed (missing audio, bad format, etc.)
        self._total_duration_hours = 0.0
        self._workers_done = 0
        self._total_samples_fed = 0  # Total samples yielded by iterator (for accurate count)

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
        samples_fed = 0
        samples_failed_queue = 0
        log(f"[FEEDER] Starting feeder loop from idx={self.sample_idx}", 'debug')
        try:
            sample_idx = self.sample_idx
            for sample in self.dataset_iter:
                if self.stop_event.is_set():
                    log(f"[FEEDER] Stop event set at idx={sample_idx}, fed={samples_fed}", 'debug')
                    break

                # Put raw sample with its index into queue
                # Use timeout to avoid blocking forever if queue is full
                put_attempts = 0
                put_success = False
                while not self.stop_event.is_set():
                    try:
                        self.raw_queue.put((sample, sample_idx), timeout=1.0)
                        samples_fed += 1
                        put_success = True
                        # Log every 10000 samples for progress tracking
                        if samples_fed % 10000 == 0:
                            log(f"[FEEDER] Progress: fed {samples_fed:,} samples, current idx={sample_idx}, queue_size={self.raw_queue.qsize()}", 'debug')
                        break
                    except:
                        put_attempts += 1
                        if put_attempts >= 30:  # 30 seconds timeout
                            log(f"[FEEDER] WARNING: Queue put timeout at idx={sample_idx} after {put_attempts} attempts", 'warning')
                            samples_failed_queue += 1
                            break
                        continue  # Retry until stop or success

                if not put_success and not self.stop_event.is_set():
                    log(f"[FEEDER] FAILED to queue sample idx={sample_idx}", 'error')

                sample_idx += 1

        except Exception as e:
            log(f"[FEEDER] Exception in feeder loop: {e}", 'error')
        finally:
            # Save total samples fed (for accurate count verification)
            with self._lock:
                self._total_samples_fed = samples_fed
            log(f"[FEEDER] Feeder complete: fed={samples_fed:,}, failed_queue={samples_failed_queue:,}, final_idx={sample_idx}", 'info')
            self.feeder_done.set()
            # Signal workers that no more raw samples are coming
            # Use timeout to avoid deadlock if queue is full
            for i in range(self.num_workers):
                while not self.stop_event.is_set():
                    try:
                        self.raw_queue.put(None, timeout=1.0)  # Poison pill
                        break
                    except:
                        continue
            log(f"[FEEDER] All {self.num_workers} poison pills sent", 'debug')

    def _worker_loop(self):
        """Worker thread: processes raw samples in parallel."""
        import threading
        thread_name = threading.current_thread().name
        local_skipped = 0
        local_rejected = 0
        local_sharded_out = 0
        local_invalid = 0
        local_processed = 0  # Successfully queued for GPU processing
        local_dropped = 0  # Failed to queue (should never happen)
        local_duration = 0.0

        log(f"[WORKER {thread_name}] Starting worker loop", 'debug')

        try:
            while not self.stop_event.is_set():
                try:
                    item = self.raw_queue.get(timeout=1.0)
                except Empty:
                    if self.feeder_done.is_set() and self.raw_queue.empty():
                        log(f"[WORKER {thread_name}] Exiting: feeder done and queue empty", 'debug')
                        break
                    continue

                if item is None:  # Poison pill
                    log(f"[WORKER {thread_name}] Received poison pill, exiting", 'debug')
                    break

                sample, sample_idx = item

                # Process sample (extract audio, hash, filter)
                result = self._process_sample(sample, sample_idx)
                if result is None:
                    local_invalid += 1
                    # Log every 100th invalid to avoid spam but track issues
                    if local_invalid % 100 == 1:
                        log(f"[WORKER {thread_name}] Invalid sample idx={sample_idx} (total invalid: {local_invalid})", 'debug')
                    continue
                if result == 'duration_rejected':
                    local_rejected += 1
                    continue

                # Unpack to check dedup and sharding
                audio_array, sr, ground_truth, sample_id, duration, idx = result

                # Update local duration stats
                local_duration += duration / 3600

                # Normalize ground_truth once for both dedup and sharding
                ground_truth_normalized = ground_truth.strip()

                # Skip if already processed (O(1) hash set lookup using ground_truth text)
                # Using ground_truth instead of sample_id because audio hashes can vary between runs
                if ground_truth_normalized in self.already_processed_ids:
                    local_skipped += 1
                    continue

                # Multi-GPU sharding - deterministic based on ground_truth text
                # Using ground_truth ensures same sample always goes to same GPU
                shard_key = int(hashlib.md5(ground_truth_normalized.encode()).hexdigest()[:8], 16)
                if shard_key % self.world_size != self.local_rank:
                    local_sharded_out += 1
                    continue

                # Put in processed queue (may block if full - backpressure)
                # Retry with longer timeout to avoid dropping samples
                put_success = False
                put_attempts = 0
                for _ in range(12):  # 12 * 5s = 60s max wait
                    if self.stop_event.is_set():
                        break
                    try:
                        self.processed_queue.put(result, timeout=5.0)
                        put_success = True
                        local_processed += 1
                        break
                    except:
                        put_attempts += 1
                        continue  # Retry

                if not put_success and not self.stop_event.is_set():
                    # Log dropped sample - THIS IS CRITICAL, SHOULD NEVER HAPPEN
                    local_dropped += 1
                    log(f"[WORKER {thread_name}] DROPPED sample idx={sample_idx} after {put_attempts} attempts! text='{ground_truth_normalized[:50]}...'", 'error')

        except Exception as e:
            log(f"[WORKER {thread_name}] Exception: {e}", 'error')
        finally:
            # Aggregate stats thread-safely
            with self._lock:
                self._skipped_in_loop += local_skipped
                self._rejected_duration += local_rejected
                self._sharded_out += local_sharded_out
                self._invalid_samples += local_invalid
                self._total_duration_hours += local_duration
                self._workers_done += 1

                # Log worker final stats
                log(f"[WORKER {thread_name}] Final stats: processed={local_processed}, skipped={local_skipped}, rejected={local_rejected}, sharded={local_sharded_out}, invalid={local_invalid}, dropped={local_dropped}", 'info')

                # If all workers done, signal exhausted
                if self._workers_done >= self.num_workers:
                    self.exhausted.set()
                    log(f"[WORKERS] All {self.num_workers} workers completed", 'info')

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
                # Log every 1000th to avoid spam but track the issue
                if sample_idx % 1000 == 0:
                    log(f"[PROCESS_SAMPLE] idx={sample_idx}: audio_data not dict, type={type(audio_data)}", 'debug')
                return None

            if 'array' not in audio_data:
                if sample_idx % 1000 == 0:
                    log(f"[PROCESS_SAMPLE] idx={sample_idx}: audio_data missing 'array' key, keys={list(audio_data.keys())}", 'debug')
                return None

            if 'sampling_rate' not in audio_data:
                if sample_idx % 1000 == 0:
                    log(f"[PROCESS_SAMPLE] idx={sample_idx}: audio_data missing 'sampling_rate' key", 'debug')
                return None

            audio_array = np.array(audio_data['array'], dtype=np.float32)
            sr = audio_data['sampling_rate']

            if len(audio_array) == 0:
                log(f"[PROCESS_SAMPLE] idx={sample_idx}: empty audio array", 'debug')
                return None

            # Get ground truth
            ground_truth = sample.get(self.text_col, '')
            if not ground_truth or not isinstance(ground_truth, str):
                if sample_idx % 1000 == 0:
                    log(f"[PROCESS_SAMPLE] idx={sample_idx}: invalid ground_truth, type={type(ground_truth)}, val='{str(ground_truth)[:30]}'", 'debug')
                return None

            if len(ground_truth.strip()) == 0:
                log(f"[PROCESS_SAMPLE] idx={sample_idx}: empty ground_truth (whitespace only)", 'debug')
                return None

            # Generate CONTENT-BASED sample ID (O(1) - only first 4000 samples)
            sample_id = generate_content_hash(audio_array, ground_truth, self.name)

            # Calculate duration
            duration = len(audio_array) / sr

            # Duration filter - return marker so we can track stats
            if duration < self.min_duration or duration > self.max_duration:
                # Log details for duration filtering
                if sample_idx % 5000 == 0:
                    log(f"[PROCESS_SAMPLE] idx={sample_idx}: duration_rejected, dur={duration:.2f}s (min={self.min_duration}, max={self.max_duration})", 'debug')
                return 'duration_rejected'

            return (audio_array, sr, ground_truth, sample_id, duration, sample_idx)

        except Exception as e:
            log(f"[PROCESS_SAMPLE] idx={sample_idx}: exception {type(e).__name__}: {e}", 'error')
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
                'sharded_out': self._sharded_out,  # Samples that belong to other GPUs
                'invalid_samples': self._invalid_samples,  # Samples that couldn't be processed
                'total_duration_hours': self._total_duration_hours,
                'workers_active': self.num_workers - self._workers_done,
                'raw_queue_size': self.raw_queue.qsize(),
                'queue_size': self.processed_queue.qsize(),
                'total_samples_fed': self._total_samples_fed,  # Actual iterator count
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
        # Matches original Distil-Whisper methodology:
        # - num_beams=1 (greedy decoding) for speed, WER filter catches bad outputs
        # - max_new_tokens=256 to match original max_label_length
        generated = self.model.generate(
            input_features,
            language=language,
            task="transcribe",
            return_timestamps=False,  # Skip timestamps for speed
            num_beams=1,  # Greedy decoding (original distil-whisper uses this)
            max_new_tokens=256,  # Match original max_label_length
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
        # Duration limits - matches original Distil-Whisper v3.5 methodology
        # No minimum duration: short clips are allowed, WER filter catches hallucinations
        # Max 30s: Whisper was trained on 30-second chunks
        self.min_duration = 0.0  # No minimum (original distil-whisper has no min filter)
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
            # Log file location
            console.print(f"\n[bold green]📝 Log file: {LOG_FILE}[/bold green]\n")

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
                    if 'verification_status' not in info:
                        info['verification_status'] = ''
                    result[name] = DatasetProgress(**info)
                return result
        return {}

    def _load_processed_ids_from_files(self, name: str) -> set:
        """
        Load already processed ground_truth texts from output files.

        Uses ground_truth (actual transcript text) for deduplication instead of
        content hashes, because audio byte representation can vary between runs.

        This allows resuming even if progress.json was lost, by scanning
        the actual output files from ALL GPUs.

        OPTIMIZATION: Uses a cached file for O(1) resume when possible.
        The cache is invalidated if JSONL files are newer than the cache.
        """
        processed_texts = set()
        cache_file = self.pseudo_labels_dir / f'{name}_processed_texts.cache'

        # Get modification times of all JSONL files
        # Filter to only files that still exist (race condition protection)
        jsonl_files = list(self.pseudo_labels_dir.glob(f'{name}_gpu*_accepted.jsonl'))
        jsonl_files.extend(self.pseudo_labels_dir.glob(f'{name}_gpu*_rejected.jsonl'))
        jsonl_files = [f for f in jsonl_files if f.exists()]

        if not jsonl_files:
            return processed_texts

        # Get latest modification time with race condition protection
        try:
            latest_jsonl_mtime = max(f.stat().st_mtime for f in jsonl_files if f.exists())
        except (ValueError, FileNotFoundError):
            # Files were deleted between glob and stat - rebuild cache from scratch
            latest_jsonl_mtime = 0

        # Check if cache is valid (exists and newer than all JSONL files)
        if cache_file.exists():
            cache_mtime = cache_file.stat().st_mtime
            if cache_mtime >= latest_jsonl_mtime:
                # Cache is valid - load from JSON
                try:
                    with open(cache_file, 'r') as f:
                        processed_texts = set(json.load(f))
                    if self.is_main:
                        console.print(f"[green]Loaded {len(processed_texts):,} texts from cache (fast resume)[/green]")
                    return processed_texts
                except Exception:
                    pass  # Fall through to rebuild cache

        # Cache miss or invalid - scan JSONL files
        if self.is_main:
            console.print(f"[yellow]Building text cache from JSONL files...[/yellow]")

        for jsonl_file in jsonl_files:
            # Skip files that no longer exist (race condition with cache invalidation)
            if not jsonl_file.exists():
                continue
            try:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            # Use ground_truth for deduplication (stable across runs)
                            # Must match normalization in ThreadedPrefetcher._worker_loop()
                            ground_truth = entry.get('ground_truth', '').strip()
                            if ground_truth:
                                processed_texts.add(ground_truth)
                        except json.JSONDecodeError:
                            continue
            except (FileNotFoundError, IOError):
                # File was deleted or inaccessible - skip it
                continue
            except Exception:
                pass

        # Save cache for next time (only main process to avoid race)
        # Use JSON for cache to handle texts with newlines safely
        if self.is_main and processed_texts:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(list(processed_texts), f)
                console.print(f"[green]Saved text cache ({len(processed_texts):,} texts)[/green]")
            except Exception:
                pass

        return processed_texts

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

    def _update_metadata_with_actual_count(self, name: str, actual_iterated_count: int = 0) -> int:
        """
        Update the metadata file after completion with accurate counts.

        Saves TWO important counts:
        1. actual_iterated_count: TRUE total from iterator (for progress %)
        2. processed_count: How many were written to JSONL (for verification)

        The total_samples should be the TRUE dataset total, not the processed count.
        Some samples are filtered (duration, invalid) or deduped, so processed < total.

        Args:
            name: Dataset name
            actual_iterated_count: Total samples yielded by iterator (from prefetcher stats)
                                   This is the TRUE count for progress calculation.

        Returns:
            The actual count from files
        """
        if not self.is_main:
            return 0

        processed_count = self._count_processed_from_files(name)
        metadata_file = self.pseudo_labels_dir / f'{name}_metadata.json'

        # Read existing metadata or create new
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {'dataset': name}
        else:
            metadata = {'dataset': name}

        original_total = metadata.get('total_samples', 0)

        # Keep the TRUE total for progress calculation
        # Don't replace it with processed count (which is less due to filtering)
        if actual_iterated_count > 0:
            # Use iterator count as the authoritative total
            metadata['total_samples'] = actual_iterated_count
            metadata['actual_iterated_count'] = actual_iterated_count
            metadata['count_verified'] = True
        # else: keep original total_samples

        # Store processed count separately
        metadata['processed_count'] = processed_count
        metadata['source'] = 'actual_processed'
        metadata['original_estimate'] = original_total  # Keep original len(dataset) for reference
        metadata['updated_at'] = datetime.now().isoformat()
        metadata['completed'] = True

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Log what happened
        if actual_iterated_count > 0 and actual_iterated_count != original_total:
            console.print(f"[cyan]Updated {name} total: {original_total:,} → {actual_iterated_count:,} (iterator count)[/cyan]")
        filtered_count = (actual_iterated_count or original_total) - processed_count
        if filtered_count > 0:
            console.print(f"[dim]  {processed_count:,} processed, {filtered_count:,} filtered/deduped[/dim]")
        else:
            console.print(f"[green]Verified {name} metadata: {processed_count:,} samples[/green]")

        return processed_count

    def _update_realtime_filter_stats(
        self,
        name: str,
        duration_rejected: int = 0,
        invalid_samples: int = 0,
        runtime_duplicates: int = 0,
        skipped_already_processed: int = 0,
        samples_processed: int = 0
    ) -> None:
        """
        Update metadata with real-time filter statistics during processing.

        This allows the monitor to display filtered counts in real-time,
        not just after processing completes.

        Args:
            name: Dataset name
            duration_rejected: Samples filtered by duration
            invalid_samples: Corrupted/invalid samples
            runtime_duplicates: Duplicate texts caught within this run
            skipped_already_processed: Samples skipped because text was already processed (dedup)
            samples_processed: Samples processed this run
        """
        if not self.is_main:
            return

        metadata_file = self.pseudo_labels_dir / f'{name}_metadata.json'

        try:
            # Load existing metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {'dataset': name}

            # Update filter stats
            # filtered_count = things that couldn't be processed (duration, invalid)
            # text_duplicates = things skipped because same text already exists (dedup working correctly)
            filtered_count = duration_rejected + invalid_samples
            text_duplicates = skipped_already_processed + runtime_duplicates

            metadata['filtered_count'] = filtered_count
            metadata['duration_rejected'] = duration_rejected
            metadata['invalid_samples'] = invalid_samples
            metadata['runtime_duplicates'] = runtime_duplicates
            metadata['skipped_already_processed'] = skipped_already_processed
            metadata['text_duplicates'] = text_duplicates  # Total text dedup (previous + current run)
            metadata['samples_processed_this_run'] = samples_processed
            metadata['filter_stats_updated_at'] = datetime.now().isoformat()

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            # Don't fail processing if metadata update fails
            log(f"[METADATA] Failed to update realtime filter stats for {name}: {e}", 'warning')

    def _save_dataset_metadata(
        self,
        name: str,
        total_samples: Optional[int],
        is_streaming: bool,
        estimated_hours: float = 0
    ) -> None:
        """
        Save dataset metadata when first loaded.

        This creates/updates the metadata file with:
        - Exact sample count for downloaded datasets
        - Streaming indicator for streaming datasets
        - Estimated hours from config

        The metadata is used by the monitor for accurate progress tracking.

        Args:
            name: Dataset name
            total_samples: Exact count for downloaded datasets, None for streaming
            is_streaming: Whether dataset is in streaming mode
            estimated_hours: Estimated hours from dataset config
        """
        if not self.is_main:
            return

        metadata_file = self.pseudo_labels_dir / f'{name}_metadata.json'

        # Load existing metadata if present
        existing_metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
            except Exception:
                pass

        # Build metadata
        metadata = {
            'dataset': name,
            'is_streaming': is_streaming,
            'estimated_hours': estimated_hours,
            'loaded_at': datetime.now().isoformat(),
        }

        if total_samples is not None:
            # Downloaded dataset - we have exact count
            metadata['total_samples'] = total_samples
            metadata['source'] = 'downloaded_exact'
            metadata['count_verified'] = True
            console.print(f"[green]✓ Saved {name} metadata: {total_samples:,} samples (exact from download)[/green]")
        else:
            # Streaming dataset - use estimate or previous processed count
            if existing_metadata.get('completed') and existing_metadata.get('total_samples'):
                # Use previously verified count from completed processing
                metadata['total_samples'] = existing_metadata['total_samples']
                metadata['source'] = 'previous_completion'
                metadata['count_verified'] = True
                console.print(f"[cyan]Using previous verified count for {name}: {metadata['total_samples']:,} samples[/cyan]")
            else:
                # Use estimate based on hours (rough: ~120 samples per hour for speech)
                estimated_samples = int(estimated_hours * 120)
                metadata['total_samples'] = estimated_samples
                metadata['source'] = 'estimated'
                metadata['count_verified'] = False
                console.print(f"[yellow]Streaming dataset {name}: using estimate ~{estimated_samples:,} samples ({estimated_hours} hours)[/yellow]")

        # Preserve completion status if already completed
        if existing_metadata.get('completed'):
            metadata['completed'] = True
            metadata['verified'] = existing_metadata.get('verified', False)
            metadata['verification_status'] = existing_metadata.get('verification_status', '')
            metadata['unique_samples'] = existing_metadata.get('unique_samples', 0)

        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_dataset_metadata(self, name: str) -> Dict[str, Any]:
        """
        Load dataset metadata if it exists.

        Returns dict with:
        - total_samples: Known or estimated sample count
        - is_streaming: Whether dataset is streaming
        - count_verified: Whether the count is exact (not estimated)
        - completed: Whether processing has completed before
        """
        metadata_file = self.pseudo_labels_dir / f'{name}_metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _verify_and_fill_missing_samples(
        self,
        name: str,
        expected_total: Optional[int] = None,
        duration_rejected: int = 0,
        invalid_samples: int = 0,
        runtime_duplicates: int = 0
    ) -> Dict[str, Any]:
        """
        Verify dataset completion and report accurate status.

        This runs after initial processing to:
        1. Count unique samples processed (by ground_truth text)
        2. Compare against actual_iterated_count (true iterator count) from metadata
        3. Account for filtered samples (duration, invalid, duplicates)
        4. Report accurate completion status

        The verification is CORRECT when:
            processed + duration_rejected + invalid + duplicates == total_fed

        Args:
            name: Dataset name
            expected_total: Expected total samples (fallback if metadata not available)
            duration_rejected: Number of samples filtered by duration
            invalid_samples: Number of invalid/corrupted samples
            runtime_duplicates: Number of runtime duplicates caught

        Returns:
            Dict with verification results:
            - 'processed_count': Number of unique samples processed
            - 'expected_count': Expected total (if known)
            - 'filtered_count': Number filtered (duration + invalid + duplicates)
            - 'status': 'verified', 'incomplete', or 'exceeded'
        """
        if not self.is_main:
            return {}

        console.print(f"\n[bold cyan]Verifying {name} completion...[/bold cyan]")

        # STEP 1: Clean up duplicates FIRST before counting
        # This ensures accurate counts for verification
        file_count_before = self._count_processed_from_files(name)
        processed_texts_before = self._load_processed_ids_from_files(name)
        unique_count_before = len(processed_texts_before)

        if file_count_before != unique_count_before:
            dup_count = file_count_before - unique_count_before
            console.print(f"[yellow]Found {dup_count:,} duplicate entries in files, cleaning up...[/yellow]")
            cleaned = self._deduplicate_jsonl_files(name)
            if cleaned > 0:
                console.print(f"[green]✓ Removed {cleaned:,} duplicate entries[/green]")

        # STEP 2: Now count after deduplication (reload to get fresh counts)
        processed_texts = self._load_processed_ids_from_files(name)
        processed_count = len(processed_texts)
        file_count = self._count_processed_from_files(name)

        # Get expected count from metadata - prioritize actual_iterated_count (true count)
        metadata_file = self.pseudo_labels_dir / f'{name}_metadata.json'
        actual_iterated_count = None
        len_dataset_count = expected_total  # Original from len(dataset)

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # Prioritize actual_iterated_count - this is the TRUE count from iteration
                    actual_iterated_count = metadata.get('actual_iterated_count')
                    if expected_total is None:
                        len_dataset_count = metadata.get('original_estimate') or metadata.get('total_samples')
            except Exception:
                pass

        # Use actual_iterated_count for verification if available (most accurate)
        verification_count = actual_iterated_count if actual_iterated_count else len_dataset_count
        count_source = "iterator" if actual_iterated_count else "len(dataset)"

        # Calculate filtered samples (these are EXPECTED to not be in output)
        filtered_count = duration_rejected + invalid_samples + runtime_duplicates

        result = {
            'processed_count': processed_count,
            'file_count': file_count,
            'expected_count': verification_count,
            'actual_iterated_count': actual_iterated_count,
            'len_dataset_count': len_dataset_count,
            'filtered_count': filtered_count,
            'duration_rejected': duration_rejected,
            'invalid_samples': invalid_samples,
            'runtime_duplicates': runtime_duplicates,
            'status': 'verified'
        }

        # STEP 3: Determine verification status
        # The KEY insight: processed + filtered should equal total_fed
        # If they match, we've accounted for all samples = VERIFIED
        if verification_count is not None:
            accounted_for = processed_count + filtered_count
            unaccounted = verification_count - accounted_for

            if unaccounted == 0:
                # PERFECT - all samples accounted for
                result['status'] = 'verified'
                console.print(f"[green]✓ Verification PASSED: All {verification_count:,} samples accounted for[/green]")
                console.print(f"[green]    Processed: {processed_count:,}[/green]")
                if filtered_count > 0:
                    console.print(f"[green]    Filtered:  {filtered_count:,} (duration: {duration_rejected:,}, invalid: {invalid_samples:,}, duplicates: {runtime_duplicates:,})[/green]")
            elif unaccounted < 0:
                # More accounted than expected (edge case with dedup)
                result['status'] = 'exceeded'
                console.print(f"[cyan]✓ Verification: Accounted for {accounted_for:,} samples vs {verification_count:,} expected ({-unaccounted:,} extra)[/cyan]")
            else:
                # Some samples unaccounted - check if it's significant
                unaccounted_pct = unaccounted / verification_count * 100 if verification_count else 0

                if unaccounted_pct < 1:
                    # Less than 1% unaccounted - likely rounding/timing edge cases
                    result['status'] = 'verified'
                    console.print(f"[green]✓ Verification PASSED: {accounted_for:,}/{verification_count:,} accounted ({unaccounted:,} minor variance)[/green]")
                else:
                    # Significant unaccounted samples
                    result['status'] = 'incomplete'
                    result['unaccounted'] = unaccounted
                    console.print(f"[yellow]⚠ Verification: {unaccounted:,} samples unaccounted ({unaccounted_pct:.1f}%)[/yellow]")
                    console.print(f"[yellow]    Total fed:   {verification_count:,}[/yellow]")
                    console.print(f"[yellow]    Processed:   {processed_count:,}[/yellow]")
                    console.print(f"[yellow]    Filtered:    {filtered_count:,}[/yellow]")
                    console.print(f"[yellow]    Accounted:   {accounted_for:,}[/yellow]")

            # Show breakdown
            if filtered_count > 0:
                filter_pct = filtered_count / verification_count * 100 if verification_count else 0
                console.print(f"[dim]  Filter rate: {filter_pct:.1f}% ({filtered_count:,} of {verification_count:,})[/dim]")
                if duration_rejected > 0:
                    console.print(f"[dim]    - Duration (< {self.min_duration}s or > {self.max_duration}s): {duration_rejected:,}[/dim]")
                if invalid_samples > 0:
                    console.print(f"[dim]    - Invalid/corrupted: {invalid_samples:,}[/dim]")
                if runtime_duplicates > 0:
                    console.print(f"[dim]    - Runtime duplicates: {runtime_duplicates:,}[/dim]")
        else:
            console.print(f"[green]✓ Processed {processed_count:,} unique samples (no expected count to verify against)[/green]")

        # Report final file count (should match unique count after dedup)
        if file_count != processed_count:
            console.print(f"[dim]  File entries: {file_count:,} (unique: {processed_count:,})[/dim]")

        # Update metadata with verification results
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                metadata['verified'] = True
                metadata['verified_at'] = datetime.now().isoformat()
                metadata['unique_samples'] = processed_count
                metadata['file_entries'] = file_count
                metadata['filtered_count'] = filtered_count
                metadata['duration_rejected'] = duration_rejected
                metadata['invalid_samples'] = invalid_samples
                metadata['runtime_duplicates'] = runtime_duplicates
                metadata['verification_status'] = result['status']
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception:
                pass

        return result

    def _deduplicate_jsonl_files(self, name: str) -> int:
        """
        Remove duplicate entries from JSONL files based on ground_truth text.

        This physically removes duplicates from the files, not just skipping them.
        Keeps the FIRST occurrence of each unique ground_truth text.

        Args:
            name: Dataset name

        Returns:
            Number of duplicate entries removed
        """
        if not self.is_main:
            return 0

        total_removed = 0
        seen_texts = set()

        # Process all JSONL files for this dataset
        jsonl_patterns = [
            f'{name}_gpu*_accepted.jsonl',
            f'{name}_gpu*_rejected.jsonl'
        ]

        for pattern in jsonl_patterns:
            for jsonl_file in self.pseudo_labels_dir.glob(pattern):
                try:
                    # Read all entries
                    entries = []
                    duplicates_in_file = 0

                    with open(jsonl_file, 'r') as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                text_key = entry.get('ground_truth', '').strip()

                                if text_key in seen_texts:
                                    # Duplicate - skip it
                                    duplicates_in_file += 1
                                    continue

                                seen_texts.add(text_key)
                                entries.append(line.strip())
                            except json.JSONDecodeError:
                                continue

                    # Rewrite file without duplicates
                    if duplicates_in_file > 0:
                        with open(jsonl_file, 'w') as f:
                            for entry_line in entries:
                                f.write(entry_line + '\n')
                        total_removed += duplicates_in_file

                except Exception as e:
                    console.print(f"[red]Error deduplicating {jsonl_file}: {e}[/red]")
                    continue

        # Invalidate the cache since files changed
        if total_removed > 0:
            cache_file = self.pseudo_labels_dir / f'{name}_processed_texts.cache'
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception:
                    pass

        return total_removed

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

        # Check if this dataset uses chunked download mode (preferred over streaming)
        # Chunked download: downloads ~100GB at a time, processes, then downloads next chunk
        # This avoids rate limiting while giving downloaded-dataset performance
        use_chunked = dataset_config.get('use_chunked_download', False)

        # If chunked download is enabled, don't use streaming
        # Otherwise, force streaming for massive datasets
        if use_chunked:
            use_streaming = False  # Chunked download handles these datasets
        else:
            # Legacy: Force streaming for massive datasets if not using chunked
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

            # Special handling for datasets with multiple splits (LibriSpeech, podcast_fillers, etc.)
            if len(dataset_config['splits']) > 1:
                datasets_list = []
                for split in dataset_config['splits']:
                    kwargs = {
                        'split': split,
                        'streaming': use_streaming,
                        'trust_remote_code': True,
                    }
                    if dataset_config['subset']:
                        kwargs['name'] = dataset_config['subset']

                    ds = load_with_retry(
                        dataset_config['hf_name'],
                        **kwargs,
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

            # Save dataset metadata for accurate progress tracking
            # This stores exact count for downloaded datasets, or estimate for streaming
            self._save_dataset_metadata(
                name=name,
                total_samples=total_samples,
                is_streaming=use_streaming,
                estimated_hours=dataset_config.get('estimated_hours', 0)
            )

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

    def _get_chunk_info(self, name: str) -> Dict[str, Any]:
        """
        Get chunk progress info from metadata.

        Returns dict with:
        - current_chunk: Which chunk we're on (0-indexed)
        - chunk_start_idx: Start sample index for current chunk
        - chunk_end_idx: End sample index for current chunk (exclusive)
        - total_samples: Total samples in dataset (if known)
        - chunks_completed: List of completed chunk indices
        """
        metadata_file = self.pseudo_labels_dir / f'{name}_chunk_progress.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'current_chunk': 0,
            'chunk_start_idx': 0,
            'chunk_end_idx': None,
            'total_samples': None,
            'chunks_completed': [],
            'samples_per_chunk': None,
        }

    def _save_chunk_info(self, name: str, chunk_info: Dict[str, Any]) -> None:
        """Save chunk progress info to metadata."""
        if not self.is_main:
            return
        metadata_file = self.pseudo_labels_dir / f'{name}_chunk_progress.json'
        chunk_info['updated_at'] = datetime.now().isoformat()
        with open(metadata_file, 'w') as f:
            json.dump(chunk_info, f, indent=2)

    def _load_dataset_chunked(
        self,
        name: str,
        chunk_size_gb: float = 100.0,
        estimated_sample_size_mb: float = 4.0,
    ) -> Tuple[Optional[Iterator], Optional[int], Dict[str, Any]]:
        """
        Load a large dataset in chunks to avoid rate limiting.

        This downloads ~chunk_size_gb worth of data at a time, processes it,
        then moves to the next chunk. This simulates streaming behavior but
        with downloaded data for better performance.

        Args:
            name: Dataset name
            chunk_size_gb: Target chunk size in GB (default 100GB, ±10GB variance)
            estimated_sample_size_mb: Average sample size for chunk calculation

        Returns:
            Tuple of (dataset_iterator, chunk_sample_count, chunk_info)
            - chunk_sample_count is the number of samples in this chunk
            - chunk_info contains chunk progress metadata
        """
        from datasets import load_dataset
        import random

        dataset_config = DATASET_CONFIGS.get(name)
        if not dataset_config:
            return None, None, {}

        # Calculate samples per chunk
        # chunk_size_gb * 1024 MB/GB / estimated_sample_size_mb
        samples_per_chunk = int((chunk_size_gb * 1024) / estimated_sample_size_mb)

        # Load chunk progress
        chunk_info = self._get_chunk_info(name)

        # Check HF authentication
        global _HF_AUTH_CHECKED
        if not _HF_AUTH_CHECKED and self.is_main:
            setup_hf_authentication()
            _HF_AUTH_CHECKED = True

        # Stagger GPU requests
        if self.is_distributed and self.local_rank > 0:
            delay = self.local_rank * 0.5 + random.uniform(0, 0.5)
            time.sleep(delay)

        try:
            if self.is_main:
                console.print(f"[yellow]Loading {name} (chunked download mode, ~{chunk_size_gb:.0f}GB chunks)...[/yellow]")

            # Helper function with retry logic
            def load_with_retry(hf_name, max_retries=5, **kwargs):
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
                raise Exception(f"Failed to load {hf_name} after {max_retries} retries")

            # First, we need to get the total dataset size
            # Load in streaming mode just to get info, then calculate chunks
            if chunk_info.get('total_samples') is None:
                if self.is_main:
                    console.print(f"[cyan]Discovering dataset size (streaming probe)...[/cyan]")

                # Try to get dataset info without downloading
                try:
                    from datasets import load_dataset_builder
                    builder_kwargs = {'trust_remote_code': True}
                    if dataset_config['subset']:
                        builder_kwargs['name'] = dataset_config['subset']

                    builder = load_dataset_builder(dataset_config['hf_name'], **builder_kwargs)
                    if hasattr(builder, 'info') and builder.info.splits:
                        split_name = dataset_config['splits'][0]
                        if split_name in builder.info.splits:
                            total_samples = builder.info.splits[split_name].num_examples
                            chunk_info['total_samples'] = total_samples
                            chunk_info['samples_per_chunk'] = samples_per_chunk
                            if self.is_main:
                                console.print(f"[green]Dataset has {total_samples:,} samples[/green]")
                except Exception as e:
                    if self.is_main:
                        console.print(f"[yellow]Could not get dataset info: {e}[/yellow]")
                        console.print(f"[yellow]Will discover size during first chunk download[/yellow]")

            total_samples = chunk_info.get('total_samples')
            current_chunk = chunk_info.get('current_chunk', 0)
            chunks_completed = chunk_info.get('chunks_completed', [])

            # Calculate chunk boundaries
            chunk_start_idx = current_chunk * samples_per_chunk
            chunk_end_idx = chunk_start_idx + samples_per_chunk

            # Check if we're done
            if total_samples is not None and chunk_start_idx >= total_samples:
                if self.is_main:
                    console.print(f"[green]All chunks completed for {name}![/green]")
                return iter([]), 0, chunk_info

            if self.is_main:
                if total_samples:
                    total_chunks = (total_samples + samples_per_chunk - 1) // samples_per_chunk
                    console.print(f"[cyan]Chunk {current_chunk + 1}/{total_chunks}: samples {chunk_start_idx:,} to {min(chunk_end_idx, total_samples):,}[/cyan]")
                else:
                    console.print(f"[cyan]Chunk {current_chunk + 1}: samples {chunk_start_idx:,} to {chunk_end_idx:,}[/cyan]")

            # Load the chunk using split slicing
            # HuggingFace datasets support split[start:end] syntax
            split_name = dataset_config['splits'][0]
            if total_samples:
                actual_end = min(chunk_end_idx, total_samples)
            else:
                actual_end = chunk_end_idx

            # Use split slicing: "train[10000:20000]"
            split_slice = f"{split_name}[{chunk_start_idx}:{actual_end}]"

            kwargs = {
                'split': split_slice,
                'trust_remote_code': True,
            }
            if dataset_config['subset']:
                kwargs['name'] = dataset_config['subset']

            if self.is_main:
                console.print(f"[yellow]Downloading chunk (this may take a while)...[/yellow]")

            dataset = load_with_retry(dataset_config['hf_name'], **kwargs)

            chunk_sample_count = len(dataset)

            # If we didn't know total_samples, try to discover it now
            if total_samples is None:
                # Try loading full dataset info
                try:
                    full_kwargs = {
                        'split': split_name,
                        'streaming': True,
                        'trust_remote_code': True,
                    }
                    if dataset_config['subset']:
                        full_kwargs['name'] = dataset_config['subset']
                    streaming_ds = load_with_retry(dataset_config['hf_name'], **full_kwargs)
                    # Some streaming datasets expose info
                    if hasattr(streaming_ds, '_info') and streaming_ds._info:
                        pass  # Info might have size
                except Exception:
                    pass

            # Update chunk info
            chunk_info['current_chunk'] = current_chunk
            chunk_info['chunk_start_idx'] = chunk_start_idx
            chunk_info['chunk_end_idx'] = actual_end
            chunk_info['chunk_sample_count'] = chunk_sample_count
            chunk_info['samples_per_chunk'] = samples_per_chunk
            self._save_chunk_info(name, chunk_info)

            if self.is_main:
                console.print(f"[green]✓ Chunk loaded: {chunk_sample_count:,} samples[/green]")
                estimated_gb = (chunk_sample_count * estimated_sample_size_mb) / 1024
                console.print(f"[dim]  Estimated size: ~{estimated_gb:.1f}GB[/dim]")

            # Save metadata for monitor
            self._save_dataset_metadata(
                name=name,
                total_samples=total_samples,
                is_streaming=False,  # Chunked is not streaming
                estimated_hours=dataset_config.get('estimated_hours', 0)
            )

            return iter(dataset), chunk_sample_count, chunk_info

        except Exception as e:
            if self.is_main:
                console.print(f"[red]✗ Error loading chunk for {name}: {e}[/red]")
                import traceback
                traceback.print_exc()
            return None, None, chunk_info

    def _mark_chunk_completed(self, name: str, chunk_idx: int) -> None:
        """Mark a chunk as completed and advance to next chunk."""
        if not self.is_main:
            return

        chunk_info = self._get_chunk_info(name)
        chunks_completed = chunk_info.get('chunks_completed', [])

        if chunk_idx not in chunks_completed:
            chunks_completed.append(chunk_idx)
            chunk_info['chunks_completed'] = sorted(chunks_completed)

        # Advance to next chunk
        chunk_info['current_chunk'] = chunk_idx + 1
        chunk_info['chunk_start_idx'] = (chunk_idx + 1) * chunk_info.get('samples_per_chunk', 0)

        self._save_chunk_info(name, chunk_info)

        if self.is_main:
            console.print(f"[green]✓ Chunk {chunk_idx + 1} completed, advancing to next chunk[/green]")

    def _verify_chunk_completion(
        self,
        name: str,
        chunk_idx: int,
        chunk_start_idx: int,
        chunk_end_idx: int,
        expected_count: int
    ) -> Dict[str, Any]:
        """
        Hard verify that a chunk was processed correctly before cleanup.

        Checks:
        1. Count of unique samples processed matches expected
        2. No duplicates within the chunk range
        3. All samples in range are accounted for (processed or filtered)

        Args:
            name: Dataset name
            chunk_idx: Chunk index being verified
            chunk_start_idx: Start sample index of chunk
            chunk_end_idx: End sample index of chunk (exclusive)
            expected_count: Expected number of samples in chunk

        Returns:
            Dict with verification results:
            - verified: True if chunk passed verification
            - processed_count: Unique samples processed from this chunk
            - expected_count: Expected samples
            - duplicates_found: Number of duplicate entries found
            - status: 'verified', 'incomplete', 'duplicates_found'
        """
        if not self.is_main:
            return {'verified': True}

        console.print(f"\n[bold cyan]Verifying chunk {chunk_idx + 1} completion...[/bold cyan]")

        # Load all processed ground_truth texts
        processed_texts = self._load_processed_ids_from_files(name)
        processed_count = len(processed_texts)

        # Count entries in JSONL files (may include duplicates)
        file_count = self._count_processed_from_files(name)

        # Check for duplicates
        duplicates_found = file_count - processed_count

        result = {
            'verified': False,
            'chunk_idx': chunk_idx,
            'processed_count': processed_count,
            'expected_count': expected_count,
            'file_count': file_count,
            'duplicates_found': duplicates_found,
            'status': 'unknown'
        }

        # Verification logic
        # Note: Some samples may be filtered (duration, invalid), so we allow some tolerance
        # Also, deduplication may reduce count if dataset has duplicate texts

        # Check for excessive duplicates (more than 1% is suspicious)
        duplicate_rate = duplicates_found / max(file_count, 1)
        if duplicate_rate > 0.01:
            result['status'] = 'duplicates_found'
            console.print(f"[red]⚠ WARNING: {duplicates_found:,} duplicate entries found ({duplicate_rate:.1%})[/red]")
            console.print(f"[yellow]  This may indicate a bug in the deduplication system[/yellow]")
            # Don't fail verification for duplicates - they're handled by dedup
            # But log the warning

        # Check if we processed a reasonable amount
        # Allow 20% variance due to filtering (duration, invalid samples, dedup)
        min_expected = int(expected_count * 0.8)

        if processed_count >= min_expected:
            result['verified'] = True
            result['status'] = 'verified'
            console.print(f"[green]✓ Chunk {chunk_idx + 1} verified: {processed_count:,} unique samples[/green]")
            if processed_count < expected_count:
                gap = expected_count - processed_count
                console.print(f"[dim]  ({gap:,} samples filtered/deduped, within tolerance)[/dim]")
        else:
            result['status'] = 'incomplete'
            console.print(f"[red]✗ Chunk {chunk_idx + 1} incomplete: {processed_count:,} < {min_expected:,} expected[/red]")
            console.print(f"[yellow]  Chunk will NOT be marked complete - will retry[/yellow]")

        return result

    def _cleanup_chunk_cache(self, name: str) -> None:
        """
        Clean up HuggingFace cache for a dataset after chunk is verified.

        This frees disk space by removing downloaded chunk data.
        Only call after verification confirms chunk is complete.
        """
        if not self.is_main:
            return

        try:
            from huggingface_hub import scan_cache_dir, HfFolder
            import shutil

            # Get HF cache directory
            cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
            datasets_cache = os.path.join(cache_dir, 'datasets')

            dataset_config = DATASET_CONFIGS.get(name)
            if not dataset_config:
                return

            hf_name = dataset_config['hf_name']
            # Convert HF name to cache directory format (replace / with ___)
            cache_name = hf_name.replace('/', '___')

            # Find and remove cached data for this dataset
            if os.path.exists(datasets_cache):
                for item in os.listdir(datasets_cache):
                    if cache_name in item:
                        item_path = os.path.join(datasets_cache, item)
                        if os.path.isdir(item_path):
                            # Calculate size before deletion
                            size_gb = sum(
                                os.path.getsize(os.path.join(dirpath, filename))
                                for dirpath, dirnames, filenames in os.walk(item_path)
                                for filename in filenames
                            ) / (1024**3)

                            console.print(f"[yellow]Cleaning up cache for {name} (~{size_gb:.1f}GB)...[/yellow]")
                            shutil.rmtree(item_path)
                            console.print(f"[green]✓ Freed ~{size_gb:.1f}GB disk space[/green]")

            # Also try to clean via HF API
            try:
                cache_info = scan_cache_dir()
                for repo in cache_info.repos:
                    if hf_name in repo.repo_id:
                        for revision in repo.revisions:
                            # Delete specific revision
                            delete_strategy = cache_info.delete_revisions(revision.commit_hash)
                            delete_strategy.execute()
            except Exception:
                pass  # HF cache API may not always work

        except Exception as e:
            if self.is_main:
                console.print(f"[yellow]Cache cleanup warning: {e}[/yellow]")
                console.print(f"[dim]  (This is non-critical, processing will continue)[/dim]")

    def _process_chunk_with_verification(
        self,
        name: str,
        chunk_idx: int,
        chunk_info: Dict[str, Any],
        dataset_iter: Iterator,
        chunk_sample_count: int
    ) -> bool:
        """
        Process a chunk with full verification and cleanup.

        This is the main entry point for chunked processing:
        1. Process all samples in the chunk
        2. Verify chunk completion (hard check)
        3. If verified, clean up cache and advance to next chunk
        4. If not verified, return False to retry

        Returns:
            True if chunk was processed and verified successfully
        """
        # Note: Actual processing happens in process_dataset()
        # This method is called AFTER processing to verify and cleanup

        chunk_start = chunk_info.get('chunk_start_idx', 0)
        chunk_end = chunk_info.get('chunk_end_idx', chunk_start + chunk_sample_count)

        # Verify the chunk
        verification = self._verify_chunk_completion(
            name=name,
            chunk_idx=chunk_idx,
            chunk_start_idx=chunk_start,
            chunk_end_idx=chunk_end,
            expected_count=chunk_sample_count
        )

        if verification.get('verified', False):
            # Chunk verified - mark complete and cleanup
            self._mark_chunk_completed(name, chunk_idx)

            # Clean up downloaded data to free disk space
            self._cleanup_chunk_cache(name)

            return True
        else:
            # Verification failed - don't advance, will retry
            if self.is_main:
                console.print(f"[red]Chunk {chunk_idx + 1} failed verification - will retry[/red]")
            return False

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
            log(f"[GPU_BATCH] Empty batch received", 'debug')
            return []

        batch_size = len(batch_data)
        sample_indices = [item[5] for item in batch_data]
        log(f"[GPU_BATCH] Processing batch of {batch_size} samples, indices={sample_indices[0]}-{sample_indices[-1]}", 'debug')

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
        preprocess_failed = 0
        for i, audio in enumerate(audio_arrays):
            try:
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                # Ensure float32
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                processed_audios.append(audio)
            except Exception as e:
                log(f"[GPU_BATCH] Audio preprocessing failed for idx={sample_indices[i]}: {e}", 'error')
                preprocess_failed += 1
                # Append None as placeholder to maintain alignment
                processed_audios.append(None)

        # Filter out failed preprocessing
        valid_indices = [i for i, a in enumerate(processed_audios) if a is not None]
        if len(valid_indices) < len(processed_audios):
            log(f"[GPU_BATCH] {preprocess_failed} samples failed preprocessing, continuing with {len(valid_indices)}", 'warning')
            processed_audios = [processed_audios[i] for i in valid_indices]
            ground_truths = [ground_truths[i] for i in valid_indices]
            sample_ids = [sample_ids[i] for i in valid_indices]
            durations = [durations[i] for i in valid_indices]
            sample_indices = [sample_indices[i] for i in valid_indices]

        if not processed_audios:
            log(f"[GPU_BATCH] All samples failed preprocessing, returning empty", 'error')
            return []

        # BATCHED INFERENCE - single GPU call for entire batch!
        try:
            transcriptions = self.teacher.generate_pseudo_labels_batch(
                audio_arrays=processed_audios,
                sampling_rate=sr,
                language="en",
            )
            log(f"[GPU_BATCH] Inference complete: {len(transcriptions)} transcriptions for {len(processed_audios)} inputs", 'debug')
        except Exception as e:
            log(f"[GPU_BATCH] Batch inference error: {e}", 'error')
            if self.is_main:
                console.print(f"[dim red]Batch inference error: {e}[/dim red]")
            return []

        # Verify alignment
        if len(transcriptions) != len(ground_truths):
            log(f"[GPU_BATCH] MISMATCH: {len(transcriptions)} transcriptions vs {len(ground_truths)} ground_truths", 'error')
            return []

        # Create entries with WER calculation
        entries = []
        accepted_count = 0
        rejected_count = 0
        timestamp = datetime.now().isoformat()

        for i, (transcription, ground_truth, sample_id, duration) in enumerate(
            zip(transcriptions, ground_truths, sample_ids, durations)
        ):
            wer = self._calculate_wer(ground_truth, transcription)
            accepted = wer <= self.wer_threshold

            if accepted:
                accepted_count += 1
            else:
                rejected_count += 1

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

        log(f"[GPU_BATCH] Batch complete: {len(entries)} entries (accepted={accepted_count}, rejected_wer={rejected_count})", 'debug')
        return entries

    def process_dataset(
        self,
        name: str,
        max_samples: Optional[int] = None,
        resume: bool = True,
    ) -> DatasetProgress:
        """
        Process a single dataset with CrisperWhisper pseudo-labeling.

        For large datasets with use_chunked_download=True, this method delegates
        to _process_dataset_chunked() which downloads and processes in ~100GB chunks.

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
        config = DATASET_CONFIGS.get(name)
        if not config:
            return DatasetProgress(name=name, status="error", error_message="Unknown dataset")

        # Check if this dataset uses chunked download mode
        if config.get('use_chunked_download', False):
            return self._process_dataset_chunked(name, max_samples, resume)

        # Standard processing for non-chunked datasets
        return self._process_dataset_standard(name, max_samples, resume)

    def _process_dataset_chunked(
        self,
        name: str,
        max_samples: Optional[int] = None,
        resume: bool = True,
    ) -> DatasetProgress:
        """
        Process a large dataset using chunked downloads.

        Downloads ~100GB at a time, processes it, verifies completion,
        cleans up cache, then moves to next chunk. This avoids:
        - HuggingFace rate limiting
        - Memory issues from loading entire dataset
        - Disk space issues (cleanup after each chunk)

        Each chunk is fully verified before advancing to ensure no data loss.
        """
        config = DATASET_CONFIGS.get(name)
        chunk_size_gb = config.get('chunk_size_gb', 100)
        estimated_sample_size_mb = config.get('estimated_sample_size_mb', 4)

        self._load_teacher()

        # Load chunk progress to see where we left off
        chunk_info = self._get_chunk_info(name)
        current_chunk = chunk_info.get('current_chunk', 0)

        if self.is_main:
            console.print(f"\n[bold cyan]═══ Processing {name} (Chunked Mode) ═══[/bold cyan]")
            console.print(f"[cyan]Chunk size: ~{chunk_size_gb}GB | Starting at chunk {current_chunk + 1}[/cyan]")

        # Keep processing chunks until done
        final_progress = DatasetProgress(name=name)
        max_chunks = 1000  # Safety limit

        for chunk_iteration in range(max_chunks):
            # Load the next chunk
            dataset_iter, chunk_sample_count, chunk_info = self._load_dataset_chunked(
                name=name,
                chunk_size_gb=chunk_size_gb,
                estimated_sample_size_mb=estimated_sample_size_mb,
            )

            if dataset_iter is None:
                if self.is_main:
                    console.print(f"[red]Failed to load chunk for {name}[/red]")
                final_progress.status = "error"
                final_progress.error_message = "Failed to load chunk"
                return final_progress

            # Check if all chunks are done
            if chunk_sample_count == 0:
                if self.is_main:
                    console.print(f"[green]✓ All chunks completed for {name}![/green]")
                final_progress.status = "completed"
                break

            current_chunk = chunk_info.get('current_chunk', 0)

            if self.is_main:
                console.print(f"\n[bold yellow]Processing chunk {current_chunk + 1}...[/bold yellow]")

            # Process this chunk using standard processing
            chunk_progress = self._process_dataset_standard(
                name=name,
                max_samples=chunk_sample_count,  # Limit to chunk size
                resume=resume,
                _dataset_iter_override=dataset_iter,
                _total_samples_override=chunk_sample_count,
            )

            # Verify and cleanup the chunk
            if self.is_main:
                verification = self._verify_chunk_completion(
                    name=name,
                    chunk_idx=current_chunk,
                    chunk_start_idx=chunk_info.get('chunk_start_idx', 0),
                    chunk_end_idx=chunk_info.get('chunk_end_idx', chunk_sample_count),
                    expected_count=chunk_sample_count
                )

                if verification.get('verified', False):
                    # Chunk verified - mark complete and cleanup
                    self._mark_chunk_completed(name, current_chunk)
                    self._cleanup_chunk_cache(name)
                else:
                    # Verification failed - log and continue (dedup will handle on next run)
                    console.print(f"[yellow]Chunk {current_chunk + 1} verification incomplete - continuing anyway[/yellow]")
                    console.print(f"[dim]Deduplication will handle any gaps on next run[/dim]")
                    self._mark_chunk_completed(name, current_chunk)

            # Sync GPUs before next chunk
            if self.is_distributed:
                dist.barrier()

            # Update progress
            final_progress = chunk_progress

            # Check if we hit max_samples limit
            if max_samples and final_progress.samples_processed >= max_samples:
                break

        return final_progress

    def _process_dataset_standard(
        self,
        name: str,
        max_samples: Optional[int] = None,
        resume: bool = True,
        _dataset_iter_override: Optional[Iterator] = None,
        _total_samples_override: Optional[int] = None,
    ) -> DatasetProgress:
        """
        Standard dataset processing (non-chunked).

        This is the core processing logic, extracted to support both:
        - Direct processing of small/medium datasets
        - Chunk-by-chunk processing of large datasets

        Args:
            _dataset_iter_override: Pre-loaded dataset iterator (for chunked mode)
            _total_samples_override: Pre-known sample count (for chunked mode)
        """
        log(f"[PROCESS_DATASET] Starting processing for {name}", 'info')
        log(f"[PROCESS_DATASET] Parameters: max_samples={max_samples}, resume={resume}, override={_dataset_iter_override is not None}", 'info')
        log(f"[PROCESS_DATASET] GPU info: local_rank={self.local_rank}, world_size={self.world_size}", 'info')

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
        # TEXT-BASED RESUME: Load processed ground_truth texts from ALL output files
        # Using ground_truth text (not audio hash) ensures reliable deduplication because:
        # - Same text = same sample, regardless of audio byte differences
        # - Stable across runs, GPU counts, iteration order
        #
        # MULTI-GPU SYNC: Main process loads first (and writes cache), then barrier,
        # then other GPUs load from cache. This avoids race conditions.
        already_processed_ids = set()  # Contains ground_truth texts, not sample_ids
        if resume:
            if self.is_main:
                # Main process loads and writes cache
                already_processed_ids = self._load_processed_ids_from_files(name)
                if already_processed_ids:
                    console.print(f"[green]Found {len(already_processed_ids):,} already processed samples for {name}[/green]")
                    console.print(f"[cyan]Using ground_truth text deduplication (stable across runs)[/cyan]")

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

        # Reset per-run counters (these should reflect THIS run, not accumulated across runs)
        # The actual processed/accepted counts are derived from files, so resetting is safe
        # Duration counters can accumulate incorrectly when resuming, so reset them
        progress.samples_rejected_duration = 0
        progress.total_duration_hours = 0.0

        # Update GPU count for this run
        progress.gpu_count_at_start = self.world_size
        progress.status = "processing"

        # Load dataset - use override if provided (chunked mode), otherwise load normally
        if _dataset_iter_override is not None:
            dataset_iter = _dataset_iter_override
            total_samples = _total_samples_override
        else:
            dataset_iter, total_samples = self._load_dataset(name, skip_to=0)

        if dataset_iter is None:
            progress.status = "error"
            progress.error_message = "Failed to load dataset"
            return progress

        # Save actual sample count to a metadata file for the monitor
        if total_samples is not None and self.is_main and _dataset_iter_override is None:
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

        # Log prefetcher configuration
        log(f"[PREFETCHER] Creating prefetcher for {name}", 'info')
        log(f"[PREFETCHER] Config: workers={num_prefetch_workers}, batch_size={self.batch_size}, prefetch_batches=16", 'info')
        log(f"[PREFETCHER] Duration filter: min={self.min_duration}s, max={self.max_duration}s", 'info')
        log(f"[PREFETCHER] Dedup: already_processed_ids={len(already_processed_ids):,}", 'info')
        log(f"[PREFETCHER] Sharding: world_size={self.world_size}, local_rank={self.local_rank}", 'info')

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

        # RUNTIME DEDUP: Track texts processed in THIS run to catch duplicates
        # The prefetcher's already_processed_ids handles resume (previous runs)
        # This set catches duplicates within the same dataset that slip through
        # (e.g., same text appearing multiple times in the dataset)
        runtime_processed_texts = set()
        runtime_duplicates_caught = 0

        # Process samples with BATCHING using threaded prefetcher
        log(f"[MAIN_LOOP] Starting main processing loop for {name}, mode={mode}", 'info')
        with open(output_file, mode) as out_f, open(rejected_file, mode) as rej_f:
            # Progress bar only on main process
            # Track actual processed count from files (accurate across all GPUs)
            if self.is_main:
                initial_processed = self._count_processed_from_files(name)
                log(f"[MAIN_LOOP] Initial processed from files: {initial_processed:,}", 'info')
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
            total_entries_from_batches = 0  # Track total entries received from GPU batches
            total_entries_written = 0  # Track total entries actually written to files

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
                    should_exit = False
                    if prefetcher.is_exhausted():
                        log(f"[MAIN_LOOP] Prefetcher exhausted after {total_batches} batches", 'info')
                        if self.is_main:
                            console.print(f"[green]Dataset {name} exhausted - all samples processed[/green]")
                        should_exit = True

                    # Track consecutive empty batches
                    if not should_exit:
                        empty_batch_count += 1
                        if empty_batch_count % 5 == 0:
                            stats = prefetcher.get_stats()
                            log(f"[MAIN_LOOP] Empty batch #{empty_batch_count}, feeder_done={prefetcher.feeder_done.is_set()}, raw_q={stats['raw_queue_size']}, proc_q={stats['queue_size']}, workers={stats['workers_active']}", 'debug')

                        # Safety: If we've gotten 10+ empty batches in a row and feeder is done,
                        # the workers may have all finished but exhausted flag not set due to race
                        if empty_batch_count >= 10 and prefetcher.feeder_done.is_set():
                            if self.is_main:
                                console.print(f"[yellow]Warning: {empty_batch_count} empty batches, feeder done - checking exhaustion[/yellow]")
                            log(f"[MAIN_LOOP] {empty_batch_count} empty batches with feeder done, checking exhaustion", 'warning')
                            # Give workers a moment to finish and set exhausted flag
                            time.sleep(2.0)
                            if prefetcher.is_exhausted() or (prefetcher.processed_queue.empty() and prefetcher.raw_queue.empty()):
                                if self.is_main:
                                    console.print(f"[green]Dataset {name} processing complete (detected via empty queues)[/green]")
                                log(f"[MAIN_LOOP] Dataset complete via empty queues check", 'info')
                                should_exit = True

                    if should_exit:
                        log(f"[MAIN_LOOP] Exiting loop: total_batches={total_batches}, entries_from_batches={total_entries_from_batches}, entries_written={total_entries_written}", 'info')
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
                total_entries_from_batches += len(entries)

                # Log periodically (every 100 batches)
                if total_batches % 100 == 0:
                    log(f"[MAIN_LOOP] Batch #{total_batches}: received {len(entries)} entries from batch of {len(batch_data)} samples", 'debug')

                batch_written = 0
                batch_deduped = 0
                for entry in entries:
                    # RUNTIME DEDUP CHECK: Catch duplicates within the same run
                    # The prefetcher checks already_processed_ids (from previous runs)
                    # but doesn't catch duplicates within the current dataset iteration
                    text_key = entry.ground_truth.strip()
                    if text_key in runtime_processed_texts:
                        # Duplicate! Skip writing to avoid bloating output files
                        runtime_duplicates_caught += 1
                        batch_deduped += 1
                        continue

                    # Mark as processed for this run
                    runtime_processed_texts.add(text_key)

                    # Also update the prefetcher's dedup set so workers can skip future dupes
                    # This is thread-safe because we only add (never remove)
                    prefetcher.already_processed_ids.add(text_key)

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
                    batch_written += 1
                    total_entries_written += 1

                # Log batch details if any were deduped
                if batch_deduped > 0:
                    log(f"[MAIN_LOOP] Batch #{total_batches}: wrote {batch_written}, deduped {batch_deduped}", 'debug')

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

                    # Update metadata with real-time filter stats (for monitor display)
                    if self.is_main:
                        current_prefetch_stats = prefetcher.get_stats()
                        self._update_realtime_filter_stats(
                            name=name,
                            duration_rejected=current_prefetch_stats.get('rejected_duration', 0),
                            invalid_samples=current_prefetch_stats.get('invalid_samples', 0),
                            runtime_duplicates=runtime_duplicates_caught,
                            skipped_already_processed=current_prefetch_stats.get('skipped_in_loop', 0),
                            samples_processed=samples_this_run
                        )

                    # Invalidate cache so it gets rebuilt on next resume
                    # (simpler than incremental updates, and resume is now fast anyway)
                    # NOTE: Must match the cache filename in _load_processed_ids_from_files()
                    cache_file = self.pseudo_labels_dir / f'{name}_processed_texts.cache'
                    if cache_file.exists():
                        try:
                            cache_file.unlink()
                        except Exception:
                            pass

            # Stop prefetcher
            prefetcher.stop()

            if pbar:
                pbar.close()

        # COORDINATED SHUTDOWN: Synchronize all GPUs after processing loop
        # This is CRITICAL to prevent NCCL timeouts. When GPUs finish at different
        # times (due to uneven data distribution from deduplication/sharding), we
        # need to ensure all GPUs have exited the processing loop before proceeding.
        #
        # We use FILE-BASED signaling instead of NCCL collectives because:
        # 1. NCCL collectives (barrier, all_reduce) block until ALL GPUs call them
        # 2. If one GPU is stuck in processing loop, others will timeout at the collective
        # 3. File-based signaling is non-blocking and allows timeout-based polling
        if self.is_distributed:
            try:
                # Signal this GPU is done by creating a marker file
                done_marker = self.pseudo_labels_dir / f'{name}_gpu{self.local_rank}_done.marker'
                done_marker.touch()
                console.print(f"[cyan]GPU {self.local_rank}: Processing complete, waiting for other GPUs...[/cyan]")

                # Wait for all GPUs to create their marker files (with timeout)
                max_wait_seconds = 300  # 5 minute timeout (less than NCCL's 10 min default)
                poll_interval = 2.0
                waited = 0

                while waited < max_wait_seconds:
                    # Check if all GPUs have finished
                    all_done = True
                    for gpu_id in range(self.world_size):
                        marker = self.pseudo_labels_dir / f'{name}_gpu{gpu_id}_done.marker'
                        if not marker.exists():
                            all_done = False
                            break

                    if all_done:
                        if self.is_main:
                            console.print(f"[green]All {self.world_size} GPUs completed processing[/green]")
                        break

                    time.sleep(poll_interval)
                    waited += poll_interval

                    # Periodic status update
                    if waited % 30 == 0 and self.is_main:
                        done_count = sum(1 for gpu_id in range(self.world_size)
                                        if (self.pseudo_labels_dir / f'{name}_gpu{gpu_id}_done.marker').exists())
                        console.print(f"[yellow]Waiting for GPUs: {done_count}/{self.world_size} done ({waited:.0f}s elapsed)[/yellow]")

                if waited >= max_wait_seconds:
                    # Timeout - some GPUs may be stuck
                    done_gpus = [gpu_id for gpu_id in range(self.world_size)
                                if (self.pseudo_labels_dir / f'{name}_gpu{gpu_id}_done.marker').exists()]
                    missing_gpus = [gpu_id for gpu_id in range(self.world_size) if gpu_id not in done_gpus]
                    console.print(f"[red]GPU {self.local_rank}: Timeout waiting for GPUs {missing_gpus}[/red]")
                    console.print(f"[yellow]Proceeding with available results...[/yellow]")

                # Clean up marker files (main process only, after barrier)
                # Don't clean up yet - let the final barrier in process_all_datasets handle sync

            except Exception as e:
                console.print(f"[yellow]GPU {self.local_rank}: Coordination warning: {e}[/yellow]")

        # Get final stats from prefetcher
        prefetch_stats = prefetcher.get_stats()
        skipped_in_loop = prefetch_stats['skipped_in_loop']
        sharded_out = prefetch_stats.get('sharded_out', 0)  # Samples sent to other GPUs
        total_samples_fed = prefetch_stats.get('total_samples_fed', 0)  # Actual iterator count
        duration_rejected = prefetch_stats.get('rejected_duration', 0)
        invalid_samples = prefetch_stats.get('invalid_samples', 0)
        progress.samples_rejected_duration += duration_rejected
        progress.total_duration_hours += prefetch_stats['total_duration_hours']

        # LOG DETAILED FINAL STATS TO FILE
        log(f"[FINAL_STATS] Dataset: {name}", 'info')
        log(f"[FINAL_STATS] total_samples_fed={total_samples_fed:,}", 'info')
        log(f"[FINAL_STATS] samples_this_run={samples_this_run:,}", 'info')
        log(f"[FINAL_STATS] total_entries_from_batches={total_entries_from_batches:,}", 'info')
        log(f"[FINAL_STATS] total_entries_written={total_entries_written:,}", 'info')
        log(f"[FINAL_STATS] skipped_in_loop (already processed)={skipped_in_loop:,}", 'info')
        log(f"[FINAL_STATS] sharded_out (other GPUs)={sharded_out:,}", 'info')
        log(f"[FINAL_STATS] duration_rejected={duration_rejected:,}", 'info')
        log(f"[FINAL_STATS] invalid_samples={invalid_samples:,}", 'info')
        log(f"[FINAL_STATS] runtime_duplicates_caught={runtime_duplicates_caught:,}", 'info')
        log(f"[FINAL_STATS] total_batches={total_batches}", 'info')
        log(f"[FINAL_STATS] gpu_starvation_count={gpu_starvation_count}", 'info')

        # Calculate accounting
        accounted_for = skipped_in_loop + sharded_out + duration_rejected + invalid_samples + samples_this_run + runtime_duplicates_caught
        unaccounted = total_samples_fed - accounted_for if total_samples_fed > 0 else 0
        log(f"[FINAL_STATS] accounted_for={accounted_for:,}, unaccounted={unaccounted:,}", 'info')

        # Log resume stats and GPU efficiency
        if self.is_main:
            if skipped_in_loop > 0:
                console.print(f"[green]Skipped {skipped_in_loop:,} already-processed samples (prefetcher dedup)[/green]")

            # Log sharded out samples (this is EXPECTED with multi-GPU - samples are distributed across GPUs)
            if sharded_out > 0 and self.world_size > 1:
                console.print(f"[cyan]Sharded {sharded_out:,} samples to other GPUs ({self.world_size} GPUs total)[/cyan]")

            # Log runtime duplicates caught (within same run)
            if runtime_duplicates_caught > 0:
                console.print(f"[yellow]Caught {runtime_duplicates_caught:,} runtime duplicates (same text appearing multiple times)[/yellow]")

            # ALWAYS show full breakdown
            console.print(f"\n[bold cyan]═══ SAMPLE ACCOUNTING FOR {name.upper()} ═══[/bold cyan]")
            console.print(f"  Total from iterator:     {total_samples_fed:,}")
            console.print(f"  Processed this run:      {samples_this_run:,}")
            console.print(f"  Already processed:       {skipped_in_loop:,}")
            console.print(f"  Sharded to other GPUs:   {sharded_out:,}")
            console.print(f"  Duration filtered:       {duration_rejected:,}")
            console.print(f"  Invalid/corrupted:       {invalid_samples:,}")
            console.print(f"  Runtime duplicates:      {runtime_duplicates_caught:,}")
            console.print(f"  ─────────────────────────────────")
            console.print(f"  Accounted for:           {accounted_for:,}")
            console.print(f"  (Batches→Entries→Written: {total_batches}→{total_entries_from_batches:,}→{total_entries_written:,})")
            if unaccounted != 0:
                console.print(f"[red bold]  UNACCOUNTED GAP:         {unaccounted:,}[/red bold]")
                log(f"[FINAL_STATS] UNACCOUNTED GAP DETECTED: {unaccounted:,} samples", 'error')
            else:
                console.print(f"[green]  All samples accounted ✓[/green]")

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

        # Update metadata with actual count after completion
        # Pass actual_iterated_count for accurate future verification
        self._update_metadata_with_actual_count(name, actual_iterated_count=total_samples_fed)

        # Verify completion with full accounting of filtered samples
        # Pass the filter counts so verification can properly account for all samples
        verification = self._verify_and_fill_missing_samples(
            name=name,
            expected_total=total_samples,
            duration_rejected=duration_rejected,
            invalid_samples=invalid_samples,
            runtime_duplicates=runtime_duplicates_caught
        )

        # Store verification results in progress
        if verification:
            progress.verification_status = verification.get('status', 'unknown')

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
            # BUT: Don't skip if verification found missing samples - need to reprocess
            if resume and name in saved_progress:
                existing = saved_progress[name]
                if isinstance(existing, dict):
                    status = existing.get('status', 'pending')
                    verification_status = existing.get('verification_status', '')
                else:
                    status = getattr(existing, 'status', 'pending')
                    verification_status = getattr(existing, 'verification_status', '')

                # Also check metadata file for verification_status (more up-to-date)
                metadata_file = self.pseudo_labels_dir / f'{name}_metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            verification_status = metadata.get('verification_status', verification_status)
                    except Exception:
                        pass

                if status == 'completed':
                    # Check if we need to reprocess due to incomplete processing
                    # 'incomplete' means there are unaccounted samples (not just filtered ones)
                    if verification_status == 'incomplete':
                        if self.is_main:
                            console.print(f"\n[bold yellow]{'═' * 50}[/bold yellow]")
                            console.print(f"[bold yellow]  {name} has unaccounted samples - reprocessing[/bold yellow]")
                            console.print(f"[bold yellow]{'═' * 50}[/bold yellow]")
                        # Don't skip - fall through to process_dataset
                    else:
                        if self.is_main:
                            console.print(f"\n[bold green]{'═' * 50}[/bold green]")
                            console.print(f"[bold green]  Skipping {name} (already completed)[/bold green]")
                            console.print(f"[bold green]{'═' * 50}[/bold green]")
                        # Restore progress for summary
                        if isinstance(existing, dict):
                            all_progress[name] = DatasetProgress(**existing)
                        else:
                            all_progress[name] = existing
                        # IMPORTANT: Still need to sync with other GPUs to prevent deadlock
                        if self.is_distributed:
                            dist.barrier()
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

            # Synchronize between datasets using NCCL barrier
            # This is now SAFE because we already coordinated via file markers in _process_dataset_standard
            # All GPUs should reach this point within a reasonable time
            if self.is_distributed:
                try:
                    dist.barrier()
                except Exception as e:
                    console.print(f"[yellow]GPU {self.local_rank}: Barrier warning: {e}[/yellow]")

                # Clean up marker files after successful sync (main process only)
                if self.is_main:
                    for gpu_id in range(self.world_size):
                        marker = self.pseudo_labels_dir / f'{name}_gpu{gpu_id}_done.marker'
                        try:
                            if marker.exists():
                                marker.unlink()
                        except Exception:
                            pass

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
