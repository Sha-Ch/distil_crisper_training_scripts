#!/usr/bin/env python3
"""
=============================================================================
Deduplication and Storage Utilities for Pseudo-Label Generation
=============================================================================

This module provides:
1. Global cross-dataset deduplication tracking
2. Audio fingerprinting for near-duplicate detection
3. Runtime duplicate detection within batches
4. Storage space monitoring and management
5. Duplicate statistics logging

Usage:
    from dedup_utils import GlobalDeduplicator, StorageManager, AudioFingerprinter

References:
- People's Speech has NO duplicate detection (critical)
- Common Voice v1 had >99% train/test overlap (fixed in newer versions)
- Cross-dataset duplicates (e.g., TED talks in both TED-LIUM and YODAS)
=============================================================================
"""

import os
import json
import hashlib
import shutil
import logging
from pathlib import Path
from typing import Set, Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import threading

import numpy as np

# Get logger from main module
logger = logging.getLogger('pseudo_labels')


def log(message: str, level: str = 'info'):
    """Log message to file."""
    if level == 'debug':
        logger.debug(message)
    elif level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)
    else:
        logger.info(message)


# =============================================================================
# Audio Fingerprinting for Near-Duplicate Detection
# =============================================================================

class AudioFingerprinter:
    """
    Generate audio fingerprints to detect near-duplicate audio.

    Uses multiple techniques:
    1. Perceptual hash based on spectral features
    2. Energy-based signature
    3. Zero-crossing rate signature

    This catches duplicates even when:
    - Audio is re-encoded (different byte representation)
    - Slight volume differences
    - Minor trimming at start/end
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        # Number of frames for fingerprint (covers ~2-3 seconds)
        self.n_frames = 32
        self.frame_size = 512

    def compute_fingerprint(self, audio: np.ndarray) -> str:
        """
        Compute a perceptual fingerprint for audio.

        Args:
            audio: Audio samples as numpy array (mono, any sample rate)

        Returns:
            64-character hex string fingerprint
        """
        if len(audio) == 0:
            return "0" * 64

        # Normalize audio
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Compute energy in frames
        n_samples = len(audio)
        frame_hop = max(1, n_samples // self.n_frames)

        energies = []
        zcrs = []  # Zero crossing rates

        for i in range(self.n_frames):
            start = i * frame_hop
            end = min(start + self.frame_size, n_samples)
            if start >= n_samples:
                break
            frame = audio[start:end]

            # Energy
            energy = np.sum(frame ** 2) / len(frame) if len(frame) > 0 else 0
            energies.append(energy)

            # Zero crossing rate
            if len(frame) > 1:
                zcr = np.sum(np.abs(np.diff(np.signbit(frame)))) / len(frame)
            else:
                zcr = 0
            zcrs.append(zcr)

        # Pad to fixed length
        while len(energies) < self.n_frames:
            energies.append(0)
            zcrs.append(0)

        # Create binary hash from comparisons
        # Compare each frame to mean (like perceptual hashing)
        energy_mean = np.mean(energies)
        zcr_mean = np.mean(zcrs)

        bits = []
        for i in range(self.n_frames):
            bits.append('1' if energies[i] > energy_mean else '0')
            bits.append('1' if zcrs[i] > zcr_mean else '0')

        # Convert to hex
        bit_string = ''.join(bits)
        # Pad to multiple of 4 for hex conversion
        while len(bit_string) % 4 != 0:
            bit_string += '0'

        hex_string = hex(int(bit_string, 2))[2:].zfill(16)

        # Also add content hash for exact matches
        content_hash = hashlib.md5(audio[:8000].tobytes()).hexdigest()[:16]

        return f"{hex_string}{content_hash}"[:32]

    def hamming_distance(self, fp1: str, fp2: str) -> int:
        """
        Compute Hamming distance between two fingerprints.

        Lower distance = more similar audio.
        Threshold of ~8-12 typically indicates near-duplicates.
        """
        if len(fp1) != len(fp2):
            return 999

        # Convert hex to binary
        try:
            bin1 = bin(int(fp1[:16], 16))[2:].zfill(64)
            bin2 = bin(int(fp2[:16], 16))[2:].zfill(64)
            return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
        except ValueError:
            return 999

    def is_near_duplicate(self, fp1: str, fp2: str, threshold: int = 10) -> bool:
        """Check if two fingerprints represent near-duplicate audio."""
        return self.hamming_distance(fp1, fp2) <= threshold


# =============================================================================
# Global Cross-Dataset Deduplication
# =============================================================================

@dataclass
class DuplicateStats:
    """Statistics about duplicates found."""
    total_checked: int = 0
    exact_text_duplicates: int = 0
    near_audio_duplicates: int = 0
    cross_dataset_duplicates: int = 0
    within_batch_duplicates: int = 0
    duplicates_by_dataset: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GlobalDeduplicator:
    """
    Global deduplication tracker across all datasets.

    Maintains:
    1. Set of processed ground_truth texts (exact match)
    2. Set of audio fingerprints (near-duplicate detection)
    3. Statistics on duplicates found

    Thread-safe for multi-GPU processing.
    """

    def __init__(
        self,
        cache_dir: Path,
        enable_audio_fingerprinting: bool = True,
        fingerprint_threshold: int = 10,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.enable_audio_fingerprinting = enable_audio_fingerprinting
        self.fingerprint_threshold = fingerprint_threshold

        # Thread-safe sets
        self._lock = threading.Lock()
        self._processed_texts: Set[str] = set()
        self._audio_fingerprints: Dict[str, str] = {}  # fingerprint -> dataset:sample_id
        self._stats = DuplicateStats()

        # Audio fingerprinter
        self._fingerprinter = AudioFingerprinter() if enable_audio_fingerprinting else None

        # Load existing data
        self._load_cache()

    def _load_cache(self):
        """Load cached deduplication data."""
        # Load processed texts
        texts_cache = self.cache_dir / 'global_processed_texts.json'
        if texts_cache.exists():
            try:
                with open(texts_cache, 'r', encoding='utf-8') as f:
                    self._processed_texts = set(json.load(f))
                log(f"[DEDUP] Loaded {len(self._processed_texts):,} processed texts from global cache", 'info')
            except Exception as e:
                log(f"[DEDUP] Failed to load texts cache: {e}", 'warning')

        # Load audio fingerprints
        fp_cache = self.cache_dir / 'global_audio_fingerprints.json'
        if fp_cache.exists():
            try:
                with open(fp_cache, 'r', encoding='utf-8') as f:
                    self._audio_fingerprints = json.load(f)
                log(f"[DEDUP] Loaded {len(self._audio_fingerprints):,} audio fingerprints from global cache", 'info')
            except Exception as e:
                log(f"[DEDUP] Failed to load fingerprints cache: {e}", 'warning')

        # Load stats
        stats_cache = self.cache_dir / 'global_dedup_stats.json'
        if stats_cache.exists():
            try:
                with open(stats_cache, 'r', encoding='utf-8') as f:
                    stats_dict = json.load(f)
                    self._stats = DuplicateStats(**stats_dict)
                log(f"[DEDUP] Loaded dedup stats from cache", 'info')
            except Exception as e:
                log(f"[DEDUP] Failed to load stats cache: {e}", 'warning')

    def save_cache(self):
        """Save deduplication data to cache."""
        with self._lock:
            # Save processed texts
            texts_cache = self.cache_dir / 'global_processed_texts.json'
            try:
                with open(texts_cache, 'w', encoding='utf-8') as f:
                    json.dump(list(self._processed_texts), f)
            except Exception as e:
                log(f"[DEDUP] Failed to save texts cache: {e}", 'warning')

            # Save audio fingerprints
            fp_cache = self.cache_dir / 'global_audio_fingerprints.json'
            try:
                with open(fp_cache, 'w', encoding='utf-8') as f:
                    json.dump(self._audio_fingerprints, f)
            except Exception as e:
                log(f"[DEDUP] Failed to save fingerprints cache: {e}", 'warning')

            # Save stats
            stats_cache = self.cache_dir / 'global_dedup_stats.json'
            try:
                with open(stats_cache, 'w', encoding='utf-8') as f:
                    json.dump(self._stats.to_dict(), f, indent=2)
            except Exception as e:
                log(f"[DEDUP] Failed to save stats cache: {e}", 'warning')

        log(f"[DEDUP] Saved global dedup cache: {len(self._processed_texts):,} texts, {len(self._audio_fingerprints):,} fingerprints", 'info')

    def check_and_add(
        self,
        ground_truth: str,
        audio: Optional[np.ndarray],
        dataset_name: str,
        sample_id: str,
    ) -> Tuple[bool, str]:
        """
        Check if sample is duplicate and add to tracking.

        Args:
            ground_truth: Transcript text
            audio: Audio samples (optional, for fingerprinting)
            dataset_name: Name of source dataset
            sample_id: Unique sample identifier

        Returns:
            Tuple of (is_duplicate, reason)
            - is_duplicate: True if this is a duplicate
            - reason: Why it's a duplicate (or empty string)
        """
        with self._lock:
            self._stats.total_checked += 1

            # Normalize text
            text_normalized = ground_truth.strip().lower()

            # Check exact text match
            if text_normalized in self._processed_texts:
                self._stats.exact_text_duplicates += 1
                self._stats.duplicates_by_dataset[dataset_name] = \
                    self._stats.duplicates_by_dataset.get(dataset_name, 0) + 1
                return True, "exact_text_match"

            # Check audio fingerprint (if enabled and audio provided)
            if self.enable_audio_fingerprinting and audio is not None and self._fingerprinter:
                fingerprint = self._fingerprinter.compute_fingerprint(audio)

                # Check for near-duplicate audio
                for existing_fp, existing_id in self._audio_fingerprints.items():
                    if self._fingerprinter.is_near_duplicate(
                        fingerprint, existing_fp, self.fingerprint_threshold
                    ):
                        self._stats.near_audio_duplicates += 1
                        existing_dataset = existing_id.split(':')[0] if ':' in existing_id else 'unknown'
                        if existing_dataset != dataset_name:
                            self._stats.cross_dataset_duplicates += 1
                        return True, f"near_audio_duplicate_of_{existing_id}"

                # Add fingerprint
                self._audio_fingerprints[fingerprint] = f"{dataset_name}:{sample_id}"

            # Add text to processed set
            self._processed_texts.add(text_normalized)

            return False, ""

    def check_batch_duplicates(
        self,
        batch_texts: List[str],
    ) -> List[int]:
        """
        Check for duplicates within a batch.

        Args:
            batch_texts: List of ground_truth texts in batch

        Returns:
            List of indices that are duplicates within the batch
        """
        seen = {}
        duplicates = []

        for i, text in enumerate(batch_texts):
            text_normalized = text.strip().lower()
            if text_normalized in seen:
                duplicates.append(i)
                with self._lock:
                    self._stats.within_batch_duplicates += 1
            else:
                seen[text_normalized] = i

        return duplicates

    def get_stats(self) -> DuplicateStats:
        """Get current duplicate statistics."""
        with self._lock:
            return DuplicateStats(
                total_checked=self._stats.total_checked,
                exact_text_duplicates=self._stats.exact_text_duplicates,
                near_audio_duplicates=self._stats.near_audio_duplicates,
                cross_dataset_duplicates=self._stats.cross_dataset_duplicates,
                within_batch_duplicates=self._stats.within_batch_duplicates,
                duplicates_by_dataset=dict(self._stats.duplicates_by_dataset),
            )

    def log_stats(self):
        """Log current duplicate statistics."""
        stats = self.get_stats()
        log(f"[DEDUP STATS] Total checked: {stats.total_checked:,}", 'info')
        log(f"[DEDUP STATS] Exact text duplicates: {stats.exact_text_duplicates:,}", 'info')
        log(f"[DEDUP STATS] Near audio duplicates: {stats.near_audio_duplicates:,}", 'info')
        log(f"[DEDUP STATS] Cross-dataset duplicates: {stats.cross_dataset_duplicates:,}", 'info')
        log(f"[DEDUP STATS] Within-batch duplicates: {stats.within_batch_duplicates:,}", 'info')
        if stats.duplicates_by_dataset:
            log(f"[DEDUP STATS] By dataset: {stats.duplicates_by_dataset}", 'info')

    @property
    def processed_count(self) -> int:
        """Number of unique samples processed."""
        with self._lock:
            return len(self._processed_texts)


# =============================================================================
# Storage Space Management
# =============================================================================

class StorageManager:
    """
    Monitor and manage storage space for chunked downloads.

    Features:
    1. Check available space before downloads
    2. Automatic cleanup when space is low
    3. Track space usage per dataset
    4. Emergency cleanup if space runs critically low
    """

    def __init__(
        self,
        cache_dir: str,
        output_dir: str,
        min_free_space_gb: float = 50.0,
        emergency_free_space_gb: float = 20.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.min_free_space_gb = min_free_space_gb
        self.emergency_free_space_gb = emergency_free_space_gb

        # Track space usage
        self._space_log: List[Dict[str, Any]] = []

    def get_free_space_gb(self, path: Optional[Path] = None) -> float:
        """Get free space in GB for the given path."""
        if path is None:
            path = self.cache_dir
        try:
            stat = shutil.disk_usage(path)
            return stat.free / (1024 ** 3)
        except Exception as e:
            log(f"[STORAGE] Failed to get disk usage: {e}", 'warning')
            return 999.0  # Assume plenty of space if we can't check

    def get_cache_size_gb(self) -> float:
        """Get total size of HuggingFace cache in GB."""
        hf_cache = Path(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')))
        return self._get_dir_size_gb(hf_cache)

    def _get_dir_size_gb(self, path: Path) -> float:
        """Get total size of a directory in GB."""
        if not path.exists():
            return 0.0
        try:
            total = sum(
                f.stat().st_size
                for f in path.rglob('*')
                if f.is_file()
            )
            return total / (1024 ** 3)
        except Exception:
            return 0.0

    def has_enough_space(self, required_gb: float) -> bool:
        """Check if there's enough free space for a download."""
        free_space = self.get_free_space_gb()
        has_space = free_space >= (required_gb + self.min_free_space_gb)

        if not has_space:
            log(f"[STORAGE] Insufficient space: {free_space:.1f}GB free, need {required_gb + self.min_free_space_gb:.1f}GB", 'warning')

        return has_space

    def check_and_cleanup_if_needed(self, required_gb: float = 0) -> bool:
        """
        Check space and cleanup if needed.

        Returns True if there's enough space (after cleanup if performed).
        """
        free_space = self.get_free_space_gb()

        # Log current status
        cache_size = self.get_cache_size_gb()
        log(f"[STORAGE] Status: {free_space:.1f}GB free, {cache_size:.1f}GB in cache", 'info')

        # Emergency cleanup if space is critically low
        if free_space < self.emergency_free_space_gb:
            log(f"[STORAGE] EMERGENCY: Only {free_space:.1f}GB free! Running emergency cleanup...", 'warning')
            self.emergency_cleanup()
            free_space = self.get_free_space_gb()

        # Normal cleanup if below minimum
        if free_space < self.min_free_space_gb + required_gb:
            log(f"[STORAGE] Low space: {free_space:.1f}GB. Running cleanup...", 'warning')
            self.cleanup_old_cache()
            free_space = self.get_free_space_gb()

        return free_space >= (required_gb + self.min_free_space_gb)

    def cleanup_old_cache(self):
        """Clean up old HuggingFace cache entries."""
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()

            # Sort by last accessed time, delete oldest first
            repos_by_age = []
            for repo in cache_info.repos:
                for revision in repo.revisions:
                    repos_by_age.append((revision.last_accessed, repo, revision))

            repos_by_age.sort(key=lambda x: x[0])  # Oldest first

            freed = 0.0
            for last_accessed, repo, revision in repos_by_age:
                if self.get_free_space_gb() >= self.min_free_space_gb:
                    break

                size_gb = revision.size_on_disk / (1024 ** 3)
                try:
                    delete_strategy = cache_info.delete_revisions(revision.commit_hash)
                    delete_strategy.execute()
                    freed += size_gb
                    log(f"[STORAGE] Deleted {repo.repo_id} revision ({size_gb:.1f}GB)", 'info')
                except Exception as e:
                    log(f"[STORAGE] Failed to delete {repo.repo_id}: {e}", 'warning')

            if freed > 0:
                log(f"[STORAGE] Cleanup complete: freed {freed:.1f}GB", 'info')

        except Exception as e:
            log(f"[STORAGE] Cleanup failed: {e}", 'warning')

    def emergency_cleanup(self):
        """
        Emergency cleanup when space is critically low.

        More aggressive than normal cleanup:
        1. Delete all dataset caches
        2. Clear HuggingFace hub cache
        3. Clear any temporary files
        """
        log("[STORAGE] Running EMERGENCY cleanup...", 'warning')

        freed_total = 0.0

        # Clear HF datasets cache
        hf_cache = Path(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')))
        datasets_cache = hf_cache / 'datasets'

        if datasets_cache.exists():
            size_before = self._get_dir_size_gb(datasets_cache)
            try:
                shutil.rmtree(datasets_cache)
                freed_total += size_before
                log(f"[STORAGE] Cleared datasets cache: {size_before:.1f}GB", 'info')
            except Exception as e:
                log(f"[STORAGE] Failed to clear datasets cache: {e}", 'warning')

        # Clear HF hub cache (downloaded model files, etc.)
        hub_cache = hf_cache / 'hub'
        if hub_cache.exists():
            size_before = self._get_dir_size_gb(hub_cache)
            # Only delete .incomplete files and old downloads, not model weights
            for item in hub_cache.glob('**/.incomplete'):
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception:
                    pass

        log(f"[STORAGE] Emergency cleanup complete: freed ~{freed_total:.1f}GB", 'warning')

    def log_space_status(self):
        """Log current storage space status."""
        free_space = self.get_free_space_gb()
        cache_size = self.get_cache_size_gb()

        status = "OK" if free_space >= self.min_free_space_gb else "LOW"
        if free_space < self.emergency_free_space_gb:
            status = "CRITICAL"

        log(f"[STORAGE] Free: {free_space:.1f}GB | Cache: {cache_size:.1f}GB | Status: {status}", 'info')

        self._space_log.append({
            'timestamp': datetime.now().isoformat(),
            'free_gb': free_space,
            'cache_gb': cache_size,
            'status': status,
        })


# =============================================================================
# Integration Helper
# =============================================================================

def create_dedup_and_storage_managers(
    output_dir: Path,
    enable_audio_fingerprinting: bool = True,
    min_free_space_gb: float = 50.0,
) -> Tuple[GlobalDeduplicator, StorageManager]:
    """
    Create deduplication and storage managers.

    Args:
        output_dir: Directory for output files and caches
        enable_audio_fingerprinting: Whether to enable audio fingerprinting
        min_free_space_gb: Minimum free space to maintain

    Returns:
        Tuple of (GlobalDeduplicator, StorageManager)
    """
    cache_dir = Path(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')))

    deduplicator = GlobalDeduplicator(
        cache_dir=output_dir,
        enable_audio_fingerprinting=enable_audio_fingerprinting,
    )

    storage_manager = StorageManager(
        cache_dir=str(cache_dir),
        output_dir=str(output_dir),
        min_free_space_gb=min_free_space_gb,
    )

    return deduplicator, storage_manager
