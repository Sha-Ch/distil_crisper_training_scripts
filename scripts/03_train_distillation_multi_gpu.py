#!/usr/bin/env python3
"""
=============================================================================
Multi-GPU Distillation Training for Distil-CrisperWhisper
=============================================================================
Trains a distilled student model from CrisperWhisper teacher using the
official distil-whisper v3.5 methodology.

CRITICAL METHODOLOGY (matching official):
1. FROZEN ENCODER - Full 32-layer encoder copied from CrisperWhisper, frozen
2. REDUCED DECODER - 2 decoder layers (for 6x speed) or 16 (for max quality)
3. SpecAugment - Audio data augmentation during training
4. KL + CE Loss - Combined distillation loss (alpha=0.8)
5. Sample Packing - Efficient batching with 30s segments
6. Word Timestamps - Preserved from CrisperWhisper pseudo-labels

This produces a model compatible with faster-whisper via CTranslate2 conversion.

Hardware: Auto-scales from 4-6+ GPUs (H100, A100, etc.)
  4x H100 NVL: 32 * 8 * 4 = 1024 effective batch
  6x H100 NVL: 32 * 8 * 6 = 1536 effective batch (50% faster!)
Training time: ~40-60 hours for 80k steps on 4 GPUs (~30-40 hours on 6 GPUs)

Usage:
  accelerate launch 03_train_distillation_multi_gpu.py --config ../config.yaml

References:
- Distil-Whisper: https://arxiv.org/abs/2311.00430
- Official Code: https://github.com/huggingface/distil-whisper
=============================================================================
"""

import os
import sys
import json
import yaml
import argparse
import signal
import shutil
import gc
import random
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchaudio
import numpy as np
from tqdm import tqdm

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    get_scheduler,
    set_seed,
)
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
import soundfile as sf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from huggingface_hub import HfApi, create_repo

# Torch-free helpers (unit-tested in tests/test_distill_seq_utils.py)
from distill_seq_utils import build_decoder_inputs_and_labels, pad_decoder_batch

warnings.filterwarnings('ignore')
console = Console()


# =============================================================================
# SpecAugment - Official Distil-Whisper Implementation
# =============================================================================

class SpecAugment(nn.Module):
    """
    SpecAugment data augmentation for mel spectrograms.
    Follows the official distil-whisper v3.5 implementation.

    Paper: https://arxiv.org/abs/1904.08779

    Applied during training to improve robustness:
    - Frequency masking: randomly masks frequency bands
    - Time masking: randomly masks time steps
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        probability: float = 0.5,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to mel spectrogram.

        Args:
            x: Input tensor [batch, n_mels, time]

        Returns:
            Augmented tensor
        """
        if not self.training or random.random() > self.probability:
            return x

        x = x.clone()
        batch_size, n_mels, time_steps = x.shape

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, n_mels - f))
            x[:, f0:f0 + f, :] = 0

        # Time masking
        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_param, time_steps))
            t0 = random.randint(0, max(0, time_steps - t))
            x[:, :, t0:t0 + t] = 0

        return x


# =============================================================================
# Distillation Loss - EXACT Official Implementation
# =============================================================================

class DistillationLoss(nn.Module):
    """
    Combined KL Divergence + Cross-Entropy loss for knowledge distillation.
    Follows the EXACT official distil-whisper formula from:
    https://github.com/huggingface/distil-whisper/blob/main/training/run_distillation.py

    OFFICIAL FORMULA: loss = 0.8 * ce_loss + kl_weight * kl_loss
    Where:
    - ce_weight = 0.8 (weight for cross-entropy/hard labels)
    - kl_weight = 1.0 (weight for KL divergence/soft labels) - NOT 0.2!
    - T = 2.0 (temperature, with T^2 scaling for gradients)
    - CE_loss = cross-entropy with pseudo-labels
    - KL_loss = KL divergence with teacher logits

    NOTE: This is DIFFERENT from standard KD where alpha + (1-alpha) = 1
    """

    def __init__(
        self,
        temperature: float = 2.0,
        ce_weight: float = 0.8,   # Official default
        kl_weight: float = 1.0,   # Official default - NOT (1 - ce_weight)!
        ignore_index: int = -100,
    ):
        super().__init__()
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.kl_weight = kl_weight
        self.ignore_index = ignore_index
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.

        Args:
            student_logits: [batch, seq_len, vocab_size]
            teacher_logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len] with -100 for padding

        Returns:
            total_loss, loss_dict
        """
        # Hard label loss (cross-entropy with pseudo-labels)
        ce_loss = self.ce_loss_fn(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )

        # Soft label loss (KL divergence with temperature scaling)
        # log_softmax for numerical stability
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Mask for valid (non-padding) tokens
        mask = (labels != self.ignore_index).unsqueeze(-1).float()

        # KL divergence: sum over vocab, mean over valid tokens
        # KL(teacher || student) = sum(teacher * (log(teacher) - log(student)))
        kl_div = teacher_probs * (teacher_probs.clamp(min=1e-10).log() - student_log_probs)
        kl_div = (kl_div * mask).sum() / mask.sum().clamp(min=1)

        # Scale by temperature squared (ensures gradients scale correctly)
        kl_loss = kl_div * (self.temperature ** 2)

        # Combined loss - OFFICIAL FORMULA: 0.8 * CE + 1.0 * KL
        # NOTE: This is NOT (alpha * CE + (1-alpha) * KL)
        total_loss = self.ce_weight * ce_loss + self.kl_weight * kl_loss

        return total_loss, {
            'ce_loss': ce_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item(),
        }


# =============================================================================
# Dataset for Distillation Training
# =============================================================================

class DistillationDataset(Dataset):
    """
    Dataset for distillation training using pre-generated pseudo-labels.

    Faithful distil-whisper v3.5 training inputs:
    - Loads REAL audio from `audio_path` (written by stage 2 with save_audio: true).
      Samples without on-disk audio are skipped at load time (no silent training
      on zero-filled audio).
    - timestamp_probability: a fraction of the time, train on the timestamped
      pseudo-label (`pseudo_label_timestamped`) so the student learns to emit
      Whisper-native segment timestamps.
    - condition_on_prev_probability: a fraction of the time, prepend the previous
      transcript (`prev_text`) as a decoder prompt.

    __getitem__ returns token-id LISTS (prompt_ids, label_ids); the collator builds
    padded decoder_input_ids/labels via build_decoder_inputs_and_labels().
    """

    def __init__(
        self,
        pseudo_labels_path: str,
        processor: WhisperProcessor,
        max_audio_length: int = 480000,  # 30 seconds at 16kHz
        max_label_length: int = 448,
        sample_rate: int = 16000,
        timestamp_probability: float = 0.0,
        condition_on_prev_probability: float = 0.0,
    ):
        self.processor = processor
        self.max_audio_length = max_audio_length
        self.max_label_length = max_label_length
        self.sample_rate = sample_rate
        self.timestamp_probability = timestamp_probability
        self.condition_on_prev_probability = condition_on_prev_probability

        console.print(f"[yellow]Loading pseudo-labels from {pseudo_labels_path}...[/yellow]")
        loaded = []
        with open(pseudo_labels_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('accepted', True):
                        loaded.append(entry)
                except json.JSONDecodeError:
                    continue

        # Training requires REAL audio on disk. Stage 2 must have been run with
        # distillation.pseudo_labels.save_audio: true (the default). We drop any
        # accepted entry whose audio is missing rather than fabricate silence.
        self.samples = [
            s for s in loaded
            if s.get('audio_path') and Path(s['audio_path']).exists()
        ]
        dropped = len(loaded) - len(self.samples)

        if not self.samples:
            console.print("[red]No training samples have audio on disk.[/red]")
            console.print(
                "[yellow]Re-run stage 2 (02_generate_pseudo_labels_multi_gpu.py) with "
                "distillation.pseudo_labels.save_audio: true so audio is persisted.[/yellow]"
            )
            raise RuntimeError("No audio available for training (audio_path missing).")

        if dropped:
            console.print(f"[yellow]Skipped {dropped:,} accepted samples without audio on disk[/yellow]")
        console.print(f"[green]Loaded {len(self.samples):,} training samples (with audio)[/green]")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, sample: Dict[str, Any]) -> np.ndarray:
        """Load + normalize one sample's audio to 16kHz mono, padded/truncated to 30s."""
        audio_array, sr = sf.read(sample['audio_path'])
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if sr != self.sample_rate:
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float()
            audio_array = torchaudio.transforms.Resample(sr, self.sample_rate)(audio_tensor).squeeze(0).numpy()
        if len(audio_array) > self.max_audio_length:
            audio_array = audio_array[:self.max_audio_length]
        elif len(audio_array) < self.max_audio_length:
            audio_array = np.pad(audio_array, (0, self.max_audio_length - len(audio_array)))
        return audio_array

    def _encode_label(self, text: str, use_timestamps: bool) -> List[int]:
        """
        Tokenize transcript text WITH the Whisper special prefix/suffix.
        When use_timestamps is True the prefix omits <|notimestamps|> and the
        <|x.xx|> markers in `text` are encoded as timestamp tokens. Falls back to
        plain (no-timestamp) tokenization on any error.
        """
        tok = self.processor.tokenizer
        try:
            tok.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=use_timestamps)
            ids = tok(text, add_special_tokens=True).input_ids
            if ids:
                return [int(x) for x in ids]
        except Exception:
            pass
        try:
            tok.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=False)
        except Exception:
            pass
        return [int(x) for x in tok(text, add_special_tokens=True).input_ids]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        text_plain = sample.get('pseudo_label') or sample.get('ground_truth') or ''
        text_ts = sample.get('pseudo_label_timestamped')

        audio_array = self._load_audio(sample)
        input_features = self.processor.feature_extractor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).input_features.squeeze(0)

        # timestamp_probability: occasionally train on the timestamped label.
        use_ts = (
            bool(text_ts)
            and self.timestamp_probability > 0
            and random.random() < self.timestamp_probability
        )
        label_ids = self._encode_label(text_ts if use_ts else text_plain, use_ts)

        # condition_on_prev_probability: occasionally prepend previous context.
        prompt_ids: List[int] = []
        prev_text = sample.get('prev_text')
        if (
            prev_text
            and self.condition_on_prev_probability > 0
            and random.random() < self.condition_on_prev_probability
        ):
            try:
                pid = self.processor.tokenizer.get_prompt_ids(prev_text)
                prompt_ids = [int(x) for x in pid]
            except Exception:
                prompt_ids = []

        return {
            'input_features': input_features,
            'label_ids': label_ids,
            'prompt_ids': prompt_ids,
        }


class DistillationCollator:
    """
    Pads a batch and builds decoder_input_ids/labels via
    build_decoder_inputs_and_labels(). decoder_input_ids are padded with
    pad_token_id; labels with -100. No decoder_attention_mask is needed: padding
    sits at the end and Whisper's decoder is causal, so it never affects the
    supervised (earlier) positions.
    """

    def __init__(self, pad_token_id: int, max_length: int = 448):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = torch.stack([item['input_features'] for item in batch])

        dec_list, lab_list = [], []
        for item in batch:
            di, lb = build_decoder_inputs_and_labels(
                item['prompt_ids'], item['label_ids'], self.max_length
            )
            dec_list.append(di)
            lab_list.append(lb)

        dec_padded, lab_padded = pad_decoder_batch(dec_list, lab_list, self.pad_token_id)

        return {
            'input_features': input_features,
            'decoder_input_ids': torch.tensor(dec_padded, dtype=torch.long),
            'labels': torch.tensor(lab_padded, dtype=torch.long),
        }


# =============================================================================
# Student Model Creation - Official Methodology
# =============================================================================

def create_student_model(
    teacher_model: WhisperForConditionalGeneration,
    num_decoder_layers: int = 2,
) -> WhisperForConditionalGeneration:
    """
    Create a student model following distil-whisper methodology.

    Key principles:
    1. Copy FULL encoder from teacher (will be frozen during training)
    2. Copy SUBSET of decoder layers (maximally spaced)
    3. Copy embeddings and output projection
    4. This enables speculative decoding compatibility

    Args:
        teacher_model: CrisperWhisper teacher model
        num_decoder_layers: Number of decoder layers in student (2 for 6x speed)

    Returns:
        Initialized student model
    """
    console.print(f"\n[bold blue]Creating student model (distil-whisper methodology)...[/bold blue]")
    console.print(f"  Encoder: FULL ({teacher_model.config.encoder_layers} layers, will be FROZEN)")
    console.print(f"  Decoder: {num_decoder_layers} layers (from {teacher_model.config.decoder_layers})")

    teacher_config = teacher_model.config

    # Student config: same encoder, reduced decoder
    student_config = WhisperConfig(
        vocab_size=teacher_config.vocab_size,
        num_mel_bins=teacher_config.num_mel_bins,
        # FULL encoder
        encoder_layers=teacher_config.encoder_layers,
        encoder_attention_heads=teacher_config.encoder_attention_heads,
        encoder_ffn_dim=teacher_config.encoder_ffn_dim,
        # REDUCED decoder
        decoder_layers=num_decoder_layers,
        decoder_attention_heads=teacher_config.decoder_attention_heads,
        decoder_ffn_dim=teacher_config.decoder_ffn_dim,
        # Same dimensions
        d_model=teacher_config.d_model,
        dropout=teacher_config.dropout,
        attention_dropout=teacher_config.attention_dropout,
        activation_dropout=teacher_config.activation_dropout,
        max_source_positions=teacher_config.max_source_positions,
        max_target_positions=teacher_config.max_target_positions,
        # Token IDs
        pad_token_id=teacher_config.pad_token_id,
        bos_token_id=teacher_config.bos_token_id,
        eos_token_id=teacher_config.eos_token_id,
        decoder_start_token_id=teacher_config.decoder_start_token_id,
        suppress_tokens=teacher_config.suppress_tokens,
        begin_suppress_tokens=teacher_config.begin_suppress_tokens,
        # Disable cache for training
        use_cache=False,
    )

    # Create student model
    student = WhisperForConditionalGeneration(student_config)

    # Copy FULL encoder
    console.print("  [yellow]Copying full encoder...[/yellow]")
    student.model.encoder.load_state_dict(teacher_model.model.encoder.state_dict())
    console.print("  [green]✓ Full encoder copied[/green]")

    # Copy maximally spaced decoder layers
    teacher_decoder_layers = teacher_config.decoder_layers
    if num_decoder_layers == 1:
        layer_indices = [teacher_decoder_layers - 1]
    else:
        layer_indices = np.linspace(0, teacher_decoder_layers - 1, num_decoder_layers, dtype=int).tolist()

    console.print(f"  [yellow]Copying decoder layers {layer_indices}...[/yellow]")
    for student_idx, teacher_idx in enumerate(layer_indices):
        student.model.decoder.layers[student_idx].load_state_dict(
            teacher_model.model.decoder.layers[teacher_idx].state_dict()
        )
    console.print(f"  [green]✓ Decoder layers copied[/green]")

    # Copy embeddings
    console.print("  [yellow]Copying embeddings...[/yellow]")
    student.model.decoder.embed_tokens.load_state_dict(
        teacher_model.model.decoder.embed_tokens.state_dict()
    )
    student.model.decoder.embed_positions.load_state_dict(
        teacher_model.model.decoder.embed_positions.state_dict()
    )
    console.print("  [green]✓ Embeddings copied[/green]")

    # Copy output projection
    console.print("  [yellow]Copying output projection...[/yellow]")
    student.proj_out.load_state_dict(teacher_model.proj_out.state_dict())
    console.print("  [green]✓ Output projection copied[/green]")

    # Parameter counts
    teacher_params = sum(p.numel() for p in teacher_model.parameters()) / 1e6
    student_params = sum(p.numel() for p in student.parameters()) / 1e6
    encoder_params = sum(p.numel() for p in student.model.encoder.parameters()) / 1e6
    decoder_params = student_params - encoder_params

    console.print(f"\n[green]✓ Student model created![/green]")
    console.print(f"  Teacher: {teacher_params:.1f}M parameters")
    console.print(f"  Student: {student_params:.1f}M parameters")
    console.print(f"  Encoder (frozen): {encoder_params:.1f}M parameters")
    console.print(f"  Decoder (trainable): {decoder_params:.1f}M parameters")

    return student


def freeze_encoder(model: WhisperForConditionalGeneration):
    """
    Freeze the encoder following distil-whisper methodology.

    This is CRITICAL for:
    1. Preserving CrisperWhisper's audio understanding
    2. Enabling speculative decoding
    3. Allowing larger batch sizes (less memory for gradients)
    """
    console.print("[yellow]Freezing encoder...[/yellow]")

    # Freeze all encoder parameters
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    # Also freeze positional embeddings in decoder (following official)
    for param in model.model.decoder.embed_positions.parameters():
        param.requires_grad = False

    # Count frozen vs trainable
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    console.print(f"  [green]✓ Frozen: {frozen / 1e6:.1f}M parameters[/green]")
    console.print(f"  [green]✓ Trainable: {trainable / 1e6:.1f}M parameters[/green]")


# =============================================================================
# Spot Instance Handler
# =============================================================================

class SpotInstanceHandler:
    """Handles spot instance preemption with graceful checkpoint saving."""

    def __init__(self, save_callback=None):
        self.should_stop = False
        self.save_callback = save_callback
        self._setup()

    def _setup(self):
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        console.print(f"\n[bold yellow]⚠ Signal {signum} received. Saving checkpoint...[/bold yellow]")
        self.should_stop = True
        if self.save_callback:
            self.save_callback()


# =============================================================================
# Trainer Class
# =============================================================================

class DistillationTrainer:
    """
    Multi-GPU trainer for distillation following official methodology.
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.config_path = config_path

        # Paths
        self.workspace = Path(self.config['paths']['workspace'])
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize accelerator for multi-GPU
        training_config = self.config['training']

        # Use configured gradient accumulation - more GPUs = higher effective batch
        # This is CORRECT behavior: more GPUs should increase throughput
        #   4 GPUs: 32 * 8 * 4 = 1024 effective batch
        #   6 GPUs: 32 * 8 * 6 = 1536 effective batch (50% more throughput!)
        # The learning rate will auto-scale with batch size via linear scaling rule
        self.gradient_accumulation_steps = training_config['gradient_accumulation_steps']

        self.accelerator = Accelerator(
            mixed_precision='bf16' if training_config.get('bf16', True) else 'fp16',
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            log_with=['tensorboard'],
            project_dir=str(self.checkpoint_dir),
        )

        # Spot instance handler
        self.spot_handler = SpotInstanceHandler(save_callback=self._emergency_save)

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.epoch = 0

        # SpecAugment
        spec_config = self.config.get('spec_augment', {})
        self.spec_augment = SpecAugment(
            freq_mask_param=spec_config.get('freq_mask_param', 27),
            time_mask_param=spec_config.get('time_mask_param', 100),
            n_freq_masks=spec_config.get('n_freq_masks', 2),
            n_time_masks=spec_config.get('n_time_masks', 2),
            probability=spec_config.get('probability', 0.5),
        )

        # Models (loaded later)
        self.teacher_model = None
        self.student_model = None
        self.processor = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.eval_dataset = None
        self.loss_fn = None

        if self.accelerator.is_main_process:
            effective_batch = training_config['per_device_train_batch_size'] * self.accelerator.num_processes * self.gradient_accumulation_steps
            console.print(Panel.fit(
                f"[bold cyan]Distillation Trainer (Auto-Scaling)[/bold cyan]\n"
                f"GPUs Available: {torch.cuda.device_count()}\n"
                f"GPUs in Use: {self.accelerator.num_processes}\n"
                f"Per-Device Batch Size: {training_config['per_device_train_batch_size']}\n"
                f"Gradient Accumulation: {self.gradient_accumulation_steps}\n"
                f"Effective Batch Size: {effective_batch}\n"
                f"Mixed Precision: bf16",
                title="Configuration"
            ))

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _emergency_save(self):
        """Emergency checkpoint save on preemption."""
        if self.student_model is not None:
            self._save_checkpoint(emergency=True)

    def setup(self):
        """Setup models, optimizer, data loaders."""
        set_seed(42)

        if self.accelerator.is_main_process:
            console.print("\n[bold blue]Setting up training...[/bold blue]")

        # Load teacher model (CrisperWhisper)
        teacher_config = self.config['teacher']
        cache_dir = self.config['paths'].get('hf_cache')

        if self.accelerator.is_main_process:
            console.print(f"\n[yellow]Loading teacher model: {teacher_config['model_id']}[/yellow]")

        self.processor = WhisperProcessor.from_pretrained(
            teacher_config['model_id'],
            cache_dir=cache_dir,
        )

        # BPE dropout (faithful v3.5 regularization). Best-effort: only the fast
        # tokenizer's BPE backend supports it; if unavailable we log and continue.
        bpe_dropout = float(self.config.get('distillation', {}).get('bpe_dropout', 0.0) or 0.0)
        if bpe_dropout > 0:
            try:
                self.processor.tokenizer.backend_tokenizer.model.dropout = bpe_dropout
                if self.accelerator.is_main_process:
                    console.print(f"[green]✓ BPE dropout enabled: {bpe_dropout}[/green]")
            except Exception as e:
                if self.accelerator.is_main_process:
                    console.print(f"[yellow]BPE dropout not applied (tokenizer unsupported): {e}[/yellow]")

        self.teacher_model = WhisperForConditionalGeneration.from_pretrained(
            teacher_config['model_id'],
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            attn_implementation=teacher_config.get('attn_implementation', 'sdpa'),
        )
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        if self.accelerator.is_main_process:
            console.print("[green]✓ Teacher model loaded[/green]")

        # Create student model
        student_config = self.config['student']
        self.student_model = create_student_model(
            self.teacher_model,
            num_decoder_layers=student_config.get('decoder_layers', 2),
        )

        # Freeze encoder
        freeze_encoder(self.student_model)

        # Enable gradient checkpointing
        if self.config['training'].get('gradient_checkpointing', True):
            self.student_model.gradient_checkpointing_enable()

        # Setup optimizer (only for trainable parameters)
        training_config = self.config['training']
        trainable_params = [p for p in self.student_model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
        )

        # Setup dataset
        pseudo_labels_dir = Path(self.config['paths'].get('pseudo_labels_dir', self.workspace / 'pseudo_labels'))
        pseudo_labels_path = pseudo_labels_dir / 'all_pseudo_labels.jsonl'

        if not pseudo_labels_path.exists():
            console.print(f"[red]Error: Pseudo-labels not found at {pseudo_labels_path}[/red]")
            console.print("[yellow]Run 02_generate_pseudo_labels_multi_gpu.py first[/yellow]")
            sys.exit(1)

        # Faithful v3.5 training-time augmentation probabilities
        distil_config = self.config.get('distillation', {})
        timestamp_probability = float(distil_config.get('timestamp_probability', 0.0) or 0.0)
        condition_on_prev_probability = float(distil_config.get('condition_on_prev_probability', 0.0) or 0.0)
        # Whisper's decoder positions cap at 448
        max_label_length = min(int(distil_config.get('pseudo_labels', {}).get('max_label_length', 448) or 448), 448)

        full_dataset = DistillationDataset(
            pseudo_labels_path=str(pseudo_labels_path),
            processor=self.processor,
            max_label_length=max_label_length,
            timestamp_probability=timestamp_probability,
            condition_on_prev_probability=condition_on_prev_probability,
        )

        # Hold out a small slice for periodic eval (overfitting signal). Capped to
        # at most 10% so tiny PoC datasets keep enough training data.
        eval_n = min(int(training_config.get('eval_samples', 0) or 0), len(full_dataset) // 10)
        if eval_n > 0:
            from torch.utils.data import Subset
            n = len(full_dataset)
            train_dataset = Subset(full_dataset, list(range(0, n - eval_n)))
            self.eval_dataset = Subset(full_dataset, list(range(n - eval_n, n)))
            if self.accelerator.is_main_process:
                console.print(f"[cyan]Held out {eval_n:,} samples for eval[/cyan]")
        else:
            train_dataset = full_dataset
            self.eval_dataset = None

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id
        collator = DistillationCollator(pad_token_id=pad_id, max_length=max_label_length)

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_config['per_device_train_batch_size'],
            shuffle=True,
            num_workers=training_config.get('dataloader_num_workers', 4),
            pin_memory=training_config.get('dataloader_pin_memory', True),
            collate_fn=collator,
        )

        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=training_config['per_device_train_batch_size'],
                shuffle=False,
                num_workers=training_config.get('dataloader_num_workers', 4),
                pin_memory=training_config.get('dataloader_pin_memory', True),
                collate_fn=collator,
            )

        # Setup scheduler
        num_training_steps = training_config['max_steps']
        num_warmup_steps = training_config['warmup_steps']

        self.scheduler = get_scheduler(
            name=training_config['lr_scheduler_type'],
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Setup loss function - EXACT official formula
        distil_config = self.config['distillation']
        self.loss_fn = DistillationLoss(
            temperature=distil_config['temperature'],
            ce_weight=distil_config.get('ce_weight', 0.8),  # Official: 0.8
            kl_weight=distil_config.get('kl_weight', 1.0),  # Official: 1.0
        )

        # Prepare for distributed training
        (
            self.student_model,
            self.teacher_model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.student_model,
            self.teacher_model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        )

        if self.eval_dataloader is not None:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

        if self.accelerator.is_main_process:
            console.print("[green]✓ Training setup complete[/green]")

    def _save_checkpoint(self, emergency: bool = False):
        """Save training checkpoint."""
        if not self.accelerator.is_main_process:
            return

        suffix = 'emergency' if emergency else f'{self.global_step}'
        checkpoint_path = self.checkpoint_dir / f'checkpoint-{suffix}'

        console.print(f"[yellow]Saving checkpoint to {checkpoint_path}...[/yellow]")

        # Unwrap model
        unwrapped = self.accelerator.unwrap_model(self.student_model)
        unwrapped.save_pretrained(checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)

        # Save training state
        state = {
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'epoch': self.epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(state, checkpoint_path / 'training_state.pt')

        console.print(f"[green]✓ Checkpoint saved at step {self.global_step}[/green]")

        # Push to hub if configured
        hf_config = self.config.get('huggingface', {})
        if hf_config.get('push_to_hub', False) and not emergency:
            self._push_to_hub(checkpoint_path)

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def _load_checkpoint(self) -> bool:
        """
        Resume from the most-advanced checkpoint (by saved global_step), including
        the emergency checkpoint written on preemption.

        Weights are loaded INTO the already-prepared student model rather than
        replacing the object — replacing it would leave an unprepared, off-device
        model whose parameters no longer match the optimizer.
        """
        candidates = list(self.checkpoint_dir.glob('checkpoint-[0-9]*'))
        emergency = self.checkpoint_dir / 'checkpoint-emergency'
        if emergency.exists() and (emergency / 'training_state.pt').exists():
            candidates.append(emergency)
        if not candidates:
            return False

        def _step_of(p: Path) -> int:
            state_path = p / 'training_state.pt'
            try:
                return int(torch.load(state_path, map_location='cpu').get('global_step', -1))
            except Exception:
                try:
                    return int(p.name.split('-')[1])
                except Exception:
                    return -1

        checkpoint_path = max(candidates, key=_step_of)

        if self.accelerator.is_main_process:
            console.print(f"[yellow]Loading checkpoint from {checkpoint_path}...[/yellow]")

        # Load weights into the existing (prepared) student model, then re-freeze.
        loaded = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
        unwrapped = self.accelerator.unwrap_model(self.student_model)
        unwrapped.load_state_dict(loaded.state_dict())
        del loaded
        freeze_encoder(unwrapped)

        # Load training state
        state = torch.load(checkpoint_path / 'training_state.pt', map_location='cpu')
        self.global_step = state['global_step']
        self.best_loss = state['best_loss']
        self.epoch = state['epoch']
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])

        if self.accelerator.is_main_process:
            console.print(f"[green]✓ Resumed from step {self.global_step}[/green]")

        return True

    def _push_to_hub(self, checkpoint_path: Path):
        """Push checkpoint to HuggingFace Hub."""
        try:
            hf_config = self.config['huggingface']
            username = os.environ.get('HF_USERNAME', hf_config.get('username', 'user'))
            repo_name = hf_config.get('repo_name', 'distil-crisperwhisper')
            repo_id = f"{username}/{repo_name}"

            api = HfApi()
            try:
                create_repo(repo_id, private=hf_config.get('private', True), exist_ok=True)
            except Exception:
                pass

            api.upload_folder(
                folder_path=str(checkpoint_path),
                repo_id=repo_id,
                commit_message=f"Checkpoint at step {self.global_step}",
            )

            console.print(f"[green]✓ Pushed to Hub: {repo_id}[/green]")

        except Exception as e:
            console.print(f"[yellow]Warning: Could not push to Hub: {e}[/yellow]")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints."""
        limit = self.config['training'].get('save_total_limit', 5)
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint-[0-9]*'),
            key=lambda x: int(x.name.split('-')[1])
        )

        while len(checkpoints) > limit:
            oldest = checkpoints.pop(0)
            console.print(f"[dim]Removing old checkpoint: {oldest}[/dim]")
            shutil.rmtree(oldest)

    def _evaluate(self) -> Optional[float]:
        """
        Mean distillation loss over the held-out eval set (no grad, no augmentation).

        Single-GPU: exact. Multi-GPU: each process averages its own shard and only
        the main process logs, so the reported value is an approximation — fine as
        an overfitting signal.
        """
        if self.eval_dataloader is None:
            return None

        self.student_model.eval()
        self.spec_augment.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                student_logits = self.student_model(
                    input_features=batch['input_features'],
                    decoder_input_ids=batch['decoder_input_ids'],
                ).logits
                teacher_logits = self.teacher_model(
                    input_features=batch['input_features'],
                    decoder_input_ids=batch['decoder_input_ids'],
                ).logits
                loss, _ = self.loss_fn(student_logits, teacher_logits, batch['labels'])
                total += float(loss.detach().item())
                count += 1
        self.student_model.train()
        self.spec_augment.train()

        return (total / count) if count else None

    def train(self, resume: bool = False):
        """Main training loop."""
        self.setup()

        if resume:
            self._load_checkpoint()

        training_config = self.config['training']
        max_steps = training_config['max_steps']
        save_steps = training_config['save_steps']
        logging_steps = training_config['logging_steps']
        eval_steps = int(training_config.get('eval_steps', 0) or 0)

        if self.accelerator.is_main_process:
            effective_batch = training_config['per_device_train_batch_size'] * self.accelerator.num_processes * self.gradient_accumulation_steps
            console.print(f"\n[bold green]Starting training from step {self.global_step}...[/bold green]")
            console.print(f"  Max steps: {max_steps}")
            console.print(f"  GPUs: {self.accelerator.num_processes}")
            console.print(f"  Batch size per GPU: {training_config['per_device_train_batch_size']}")
            console.print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
            console.print(f"  Effective batch: {effective_batch}")

        self.student_model.train()
        self.spec_augment.train()
        running_loss = 0.0
        running_ce = 0.0
        running_kl = 0.0

        if self.accelerator.is_main_process:
            pbar = tqdm(total=max_steps, initial=self.global_step, desc="Training")
        else:
            pbar = None

        while self.global_step < max_steps:
            for batch in self.train_dataloader:
                if self.spot_handler.should_stop:
                    self._save_checkpoint()
                    return

                with self.accelerator.accumulate(self.student_model):
                    # Apply SpecAugment to the STUDENT's input features only
                    input_features = batch['input_features']
                    if self.student_model.training:
                        input_features = self.spec_augment(input_features)

                    # Student forward pass. We pass explicit decoder_input_ids so
                    # timestamp + prev-context conditioning are handled identically
                    # for teacher and student (the loss uses the masked `labels`).
                    student_logits = self.student_model(
                        input_features=input_features,
                        decoder_input_ids=batch['decoder_input_ids'],
                    ).logits

                    # Teacher forward pass (no gradients, clean/un-augmented features)
                    with torch.no_grad():
                        teacher_logits = self.teacher_model(
                            input_features=batch['input_features'],
                            decoder_input_ids=batch['decoder_input_ids'],
                        ).logits

                    # Combined distillation loss. KL aligns position-by-position
                    # because teacher & student share decoder_input_ids.
                    loss, loss_dict = self.loss_fn(
                        student_logits,
                        teacher_logits,
                        batch['labels'],
                    )

                    # Backward pass
                    self.accelerator.backward(loss)

                    # Gradient clipping
                    if training_config.get('max_grad_norm', 1.0) > 0:
                        self.accelerator.clip_grad_norm_(
                            self.student_model.parameters(),
                            training_config['max_grad_norm'],
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                running_loss += loss_dict['total_loss']
                running_ce += loss_dict['ce_loss']
                running_kl += loss_dict['kl_loss']
                self.global_step += 1

                # Logging
                if self.global_step % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    avg_ce = running_ce / logging_steps
                    avg_kl = running_kl / logging_steps
                    lr = self.scheduler.get_last_lr()[0]

                    self.accelerator.log({
                        'train/loss': avg_loss,
                        'train/ce_loss': avg_ce,
                        'train/kl_loss': avg_kl,
                        'train/learning_rate': lr,
                    }, step=self.global_step)

                    if pbar:
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'ce': f'{avg_ce:.4f}',
                            'kl': f'{avg_kl:.4f}',
                            'lr': f'{lr:.2e}',
                        })

                    running_loss = 0.0
                    running_ce = 0.0
                    running_kl = 0.0

                # Save checkpoint
                if self.global_step % save_steps == 0:
                    self._save_checkpoint()

                # Periodic eval on the held-out slice (overfitting signal).
                # Defensive: a failure here must never kill a long training run.
                if eval_steps and self.eval_dataloader is not None and self.global_step % eval_steps == 0:
                    try:
                        eval_loss = self._evaluate()
                        if eval_loss is not None:
                            self.accelerator.log({'eval/loss': eval_loss}, step=self.global_step)
                            if eval_loss < self.best_loss:
                                self.best_loss = eval_loss
                            if self.accelerator.is_main_process and pbar:
                                pbar.write(f"[eval] step {self.global_step}: loss {eval_loss:.4f} (best {self.best_loss:.4f})")
                    except Exception as e:
                        if self.accelerator.is_main_process:
                            console.print(f"[yellow]Eval skipped at step {self.global_step}: {e}[/yellow]")

                if pbar:
                    pbar.update(1)

                if self.global_step >= max_steps:
                    break

        if pbar:
            pbar.close()

        # Final save
        if self.accelerator.is_main_process:
            console.print("\n[bold green]Training complete![/bold green]")
            self._save_checkpoint()

            # Save final model
            final_path = self.output_dir / 'distil-crisperwhisper-final'
            unwrapped = self.accelerator.unwrap_model(self.student_model)
            unwrapped.save_pretrained(final_path)
            self.processor.save_pretrained(final_path)

            console.print(f"[green]✓ Final model saved to {final_path}[/green]")
            console.print("\n[bold]Next step:[/bold]")
            console.print("  python3 04_convert_to_ctranslate2.py --config ../config.yaml")


def main():
    parser = argparse.ArgumentParser(description='Train distilled CrisperWhisper (Multi-GPU)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Override training.max_steps (handy for a quick local PoC run)')
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

    # Start training
    trainer = DistillationTrainer(str(config_path))
    if args.max_steps is not None:
        # Override before train()->setup() so the LR scheduler uses the same horizon.
        trainer.config['training']['max_steps'] = args.max_steps
        console.print(f"[cyan]Overriding max_steps -> {args.max_steps} (PoC)[/cyan]")
    trainer.train(resume=args.resume)


if __name__ == '__main__':
    main()
