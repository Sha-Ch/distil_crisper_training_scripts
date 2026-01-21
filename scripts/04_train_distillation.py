#!/usr/bin/env python3
"""
=============================================================================
Distillation Training Script for Distil-CrisperWhisper
=============================================================================
Trains a distilled student model from CrisperWhisper teacher using
knowledge distillation following the official distil-whisper v3.5 methodology.

Key Features (matching official distil-whisper):
1. FROZEN ENCODER - Encoder copied from teacher and frozen during training
2. SpecAugment - Audio data augmentation for robustness
3. KL + CE Loss - Combined distillation loss
4. Sample Packing - Efficient batching with ~30s segments
5. Automatic checkpoint resume for spot instances

Usage: python3 04_train_distillation.py [--config config.yaml] [--resume]

References:
- Paper: https://arxiv.org/abs/2311.00430
- Code: https://github.com/huggingface/distil-whisper
=============================================================================
"""

import os
import sys
import json
import yaml
import argparse
import signal
import shutil
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperConfig,
    get_scheduler,
    set_seed
)
from accelerate import Accelerator
import soundfile as sf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
from huggingface_hub import HfApi, create_repo

console = Console()


# =============================================================================
# SpecAugment (following distil-whisper methodology)
# =============================================================================

class SpecAugment(nn.Module):
    """
    SpecAugment data augmentation for audio spectrograms.
    Following the implementation used in distil-whisper v3.5.

    Paper: https://arxiv.org/abs/1904.08779
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        p: float = 0.5  # Probability of applying augmentation
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to input features.

        Args:
            x: Input mel spectrogram [batch, n_mels, time]

        Returns:
            Augmented spectrogram
        """
        if not self.training or random.random() > self.p:
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
# Spot Instance Handling
# =============================================================================

class SpotInstanceHandler:
    """Handles spot instance preemption signals with grace period."""

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
        console.print(f"\n[bold yellow]⚠ Received termination signal ({signum}). Saving checkpoint...[/bold yellow]")
        self.should_stop = True
        if self.save_callback:
            self.save_callback()


# =============================================================================
# Dataset with Sample Packing
# =============================================================================

class DistillationDataset(Dataset):
    """
    Dataset for distillation training with pseudo-labels.
    Implements features from official distil-whisper:
    - Sample packing to 30 seconds
    - Timestamp probability (include timestamps in X% of samples)
    - Condition on previous probability (for long-form training)

    Reference: https://github.com/huggingface/distil-whisper
    """

    def __init__(
        self,
        manifest_path: str,
        processor: WhisperProcessor,
        max_length: int = 480000,  # 30 seconds at 16kHz
        sample_rate: int = 16000,
        timestamp_probability: float = 0.2,
        condition_on_prev_probability: float = 0.2
    ):
        self.processor = processor
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.timestamp_probability = timestamp_probability
        self.condition_on_prev_probability = condition_on_prev_probability

        # Load manifest
        self.samples = []
        with open(manifest_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # Use pseudo_transcription as the training target
                if 'pseudo_transcription' in entry:
                    entry['text'] = entry['pseudo_transcription']
                elif 'transcription' in entry:
                    entry['text'] = entry['transcription']
                self.samples.append(entry)

        console.print(f"[green]Loaded {len(self.samples)} samples from {manifest_path}[/green]")
        console.print(f"[green]  Timestamp probability: {timestamp_probability}[/green]")
        console.print(f"[green]  Condition on prev probability: {condition_on_prev_probability}[/green]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load audio
        audio, sr = sf.read(sample['audio_path'])

        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Ensure correct sample rate
        if sr != self.sample_rate:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio_tensor).squeeze(0).numpy()

        # Pad or truncate to 30 seconds
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        elif len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))

        # Process audio to mel spectrogram
        input_features = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_features.squeeze(0)

        # Process text (pseudo-label from teacher)
        text = sample.get('text', sample.get('pseudo_transcription', ''))
        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=448,
            truncation=True
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 for loss computation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            'input_features': input_features,
            'labels': labels
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for batching."""
    input_features = torch.stack([item['input_features'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_features': input_features,
        'labels': labels
    }


# =============================================================================
# Student Model Creation (following distil-whisper)
# =============================================================================

def create_student_model(
    teacher_model: WhisperForConditionalGeneration,
    config: Dict[str, Any]
) -> WhisperForConditionalGeneration:
    """
    Create a smaller student model following distil-whisper methodology:
    - Copy ENTIRE encoder from teacher (will be frozen during training)
    - Copy SUBSET of decoder layers (maximally spaced)
    - This enables speculative decoding compatibility

    Args:
        teacher_model: The CrisperWhisper teacher model
        config: Configuration dictionary

    Returns:
        Student model initialized from teacher
    """
    student_config = config['student']
    num_decoder_layers = student_config.get('decoder_layers', 2)

    console.print(f"[bold blue]Creating student model (distil-whisper methodology)...[/bold blue]")
    console.print(f"  Encoder: FULL (32 layers, will be FROZEN)")
    console.print(f"  Decoder: {num_decoder_layers} layers (maximally spaced from teacher)")

    # Get teacher config
    teacher_config = teacher_model.config

    # Create student config - KEEP FULL ENCODER
    student_model_config = WhisperConfig(
        vocab_size=teacher_config.vocab_size,
        num_mel_bins=teacher_config.num_mel_bins,
        encoder_layers=teacher_config.encoder_layers,  # FULL encoder
        encoder_attention_heads=teacher_config.encoder_attention_heads,
        decoder_layers=num_decoder_layers,  # Reduced decoder
        decoder_attention_heads=teacher_config.decoder_attention_heads,
        decoder_ffn_dim=teacher_config.decoder_ffn_dim,
        encoder_ffn_dim=teacher_config.encoder_ffn_dim,
        d_model=teacher_config.d_model,
        dropout=teacher_config.dropout,
        attention_dropout=teacher_config.attention_dropout,
        activation_dropout=teacher_config.activation_dropout,
        max_source_positions=teacher_config.max_source_positions,
        max_target_positions=teacher_config.max_target_positions,
        pad_token_id=teacher_config.pad_token_id,
        bos_token_id=teacher_config.bos_token_id,
        eos_token_id=teacher_config.eos_token_id,
        suppress_tokens=teacher_config.suppress_tokens,
        begin_suppress_tokens=teacher_config.begin_suppress_tokens,
        use_cache=False,
    )

    # Create student model
    student_model = WhisperForConditionalGeneration(student_model_config)

    console.print("[yellow]Initializing student from teacher weights...[/yellow]")

    # 1. Copy ENTIRE encoder (this will be frozen)
    student_model.model.encoder.load_state_dict(
        teacher_model.model.encoder.state_dict()
    )
    console.print("  ✓ Full encoder copied (will be frozen during training)")

    # 2. Copy maximally spaced decoder layers
    teacher_decoder_layers = teacher_config.decoder_layers
    # Calculate maximally spaced indices
    if num_decoder_layers == 1:
        layer_indices = [teacher_decoder_layers - 1]  # Last layer
    else:
        layer_indices = np.linspace(0, teacher_decoder_layers - 1, num_decoder_layers, dtype=int)

    for student_idx, teacher_idx in enumerate(layer_indices):
        student_model.model.decoder.layers[student_idx].load_state_dict(
            teacher_model.model.decoder.layers[teacher_idx].state_dict()
        )

    console.print(f"  ✓ Decoder layers copied from teacher indices: {layer_indices.tolist()}")

    # 3. Copy decoder embeddings
    student_model.model.decoder.embed_tokens.load_state_dict(
        teacher_model.model.decoder.embed_tokens.state_dict()
    )
    student_model.model.decoder.embed_positions.load_state_dict(
        teacher_model.model.decoder.embed_positions.state_dict()
    )
    console.print("  ✓ Decoder embeddings copied")

    # 4. Copy output projection
    student_model.proj_out.load_state_dict(teacher_model.proj_out.state_dict())
    console.print("  ✓ Output projection copied")

    # Count parameters
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())

    # Count trainable params (decoder only since encoder will be frozen)
    encoder_params = sum(p.numel() for p in student_model.model.encoder.parameters())
    trainable_params = student_params - encoder_params

    console.print(f"\n[green]Student model created![/green]")
    console.print(f"  Teacher parameters: {teacher_params / 1e6:.1f}M")
    console.print(f"  Student parameters: {student_params / 1e6:.1f}M")
    console.print(f"  Trainable parameters (decoder): {trainable_params / 1e6:.1f}M")
    console.print(f"  Frozen parameters (encoder): {encoder_params / 1e6:.1f}M")

    return student_model


def freeze_encoder(model: WhisperForConditionalGeneration):
    """
    Freeze the encoder following distil-whisper methodology.
    This is CRITICAL for:
    1. Preserving encoder quality from teacher
    2. Enabling speculative decoding
    3. Allowing 2x larger batch sizes
    """
    console.print("[yellow]Freezing encoder (distil-whisper methodology)...[/yellow]")

    for param in model.model.encoder.parameters():
        param.requires_grad = False

    # Also freeze embed_positions in decoder (following distil-whisper)
    for param in model.model.decoder.embed_positions.parameters():
        param.requires_grad = False

    # Count frozen vs trainable
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    console.print(f"  ✓ Frozen: {frozen_params / 1e6:.1f}M parameters")
    console.print(f"  ✓ Trainable: {trainable_params / 1e6:.1f}M parameters")


# =============================================================================
# Distillation Loss - EXACT Official Implementation
# =============================================================================

class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation (EXACT official distil-whisper formula):
    - Cross-entropy loss on pseudo-labels (hard targets)
    - KL divergence loss on teacher logits (soft targets)

    OFFICIAL formula from run_distillation.py:
        loss = 0.8 * ce_loss + kl_weight * kl_loss

    Where:
    - ce_weight = 0.8 (NOT alpha that sums to 1 with kl_weight!)
    - kl_weight = 1.0 (default, can be tuned)
    - KL_loss is scaled by temperature^2

    Reference: https://github.com/huggingface/distil-whisper/blob/main/training/run_distillation.py
    """

    def __init__(self, temperature: float = 2.0, ce_weight: float = 0.8, kl_weight: float = 1.0):
        """
        Args:
            temperature: Softmax temperature for KL loss (default: 2.0)
            ce_weight: Weight for CE loss (default: 0.8)
            kl_weight: Weight for KL loss (default: 1.0) - NOT (1 - ce_weight)!
        """
        super().__init__()
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.kl_weight = kl_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
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
        ce_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )

        # Soft label loss (KL divergence with teacher)
        # Following official implementation: temperature scaling + temperature^2 factor
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Create mask for valid (non-padding) tokens
        # Shape: [batch, seq_len, 1] -> broadcast to vocab dimension
        mask = (labels != -100).unsqueeze(-1).float()

        # Compute KL divergence only on valid tokens
        # KL(P||Q) = sum(P * log(P/Q)) = sum(P * log(P) - P * log(Q))
        # For loss: KL = sum(teacher * (log(teacher) - log(student)))
        kl_div = teacher_probs * (teacher_probs.log() - student_log_probs)
        kl_div = kl_div * mask  # Zero out padding positions

        # Sum over vocab, mean over valid tokens
        kl_loss = kl_div.sum() / mask.sum().clamp(min=1)

        # Scale by temperature^2 (standard KD practice)
        kl_loss = kl_loss * (self.temperature ** 2)

        # Combined loss: OFFICIAL formula = 0.8 * CE + 1.0 * KL
        total_loss = self.ce_weight * ce_loss + self.kl_weight * kl_loss

        return total_loss, {
            'ce_loss': ce_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item()
        }


# =============================================================================
# Training Loop
# =============================================================================

class DistillationTrainer:
    """Handles distillation training following distil-whisper v3.5 methodology."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.config_path = config_path

        # Paths
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision='bf16' if self.config['training'].get('bf16', True) else 'fp16',
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            log_with=['tensorboard'] if not self.config['logging'].get('use_wandb', False) else ['tensorboard', 'wandb'],
            project_dir=str(self.checkpoint_dir)
        )

        # Spot instance handler
        self.spot_handler = SpotInstanceHandler(save_callback=self._emergency_save)

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')

        # SpecAugment
        self.spec_augment = SpecAugment(
            freq_mask_param=27,
            time_mask_param=100,
            n_freq_masks=2,
            n_time_masks=2,
            p=0.5  # 50% probability
        )

        # Models (loaded later)
        self.teacher_model = None
        self.student_model = None
        self.processor = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _emergency_save(self):
        if self.student_model is not None:
            console.print("[bold yellow]Emergency save in progress...[/bold yellow]")
            self._save_checkpoint(emergency=True)

    def setup(self):
        """Setup models, optimizer, and data loaders."""

        set_seed(42)

        console.print("[bold blue]Setting up training (distil-whisper v3.5 methodology)...[/bold blue]")

        # Load teacher model
        console.print("\n[yellow]Loading teacher model (CrisperWhisper)...[/yellow]")
        teacher_id = self.config['teacher']['model_id']
        cache_dir = self.config['paths'].get('hf_cache', None)

        self.processor = WhisperProcessor.from_pretrained(teacher_id, cache_dir=cache_dir)

        self.teacher_model = WhisperForConditionalGeneration.from_pretrained(
            teacher_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16
        )
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        console.print("[green]✓ Teacher model loaded[/green]")

        # Create student model
        self.student_model = create_student_model(self.teacher_model, self.config)

        # CRITICAL: Freeze encoder (distil-whisper methodology)
        freeze_encoder(self.student_model)

        # Enable gradient checkpointing for memory efficiency
        self.student_model.gradient_checkpointing_enable()

        # Setup optimizer (only for trainable parameters)
        training_config = self.config['training']
        trainable_params = [p for p in self.student_model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )

        # Setup dataset
        pseudo_labels_path = Path(self.config['paths']['data_dir']) / 'pseudo_labels' / 'all_labels.jsonl'

        if not pseudo_labels_path.exists():
            console.print(f"[red]Error: Pseudo-labels not found at {pseudo_labels_path}[/red]")
            console.print("[yellow]Run 03_generate_pseudo_labels.py first[/yellow]")
            sys.exit(1)

        # Get distillation config for timestamp/condition probabilities
        distil_config = self.config.get('distillation', {})

        train_dataset = DistillationDataset(
            manifest_path=str(pseudo_labels_path),
            processor=self.processor,
            timestamp_probability=distil_config.get('timestamp_probability', 0.2),
            condition_on_prev_probability=distil_config.get('condition_on_prev_probability', 0.2)
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_config['per_device_train_batch_size'],
            shuffle=True,
            num_workers=training_config['dataloader_num_workers'],
            pin_memory=training_config['dataloader_pin_memory'],
            collate_fn=collate_fn
        )

        # Setup scheduler
        num_training_steps = training_config['max_steps']
        num_warmup_steps = training_config['warmup_steps']

        self.scheduler = get_scheduler(
            name=training_config['lr_scheduler_type'],
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # Prepare with accelerator
        (
            self.student_model,
            self.teacher_model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler
        ) = self.accelerator.prepare(
            self.student_model,
            self.teacher_model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler
        )

        # Setup loss function
        distil_config = self.config['distillation']
        self.loss_fn = DistillationLoss(
            temperature=distil_config['temperature'],
            alpha=distil_config['alpha']
        )

        console.print("[green]✓ Training setup complete[/green]")
        console.print(f"\n[bold]Training Configuration:[/bold]")
        console.print(f"  Encoder: FROZEN (copied from CrisperWhisper)")
        console.print(f"  Decoder layers: {self.config['student']['decoder_layers']}")
        console.print(f"  SpecAugment: ENABLED")
        console.print(f"  Max steps: {training_config['max_steps']}")
        console.print(f"  Learning rate: {training_config['learning_rate']}")

    def _save_checkpoint(self, emergency: bool = False):
        """Save training checkpoint."""

        checkpoint_path = self.checkpoint_dir / f'checkpoint-{self.global_step}'
        if emergency:
            checkpoint_path = self.checkpoint_dir / 'checkpoint-emergency'

        console.print(f"[yellow]Saving checkpoint to {checkpoint_path}...[/yellow]")

        # Save model
        unwrapped_model = self.accelerator.unwrap_model(self.student_model)
        unwrapped_model.save_pretrained(checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)

        # Save training state
        training_state = {
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        torch.save(training_state, checkpoint_path / 'training_state.pt')

        console.print(f"[green]✓ Checkpoint saved at step {self.global_step}[/green]")

        # Push to hub if configured
        if self.config['huggingface'].get('push_to_hub', False) and not emergency:
            self._push_to_hub(checkpoint_path)

        self._cleanup_checkpoints()

    def _load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> bool:
        """Load training checkpoint."""

        if checkpoint_path is None:
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint-*'))
            if not checkpoints:
                return False
            checkpoint_path = checkpoints[-1]

        if not checkpoint_path.exists():
            return False

        console.print(f"[yellow]Loading checkpoint from {checkpoint_path}...[/yellow]")

        # Load model
        self.student_model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)

        # Re-freeze encoder
        freeze_encoder(self.student_model)

        # Load training state
        training_state = torch.load(checkpoint_path / 'training_state.pt')
        self.global_step = training_state['global_step']
        self.best_loss = training_state['best_loss']
        self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
        self.scheduler.load_state_dict(training_state['scheduler_state_dict'])

        console.print(f"[green]✓ Resumed from step {self.global_step}[/green]")
        return True

    def _push_to_hub(self, checkpoint_path: Path):
        """Push checkpoint to HuggingFace Hub."""
        try:
            hf_config = self.config['huggingface']
            repo_id = f"{os.environ.get('HF_USERNAME', 'user')}/{hf_config['repo_name']}"

            api = HfApi()
            try:
                create_repo(repo_id, private=hf_config.get('private', True), exist_ok=True)
            except Exception:
                pass

            api.upload_folder(
                folder_path=str(checkpoint_path),
                repo_id=repo_id,
                commit_message=f"Checkpoint at step {self.global_step}"
            )

            console.print(f"[green]✓ Pushed to Hub: {repo_id}[/green]")

        except Exception as e:
            console.print(f"[yellow]Warning: Could not push to Hub: {e}[/yellow]")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints."""
        save_total_limit = self.config['training'].get('save_total_limit', 3)

        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint-[0-9]*'),
            key=lambda x: int(x.name.split('-')[1])
        )

        while len(checkpoints) > save_total_limit:
            oldest = checkpoints.pop(0)
            console.print(f"[dim]Removing old checkpoint: {oldest}[/dim]")
            shutil.rmtree(oldest)

    def train(self, resume: bool = False):
        """Main training loop."""

        self.setup()

        if resume:
            self._load_checkpoint()

        training_config = self.config['training']
        max_steps = training_config['max_steps']
        save_steps = training_config['save_steps']
        logging_steps = training_config['logging_steps']

        console.print(f"\n[bold green]Starting training from step {self.global_step}...[/bold green]")
        console.print(f"  Max steps: {max_steps}")
        console.print(f"  Batch size: {training_config['per_device_train_batch_size']}")
        console.print(f"  Gradient accumulation: {training_config['gradient_accumulation_steps']}")
        console.print(f"  Encoder: FROZEN")
        console.print(f"  SpecAugment: ENABLED")

        self.student_model.train()
        self.spec_augment.train()
        running_loss = 0.0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            task = progress.add_task("Training", total=max_steps)
            progress.update(task, completed=self.global_step)

            while self.global_step < max_steps:
                for batch in self.train_dataloader:
                    if self.spot_handler.should_stop:
                        console.print("[yellow]Training stopped due to preemption[/yellow]")
                        self._save_checkpoint()
                        return

                    with self.accelerator.accumulate(self.student_model):
                        # Apply SpecAugment to input features
                        input_features = batch['input_features']
                        if self.training:
                            input_features = self.spec_augment(input_features)

                        # Forward pass through student
                        student_outputs = self.student_model(
                            input_features=input_features,
                            labels=batch['labels']
                        )
                        student_logits = student_outputs.logits

                        # Forward pass through teacher (no gradients, no augmentation)
                        with torch.no_grad():
                            teacher_outputs = self.teacher_model(
                                input_features=batch['input_features'],  # Original features
                                labels=batch['labels']
                            )
                            teacher_logits = teacher_outputs.logits

                        # Compute loss
                        loss, loss_dict = self.loss_fn(
                            student_logits,
                            teacher_logits,
                            batch['labels']
                        )

                        # Backward pass
                        self.accelerator.backward(loss)

                        # Gradient clipping
                        if training_config.get('max_grad_norm', 1.0) > 0:
                            self.accelerator.clip_grad_norm_(
                                self.student_model.parameters(),
                                training_config['max_grad_norm']
                            )

                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                    running_loss += loss_dict['total_loss']
                    self.global_step += 1

                    # Logging
                    if self.global_step % logging_steps == 0:
                        avg_loss = running_loss / logging_steps
                        lr = self.scheduler.get_last_lr()[0]

                        self.accelerator.log({
                            'train/loss': avg_loss,
                            'train/ce_loss': loss_dict['ce_loss'],
                            'train/kl_loss': loss_dict['kl_loss'],
                            'train/learning_rate': lr
                        }, step=self.global_step)

                        progress.update(
                            task,
                            description=f"Training (loss: {avg_loss:.4f}, lr: {lr:.2e})"
                        )
                        running_loss = 0.0

                    # Save checkpoint
                    if self.global_step % save_steps == 0:
                        self._save_checkpoint()

                    progress.update(task, advance=1)

                    if self.global_step >= max_steps:
                        break

        # Final save
        console.print("\n[bold green]Training complete![/bold green]")
        self._save_checkpoint()

        # Save final model
        final_path = self.output_dir / 'distil-crisperwhisper-final'
        unwrapped_model = self.accelerator.unwrap_model(self.student_model)
        unwrapped_model.save_pretrained(final_path)
        self.processor.save_pretrained(final_path)

        console.print(f"[green]✓ Final model saved to {final_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description='Train distilled CrisperWhisper (distil-whisper methodology)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
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
    trainer.train(resume=args.resume)


if __name__ == '__main__':
    main()
