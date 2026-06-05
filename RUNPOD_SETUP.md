# Distil-CrisperWhisper: 4x H100 NVL Setup Guide

Complete guide for training a distilled CrisperWhisper model on RunPod with **4x H100 NVL GPUs** following the **official distil-whisper v3.5 methodology**.

> **Single local RTX 4090 instead of a cloud pod?** See [LOCAL_4090.md](LOCAL_4090.md)
> (Docker on WSL2, `config.local.yaml`). The scripts auto-detect one GPU; full-scale
> training (4096 batch / 80 epochs) remains a cloud multi-GPU job.

## Overview

This setup will create a distilled CrisperWhisper model that:
- ✅ Preserves CrisperWhisper's word-level timestamp quality
- ✅ Runs **6x faster** than the original model
- ✅ Works with **faster-whisper** backend
- ✅ Uses only **~2GB VRAM** (vs 6GB for original)

### Cost & Time Estimates

| Phase | Time | Cost (Spot) |
|-------|------|-------------|
| Setup & Dependencies | 30 min | ~$5 |
| Pseudo-Label Generation | 24-48 hrs | ~$250-500 |
| Training (80k steps) | 40-60 hrs | ~$400-600 |
| Conversion & Testing | 1 hr | ~$10 |
| **Total** | **3-5 days** | **~$665-1115** |

With your $900 budget, this is achievable with spot instances.

---

## Prerequisites

Before starting, ensure you have:

### 1. HuggingFace Account
- Create account at https://huggingface.co/join
- Generate an access token with **Write** permissions
- Accept the terms for these datasets:
  - [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech)
  - [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
  - [People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech)

### 2. RunPod Account
- Sign up at https://runpod.io
- Add funds (~$900 for full training)

### 3. (Optional) Weights & Biases Account
- Sign up at https://wandb.ai for training monitoring

---

## Step 1: Create RunPod Instance

### Pod Configuration

1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Click **+ Deploy**
3. Select **4x H100 NVL** (or "4x H100 80GB")
4. Configure:
   ```
   Template:        RunPod PyTorch 2.1+
   Container Disk:  50 GB
   Volume Disk:     800 GB (or more for full dataset)
   Volume Path:     /workspace
   Spot Instance:   ✓ (recommended for cost savings)
   ```

5. Click **Deploy**

### Why 4x H100 NVL?

- 94GB VRAM per GPU = 376GB total
- Enables batch size of 256 (matching official)
- NVLink for fast GPU communication
- ~$10.44/hr spot pricing

---

## Step 2: Initial Setup

Once your pod is running, connect via Web Terminal or SSH.

### 2.1 Set Environment Variables

```bash
# REQUIRED: Your HuggingFace token
export HF_TOKEN="hf_your_token_here"

# REQUIRED: Your HuggingFace username
export HF_USERNAME="your_username"

# OPTIONAL: Weights & Biases
export WANDB_API_KEY="your_wandb_key"

# Persist across restarts
cat >> ~/.bashrc << 'EOF'
export HF_TOKEN="hf_your_token_here"
export HF_USERNAME="your_username"
export WANDB_API_KEY="your_wandb_key"
export HF_HOME="/workspace/hf_cache"
export HF_DATASETS_CACHE="/workspace/hf_cache/datasets"
EOF

source ~/.bashrc
```

### 2.2 Upload Training Scripts

**Option A: Clone from GitHub**
```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git distil_crisper_training
```

**Option B: Upload ZIP via RunPod File Manager**
1. Zip the `distil_crisper_training` folder locally
2. Use RunPod's file upload feature
3. Unzip:
   ```bash
   cd /workspace
   unzip distil_crisper_training.zip
   ```

### 2.3 Run Setup Script

```bash
cd /workspace/distil_crisper_training/scripts
chmod +x *.sh *.py
bash 01_multi_gpu_setup.sh
```

This will:
- Check GPU configuration (4x H100 NVL)
- Check storage (need 800GB+)
- Install system dependencies (FFmpeg, etc.)
- Install Python packages (PyTorch, Transformers, etc.)
- Configure Accelerate for multi-GPU
- Login to HuggingFace

---

## Step 3: Generate Pseudo-Labels

This is the longest phase. We generate labels using CrisperWhisper to preserve its quality.

### 3.1 Start in tmux (Recommended)

```bash
tmux new -s pseudo_labels
```

### 3.2 Run Pseudo-Label Generation

```bash
cd /workspace/distil_crisper_training/scripts

# Multi-GPU pseudo-label generation
accelerate launch 02_generate_pseudo_labels_multi_gpu.py --config ../config.yaml
```

This will:
- Load CrisperWhisper as teacher model
- Stream all 8 datasets (~196,000 hours)
- Generate pseudo-labels with word timestamps
- Filter by WER (discard >10% WER)
- Save accepted samples (~98,000 hours after filtering)

### 3.3 Monitor Progress

```bash
# In another terminal or detach with Ctrl+B, D
tail -f /workspace/pseudo_labels/generation_progress.json

# Check accepted samples
wc -l /workspace/pseudo_labels/*_accepted.jsonl
```

### 3.4 Resume After Interruption

If spot instance gets interrupted:
```bash
# Re-run setup
source ~/.bashrc
cd /workspace/distil_crisper_training/scripts
bash 01_multi_gpu_setup.sh

# Resume (automatically continues from last checkpoint)
accelerate launch 02_generate_pseudo_labels_multi_gpu.py --config ../config.yaml
```

### 3.5 Estimated Time

| Dataset | Hours | Samples | Time (4x H100) |
|---------|-------|---------|----------------|
| LibriSpeech | 960 | ~280k | 2-3 hrs |
| GigaSpeech | 10,000 | ~3M | 8-12 hrs |
| VoxPopuli | 1,800 | ~500k | 3-4 hrs |
| Common Voice | 3,000 | ~1M | 5-6 hrs |
| TED-LIUM | 450 | ~100k | 1-2 hrs |
| AMI | 100 | ~30k | 30 min |
| People's Speech | 30,000 | ~9M | 15-20 hrs |
| YODAS | 150,000 | ~50M | 24-48 hrs |

**Total: ~48-96 hours** (can be reduced by skipping YODAS for initial training)

---

## Step 4: Train the Model

### 4.1 Verify Pseudo-Labels

```bash
# Check merged labels
ls -lh /workspace/pseudo_labels/all_pseudo_labels.jsonl

# Count samples
wc -l /workspace/pseudo_labels/all_pseudo_labels.jsonl
```

### 4.2 Start Training

```bash
tmux new -s training

cd /workspace/distil_crisper_training/scripts

# Launch multi-GPU training
accelerate launch 03_train_distillation_multi_gpu.py --config ../config.yaml
```

### 4.3 Training Configuration

The training follows official distil-whisper v3.5:

| Parameter | Value |
|-----------|-------|
| Max Steps | 80,000 |
| Batch Size (per GPU) | 16 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 256 |
| Learning Rate | 1e-4 |
| Warmup Steps | 500 |
| Loss | 0.8×CE + 0.2×KL |
| Temperature | 2.0 |
| Encoder | Frozen (32 layers) |
| Decoder | 2 layers |
| SpecAugment | Enabled |

### 4.4 Monitor Training

```bash
# TensorBoard
tensorboard --logdir /workspace/tensorboard --port 6006

# Check latest checkpoint
ls -la /workspace/checkpoints/

# View training logs
tail -f /workspace/logs/training.log
```

### 4.5 Resume Training

```bash
accelerate launch 03_train_distillation_multi_gpu.py --config ../config.yaml --resume
```

### 4.6 Training Time

With 4x H100 NVL:
- ~80,000 steps
- ~2 steps/second
- **~40,000 seconds = ~11 hours** for training loop
- With I/O overhead: **~40-60 hours total**

---

## Step 5: Convert to CTranslate2

After training completes:

```bash
cd /workspace/distil_crisper_training/scripts

python3 04_convert_to_ctranslate2.py --config ../config.yaml
```

This creates a model compatible with **faster-whisper**.

### Output Location

```
/workspace/output/distil-crisperwhisper-ct2/
├── model.bin
├── config.json
├── tokenizer.json
├── vocabulary.txt
└── README.md
```

---

## Step 6: Test Your Model

```python
from faster_whisper import WhisperModel

# Load your distilled model
model = WhisperModel(
    "/workspace/output/distil-crisperwhisper-ct2",
    device="cuda",
    compute_type="float16",
)

# Test transcription with word timestamps
segments, info = model.transcribe(
    "test_audio.wav",
    word_timestamps=True,
    language="en",
)

print(f"Detected language: {info.language}")
print(f"Duration: {info.duration:.2f}s")

for segment in segments:
    print(f"\n[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    for word in segment.words:
        print(f"  {word.word} ({word.start:.3f}s - {word.end:.3f}s)")
```

---

## Step 7: Download Your Model

### Option A: HuggingFace Hub

Your model is automatically pushed to:
```
https://huggingface.co/{HF_USERNAME}/distil-crisperwhisper-large-v3
```

Download locally:
```bash
huggingface-cli download {HF_USERNAME}/distil-crisperwhisper-large-v3
```

### Option B: Direct Download

Use RunPod's file browser to download:
```
/workspace/output/distil-crisperwhisper-ct2/
```

---

## Step 8: Stop Your Pod!

**IMPORTANT**: Don't forget to stop your pod when done!

1. Go to RunPod Console
2. Find your pod
3. Click **Stop** (keeps volume data) or **Terminate** (deletes everything)

---

## Troubleshooting

### Out of Memory

```yaml
# In config.yaml, reduce batch size:
training:
  per_device_train_batch_size: 8  # Reduce from 16
  gradient_accumulation_steps: 8  # Increase to maintain effective batch
```

### Dataset Access Denied

```
Error: 401 Unauthorized for GigaSpeech
```

Solution:
1. Go to https://huggingface.co/datasets/speechcolab/gigaspeech
2. Click "Agree and access repository"
3. Wait 5 minutes for access to propagate

### Spot Instance Interrupted

All scripts have automatic resume support:
```bash
# Resume pseudo-label generation
accelerate launch 02_generate_pseudo_labels_multi_gpu.py --config ../config.yaml

# Resume training
accelerate launch 03_train_distillation_multi_gpu.py --config ../config.yaml --resume
```

### Low GPU Utilization

Check if data loading is the bottleneck:
```bash
nvidia-smi -l 1
```

If GPU utilization is low:
```yaml
# In config.yaml:
training:
  dataloader_num_workers: 16  # Increase from 8
  dataloader_prefetch_factor: 8  # Add prefetching
```

### Training Too Slow

Try DeepSpeed for faster training:
```bash
accelerate launch --use_deepspeed 03_train_distillation_multi_gpu.py --config ../config.yaml
```

---

## Quick Reference Commands

```bash
# Check GPU status
nvidia-smi

# Check storage
df -h /workspace

# Check running jobs
htop

# Detach from tmux
Ctrl+B, D

# Reattach to tmux
tmux attach -t training

# Kill a tmux session
tmux kill-session -t training

# Check progress
wc -l /workspace/pseudo_labels/*_accepted.jsonl
ls -la /workspace/checkpoints/

# Emergency checkpoint save
# (Send SIGTERM to training process)
pkill -TERM -f train_distillation
```

---

## Support

If you encounter issues:
1. Check logs in `/workspace/logs/`
2. Review this troubleshooting guide
3. Check the config.yaml settings
4. Verify HuggingFace access for all datasets

---

## Expected Results

After successful training, your distil-CrisperWhisper model will:

| Metric | Original CrisperWhisper | Distil-CrisperWhisper |
|--------|------------------------|----------------------|
| Inference Speed | 1x (baseline) | **6x faster** |
| VRAM Usage | ~6 GB | **~2 GB** |
| WER (LibriSpeech) | 2.8% | ~3.0% |
| Word Timestamp Quality | Excellent | Excellent |
| Model Size | ~3 GB | ~1.5 GB |

The distilled model preserves CrisperWhisper's key advantages while being **6x faster** for production use.
