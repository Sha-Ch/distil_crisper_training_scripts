# Distil-CrisperWhisper Training Guide

Complete step-by-step guide for training a distilled CrisperWhisper model on cloud GPU.

> **Running data prep / a PoC on one local RTX 4090 (Docker on WSL2)?** See
> [LOCAL_4090.md](LOCAL_4090.md) and use `config.local.yaml`. Note: stage 2 now
> persists accepted audio (`save_audio: true`) so stage 3 can train on real
> features; the faithful v3.5 fidelity flags (timestamps, prev-context, BPE
> dropout, `weight_decay: 0.0`) are wired in.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Setup](#local-setup)
3. [Cloud Setup (RunPod)](#cloud-setup-runpod)
4. [Running the Training](#running-the-training)
5. [Monitoring & Resuming](#monitoring--resuming)
6. [Final Steps](#final-steps)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Before You Start

1. **HuggingFace Account** (Free)
   - Go to https://huggingface.co/join
   - Create an account
   - Go to Settings → Access Tokens → New Token
   - Create a token with **Write** permissions
   - Save this token securely

2. **GigaSpeech Access** (Free, requires agreement)
   - Go to https://huggingface.co/datasets/speechcolab/gigaspeech
   - Click "Agree and access repository"
   - This is required to download the training data

3. **Weights & Biases Account** (Optional, Free)
   - Go to https://wandb.ai/signup
   - Get your API key from https://wandb.ai/settings

4. **RunPod Account**
   - Go to https://runpod.io
   - Add funds ($50-100 should be plenty for initial training)

---

## Local Setup

### Step 1: Prepare Your Files

On your local machine, you need to upload the training scripts to the cloud. You have two options:

**Option A: Direct Upload (Recommended)**
1. Zip the `distil_crisper_training` folder
2. Upload to the cloud instance later

**Option B: Push to GitHub**
1. Create a private repository
2. Push the `distil_crisper_training` folder
3. Clone on the cloud instance

### Step 2: Review Configuration

Edit `config.yaml` to customize your training:

```yaml
# Key settings to review:

student:
  decoder_layers: 8  # Fewer = faster, more = better quality
  encoder_layers: 32 # Keep at 32 for best quality

training:
  max_steps: 50000   # Increase for better quality
  per_device_train_batch_size: 8  # Reduce if OOM

huggingface:
  username: "YOUR_USERNAME"  # Your HF username
  repo_name: "distil-crisperwhisper-large-v3"
```

---

## Cloud Setup (RunPod)

### Step 1: Create a Pod

1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Click **+ Deploy**
3. Select GPU:
   - **Recommended**: A100 80GB ($1.99-2.49/hr)
   - **Budget**: A100 40GB ($1.49/hr) - may need smaller batch size
4. Select template: **RunPod Pytorch 2.1**
5. Configure:
   - Container Disk: 50 GB
   - Volume Disk: **200 GB** (for persistent storage)
   - Volume Mount Path: `/workspace`
6. Check **Spot Instance** for lower cost (scripts handle interruptions)
7. Click **Deploy**

### Step 2: Connect to Your Pod

1. Wait for pod to be "Running"
2. Click **Connect** → **Start Web Terminal** or use SSH
3. You should see a terminal prompt

### Step 3: Upload Training Files

**If using direct upload:**
```bash
# In the web terminal or SSH
cd /workspace

# Upload your zip file using the RunPod file manager
# Then unzip:
unzip distil_crisper_training.zip
```

**If using GitHub:**
```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git distil_crisper_training
```

### Step 4: Set Environment Variables

```bash
# Required: Your HuggingFace token
export HF_TOKEN="hf_your_token_here"

# Required: Your HuggingFace username
export HF_USERNAME="your_username"

# Optional: Weights & Biases key
export WANDB_API_KEY="your_wandb_key"

# Save to bashrc for persistence
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
echo 'export HF_USERNAME="your_username"' >> ~/.bashrc
```

---

## Running the Training

### Option 1: Full Automated Pipeline

This runs everything from setup to conversion:

```bash
cd /workspace/distil_crisper_training/scripts
chmod +x *.sh
bash run_training.sh
```

This will:
1. Install all dependencies (~5 min)
2. Download models (~10 min)
3. Prepare datasets (~30-60 min for streaming setup)
4. Generate pseudo-labels (~2-4 hours)
5. Train the model (~20-40 hours)
6. Convert to CTranslate2 (~5 min)

### Option 2: Step-by-Step (Recommended for First Time)

Run each step individually to monitor progress:

```bash
cd /workspace/distil_crisper_training/scripts

# Step 1: Setup environment
bash 01_cloud_setup.sh

# Step 2: Prepare datasets
python3 02_prepare_data.py --config ../config.yaml

# Step 3: Generate pseudo-labels
python3 03_generate_pseudo_labels.py --config ../config.yaml

# Step 4: Train (this is the long step)
python3 04_train_distillation.py --config ../config.yaml

# Step 5: Convert to CTranslate2
python3 05_convert_to_ctranslate2.py --config ../config.yaml
```

---

## Monitoring & Resuming

### Monitoring Training

**Option 1: TensorBoard**
```bash
# In a new terminal
tensorboard --logdir /workspace/tensorboard --port 6006

# Access via RunPod's HTTP port mapping
```

**Option 2: Weights & Biases**
- Go to https://wandb.ai
- Find your project "distil-crisperwhisper"
- View live metrics

**Option 3: Check Logs**
```bash
tail -f /workspace/logs/training.log
```

### Resuming After Interruption

If your spot instance gets interrupted:

1. Start a new pod (or restart existing)
2. Set environment variables again
3. Resume training:

```bash
cd /workspace/distil_crisper_training/scripts

# Resume from last checkpoint
python3 04_train_distillation.py --config ../config.yaml --resume
```

The script automatically:
- Finds the latest checkpoint
- Loads model weights
- Resumes from the correct training step
- Continues pushing checkpoints to HuggingFace Hub

### Checking Progress

```bash
# See checkpoints
ls -la /workspace/checkpoints/

# See training step from checkpoint
cat /workspace/checkpoints/checkpoint-*/training_state.pt | python3 -c "
import torch, sys
state = torch.load(sys.stdin.buffer)
print(f'Step: {state[\"global_step\"]}')
print(f'Best Loss: {state[\"best_loss\"]:.4f}')
"
```

---

## Final Steps

### 1. Download Your Model

After training completes:

```bash
# The final model is at:
ls /workspace/output/distil-crisperwhisper-ct2/

# Download using RunPod file manager
# Or copy to your HuggingFace Hub
```

### 2. Test Your Model

```bash
python3 << 'EOF'
from faster_whisper import WhisperModel

model = WhisperModel(
    "/workspace/output/distil-crisperwhisper-ct2",
    device="cuda",
    compute_type="float16"
)

# Test with a sample audio file
segments, info = model.transcribe("/path/to/test.wav")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
EOF
```

### 3. Copy to Your Local Machine

**Option A: Download from HuggingFace Hub**
```bash
# On your local machine
pip install huggingface_hub
huggingface-cli download YOUR_USERNAME/distil-crisperwhisper-large-v3
```

**Option B: Direct Download**
- Use RunPod's file browser to download the `/workspace/output/` folder

### 4. Stop Your Pod

**Important**: Don't forget to stop your pod when done!

1. Go to RunPod Console
2. Click on your pod
3. Click **Stop** (keeps data) or **Terminate** (deletes everything)

---

## Troubleshooting

### Out of Memory (OOM) Errors

**During pseudo-label generation:**
```yaml
# In config.yaml, reduce:
teacher:
  pseudo_label_batch_size: 16  # Try 8 or 4
```

**During training:**
```yaml
# In config.yaml, reduce:
training:
  per_device_train_batch_size: 4  # Try 2
  gradient_accumulation_steps: 16  # Increase to maintain effective batch size
```

### GigaSpeech Access Denied

```
Error: 401 Unauthorized
```

**Solution:**
1. Go to https://huggingface.co/datasets/speechcolab/gigaspeech
2. Click "Agree and access repository"
3. Wait a few minutes for access to propagate
4. Re-run the data preparation

### Connection Lost / Pod Stopped

This is normal with spot instances!

**Solution:**
1. Start a new pod
2. Your data is safe on the persistent volume
3. Resume: `python3 04_train_distillation.py --resume`

### Training Too Slow

**Check GPU utilization:**
```bash
nvidia-smi -l 1
```

If GPU utilization is low:
- Increase batch size
- Increase dataloader workers
- Check if data loading is the bottleneck

### Model Quality Issues

If the distilled model quality is poor:

1. **Train longer**: Increase `max_steps` to 100000
2. **Keep more layers**: Increase `decoder_layers` to 12 or 16
3. **More data**: Ensure GigaSpeech and VoxPopuli are both being used
4. **Lower learning rate**: Try `5e-5` instead of `1e-4`

---

## Cost Estimates

| Component | Time | Cost (A100 80GB) |
|-----------|------|------------------|
| Setup & Data Prep | ~2 hrs | ~$5 |
| Pseudo-Labels | ~4 hrs | ~$10 |
| Training (50k steps) | ~30 hrs | ~$75 |
| Conversion & Testing | ~1 hr | ~$2 |
| **Total** | ~37 hrs | **~$92** |

With spot instances (50% cheaper): **~$46**

Storage costs: ~$20/month for 200GB volume

**Your $2500 budget allows for:**
- Multiple training runs with different hyperparameters
- Longer training for better quality
- Experimentation with different architectures

---

## Next Steps

After successful training:

1. **Integrate with VeilVoice**: Copy the CTranslate2 model to your VeilVoice installation
2. **Benchmark**: Compare speed and accuracy against the original CrisperWhisper
3. **Fine-tune**: If needed, fine-tune on your specific domain
4. **Share**: Consider publishing your model to HuggingFace Hub (make repo public)

---

## Support

If you encounter issues:
1. Check the logs in `/workspace/logs/`
2. Review this troubleshooting guide
3. Check the config.yaml settings
