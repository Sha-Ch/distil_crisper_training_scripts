#!/bin/bash
# =============================================================================
# Multi-GPU Setup Script for 4x H100 NVL on RunPod
# =============================================================================
# This script sets up the environment for training distil-crisperwhisper
# following the official distil-whisper v3.5 methodology.
#
# Hardware: 4x H100 NVL (94GB each) = 376GB total VRAM
# Budget: ~$900 (~10.44/hr spot for 86 hours)
#
# Usage: bash 01_multi_gpu_setup.sh
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

WORKSPACE="/workspace"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Distil-CrisperWhisper: Official Methodology Setup            ║${NC}"
echo -e "${BLUE}║     4x H100 NVL (376GB VRAM) - All-In Configuration              ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════╝${NC}"

# -----------------------------------------------------------------------------
# Check GPU Configuration
# -----------------------------------------------------------------------------
check_gpus() {
    echo -e "\n${YELLOW}[1/8] Checking GPU configuration...${NC}"

    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}ERROR: nvidia-smi not found. Are you on a GPU instance?${NC}"
        exit 1
    fi

    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

    echo -e "  GPUs detected: ${CYAN}${GPU_COUNT}x ${GPU_NAME}${NC}"
    echo -e "  Memory per GPU: ${CYAN}${GPU_MEMORY} MB${NC}"

    if [ "$GPU_COUNT" -lt 4 ]; then
        echo -e "${YELLOW}WARNING: Expected 4 GPUs but found ${GPU_COUNT}.${NC}"
        echo -e "${YELLOW}Training will work but may be slower than estimated.${NC}"
    fi

    if [ "$GPU_COUNT" -ge 4 ]; then
        echo -e "${GREEN}✓ 4+ GPUs detected - optimal configuration${NC}"
    fi

    # Check NVLINK topology
    echo -e "\n  GPU Topology:"
    nvidia-smi topo -m 2>/dev/null || echo "  (topology info not available)"
}

# -----------------------------------------------------------------------------
# Check Storage
# -----------------------------------------------------------------------------
check_storage() {
    echo -e "\n${YELLOW}[2/8] Checking storage configuration...${NC}"

    WORKSPACE_SIZE=$(df -BG "${WORKSPACE}" 2>/dev/null | tail -1 | awk '{print $2}' | tr -d 'G')
    WORKSPACE_AVAIL=$(df -BG "${WORKSPACE}" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')

    echo -e "  Workspace: ${WORKSPACE}"
    echo -e "  Total size: ${CYAN}${WORKSPACE_SIZE}GB${NC}"
    echo -e "  Available: ${CYAN}${WORKSPACE_AVAIL}GB${NC}"

    if [ "$WORKSPACE_AVAIL" -lt 200 ]; then
        echo -e "${RED}ERROR: Insufficient storage. Need at least 200GB available.${NC}"
        echo -e "${RED}Current available: ${WORKSPACE_AVAIL}GB${NC}"
        exit 1
    fi

    if [ "$WORKSPACE_AVAIL" -lt 800 ]; then
        echo -e "${YELLOW}WARNING: Limited storage (${WORKSPACE_AVAIL}GB available).${NC}"
        echo -e "${YELLOW}Will use streaming for large datasets.${NC}"
    fi

    if [ "$WORKSPACE_AVAIL" -ge 800 ]; then
        echo -e "${GREEN}✓ Good storage capacity for training${NC}"
    fi

    # Create directories
    mkdir -p "${WORKSPACE}/data"
    mkdir -p "${WORKSPACE}/checkpoints"
    mkdir -p "${WORKSPACE}/output"
    mkdir -p "${WORKSPACE}/hf_cache"
    mkdir -p "${WORKSPACE}/pseudo_labels"
    mkdir -p "${WORKSPACE}/tensorboard"
    mkdir -p "${WORKSPACE}/logs"

    echo -e "${GREEN}✓ Directories created${NC}"
}

# -----------------------------------------------------------------------------
# Check Environment Variables
# -----------------------------------------------------------------------------
check_env() {
    echo -e "\n${YELLOW}[3/8] Checking environment variables...${NC}"

    if [ -z "$HF_TOKEN" ]; then
        echo -e "${RED}ERROR: HF_TOKEN not set${NC}"
        echo -e "Please run: export HF_TOKEN=\"your_huggingface_token\""
        exit 1
    fi
    echo -e "${GREEN}✓ HF_TOKEN is set${NC}"

    if [ -z "$HF_USERNAME" ]; then
        echo -e "${YELLOW}WARNING: HF_USERNAME not set${NC}"
        echo -e "Please run: export HF_USERNAME=\"your_username\""
    else
        echo -e "${GREEN}✓ HF_USERNAME is set: ${HF_USERNAME}${NC}"
    fi

    if [ -z "$WANDB_API_KEY" ]; then
        echo -e "${YELLOW}NOTE: WANDB_API_KEY not set (optional)${NC}"
    else
        echo -e "${GREEN}✓ WANDB_API_KEY is set${NC}"
    fi

    # Set HF_HOME to workspace for caching
    export HF_HOME="${WORKSPACE}/hf_cache"
    export HF_DATASETS_CACHE="${WORKSPACE}/hf_cache/datasets"
    export TRANSFORMERS_CACHE="${WORKSPACE}/hf_cache/transformers"

    echo -e "${GREEN}✓ Cache directories set to workspace${NC}"
}

# -----------------------------------------------------------------------------
# Install System Dependencies
# -----------------------------------------------------------------------------
install_system_deps() {
    echo -e "\n${YELLOW}[4/8] Installing system dependencies...${NC}"

    apt-get update -qq
    apt-get install -y -qq ffmpeg libsndfile1 git-lfs htop tmux > /dev/null 2>&1

    echo -e "${GREEN}✓ System dependencies installed${NC}"
}

# -----------------------------------------------------------------------------
# Install Python Dependencies
# -----------------------------------------------------------------------------
install_python_deps() {
    echo -e "\n${YELLOW}[5/8] Installing Python dependencies...${NC}"

    # Upgrade pip
    pip install --upgrade pip -q

    # Core ML libraries (compatible with H100)
    pip install -q torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

    # Transformers ecosystem
    pip install -q \
        transformers>=4.40.0 \
        datasets>=2.18.0 \
        accelerate>=0.29.0 \
        evaluate>=0.4.0

    # Audio processing
    pip install -q \
        soundfile>=0.12.0 \
        librosa>=0.10.0 \
        audioread>=3.0.0

    # Training utilities
    pip install -q \
        jiwer>=3.0.0 \
        wandb>=0.16.0 \
        tensorboard>=2.16.0 \
        rich>=13.0.0 \
        tqdm>=4.66.0

    # Multi-GPU training
    pip install -q \
        deepspeed>=0.14.0 \
        ninja

    # Final conversion
    pip install -q ctranslate2>=4.0.0

    echo -e "${GREEN}✓ Python dependencies installed${NC}"

    # Verify key imports
    echo -e "  Verifying installations..."
    python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
    python3 -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
    python3 -c "import accelerate; print(f'  Accelerate: {accelerate.__version__}')"
    python3 -c "import datasets; print(f'  Datasets: {datasets.__version__}')"
}

# -----------------------------------------------------------------------------
# Setup Accelerate for Multi-GPU
# -----------------------------------------------------------------------------
setup_accelerate() {
    echo -e "\n${YELLOW}[6/8] Configuring Accelerate for multi-GPU training...${NC}"

    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    # Create accelerate config
    mkdir -p ~/.cache/huggingface/accelerate

    cat > ~/.cache/huggingface/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: ${GPU_COUNT}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

    echo -e "${GREEN}✓ Accelerate configured for ${GPU_COUNT} GPUs with bf16${NC}"

    # Verify configuration
    accelerate env
}

# -----------------------------------------------------------------------------
# Login to HuggingFace
# -----------------------------------------------------------------------------
login_hf() {
    echo -e "\n${YELLOW}[7/8] Logging into HuggingFace...${NC}"

    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

    # Verify access to required datasets
    echo -e "  Checking dataset access..."

    # LibriSpeech (public)
    python3 -c "from datasets import load_dataset; load_dataset('librispeech_asr', 'clean', split='test', streaming=True, trust_remote_code=True)" 2>/dev/null && \
        echo -e "${GREEN}  ✓ LibriSpeech accessible${NC}" || \
        echo -e "${YELLOW}  ⚠ LibriSpeech check failed (may work anyway)${NC}"

    echo -e "${GREEN}✓ HuggingFace login complete${NC}"
}

# -----------------------------------------------------------------------------
# Clone Official Distil-Whisper Repository
# -----------------------------------------------------------------------------
clone_distil_whisper() {
    echo -e "\n${YELLOW}[8/8] Setting up official distil-whisper reference...${NC}"

    DISTIL_REPO="${WORKSPACE}/distil-whisper-official"

    if [ -d "$DISTIL_REPO" ]; then
        echo -e "  Repository exists, pulling latest..."
        cd "$DISTIL_REPO" && git pull --quiet
    else
        echo -e "  Cloning official repository..."
        git clone --quiet https://github.com/huggingface/distil-whisper.git "$DISTIL_REPO"
    fi

    echo -e "${GREEN}✓ Official distil-whisper repository ready${NC}"
    echo -e "  Location: ${DISTIL_REPO}"
}

# -----------------------------------------------------------------------------
# Print Summary
# -----------------------------------------------------------------------------
print_summary() {
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    WORKSPACE_AVAIL=$(df -BG "${WORKSPACE}" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')

    echo -e "\n${BLUE}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                      SETUP COMPLETE                               ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════╝${NC}"

    echo -e "\n${GREEN}Configuration Summary:${NC}"
    echo -e "  GPUs: ${CYAN}${GPU_COUNT}x ${GPU_NAME}${NC}"
    echo -e "  Storage: ${CYAN}${WORKSPACE_AVAIL}GB available${NC}"
    echo -e "  HF User: ${CYAN}${HF_USERNAME:-'not set'}${NC}"
    echo -e "  W&B: ${CYAN}$([ -n "$WANDB_API_KEY" ] && echo 'configured' || echo 'not configured')${NC}"

    echo -e "\n${GREEN}Next Steps:${NC}"
    echo -e "  ${CYAN}1.${NC} Generate pseudo-labels with CrisperWhisper:"
    echo -e "     ${YELLOW}python3 02_prepare_data_streaming.py --config ../config.yaml${NC}"
    echo -e ""
    echo -e "  ${CYAN}2.${NC} Start distillation training:"
    echo -e "     ${YELLOW}accelerate launch 04_train_distillation.py --config ../config.yaml${NC}"
    echo -e ""
    echo -e "  ${CYAN}3.${NC} Or run full pipeline:"
    echo -e "     ${YELLOW}bash run_full_pipeline.sh${NC}"

    echo -e "\n${GREEN}Estimated Timeline:${NC}"
    echo -e "  Pseudo-label generation: ~24-48 hours"
    echo -e "  Training (80k steps): ~40-60 hours"
    echo -e "  Total: ~3-5 days"
    echo -e "  Estimated cost: ~\$500-900 (spot instances)"

    echo -e "\n${YELLOW}TIP: Use 'tmux' to keep training running after disconnect:${NC}"
    echo -e "  tmux new -s training"
    echo -e "  # ... start training ..."
    echo -e "  # Press Ctrl+B, then D to detach"
    echo -e "  tmux attach -t training  # to reattach"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    check_gpus
    check_storage
    check_env
    install_system_deps
    install_python_deps
    setup_accelerate
    login_hf
    clone_distil_whisper
    print_summary
}

main "$@"
