#!/bin/bash
# =============================================================================
# Cloud Bootstrap Script for Distil-CrisperWhisper Training
# =============================================================================
# Run this ONCE when you first start your cloud instance
# This sets up the environment, downloads models, and prepares everything
#
# Usage: bash 01_cloud_setup.sh
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Distil-CrisperWhisper Cloud Setup${NC}"
echo -e "${BLUE}=============================================${NC}"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WORKSPACE="/workspace"
DATA_DIR="${WORKSPACE}/data"
CHECKPOINT_DIR="${WORKSPACE}/checkpoints"
OUTPUT_DIR="${WORKSPACE}/output"
HF_CACHE="${WORKSPACE}/hf_cache"
SCRIPTS_DIR="${WORKSPACE}/distil_crisper_training"

# -----------------------------------------------------------------------------
# Check if running on cloud
# -----------------------------------------------------------------------------
check_environment() {
    echo -e "\n${YELLOW}[1/8] Checking environment...${NC}"

    # Check for GPU
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}ERROR: nvidia-smi not found. Are you on a GPU instance?${NC}"
        exit 1
    fi

    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    echo -e "${GREEN}✓ GPU detected: ${GPU_INFO}${NC}"

    # Check VRAM
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$VRAM_MB" -lt 40000 ]; then
        echo -e "${YELLOW}WARNING: Less than 40GB VRAM detected. Training may be slow.${NC}"
        echo -e "${YELLOW}Recommended: A100 80GB or better${NC}"
    fi

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}ERROR: Python3 not found${NC}"
        exit 1
    fi
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ ${PYTHON_VERSION}${NC}"

    # Check CUDA
    if [ -n "$CUDA_HOME" ]; then
        echo -e "${GREEN}✓ CUDA_HOME: ${CUDA_HOME}${NC}"
    else
        echo -e "${YELLOW}WARNING: CUDA_HOME not set${NC}"
    fi
}

# -----------------------------------------------------------------------------
# Create directory structure
# -----------------------------------------------------------------------------
create_directories() {
    echo -e "\n${YELLOW}[2/8] Creating directory structure...${NC}"

    mkdir -p "${DATA_DIR}"
    mkdir -p "${DATA_DIR}/gigaspeech"
    mkdir -p "${DATA_DIR}/voxpopuli"
    mkdir -p "${DATA_DIR}/librispeech"
    mkdir -p "${DATA_DIR}/pseudo_labels"
    mkdir -p "${CHECKPOINT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "${HF_CACHE}"
    mkdir -p "${WORKSPACE}/logs"
    mkdir -p "${WORKSPACE}/tensorboard"

    echo -e "${GREEN}✓ Directory structure created${NC}"
    echo "  ${DATA_DIR}"
    echo "  ${CHECKPOINT_DIR}"
    echo "  ${OUTPUT_DIR}"
}

# -----------------------------------------------------------------------------
# Install Python dependencies
# -----------------------------------------------------------------------------
install_dependencies() {
    echo -e "\n${YELLOW}[3/8] Installing Python dependencies...${NC}"

    # Upgrade pip
    pip install --upgrade pip

    # Install requirements
    pip install -r "${SCRIPTS_DIR}/requirements.txt"

    # Verify critical imports
    python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
    python3 -c "import transformers; print(f'Transformers {transformers.__version__}')"
    python3 -c "import datasets; print(f'Datasets {datasets.__version__}')"

    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# -----------------------------------------------------------------------------
# Setup HuggingFace authentication
# -----------------------------------------------------------------------------
setup_huggingface() {
    echo -e "\n${YELLOW}[4/8] Setting up HuggingFace authentication...${NC}"

    # Check if HF_TOKEN is set
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${RED}ERROR: HF_TOKEN environment variable not set${NC}"
        echo -e "${YELLOW}Please set it with: export HF_TOKEN=your_token_here${NC}"
        echo -e "${YELLOW}Get your token from: https://huggingface.co/settings/tokens${NC}"
        exit 1
    fi

    # Login to HuggingFace
    huggingface-cli login --token "$HF_TOKEN"

    # Set cache directory
    export HF_HOME="${HF_CACHE}"
    export TRANSFORMERS_CACHE="${HF_CACHE}"
    export HF_DATASETS_CACHE="${HF_CACHE}/datasets"

    echo -e "${GREEN}✓ HuggingFace authenticated${NC}"
}

# -----------------------------------------------------------------------------
# Download CrisperWhisper teacher model
# -----------------------------------------------------------------------------
download_teacher_model() {
    echo -e "\n${YELLOW}[5/8] Downloading CrisperWhisper teacher model...${NC}"

    python3 << 'EOF'
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_id = "nyrahealth/CrisperWhisper"
cache_dir = os.environ.get("HF_HOME", "/workspace/hf_cache")

print(f"Downloading {model_id}...")
print(f"Cache directory: {cache_dir}")

# Download model
model = WhisperForConditionalGeneration.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype="auto"
)

# Download processor
processor = WhisperProcessor.from_pretrained(
    model_id,
    cache_dir=cache_dir
)

print(f"✓ Model downloaded successfully")
print(f"  Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
EOF

    echo -e "${GREEN}✓ Teacher model ready${NC}"
}

# -----------------------------------------------------------------------------
# Download base student model architecture
# -----------------------------------------------------------------------------
download_student_base() {
    echo -e "\n${YELLOW}[6/8] Downloading Whisper-large-v3 base architecture...${NC}"

    python3 << 'EOF'
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_id = "openai/whisper-large-v3"
cache_dir = os.environ.get("HF_HOME", "/workspace/hf_cache")

print(f"Downloading {model_id}...")

# Download model (for architecture reference)
model = WhisperForConditionalGeneration.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype="auto"
)

# Download processor
processor = WhisperProcessor.from_pretrained(
    model_id,
    cache_dir=cache_dir
)

print(f"✓ Base model downloaded")
EOF

    echo -e "${GREEN}✓ Student base architecture ready${NC}"
}

# -----------------------------------------------------------------------------
# Setup Weights & Biases (optional)
# -----------------------------------------------------------------------------
setup_wandb() {
    echo -e "\n${YELLOW}[7/8] Setting up Weights & Biases...${NC}"

    if [ -z "$WANDB_API_KEY" ]; then
        echo -e "${YELLOW}WANDB_API_KEY not set. Skipping W&B setup.${NC}"
        echo -e "${YELLOW}To enable: export WANDB_API_KEY=your_key${NC}"
        return 0
    fi

    wandb login "$WANDB_API_KEY"
    echo -e "${GREEN}✓ Weights & Biases configured${NC}"
}

# -----------------------------------------------------------------------------
# Create environment file
# -----------------------------------------------------------------------------
create_env_file() {
    echo -e "\n${YELLOW}[8/8] Creating environment configuration...${NC}"

    cat > "${WORKSPACE}/.env" << EOF
# Distil-CrisperWhisper Environment Configuration
# Generated by setup script

# Paths
WORKSPACE=${WORKSPACE}
DATA_DIR=${DATA_DIR}
CHECKPOINT_DIR=${CHECKPOINT_DIR}
OUTPUT_DIR=${OUTPUT_DIR}

# HuggingFace
HF_HOME=${HF_CACHE}
TRANSFORMERS_CACHE=${HF_CACHE}
HF_DATASETS_CACHE=${HF_CACHE}/datasets

# CUDA
CUDA_VISIBLE_DEVICES=0

# Training
TOKENIZERS_PARALLELISM=false
EOF

    # Also add to bashrc for persistence
    echo "source ${WORKSPACE}/.env" >> ~/.bashrc

    echo -e "${GREEN}✓ Environment file created at ${WORKSPACE}/.env${NC}"
}

# -----------------------------------------------------------------------------
# Print summary
# -----------------------------------------------------------------------------
print_summary() {
    echo -e "\n${BLUE}=============================================${NC}"
    echo -e "${GREEN}  Setup Complete!${NC}"
    echo -e "${BLUE}=============================================${NC}"
    echo ""
    echo -e "Directory structure:"
    echo -e "  Data:        ${DATA_DIR}"
    echo -e "  Checkpoints: ${CHECKPOINT_DIR}"
    echo -e "  Output:      ${OUTPUT_DIR}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Run: source ${WORKSPACE}/.env"
    echo -e "  2. Run: python3 02_prepare_data.py"
    echo -e "  3. Run: python3 03_generate_pseudo_labels.py"
    echo -e "  4. Run: python3 04_train_distillation.py"
    echo ""
    echo -e "${YELLOW}To resume training after spot instance interruption:${NC}"
    echo -e "  python3 04_train_distillation.py --resume"
    echo ""
}

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
main() {
    check_environment
    create_directories
    install_dependencies
    setup_huggingface
    download_teacher_model
    download_student_base
    setup_wandb
    create_env_file
    print_summary
}

main "$@"
