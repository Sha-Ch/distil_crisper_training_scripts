#!/bin/bash
# =============================================================================
# Master Training Script for Distil-CrisperWhisper
# =============================================================================
# This script orchestrates the entire training pipeline:
# 1. Setup cloud environment
# 2. Prepare datasets
# 3. Generate pseudo-labels
# 4. Train distillation
# 5. Convert to CTranslate2
#
# Usage:
#   Full pipeline:    bash run_training.sh
#   Resume training:  bash run_training.sh --resume
#   Specific step:    bash run_training.sh --step train
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="/workspace"
CONFIG_FILE="${SCRIPT_DIR}/../config.yaml"
LOG_DIR="${WORKSPACE}/logs"

# Parse arguments
RESUME=false
STEP="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME=true
            shift
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_DIR}/training.log"
}

# Print header
print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║          Distil-CrisperWhisper Training Pipeline             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check environment
check_environment() {
    log "INFO" "Checking environment..."

    # Check GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log "ERROR" "nvidia-smi not found. GPU required."
        exit 1
    fi

    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    log "INFO" "GPU: ${GPU_INFO}"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log "ERROR" "Python3 not found"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version)
    log "INFO" "${PYTHON_VERSION}"

    # Check HF token
    if [ -z "$HF_TOKEN" ]; then
        log "WARN" "HF_TOKEN not set. Some features may not work."
    fi

    # Check config file
    if [ ! -f "${CONFIG_FILE}" ]; then
        log "ERROR" "Config file not found: ${CONFIG_FILE}"
        exit 1
    fi

    log "INFO" "Using config: ${CONFIG_FILE}"
}

# Step 1: Setup
run_setup() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Step 1: Environment Setup${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

    log "INFO" "Running setup script..."
    bash "${SCRIPT_DIR}/01_cloud_setup.sh"
    log "INFO" "Setup complete"
}

# Step 2: Prepare data
run_prepare_data() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Step 2: Dataset Preparation${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

    log "INFO" "Preparing datasets..."
    python3 "${SCRIPT_DIR}/02_prepare_data.py" --config "${CONFIG_FILE}" 2>&1 | tee "${LOG_DIR}/prepare_data.log"
    log "INFO" "Dataset preparation complete"
}

# Step 3: Generate pseudo-labels
run_pseudo_labels() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Step 3: Pseudo-Label Generation${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

    RESUME_FLAG=""
    if [ "$RESUME" = true ]; then
        RESUME_FLAG="--resume"
    fi

    log "INFO" "Generating pseudo-labels..."
    python3 "${SCRIPT_DIR}/03_generate_pseudo_labels.py" --config "${CONFIG_FILE}" ${RESUME_FLAG} 2>&1 | tee "${LOG_DIR}/pseudo_labels.log"
    log "INFO" "Pseudo-label generation complete"
}

# Step 4: Train distillation
run_train() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Step 4: Distillation Training${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

    RESUME_FLAG=""
    if [ "$RESUME" = true ]; then
        RESUME_FLAG="--resume"
    fi

    log "INFO" "Starting distillation training..."
    python3 "${SCRIPT_DIR}/04_train_distillation.py" --config "${CONFIG_FILE}" ${RESUME_FLAG} 2>&1 | tee "${LOG_DIR}/training.log"
    log "INFO" "Training complete"
}

# Step 5: Convert to CTranslate2
run_convert() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Step 5: CTranslate2 Conversion${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

    log "INFO" "Converting to CTranslate2..."
    python3 "${SCRIPT_DIR}/05_convert_to_ctranslate2.py" --config "${CONFIG_FILE}" 2>&1 | tee "${LOG_DIR}/conversion.log"
    log "INFO" "Conversion complete"
}

# Print summary
print_summary() {
    echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Training Pipeline Complete!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}\n"

    OUTPUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG_FILE}'))['paths']['output_dir'])")

    echo -e "Output files:"
    echo -e "  HuggingFace model: ${OUTPUT_DIR}/distil-crisperwhisper-final"
    echo -e "  CTranslate2 model: ${OUTPUT_DIR}/distil-crisperwhisper-ct2"
    echo -e ""
    echo -e "Logs: ${LOG_DIR}"
    echo -e ""
    echo -e "To use with faster-whisper:"
    echo -e "  from faster_whisper import WhisperModel"
    echo -e "  model = WhisperModel('${OUTPUT_DIR}/distil-crisperwhisper-ct2')"
    echo -e ""
}

# Main execution
main() {
    print_header
    check_environment

    case $STEP in
        all)
            run_setup
            run_prepare_data
            run_pseudo_labels
            run_train
            run_convert
            print_summary
            ;;
        setup)
            run_setup
            ;;
        data)
            run_prepare_data
            ;;
        labels)
            run_pseudo_labels
            ;;
        train)
            run_train
            ;;
        convert)
            run_convert
            ;;
        *)
            echo -e "${RED}Unknown step: $STEP${NC}"
            echo "Valid steps: all, setup, data, labels, train, convert"
            exit 1
            ;;
    esac

    log "INFO" "Pipeline completed successfully!"
}

# Run
main
