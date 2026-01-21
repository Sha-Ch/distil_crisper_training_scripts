#!/bin/bash
# =============================================================================
# Full Pipeline: Distil-CrisperWhisper Training
# =============================================================================
# Runs the complete training pipeline for distilling CrisperWhisper:
# 1. Environment setup and dependency installation
# 2. Pseudo-label generation with CrisperWhisper
# 3. Distillation training (80k steps)
# 4. CTranslate2 conversion for faster-whisper
#
# Hardware: 4x H100 NVL (376GB VRAM)
# Estimated time: 3-5 days
# Estimated cost: ~$600-900 (spot instances)
#
# Usage:
#   bash run_full_pipeline.sh              # Full pipeline
#   bash run_full_pipeline.sh --skip-setup # Skip setup (already done)
#   bash run_full_pipeline.sh --resume     # Resume from checkpoint
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${CONFIG_DIR}/config.yaml"
WORKSPACE="/workspace"
LOG_DIR="${WORKSPACE}/logs"

# Parse arguments
SKIP_SETUP=false
RESUME=false

for arg in "$@"; do
    case $arg in
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# Function to log messages
log() {
    echo -e "$1"
    echo -e "$(date +%Y-%m-%d\ %H:%M:%S) $1" >> "$MAIN_LOG"
}

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                           ║"
    echo "║     ██████╗ ██╗███████╗████████╗██╗██╗                                    ║"
    echo "║     ██╔══██╗██║██╔════╝╚══██╔══╝██║██║                                    ║"
    echo "║     ██║  ██║██║███████╗   ██║   ██║██║                                    ║"
    echo "║     ██║  ██║██║╚════██║   ██║   ██║██║                                    ║"
    echo "║     ██████╔╝██║███████║   ██║   ██║███████╗                               ║"
    echo "║     ╚═════╝ ╚═╝╚══════╝   ╚═╝   ╚═╝╚══════╝                               ║"
    echo "║                                                                           ║"
    echo "║           C R I S P E R W H I S P E R   T R A I N I N G                   ║"
    echo "║                                                                           ║"
    echo "║     Following Official Distil-Whisper v3.5 Methodology                    ║"
    echo "║     Hardware: 4x H100 NVL (376GB VRAM)                                    ║"
    echo "║                                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "${YELLOW}[CHECK] Verifying prerequisites...${NC}"

    # Check GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log "${RED}ERROR: nvidia-smi not found. Are you on a GPU instance?${NC}"
        exit 1
    fi

    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log "  GPUs detected: ${GPU_COUNT}"

    if [ "$GPU_COUNT" -lt 4 ]; then
        log "${YELLOW}  WARNING: Expected 4 GPUs, found ${GPU_COUNT}. Training may be slower.${NC}"
    fi

    # Check HF_TOKEN
    if [ -z "$HF_TOKEN" ]; then
        log "${RED}ERROR: HF_TOKEN environment variable not set${NC}"
        log "  Run: export HF_TOKEN=\"your_huggingface_token\""
        exit 1
    fi

    # Check config file
    if [ ! -f "$CONFIG_FILE" ]; then
        log "${RED}ERROR: Config file not found at ${CONFIG_FILE}${NC}"
        exit 1
    fi

    # Check storage
    WORKSPACE_AVAIL=$(df -BG "${WORKSPACE}" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
    log "  Storage available: ${WORKSPACE_AVAIL}GB"

    if [ "$WORKSPACE_AVAIL" -lt 200 ]; then
        log "${RED}ERROR: Insufficient storage. Need at least 200GB.${NC}"
        exit 1
    fi

    log "${GREEN}[CHECK] Prerequisites verified ✓${NC}"
}

# Step 1: Setup environment
run_setup() {
    log "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    log "${CYAN}  STEP 1: Environment Setup${NC}"
    log "${CYAN}═══════════════════════════════════════════════════════════════${NC}\n"

    if [ "$SKIP_SETUP" = true ]; then
        log "${YELLOW}  Skipping setup (--skip-setup flag)${NC}"
        return 0
    fi

    log "  Running setup script..."
    bash "${SCRIPT_DIR}/01_multi_gpu_setup.sh" 2>&1 | tee -a "$MAIN_LOG"

    log "${GREEN}  ✓ Setup complete${NC}"
}

# Step 2: Generate pseudo-labels
run_pseudo_labeling() {
    log "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    log "${CYAN}  STEP 2: Pseudo-Label Generation with CrisperWhisper${NC}"
    log "${CYAN}═══════════════════════════════════════════════════════════════${NC}\n"

    PSEUDO_LABELS_FILE="${WORKSPACE}/pseudo_labels/all_pseudo_labels.jsonl"

    # Check if already done
    if [ -f "$PSEUDO_LABELS_FILE" ] && [ "$RESUME" = false ]; then
        SAMPLE_COUNT=$(wc -l < "$PSEUDO_LABELS_FILE")
        log "${YELLOW}  Pseudo-labels already exist (${SAMPLE_COUNT} samples).${NC}"
        log "  Use --resume to regenerate or skip this step."

        read -p "  Skip pseudo-labeling? [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            log "  Skipping pseudo-labeling."
            return 0
        fi
    fi

    log "  Starting pseudo-label generation..."
    log "  This will take 24-48 hours on 4x H100 NVL."
    log "  Output: ${WORKSPACE}/pseudo_labels/"
    log ""

    cd "$SCRIPT_DIR"

    # Run with accelerate for multi-GPU
    accelerate launch 02_generate_pseudo_labels_multi_gpu.py \
        --config "$CONFIG_FILE" \
        2>&1 | tee -a "${LOG_DIR}/pseudo_labels_${TIMESTAMP}.log"

    # Verify output
    if [ -f "$PSEUDO_LABELS_FILE" ]; then
        SAMPLE_COUNT=$(wc -l < "$PSEUDO_LABELS_FILE")
        log "${GREEN}  ✓ Pseudo-labeling complete: ${SAMPLE_COUNT} samples${NC}"
    else
        log "${RED}  ERROR: Pseudo-labels file not created${NC}"
        exit 1
    fi
}

# Step 3: Train distillation model
run_training() {
    log "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    log "${CYAN}  STEP 3: Distillation Training${NC}"
    log "${CYAN}═══════════════════════════════════════════════════════════════${NC}\n"

    FINAL_MODEL="${WORKSPACE}/output/distil-crisperwhisper-final"

    # Check if already done
    if [ -d "$FINAL_MODEL" ] && [ "$RESUME" = false ]; then
        log "${YELLOW}  Final model already exists at ${FINAL_MODEL}${NC}"

        read -p "  Skip training? [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            log "  Skipping training."
            return 0
        fi
    fi

    log "  Starting distillation training..."
    log "  This will take 40-60 hours on 4x H100 NVL."
    log "  Checkpoints: ${WORKSPACE}/checkpoints/"
    log ""

    cd "$SCRIPT_DIR"

    # Build accelerate command
    ACCELERATE_CMD="accelerate launch 03_train_distillation_multi_gpu.py --config $CONFIG_FILE"

    if [ "$RESUME" = true ]; then
        ACCELERATE_CMD="$ACCELERATE_CMD --resume"
        log "  Resuming from latest checkpoint..."
    fi

    # Run training
    $ACCELERATE_CMD 2>&1 | tee -a "${LOG_DIR}/training_${TIMESTAMP}.log"

    # Verify output
    if [ -d "$FINAL_MODEL" ]; then
        log "${GREEN}  ✓ Training complete${NC}"
    else
        log "${RED}  ERROR: Final model not created${NC}"
        exit 1
    fi
}

# Step 4: Convert to CTranslate2
run_conversion() {
    log "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    log "${CYAN}  STEP 4: CTranslate2 Conversion${NC}"
    log "${CYAN}═══════════════════════════════════════════════════════════════${NC}\n"

    CT2_MODEL="${WORKSPACE}/output/distil-crisperwhisper-ct2"

    # Check if already done
    if [ -d "$CT2_MODEL" ]; then
        log "${YELLOW}  CTranslate2 model already exists at ${CT2_MODEL}${NC}"

        read -p "  Reconvert? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "  Skipping conversion."
            return 0
        fi
    fi

    log "  Converting to CTranslate2 format for faster-whisper..."

    cd "$SCRIPT_DIR"
    python3 04_convert_to_ctranslate2.py --config "$CONFIG_FILE" --force \
        2>&1 | tee -a "${LOG_DIR}/conversion_${TIMESTAMP}.log"

    # Verify output
    if [ -d "$CT2_MODEL" ]; then
        log "${GREEN}  ✓ Conversion complete${NC}"
    else
        log "${RED}  ERROR: CTranslate2 model not created${NC}"
        exit 1
    fi
}

# Print final summary
print_summary() {
    log "\n${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    log "${GREEN}  PIPELINE COMPLETE!${NC}"
    log "${GREEN}═══════════════════════════════════════════════════════════════${NC}\n"

    log "  ${BOLD}Your distil-CrisperWhisper model is ready!${NC}\n"

    log "  ${CYAN}Model locations:${NC}"
    log "    HuggingFace format: ${WORKSPACE}/output/distil-crisperwhisper-final/"
    log "    CTranslate2 format: ${WORKSPACE}/output/distil-crisperwhisper-ct2/"

    log "\n  ${CYAN}Usage with faster-whisper:${NC}"
    log "    from faster_whisper import WhisperModel"
    log "    model = WhisperModel("
    log "        \"${WORKSPACE}/output/distil-crisperwhisper-ct2\","
    log "        device=\"cuda\","
    log "        compute_type=\"float16\","
    log "    )"
    log "    segments, info = model.transcribe(\"audio.wav\", word_timestamps=True)"

    log "\n  ${CYAN}Performance:${NC}"
    log "    Speed: 6x faster than original CrisperWhisper"
    log "    VRAM:  ~2GB (vs 6GB original)"
    log "    Quality: Word-level timestamps preserved"

    log "\n  ${CYAN}Next steps:${NC}"
    log "    1. Download model from RunPod"
    log "    2. Integrate with your application"
    log "    3. Don't forget to STOP YOUR POD!"

    log "\n  ${BOLD}Logs saved to: ${LOG_DIR}/${NC}"
}

# Main execution
main() {
    print_banner

    log "Starting pipeline at $(date)"
    log "Config: ${CONFIG_FILE}"
    log "Logs: ${MAIN_LOG}"

    check_prerequisites
    run_setup
    run_pseudo_labeling
    run_training
    run_conversion
    print_summary

    log "\nPipeline finished at $(date)"
}

# Run main
main "$@"
