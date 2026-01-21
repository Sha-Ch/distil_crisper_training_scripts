#!/bin/bash
# =============================================================================
# Run Official HuggingFace Distil-Whisper Training Scripts
# =============================================================================
# This script uses the OFFICIAL distil-whisper repository from HuggingFace
# for maximum compatibility and proven results.
#
# Reference: https://github.com/huggingface/distil-whisper
#
# Usage: bash run_official_distilwhisper.sh
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

WORKSPACE="/workspace"
DISTIL_REPO="${WORKSPACE}/distil-whisper"

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Official Distil-Whisper Training Pipeline${NC}"
echo -e "${BLUE}=============================================${NC}"

# -----------------------------------------------------------------------------
# Step 1: Clone official repository
# -----------------------------------------------------------------------------
clone_repo() {
    echo -e "\n${YELLOW}[1/6] Cloning official distil-whisper repository...${NC}"

    if [ -d "${DISTIL_REPO}" ]; then
        echo -e "${GREEN}Repository already exists, pulling latest...${NC}"
        cd "${DISTIL_REPO}" && git pull
    else
        git clone https://github.com/huggingface/distil-whisper.git "${DISTIL_REPO}"
    fi

    cd "${DISTIL_REPO}/training"
    pip install -r requirements.txt

    echo -e "${GREEN}✓ Repository ready${NC}"
}

# -----------------------------------------------------------------------------
# Step 2: Create student model
# -----------------------------------------------------------------------------
create_student() {
    echo -e "\n${YELLOW}[2/6] Creating student model from CrisperWhisper...${NC}"

    # Use CrisperWhisper as teacher, create student with 2 decoder layers
    # (following official distil-whisper architecture)
    python create_student_model.py \
        --teacher_checkpoint "nyrahealth/CrisperWhisper" \
        --encoder_layers 32 \
        --decoder_layers 2 \
        --save_dir "${WORKSPACE}/distil-crisperwhisper-init"

    echo -e "${GREEN}✓ Student model created at ${WORKSPACE}/distil-crisperwhisper-init${NC}"
}

# -----------------------------------------------------------------------------
# Step 3: Generate pseudo-labels
# -----------------------------------------------------------------------------
generate_pseudo_labels() {
    echo -e "\n${YELLOW}[3/6] Generating pseudo-labels with CrisperWhisper...${NC}"

    # For GigaSpeech
    python run_pseudo_labelling.py \
        --model_name_or_path "nyrahealth/CrisperWhisper" \
        --dataset_name "speechcolab/gigaspeech" \
        --dataset_config_name "l" \
        --dataset_split_name "train" \
        --text_column_name "text" \
        --id_column_name "segment_id" \
        --output_dir "${WORKSPACE}/pseudo_labels/gigaspeech" \
        --wandb_project "distil-crisperwhisper-pseudo-labelling" \
        --per_device_eval_batch_size 32 \
        --dtype "float16" \
        --attn_implementation "sdpa" \
        --logging_steps 500 \
        --max_label_length 256 \
        --concatenate_audio \
        --preprocessing_batch_size 500 \
        --preprocessing_num_workers 8 \
        --dataloader_num_workers 8 \
        --report_to "wandb" \
        --language "en" \
        --task "transcribe" \
        --return_timestamps \
        --streaming

    # For VoxPopuli
    python run_pseudo_labelling.py \
        --model_name_or_path "nyrahealth/CrisperWhisper" \
        --dataset_name "facebook/voxpopuli" \
        --dataset_config_name "en" \
        --dataset_split_name "train" \
        --text_column_name "normalized_text" \
        --output_dir "${WORKSPACE}/pseudo_labels/voxpopuli" \
        --wandb_project "distil-crisperwhisper-pseudo-labelling" \
        --per_device_eval_batch_size 32 \
        --dtype "float16" \
        --attn_implementation "sdpa" \
        --logging_steps 500 \
        --max_label_length 256 \
        --concatenate_audio \
        --preprocessing_batch_size 500 \
        --preprocessing_num_workers 8 \
        --dataloader_num_workers 8 \
        --report_to "wandb" \
        --language "en" \
        --task "transcribe" \
        --return_timestamps \
        --streaming

    echo -e "${GREEN}✓ Pseudo-labels generated${NC}"
}

# -----------------------------------------------------------------------------
# Step 4: Run distillation training
# -----------------------------------------------------------------------------
run_distillation() {
    echo -e "\n${YELLOW}[4/6] Running distillation training...${NC}"

    # Train using official script with multiple datasets
    accelerate launch run_distillation.py \
        --model_name_or_path "${WORKSPACE}/distil-crisperwhisper-init" \
        --teacher_model_name_or_path "nyrahealth/CrisperWhisper" \
        --train_dataset_name "${WORKSPACE}/pseudo_labels/gigaspeech+${WORKSPACE}/pseudo_labels/voxpopuli" \
        --train_split_name "train+train" \
        --text_column_name "whisper_transcript+whisper_transcript" \
        --eval_dataset_name "librispeech_asr" \
        --eval_dataset_config_name "clean" \
        --eval_split_name "test" \
        --eval_text_column_name "text" \
        --output_dir "${WORKSPACE}/distil-crisperwhisper-trained" \
        --wandb_project "distil-crisperwhisper" \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --learning_rate 0.0001 \
        --warmup_steps 500 \
        --max_steps 80000 \
        --evaluation_strategy "steps" \
        --eval_steps 5000 \
        --save_strategy "steps" \
        --save_steps 2500 \
        --logging_steps 25 \
        --report_to "wandb" \
        --dtype "bfloat16" \
        --attn_implementation "sdpa" \
        --gradient_checkpointing \
        --freeze_encoder \
        --freeze_embed_positions \
        --wer_threshold 10 \
        --timestamp_probability 0.2 \
        --condition_on_prev_probability 0.2 \
        --push_to_hub \
        --hub_model_id "${HF_USERNAME}/distil-crisperwhisper" \
        --hub_private_repo

    echo -e "${GREEN}✓ Training complete${NC}"
}

# -----------------------------------------------------------------------------
# Step 5: Evaluate model
# -----------------------------------------------------------------------------
evaluate_model() {
    echo -e "\n${YELLOW}[5/6] Evaluating trained model...${NC}"

    python run_eval.py \
        --model_name_or_path "${WORKSPACE}/distil-crisperwhisper-trained" \
        --dataset_name "librispeech_asr" \
        --dataset_config_name "clean+other" \
        --dataset_split_name "test+test" \
        --text_column_name "text" \
        --output_dir "${WORKSPACE}/eval_results" \
        --per_device_eval_batch_size 16 \
        --dtype "float16" \
        --attn_implementation "sdpa" \
        --generation_max_length 256 \
        --report_to "wandb" \
        --wandb_project "distil-crisperwhisper-eval"

    echo -e "${GREEN}✓ Evaluation complete${NC}"
}

# -----------------------------------------------------------------------------
# Step 6: Convert to CTranslate2
# -----------------------------------------------------------------------------
convert_model() {
    echo -e "\n${YELLOW}[6/6] Converting to CTranslate2 format...${NC}"

    ct2-transformers-converter \
        --model "${WORKSPACE}/distil-crisperwhisper-trained" \
        --output_dir "${WORKSPACE}/distil-crisperwhisper-ct2" \
        --quantization float16 \
        --copy_files tokenizer.json preprocessor_config.json

    echo -e "${GREEN}✓ Model converted to CTranslate2${NC}"
    echo -e "${GREEN}  Output: ${WORKSPACE}/distil-crisperwhisper-ct2${NC}"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    # Check environment
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${RED}ERROR: HF_TOKEN not set${NC}"
        exit 1
    fi

    if [ -z "$HF_USERNAME" ]; then
        echo -e "${RED}ERROR: HF_USERNAME not set${NC}"
        exit 1
    fi

    clone_repo
    create_student
    generate_pseudo_labels
    run_distillation
    evaluate_model
    convert_model

    echo -e "\n${GREEN}=============================================${NC}"
    echo -e "${GREEN}  Training Pipeline Complete!${NC}"
    echo -e "${GREEN}=============================================${NC}"
    echo -e "\nFinal model: ${WORKSPACE}/distil-crisperwhisper-ct2"
}

main "$@"
