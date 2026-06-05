# =============================================================================
# Distil-CrisperWhisper — local single-RTX-4090 image (CUDA 12.4)
# =============================================================================
# Reproduces the project's tested stack (PyTorch 2.5.1 + cu124, the pins from
# scripts/01_multi_gpu_setup.sh) for ONE NVIDIA GPU under Docker on WSL2.
#
# Build/run via docker-compose (see docker-compose.yml + LOCAL_4090.md):
#   docker compose build
#   docker compose run --rm distil
#
# Code is bind-mounted at /app and the host data dir at /workspace, so the image
# itself is just the environment. deepspeed is intentionally OMITTED (unused on a
# single GPU and its source build is fragile).
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf_cache \
    HF_DATASETS_CACHE=/workspace/hf_cache/datasets \
    TRANSFORMERS_CACHE=/workspace/hf_cache/transformers \
    HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub \
    HF_HUB_DOWNLOAD_TIMEOUT=1800 \
    HF_HUB_DISABLE_XET=1 \
    HF_XET_CACHE=/workspace/hf_cache/xet

# System dependencies (Python 3.10 + audio libs). Mirrors install_system_deps().
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        ffmpeg libsndfile1 git git-lfs tmux ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# PyTorch 2.5.1 + CUDA 12.4 (the project's tested versions).
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124

# Python deps. Pins mirror scripts/01_multi_gpu_setup.sh:
#   - transformers <4.50 ; datasets <3.0 (so audio decodes via soundfile, not torchcodec)
RUN pip install \
        "numpy>=1.26.0" "fsspec>=2023.5.0" \
        "transformers>=4.40.0,<4.50.0" \
        "datasets>=2.18.0,<3.0.0" \
        "accelerate>=0.29.0" \
        "evaluate>=0.4.0" \
        "jiwer>=3.0.0" \
        "soundfile>=0.12.0" "librosa>=0.10.0" "audioread>=3.0.0" \
        "ctranslate2>=4.0.0" "faster-whisper>=1.0.0" \
        "tensorboard>=2.16.0" "wandb>=0.16.0" \
        "huggingface_hub>=0.21.0" \
        "pandas>=2.0.0" "pyarrow>=14.0.0" "tqdm>=4.66.0" \
        "pyyaml>=6.0" "rich>=13.0.0" "psutil>=5.9.0" \
        "python-dotenv>=1.0.0" "python-dateutil>=2.8.0"

# Optional faster HF downloads (scripts enable it only if installed).
RUN pip install hf_transfer || true

WORKDIR /app/scripts

CMD ["/bin/bash"]
