# Distil-CrisperWhisper Training — Claude Code Context

## Workspace Rules (MUST FOLLOW)

**CRITICAL: All development work must be performed in the worktree, never in the original directory.**

**CRITICAL: This project's code does NOT run on native Windows.** Training, pseudo-label generation, conversion, and testing all require a Linux host with NVIDIA GPU(s) (CUDA). Two supported execution hosts: (1) a **RunPod cloud pod** for the full multi-GPU run (see [RUNPOD_SETUP.md](RUNPOD_SETUP.md), [GUIDE.md](GUIDE.md), `config.yaml`); (2) a **local single RTX 4090 inside a Docker container on WSL2** for data prep + a PoC training run (see [LOCAL_4090.md](LOCAL_4090.md), `config.local.yaml`, `Dockerfile`, `docker-compose.yml`). The Docker/WSL2 path is Linux/CUDA, so it satisfies the Linux requirement — **native Windows execution remains unsupported**. The Windows side itself is for **editing, reading, and reviewing** only: never run the GPU scripts directly on Windows, and never claim a GPU result you didn't actually observe (on the pod or in the local container). Static checks that do not import torch/transformers/datasets — `py_compile` syntax parse, reading code, grep, and the **import-light `tests/` via `pytest`** — ARE fine locally on Windows.

**CRITICAL: Always invoke the `andrej-karpathy-skills:karpathy-guidelines` skill at the start of every session before doing any coding, reviewing, or refactoring work on this project. These guidelines must be applied to all code changes here.**

**CRITICAL: When coding, reviewing code, reading files, or performing any local codebase analysis, always read the entire related logic — the full function body, every call site that reaches it, every helper/dependency it calls into, and any sibling modules that share state with it — rather than partial snippets or the first N lines. Never assume behavior, type, control flow, threading model, lifecycle, or side effects when you are uncertain: stop and dig deeper into the codebase (more `Read`, `Grep`, `Glob`, subagent exploration) until you have the complete picture. Explicitly ignore context-window size, token budget, and response-length worries when deciding how much to read — getting the full context always wins over saving tokens. An assumption made without verification is a bug waiting to happen, and a half-read function is worse than an unread one because it gives false confidence. The two big files here — `scripts/02_generate_pseudo_labels_multi_gpu.py` (~4,200 lines) and `scripts/03_train_distillation_multi_gpu.py` (~980 lines) — are exactly where partial reads cause regressions; read them whole.**

| Directory | Path | Purpose |
|-----------|------|---------|
| **Worktree (Work Here)** | e.g. `C:\Users\shat\.claude-worktrees\distil_crisper_training\<worktree-name>` *(varies per session)* | All modifications, edits, and review |
| **Original (Read-Only)** | `C:\Users\shat\Desktop\distil_crisper_training` | Source of truth — NEVER modify directly |
| **Execution Host** | RunPod pod (`/workspace`) **or** local single 4090 in Docker/WSL2 (`/workspace` = host mount) | Where the scripts actually run (Linux + NVIDIA GPUs). Cloud = `config.yaml`; local = `config.local.yaml` ([LOCAL_4090.md](LOCAL_4090.md)) |

### Workflow Requirements

1. **All work is done in the worktree** — Make all code changes, file edits, and modifications exclusively in the worktree directory.
2. **Double-check all changes** — Before copying to the original directory, thoroughly review and verify all modifications in detail. Ensure correctness, completeness, proper error handling, and that no regressions are introduced. Because the code cannot be executed locally, verification is by **careful reading + reasoning** (and, when possible, a real run on the pod). State explicitly when something has only been reasoned about vs. actually run on a GPU.
3. **Copy with `cp` command only** — After verification, use the `cp` command to copy finalized changes to the original directory. This is the user's explicit instruction (see `toDoList`): *"copy the changes to the original directory with command cp and never to modify it directly. Original Directory: C:\Users\shat\Desktop\distil_crisper_training."*
4. **Re-read CLAUDE.md fully AND follow everything in it strictly AND reactivate karpathy-guidelines after every cp** — Immediately after copying files to the original directory: (a) re-read this entire CLAUDE.md from top to bottom, (b) treat the re-read as active re-commitment — every rule, path, convention, and gotcha in this file is binding and must be followed strictly and fully from this point in the session, and (c) re-invoke the `andrej-karpathy-skills:karpathy-guidelines` skill. Passive skimming does NOT satisfy this step. Long sessions drift: rules get paraphrased, path references go stale, and the karpathy anchors (think before coding, simplicity first, surgical changes, goal-driven execution) weaken as context accumulates. The re-read rebinds the canonical rules; the skill reactivation re-seeds the guardrails. All three — read, strict compliance, skill reactivation — happen together on every cp.
5. **Never modify original directly** — The original directory must never be edited directly; it receives only verified, completed work.
6. **Keep documentation in sync** — After significant changes (new scripts, renamed stages, config-schema or data-format changes, dataset list changes), update this CLAUDE.md, [GUIDE.md](GUIDE.md), and [RUNPOD_SETUP.md](RUNPOD_SETUP.md) so they stay accurate.
7. **Never commit secrets** — All credentials (HuggingFace token, W&B key) are passed via environment variables, never hard-coded. ⚠️ `toDoList` currently contains a real `HF_TOKEN` in plaintext — it must be **rotated** and never reproduced in code, commits, or this file.

### Example Workflow

```bash
# 1. Make changes in worktree (automatic - this is the working directory)
# 2. Verify changes are correct and complete (read the full logic; reason about GPU/dist behavior)
# 3. Copy verified files to original (substitute the actual worktree path for the session):
cp "C:\Users\shat\.claude-worktrees\distil_crisper_training\<worktree-name>\scripts\02_generate_pseudo_labels_multi_gpu.py" "C:\Users\shat\Desktop\distil_crisper_training\scripts\02_generate_pseudo_labels_multi_gpu.py"
# 4. Re-read CLAUDE.md in full, follow everything strictly, AND re-invoke andrej-karpathy-skills:karpathy-guidelines before any further work
```

---

## Project Overview

**Distil-CrisperWhisper Training** is a cloud-GPU pipeline that **knowledge-distills** the [CrisperWhisper](https://huggingface.co/nyrahealth/CrisperWhisper) ASR model (the *teacher*) into a small, fast *student* model, following the **official Distil-Whisper v3.5 methodology** ([paper](https://arxiv.org/abs/2311.00430), [code](https://github.com/huggingface/distil-whisper)). The student is then converted to **CTranslate2** format for use with **faster-whisper**.

The end goal is a model that preserves CrisperWhisper's strengths (accurate **word-level timestamps**, verbatim transcription of disfluencies/filler words, reduced hallucination on silence) while running **~6× faster** at **~2 GB VRAM** (vs ~6 GB), suitable for production. The intended downstream consumer is the sibling **VeilVoice** project (`C:\Users\shat\Desktop\VeilVoice`), which uses distil-whisper-class CTranslate2 models as a two-pass verifier.

**The pipeline has four stages:**
1. **Setup** — provision the pod, install deps, configure Accelerate, log in to HuggingFace.
2. **Pseudo-label generation** — run the CrisperWhisper teacher over millions of audio samples from 8 public ASR datasets, keep only transcriptions that pass a **WER ≤ 10%** quality gate, and write them as JSONL.
3. **Distillation training** — train a student (full **frozen** 32-layer encoder + **8** decoder layers, chosen for WER parity; distil-v3.5 used 2) on those pseudo-labels with a combined cross-entropy + KL-divergence loss.
4. **Conversion & test** — export the trained checkpoint to CTranslate2 (float16) and validate it with faster-whisper.

Because spot instances get preempted, **every long stage is checkpointed and resumable**, and disk pressure on the pod is actively managed (chunked downloads + HF-cache cleanup).

**Key characteristics:**
- Teacher: `nyrahealth/CrisperWhisper` (a Whisper-large-v3 fine-tune), float16, SDPA attention.
- Student: built from the **teacher's** weights/tokenizer (CrisperWhisper uses a retokenized tokenizer, so `student.base_model` is vestigial); 32 encoder layers copied & frozen, **8 decoder layers** (maximally-spaced; chosen for WER parity — distil-v3.5 used 2), bfloat16.
- Distillation loss (official): `loss = 0.8·CE + 1.0·KL`, temperature 2.0.
- Quality filter: spelling-tolerant, filler-aware WER with a 10% threshold; all-caps outputs rejected as hallucinations.
- Multi-GPU via 🤗 Accelerate, auto-scaling to whatever GPU count the pod has.

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.10 (pod); edited on Windows 11 |
| **Execution host** | RunPod cloud pod, Linux, NVIDIA GPUs, `/workspace` persistent volume |
| **Target hardware** | Multi-GPU. Config targets **7× H200 (141 GB)**; docstrings/guides reference **4× H100 NVL (94 GB)** and **A100 80/40 GB**. Scripts **auto-detect GPU count and per-GPU VRAM** and scale batch size/workers accordingly — see *Hardware drift* gotcha. |
| **ML framework** | PyTorch 2.5.1 + CUDA 12.4 (`--index-url .../cu124`) |
| **Teacher / Student models** | HuggingFace `transformers` (`WhisperForConditionalGeneration`, `WhisperProcessor`) |
| **Distributed training** | 🤗 `accelerate` (+ optional `deepspeed` ZeRO); NCCL backend |
| **Datasets** | 🤗 `datasets` (streaming + chunked download); 8 public ASR corpora |
| **Conversion** | `ctranslate2` (`TransformersConverter` / `ct2-transformers-converter`) |
| **Inference (test)** | `faster-whisper` (`WhisperModel`) |
| **Audio** | `torchaudio`, `soundfile`, `librosa`, `audioread` |
| **Metrics** | `jiwer` + custom spelling-tolerant WER + Whisper `EnglishTextNormalizer` |
| **Monitoring** | `rich` (live TUI dashboard), `tensorboard`, `wandb` |
| **Utilities** | `numpy`, `pandas`, `tqdm`, `pyarrow`, `pyyaml`, `psutil` |

PyTorch is **not** in `requirements.txt` — it must be installed first from the CUDA 12.4 index (the setup scripts do this). See [requirements.txt](requirements.txt) for the rest.

## Project Structure

```
distil_crisper_training/
├── CLAUDE.md                      # This file
├── GUIDE.md                       # Single-GPU / general step-by-step training guide (A100)
├── RUNPOD_SETUP.md                # 4×H100-NVL / multi-GPU setup guide (official methodology)
├── LOCAL_4090.md                  # Local single-RTX-4090 (Docker on WSL2) data-prep + PoC guide
├── config.yaml                    # Single source of truth for all hyperparameters & dataset list (CLOUD)
├── config.local.yaml              # Single-4090 profile: local paths, 1-GPU batch, fidelity flags
├── Dockerfile                     # Local CUDA 12.4 image (torch 2.5.1+cu124) for WSL2 (single-stage, no COPY)
├── docker-compose.yml             # GPU passthrough + host data mount (/workspace); reads .env
├── .dockerignore                  # Keeps build context tiny (excludes .venv/_workspace/.git/...)
├── .env.example                   # Template for HF_TOKEN / HF_USERNAME / DATA_DIR (copy to .env)
├── start.ps1                      # Windows one-shot: venv+tests, pick storage folder, write .env, (opt) docker build/run
├── .gitignore                     # NEW: protects .env, ignores .venv/, _workspace/, __pycache__, caches
├── requirements.txt               # Python deps (PyTorch installed separately, CUDA 12.4)
├── requirements-dev.txt           # Test-only deps (pytest, pyyaml) for the import-light tests
├── toDoList                       # User scratchpad (⚠ contains a plaintext HF_TOKEN — rotate it)
│
├── tests/                         # Import-light pytest suite (no torch) — runs on Windows & in Docker
│   ├── test_distill_seq_utils.py  # teacher-forcing / prompt-mask / truncation logic
│   ├── test_sharding.py           # deterministic, GPU-count-agnostic shard contract
│   └── test_wer_utils.py          # WER accept/reject gate + content-hash (resume/dedup) logic
│
├── scripts/                       # The entire pipeline (no package, flat scripts run by number)
│   │
│   │   ===== CANONICAL MULTI-GPU PIPELINE (current / actively developed) =====
│   ├── 01_multi_gpu_setup.sh              # Pod bootstrap: apt deps, PyTorch 2.5.1+cu124, Accelerate bf16 config, HF login, clone official distil-whisper
│   ├── 02_generate_pseudo_labels_multi_gpu.py  # ★ HEART OF THE PROJECT (~4,200 lines). Teacher inference + WER filter + dedup + chunked download + resume, sharded across GPUs
│   ├── 03_train_distillation_multi_gpu.py # Distillation trainer (Accelerate, frozen encoder, 8 decoder layers [chosen; 2=faithful-v3.5], 0.8·CE+1.0·KL)
│   ├── 04_convert_to_ctranslate2.py       # HF checkpoint → CTranslate2 (float16) + model card + synthetic-audio verify
│   ├── run_full_pipeline.sh               # Orchestrates setup → pseudo-labels → train → convert (4×H100), --skip-setup/--resume
│   │
│   │   ===== SUPPORTING UTILITIES (used by the multi-GPU pipeline) =====
│   ├── dedup_utils.py                     # GlobalDeduplicator (text + perceptual audio fingerprint), StorageManager (disk watchdog)
│   ├── distill_seq_utils.py               # Torch-free teacher-forcing/prompt-mask helpers used by stage 3 (unit-tested in tests/)
│   ├── wer_utils.py                       # Torch-free WER-gate + content-hash primitives used by stage 2 (unit-tested in tests/)
│   ├── monitor_progress.py                # Live `rich` TUI: per-GPU stats + per-dataset progress/acceptance/WER/ETA
│   ├── 04_reprocess_pseudo_labels.py      # CPU multiprocessing re-scorer: recompute WER with improved logic, reclassify accepted⇄rejected
│   ├── 05_check_duplicates.py             # Post-hoc duplicate scan/removal on *_gpu*_accepted/rejected.jsonl (--dry-run/--fix)
│   ├── remove_ami_entries.py              # One-off: strip dataset=="ami" rows from all_pseudo_labels.jsonl (timestamped backup)
│   ├── 05_convert_to_ctranslate2.py       # Alt converter: + real-audio benchmark, YAML model card, copies config.json
│   ├── 05_test_faster_whisper.py          # 5-test faster-whisper QA suite (load/transcribe/word-ts/memory<4GB/speed RTF)
│   ├── 06_eval_open_asr.py                # WER eval over Open-ASR-Leaderboard test splits (student[/teacher] vs CrisperWhisper published) — parity check
│   │
│   │   ===== LEGACY SINGLE-GPU PIPELINE (earlier, superseded; kept for reference) =====
│   ├── 01_cloud_setup.sh                  # Single-GPU pod bootstrap (CUDA_VISIBLE_DEVICES=0)
│   ├── 02_prepare_data.py                 # Download+resample 3 datasets → manifest.jsonl + WAVs
│   ├── 02_prepare_data_streaming.py       # Earlier all-in-one streaming pseudo-labeller (8 datasets)
│   ├── 03_generate_pseudo_labels.py       # Single-GPU pseudo-label generator (threaded audio load)
│   ├── 04_train_distillation.py           # Single-GPU trainer (same loss; SIGUSR1; stores config in state)
│   ├── 05_convert_to_ctranslate2.py       # (shared with multi-GPU — see above)
│   ├── run_training.sh                    # Step-selectable single-GPU orchestrator (--step setup|data|labels|train|convert)
│   └── run_official_distilwhisper.sh      # Alternative: run HF's official create_student/run_pseudo_labelling/run_distillation/run_eval scripts verbatim
│
├── Server logs/                   # Captured pod run logs (pseudo_labels_*.log) — untracked
├── Backupserverdata/              # Backed-up pseudo_labels output (pseudo_labels.zip, "pseudo_labels (old)/") — untracked
├── .idea/                         # PyCharm project files — untracked
└── .claude/                       # Claude Code session data — untracked
```

**A root `.gitignore` now exists** (added this session): it covers `.env`, `.venv/`, `_workspace/`, `__pycache__/`, `.pytest_cache/`, `.idea/`, `.claude/`, `Server logs/`, `Backupserverdata/`, and `*.zip`. `toDoList` is not ignored (still shows untracked). Never commit `Backupserverdata/pseudo_labels.zip` (~260 MB) or any real `.env`. A Windows helper `start.ps1` creates the `.venv`, runs the import-light tests, prompts for the storage folder (`DATA_DIR`), and writes `.env`.

### The "two pipelines" gotcha (read before touching any numbered script)

The `scripts/` directory contains **two overlapping numbered pipelines plus one official-repo wrapper**, and the numbers collide. Do not assume a number means a stage:

| Stage | **Canonical (multi-GPU, current)** | Legacy (single-GPU) | Official-repo wrapper |
|-------|------------------------------------|---------------------|-----------------------|
| Setup | `01_multi_gpu_setup.sh` | `01_cloud_setup.sh` | (uses `01_multi_gpu_setup.sh`) |
| Data/labels | `02_generate_pseudo_labels_multi_gpu.py` | `02_prepare_data.py` → `03_generate_pseudo_labels.py` (or `02_prepare_data_streaming.py`) | `run_pseudo_labelling.py` (HF) |
| Train | `03_train_distillation_multi_gpu.py` | `04_train_distillation.py` | `run_distillation.py` (HF) |
| Convert | `04_convert_to_ctranslate2.py` | `05_convert_to_ctranslate2.py` | `ct2-transformers-converter` |
| Orchestrator | `run_full_pipeline.sh` | `run_training.sh` | `run_official_distilwhisper.sh` |

**The multi-GPU pipeline is the one under active development** (all recent commits touch `02_generate_pseudo_labels_multi_gpu.py`; the live `Server logs/` are from it). When in doubt, the `*_multi_gpu*` scripts + `run_full_pipeline.sh` are authoritative; the legacy scripts and the official-repo wrapper are kept for fallback/reference.

---

## Pipeline Architecture

### End-to-End Data Flow (canonical multi-GPU path)

```
RunPod pod (Linux, N×GPU, /workspace volume)
    │
    ▼  bash 01_multi_gpu_setup.sh
[ deps installed · Accelerate bf16 config · HF login · official repo cloned ]
    │
    ▼  accelerate launch 02_generate_pseudo_labels_multi_gpu.py --config ../config.yaml
8 HF datasets ──► CrisperWhisper teacher (float16, greedy, batched) ──► WER filter (≤10%) ──► dedup/shard
    │                                                                                          │
    ▼                                                                                          ▼
/workspace/pseudo_labels/{name}_gpu{rank}_accepted.jsonl  +  {name}_gpu{rank}_rejected.jsonl
    │   (+ generation_progress.json, {name}_metadata.json, dedup caches)
    ▼  merge (main rank)
/workspace/pseudo_labels/all_pseudo_labels.jsonl
    │
    ▼  accelerate launch 03_train_distillation_multi_gpu.py --config ../config.yaml [--resume]
Student (frozen 32L encoder + 8L decoder)  ◄── distill ──  Teacher (frozen)
loss = 0.8·CE(student, pseudo-label) + 1.0·KL(softmax_T(teacher) ‖ softmax_T(student))
    │   checkpoints every save_steps → /workspace/checkpoints/checkpoint-{step}/ (+ push to HF Hub)
    ▼
/workspace/output/distil-crisperwhisper-final/   (HF format)
    │
    ▼  python3 04_convert_to_ctranslate2.py --config ../config.yaml
/workspace/output/distil-crisperwhisper-ct2/     (model.bin + tokenizer + README.md)
    │
    ▼  python3 05_test_faster_whisper.py --config ../config.yaml
[ load · transcribe · word-timestamps · VRAM<4GB · RTF speed → PASS/FAIL ]
```

### Stage 2 internals — `02_generate_pseudo_labels_multi_gpu.py`

This single file is the most complex and most-edited part of the project. Its components:

| Component | Lines (approx) | Purpose |
|-----------|----------------|---------|
| `DATASET_CONFIGS` | 213 | Hard-coded registry of the 8 datasets (HF name, subset, splits, text/audio columns, est. hours, auth, priority, chunked-download flags). **This — not `config.yaml` — is what the multi-GPU script iterates.** |
| `PseudoLabelEntry` (dataclass) | 368 | One output record: `sample_id, dataset, ground_truth, pseudo_label, word_timestamps, wer, duration_seconds, audio_path, accepted, rejection_reason`. |
| `DatasetProgress` (dataclass) | 383 | Per-dataset resume/stats state (processed/accepted/rejected counts, hours, `last_sample_idx`, `status`, `verification_status`, `processed_sample_ids`). |
| `generate_content_hash()` | 424 | Deterministic `{dataset}_{md5_16hex}` ID from first 4000 audio samples + text + dataset name. **Do not change the 16-char length** without a migration plan (existing IDs depend on it). |
| `ThreadedPrefetcher` | 461 | CPU-side producer: 1 feeder thread iterates the dataset → raw queue → N worker threads (extract audio, hash, duration-filter, **dedup by text**, **shard by text**) → bounded processed queue feeding GPU batches. Keeps the GPU saturated. |
| `SpotInstanceHandler` | 874 | SIGTERM/SIGINT/SIGUSR1 → set `should_stop` + run save callback for graceful preemption. |
| `CrisperWhisperTeacher` | 895 | Teacher wrapper. `generate_pseudo_labels_batch()` is the hot path: greedy (`num_beams=1`), `max_new_tokens=256`, timestamps off for speed. `generate_pseudo_label()` (single) can emit word timestamps via `decode_with_timestamps` + regex parse. |
| `MultiGPUPseudoLabelGenerator` | 1151 | The orchestrator (~2,900 lines): config, distributed setup, batch-size auto-tuning, dataset loading (streaming + chunked), the main GPU loop, WER scoring, resume/verification, per-GPU output, and the final merge. |

**Threading model (per GPU process):**

| Thread(s) | Role |
|-----------|------|
| Main | Pulls preprocessed batches, runs `teacher.generate_pseudo_labels_batch()`, scores WER, writes JSONL |
| 1 × feeder | Iterates the HF dataset iterator, enqueues raw samples |
| N × prefetch workers | `cpu_count()//4`, clamped 8–64; extract audio, content-hash, duration-filter, dedup-by-text, shard-by-text |

**Multi-GPU sharding is deterministic by ground-truth text:** `int(md5(text)[:8],16) % world_size == local_rank`. The same sample always lands on the same rank regardless of GPU count, so resume is robust even if the pod restarts with a different number of GPUs. Each rank writes its own `{name}_gpu{rank}_*.jsonl`; the main rank merges them at the end.

### Stage 3 internals — `03_train_distillation_multi_gpu.py`

| Component | Purpose |
|-----------|---------|
| `create_student_model()` | Copies the **full 32-layer encoder** from the teacher, then selects decoder layers via `np.linspace(0, 31, n).astype(int)` (n=2 → [0,31]; **n=8 (chosen) → [0,4,9,13,18,22,27,31]**); copies embeddings & `proj_out`. |
| `freeze_encoder()` | Sets `requires_grad=False` on **all encoder params** + decoder `embed_positions`. Must be re-applied after every checkpoint resume. |
| `DistillationLoss(nn.Module)` | **Official formula** `0.8·CE + 1.0·KL` (weights do **not** sum to 1). KL uses temperature `T=2.0`, scaled by `T²`; padding masked with `ignore_index=-100`. Student sees SpecAugment-masked features; teacher sees originals (no_grad). |
| `SpecAugment(nn.Module)` | freq mask 27 / time mask 100, 2 masks each, p=0.5; applied to **student input only**. |
| `DistillationDataset` / `DistillationCollator` | Loads pseudo-label JSONL, **loads REAL audio from `audio_path`** (drops samples with no on-disk audio — no zero-silence fallback), pads to 480k samples (30 s @ 16 kHz) → `input_features`. Per item returns `prompt_ids` + `label_ids` (with `timestamp_probability` choosing timestamped labels and `condition_on_prev_probability` prepending `<|startofprev|>` context). The collator builds padded `decoder_input_ids` + masked `labels` via `build_decoder_inputs_and_labels`/`pad_decoder_batch` (`distill_seq_utils.py`, unit-tested). |
| `build_decoder_inputs_and_labels` (`distill_seq_utils.py`) | Torch-free teacher-forcing builder: `full = prompt+label`, `decoder_input_ids=full[:-1]`, `labels=full[1:]` with the prompt region masked to -100. **Unit-tested** (the trainer can't be GPU-run locally). |
| `DistillationTrainer` | Accelerate setup, training loop (explicit `decoder_input_ids` to BOTH teacher & student so KL aligns), periodic held-out **eval loop**, checkpoint save/**resume-into-prepared-model** (incl. `checkpoint-emergency`), optional BPE dropout, HF Hub push, emergency save on preemption. `--max-steps` overrides the horizon for a quick PoC. |

---

## Configuration

### `config.yaml` (single source of truth for hyperparameters)

All tunable settings live in [config.yaml](config.yaml). Key sections:

| Section | Key settings | Notes / defaults |
|---------|-------------|------------------|
| `paths` | `workspace`, `data_dir`, `checkpoint_dir`, `output_dir`, `hf_cache`, `pseudo_labels_dir` | All under `/workspace` (the persistent RunPod volume) |
| `teacher` | `model_id` (`nyrahealth/CrisperWhisper`), `dtype` (`float16`), `pseudo_label_batch_size` (64), `attn_implementation` (`sdpa`) | Batch size is a **ceiling** — auto-reduced per GPU VRAM |
| `student` | `base_model` (vestigial — student built from teacher), `decoder_layers` (**8** chosen for WER parity; 2 = 6× speed/faithful-v3.5, 16 = max quality), `encoder_layers` (32, frozen), `dtype` (`bfloat16`) | |
| `storage` | `min_free_space_gb`, `emergency_free_space_gb`, `chunk_size_gb` (overrides DATASET_CONFIGS chunk size) | cloud 50/20; `config.local.yaml` 200/100 + `chunk_size_gb: 100` for a 10 TiB disk (bigger chunks ⇒ fewer HF requests) |
| `datasets` | per-dataset `enabled / name / subset / split / streaming / hours / text_column` | Consumed mainly by the **legacy** path; the multi-GPU path reads `enabled` to filter, but dataset details come from `DATASET_CONFIGS` in the script |
| `training` | `max_steps` (80000), `per_device_train_batch_size` (48), `gradient_accumulation_steps` (8), `learning_rate` (1e-4), `lr_scheduler_type` (`linear`), `warmup_steps` (500), `weight_decay` (**0.0** — v3.5 uses no weight decay), `max_grad_norm` (1.0), `bf16` (true), `save_steps` (2500), `save_total_limit` (5), `eval_steps` (5000), `eval_samples` (held out for the periodic eval-loss), `gradient_checkpointing` (true), dataloader workers/prefetch | Effective batch = per_device × grad_accum × num_gpus |
| `distillation` | `temperature` (2.0), `ce_weight` (0.8), `kl_weight` (1.0), `timestamp_probability` (0.2), `condition_on_prev_probability` (0.2), `bpe_dropout` (0.1), `pseudo_labels.wer_threshold` (0.10), `pseudo_labels.save_audio` (true), `pseudo_labels.audio_format` (flac), `pseudo_labels.generate_timestamps` (true), `chunk_duration_seconds` (30), `max_label_length` (256) | `timestamp_probability` / `condition_on_prev_probability` / `bpe_dropout` are now **IMPLEMENTED** in stage 3 (`DistillationDataset.__getitem__` + the collator + the explicit-`decoder_input_ids` train step). `save_audio`/`generate_timestamps` drive stage-2 audio + timestamped-label persistence. |
| `spec_augment` | `enabled`, `freq_mask_param` (27), `n_freq_masks` (2), `time_mask_param` (100), `n_time_masks` (2), `probability` (0.5) | |
| `huggingface` | `username` (`${HF_USERNAME}`), `repo_name` (`distil-crisperwhisper-large-v3`), `push_to_hub` (true), `push_every_n_steps` (2500), `private` (true) | |
| `spot_instance` | `enabled`, `check_interval_seconds` (30), `save_on_preemption`, `auto_resume` | |
| `logging` | `use_wandb`, `wandb_project` (`distil-crisperwhisper`), `use_tensorboard`, `tensorboard_dir` | |
| `multi_gpu` | `use_accelerate`, `mixed_precision` (`bf16`), `deepspeed_config` (null), `min_gpus` (4), `max_gpus` (0 = all) | |
| `conversion` | `quantization` (`float16`), `output_format` (`ctranslate2`) | |

### Environment Variables (set on the pod)

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_TOKEN` | **Yes** | HuggingFace auth — avoids 429 rate-limiting when many GPUs hit the Hub; needed for gated datasets |
| `HF_USERNAME` | For `push_to_hub` | Target Hub namespace |
| `WANDB_API_KEY` | Optional | Weights & Biases logging |
| `HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, `HUGGINGFACE_HUB_CACHE` | Set by scripts | Pin all HF caches under `/workspace/hf_cache` so downloads don't fill the small container root |
| `HF_HUB_ENABLE_HF_TRANSFER` | Auto | `1` only if `hf_transfer` is installed, else `0` (set at the top of the multi-GPU script before importing `datasets`) |
| `HF_HUB_DOWNLOAD_TIMEOUT` | Auto (1800) | 30-min per-file download timeout |
| `NCCL_TIMEOUT` / `TORCH_NCCL_BLOCKING_WAIT` | Auto (1800 / 1) | Prevents the NCCL watchdog from killing ranks during slow dataset downloads |

### The 8 datasets (Distil-Whisper v3.5, ~196,000 raw hours)

Defined in `DATASET_CONFIGS` (multi-GPU script) and `config.yaml`. Listed by processing **priority** (lower = first):

| Priority | Dataset | HF name | Subset | Auth | ~Hours | Download mode |
|----------|---------|---------|--------|------|--------|---------------|
| 1 | LibriSpeech | `librispeech_asr` | — (clean.100+clean.360+other.500) | No | 960 | full |
| 2 | AMI (IHM) | `edinburghcstr/ami` | `ihm` | No | 100 | full · verbatim/fillers |
| 4 | VoxPopuli | `facebook/voxpopuli` | `en` | No | 1,800 | full |
| 5 | Common Voice 17 | `mozilla-foundation/common_voice_17_0` | `en` | **Yes** | 3,000 | full |
| 6 | TED-LIUM | `distil-whisper/tedlium` | `release3` | No | 450 | full |
| 7 | GigaSpeech | `speechcolab/gigaspeech` | `xl` | **Yes** | 10,000 | **chunked** (20 GB) |
| 8 | People's Speech | `MLCommons/peoples_speech` | `clean` | **Yes** | 30,000 | **chunked** (20 GB) |
| 9 | YODAS | `espnet/yodas` | `en000` | No | 150,000 | **chunked** (20 GB) |

After WER filtering, roughly **~50% are accepted** (~98,000 hours in the full run). Gated datasets require accepting their terms on the HF website with the same account as `HF_TOKEN`.

✅ **`podcast_fillers` reconciled:** previously `config.yaml` listed `podcast_fillers` (`ylacombe/podcast_fillers_by_license`) while `DATASET_CONFIGS` had removed it — now **both** drop it (only 199 episodes, no transcriptions, non-commercial annotations). If you re-enable filler training, do it via AMI (verbatim fillers) — don't resurrect `podcast_fillers`. Keep `config.yaml`/`config.local.yaml` and `DATASET_CONFIGS` reconciled going forward.

---

## Running the Pipeline (on the RunPod pod)

> All commands run **on the pod**, from `/workspace/distil_crisper_training/scripts`, after setting `HF_TOKEN` / `HF_USERNAME`. Use `tmux` so a dropped SSH session doesn't kill the job.

### Full automated run (multi-GPU)
```bash
cd /workspace/distil_crisper_training/scripts
chmod +x *.sh
bash run_full_pipeline.sh                 # setup → pseudo-labels → train → convert
bash run_full_pipeline.sh --skip-setup    # skip env bootstrap if already done
bash run_full_pipeline.sh --resume        # resume training/labels from last checkpoint
```

### Step-by-step (multi-GPU)
```bash
bash 01_multi_gpu_setup.sh                                            # one-time env bootstrap
accelerate launch 02_generate_pseudo_labels_multi_gpu.py --config ../config.yaml
accelerate launch 03_train_distillation_multi_gpu.py   --config ../config.yaml [--resume]
python3 04_convert_to_ctranslate2.py --config ../config.yaml [--force]
python3 05_test_faster_whisper.py    --config ../config.yaml
```

### Useful flags — `02_generate_pseudo_labels_multi_gpu.py`
```bash
--config ../config.yaml         # config path (falls back to repo-root config.yaml)
--datasets librispeech ami      # process only these datasets
--max-samples 1000              # cap per dataset (smoke test)
--no-resume                     # start fresh (ignore progress/output files)
--merge-only                    # just merge existing *_accepted.jsonl → all_pseudo_labels.jsonl
--list-datasets                 # print the dataset table and exit
```

### Monitoring (separate terminal/tmux pane, on the pod)
```bash
python3 monitor_progress.py --dir /workspace/pseudo_labels --refresh 1.0   # live rich TUI
tail -f /workspace/logs/pseudo_labels_*.log                                # raw log
tensorboard --logdir /workspace/tensorboard --port 6006                    # training curves
```

### Post-hoc data tools
```bash
python3 04_reprocess_pseudo_labels.py -i /workspace/pseudo_labels [--dry-run] [--replace-originals]
python3 05_check_duplicates.py --dir /workspace/pseudo_labels [--details] [--fix --dry-run]
python3 remove_ami_entries.py    # strips dataset=="ami" from all_pseudo_labels.jsonl (makes a backup)
```

### Legacy single-GPU path (fallback)
```bash
bash run_training.sh                       # full single-GPU run
bash run_training.sh --step train --resume # one stage only
bash run_official_distilwhisper.sh         # use HuggingFace's own distil-whisper scripts
```

---

## Data Formats & On-Disk Layout (`/workspace`)

```
/workspace/
├── hf_cache/                       # HF_HOME — datasets/transformers/hub caches (cleaned under disk pressure)
├── data/                           # legacy path: downloaded WAVs + manifest.jsonl per dataset
├── pseudo_labels/
│   ├── audio/{name}/{sample_id}.flac     # persisted 16kHz mono audio for ACCEPTED samples (save_audio) — stage-3 training input
│   ├── {name}_gpu{rank}_accepted.jsonl   # per-GPU accepted records (PseudoLabelEntry, JSONL)
│   ├── {name}_gpu{rank}_rejected.jsonl   # per-GPU rejected records (WER > threshold etc.)
│   ├── all_pseudo_labels.jsonl           # merged accepted records (training input)
│   ├── generation_progress.json          # resume state: {world_size, datasets:{name:DatasetProgress}}
│   ├── {name}_metadata.json              # per-dataset counts/verification (consumed by monitor)
│   ├── global_processed_texts.json       # dedup cache: normalized ground-truth texts seen
│   ├── global_audio_fingerprints.json    # dedup cache: fingerprint → "dataset:sample_id"
│   └── global_dedup_stats.json           # dedup counters
├── checkpoints/
│   ├── checkpoint-{step}/                # model + processor + training_state.pt; pruned to save_total_limit
│   └── checkpoint-emergency/             # written on SIGTERM (NOT pushed to Hub, NOT auto-pruned)
├── output/
│   ├── distil-crisperwhisper-final/      # final HF-format student
│   └── distil-crisperwhisper-ct2/        # CTranslate2 model: model.bin + tokenizer files + README.md
├── logs/                            # pseudo_labels_<ts>.log, training.log, etc.
└── tensorboard/                     # TB event files
```

**`PseudoLabelEntry` JSONL record** (one per line in `*_accepted.jsonl` / `*_rejected.jsonl`):
```json
{
  "sample_id": "librispeech_a1b2c3d4e5f6a7b8",
  "dataset": "librispeech",
  "ground_truth": "the original reference transcript",
  "pseudo_label": "crisperwhisper's generated transcript",
  "word_timestamps": [],
  "wer": 0.041,
  "duration_seconds": 5.3,
  "audio_path": "/workspace/pseudo_labels/audio/librispeech/librispeech_a1b2c3d4e5f6a7b8.flac",
  "accepted": true,
  "rejection_reason": null,
  "pseudo_label_timestamped": "<|0.00|> crisperwhisper's generated transcript<|5.20|>",
  "prev_text": "the previous accepted sample's transcript"
}
```
> Notes:
> - **`audio_path` is now populated for accepted samples** when `save_audio: true` (default): stage 2 writes 16 kHz mono FLAC under `pseudo_labels/audio/{dataset}/{sample_id}.flac`. This is what makes stage 3 trainable — earlier it was always `null` and the trainer fell back to silence. Rejected samples keep `audio_path: null` (audio not saved).
> - **`pseudo_label_timestamped`** holds the transcription WITH Whisper segment `<|x.xx|>` tokens (only when `generate_timestamps: true`); **`prev_text`** holds the previous accepted sample's text. Both feed the trainer's faithful `timestamp_probability` / `condition_on_prev_probability`. `word_timestamps` stays empty (word-level timestamps are inherited via the frozen copied encoder, not stored per-label).
> - The legacy single-GPU script uses field name `pseudo_transcription` instead of `pseudo_label` — watch for this when sharing JSONL between paths.

**`training_state.pt`** (in each checkpoint): `global_step`, `best_loss`, `epoch`, optimizer & scheduler state. Multi-GPU uses keys `optimizer`/`scheduler`; the legacy single-GPU trainer uses `optimizer_state_dict`/`scheduler_state_dict` and also stores the full `config` — the two are **not** checkpoint-compatible.

---

## Important Implementation Details

### WER quality gate (the accept/reject decision)
- Threshold: `distillation.pseudo_labels.wer_threshold` (0.10). `accepted = wer <= threshold`.
- Normalization (`_normalize_text`): strips bracketed fillers (`[Um]`, `[Uh]`, …) → Whisper `EnglishTextNormalizer` (lowercase, punctuation, numbers→words, contractions, unicode) → strips standalone filler words (`um, uh, er, ah, uhm, erm, hmm, hm, mm, mhm, uh huh, mm hmm`). This is so CrisperWhisper's verbatim filler output isn't penalized against filler-free ground truth.
- WER algorithm (`_calculate_wer_spelling_tolerant`): Levenshtein DP over word lists, but British/American spelling variants count as a **match** (zero cost) when they share a first letter and `difflib.SequenceMatcher` ratio ≥ 0.85 (e.g. colour/color, realise/realize).
- All-caps hypotheses are rejected outright (`wer = 1.0`) — a known Whisper hallucination signature. Empty ref or empty hyp ⇒ `1.0`.

### Deduplication (three layers)
1. **Resume dedup** — `already_processed_ids` is a set of **normalized ground-truth texts** loaded from existing output files; prefetch workers skip any text already present. (Text, not audio hash, because audio bytes vary across runs.)
2. **Within-batch dedup** — `GlobalDeduplicator.check_batch_duplicates()` removes repeated texts inside a single GPU batch before inference.
3. **Cross-dataset dedup** (`dedup_utils.GlobalDeduplicator`) — exact normalized-text match **plus** a 32-char perceptual **audio fingerprint** (64-bit energy/ZCR hash over 32 frames + md5 of first 8000 samples); near-duplicates flagged by Hamming distance ≤ 10. Thread-safe; persisted to the `global_*` JSON caches.

### Resume & verification (GPU-count agnostic)
- Completed datasets are skipped on resume **unless** their `verification_status == "incomplete"` (unaccounted samples), in which case they're reprocessed.
- `verification_status` values: `verified`, `missing_samples`/`incomplete`, `exceeded` (iteration yielded more than the metadata estimate — normal for some streaming/concatenated datasets).
- `generate_content_hash()` and the text-based shard/dedup keys make IDs stable across runs and GPU counts, so a pod can restart with a different GPU count and pick up cleanly.

### Chunked download (disk-pressure management)
- GigaSpeech / People's Speech / YODAS set `use_chunked_download: True` with `chunk_size_gb: 20`. The script downloads ~20 GB at a time, processes it, marks the chunk done, and **cleans the HF cache** before the next chunk — this is what lets a multi-TB corpus run on a ~2 TB pod disk (see commit *"optimize for 2TB disk limit"*).
- `StorageManager` (in `dedup_utils.py`) watches free space: normal cleanup below `storage.min_free_space_gb` free, **emergency cleanup** (delete the entire `datasets/` cache + `.incomplete` files) below `storage.emergency_free_space_gb`. These are now **config-driven** (defaults 50/20 in `config.yaml`; **200/100 in `config.local.yaml` for the 10 TiB disk**). `config.local.yaml` also sets `storage.chunk_size_gb: 100`, which **overrides** the per-dataset 20 GB `chunk_size_gb` in `DATASET_CONFIGS` (bigger rolling chunks ⇒ far fewer HuggingFace requests ⇒ lower rate-limit risk).
- On **incomplete chunk verification**, stage 2 now records the chunk index under `incomplete_chunks` in `{name}_chunk_progress.json` (loud warning) before advancing + cleaning cache — so a genuine gap is auditable / re-processable (`--no-resume`) instead of silently lost, while still avoiding an infinite re-download loop on a small disk.

### Distributed coordination & NCCL
- NCCL timeout is raised to 1800 s and `TORCH_NCCL_BLOCKING_WAIT=1` so the watchdog doesn't kill ranks during long downloads. The most recent commit (*"replace barriers with file-based signaling"*) moved cross-rank coordination toward `{name}_gpu{id}_done.marker` files instead of `dist.barrier()` to avoid timeout deadlocks — prefer the file-marker pattern when adding new cross-rank sync points.
- Batch size is auto-tuned per GPU VRAM (`_calculate_optimal_batch_size`): ≥90 GB→64, ≥75→48, ≥35→24, ≥20→12, ≥10→6, else 4 (capped by `pseudo_label_batch_size`).

### Distillation specifics
- Student decoder layers chosen by `np.linspace(0, 31, n)` (maximally spaced); n=8 (chosen) = [0,4,9,13,18,22,27,31]; n=2 = [0, 31].
- Encoder + decoder positional embeddings frozen — preserves the teacher's "ears" and keeps speculative-decoding compatibility. **Re-freeze after every checkpoint load.**
- Loss is `0.8·CE + 1.0·KL` (do **not** "normalize" the weights to sum to 1 — that's a common and wrong "fix"). KL is temperature-scaled by `T²`.
- SpecAugment is applied to the **student's** input features only; the teacher always sees clean features under `no_grad`.

### Hardware drift (be skeptical of any single hardware claim)
The repo was retargeted over time: `config.yaml` headers say **7× H200**, the `02_*` script docstring says **4× H100 NVL (376 GB)**, [RUNPOD_SETUP.md](RUNPOD_SETUP.md) says **4× H100 NVL**, [GUIDE.md](GUIDE.md) says **A100 80/40 GB**. None is canonical — the scripts **auto-detect** GPU count and per-GPU VRAM and scale. Trust `torch.cuda.device_count()` / `get_device_properties()` at runtime, not the prose. `multi_gpu.min_gpus` is 4; `max_gpus: 0` means use all.

---

## Code Conventions (match the existing style)

This is a research/training codebase, not a packaged application — its conventions differ from a typical app, and you must **match what's already here** rather than impose an app-style standard:

- **Flat scripts, not a package.** No `Modules/`, no `__init__.py`, no installable package. Scripts are run directly (`python3 NN_*.py` / `accelerate launch …`). Cross-script reuse is by plain import of sibling files (e.g. `from dedup_utils import GlobalDeduplicator`) — they rely on being run from `scripts/`.
- **Heavy inline comments are the norm.** Unlike some sibling projects, this code uses extensive `#` comments and banner blocks (`# ====`) to explain intent, cite the distil-whisper paper, and record *why* a value was chosen. **Keep that style** — explanatory comments here are a feature, not a smell. Do not strip them.
- **Dataclasses** for structured records (`PseudoLabelEntry`, `DatasetProgress`, `ReprocessStats`, …) with docstrings.
- **Dual logging**: every script logs to both the `rich` console and a timestamped file via the `log()` helper / `console.print` wrapper. New diagnostics should go through these, with bracketed tags like `[FEEDER]`, `[WORKER ...]`, `[GPU_BATCH]`, `[PROCESS_DATASET]` for grep-ability.
- **Multiprocessing-safe helpers** in `04_reprocess_pseudo_labels.py` are module-level functions (not methods) because `ProcessPoolExecutor` must pickle them — keep new workers module-level.
- **`config.yaml` is the single source of truth for hyperparameters**; do not hard-code values that belong there. (Dataset *registry* details, however, live in `DATASET_CONFIGS` in the multi-GPU script — see the discrepancy note above.)
- **Resilience over strictness in the data loop**: malformed samples/lines are skipped (and counted), not fatal. Preserve this — one bad sample in millions must never crash a 40-hour run.

---

## Testing & Verification

There is now an **import-light `tests/` suite** (pure Python, no torch/transformers) covering the core decision logic that can't be exercised by a GPU run — the **WER accept/reject gate + content-hash** (`wer_utils.py`), the teacher-forcing/prompt-masking builder (`distill_seq_utils.py`), and the shard contract. Stage 2/3 **delegate** to those torch-free modules so the tests cover the real code paths. Run locally on Windows (`start.ps1` creates the venv) or in the container: `python -m pytest tests/ -q`. Everything else is still verified empirically on a GPU (pod or local Docker):

1. **Smoke a stage cheaply** before a full run:
   ```bash
   accelerate launch 02_generate_pseudo_labels_multi_gpu.py --config ../config.yaml --datasets librispeech --max-samples 200
   ```
   Confirm `librispeech_gpu0_accepted.jsonl` fills, acceptance rate looks sane (~50–80% on clean speech), and `monitor_progress.py` shows progress.
2. **Resume correctness** — kill the job (Ctrl-C/SIGTERM), restart the same command, and confirm it **skips already-processed samples** (logs `Found N already processed samples`) rather than redoing or duplicating them.
3. **Converted-model QA** — `05_test_faster_whisper.py` runs five checks and exits non-zero on failure: model loads, basic transcription, **word-level timestamps populate** (CrisperWhisper's key feature), peak VRAM **< 4 GB**, and speed (GPU target RTF < 0.1). It uses synthetic audio, so empty transcription / unreliable language detection on the sine-wave inputs are expected and treated as pass.
4. **Data hygiene** — after generation, optionally run `05_check_duplicates.py --dir … --fix --dry-run` and `04_reprocess_pseudo_labels.py -i … --dry-run` to preview duplicate removal / WER re-scoring before committing changes.
5. **WER parity** — `06_eval_open_asr.py --config … [--eval-teacher] [--max-samples N]` runs the student (and optionally the teacher) over the Open-ASR-Leaderboard test splits and prints per-dataset WER vs CrisperWhisper's published numbers. Gated/uncertain-id sets are skipped (logged) if unavailable. Some gap on cleaned-reference sets (Earnings22/GigaSpeech) is expected (verbatim-model penalty).

When you change code you cannot run locally, **say so explicitly** and describe the exact pod command the user should run to verify, plus what a passing result looks like.

---

## Known Gotchas (quick reference)

- **Two colliding numbered pipelines** + one official-repo wrapper. The `*_multi_gpu*` scripts are canonical; numbers do not denote stages across pipelines. See the table above.
- **`config.yaml` ↔ `DATASET_CONFIGS` can disagree** (e.g. `podcast_fillers`). The multi-GPU script's `DATASET_CONFIGS` wins for that path.
- **JSONL field name differs** between paths: `pseudo_label` (multi-GPU) vs `pseudo_transcription` (legacy). The multi-GPU record now also carries `pseudo_label_timestamped` and `prev_text` (extra keys; old readers ignore them).
- **Stage 3 needs persisted audio.** Training loads real audio from `audio_path`; run stage 2 with `save_audio: true` (default) or the trainer drops audio-less samples (and errors if NONE have audio). Don't disable `save_audio` if you intend to train from that data.
- **`training_state.pt` keys differ** between trainers — checkpoints are not cross-compatible.
- **A root `.gitignore` now exists** (protects `.env`, ignores `_workspace/`, `__pycache__/`, caches, `*.zip`). Still never commit `Backupserverdata/pseudo_labels.zip` or any real `.env`.
- **Plaintext `HF_TOKEN` in `toDoList`** — rotate it; always use the `HF_TOKEN` env var instead (locally via `.env`, never committed).
- **Don't change `generate_content_hash` length (16)** without a migration plan.
- **Re-freeze the encoder after every checkpoint resume** in the trainer (still done; resume now loads weights INTO the prepared model rather than replacing it).
- **Keep the distillation loss weights at 0.8 CE / 1.0 KL** (not summing to 1) — **verified against `run_distillation.py`** (`loss = 0.8 * ce_loss + kl_weight * kl_loss`, `kl_weight` default 1.0). Do not "normalize" them.
- **`weight_decay` is 0.0** (v3.5 uses no weight decay) — was 0.01.
- **Single-GPU (4090):** prefer plain `python3 <script>` (no `accelerate launch` needed); `dist.barrier()` calls are guarded by `world_size > 1`.
- **Prefer file-marker signaling over `dist.barrier()`** for new cross-rank sync (NCCL timeout safety).
- **Code can't run on native Windows** — verify on the pod or in the local Docker/WSL2 container, and never claim a GPU result you didn't observe. Import-light `tests/` DO run on Windows.
