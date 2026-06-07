# Local single-RTX-4090 setup (Docker on WSL2)

Run pseudo-label **data preparation** (and a small **proof-of-concept training**)
for Distil-CrisperWhisper on one local NVIDIA RTX 4090 (24 GB), inside a Docker
container on WSL2. This keeps the code on Linux/CUDA (it does **not** run on native
Windows) while using your local GPU instead of a cloud pod.

The cloud/multi-GPU path is unchanged — see [RUNPOD_SETUP.md](RUNPOD_SETUP.md) and
[config.yaml](config.yaml). This guide uses [config.local.yaml](config.local.yaml).

---

## Quick start (Windows)
From a PowerShell window in this folder:
```powershell
.\start.ps1
```
It (1) creates a Python venv and runs the import-light tests, (2) writes `.env`
(HF creds), and (3) prints the Docker next steps. Add `-Build -Run` to also build
and drop into the container. If PowerShell blocks the script:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

> **Storage changed:** outputs now live on a **TrueNAS SMB share** (`/workspace`)
> and the HF cache on a **local docker volume** (`/hf_cache`) — see "Storage
> architecture" below. The old `DATA_DIR` folder-picker that `start.ps1` still writes
> is no longer used by docker-compose; set the `SMB_*` vars in `.env` instead
> (updating `start.ps1` to prompt for them is a TODO).

The rest of this guide is the manual / step-by-step version of the same thing.

---

## Storage architecture

Two locations, each matched to its access pattern:

| What | Where | Why |
|------|-------|-----|
| **Outputs** — pseudo-labels (+ FLAC audio), checkpoints, final/CT2 models | **TrueNAS SMB share** at `/workspace` (10 TB, multi-device, Windows-readable) | what you keep; sequential single-writer files CIFS handles fine |
| **HF cache** — teacher download + dataset Arrow cache | **Local docker volume** at `/hf_cache` (ext4, fast) | transient scratch with a network-hostile access pattern; cycles per dataset |

Why not put the cache on SMB too? `datasets` memory-maps Arrow files and the hub
cache uses symlinks — both fight CIFS. Keeping the cache on local ext4 avoids that;
the cache only needs to hold **one dataset (or chunk) at a time** because it is
**auto-flushed after each dataset** (`storage.flush_dataset_cache_after_completion:
true` in `config.local.yaml`), so a ~160 GB local disk cycles through any number of
datasets while the kept audio accumulates on the 10 TB share.

**TrueNAS prerequisite (one-time):** create a ZFS **dataset** + an **SMB share** for
it (delete the old iSCSI zvol first to free the space — iSCSI block devices can't be
bind-mounted into WSL2/Docker, which is the whole reason for using SMB). Then set in
`.env`: `SMB_HOST`, `SMB_SHARE`, `SMB_USER`, `SMB_PASS`. The container mounts the
share via CIFS; if it can't mount, the container **fails fast** (safer than silently
writing to the ephemeral overlay).

> A single **non-streaming** dataset whose peak (download + Arrow) exceeds the local
> cache still won't fit — librispeech-all-splits (~120–180 GB) is the one at risk on
> a ~160 GB cache. Use a smaller set for the smoke test (`--datasets ami`, ~15 GB);
> for librispeech either set only its `streaming: true` or process one split.

---

## 1. Prerequisites (on Windows + WSL2)

1. **WSL2 + Ubuntu** installed (`wsl --install`), and an up-to-date **NVIDIA driver
   on Windows** (the WSL2 CUDA driver is included — do *not* install a driver inside
   WSL).
2. **Docker** with the WSL2 backend and **NVIDIA Container Toolkit** (Docker Desktop
   ≥ 4.x with WSL integration already includes GPU support; or install
   `nvidia-container-toolkit` in your WSL distro).
3. Verify GPU passthrough works:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```
   You should see your RTX 4090 listed.

Work from the repo checked out **inside WSL2** (e.g. `~/distil_crisper_training`).

---

## 2. Configure

```bash
cp .env.example .env
# Edit .env:
#   HF_TOKEN=...        (required; gated datasets + avoids rate limiting)
#   HF_USERNAME=...     (only if you later enable push_to_hub)
#   SMB_HOST=192.168.1.55  SMB_SHARE=distil  SMB_USER=...  SMB_PASS=...
#                       (TrueNAS SMB share mounted at /workspace — see "Storage architecture")
#   HF_HUB_ENABLE_HF_TRANSFER=0   (legible download errors; set 1 for speed once stable)
```

The HF cache goes on a **local** docker volume automatically (no path to set). The
legacy `DATA_DIR` is no longer used by docker-compose.

### How much disk? (`DATA_DIR`)
The currently-configured English mixture is ~50k raw hours (~2.4 TB of downloads),
but downloads are **chunked and auto-deleted**, so only a rolling chunk is on disk
at any moment. The real driver is **retained accepted audio** (`save_audio: true`):

| Scope | Retained audio (FLAC) | Notes |
|------|------|------|
| Core-5 (LibriSpeech, AMI, TED-LIUM, VoxPopuli, Common Voice) | ~80 GB | days of 4090 time |
| Full configured mixture (~50k h) | ~0.8 TB | weeks–months of 4090 time |

**The 10 TB SMB share holds the retained audio** (~80 GB core-5, ~0.8 TB full
mixture) — ample. The **local HF cache** is the tighter resource: `config.local.yaml`
is tuned for a ~160 GB local cache (`storage.chunk_size_gb: 40` → ~60 GB working set,
`min_free_space_gb: 30 / emergency: 15`) and **auto-flushes each dataset's cache** so
it cycles. A single non-chunked dataset must still fit the local cache at peak (see
the librispeech note under "Storage architecture").

### Gated datasets & avoiding rate limits
**Accept the dataset terms first** (one-time, same HF account as `HF_TOKEN`) or the
gated ones are skipped with a 401/403:
- GigaSpeech — https://huggingface.co/datasets/speechcolab/gigaspeech
- People's Speech — https://huggingface.co/datasets/MLCommons/peoples_speech
- Common Voice 17 — https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
- (eval only) SPGISpeech — https://huggingface.co/datasets/kensho/spgispeech

**Rate limits** are unlikely on a single machine (one process → no fan-out), and are
further reduced by the settings already in place:
- `HF_TOKEN` set (authenticated requests have far higher limits than anonymous).
- `hf_transfer` installed in the image + `HF_HUB_DOWNLOAD_TIMEOUT=1800`.
- **Large chunks** (`storage.chunk_size_gb: 100`) → ~5–60× fewer `load_dataset`
  calls on the giant sets than the cloud's 20 GB chunks.
- Built-in exponential-backoff retry on 429 / timeout (5 attempts), and the
  chunked path downloads with a 2-hour per-attempt timeout.

If you still see 429s, lower `--batch-size`/parallelism is irrelevant (downloads are
serial); instead just re-run — completed chunks/samples are skipped on resume.

---

## 3. Build & enter the container

```bash
docker compose build
docker compose run --rm distil      # drops you into /app/scripts with the GPU
```

Inside the container, confirm the GPU: `nvidia-smi`.

### Docker images & keeping it clean
The Dockerfile is **single-stage** and does **not** `COPY` the repo (code is
bind-mounted), so a build produces just:
- **`distil-crisperwhisper:local`** — the one image you run, and
- **`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`** — the base, pulled once and reused.

With the default BuildKit builder there are **no leftover intermediate `<none>`
images** on a first build (build layers live in the build cache, not as images),
and a `.dockerignore` keeps the build context tiny (no `.venv`/`_workspace`/`.git`).
`docker compose run --rm` removes the container on exit, so no stray containers.

The only residue appears on **re-builds**: re-tagging `distil-crisperwhisper:local`
leaves the previous image dangling (`<none>`). Clean up when you want:
```bash
docker image prune -f        # remove dangling <none> images (safe)
docker builder prune         # reclaim BuildKit cache (optional)
docker compose down          # stop/remove the compose container if left running
```
`docker images` should normally show exactly those two tagged images.

### Troubleshooting: `No space left on device` during a download
If a model/dataset download dies with `No space left on device (os error 28)` even
though `D:` has TBs free, HF's **Xet** downloader was staging the file on the
container **overlay** (Docker's WSL2 VM, which lives on **C:** and is easily full),
not on `/workspace` (= D:). The image now sets `HF_HUB_DISABLE_XET=1` +
`HF_XET_CACHE=/workspace/hf_cache/xet` to force everything onto D — `git pull` and
re-run `docker compose run --rm distil` (compose env applies with no rebuild). To
patch a *running* container without pulling:
```bash
export HF_HUB_DISABLE_XET=1
export HF_XET_CACHE=/workspace/hf_cache/xet
# then re-run your 02_… command
```
If it STILL fails, check both mounts: `df -h /workspace` (the SMB share — should be
~10 TB) and `df -h /hf_cache` (the local cache volume). A tiny `/workspace` (e.g.
137 MB) means the SMB mount didn't take — verify the `SMB_*` vars in `.env` and that
the TrueNAS share is reachable. Note: an **iSCSI** disk cannot be bind-mounted into
WSL2/Docker (WSL2 only auto-mounts physical fixed disks) — that is exactly why
`/workspace` uses an **SMB share** (CIFS) here, not a Windows drive path.

---

## 4. Smoke test (do this first)

All commands run inside the container, from `/app/scripts`, with
`--config ../config.local.yaml`.

```bash
# (a) Tiny pseudo-label run on LibriSpeech (50 samples)
python3 02_generate_pseudo_labels_multi_gpu.py \
    --config ../config.local.yaml --datasets librispeech --max-samples 50

# Expect: /workspace/pseudo_labels/librispeech_gpu0_accepted.jsonl with
#         non-null "audio_path", and matching .flac files under
#         /workspace/pseudo_labels/audio/librispeech/
ls /workspace/pseudo_labels/audio/librispeech/ | head
python3 - <<'PY'
import json
p="/workspace/pseudo_labels/librispeech_gpu0_accepted.jsonl"
e=json.loads(open(p).readline())
print("audio_path:", e["audio_path"])
print("has timestamps:", bool(e.get("pseudo_label_timestamped")))
PY

# (b) Merge to the training file
python3 02_generate_pseudo_labels_multi_gpu.py --config ../config.local.yaml --merge-only

# (c) PoC training (a few steps) on that tiny set
python3 03_train_distillation_multi_gpu.py --config ../config.local.yaml --max-steps 20

# (d) Convert + validate
python3 04_convert_to_ctranslate2.py --config ../config.local.yaml
python3 05_test_faster_whisper.py    --config ../config.local.yaml
```

**Resume check:** re-run command (a); it should log `Found N already processed
samples` and skip them rather than redoing/duplicating work.

---

## 5. Real data-prep runs

Process datasets smallest-first (they're prioritized automatically). On one 4090
the giant sets (GigaSpeech/People's Speech/YODAS) take weeks-to-months — start with
the core sets:

```bash
python3 02_generate_pseudo_labels_multi_gpu.py \
    --config ../config.local.yaml --datasets librispeech ami tedlium voxpopuli common_voice

# Live progress (separate container shell: `docker compose exec distil bash`)
python3 monitor_progress.py --dir /workspace/pseudo_labels --refresh 1.0
```

Use `tmux` (preinstalled) so a closed terminal doesn't kill a long run.

### Measure WER parity (Open ASR Leaderboard)
After training a model, check how close it is to CrisperWhisper on the test sets
it's benchmarked on:
```bash
# Student only (quick, 200 samples/set):
python3 06_eval_open_asr.py --config ../config.local.yaml --max-samples 200
# Full sets + side-by-side teacher baseline:
python3 06_eval_open_asr.py --config ../config.local.yaml --eval-teacher
# A subset:
python3 06_eval_open_asr.py --config ../config.local.yaml --datasets librispeech_clean ami tedlium
```
Prints per-dataset WER vs CrisperWhisper's published numbers. Gated/uncertain-id
sets are skipped (logged) if unavailable — accept their terms (above) to include them.
Expect some gap on cleaned-reference sets (Earnings22/GigaSpeech) — that's the
verbatim-model penalty the teacher also pays.

---

## 6. Notes

- **8 decoder layers** (`config.*.yaml` `student.decoder_layers: 8`) — chosen for
  closer WER parity (~3× speed) over the faithful distil-v3.5 2 layers (~6×). If
  training OOMs on the 4090, drop `per_device_train_batch_size` to 4–6.
- **Word timestamps are vanilla-Whisper quality**, not CrisperWhisper-quality —
  CrisperWhisper's DTW/alignment-head timestamps don't transfer through standard
  distillation (by design choice). WER + verbatim/filler behavior do transfer.

- **Single GPU, no launcher needed.** Plain `python3 <script>` runs one process
  (no `torch.distributed`, no NCCL). `accelerate launch --num_processes 1 ...` also
  works if you prefer.
- **Fidelity flags** (in `config.local.yaml`): `save_audio`, `generate_timestamps`,
  `timestamp_probability: 0.2`, `condition_on_prev_probability: 0.2`, `bpe_dropout`,
  `weight_decay: 0.0` — the faithful distil-large-v3.5 recipe. Turn off
  `generate_timestamps` to speed up stage-2 generation (you then lose timestamped
  training targets).
- **Full-scale training** (4096 batch / 80 epochs) is a cloud multi-GPU job — the
  trainer stays multi-GPU-capable via `config.yaml`. The 4090 is for data prep + PoC.
- **GPU verification only happens here**, not on the Windows host. If a step fails,
  capture the console output; stage 2 also logs to `/app/logs/` (the repo
  bind-mount), not `/workspace`.
