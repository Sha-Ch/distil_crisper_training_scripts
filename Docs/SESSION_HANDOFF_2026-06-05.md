# Session Handoff — Distil-CrisperWhisper (2026-06-05)

> **How to use this document.** This is a **starting-point map** of what was done and
> learned in the 2026-06-05 session, written for the next agent. It is **not**
> authoritative and may drift from the code. **Before acting on anything here, do
> REAL research in the codebase** — read `CLAUDE.md` in full first (it is the canonical
> project context and is kept in sync), then read the actual files you'll touch
> *whole* (the two big scripts are ~4,200 and ~1,000 lines — read them, don't skim).
> Verify every claim below against the current code/git before relying on it.

---

## 0. The non-negotiable workflow rules (from CLAUDE.md — read it)
- **Work only in the git worktree**, never edit the original directory directly.
  Original (read-only source of truth): `C:\Users\shat\Desktop\distil_crisper_training`.
- After verifying, **copy to the original with `cp`** only. **After every `cp`,
  re-read `CLAUDE.md` fully AND re-invoke the `andrej-karpathy-skills:karpathy-guidelines`
  skill** (invoke it at session start too).
- **Code does NOT run on native Windows.** It runs on Linux+CUDA: either a RunPod
  pod (cloud, `config.yaml`) or a **local single RTX 4090 in Docker on WSL2**
  (`config.local.yaml`, `Dockerfile`, `docker-compose.yml`, `start.ps1`,
  `LOCAL_4090.md`). On Windows you may only: edit/read, `py_compile`, grep, and run
  the import-light `tests/` via pytest. Never claim a GPU result you didn't observe.
- **Never commit secrets.** (`toDoList` once held a real HF token — it was purged
  from git history this session and gitignored. See §7.)
- **Read full logic before editing** the heart files `scripts/02_generate_pseudo_labels_multi_gpu.py`
  (~4,200 lines) and `scripts/03_train_distillation_multi_gpu.py` (~1,000 lines).

## 1. What this project is
Knowledge-distills the **CrisperWhisper** teacher (a whisper-large-v3 fine-tune;
verbatim/disfluency transcription + accurate word timestamps) into a small fast
student, following the **distil-whisper v3.5** methodology, then exports to
CTranslate2 for faster-whisper. Downstream consumer: the sibling **VeilVoice** project.
Pipeline stages (canonical `*_multi_gpu*` scripts): setup → **02** pseudo-label
generation (teacher inference + WER≤10% gate + dedup) → **03** distillation training
→ **04** CTranslate2 convert → **05** faster-whisper QA → **06** Open-ASR WER eval.

## 2. What the user is doing right now (the live task)
Running **data prep + a PoC training on ONE local RTX 4090** via Docker on WSL2.
Storage drive = **`D:\Storage`** (10 TiB), bind-mounted to `/workspace` in the
container. They cloned the repo at
`C:\Users\shat\Desktop\distil_crisper_training_scripts\distil_crisper_training_scripts`
(same machine) and launched with `.\start.ps1 -Build -Run`. Status as of handoff:
- ✅ Docker image builds cleanly (`distil-crisperwhisper:local`).
- ✅ GPU passthrough works — `nvidia-smi` shows the RTX 4090 inside the container.
- ✅ Stage 2 starts, auto-tunes `batch_size=12` for the 4090.
- ❌ Teacher download hit `No space left on device` — **fixed this session** (see §6).
- ⏳ The 50-sample smoke test has **not** yet completed end-to-end. **This is the next
  thing to verify** (see §8).

## 3. What was built/changed this session (high level — verify against git)
The whole local-4090 + faithful-recipe rework. Major pieces:
- **Stage 2** (`02_…`): **persist accepted audio** as 16kHz FLAC + set `audio_path`
  (previously `null`, so stage 3 trained on silence — this was the #1 fix); store
  `pseudo_label_timestamped` + `prev_text`; guard `dist.barrier()` for single-GPU;
  config-driven storage thresholds + `storage.chunk_size_gb` override; chunk-verify
  audit trail; cache-unlink race fix. Extracted WER gate + content-hash to
  `scripts/wer_utils.py` (torch-free, unit-tested).
- **Stage 3** (`03_…`): load REAL audio (no silence fallback); **8 decoder layers**
  (chosen — see §4); implement `timestamp_probability` / `condition_on_prev_probability`
  / BPE dropout via an explicit `decoder_input_ids` path (teacher+student aligned);
  `weight_decay=0.0`; held-out eval loop; fixed resume (load into prepared model +
  emergency checkpoint). Extracted teacher-forcing/mask builder to
  `scripts/distill_seq_utils.py` (torch-free, unit-tested).
- **New env/tooling:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`,
  `.env.example`, `config.local.yaml`, `start.ps1` (Windows one-shot: venv+tests,
  storage-folder picker, `.env`), root `.gitignore`, `requirements-dev.txt`.
- **New eval:** `scripts/06_eval_open_asr.py` — Open-ASR-Leaderboard WER vs CrisperWhisper.
- **Tests:** import-light `tests/` (23 passing) — WER gate, content-hash,
  teacher-forcing/mask, shard contract.
- **Docs:** `CLAUDE.md`, `GUIDE.md`, `RUNPOD_SETUP.md`, `LOCAL_4090.md` all updated.

## 4. Key decisions + rationale (these were the user's calls — don't silently revert)
- **8 decoder layers** (`student.decoder_layers: 8` in both configs). Trades the
  ~6× speed of the faithful-v3.5 2-layer student for **closer WER parity** (~3×).
  Layers chosen via `np.linspace(0,31,8)` = `[0,4,9,13,18,22,27,31]`.
- **Vanilla timestamps** (no extra work). CrisperWhisper's precise word timestamps
  come from DTW over 15 specific decoder cross-attention alignment heads + a
  retokenized tokenizer — these do **NOT** transfer through standard 2/8-layer
  distillation. The student gets ordinary-Whisper timestamps; WER + verbatim/filler
  behaviour DO transfer. User accepted this.
- **Loss is `0.8·CE + 1.0·KL`, T=2.0, KL×T²** — **verified against the real
  `run_distillation.py`** (`loss = 0.8 * ce_loss + kl_weight * kl_loss`, kl_weight
  default 1.0). Do NOT "normalize" the weights. (A research agent once reported these
  swapped — it was wrong; the source code is authoritative.)
- **Student is built FROM the teacher** (CrisperWhisper), not from openai/whisper-large-v3.
  CrisperWhisper has a *retokenized* tokenizer (vocab 51866 same as large-v3, but
  token↔string mapping differs), so the student must inherit the teacher's
  tokenizer/embeddings. `student.base_model` in config is **vestigial**.
- **Data scope:** "full faithful" mixture supported, English-only. Note YODAS config
  uses only the `en000` shard (~5k h), so the configured mixture is ~50k raw hours
  (~2.4 TB downloads), NOT 196k. Composition > volume for capturing the verbatim
  specialty (weight toward spontaneous speech: AMI/GigaSpeech/People's Speech/YODAS).
- **10 TiB disk:** `config.local.yaml` sets `storage.chunk_size_gb: 100` +
  thresholds `200/100`. Disk math: retained accepted audio (FLAC) for the full
  configured mixture ≈ 0.8 TB; 5 TB was already "enough", 10 TiB is ample.

## 5. CrisperWhisper research findings (cite the sources; re-verify if it matters)
Teacher = whisper-large-v3 fine-tune, **English+German**, retokenized tokenizer,
3-stage training; named training data: **AMI IHM, TIMIT, CommonVoice (CTC-aligned)**,
+ unnamed verbatim sets, + noise aug. Its strengths/benchmarks (model card / paper):
- **Avg WER 6.66** on the Open-ASR set vs 7.7 for large-v3; best on **verbatim**
  corpora (AMI 8.72 vs 16.01; TED-LIUM 3.35 vs 3.9). Tied/slightly behind on
  cleaned-reference sets (Earnings22 12.37 vs 11.3, GigaSpeech 10.27 vs 10.02) —
  the verbatim-output penalty (which is why the WER gate strips fillers).
- Timestamp segmentation F1 ≈ 0.79–0.80 (vs ~0.48–0.66 for large-v3).
- Sources: https://huggingface.co/nyrahealth/CrisperWhisper ,
  https://arxiv.org/abs/2408.16589 , distil-whisper repo `training/run_distillation.py`.
`06_eval_open_asr.py` measures parity on these test splits (a few HF ids —
Earnings22/SPGISpeech/GigaSpeech-test — may need tweaks; it skips-and-logs on failure).

## 6. The disk bug — root cause + fix (committed this session)
**Symptom:** stage 2 teacher download → `OSError: No space left on device (os error 28)`
(then a `hf_xet` segfault), despite D: having 10 TB free.
**Cause:** HF's **Xet** downloader (`xet_get`) staged the 3 GB model into the container
**overlay** filesystem — which lives in Docker's WSL2 VM **on C:** (small/full) — instead
of `/workspace` (= D:).
**Fix (permanent):** set `HF_HUB_DISABLE_XET=1` + `HF_XET_CACHE=/workspace/hf_cache/xet`
in **both** `docker-compose.yml` (env, applies on next `run`, no rebuild) and the
`Dockerfile` (ENV, baked in). This routes downloads through the standard downloader,
which stages inside the HF cache dir on `/workspace` (D:).
**Immediate workaround in a running container:** `export HF_HUB_DISABLE_XET=1` then re-run.
**If it still fails:** confirm the mount with `df -h /workspace` (should be ~10 TB);
if it's a small filesystem the `D:\Storage→/workspace` bind mount didn't take.
(Detail in `LOCAL_4090.md` → "Troubleshooting: No space left on device".)

## 7. Git / secret remediation (done this session)
- Remote `origin` updated to **`https://github.com/Dat-Ngu/distil_crisper_training_scripts.git`**.
- A prior commit had committed `toDoList` containing a **real HF token** → GitHub
  push-protection blocked it. It was **purged from history** (amended out), `_nul` and
  a stray `.txt` removed, and `toDoList`/`_nul` added to `.gitignore`. `master` is now
  on GitHub, **secret-free** (verified `git log --all -- toDoList` is empty).
- **The exposed token must still be rotated by the user** (removing from git ≠ un-leaking).
  Also: `start.ps1` currently echoes the token at the prompt (plain `Read-Host`) — a
  good follow-up is to mask it (`-AsSecureString`).
- Note the branch tangle: GitHub `master` carries all the work; the worktree branch
  `local-4090-faithful` is a now-redundant parallel commit of the same content.

## 8. Verification status — what's proven vs NOT
- ✅ Local static: `py_compile` clean on all scripts; **23 import-light tests pass**
  in a venv (`python -m pytest tests/ -q`).
- ✅ On the 4090: Docker build, GPU passthrough (`nvidia-smi`), stage-2 startup +
  batch auto-tune.
- ❌ **NOT yet proven end-to-end:** the 50-sample smoke test (stage 2 generation →
  audio FLAC written → merge → stage 3 PoC train → 04 convert → 05 test), the eval
  harness against real datasets, timestamp tokenization, BPE-dropout application,
  `condition_on_prev` prompt path. **These are the immediate next checks.**
- **Next step for the live task:** re-run the smoke test with the Xet fix:
  `python3 02_generate_pseudo_labels_multi_gpu.py --config ../config.local.yaml --datasets librispeech --max-samples 50`
  then the rest of the sequence in `LOCAL_4090.md` §4. Watch for: audio FLACs under
  `/workspace/pseudo_labels/audio/librispeech/`, `audio_path` non-null in the JSONL,
  and stage-3 loss decreasing on real (non-silent) features.

## 9. Known gotchas (verify in code; CLAUDE.md has the full list)
- **Two colliding numbered pipelines** + an official-repo wrapper. `*_multi_gpu*` are
  canonical; numbers don't denote stages across pipelines.
- **`config.yaml` (cloud) vs `config.local.yaml` (local 4090)** — pick the right one.
  `DATASET_CONFIGS` (in `02_…`), not `config.yaml`, is what the multi-GPU path iterates.
- **Stage 3 requires persisted audio** (`save_audio: true`, default) or it drops
  audio-less samples / errors if none have audio.
- **Don't change `generate_content_hash` length (16)** without a migration plan.
- **Re-freeze the encoder after every checkpoint resume.**
- **Prefer file-marker signaling over `dist.barrier()`** for new cross-rank sync.
- Pure logic lives in torch-free modules `wer_utils.py` / `distill_seq_utils.py`
  (stage 2/3 delegate to them) — extend tests there.

## 10. Open items / good next tasks
- Finish the smoke test on the 4090; fix anything it surfaces (likely candidates:
  HF gated-dataset auth, `set_prefix_tokens`/`get_prompt_ids` timestamp tokenization,
  BPE-dropout on the fast tokenizer, CTranslate2 convert of an 8-layer student).
- Real data-prep run (spontaneous-speech-weighted), then full/cloud training.
- Mask the token prompt in `start.ps1`; optionally move Docker's disk image to D:.
- Validate `06_eval_open_asr.py` dataset ids against the live HF mirrors.
- Word-timestamp parity (if ever wanted) is a separate cross-attention-alignment
  distillation effort — out of current scope.

## 11. File map (read these, don't trust this list blindly)
- `CLAUDE.md` — canonical project context (read FIRST, in full).
- `scripts/02_generate_pseudo_labels_multi_gpu.py` — stage 2 (the heart, ~4.2k lines).
- `scripts/03_train_distillation_multi_gpu.py` — stage 3 trainer.
- `scripts/wer_utils.py`, `scripts/distill_seq_utils.py` — torch-free, unit-tested logic.
- `scripts/dedup_utils.py` — GlobalDeduplicator + StorageManager.
- `scripts/06_eval_open_asr.py` — parity eval.
- `config.yaml` (cloud) / `config.local.yaml` (local 4090).
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.env.example`, `start.ps1`,
  `LOCAL_4090.md` — the local-4090 Docker/WSL2 path.
- `tests/` — import-light suite (runs on Windows + in the container).

---
*Generated by the 2026-06-05 session. Treat as orientation, not ground truth — read
the code and `CLAUDE.md`, then verify before you change anything.*
