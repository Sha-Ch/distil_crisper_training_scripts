#!/usr/bin/env python3
"""
=============================================================================
Open ASR Leaderboard WER Evaluation for Distil-CrisperWhisper
=============================================================================
Measures WER parity against CrisperWhisper on the test sets CrisperWhisper is
benchmarked on (the HuggingFace Open ASR Leaderboard mixture). Runs the distilled
HF student (and optionally the CrisperWhisper teacher) over each test split,
normalizes with the official Whisper EnglishTextNormalizer, computes WER with
jiwer, and prints a table comparing to CrisperWhisper's published numbers.

This is how you verify "did distillation reach parity?". The student emits
verbatim/filler output like the teacher, so on cleaned-reference sets
(Earnings22, GigaSpeech) some gap is EXPECTED — the teacher is also penalized
there (see published numbers). The verbatim sets (AMI, TED-LIUM) are where the
parity question really matters.

Usage (on a GPU host — pod or local Docker/WSL2; NOT native Windows):
  python3 06_eval_open_asr.py --config ../config.local.yaml
  python3 06_eval_open_asr.py --config ../config.local.yaml --eval-teacher
  python3 06_eval_open_asr.py --config ../config.local.yaml --datasets librispeech_clean ami --max-samples 200
  python3 06_eval_open_asr.py --config ../config.local.yaml --model /workspace/output/distil-crisperwhisper-final

Notes:
- Gated sets (Common Voice, GigaSpeech, SPGISpeech) require accepting their terms
  on HuggingFace with the same account as $HF_TOKEN, or they are SKIPPED.
- A few eval-set HF ids/columns vary across mirrors; the EVAL_DATASETS registry
  below is intentionally editable, and any dataset that fails to load is skipped
  (logged) rather than aborting the whole run.

References:
- CrisperWhisper: https://huggingface.co/nyrahealth/CrisperWhisper
- Open ASR Leaderboard: https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
=============================================================================
"""

import os
import sys
import re
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml

# Faster, resumable HF downloads only if the package is present (matches stage 2).
try:
    import importlib.util
    if importlib.util.find_spec('hf_transfer') is not None:
        os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
except Exception:
    pass
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '1800')

import torch
import numpy as np
import jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from rich.console import Console
from rich.table import Table

warnings.filterwarnings('ignore')
console = Console()


# =============================================================================
# Eval dataset registry (Open ASR Leaderboard test splits).
# Edit ids/columns here if a mirror differs. `published` = CrisperWhisper's
# reported WER (model card / paper) for the comparison column.
# =============================================================================
EVAL_DATASETS: List[Dict[str, Any]] = [
    {"key": "librispeech_clean", "hf_name": "librispeech_asr", "subset": "clean",
     "split": "test", "text_column": "text", "audio_column": "audio",
     "gated": False, "published": 1.74},
    {"key": "librispeech_other", "hf_name": "librispeech_asr", "subset": "other",
     "split": "test", "text_column": "text", "audio_column": "audio",
     "gated": False, "published": 3.97},
    {"key": "ami", "hf_name": "edinburghcstr/ami", "subset": "ihm",
     "split": "test", "text_column": "text", "audio_column": "audio",
     "gated": False, "published": 8.72},
    {"key": "tedlium", "hf_name": "LIUM/tedlium", "subset": "release3",
     "split": "test", "text_column": "text", "audio_column": "audio",
     "gated": False, "published": 3.35},
    {"key": "voxpopuli", "hf_name": "facebook/voxpopuli", "subset": "en",
     "split": "test", "text_column": "normalized_text", "audio_column": "audio",
     "gated": False, "published": 8.61},
    {"key": "commonvoice", "hf_name": "mozilla-foundation/common_voice_17_0", "subset": "en",
     "split": "test", "text_column": "sentence", "audio_column": "audio",
     "gated": True, "published": 8.19},
    {"key": "gigaspeech", "hf_name": "speechcolab/gigaspeech", "subset": "xl",
     "split": "test", "text_column": "text", "audio_column": "audio",
     "gated": True, "published": 10.27, "clean": "gigaspeech"},
    # NOTE: ids/columns below vary across mirrors — adjust if they fail to load.
    {"key": "earnings22", "hf_name": "revdotcom/earnings22", "subset": None,
     "split": "test", "text_column": "sentence", "audio_column": "audio",
     "gated": False, "published": 12.37},
    {"key": "spgispeech", "hf_name": "kensho/spgispeech", "subset": "test",
     "split": "test", "text_column": "transcript", "audio_column": "audio",
     "gated": True, "published": 2.71},
]

# GigaSpeech (and some others) use literal punctuation tokens + non-speech tags
# that must be stripped BEFORE the Whisper normalizer (else "<COMMA>" -> "comma").
_GIGA_PUNCT = {
    " <COMMA>": ",", " <PERIOD>": ".", " <QUESTIONMARK>": "?", " <EXCLAMATIONPOINT>": "!",
}
_GIGA_TAGS = re.compile(r"<(SIL|MUSIC|NOISE|OTHER|UNK)>", re.IGNORECASE)


def clean_reference(text: str, clean_kind: Optional[str]) -> str:
    """Dataset-specific cleanup applied to the REFERENCE before normalization."""
    if clean_kind == "gigaspeech":
        for tok, repl in _GIGA_PUNCT.items():
            text = text.replace(tok, repl)
        text = _GIGA_TAGS.sub(" ", text)
    return text


def load_model(model_path: str, cache_dir: Optional[str], device: torch.device):
    """Load a Whisper model + its own processor (tokenizer)."""
    processor = WhisperProcessor.from_pretrained(model_path, cache_dir=cache_dir)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, cache_dir=cache_dir, torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return model, processor


@torch.inference_mode()
def transcribe_batch(model, processor, audios: List[np.ndarray], device: torch.device) -> List[str]:
    """Greedy batched transcription (matches the pseudo-labelling decode settings)."""
    feats = processor(
        audios, sampling_rate=16000, return_tensors="pt", padding=True
    ).input_features.to(device, dtype=torch.float16)
    generated = model.generate(
        feats, language="en", task="transcribe",
        num_beams=1, max_new_tokens=256, return_timestamps=False,
    )
    return [t.strip() for t in processor.batch_decode(generated, skip_special_tokens=True)]


def eval_dataset(model, processor, spec: Dict[str, Any], normalizer, device,
                 max_samples: Optional[int], batch_size: int) -> Optional[Tuple[float, int]]:
    """Return (WER%, n_samples) for one dataset, or None if it could not run."""
    from datasets import load_dataset, Audio

    key = spec["key"]
    try:
        kwargs = {"split": spec["split"], "streaming": True, "trust_remote_code": True}
        if spec["subset"]:
            kwargs["name"] = spec["subset"]
        ds = load_dataset(spec["hf_name"], **kwargs)
        # Let `datasets` resample to 16kHz mono on access.
        ds = ds.cast_column(spec["audio_column"], Audio(sampling_rate=16000))
    except Exception as e:
        console.print(f"[yellow]  ⚠ {key}: could not load ({type(e).__name__}: {str(e)[:120]}) — skipping[/yellow]")
        return None

    refs: List[str] = []
    hyps: List[str] = []
    audio_buf: List[np.ndarray] = []
    ref_buf: List[str] = []
    n = 0

    def flush():
        if not audio_buf:
            return
        try:
            hyps.extend(transcribe_batch(model, processor, audio_buf, device))
            refs.extend(ref_buf)
        except Exception as e:
            console.print(f"[yellow]  ⚠ {key}: inference error on a batch ({str(e)[:80]}) — skipping batch[/yellow]")
        audio_buf.clear()
        ref_buf.clear()

    try:
        for sample in ds:
            ref = sample.get(spec["text_column"], "")
            audio = sample.get(spec["audio_column"], {})
            if not isinstance(ref, str) or not ref.strip():
                continue
            if not isinstance(audio, dict) or "array" not in audio:
                continue
            arr = np.asarray(audio["array"], dtype=np.float32)
            if arr.size == 0:
                continue
            audio_buf.append(arr)
            ref_buf.append(ref)
            n += 1
            if len(audio_buf) >= batch_size:
                flush()
            if max_samples and n >= max_samples:
                break
        flush()
    except Exception as e:
        console.print(f"[yellow]  ⚠ {key}: iteration error ({str(e)[:100]}) — using partial results[/yellow]")

    # Normalize + drop empty-reference pairs, then compute corpus WER.
    norm_refs, norm_hyps = [], []
    for r, h in zip(refs, hyps):
        r = normalizer(clean_reference(r, spec.get("clean")))
        h = normalizer(h)
        if r.strip():
            norm_refs.append(r)
            norm_hyps.append(h)
    if not norm_refs:
        console.print(f"[yellow]  ⚠ {key}: no usable samples — skipping[/yellow]")
        return None

    wer = jiwer.wer(norm_refs, norm_hyps) * 100.0
    return wer, len(norm_refs)


def evaluate_model(label: str, model_path: str, specs: List[Dict[str, Any]],
                   cache_dir, device, normalizer, max_samples, batch_size) -> Dict[str, Tuple[float, int]]:
    console.print(f"\n[bold blue]Evaluating {label}: {model_path}[/bold blue]")
    model, processor = load_model(model_path, cache_dir, device)
    results: Dict[str, Tuple[float, int]] = {}
    try:
        for spec in specs:
            console.print(f"[cyan]  • {spec['key']}{' (gated)' if spec.get('gated') else ''}...[/cyan]")
            out = eval_dataset(model, processor, spec, normalizer, device, max_samples, batch_size)
            if out is not None:
                wer, n = out
                results[spec["key"]] = (wer, n)
                console.print(f"[green]    {spec['key']}: WER {wer:.2f}%  (n={n:,})[/green]")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="Open ASR Leaderboard WER eval for the distilled student")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, default=None,
                        help="HF student path (default: <output_dir>/distil-crisperwhisper-final)")
    parser.add_argument("--eval-teacher", action="store_true",
                        help="Also evaluate the CrisperWhisper teacher for a side-by-side baseline")
    parser.add_argument("--datasets", nargs="+", help="Subset of dataset keys to run (default: all)")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap samples per dataset (quick run)")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        console.print(f"[red]Config not found: {args.config}[/red]")
        sys.exit(1)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cache_dir = config.get("paths", {}).get("hf_cache")
    output_dir = Path(config.get("paths", {}).get("output_dir", "/workspace/output"))
    teacher_id = config.get("teacher", {}).get("model_id", "nyrahealth/CrisperWhisper")

    student_path = args.model or str(output_dir / "distil-crisperwhisper-final")
    if not Path(student_path).exists():
        console.print(f"[red]Student model not found at {student_path}[/red]")
        console.print("[yellow]Train + save a model first (03_train_distillation_multi_gpu.py), or pass --model.[/yellow]")
        sys.exit(1)

    if not torch.cuda.is_available():
        console.print("[red]No CUDA device. Run this on a GPU host (pod or local Docker/WSL2).[/red]")
        sys.exit(1)
    device = torch.device("cuda:0")
    normalizer = EnglishTextNormalizer({})

    specs = EVAL_DATASETS
    if args.datasets:
        wanted = set(args.datasets)
        specs = [s for s in EVAL_DATASETS if s["key"] in wanted]
        if not specs:
            console.print(f"[red]No matching dataset keys. Available: {[s['key'] for s in EVAL_DATASETS]}[/red]")
            sys.exit(1)

    student_res = evaluate_model("STUDENT", student_path, specs, cache_dir, device,
                                 normalizer, args.max_samples, args.batch_size)
    teacher_res: Dict[str, Tuple[float, int]] = {}
    if args.eval_teacher:
        teacher_res = evaluate_model("TEACHER (CrisperWhisper)", teacher_id, specs, cache_dir, device,
                                     normalizer, args.max_samples, args.batch_size)

    # ---- Results table ----
    table = Table(title="Open ASR Leaderboard — WER% (lower is better)")
    table.add_column("Dataset", style="cyan")
    table.add_column("Student", justify="right")
    if args.eval_teacher:
        table.add_column("Teacher", justify="right")
    table.add_column("CrisperWhisper (pub)", justify="right")
    table.add_column("Δ vs pub", justify="right")

    student_vals, pub_vals = [], []
    for spec in specs:
        key = spec["key"]
        pub = spec.get("published")
        s = student_res.get(key)
        s_str = f"{s[0]:.2f}" if s else "—"
        row = [key, s_str]
        if args.eval_teacher:
            t = teacher_res.get(key)
            row.append(f"{t[0]:.2f}" if t else "—")
        row.append(f"{pub:.2f}" if pub is not None else "—")
        if s and pub is not None:
            delta = s[0] - pub
            row.append(f"{delta:+.2f}")
            student_vals.append(s[0])
            pub_vals.append(pub)
        else:
            row.append("—")
        table.add_row(*row)

    if student_vals:
        avg_student = sum(student_vals) / len(student_vals)
        avg_pub = sum(pub_vals) / len(pub_vals)
        avg_row = ["[bold]AVG (matched)[/bold]", f"[bold]{avg_student:.2f}[/bold]"]
        if args.eval_teacher:
            avg_row.append("")
        avg_row.append(f"[bold]{avg_pub:.2f}[/bold]")
        avg_row.append(f"[bold]{avg_student - avg_pub:+.2f}[/bold]")
        table.add_row(*avg_row)

    console.print()
    console.print(table)
    console.print("\n[dim]Δ vs pub > 0 means the student is worse than CrisperWhisper's published WER. "
                  "Some gap on cleaned-reference sets (Earnings22/GigaSpeech) is expected for a verbatim model.[/dim]")


if __name__ == "__main__":
    main()
