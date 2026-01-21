#!/usr/bin/env python3
"""
=============================================================================
CTranslate2 Conversion Script for Distil-CrisperWhisper
=============================================================================
Converts the trained distilled model to CTranslate2 format for use with
faster-whisper, providing significant inference speedup.

Usage: python3 05_convert_to_ctranslate2.py [--config config.yaml]
=============================================================================
"""

import os
import sys
import yaml
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def convert_to_ctranslate2(
    model_path: str,
    output_path: str,
    quantization: str = "float16",
    copy_files: bool = True
) -> Path:
    """
    Convert a HuggingFace Whisper model to CTranslate2 format.

    Args:
        model_path: Path to the HuggingFace model directory
        output_path: Path for the CTranslate2 output
        quantization: Quantization type (float32, float16, int8, int8_float16)
        copy_files: Whether to copy tokenizer files

    Returns:
        Path to the converted model
    """
    import ctranslate2

    model_path = Path(model_path)
    output_path = Path(output_path)

    console.print(f"[bold blue]Converting model to CTranslate2...[/bold blue]")
    console.print(f"  Source: {model_path}")
    console.print(f"  Output: {output_path}")
    console.print(f"  Quantization: {quantization}")

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        # Step 1: Convert model
        task = progress.add_task("Converting model weights...", total=None)

        converter = ctranslate2.converters.TransformersConverter(str(model_path))
        converter.convert(
            str(output_path),
            quantization=quantization,
            force=True
        )

        progress.update(task, completed=True)

        # Step 2: Copy tokenizer files if needed
        if copy_files:
            task = progress.add_task("Copying tokenizer files...", total=None)

            files_to_copy = [
                'tokenizer.json',
                'vocab.json',
                'merges.txt',
                'normalizer.json',
                'added_tokens.json',
                'special_tokens_map.json',
                'tokenizer_config.json',
                'preprocessor_config.json',
                'config.json'
            ]

            for filename in files_to_copy:
                src = model_path / filename
                if src.exists():
                    shutil.copy2(src, output_path / filename)

            progress.update(task, completed=True)

    console.print(f"[green]✓ Model converted successfully![/green]")
    return output_path


def verify_conversion(
    ctranslate2_path: str,
    test_audio_path: Optional[str] = None
) -> bool:
    """
    Verify the converted model works correctly.

    Args:
        ctranslate2_path: Path to the CTranslate2 model
        test_audio_path: Optional path to a test audio file

    Returns:
        True if verification passes
    """
    try:
        from faster_whisper import WhisperModel

        console.print("\n[bold blue]Verifying converted model...[/bold blue]")

        # Load model
        model = WhisperModel(
            ctranslate2_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "float32"
        )

        console.print("[green]✓ Model loaded successfully[/green]")

        # Get model info
        console.print(f"  Model path: {ctranslate2_path}")

        if test_audio_path and Path(test_audio_path).exists():
            console.print(f"\n[yellow]Testing with audio file: {test_audio_path}[/yellow]")

            segments, info = model.transcribe(test_audio_path)

            console.print(f"  Detected language: {info.language} ({info.language_probability:.2%})")
            console.print(f"  Duration: {info.duration:.1f}s")

            console.print("\n  Transcription:")
            for segment in segments:
                console.print(f"    [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        console.print("\n[green]✓ Verification passed![/green]")
        return True

    except Exception as e:
        console.print(f"[red]✗ Verification failed: {e}[/red]")
        return False


def benchmark_model(
    ctranslate2_path: str,
    test_audio_path: str,
    num_runs: int = 5
) -> Dict[str, float]:
    """
    Benchmark the converted model's inference speed.

    Args:
        ctranslate2_path: Path to the CTranslate2 model
        test_audio_path: Path to a test audio file
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with benchmark results
    """
    import time
    from faster_whisper import WhisperModel

    console.print("\n[bold blue]Benchmarking inference speed...[/bold blue]")

    model = WhisperModel(
        ctranslate2_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "float32"
    )

    # Warmup
    console.print("[yellow]Warming up...[/yellow]")
    for _ in range(2):
        list(model.transcribe(test_audio_path))

    # Benchmark
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        segments, info = model.transcribe(test_audio_path)
        # Consume the generator
        list(segments)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        console.print(f"  Run {i+1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Calculate RTF (Real-Time Factor)
    import soundfile as sf
    audio_data, sr = sf.read(test_audio_path)
    audio_duration = len(audio_data) / sr
    rtf = avg_time / audio_duration

    results = {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'audio_duration': audio_duration,
        'rtf': rtf
    }

    console.print(f"\n[green]Benchmark Results:[/green]")
    console.print(f"  Audio duration: {audio_duration:.1f}s")
    console.print(f"  Average inference: {avg_time:.3f}s")
    console.print(f"  Min/Max: {min_time:.3f}s / {max_time:.3f}s")
    console.print(f"  Real-Time Factor: {rtf:.2f}x")

    if rtf < 1.0:
        console.print(f"  [green]✓ Faster than real-time! ({1/rtf:.1f}x real-time)[/green]")

    return results


def create_model_card(
    output_path: str,
    config: Dict[str, Any],
    benchmark_results: Optional[Dict[str, float]] = None
):
    """Create a model card for the converted model."""

    card_content = f"""---
language: en
tags:
  - whisper
  - speech-recognition
  - ctranslate2
  - faster-whisper
  - distillation
license: apache-2.0
---

# Distil-CrisperWhisper (CTranslate2)

This is a distilled version of [CrisperWhisper](https://huggingface.co/nyrahealth/CrisperWhisper),
converted to CTranslate2 format for use with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Model Details

- **Base Model**: CrisperWhisper (nyrahealth/CrisperWhisper)
- **Distillation**: {config['student']['decoder_layers']} decoder layers (reduced from 32)
- **Encoder Layers**: {config['student']['encoder_layers']}
- **Quantization**: {config['conversion']['quantization']}

## Usage

```python
from faster_whisper import WhisperModel

model = WhisperModel("path/to/model", device="cuda", compute_type="float16")

segments, info = model.transcribe("audio.wav")
for segment in segments:
    print(f"[{{segment.start:.2f}}s -> {{segment.end:.2f}}s] {{segment.text}}")
```

## Performance

"""

    if benchmark_results:
        card_content += f"""
| Metric | Value |
|--------|-------|
| Average Inference Time | {benchmark_results['avg_time']:.3f}s |
| Real-Time Factor | {benchmark_results['rtf']:.2f}x |
| Speed | {1/benchmark_results['rtf']:.1f}x real-time |
"""

    card_content += """
## Training

This model was trained using knowledge distillation from CrisperWhisper on:
- GigaSpeech
- VoxPopuli
- LibriSpeech

## License

Apache 2.0
"""

    card_path = Path(output_path) / 'README.md'
    with open(card_path, 'w') as f:
        f.write(card_content)

    console.print(f"[green]✓ Model card created: {card_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description='Convert distilled model to CTranslate2')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model-path', type=str, help='Override model path from config')
    parser.add_argument('--output-path', type=str, help='Override output path from config')
    parser.add_argument('--quantization', type=str, choices=['float32', 'float16', 'int8', 'int8_float16'],
                        help='Override quantization from config')
    parser.add_argument('--test-audio', type=str, help='Path to test audio file for verification')
    parser.add_argument('--benchmark', action='store_true', help='Run inference benchmark')
    parser.add_argument('--skip-verify', action='store_true', help='Skip verification step')
    args = parser.parse_args()

    # Find config file
    config_path = Path(args.config)
    if not config_path.exists():
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / 'config.yaml'

    if not config_path.exists():
        console.print(f"[red]Config file not found: {args.config}[/red]")
        sys.exit(1)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Determine paths
    model_path = args.model_path or str(Path(config['paths']['output_dir']) / 'distil-crisperwhisper-final')
    output_path = args.output_path or str(Path(config['paths']['output_dir']) / 'distil-crisperwhisper-ct2')
    quantization = args.quantization or config['conversion']['quantization']

    # Check if source model exists
    if not Path(model_path).exists():
        console.print(f"[red]Source model not found: {model_path}[/red]")
        console.print("[yellow]Run 04_train_distillation.py first[/yellow]")
        sys.exit(1)

    # Convert model
    convert_to_ctranslate2(
        model_path=model_path,
        output_path=output_path,
        quantization=quantization
    )

    # Verify conversion
    if not args.skip_verify:
        success = verify_conversion(output_path, args.test_audio)
        if not success:
            sys.exit(1)

    # Run benchmark if requested
    benchmark_results = None
    if args.benchmark and args.test_audio:
        benchmark_results = benchmark_model(output_path, args.test_audio)

    # Create model card
    create_model_card(output_path, config, benchmark_results)

    console.print(f"\n[bold green]Conversion complete![/bold green]")
    console.print(f"CTranslate2 model saved to: {output_path}")


if __name__ == '__main__':
    main()
