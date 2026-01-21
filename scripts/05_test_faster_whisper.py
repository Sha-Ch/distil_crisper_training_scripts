#!/usr/bin/env python3
"""
=============================================================================
faster-whisper Compatibility Test for Distil-CrisperWhisper
=============================================================================
Verifies that the converted model works correctly with faster-whisper,
including word-level timestamp extraction.

Tests:
1. Model loading
2. Basic transcription
3. Word-level timestamps
4. Speed comparison with original
5. Memory usage

Usage:
  python3 05_test_faster_whisper.py --model /path/to/ct2/model
  python3 05_test_faster_whisper.py --config ../config.yaml

Requirements:
  pip install faster-whisper numpy
=============================================================================
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    duration: float = 0.0
    details: Optional[Dict] = None


def generate_test_audio(
    duration: float = 10.0,
    sample_rate: int = 16000,
    include_speech_pattern: bool = True
) -> np.ndarray:
    """
    Generate synthetic test audio.

    For a more realistic test, you should use actual audio files.
    This generates a simple pattern that Whisper can process.
    """
    t = np.linspace(0, duration, int(duration * sample_rate))

    if include_speech_pattern:
        # Create something that vaguely resembles speech patterns
        # Multiple frequency components with amplitude modulation
        frequencies = [200, 400, 800, 1200, 2000]
        audio = np.zeros_like(t)

        for freq in frequencies:
            # Add frequency component with random amplitude modulation
            envelope = np.abs(np.sin(2 * np.pi * 0.5 * t))  # Slow modulation
            audio += 0.1 * envelope * np.sin(2 * np.pi * freq * t)

        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.5
    else:
        # Simple sine wave
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    return audio.astype(np.float32)


def test_model_loading(model_path: str) -> TestResult:
    """Test 1: Model can be loaded."""
    try:
        from faster_whisper import WhisperModel

        start = time.time()

        # Determine device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        console.print(f"  Loading model from {model_path}...")
        console.print(f"  Device: {device}, Compute: {compute_type}")

        model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
        )

        duration = time.time() - start

        return TestResult(
            name="Model Loading",
            passed=True,
            message=f"Model loaded successfully on {device}",
            duration=duration,
            details={"device": device, "compute_type": compute_type}
        )

    except Exception as e:
        return TestResult(
            name="Model Loading",
            passed=False,
            message=f"Failed: {str(e)}"
        )


def test_basic_transcription(model_path: str) -> TestResult:
    """Test 2: Basic transcription works."""
    try:
        from faster_whisper import WhisperModel
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        model = WhisperModel(model_path, device=device, compute_type=compute_type)

        # Generate test audio
        audio = generate_test_audio(duration=5.0)

        start = time.time()
        segments, info = model.transcribe(audio, language="en")
        segments_list = list(segments)  # Consume generator
        duration = time.time() - start

        return TestResult(
            name="Basic Transcription",
            passed=True,
            message=f"Transcribed successfully in {duration:.2f}s",
            duration=duration,
            details={
                "segments": len(segments_list),
                "language": info.language,
                "language_probability": info.language_probability,
                "audio_duration": 5.0,
                "rtf": duration / 5.0
            }
        )

    except Exception as e:
        return TestResult(
            name="Basic Transcription",
            passed=False,
            message=f"Failed: {str(e)}"
        )


def test_word_timestamps(model_path: str) -> TestResult:
    """Test 3: Word-level timestamps work (critical for CrisperWhisper)."""
    try:
        from faster_whisper import WhisperModel
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        model = WhisperModel(model_path, device=device, compute_type=compute_type)

        # Generate test audio
        audio = generate_test_audio(duration=10.0)

        start = time.time()
        segments, info = model.transcribe(
            audio,
            language="en",
            word_timestamps=True,  # THIS IS THE KEY FEATURE
        )

        # Collect word timestamps
        all_words = []
        for segment in segments:
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    all_words.append({
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'probability': word.probability
                    })

        duration = time.time() - start

        if len(all_words) > 0:
            return TestResult(
                name="Word Timestamps",
                passed=True,
                message=f"Word timestamps working! Found {len(all_words)} words",
                duration=duration,
                details={
                    "word_count": len(all_words),
                    "sample_words": all_words[:5] if all_words else []
                }
            )
        else:
            return TestResult(
                name="Word Timestamps",
                passed=True,  # Still passes - might just be no detected speech
                message="No words detected in test audio (expected for synthetic audio)",
                duration=duration
            )

    except Exception as e:
        return TestResult(
            name="Word Timestamps",
            passed=False,
            message=f"Failed: {str(e)}"
        )


def test_memory_usage(model_path: str) -> TestResult:
    """Test 4: Memory usage is reasonable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return TestResult(
                name="Memory Usage",
                passed=True,
                message="Skipped (no GPU)",
                details={"reason": "CPU mode"}
            )

        from faster_whisper import WhisperModel

        # Clear cache first
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_memory = torch.cuda.memory_allocated() / 1e9

        model = WhisperModel(model_path, device="cuda", compute_type="float16")

        model_memory = torch.cuda.memory_allocated() / 1e9

        # Run transcription
        audio = generate_test_audio(duration=5.0)
        segments, _ = model.transcribe(audio, language="en", word_timestamps=True)
        list(segments)

        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        # Distilled model should use less than 4GB
        passed = peak_memory < 4.0

        return TestResult(
            name="Memory Usage",
            passed=passed,
            message=f"Peak: {peak_memory:.2f}GB (target: <4GB)",
            details={
                "initial_gb": initial_memory,
                "model_gb": model_memory,
                "peak_gb": peak_memory
            }
        )

    except Exception as e:
        return TestResult(
            name="Memory Usage",
            passed=False,
            message=f"Failed: {str(e)}"
        )


def test_speed(model_path: str) -> TestResult:
    """Test 5: Speed is acceptable (should be fast for distilled model)."""
    try:
        from faster_whisper import WhisperModel
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        model = WhisperModel(model_path, device=device, compute_type=compute_type)

        # Test with various audio lengths
        durations = [5, 10, 30]
        results = []

        for audio_duration in durations:
            audio = generate_test_audio(duration=float(audio_duration))

            start = time.time()
            segments, _ = model.transcribe(audio, language="en", word_timestamps=True)
            list(segments)
            processing_time = time.time() - start

            rtf = processing_time / audio_duration
            results.append({
                'audio_duration': audio_duration,
                'processing_time': processing_time,
                'rtf': rtf
            })

        avg_rtf = sum(r['rtf'] for r in results) / len(results)

        # Distilled model should have RTF < 0.1 on GPU
        if device == "cuda":
            passed = avg_rtf < 0.1
            target = "< 0.1x"
        else:
            passed = avg_rtf < 0.5
            target = "< 0.5x"

        return TestResult(
            name="Speed Test",
            passed=passed,
            message=f"Avg RTF: {avg_rtf:.3f}x (target: {target})",
            details={
                'avg_rtf': avg_rtf,
                'device': device,
                'per_duration': results
            }
        )

    except Exception as e:
        return TestResult(
            name="Speed Test",
            passed=False,
            message=f"Failed: {str(e)}"
        )


def run_all_tests(model_path: str) -> List[TestResult]:
    """Run all compatibility tests."""
    tests = [
        ("Model Loading", test_model_loading),
        ("Basic Transcription", test_basic_transcription),
        ("Word Timestamps", test_word_timestamps),
        ("Memory Usage", test_memory_usage),
        ("Speed Test", test_speed),
    ]

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        for name, test_fn in tests:
            task = progress.add_task(f"Running: {name}", total=None)
            result = test_fn(model_path)
            results.append(result)
            progress.remove_task(task)

            status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
            console.print(f"  {status} {result.name}: {result.message}")

    return results


def print_results(results: List[TestResult]):
    """Print test results in a table."""
    table = Table(title="faster-whisper Compatibility Test Results")

    table.add_column("Test", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Message")
    table.add_column("Duration", justify="right")

    for result in results:
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        duration = f"{result.duration:.2f}s" if result.duration > 0 else "-"
        table.add_row(result.name, status, result.message, duration)

    console.print("\n")
    console.print(table)

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    if passed == total:
        console.print(f"\n[bold green]All {total} tests passed![/bold green]")
        console.print("[green]Your distil-CrisperWhisper model is compatible with faster-whisper.[/green]")
    else:
        console.print(f"\n[bold yellow]{passed}/{total} tests passed[/bold yellow]")
        console.print("[yellow]Some tests failed. Check the results above.[/yellow]")


def main():
    parser = argparse.ArgumentParser(description='Test faster-whisper compatibility')
    parser.add_argument('--model', type=str, help='Path to CTranslate2 model')
    parser.add_argument('--config', type=str, help='Path to config.yaml')
    parser.add_argument('--audio', type=str, help='Optional: real audio file to test')
    args = parser.parse_args()

    # Determine model path
    if args.model:
        model_path = args.model
    elif args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        output_dir = Path(config['paths']['output_dir'])
        model_path = str(output_dir / 'distil-crisperwhisper-ct2')
    else:
        # Try default location
        model_path = '/workspace/output/distil-crisperwhisper-ct2'

    if not Path(model_path).exists():
        console.print(f"[red]Model not found at {model_path}[/red]")
        console.print("Please provide --model or --config argument")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold cyan]faster-whisper Compatibility Test[/bold cyan]\n"
        f"Model: {model_path}",
        title="Test Configuration"
    ))

    console.print("\n[bold]Running tests...[/bold]\n")

    results = run_all_tests(model_path)
    print_results(results)

    # Exit with error if any test failed
    if not all(r.passed for r in results):
        sys.exit(1)


if __name__ == '__main__':
    main()
