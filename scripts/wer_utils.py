#!/usr/bin/env python3
"""
=============================================================================
Pure (torch-free) helpers for the WER quality gate + content hashing
=============================================================================
These are the core accept/reject and sample-identity primitives of stage 2,
isolated here so they can be unit-tested WITHOUT a GPU / torch / transformers
(see tests/test_wer_utils.py). `02_generate_pseudo_labels_multi_gpu.py` imports
and delegates to them so there is a single source of truth.

The only piece NOT here is the Whisper `EnglishTextNormalizer` call (it lives in
transformers); stage 2 keeps that and wraps it with the filler-strip helpers below.

Imported by 02_generate_pseudo_labels_multi_gpu.py:
    from wer_utils import (content_hash, are_spelling_variants,
                           calculate_wer_spelling_tolerant, is_all_caps_hallucination,
                           strip_bracketed_fillers, drop_standalone_fillers)
=============================================================================
"""

import re
import hashlib
from difflib import SequenceMatcher
from typing import List

# CrisperWhisper emits fillers verbatim; ground truth usually omits them. We strip
# fillers before WER so verbatim output isn't unfairly penalised. Keep these EXACTLY
# in sync with the values historically used in stage 2 (behaviour-preserving).
BRACKETED_FILLER_RE = re.compile(r'\[(?:um|uh|er|ah|uhm|erm|hmm|hm|mm|mhm)\]', re.IGNORECASE)
FILLER_WORDS = {'um', 'uh', 'er', 'ah', 'uhm', 'erm', 'hmm', 'hm', 'mm', 'mhm', 'uh huh', 'mm hmm'}


def strip_bracketed_fillers(text: str) -> str:
    """Remove CrisperWhisper bracketed fillers like [Um], [Uh] (case-insensitive)."""
    return BRACKETED_FILLER_RE.sub('', text)


def drop_standalone_fillers(text: str) -> str:
    """Drop standalone filler words from already-normalized (whitespace-split) text."""
    return ' '.join(w for w in text.split() if w not in FILLER_WORDS)


def is_all_caps_hallucination(hypothesis: str) -> bool:
    """
    True if the hypothesis is entirely upper-case (a known Whisper hallucination
    signature that the official distil-whisper filter rejects outright).
    """
    return hypothesis is not None and len(hypothesis) > 0 and hypothesis.upper() == hypothesis


def are_spelling_variants(word1: str, word2: str, threshold: float = 0.85) -> bool:
    """
    True if two words are spelling variants (e.g. British vs American):
    same first letter and SequenceMatcher ratio >= threshold.
    Examples: colour/color (~0.91), realise/realize (~0.86), behaviour/behavior (~0.94).
    """
    if word1 == word2:
        return True
    if not word1 or not word2 or word1[0] != word2[0]:
        return False
    return SequenceMatcher(None, word1, word2).ratio() >= threshold


def calculate_wer_spelling_tolerant(ref_words: List[str], hyp_words: List[str]) -> float:
    """
    Levenshtein WER over word lists, treating spelling variants as zero-cost matches.
    Edge cases match the official implementation: empty ref with non-empty hyp -> 1.0,
    both empty -> 0.0, empty hyp -> 1.0.
    """
    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return 1.0 if m > 0 else 0.0
    if m == 0:
        return 1.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if are_spelling_variants(ref_words[i - 1], hyp_words[j - 1]):
                dp[i][j] = dp[i - 1][j - 1]  # equivalent words, no cost
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + 1,  # substitution
                )

    return dp[n][m] / n


def content_hash(audio_array, text: str, dataset_name: str) -> str:
    """
    Deterministic content-based sample ID: "{dataset}_{md5_16hex}" from the first
    4000 audio samples + text + dataset name. Stable across runs / GPU counts, which
    is what makes resume + dedup robust.

    DO NOT change the 16-char length without a migration plan (existing IDs use 16).
    `audio_array` is a numpy array (its `.tobytes()` is used; numpy is not imported
    here so this module stays torch/numpy-free at import time).
    """
    hasher = hashlib.md5()
    audio_slice = audio_array[:4000] if len(audio_array) > 4000 else audio_array
    hasher.update(audio_slice.tobytes())
    hasher.update(text.encode('utf-8'))
    hasher.update(dataset_name.encode('utf-8'))
    return f"{dataset_name}_{hasher.hexdigest()[:16]}"
