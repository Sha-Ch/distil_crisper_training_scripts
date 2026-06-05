#!/usr/bin/env python3
"""
=============================================================================
Pure (torch-free) sequence helpers for distillation training
=============================================================================
Kept dependency-light on purpose: the teacher-forcing / prompt-masking logic is
the trickiest part of the trainer's data path, and isolating it here lets it be
unit-tested WITHOUT a GPU / torch / transformers (see tests/test_distill_seq_utils.py).

Imported by 03_train_distillation_multi_gpu.py:
    from distill_seq_utils import build_decoder_inputs_and_labels, pad_decoder_batch
=============================================================================
"""

from typing import List, Tuple


def build_decoder_inputs_and_labels(
    prompt_ids: List[int],
    label_ids: List[int],
    max_length: int = 448,
) -> Tuple[List[int], List[int]]:
    """
    Build (decoder_input_ids, labels) for ONE sample using teacher forcing.

    Faithful distil-whisper construction with optional prompt conditioning:

        full = prompt_ids + label_ids        # what the decoder reads, left-to-right
        decoder_input_ids = full[:-1]
        labels            = full[1:]          # next-token targets

    `label_ids` already carries the Whisper special prefix/suffix
    (<|startoftranscript|> ... <|endoftext|>). `prompt_ids`, when present, is the
    previous-context prompt (starts with <|startofprev|>). Loss must NOT be trained
    to predict the prompt, so every label position that predicts a token inside the
    prompt region is masked to -100. With no prompt, the whole transcript is
    supervised (the standard Whisper objective).

    The sequence is capped so decoder_input_ids fits in `max_length` positions:
    prompt tokens are dropped from the left first (preserving the transcript), then
    a hard truncation is applied as a last resort.

    Returns (decoder_input_ids, labels), both lists of equal length.
    """
    prompt_ids = list(prompt_ids)
    label_ids = list(label_ids)
    full = prompt_ids + label_ids

    # Cap so decoder_input_ids (= full[:-1]) fits within max_length positions.
    if len(full) - 1 > max_length:
        overflow = (len(full) - 1) - max_length
        drop = min(overflow, len(prompt_ids))
        prompt_ids = prompt_ids[drop:]
        full = prompt_ids + label_ids
        if len(full) - 1 > max_length:
            # Transcript alone is still too long: keep the last max_length+1 tokens.
            full = full[-(max_length + 1):]
            prompt_ids = []  # transcript was truncated; nothing left to mask

    if len(full) < 2:
        # Degenerate (shouldn't happen with real labels); emit a 1-token no-op.
        return [full[0] if full else 0], [-100]

    decoder_input_ids = full[:-1]
    labels = full[1:]

    # labels[j] predicts full[j+1]; it is a transcript token when
    # j + 1 >= len(prompt_ids), i.e. mask j < len(prompt_ids) - 1.
    mask_upto = max(len(prompt_ids) - 1, 0)
    for j in range(min(mask_upto, len(labels))):
        labels[j] = -100

    return decoder_input_ids, labels


def pad_decoder_batch(
    dec_list: List[List[int]],
    lab_list: List[List[int]],
    pad_token_id: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Right-pad a batch to its longest sequence: decoder_input_ids with
    pad_token_id, labels with -100. No decoder_attention_mask is needed — padding
    sits at the end and Whisper's decoder is causal, so it never affects the
    supervised (earlier) positions. Returns (padded_dec, padded_lab).
    """
    max_len = max((len(d) for d in dec_list), default=1)
    max_len = max(max_len, 1)
    dec_padded, lab_padded = [], []
    for di, lb in zip(dec_list, lab_list):
        pad_n = max_len - len(di)
        dec_padded.append(list(di) + [pad_token_id] * pad_n)
        lab_padded.append(list(lb) + [-100] * pad_n)
    return dec_padded, lab_padded
