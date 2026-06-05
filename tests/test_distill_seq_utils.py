"""
Unit tests for the torch-free distillation sequence helpers.

These cover the trickiest part of the trainer's data path (teacher-forcing
alignment + prompt masking + truncation), which cannot be exercised on this
machine via a real GPU run. Pure Python — runs on Windows and in the container:

    python -m pytest tests/test_distill_seq_utils.py -q
"""
import sys
import pathlib

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from distill_seq_utils import build_decoder_inputs_and_labels, pad_decoder_batch  # noqa: E402


# Stand-in token ids (values are irrelevant to the logic under test).
SOT, LANG, TASK, EOT = 50258, 50259, 50360, 50257
SOP = 50361  # <|startofprev|>


def test_no_prompt_supervises_full_transcript():
    label_ids = [SOT, LANG, TASK, 100, 101, 102, EOT]
    dec, lab = build_decoder_inputs_and_labels([], label_ids)
    assert dec == label_ids[:-1]
    assert lab == label_ids[1:]
    assert len(dec) == len(lab)
    assert -100 not in lab  # nothing masked when there is no prompt


def test_prompt_region_is_masked():
    prompt = [SOP, 7, 8, 9]            # 4 prompt tokens
    label_ids = [SOT, LANG, TASK, 100, 101, EOT]
    dec, lab = build_decoder_inputs_and_labels(prompt, label_ids)

    full = prompt + label_ids
    assert dec == full[:-1]
    assert lab[:3] == [-100, -100, -100]          # mask len(prompt)-1 positions
    assert lab.count(-100) == len(prompt) - 1
    # First supervised target is the first transcript token (sot), predicted from
    # the last prompt token.
    assert lab[3] == SOT
    assert len(dec) == len(lab)


def test_lengths_always_equal():
    cases = [
        ([], [SOT, LANG, TASK, 1, EOT]),
        ([SOP, 1, 2], [SOT, LANG, TASK, 1, 2, 3, EOT]),
        ([SOP], [SOT, EOT]),
    ]
    for prompt, label_ids in cases:
        dec, lab = build_decoder_inputs_and_labels(prompt, label_ids)
        assert len(dec) == len(lab)


def test_truncation_drops_prompt_from_left_first():
    # Long prompt + short transcript, tight max_length -> prompt is dropped,
    # transcript preserved intact, nothing left to mask.
    prompt = [SOP, 1, 2, 3, 4, 5, 6, 7]
    label_ids = [SOT, LANG, TASK, 100, EOT]   # 5 tokens
    dec, lab = build_decoder_inputs_and_labels(prompt, label_ids, max_length=4)
    assert len(dec) == 4
    assert len(lab) == 4
    # Transcript tokens survive (last token of labels is the eot target).
    assert lab[-1] == EOT
    assert -100 not in lab


def test_truncation_when_transcript_alone_too_long():
    label_ids = [SOT, LANG, TASK, 100, 101, 102, 103, EOT]  # 8 tokens
    dec, lab = build_decoder_inputs_and_labels([], label_ids, max_length=3)
    assert len(dec) == 3
    assert len(lab) == 3


def test_pad_decoder_batch():
    dec_list = [[1, 2], [1, 2, 3, 4]]
    lab_list = [[5, 6], [5, 6, 7, 8]]
    dec_p, lab_p = pad_decoder_batch(dec_list, lab_list, pad_token_id=99)
    assert dec_p == [[1, 2, 99, 99], [1, 2, 3, 4]]
    assert lab_p == [[5, 6, -100, -100], [5, 6, 7, 8]]
    # decoder_input_ids pad with pad_token_id; labels pad with -100
    for d, l in zip(dec_p, lab_p):
        assert len(d) == len(l) == 4


def test_pad_decoder_batch_uniform_lengths():
    dec_list = [[1, 2, 3], [4, 5, 6]]
    lab_list = [[1, 2, 3], [4, 5, 6]]
    dec_p, lab_p = pad_decoder_batch(dec_list, lab_list, pad_token_id=0)
    assert dec_p == dec_list
    assert lab_p == lab_list
