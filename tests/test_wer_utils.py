"""
Unit tests for the WER quality gate + content-hash primitives (wer_utils.py).

This is the accept/reject decision that determines which pseudo-labels become
training data, plus the sample-ID hashing that resume/dedup rely on — the most
important pure logic in stage 2, and not exercisable via a GPU run on Windows.

    python -m pytest tests/test_wer_utils.py -q
"""
import sys
import pathlib

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from wer_utils import (  # noqa: E402
    are_spelling_variants,
    calculate_wer_spelling_tolerant,
    is_all_caps_hallucination,
    strip_bracketed_fillers,
    drop_standalone_fillers,
    content_hash,
)


# --- spelling variants -----------------------------------------------------
def test_spelling_variants_true():
    assert are_spelling_variants("colour", "color")
    assert are_spelling_variants("realise", "realize")
    assert are_spelling_variants("behaviour", "behavior")
    assert are_spelling_variants("word", "word")  # identical


def test_spelling_variants_false():
    assert not are_spelling_variants("cat", "dog")          # unrelated
    assert not are_spelling_variants("colour", "flavour")   # different first letter
    assert not are_spelling_variants("", "x")               # empty
    assert not are_spelling_variants("a", "")               # empty


# --- WER (spelling-tolerant Levenshtein) -----------------------------------
def test_wer_identical_is_zero():
    assert calculate_wer_spelling_tolerant(["the", "cat", "sat"], ["the", "cat", "sat"]) == 0.0


def test_wer_substitution():
    # 1 substitution / 2 ref words
    assert calculate_wer_spelling_tolerant(["the", "cat"], ["the", "dog"]) == pytest.approx(0.5)


def test_wer_deletion():
    # 1 deletion / 3 ref words
    assert calculate_wer_spelling_tolerant(["a", "b", "c"], ["a", "c"]) == pytest.approx(1 / 3)


def test_wer_insertion():
    # 1 insertion / 2 ref words
    assert calculate_wer_spelling_tolerant(["a", "c"], ["a", "b", "c"]) == pytest.approx(0.5)


def test_wer_spelling_variant_is_free():
    # colour vs color must NOT count as an error
    assert calculate_wer_spelling_tolerant(["the", "colour"], ["the", "color"]) == 0.0


def test_wer_edge_cases():
    assert calculate_wer_spelling_tolerant([], ["x"]) == 1.0   # empty ref, non-empty hyp
    assert calculate_wer_spelling_tolerant([], []) == 0.0      # both empty
    assert calculate_wer_spelling_tolerant(["x"], []) == 1.0   # empty hyp


# --- all-caps hallucination guard ------------------------------------------
def test_all_caps():
    assert is_all_caps_hallucination("HELLO WORLD")
    assert not is_all_caps_hallucination("Hello World")
    assert not is_all_caps_hallucination("hello world")
    assert not is_all_caps_hallucination("")          # empty -> not flagged
    assert not is_all_caps_hallucination(None)        # None -> not flagged


# --- filler stripping ------------------------------------------------------
def test_strip_bracketed_fillers():
    assert "[" not in strip_bracketed_fillers("[Um] hello [Uh] world")
    # case-insensitive, only the listed fillers
    assert strip_bracketed_fillers("[UM]hi[uh]").strip() == "hi"


def test_drop_standalone_fillers():
    assert drop_standalone_fillers("um hello uh world mm") == "hello world"
    assert drop_standalone_fillers("the quick brown fox") == "the quick brown fox"


# --- content hash (resume/dedup identity) ----------------------------------
def test_content_hash():
    np = pytest.importorskip("numpy")
    a = np.zeros(8000, dtype=np.float32)
    b = np.ones(8000, dtype=np.float32)

    id1 = content_hash(a, "hello world", "librispeech")
    id2 = content_hash(a, "hello world", "librispeech")
    assert id1 == id2                              # deterministic
    assert id1.startswith("librispeech_")          # dataset namespace prefix
    assert len(id1.split("_", 1)[1]) == 16         # 16-char hex (DO NOT change)

    # any component change -> different id
    assert content_hash(a, "hello world", "ami") != id1          # dataset
    assert content_hash(a, "different text", "librispeech") != id1  # text
    assert content_hash(b, "hello world", "librispeech") != id1   # audio
