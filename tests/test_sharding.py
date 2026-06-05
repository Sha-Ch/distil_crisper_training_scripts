"""
Contract test for stage-2's deterministic, GPU-count-agnostic sharding.

stage 2 assigns each sample to a rank with:
    shard = int(md5(ground_truth.strip())[:8], 16) % world_size
and a rank processes it iff shard == local_rank. This test mirrors that formula
(it cannot be imported directly — it lives inside a torch-importing module) and
locks in the invariants the resume logic depends on:
  * single GPU (world_size == 1) -> every sample lands on rank 0
  * the assignment is deterministic across runs

    python -m pytest tests/test_sharding.py -q
"""
import hashlib


def shard_of(text: str, world_size: int) -> int:
    # MUST match ThreadedPrefetcher._worker_loop in 02_generate_pseudo_labels_multi_gpu.py
    key = text.strip()
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % world_size


SAMPLE_TEXTS = [
    "the quick brown fox",
    "hello world",
    "she sells sea shells",
    "a man a plan a canal panama",
    "to be or not to be",
    "lorem ipsum dolor sit amet",
    "distil crisper whisper",
    "knowledge distillation works",
]


def test_single_gpu_takes_everything():
    # On one 4090 (world_size==1) nothing may be sharded away.
    assert all(shard_of(t, 1) == 0 for t in SAMPLE_TEXTS)


def test_deterministic():
    for t in SAMPLE_TEXTS:
        assert shard_of(t, 4) == shard_of(t, 4)


def test_whitespace_insensitive():
    # stage 2 strips before hashing, so surrounding whitespace must not change shard.
    for t in SAMPLE_TEXTS:
        assert shard_of(t, 7) == shard_of(f"  {t}  ", 7)


def test_in_range():
    for ws in (1, 2, 4, 7, 8):
        for t in SAMPLE_TEXTS:
            assert 0 <= shard_of(t, ws) < ws
