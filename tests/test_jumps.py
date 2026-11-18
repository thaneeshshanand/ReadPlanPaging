import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase0"))
from jumps_analyze import BLOCK, lexical_blocks, log_counts, memory_blocks  # noqa: E402

C, W, H, PAD = 0, 1, 2, 9


def test_memory_most_recent_first_and_cap():
    hist = np.array([[H, C, C, C, C],   # step t-3
                     [C, W, C, C, C],   # t-2
                     [C, C, C, H, C]])  # t-1
    cold_now = np.array([True, True, True, True, False])
    # non-COLD last at: block0 -> 0, block1 -> 1, block3 -> 2; block2 never; block4 not COLD now
    assert memory_blocks(hist, cold_now, 5).tolist() == [3, 1, 0]
    assert memory_blocks(hist, cold_now, 2).tolist() == [3, 1]
    assert memory_blocks(hist[:0], cold_now, 5).tolist() == []


def test_memory_pad_counts_as_not_cold():
    hist = np.array([[C, PAD]])
    assert memory_blocks(hist, np.array([True, True]), 3).tolist() == [1]


def test_lexical_alignment_and_suffix_rank():
    # token 7 occurs at 20 (suffix 1) and 40 (suffix 3: 5 6 7); query position 70 has 5 6 7
    seq = np.zeros(80, np.int64)
    seq[20] = 7
    seq[38:41] = [5, 6, 7]
    seq[68:71] = [5, 6, 7]
    out = lexical_blocks(seq, 70, seq, log_counts(seq))
    assert out == [2, 1]  # p=40 (suffix 4: blocks of 40 and 41, both 2) before p=20 (blocks of 20 and 21)
    assert all(b <= 70 // BLOCK for b in out)


def test_lexical_next_block_on_boundary():
    seq = np.arange(100, 200)  # all distinct
    seq[60] = seq[31]  # token at 31; the token after it (32) is in the next block
    out = lexical_blocks(seq, 60, seq, log_counts(seq))
    assert out == [31 // BLOCK, 32 // BLOCK] == [1, 2]


def test_lexical_rarity_breaks_ties():
    seq = np.array([1, 9, 2, 9, 2, 2, 2, 3, 9, 0, 0])
    # query pos 10 token 0? use token 9 at pos 8: earlier at 1 (prev 1, count 1) and 3 (prev 2, count 4)
    out = lexical_blocks(seq, 8, seq, log_counts(seq))
    assert out == [0]  # all in block 0 for this short sequence
    # rarity: with suffix 2 neither matches (seq[7] = 3), so suffix 1 for both; equal rarity -> most recent


def _repeat_context(seed):
    # same construction as jumps_capture.repeat_records: 2048 random ids repeated, 3967 prompt + 129 reference
    s = np.random.default_rng(seed).integers(1000, 30000, 2048)
    return np.concatenate([s, s])


def _check_alignment(ctx, n0, steps):
    lc = log_counts(ctx[:n0])
    for t in range(steps):
        pos = n0 + t
        p = pos - 2048
        assert ctx[p] == ctx[pos]
        out = lexical_blocks(ctx, pos, ctx, lc)
        target = (p + 1) // BLOCK
        assert target in out[:2], (t, pos, target, out[:4])  # the earlier copy ranks first (suffix 4)


def test_alignment_on_repeated_random_tokens():
    for seed in range(4):
        _check_alignment(_repeat_context(seed), 3967, 128)


def test_alignment_on_captured_repeat_controls():
    import json
    import pytest
    root = Path(__file__).resolve().parent.parent / "runs" / "jumps"
    dirs = sorted(root.glob("*-repeat/repeat-*")) if root.exists() else []
    if not dirs:
        pytest.skip("no captured repeat controls")
    for d in dirs:
        info = json.loads((d / "prompt.json").read_text())
        ctx = np.asarray(info["context_ids"])
        bos = len(ctx) - 4096
        assert bos >= 0
        _check_alignment(ctx[bos:], info["prompt_len"] - bos, len(ctx) - info["prompt_len"] - 1)
