"""Hand-computable checks of the target's reference contract (CPU only)."""

import math

import pytest
import torch

from examples.ascend.hy4_mla.reference import sparse_mla_reference


def test_sink_duplicates_lengths_and_empty_sentinels():
    q = torch.zeros((3, 64, 576), dtype=torch.bfloat16)
    kv = torch.zeros((2, 576), dtype=torch.bfloat16)
    kv[0, :512], kv[1, :512] = 3, 9
    indices = torch.tensor([[0, 0, 1, -1, 2], [0, 1, 0, 0, 0], [-9, 2, -1, 99, -1]], dtype=torch.int32)
    lengths = torch.tensor([5, 1, 5], dtype=torch.int32)
    out, maximum, lse = sparse_mla_reference(q, kv, indices, torch.zeros(64), lengths, scale=1 / 16)
    # Duplicate 0 contributes twice: (3+3+9)/(3 valid slots + sink).
    torch.testing.assert_close(out[0], torch.full_like(out[0], 3.75), rtol=0, atol=0)
    torch.testing.assert_close(out[1], torch.full_like(out[1], 1.5), rtol=0, atol=0)
    assert torch.count_nonzero(out[2]) == 0
    assert (maximum[:2] == 0).all() and torch.isneginf(maximum[2]).all()
    torch.testing.assert_close(lse[0], torch.full_like(lse[0], math.log(3)))
    assert (lse[1] == 0).all() and torch.isposinf(lse[2]).all()


def test_rope_affects_scores_but_is_not_a_value():
    q = torch.zeros((1, 64, 576), dtype=torch.bfloat16)
    q[..., 512:] = 1
    kv = torch.zeros((2, 576), dtype=torch.bfloat16)
    kv[1, 512:] = 1
    kv[0, :512], kv[1, :512] = 2, 6
    indices = torch.tensor([[0, 1]], dtype=torch.int32)
    out, maximum, lse = sparse_mla_reference(q, kv, indices, torch.zeros(64), scale=1 / 16, sink=False)
    expected = (2 + 6 * math.exp(4)) / (1 + math.exp(4))
    torch.testing.assert_close(out, torch.full_like(out, expected), rtol=0, atol=0)
    torch.testing.assert_close(maximum, torch.full_like(maximum, 4))
    torch.testing.assert_close(lse, torch.full_like(lse, math.log1p(math.exp(4))))


def test_large_sink_is_stable_and_does_not_change_lse():
    q = torch.zeros((1, 64, 576), dtype=torch.bfloat16)
    kv = torch.ones((1, 576), dtype=torch.bfloat16)
    out, maximum, lse = sparse_mla_reference(q, kv, torch.zeros((1, 1), dtype=torch.int32), torch.full((64,), 1000.0), scale=1 / 16)
    assert (out == 0).all() and (maximum == 0).all() and (lse == 0).all()


@pytest.mark.parametrize("scale", [0, -1, math.inf, math.nan])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError, match="scale"):
        sparse_mla_reference(torch.zeros(1, 64, 576), torch.zeros(1, 576), torch.zeros(1, 1), torch.zeros(64), scale=scale)
