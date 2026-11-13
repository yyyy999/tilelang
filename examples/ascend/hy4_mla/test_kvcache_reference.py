"""CPU checks for the 656-byte record and the FP8 decode reference."""

import math

import torch

from examples.ascend.hy4_mla.kvcache_reference import fp8_sparse_kvcache_reference
from examples.ascend.hy4_mla.fp8_cache import PAGE_SIZE, pack_records, physical_rows, records_to_pages, unpack_records


def _record(latent, scales, rope):
    fp8 = torch.tensor(latent, dtype=torch.float32).to(torch.float8_e4m3fn)
    return pack_records(fp8.view(1, -1), torch.tensor(scales, dtype=torch.float32).view(1, 4), torch.tensor(rope, dtype=torch.bfloat16).view(1, 64))


def test_four_scales_round_to_bf16_independently():
    latent = [1.0] * 512
    scales = [1.0, 2.0, 4.0, 8.0]
    raw = _record(latent, scales, [0.0] * 64)
    values, rope = unpack_records(raw)
    for group, scale in enumerate(scales):
        piece = values[0, group * 128 : (group + 1) * 128]
        torch.testing.assert_close(piece, torch.full_like(piece, scale))
    assert torch.count_nonzero(rope) == 0


def test_rope_bytes_round_trip_without_scaling():
    rope = torch.arange(64, dtype=torch.bfloat16)
    raw = pack_records(
        torch.zeros(1, 512, dtype=torch.float8_e4m3fn),
        torch.ones(1, 4, dtype=torch.float32),
        rope.view(1, 64),
    )
    _, restored = unpack_records(raw)
    assert torch.equal(restored[0], rope)


def test_cross_page_slot_selects_the_addressed_row():
    pages = 2
    tokens = pages * PAGE_SIZE
    latent = torch.zeros(tokens, 512, dtype=torch.float8_e4m3fn)
    scales = torch.ones(tokens, 4, dtype=torch.float32)
    rope = torch.zeros(tokens, 64, dtype=torch.bfloat16)
    slot = PAGE_SIZE + 3
    latent[slot] = torch.full((512,), 2.0, dtype=torch.float32).to(torch.float8_e4m3fn)
    rope[slot] = 1
    cache = records_to_pages(pack_records(latent, scales, rope))
    rows = physical_rows(cache)
    torch.testing.assert_close(rows[slot, :512].float(), torch.full((512,), 2.0))
    assert torch.all(rows[0] == 0)
    assert torch.all(rows[slot, 512:] == 1)


def test_duplicate_slots_both_contribute():
    latent = torch.zeros(PAGE_SIZE, 512, dtype=torch.float8_e4m3fn)
    latent[1] = torch.full((512,), 3.0, dtype=torch.float32).to(torch.float8_e4m3fn)
    cache = records_to_pages(
        pack_records(latent, torch.ones(PAGE_SIZE, 4, dtype=torch.float32), torch.zeros(PAGE_SIZE, 64, dtype=torch.bfloat16))
    )
    q = torch.zeros(1, 1, 64, 576, dtype=torch.bfloat16)
    indices = torch.tensor([[[1, 1]]], dtype=torch.int32)
    output, lse = fp8_sparse_kvcache_reference(q, cache, indices, None, scale=1 / 16)
    # Two copies of the same row, zero query scores: the value is unchanged and the LSE counts both copies.
    torch.testing.assert_close(output, torch.full_like(output, 3.0))
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(2)))
    assert lse.shape == (1, 64, 1)


def test_invalid_rows_and_positive_inf_sink():
    cache = records_to_pages(
        pack_records(
            torch.ones(PAGE_SIZE, 512, dtype=torch.float8_e4m3fn),
            torch.ones(PAGE_SIZE, 4, dtype=torch.float32),
            torch.zeros(PAGE_SIZE, 64, dtype=torch.bfloat16),
        )
    )
    q = torch.zeros(1, 2, 64, 576, dtype=torch.bfloat16)
    indices = torch.tensor([[[-1, PAGE_SIZE], [0, -1]]], dtype=torch.int32)
    output, lse = fp8_sparse_kvcache_reference(q, cache, indices, None, scale=1 / 16)
    assert torch.count_nonzero(output[0, 0]) == 0
    torch.testing.assert_close(output[0, 1], torch.ones_like(output[0, 1]))
    assert torch.isposinf(lse[0, :, 0]).all()
    torch.testing.assert_close(lse[0, :, 1], torch.zeros(64))

    sink = torch.full((64,), float("inf"))
    sunk, sunk_lse = fp8_sparse_kvcache_reference(q[:, 1:], cache, indices[:, 1:], sink, scale=1 / 16)
    assert torch.count_nonzero(sunk) == 0
    torch.testing.assert_close(sunk_lse, torch.zeros_like(sunk_lse))
