"""Device acceptance for the Hy4 656-byte sparse decode kernel."""

import pytest
import torch

from examples.ascend.hy4_mla.decode import flash_mla_with_kvcache
from examples.ascend.hy4_mla.decode_reference import fp8_sparse_decode_reference
from examples.ascend.hy4_mla.fp8_cache import pack_records, records_to_pages
from examples.ascend.hy4_mla.test_interface_npu import npu as npu


@pytest.mark.parametrize("batch,seqlen,topk,use_sink", [(1, 1, 64, False), (2, 4, 2048, True)])
def test_fp8_sparse_decode(npu, batch, seqlen, topk, use_sink):
    rng = torch.Generator().manual_seed(2026)
    tokens = 128
    q = (torch.randn(batch, seqlen, 64, 576, generator=rng) * 0.3).bfloat16()
    latent = torch.randn(tokens, 512, generator=rng).to(torch.float8_e4m3fn)
    scales = (0.1 + torch.rand(tokens, 4, generator=rng)) * torch.tensor([0.15, 0.3, 0.45, 0.6])
    rope = (torch.randn(tokens, 64, generator=rng) * 0.4).bfloat16()
    cache = records_to_pages(pack_records(latent, scales, rope))
    indices = torch.randint(tokens, (batch, seqlen, topk), generator=rng, dtype=torch.int32)
    indices[..., :8] = torch.tensor([64, 63, 64, -1, 127, 128, 0, 0], dtype=torch.int32)
    if use_sink:
        indices[-1, -1].fill_(-1)
    sink = torch.linspace(-3, 3, 64) if use_sink else None
    if sink is not None:
        sink[0], sink[-1] = -torch.inf, torch.inf

    scale = 1 / 16
    expected = fp8_sparse_decode_reference(q, cache, indices, sink, scale=scale)
    inputs = tuple(t.to(npu) for t in (q, cache, indices))
    sink_npu = None if sink is None else sink.to(npu)
    for _ in range(2):
        output, lse = flash_mla_with_kvcache(*inputs, softmax_scale=scale, attn_sink=sink_npu)
        torch.npu.synchronize()
        assert output.is_contiguous() and lse.is_contiguous()
        torch.testing.assert_close(output.cpu(), expected[0], rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse.cpu(), expected[1], rtol=2e-4, atol=2e-4)
        if use_sink:
            assert torch.count_nonzero(output[-1, -1]).item() == 0
            assert torch.isposinf(lse[-1, :, -1]).all().item()
