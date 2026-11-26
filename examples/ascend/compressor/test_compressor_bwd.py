"""Compressor backward: reference autograd check and Ascend kernel checks."""

import torch
import tilelang.testing

from reference import backward_math, compress_forward


def _autograd_case(x, wkv, wgate, ape, cmp_ratio, coff, seqused=None, start_pos=None, cu_seqlens=None):
    x = x.detach().requires_grad_(True)
    wkv = wkv.detach().requires_grad_(True)
    wgate = wgate.detach().requires_grad_(True)
    ape = ape.detach().requires_grad_(True)
    cmp, softmax, kv = compress_forward(x, wkv, wgate, ape, cmp_ratio, coff, seqused, start_pos, cu_seqlens)
    grad = torch.randn_like(cmp)
    (cmp * grad).sum().backward()
    d_x, d_wkv, d_wgate, d_ape, _, _ = backward_math(
        x.detach(),
        wkv.detach(),
        wgate.detach(),
        grad,
        softmax.detach(),
        kv.detach(),
        cmp_ratio,
        coff,
        seqused,
        start_pos,
        cu_seqlens,
    )
    torch.testing.assert_close(d_x.view_as(x), x.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(d_wkv, wkv.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(d_wgate, wgate.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(d_ape, ape.grad, rtol=1e-4, atol=1e-4)


def test_reference_matches_autograd():
    torch.manual_seed(0)
    hidden = 16
    cases = [
        dict(batch=2, seq=16, head_dim=8, cmp_ratio=4, coff=1),
        dict(batch=1, seq=16, head_dim=8, cmp_ratio=4, coff=2, start_pos=[1]),
        dict(batch=2, seq=16, head_dim=4, cmp_ratio=4, coff=2, seqused=[8, 16], start_pos=[2, 0]),
    ]
    for case in cases:
        batch, seq = case["batch"], case["seq"]
        head_dim, coff = case["head_dim"], case["coff"]
        compress_dim = coff * head_dim
        x = torch.randn(batch, seq, hidden)
        wkv = torch.randn(compress_dim, hidden) / hidden**0.5
        wgate = torch.randn(compress_dim, hidden) / hidden**0.5
        ape = torch.randn(case["cmp_ratio"], compress_dim) * 0.1
        _autograd_case(
            x,
            wkv,
            wgate,
            ape,
            case["cmp_ratio"],
            coff,
            case.get("seqused"),
            case.get("start_pos"),
        )


def test_reference_packed_segments_match_autograd():
    torch.manual_seed(1)
    hidden, head_dim, ratio, coff = 16, 8, 4, 2
    lengths = [8, 16]
    x = torch.randn(sum(lengths), hidden)
    cu = torch.tensor([0, 8, 24], dtype=torch.int32)
    compress_dim = coff * head_dim
    wkv = torch.randn(compress_dim, hidden) / hidden**0.5
    wgate = torch.randn(compress_dim, hidden) / hidden**0.5
    ape = torch.randn(ratio, compress_dim) * 0.1
    _autograd_case(x, wkv, wgate, ape, ratio, coff, seqused=[8, 8], start_pos=[0, 3], cu_seqlens=cu)


@tilelang.testing.requires_ascend
def test_compressor_bwd_kernel():
    from example_compressor_bwd import run_correctness

    run_correctness()
    run_correctness(seq=256, head_dim=512, coff=1, cmp_ratio=128)
    run_correctness(batch=2, seq=128, dtype=torch.float16, seqused=[64, 128], start_pos=[1, 0], packed=True)
    # dX has five M tiles and two N tiles, so the merged tail spans two windows.
    run_correctness(seq=1280, hidden=512)
