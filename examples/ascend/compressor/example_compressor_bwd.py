"""Ascend compressor backward: dX, dWkv, dWgate and dApe.

The launch follows the CANN CompressorGrad pipeline. Vector cores apply the
Hadamard and softmax backward and scatter the result onto tokens; Cube cores
then evaluate the four projection GEMMs. See ``README.md`` for the layout and
the supported shapes.
"""

from __future__ import annotations

import argparse

import torch
import tilelang

from kernel import compressor_dw, compressor_dx, compressor_reduce_ape, compressor_scatter
from reference import backward_rounded, compress_forward, normalize_layout
from tilelang.profiler import do_bench

_CACHE = {}


def _compile(name, key, builder, out_idx):
    slot = (name, key)
    kernel = _CACHE.get(slot)
    if kernel is None:
        kernel = tilelang.compile(builder(), out_idx=out_idx, target="ascend")
        _CACHE[slot] = kernel
    return kernel


def _dtype_name(dtype):
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError("x, wkv, wgate and d_cmp must be bfloat16 or float16")


def _segment_lengths(flat, cu):
    batch = cu.numel() - 1
    return [int(cu[i + 1] - cu[i]) for i in range(batch)]


def build_meta(lengths, used, starts, cmp_ratio, coff, device):
    """Per-group routing: cur token, prev token, APE base, valid, zero the low half."""
    groups = sum(length // cmp_ratio for length in lengths)
    cur = torch.empty(groups, dtype=torch.int32, device=device)
    prev = torch.empty(groups, dtype=torch.int32, device=device)
    ape = torch.empty(groups, dtype=torch.int32, device=device)
    valid = torch.empty(groups, dtype=torch.int32, device=device)
    zero_lo = torch.empty(groups, dtype=torch.int32, device=device)
    base = 0
    offset = 0
    for length, n_used, start in zip(lengths, used, starts, strict=True):
        n_groups = length // cmp_ratio
        n_valid = n_used // cmp_ratio
        idx = torch.arange(n_groups, device=device, dtype=torch.int32)
        slot = slice(offset, offset + n_groups)
        is_valid = idx < n_valid
        cur[slot] = base + idx * cmp_ratio
        ape[slot] = start
        valid[slot] = is_valid.to(torch.int32)
        if coff == 2:
            prev_token = base + (idx - 1) * cmp_ratio
            prev[slot] = torch.where(is_valid & (idx > 0), prev_token, torch.full_like(prev_token, -1))
            zero_lo[slot] = ((idx == n_valid - 1) & is_valid).to(torch.int32)
        else:
            prev[slot] = -1
            zero_lo[slot] = 0
        base += length
        offset += n_groups
    return torch.stack((cur, prev, ape, valid, zero_lo), dim=1)


def _pad_rows(tensor, rows):
    if rows == 0:
        return tensor
    pad = tensor.new_zeros((rows, tensor.shape[-1]))
    return torch.cat((tensor, pad), dim=0)


def _resolve_cube_blocks(device, cube_blocks):
    if cube_blocks is None:
        cube_blocks = int(torch.npu.get_device_properties(device.index).cube_core_num)
    if cube_blocks <= 0:
        raise ValueError("cube_blocks must be positive")
    return cube_blocks


def compressor_bwd(
    x, wkv, wgate, d_cmp, softmax, kv, *, cmp_ratio, coff=1, seqused=None, start_pos=None, cu_seqlens=None, cube_blocks=None
):
    """Compute ``d_x``, ``d_wkv``, ``d_wgate`` and ``d_ape``.

    ``x`` is ``[B, S, H]`` or packed ``[T, H]`` with ``cu_seqlens``. ``wkv`` and
    ``wgate`` are ``[coff * D, H]``. ``d_cmp`` is ``[groups, D]``, and
    ``softmax`` / ``kv`` are ``[groups, coff * cmp_ratio, D]`` in segment order.
    ``H`` must be a multiple of 256. ``D`` is 128 or 512. Every segment length
    and ``seqused`` entry must be a multiple of ``cmp_ratio``.
    ``cube_blocks`` defaults to the number of AICs on the input device.
    """
    if coff not in (1, 2):
        raise ValueError(f"coff must be 1 or 2, got {coff}")
    if cmp_ratio < 2 or cmp_ratio > 128:
        raise ValueError(f"cmp_ratio must be in [2, 128], got {cmp_ratio}")
    dtype = _dtype_name(x.dtype)
    if wkv.dtype != x.dtype or wgate.dtype != x.dtype or d_cmp.dtype != x.dtype:
        raise ValueError("x, wkv, wgate and d_cmp must use the same dtype")
    flat, cu, used, starts = normalize_layout(x, seqused, start_pos, cu_seqlens)
    flat = flat.contiguous()
    tokens, hidden = flat.shape
    compress_dim = wkv.shape[0]
    head_dim = compress_dim // coff
    if head_dim not in (128, 512) or compress_dim != coff * head_dim:
        raise ValueError("D must be 128 or 512 and wkv/wgate must be [coff * D, H]")
    if hidden % 256 != 0 or hidden < 256:
        raise ValueError(f"H must be a positive multiple of 256, got {hidden}")
    if wkv.shape != (compress_dim, hidden) or wgate.shape != (compress_dim, hidden):
        raise ValueError("wkv and wgate must both be [coff * D, H]")
    lengths = _segment_lengths(flat, cu)
    for length, n_used in zip(lengths, used, strict=True):
        if length <= 0 or length % cmp_ratio != 0 or n_used % cmp_ratio != 0 or n_used < 0 or n_used > length:
            raise ValueError("each segment length and seqused must be a positive multiple of cmp_ratio and within the segment")
    if any(start < 0 for start in starts):
        raise ValueError("start_pos must be non-negative")
    groups = tokens // cmp_ratio
    window = coff * cmp_ratio
    if d_cmp.shape != (groups, head_dim):
        raise ValueError(f"d_cmp must be [{groups}, {head_dim}]")
    if softmax.shape != (groups, window, head_dim) or kv.shape != softmax.shape:
        raise ValueError(f"softmax and kv must be [{groups}, {window}, {head_dim}]")
    if softmax.dtype != torch.float32 or kv.dtype != torch.float32:
        raise ValueError("softmax and kv must be float32")

    device = flat.device
    cube_blocks = _resolve_cube_blocks(device, cube_blocks)
    wkv = wkv.contiguous()
    wgate = wgate.contiguous()
    d_cmp = d_cmp.contiguous()
    softmax = softmax.contiguous()
    kv = kv.contiguous()
    meta = build_meta(lengths, used, starts, cmp_ratio, coff, device)

    scatter = _compile(
        "scatter",
        (groups, tokens, head_dim, cmp_ratio, coff, dtype),
        lambda: compressor_scatter(groups, tokens, head_dim, cmp_ratio, coff, dtype),
        [4, 5, 6],
    )
    reduce = _compile(
        "reduce",
        (cmp_ratio, compress_dim),
        lambda: compressor_reduce_ape(cmp_ratio, compress_dim),
        -1,
    )
    d_new_kv, d_new_score, partial = scatter(d_cmp, softmax, kv, meta)
    d_ape = reduce(partial)

    pad = (256 - tokens % 256) % 256
    gemm_tokens = tokens + pad
    d_kv = _pad_rows(d_new_kv, pad)
    d_score = _pad_rows(d_new_score, pad)
    x_pad = _pad_rows(flat, pad)
    dx = _compile(
        "dx",
        (gemm_tokens, hidden, compress_dim, dtype, cube_blocks),
        lambda: compressor_dx(gemm_tokens, hidden, compress_dim, dtype, cube_blocks),
        -1,
    )
    dw = _compile(
        "dw",
        (gemm_tokens, hidden, compress_dim, dtype, cube_blocks),
        lambda: compressor_dw(gemm_tokens, hidden, compress_dim, dtype, cube_blocks),
        [3, 4],
    )
    d_x = dx(d_kv, d_score, wkv, wgate)[:tokens]
    d_wkv, d_wgate = dw(d_kv, d_score, x_pad)
    if x.ndim == 3:
        d_x = d_x.view(x.shape)
    return d_x, d_wkv, d_wgate, d_ape


def _make_inputs(batch, seq, hidden, head_dim, cmp_ratio, coff, dtype, device="cpu"):
    torch.manual_seed(0)
    x = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
    compress_dim = coff * head_dim
    scale = hidden**-0.5
    wkv = torch.randn(compress_dim, hidden, device=device, dtype=dtype) * scale
    wgate = torch.randn(compress_dim, hidden, device=device, dtype=dtype) * scale
    ape = torch.randn(cmp_ratio, compress_dim, device=device, dtype=torch.float32) * 0.1
    return x, wkv, wgate, ape


def run_correctness(
    batch=1,
    seq=128,
    hidden=512,
    head_dim=128,
    cmp_ratio=4,
    coff=2,
    dtype=torch.bfloat16,
    seqused=None,
    start_pos=None,
    packed=False,
    *,
    cube_blocks=None,
):
    x, wkv, wgate, ape = _make_inputs(batch, seq, hidden, head_dim, cmp_ratio, coff, dtype)
    cu = None
    if packed:
        x = x.reshape(batch * seq, hidden)
        cu = torch.arange(0, batch * seq + 1, seq, dtype=torch.int32)
    _, softmax, kv = compress_forward(x, wkv, wgate, ape, cmp_ratio, coff, seqused, start_pos, cu)
    groups = softmax.shape[0]
    d_cmp = torch.randn(groups, head_dim, dtype=dtype)
    if seqused is not None:
        lengths = [seq] * batch
        used = seqused if not torch.is_tensor(seqused) else seqused.tolist()
        offset = 0
        for length, n_used in zip(lengths, used, strict=True):
            n_groups = length // cmp_ratio
            n_valid = int(n_used) // cmp_ratio
            d_cmp[offset + n_valid : offset + n_groups] = 0
            offset += n_groups
    expected = backward_rounded(x, wkv, wgate, d_cmp, softmax, kv, cmp_ratio, coff, seqused, start_pos, cu)
    actual = compressor_bwd(
        x.npu(),
        wkv.npu(),
        wgate.npu(),
        d_cmp.npu(),
        softmax.npu(),
        kv.npu(),
        cmp_ratio=cmp_ratio,
        coff=coff,
        seqused=None if seqused is None else torch.tensor(seqused, dtype=torch.int32, device="npu"),
        start_pos=None if start_pos is None else torch.tensor(start_pos, dtype=torch.int32, device="npu"),
        cu_seqlens=None if cu is None else cu.npu(),
        cube_blocks=cube_blocks,
    )
    names = ("d_x", "d_wkv", "d_wgate", "d_ape")
    for name, got, ref in zip(names, actual, expected, strict=True):
        tol = 1e-3 if name == "d_ape" else 2e-2
        torch.testing.assert_close(got.cpu().float(), ref.float(), rtol=tol, atol=tol)
    replay = compressor_bwd(
        x.npu(),
        wkv.npu(),
        wgate.npu(),
        d_cmp.npu(),
        softmax.npu(),
        kv.npu(),
        cmp_ratio=cmp_ratio,
        coff=coff,
        seqused=None if seqused is None else torch.tensor(seqused, dtype=torch.int32, device="npu"),
        start_pos=None if start_pos is None else torch.tensor(start_pos, dtype=torch.int32, device="npu"),
        cu_seqlens=None if cu is None else cu.npu(),
        cube_blocks=cube_blocks,
    )
    for name, first, second in zip(names, actual, replay, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    print(f"compressor_bwd ({dtype}, coff={coff}, ratio={cmp_ratio}, D={head_dim}, S={seq}): correctness passed")


def run_regression_perf(batch=1, seq=8192, hidden=4096, head_dim=512, cmp_ratio=4, coff=2, *, cube_blocks=None):
    dtype = torch.bfloat16
    x, wkv, wgate, _ape = _make_inputs(batch, seq, hidden, head_dim, cmp_ratio, coff, dtype, device="npu")
    cube_blocks = _resolve_cube_blocks(x.device, cube_blocks)
    groups = batch * seq // cmp_ratio
    window = coff * cmp_ratio
    softmax = torch.softmax(torch.randn(groups, window, head_dim, device="npu"), dim=1)
    kv = torch.randn(groups, window, head_dim, device="npu")
    d_cmp = torch.randn(groups, head_dim, device="npu", dtype=dtype)
    def launch():
        return compressor_bwd(x, wkv, wgate, d_cmp, softmax, kv, cmp_ratio=cmp_ratio, coff=coff, cube_blocks=cube_blocks)

    # Compile and complete the first launch before entering the benchmark.
    launch()
    torch.npu.synchronize()
    latency = do_bench(launch, backend="msprof", _n_warmup=10, _n_repeat=20)
    tokens = batch * seq
    compress_dim = coff * head_dim
    flops = 8.0 * tokens * compress_dim * hidden
    tflops = flops / (latency * 1e9)
    print(
        f"compressor_bwd (T={tokens}, H={hidden}, D={head_dim}, coff={coff}, ratio={cmp_ratio}, cube_blocks={cube_blocks}): "
        f"{latency * 1000:.2f} us | {tflops:.1f} TFLOPS"
    )
    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True, help="benchmark after correctness checks")
    parser.add_argument("--cube-blocks", type=int, help="override the detected AIC count for comparison runs")
    args = parser.parse_args()
    run_correctness(cube_blocks=args.cube_blocks)
    run_correctness(seq=256, head_dim=512, coff=1, cmp_ratio=128, cube_blocks=args.cube_blocks)
    run_correctness(batch=2, seq=128, dtype=torch.float16, seqused=[64, 128], start_pos=[1, 0], packed=True, cube_blocks=args.cube_blocks)
    if args.bench:
        run_regression_perf(cube_blocks=args.cube_blocks)
