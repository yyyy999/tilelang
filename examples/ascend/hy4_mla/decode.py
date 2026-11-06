"""TileLang Ascend entry point for the Hy4 FP8 sparse decode subset."""

import math
from functools import lru_cache

import torch

from examples.ascend.hy4_mla.fp8_cache import PAGE_SIZE, RECORD_BYTES
from examples.ascend.hy4_mla.interface import _check_tensor


@lru_cache(maxsize=32)
def _get_decode_kernel(nq, nk, topk, scale, sink, device):
    import tilelang
    from tilelang.carver.arch.driver.ascend_driver import get_num_cube_cores

    from examples.ascend.hy4_mla.kernel import hy4_sparse_attention

    with torch.npu.device(device):
        num_cores = int(get_num_cube_cores(torch.npu.current_device()))
        program = hy4_sparse_attention(nq, nk, topk, scale=scale, sink=sink, num_cores=num_cores, kv_format="fp8_ds_mla")
        return tilelang.compile(
            program,
            target="ascend",
            out_idx=[5, 6],
            compile_flags=["-Ofast", "-mllvm", "-enable-hiipu-vf-loop-unroll"],
        )


def flash_mla_with_kvcache(q, k_cache, indices, *, softmax_scale, attn_sink=None):
    """Return BF16 output [B,Sq,64,512] and FP32 LSE [B,64,Sq].

    Accepts contiguous BF16 Q [B,Sq,64,576], uint8 KV [P,64,1,656],
    and int32 indices [B,Sq,K]. Indices are physical slots and K is a
    positive multiple of 64. LSE excludes the optional FP32 [64] sink.
    """
    if not isinstance(q, torch.Tensor) or q.ndim != 4:
        raise ValueError("q must have shape [B,Sq,64,576]")
    if not isinstance(k_cache, torch.Tensor) or k_cache.ndim != 4:
        raise ValueError("k_cache must have shape [P,64,1,656]")
    if not isinstance(indices, torch.Tensor) or indices.ndim != 3:
        raise ValueError("indices must have shape [B,Sq,K]")
    batch, seqlen = q.shape[:2]
    pages, topk = k_cache.shape[0], indices.shape[-1]
    if min(batch, seqlen, pages, topk) <= 0 or topk % 64:
        raise ValueError("B/Sq/P must be positive and K must be a positive multiple of 64")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("softmax_scale must be positive and finite")
    device = q.device
    inputs = [
        (q, "q", (batch, seqlen, 64, 576), torch.bfloat16),
        (k_cache, "k_cache", (pages, PAGE_SIZE, 1, RECORD_BYTES), torch.uint8),
        (indices, "indices", (batch, seqlen, topk), torch.int32),
    ]
    if attn_sink is not None:
        inputs.append((attn_sink, "attn_sink", (64,), torch.float32))
    for tensor, name, shape, dtype in inputs:
        _check_tensor(tensor, name, shape, dtype, device)
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if device.type != "npu":
        raise ValueError("Hy4 sparse decode requires an Ascend NPU")
    nq, nk = batch * seqlen, pages * PAGE_SIZE
    kernel = _get_decode_kernel(nq, nk, topk, float(softmax_scale), attn_sink is not None, device)
    sink = torch.empty(64, dtype=torch.float32, device=device) if attn_sink is None else attn_sink
    lengths = torch.empty(nq, dtype=torch.int32, device=device)
    with torch.npu.device(device):
        output, lse = kernel(q.view(nq, 64, 576), k_cache.view(nk, RECORD_BYTES), indices.view(nq, topk), sink, lengths)
    return output.view(batch, seqlen, 64, 512), lse.view(batch, seqlen, 64).transpose(1, 2).contiguous()
