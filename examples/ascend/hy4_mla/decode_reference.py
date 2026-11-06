"""FP32 reference for the Hy4 656-byte FP8 sparse decode contract.

Dequantized latent values are rounded to BF16 before the attention math, matching
the first kernel version's Cube inputs. RoPE stays in its stored BF16 values.
The returned LSE follows FlashMLA decode layout ``[batch, heads, seq]`` and
excludes the sink. Empty rows return a zero output and +inf LSE.
"""

import math

import torch

from examples.ascend.hy4_mla.fp8_cache import physical_rows
from examples.ascend.hy4_mla.reference import sparse_mla_reference


def fp8_sparse_decode_reference(q, k_cache, indices, attn_sink=None, *, scale):
    """Return ``(output, lse)`` for one paged FP8 sparse decode call.

    ``q`` is ``[B, Sq, 64, 576]`` BF16. ``k_cache`` is ``[P, 64, 1, 656]``
    uint8. ``indices`` is ``[B, Sq, K]`` int32, addressing
    ``page_id * 64 + offset``. ``attn_sink`` is ``[64]`` FP32 or None.
    """
    if q.ndim != 4 or q.shape[2:] != (64, 576) or q.dtype != torch.bfloat16:
        raise ValueError("q must have shape [B, Sq, 64, 576] and dtype bfloat16")
    if indices.ndim != 3 or indices.shape[:2] != q.shape[:2] or indices.dtype != torch.int32:
        raise ValueError("indices must have shape [B, Sq, K] and dtype int32")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    batch, seqlen, _, _ = q.shape
    rows = physical_rows(k_cache)
    flat_q = q.reshape(batch * seqlen, 64, 576)
    flat_indices = indices.reshape(batch * seqlen, indices.shape[-1])
    sink = torch.zeros(64, dtype=torch.float32, device=q.device) if attn_sink is None else attn_sink
    output, _, lse = sparse_mla_reference(flat_q, rows, flat_indices, sink, scale=scale, sink=attn_sink is not None)
    output = output.view(batch, seqlen, 64, 512)
    lse = lse.view(batch, seqlen, 64).transpose(1, 2).contiguous()
    return output, lse
