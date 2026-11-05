"""Independent FP32 gather/softmax reference; no TileLang or NPU imports.

Max and LSE exclude the sink. Empty rows use FlashMLA's compatibility
sentinels (zero output, -inf maximum, +inf LSE).
"""

import math

import torch


def sparse_mla_reference(q, kv, indices, attn_sink, lengths=None, *, scale, sink=True):
    if q.ndim != 3 or q.shape[1:] != (64, 576):
        raise ValueError("Q must have shape [nq, 64, 576]")
    if kv.ndim != 2 or kv.shape[1] != 576 or kv.shape[0] == 0:
        raise ValueError("KV must have nonempty shape [nk, 576]")
    if indices.ndim != 2 or indices.shape[0] != q.shape[0]:
        raise ValueError("Indices must have shape [nq, K]")
    if attn_sink.shape != (64,):
        raise ValueError("Sink must have shape [64]")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    nq, _, _ = q.shape
    topk = indices.shape[1]
    if lengths is not None and (lengths.shape != (nq,) or bool(((lengths < 0) | (lengths > topk)).any())):
        raise ValueError("Lengths must have shape [nq] and values in [0, K]")
    output = torch.zeros((nq, 64, 512), dtype=q.dtype, device=q.device)
    maximum = torch.full((nq, 64), -torch.inf, dtype=torch.float32, device=q.device)
    lse = torch.full_like(maximum, torch.inf)
    # Keep reference memory O(K*576), rather than materializing [nq,K,576].
    for t in range(nq):
        limit = int(lengths[t]) if lengths is not None else topk
        selected = indices[t, :limit].long()
        valid = (selected >= 0) & (selected < kv.shape[0])
        selected = selected[valid]
        if selected.numel() == 0:
            continue
        focused = kv[selected].float()
        scores = (q[t].float() @ focused.T) * scale
        maximum[t] = scores.max(dim=-1).values
        lse[t] = torch.logsumexp(scores, dim=-1)
        # A zero-value sink column gives a stable denominator even if its
        # logit is much larger than all data logits. Duplicates retain weight.
        # FlashMLA defines a +inf sink as zero output, rather than softmax's
        # undefined inf-inf case. -inf (including padded heads) is a no-op.
        positive_inf_sink = torch.isposinf(attn_sink) if sink else None
        safe_sink = attn_sink.float().masked_fill(positive_inf_sink, 0) if sink else None
        logits = torch.cat((scores, safe_sink[:, None]), dim=-1) if sink else scores
        probabilities = logits.softmax(dim=-1)[:, : selected.numel()]
        if sink:
            probabilities = probabilities.masked_fill(positive_inf_sink[:, None], 0)
        output[t] = (probabilities @ focused[:, :512]).to(q.dtype)
    return output, maximum, lse
