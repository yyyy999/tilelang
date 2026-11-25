"""PyTorch reference for the compressor backward.

The forward here is only the compression stage that CompressorGrad
differentiates: the two projections, optional overlap regrouping, APE, grouped
softmax, Hadamard product and the reduction over the compression axis. RMSNorm
and RoPE stay outside this operator, matching ``aclnnCompressorGrad``.

``kv`` is the windowed ``kv_state`` that the forward multiplies by the softmax,
not the product itself. The CPU golden consumes it as
``d_score = softmax * (dC * kv - sum(softmax * dC * kv))``. With ``coff=2`` the
window is interleaved: even rows are the previous group's first-half projection
and odd rows are this group's second-half projection.
"""

from __future__ import annotations

import torch


def _as_int_list(value, batch, name, device):
    """Accept a length-``batch`` tensor or Python sequence of segment integers."""
    if torch.is_tensor(value):
        tensor = value.to(device=device, dtype=torch.int32).reshape(-1)
    else:
        tensor = torch.tensor(list(value), device=device, dtype=torch.int32)
    if tensor.numel() != batch:
        raise ValueError(f"{name} must have {batch} elements")
    return [int(v) for v in tensor.tolist()]


def normalize_layout(x, seqused=None, start_pos=None, cu_seqlens=None):
    """Return flat ``[T, H]`` tokens plus per-segment metadata on ``x``'s device."""
    if x.ndim == 3:
        if cu_seqlens is not None:
            raise ValueError("cu_seqlens is only used with packed [T, H] input")
        batch, seq, _ = x.shape
        flat = x.reshape(batch * seq, x.shape[-1])
        cu = torch.arange(0, batch * seq + 1, seq, device=x.device, dtype=torch.int32)
    elif x.ndim == 2:
        if cu_seqlens is None:
            raise ValueError("packed [T, H] input requires cu_seqlens")
        flat = x
        cu = cu_seqlens.to(device=x.device, dtype=torch.int32)
        batch = cu.numel() - 1
    else:
        raise ValueError(f"x must be [B, S, H] or [T, H], got ndim={x.ndim}")

    if int(cu[0]) != 0 or int(cu[-1]) != flat.shape[0]:
        raise ValueError("cu_seqlens must start at 0 and end at T")
    lengths = [int(cu[b + 1] - cu[b]) for b in range(batch)]
    used = lengths if seqused is None else _as_int_list(seqused, batch, "seqused", x.device)
    starts = [0] * batch if start_pos is None else _as_int_list(start_pos, batch, "start_pos", x.device)
    return flat, cu, used, starts


def _check_segments(lengths, used, starts, cmp_ratio):
    if cmp_ratio < 2 or cmp_ratio > 128:
        raise ValueError(f"cmp_ratio must be in [2, 128], got {cmp_ratio}")
    for length, n_used, start in zip(lengths, used, starts, strict=True):
        if length % cmp_ratio != 0 or n_used % cmp_ratio != 0:
            raise ValueError("segment length and seqused must be multiples of cmp_ratio")
        if n_used < 0 or n_used > length:
            raise ValueError(f"seqused {n_used} is outside segment length {length}")
        if start < 0:
            raise ValueError("start_pos must be non-negative")


def compress_forward(x, wkv, wgate, ape, cmp_ratio, coff=1, seqused=None, start_pos=None, cu_seqlens=None):
    """Compression-stage forward. Returns ``cmp``, ``softmax`` and windowed ``kv``.

    ``cmp`` is ``[groups, D]`` in segment order, including inactive padding
    groups (those are exact zeros and do not depend on ``x``). ``softmax`` and
    ``kv`` are ``[groups, coff * cmp_ratio, D]``. ``kv`` stores the values that
    enter the Hadamard product.
    """
    if coff not in (1, 2):
        raise ValueError(f"coff must be 1 or 2, got {coff}")
    flat, cu, used, starts = normalize_layout(x, seqused, start_pos, cu_seqlens)
    hidden = flat.shape[-1]
    compress_dim = wkv.shape[0]
    head_dim = compress_dim // coff
    if wkv.shape != (compress_dim, hidden) or wgate.shape != (compress_dim, hidden):
        raise ValueError("wkv and wgate must both be [coff * D, H]")
    if ape.shape != (cmp_ratio, compress_dim):
        raise ValueError(f"ape must be [{cmp_ratio}, {compress_dim}]")
    lengths = [int(cu[b + 1] - cu[b]) for b in range(cu.numel() - 1)]
    _check_segments(lengths, used, starts, cmp_ratio)

    kv_state = flat.float() @ wkv.float().T
    score_state = flat.float() @ wgate.float().T
    window = coff * cmp_ratio
    softmax_rows = []
    kv_rows = []
    cmp_rows = []
    for batch, length, n_used, start in zip(range(len(lengths)), lengths, used, starts, strict=True):
        base = int(cu[batch])
        n_groups = length // cmp_ratio
        n_valid = n_used // cmp_ratio
        for group in range(n_groups):
            if group >= n_valid:
                softmax_rows.append(kv_state.new_zeros(window, head_dim))
                kv_rows.append(kv_state.new_zeros(window, head_dim))
                cmp_rows.append(kv_state.new_zeros(head_dim))
                continue
            scores = []
            values = []
            for row in range(cmp_ratio):
                ape_row = (start + row) % cmp_ratio
                if coff == 1:
                    token = base + group * cmp_ratio + row
                    scores.append(score_state[token] + ape[ape_row])
                    values.append(kv_state[token])
                else:
                    prev = base + (group - 1) * cmp_ratio + row
                    if group == 0:
                        scores.append(score_state.new_full((head_dim,), float("-inf")))
                        values.append(score_state.new_zeros(head_dim))
                    else:
                        scores.append(score_state[prev, :head_dim] + ape[ape_row, :head_dim])
                        values.append(kv_state[prev, :head_dim])
                    cur = base + group * cmp_ratio + row
                    scores.append(score_state[cur, head_dim:] + ape[ape_row, head_dim:])
                    values.append(kv_state[cur, head_dim:])
            score_w = torch.stack(scores, dim=0)
            value_w = torch.stack(values, dim=0)
            weights = torch.softmax(score_w, dim=0)
            softmax_rows.append(weights)
            kv_rows.append(value_w)
            cmp_rows.append((weights * value_w).sum(dim=0))
    return torch.stack(cmp_rows), torch.stack(softmax_rows), torch.stack(kv_rows)


def backward_math(x, wkv, wgate, d_cmp, softmax, kv, cmp_ratio, coff=1, seqused=None, start_pos=None, cu_seqlens=None):
    """FP32 backward of the compression stage, before the Cube bf16 rounding."""
    if coff not in (1, 2):
        raise ValueError(f"coff must be 1 or 2, got {coff}")
    flat, cu, used, starts = normalize_layout(x, seqused, start_pos, cu_seqlens)
    hidden = flat.shape[-1]
    compress_dim = wkv.shape[0]
    head_dim = compress_dim // coff
    lengths = [int(cu[b + 1] - cu[b]) for b in range(cu.numel() - 1)]
    _check_segments(lengths, used, starts, cmp_ratio)
    n_groups = sum(length // cmp_ratio for length in lengths)
    window = coff * cmp_ratio
    if d_cmp.shape != (n_groups, head_dim):
        raise ValueError(f"d_cmp must be [{n_groups}, {head_dim}]")
    if softmax.shape != (n_groups, window, head_dim) or kv.shape != softmax.shape:
        raise ValueError(f"softmax and kv must be [{n_groups}, {window}, {head_dim}]")

    grad = d_cmp.float().unsqueeze(1)
    weights = softmax.float()
    values = kv.float()
    d_kv = grad * weights
    d_score_weighted = grad * values
    reduce = (weights * d_score_weighted).sum(dim=1, keepdim=True)
    d_z = weights * (d_score_weighted - reduce)

    d_new_kv = flat.new_zeros(flat.shape[0], compress_dim, dtype=torch.float32)
    d_new_score = flat.new_zeros(flat.shape[0], compress_dim, dtype=torch.float32)
    d_ape = flat.new_zeros(cmp_ratio, compress_dim, dtype=torch.float32)
    group_id = 0
    for length, n_used, start, batch in zip(lengths, used, starts, range(len(lengths)), strict=True):
        base = int(cu[batch])
        n_valid = n_used // cmp_ratio
        for group in range(length // cmp_ratio):
            if group < n_valid:
                for row in range(cmp_ratio):
                    ape_row = (start + row) % cmp_ratio
                    if coff == 1:
                        token = base + group * cmp_ratio + row
                        d_new_kv[token] = d_kv[group_id, row]
                        d_new_score[token] = d_z[group_id, row]
                        d_ape[ape_row] = d_ape[ape_row] + d_z[group_id, row]
                    else:
                        if group > 0:
                            prev = base + (group - 1) * cmp_ratio + row
                            d_new_kv[prev, :head_dim] = d_kv[group_id, 2 * row]
                            d_new_score[prev, :head_dim] = d_z[group_id, 2 * row]
                        d_ape[ape_row, :head_dim] = d_ape[ape_row, :head_dim] + d_z[group_id, 2 * row]
                        cur = base + group * cmp_ratio + row
                        d_new_kv[cur, head_dim:] = d_kv[group_id, 2 * row + 1]
                        d_new_score[cur, head_dim:] = d_z[group_id, 2 * row + 1]
                        d_ape[ape_row, head_dim:] = d_ape[ape_row, head_dim:] + d_z[group_id, 2 * row + 1]
            group_id += 1

    x_fp = flat.float()
    d_x = d_new_kv @ wkv.float() + d_new_score @ wgate.float()
    d_wkv = d_new_kv.T @ x_fp
    d_wgate = d_new_score.T @ x_fp
    return d_x, d_wkv, d_wgate, d_ape, d_new_kv, d_new_score


def backward_rounded(x, wkv, wgate, d_cmp, softmax, kv, cmp_ratio, coff=1, seqused=None, start_pos=None, cu_seqlens=None):
    """Match the kernel's bf16/fp16 rounding of the token gradients before GEMM."""
    dtype = x.dtype
    _d_x, _, _, d_ape, d_new_kv, d_new_score = backward_math(
        x,
        wkv,
        wgate,
        d_cmp,
        softmax,
        kv,
        cmp_ratio,
        coff,
        seqused,
        start_pos,
        cu_seqlens,
    )
    flat = x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x
    d_kv = d_new_kv.to(dtype).float()
    d_score = d_new_score.to(dtype).float()
    d_x_q = (d_kv @ wkv.float() + d_score @ wgate.float()).to(dtype)
    d_wkv = (d_kv.T @ flat.float()).to(dtype)
    d_wgate = (d_score.T @ flat.float()).to(dtype)
    if x.ndim == 3:
        d_x_q = d_x_q.reshape(x.shape)
    return d_x_q, d_wkv, d_wgate, d_ape
