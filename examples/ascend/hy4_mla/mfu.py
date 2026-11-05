"""MFU for the H64 BF16 sparse MLA core.

MFU = FLOPs / (T * P_peak). One multiply-add counts as 2 FLOP. The effective
numerator keeps one QK576 product and one PV512 product for each retained
index slot and each of the 64 heads. Softmax, the sink, and data movement
stay in T.

P_peak is Cube BF16 only. Ascend 950PR figures below were queried in October
2026. The Cube+Vector total is about 12.5% higher and is not a GEMM peak.
"""

HEADS = 64
QK_K = 576
PV_K = 512
FLOPS_PER_SLOT = 2 * QK_K + 2 * PV_K

# AIC count -> Cube BF16 TFLOPS. 28 AIC is 87.5% of the 32 AIC peak.
CUBE_BF16_TFLOPS = {32: 432.0, 28: 378.0}


def cube_bf16_tflops(aic):
    try:
        return CUBE_BF16_TFLOPS[aic]
    except KeyError as exc:
        known = ", ".join(str(count) for count in sorted(CUBE_BF16_TFLOPS))
        raise ValueError(f"no 950PR Cube BF16 peak for {aic} AIC; known counts: {known}") from exc


def count_valid_slots(indices, nk, lengths):
    """Count index slots that pass the length prefix and refer to a KV row.

    Duplicates each contribute one product. Slots at or past the length, and
    indices outside ``[0, nk)``, are excluded.
    """
    if nk <= 0:
        raise ValueError("nk must be positive")
    total = 0
    for row, length in zip(indices, lengths, strict=True):
        limit = int(length)
        if limit < 0:
            raise ValueError("length must be non-negative")
        for slot, index in enumerate(row):
            if slot >= limit:
                break
            index = int(index)
            if 0 <= index < nk:
                total += 1
    return total


def gemm_flops(slots):
    """FLOPs for ``slots`` retained scores across all heads."""
    if slots < 0:
        raise ValueError("slots must be non-negative")
    return slots * HEADS * FLOPS_PER_SLOT


def mfu(flops, seconds, peak_tflops):
    """Return utilization in ``[0, +inf)``. Multiply by 100 for a percentage."""
    if flops < 0 or seconds <= 0 or peak_tflops <= 0:
        raise ValueError("flops must be non-negative and time and peak must be positive")
    return flops / (seconds * peak_tflops * 1e12)
