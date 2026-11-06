"""Pack and unpack the Hy4 fp8_ds_mla 656-byte KV record.

Each token is one contiguous record. Bytes [0, 512) hold 512 FP8 E4M3 latent
values, [512, 528) hold four FP32 scales (one per 128 latent values), and
[528, 656) hold 64 BF16 RoPE values stored without quantization.
"""

import torch

RECORD_BYTES = 656
LATENT_DIM = 512
SCALE_COUNT = 4
GROUP = LATENT_DIM // SCALE_COUNT
ROPE_DIM = 64
PAGE_SIZE = 64
FP8 = torch.float8_e4m3fn


def pack_records(latent_fp8, scales, rope):
    """Pack token records into ``[tokens, 656]`` uint8 bytes.

    ``latent_fp8`` is ``[tokens, 512]`` FP8 E4M3. ``scales`` is ``[tokens, 4]``
    FP32. ``rope`` is ``[tokens, 64]`` BF16.
    """
    if latent_fp8.dtype != FP8 or latent_fp8.ndim != 2 or latent_fp8.shape[1] != LATENT_DIM:
        raise ValueError("latent_fp8 must have shape [tokens, 512] and dtype float8_e4m3fn")
    tokens = latent_fp8.shape[0]
    if scales.shape != (tokens, SCALE_COUNT) or scales.dtype != torch.float32:
        raise ValueError("scales must have shape [tokens, 4] and dtype float32")
    if rope.shape != (tokens, ROPE_DIM) or rope.dtype != torch.bfloat16:
        raise ValueError("rope must have shape [tokens, 64] and dtype bfloat16")
    raw = torch.empty((tokens, RECORD_BYTES), dtype=torch.uint8, device=latent_fp8.device)
    raw[:, :LATENT_DIM] = latent_fp8.contiguous().view(torch.uint8)
    raw[:, LATENT_DIM : LATENT_DIM + SCALE_COUNT * 4] = scales.contiguous().view(torch.uint8).view(tokens, SCALE_COUNT * 4)
    raw[:, LATENT_DIM + SCALE_COUNT * 4 :] = rope.contiguous().view(torch.uint8).view(tokens, ROPE_DIM * 2)
    return raw


def unpack_records(raw):
    """Return BF16 latent and the original BF16 RoPE.

    Latent FP8 values are multiplied by their group scale in FP32, then rounded
    to BF16. That rounding is the value the BF16 Cube path consumes. RoPE bytes
    are interpreted as BF16 and are not scaled.
    """
    records = _as_records(raw)
    tokens = records.shape[0]
    latent_fp8 = records[:, :LATENT_DIM].view(FP8).view(tokens, SCALE_COUNT, GROUP).float()
    scales = records[:, LATENT_DIM : LATENT_DIM + SCALE_COUNT * 4].view(torch.float32).view(tokens, SCALE_COUNT)
    latent = (latent_fp8 * scales[:, :, None]).reshape(tokens, LATENT_DIM).to(torch.bfloat16)
    rope = records[:, LATENT_DIM + SCALE_COUNT * 4 :].view(torch.bfloat16).view(tokens, ROPE_DIM)
    return latent, rope


def records_to_pages(raw):
    """View token records as ``[pages, 64, 1, 656]`` uint8 pages."""
    records = _as_records(raw)
    if records.shape[0] % PAGE_SIZE:
        raise ValueError("token count must be a multiple of 64 to form pages")
    pages = records.shape[0] // PAGE_SIZE
    return records.view(pages, PAGE_SIZE, 1, RECORD_BYTES)


def physical_rows(raw):
    """Dequantize a paged cache into ``[pages * 64, 576]`` BF16 rows."""
    latent, rope = unpack_records(raw)
    return torch.cat((latent, rope), dim=-1)


def _as_records(raw):
    if raw.dtype != torch.uint8:
        raise ValueError("KV cache records must have dtype uint8")
    if raw.shape[-1] != RECORD_BYTES:
        raise ValueError("KV cache records must end with the 656-byte dimension")
    if raw.ndim == 4:
        if raw.shape[1:] != (PAGE_SIZE, 1, RECORD_BYTES):
            raise ValueError("paged KV cache must have shape [pages, 64, 1, 656]")
        return raw.view(-1, RECORD_BYTES)
    if raw.ndim == 2:
        return raw
    raise ValueError("KV cache must have shape [tokens, 656] or [pages, 64, 1, 656]")
