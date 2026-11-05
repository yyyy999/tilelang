"""Reproducible inputs and numerical checks for the BF16 core."""

import torch


def make_inputs(nq, nk, topk=2048, *, seed=42, component="both", variable_lengths=True):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn((nq, 64, 576), generator=generator) * 0.3).to(torch.bfloat16)
    kv = (torch.randn((nk, 576), generator=generator) * 0.3).to(torch.bfloat16)
    if component == "rope":
        q[..., :512] = 0
    elif component == "latent":
        q[..., 512:] = 0
    elif component != "both":
        raise ValueError("component must be both, rope or latent")
    indices = torch.randint(nk, (nq, topk), generator=generator, dtype=torch.int32)
    # Mixed invalid pairs exercise zero-burst DMA and nonadjacent gathers.
    indices[:, ::17] = -1
    indices[:, 5::29] = nk
    indices[:, 7::31] = -9
    indices[:, 11::37] = nk + 17
    attn_sink = torch.linspace(-3, 3, 64, dtype=torch.float32)
    boundaries = torch.tensor([topk, 0, 1, 63, 64, 65, 639, 640, 641, topk - 1], dtype=torch.int32).clamp(max=topk)
    lengths = boundaries.repeat((nq + len(boundaries) - 1) // len(boundaries))[:nq].contiguous()
    if not variable_lengths:
        lengths.fill_(topk)
    return q, kv, indices, attn_sink, lengths


def check_outputs(actual, expected):
    out, maximum, lse = (t.detach().cpu() for t in actual)
    ref_out, ref_max, ref_lse = (t.detach().cpu() for t in expected)
    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(maximum, ref_max, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(lse, ref_lse, rtol=2e-4, atol=2e-4)
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"
    delta = out.float() - ref_out.float()
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / ref_out.float().norm().clamp_min(1e-12)).item(),
    }
