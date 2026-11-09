"""Measure Hy4 flash_mla_with_kvcache with distinct physical KV slots."""

import argparse
import json
import statistics
import time

import torch

from examples.ascend.hy4_mla.fp8_cache import PAGE_SIZE, RECORD_BYTES, pack_records, records_to_pages

try:
    from examples.ascend.hy4_mla.flash_mla_with_kvcache import _get_kvcache_kernel, flash_mla_with_kvcache
    from examples.ascend.hy4_mla.kvcache_reference import fp8_sparse_kvcache_reference
except ModuleNotFoundError as exc:
    if exc.name != "examples.ascend.hy4_mla.flash_mla_with_kvcache":
        raise
    from examples.ascend.hy4_mla.decode import _get_decode_kernel as _get_kvcache_kernel
    from examples.ascend.hy4_mla.decode import flash_mla_with_kvcache
    from examples.ascend.hy4_mla.decode_reference import fp8_sparse_decode_reference as fp8_sparse_kvcache_reference


def make_inputs(batch, query_len, context, local_heads):
    rng = torch.Generator().manual_seed(2026)
    topk = 2048
    # Give each request its own page-aligned span. Different queries may
    # revisit the same slots, but each query selects 2048 distinct slots.
    q = torch.zeros((batch, query_len, 64, 576), dtype=torch.bfloat16)
    q[:, :, :local_heads] = (torch.randn((batch, query_len, local_heads, 576), generator=rng) * 0.3).bfloat16()
    sink = torch.full((64,), -torch.inf)
    sink[:local_heads] = torch.linspace(-3, 3, local_heads)
    indices = torch.empty((batch, query_len, topk), dtype=torch.int32)
    for b in range(batch):
        for s in range(query_len):
            indices[b, s] = torch.randperm(context, generator=rng)[:topk] + b * context
    # Write one page range at a time to limit host FP32 staging memory.
    raw = torch.empty((batch * context, RECORD_BYTES), dtype=torch.uint8)
    for start in range(0, raw.shape[0], 4096):
        rows = raw[start : start + 4096]
        n = rows.shape[0]
        latent = torch.randn((n, 512), generator=rng).to(torch.float8_e4m3fn)
        scales = (0.1 + torch.rand((n, 4), generator=rng)) * torch.tensor([0.15, 0.3, 0.45, 0.6])
        rope = (torch.randn((n, 64), generator=rng) * 0.4).bfloat16()
        rows.copy_(pack_records(latent, scales, rope))
    return q, records_to_pages(raw), indices, sink


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--context", type=int, default=32768, help="Physical KV slots per request, divisible by 64")
    parser.add_argument("--local-heads", type=int, default=8, help="Real heads before padding to the kernel's H64")
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--kernel-only", action="store_true", help="Skip timing the wrapper when collecting msprof data")
    args = parser.parse_args()
    if min(args.batch, args.query_len, args.repeat) <= 0 or not 1 <= args.local_heads <= 64:
        parser.error("batch/query-len/repeat must be positive and local-heads must be in [1,64]")
    if args.context < 2048 or args.context % PAGE_SIZE:
        parser.error("context must be at least 2048 and divisible by 64")

    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    device_id = torch.npu.current_device()
    device = torch.device("npu", device_id)
    q_cpu, cache_cpu, indices_cpu, sink_cpu = make_inputs(args.batch, args.query_len, args.context, args.local_heads)
    scale = 1 / 16
    expected = fp8_sparse_kvcache_reference(q_cpu, cache_cpu, indices_cpu, sink_cpu, scale=scale)
    q, cache, indices, sink = (tensor.to(device) for tensor in (q_cpu, cache_cpu, indices_cpu, sink_cpu))
    nq, nk = args.batch * args.query_len, args.batch * args.context
    lengths = torch.empty(nq, dtype=torch.int32, device=device)
    kernel = _get_kvcache_kernel(nq, nk, 2048, scale, True, device)
    flat_inputs = q.view(nq, 64, 576), cache.view(nk, RECORD_BYTES), indices.view(nq, 2048), sink, lengths
    out, lse = kernel(*flat_inputs)
    torch.npu.synchronize()
    torch.testing.assert_close(out.view_as(expected[0]).cpu(), expected[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse.view(args.batch, args.query_len, 64).transpose(1, 2).cpu(), expected[1], rtol=2e-4, atol=2e-4)

    for _ in range(10):
        kernel(*flat_inputs)
    torch.npu.synchronize()
    event_us = []
    for _ in range(args.repeat):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        kernel(*flat_inputs)
        end.record()
        end.synchronize()
        event_us.append(start.elapsed_time(end) * 1000)
    result = {
        "device": torch.npu.get_device_name(),
        "q_shape": list(q.shape),
        "k_cache_shape": list(cache.shape),
        "indices_shape": list(indices.shape),
        "local_heads": args.local_heads,
        "device_aic": int(torch.npu.get_device_properties(device_id).cube_core_num),
        "samples": args.repeat,
        "direct_event_median_us": statistics.median(event_us),
        "direct_event_range_us": [min(event_us), max(event_us)],
    }
    if not args.kernel_only:
        wrapper_us = []
        for _ in range(10):
            flash_mla_with_kvcache(q, cache, indices, softmax_scale=scale, attn_sink=sink)
        torch.npu.synchronize()
        for _ in range(args.repeat):
            start = time.perf_counter_ns()
            flash_mla_with_kvcache(q, cache, indices, softmax_scale=scale, attn_sink=sink)
            torch.npu.synchronize()
            wrapper_us.append((time.perf_counter_ns() - start) / 1000)
        result["wrapper_synchronized_median_us"] = statistics.median(wrapper_us)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
