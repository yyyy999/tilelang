"""Verify one BF16 case before optional warm-cache NPU-event timing."""

import argparse
import json
from pathlib import Path
import statistics

import torch
import torch_npu  # noqa: F401

from examples.ascend.hy4_mla.mfu import cube_bf16_tflops, gemm_flops, mfu
from examples.ascend.hy4_mla.compile_sparse_fwd import compile_bf16
from examples.ascend.hy4_mla.reference import sparse_mla_reference
from examples.ascend.hy4_mla.testing import check_outputs, make_inputs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--nq", type=int, default=33)
    parser.add_argument("--nk", type=int, default=257)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--scale", type=float, default=1 / 16)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-pipeline", action="store_true")
    parser.add_argument("--num-cores", type=int, help="Override the launch core count for an A/B comparison")
    parser.add_argument(
        "--implementation", choices=["hy4", "flashmla"], default="hy4", help="Hy4 QK576 or the original TileLang FlashMLA QK512 baseline"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_op_timeout_ms(30000)
    inputs = make_inputs(args.nq, args.nk, args.topk)
    properties = torch.npu.get_device_properties(torch.npu.current_device())
    device_cores = int(properties.cube_core_num)
    num_cores = device_cores if args.num_cores is None else args.num_cores
    reference_inputs = inputs
    compile_kernel = compile_bf16
    if args.implementation == "flashmla":
        from examples.ascend.flashmla.prefill import compile_prefill

        q, kv, indices, attn_sink, lengths = inputs
        # Preserve the same latent values, indices, lengths, scale and sink.
        # Zero the reference RoPE query to check the 512-dimensional baseline.
        reference_q = q.clone()
        reference_q[..., 512:] = 0
        reference_inputs = reference_q, kv, indices, attn_sink, lengths
        inputs = q[..., :512].contiguous(), kv[:, :512].contiguous(), indices, attn_sink, lengths
        compile_kernel = compile_prefill
    expected = sparse_mla_reference(*reference_inputs, scale=args.scale)
    device_inputs = tuple(t.npu() for t in inputs)
    kernel = compile_kernel(
        args.nq,
        args.nk,
        args.topk,
        scale=args.scale,
        variable_lengths=True,
        pipeline_stages=not args.no_pipeline,
        num_cores=num_cores,
    )
    actual = kernel(*device_inputs)
    torch.npu.synchronize()
    result = vars(args).copy()
    result.pop("output")
    result["accuracy"] = check_outputs(actual, expected)
    result["device"] = torch.npu.get_device_name()
    result["device_aic"] = device_cores
    result["launch_cores"] = min(args.nq, num_cores)
    result["dqk"] = 512 if args.implementation == "flashmla" else 576
    result["dv"] = 512
    if args.repeat > 0:
        for _ in range(5):
            kernel(*device_inputs)
        torch.npu.synchronize()
        times = []
        for _ in range(args.repeat):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            kernel(*device_inputs)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        result["warm_cache_ms"] = times
        result["median_ms"] = statistics.median(times)
        indices, lengths = inputs[2:]
        slots = int(((torch.arange(args.topk) < lengths[:, None]) & (indices >= 0) & (indices < args.nk)).sum())
        result["valid_slots"] = slots
        seconds = result["median_ms"] * 1e-3
        peak = cube_bf16_tflops(device_cores) if "950PR" in properties.name else None
        result["cube_peak_tflops"] = peak
        for label, count in (("effective", slots), ("padded", args.nq * args.topk)):
            flops = gemm_flops(count, dqk=result["dqk"])
            result[f"{label}_tflops"] = flops / seconds / 1e12
            if peak is not None:
                result[f"{label}_mfu"] = mfu(flops, seconds, peak)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        args.output.with_suffix(".asc").write_text(kernel.get_kernel_source())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
