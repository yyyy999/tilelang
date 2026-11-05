"""Verify one BF16 case before optional warm-cache NPU-event timing."""

import argparse
import json
from pathlib import Path
import statistics

import torch
import torch_npu  # noqa: F401

from examples.ascend.hy4_mla.prefill import compile_bf16
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
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_op_timeout_ms(30000)
    inputs = make_inputs(args.nq, args.nk, args.topk)
    expected = sparse_mla_reference(*inputs, scale=args.scale)
    device_inputs = tuple(t.npu() for t in inputs)
    kernel = compile_bf16(args.nq, args.nk, args.topk, scale=args.scale, variable_lengths=True, pipeline_stages=not args.no_pipeline)
    actual = kernel(*device_inputs)
    torch.npu.synchronize()
    result = vars(args).copy()
    result.pop("output")
    result["accuracy"] = check_outputs(actual, expected)
    result["device"] = torch.npu.get_device_name()
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
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        args.output.with_suffix(".asc").write_text(kernel.get_kernel_source())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
