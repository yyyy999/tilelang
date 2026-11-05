"""Generate Ascend source and allocation evidence without launching an NPU.

Requires a matching native TileLang build, but no tensor allocation, JIT device
compilation, kernel invocation or device synchronization is performed.
"""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--topk", type=int, nargs="+", default=[64, 2048])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Import only after CLI handling; --help works in the stdlib-only setup.
    import tilelang
    from tilelang import tvm
    from tvm import tirx

    from examples.ascend.hy4_mla.check_resources import inspect
    from examples.ascend.hy4_mla.kernel import hy4_sparse_attention

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    allocations = []

    @tvm.instrument.pass_instrument
    class CaptureAllocations:
        def run_after_pass(self, mod, info):
            if info.name != "tl.MergeUBAllocations":
                return

            def visit(node):
                if isinstance(node, tirx.AllocBuffer):
                    buf = node.buffer
                    bits = tvm.DataType(buf.dtype).bits * tvm.DataType(buf.dtype).lanes
                    allocations.append({"name": buf.name, "scope": buf.scope(), "bytes": math.prod(int(d) for d in buf.shape) * bits // 8})

            for func in mod.functions.values():
                if isinstance(func, tirx.PrimFunc):
                    tirx.stmt_functor.post_order_visit(func.body, visit)

    for topk in args.topk:
        print(f"LOWERING topk={topk}", flush=True)
        budget = inspect(topk)
        if not budget["fits"]:
            raise ValueError(f"Source allocation budget exceeded: {budget['pools']}")
        allocations.clear()
        program = hy4_sparse_attention(65, 257, topk, scale=1 / 16, variable_lengths=True)
        with tvm.transform.PassContext(instruments=[CaptureAllocations()]):
            artifact = tilelang.lower(program, target="ascend", enable_device_compile=False)
        (args.output / f"topk{topk}.asc").write_text(artifact.kernel_source)
        (args.output / f"topk{topk}.tir").write_text(artifact.device_mod.script())
        record = {"topk": topk, "source_budget": budget["pools"], "lowered_allocations": list(allocations)}
        records.append(record)
        (args.output / "lowering.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
