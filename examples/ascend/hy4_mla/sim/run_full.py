"""Run the real Hy4 kernel under npusim.

This is the compiled ``hy4_sparse_attention`` kernel: gather, QK GEMM,
softmax, and PV. The launch is one query and two 64-slot tiles so the
simulator executes the whole operator without the 4096-query grid.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch_npu  # noqa: F401

from examples.ascend.hy4_mla.compile_sparse_fwd import compile_bf16
from examples.ascend.hy4_mla.testing import make_inputs


def main():
    nq, nk, topk = 1, 128, 128
    scale = 1 / 16
    print(f"hy4_full compile nq={nq} nk={nk} topk={topk}", flush=True)
    inputs = make_inputs(nq, nk, topk, variable_lengths=False)
    kernel = compile_bf16(
        nq,
        nk,
        topk,
        scale=scale,
        variable_lengths=False,
        pipeline_stages=True,
        num_cores=1,
    )
    print("hy4_full compiled", flush=True)
    device_inputs = tuple(tensor.npu() for tensor in inputs)
    outputs = kernel(*device_inputs)
    torch.npu.synchronize()
    out = outputs[0]
    print(
        f"hy4_full_kernel done shape={tuple(out.shape)} dtype={out.dtype}",
        flush=True,
    )


if __name__ == "__main__":
    main()
