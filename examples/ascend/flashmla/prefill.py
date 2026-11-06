"""Compilation entry point for BF16 sparse prefill."""

import tilelang
import torch
from tilelang.carver.arch.driver.ascend_driver import get_num_cube_cores

from examples.ascend.flashmla.kernel import sparse_attention


def compile_prefill(nq, nk, topk, *, num_cores=None, **kwargs):
    if num_cores is None:
        num_cores = int(get_num_cube_cores(torch.npu.current_device()))
    return tilelang.compile(
        sparse_attention(nq, nk, topk, kv_format="bf16", num_cores=num_cores, **kwargs),
        target="ascend",
        out_idx=[5, 6, 7],
        compile_flags=["-Ofast", "-mllvm", "-enable-hiipu-vf-loop-unroll"],
    )
