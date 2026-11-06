"""Compilation entry point for FP8 main and optional FP8/FP4 extra KV caches."""

import tilelang
import torch
from tilelang.carver.arch.driver.ascend_driver import get_num_cube_cores

from examples.ascend.flashmla.kernel import sparse_attention


def compile_decode(nq, nk, topk, *, num_cores=None, **kwargs):
    if num_cores is None:
        num_cores = int(get_num_cube_cores(torch.npu.current_device()))
    return tilelang.compile(
        sparse_attention(nq, nk, topk, kv_format="fp8", num_cores=num_cores, **kwargs),
        target="ascend",
        out_idx=[8, 9],
        compile_flags=["-Ofast", "-mllvm", "-enable-hiipu-vf-loop-unroll"],
    )
