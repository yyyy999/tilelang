"""NPU checks for the L0C on-path dual_copy review cases.

Run on Ascend. Each kernel is the DSL shape Denverjin reproduced: a compact
MAD inside a larger L0C, an N-split whose halves are disjoint, and a
destination window smaller than the UB allocation.
"""

import pytest
import torch

import tilelang
from tilelang.ascend import language as T


def _check(kernel, args, expected, thresh, label):
    out = kernel(*args)
    torch.npu.synchronize()
    diff = (out.float() - expected.float()).abs()
    denom = expected.float().abs().max().clamp_min(1e-8)
    rel = diff.max() / denom
    assert torch.isfinite(out).all(), label
    assert rel < thresh, f"{label} rel_max={float(rel):.2e}"


def _randn(*shape, dtype):
    return torch.randn(*shape, dtype=dtype, device="npu") * 0.25


def test_full_n_compact_mad_copy_and_dual():
    """64x64 L0C, L0 GEMM writes acc[:32, :64]. Pitch must be 32."""
    m, n, k = 32, 64, 32
    a = _randn(m, k, dtype=torch.float16)
    b = _randn(n, k, dtype=torch.float16)
    ref_fp32 = a.float() @ b.float().T
    ref_fp16 = ref_fp32.to(torch.float16)

    @T.prim_func
    def copy_fp32(
        A: T.Buffer((m, k), "float16"),
        B: T.Buffer((n, k), "float16"),
        C: T.Buffer((m, n), "float32"),
    ):
        with T.MixedKernel(1) as (_bx, _sid):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            a_l0 = T.alloc_l0a((m, k), "float16")
            b_l0 = T.alloc_l0b((n, k), "float16")
            acc = T.alloc_l0c((64, 64), "float32")
            ub = T.alloc_shared((m, n), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, acc[0:m, 0:n], transpose_B=True, clear_accum=True)
            T.copy(acc[0:m, 0:n], ub)
            T.copy(ub, C)

    @T.prim_func
    def dual_fp32(
        A: T.Buffer((m, k), "float16"),
        B: T.Buffer((n, k), "float16"),
        C: T.Buffer((m, n), "float32"),
    ):
        with T.MixedKernel(1) as (_bx, _sid):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            a_l0 = T.alloc_l0a((m, k), "float16")
            b_l0 = T.alloc_l0b((n, k), "float16")
            acc = T.alloc_l0c((64, 64), "float32")
            ub = T.alloc_shared((m // 2, n), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, acc[0:m, 0:n], transpose_B=True, clear_accum=True)
            T.dual_copy(acc[0:m, 0:n], ub)
            T.dual_copy(ub, C)

    @T.prim_func
    def dual_fp16(
        A: T.Buffer((m, k), "float16"),
        B: T.Buffer((n, k), "float16"),
        C: T.Buffer((m, n), "float16"),
    ):
        with T.MixedKernel(1) as (_bx, _sid):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            a_l0 = T.alloc_l0a((m, k), "float16")
            b_l0 = T.alloc_l0b((n, k), "float16")
            acc = T.alloc_l0c((64, 64), "float32")
            ub = T.alloc_shared((m // 2, n), "float16")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, acc[0:m, 0:n], transpose_B=True, clear_accum=True)
            T.dual_copy(acc[0:m, 0:n], ub)
            T.dual_copy(ub, C)

    _check(tilelang.compile(copy_fp32, out_idx=-1), (a, b), ref_fp32, 1e-3, "fp32 copy")
    _check(tilelang.compile(dual_fp32, out_idx=-1), (a, b), ref_fp32, 1e-3, "fp32 dual_copy")
    _check(tilelang.compile(dual_fp16, out_idx=-1), (a, b), ref_fp16, 2e-2, "fp16 dual_copy")


@pytest.mark.parametrize("dst_dtype,thresh", [("float16", 2e-2), ("bfloat16", 5e-2)])
def test_n_split_compact_mad_offset(dst_dtype, thresh):
    """64x128 L0C, compact 32x64 MAD, N-split into two 32x32 UBs."""
    m, n, k = 32, 64, 32
    dtype = getattr(torch, dst_dtype)
    a = _randn(m, k, dtype=torch.float16)
    b = _randn(n, k, dtype=torch.float16)
    expected = (a.float() @ b.float().T).to(dtype)

    @T.prim_func
    def main(
        A: T.Buffer((m, k), "float16"),
        B: T.Buffer((n, k), "float16"),
        C: T.Buffer((m, n), dst_dtype),
    ):
        with T.MixedKernel(1) as (_bx, _sid):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            a_l0 = T.alloc_l0a((m, k), "float16")
            b_l0 = T.alloc_l0b((n, k), "float16")
            acc = T.alloc_l0c((64, 128), "float32")
            ub = T.alloc_shared((m, n // 2), dst_dtype)
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, acc[0:m, 0:n], transpose_B=True, clear_accum=True)
            T.dual_copy(acc[0:m, 0:n], ub)
            T.dual_copy(ub, C)

    _check(tilelang.compile(main, out_idx=-1), (a, b), expected, thresh, f"n-split {dst_dtype}")


def test_n_split_unit_flag_reuses_l0c():
    """Two GEMMs reuse one 32x64 L0C with unit_flag_ctrl=3."""
    m, n, k = 32, 64, 32
    a = _randn(2, m, k, dtype=torch.float16)
    b = _randn(2, n, k, dtype=torch.float16)
    expected = torch.cat([(a[i].float() @ b[i].float().T).to(torch.float16) for i in range(2)], dim=0)

    @T.prim_func
    def main(
        A: T.Buffer((2, m, k), "float16"),
        B: T.Buffer((2, n, k), "float16"),
        C: T.Buffer((2 * m, n), "float16"),
    ):
        with T.MixedKernel(1) as (_bx, _sid):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            acc = T.alloc_l0c((m, n), "float32")
            ub = T.alloc_shared((m, n // 2), "float16")
            for step in T.serial(2):
                T.copy(A[step, :, :], a_l1)
                T.copy(B[step, :, :], b_l1)
                T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True, unit_flag_ctrl=3)
                T.dual_copy(acc, ub, unit_flag_ctrl=3)
                T.dual_copy(ub, C[step * m : (step + 1) * m, 0:n])

    _check(tilelang.compile(main, out_idx=-1), (a, b), expected, 2e-2, "unit-flag reuse")


@pytest.mark.parametrize("row_start,row_end", [(0, 32), (16, 48)])
def test_destination_window_m_split(row_start, row_end):
    """64x64 UB; dual_copy the full accumulator into a 32-row window."""
    m, n, k = 64, 64, 32
    a = _randn(m, k, dtype=torch.float16)
    b = _randn(n, k, dtype=torch.float16)
    expected = (a.float() @ b.float().T).to(torch.float16)

    @T.prim_func
    def main(
        A: T.Buffer((m, k), "float16"),
        B: T.Buffer((n, k), "float16"),
        C: T.Buffer((m, n), "float16"),
    ):
        with T.MixedKernel(1) as (_bx, _sid):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            acc = T.alloc_l0c((m, n), "float32")
            ub = T.alloc_shared((m, n), "float16")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(acc, ub[row_start:row_end, 0:n])
            T.dual_copy(ub[row_start:row_end, 0:n], C)

    _check(
        tilelang.compile(main, out_idx=-1),
        (a, b),
        expected,
        2e-2,
        f"dst window [{row_start}:{row_end}]",
    )
