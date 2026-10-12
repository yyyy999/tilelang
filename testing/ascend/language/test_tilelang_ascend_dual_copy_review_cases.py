"""Codegen coverage for the L0C on-path dual_copy review cases.

The producer MAD's M pitch, not the allocation N, decides FixPipe's source
stride and the second N-half pointer. N-split UnitFlags clear each half.
A destination window, not the UB allocation, sizes the split.
"""

import importlib.util
from pathlib import Path

import pytest

from tilelang.ascend import language as T

_HELPER_PATH = Path(__file__).with_name("test_tilelang_ascend_dual_copy_onpath_cast.py")
_spec = importlib.util.spec_from_file_location("onpath_cast_helpers", _HELPER_PATH)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_kernel_source = _helpers._kernel_source
_cc_to_ub_argument_lists = _helpers._cc_to_ub_argument_lists
_cc_to_ub_sizes = _helpers._cc_to_ub_sizes
_cc_to_ub_tails = _helpers._cc_to_ub_tails
_unwrap_c_api_arg = _helpers._unwrap_c_api_arg


def _src_strides(source):
    return [_unwrap_c_api_arg(args[5]) for args in _cc_to_ub_argument_lists(source)]


def _assert_stride(func, expected):
    source = _kernel_source(func)
    strides = _src_strides(source)
    assert strides, source
    assert set(strides) == {expected}, (strides, source)


def test_full_n_compact_mad_uses_producer_m_pitch():
    """64x64 L0C, MAD writes 32x64. Stride stays 32 for copy and both duals."""
    alloc_m, alloc_n, m, n, k = 64, 64, 32, 64, 32

    @T.prim_func
    def copy_fp32(
        A: T.Buffer((m, k), "float16"),
        B: T.Buffer((n, k), "float16"),
        C: T.Buffer((m, n), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            a_l0 = T.alloc_l0a((m, k), "float16")
            b_l0 = T.alloc_l0b((n, k), "float16")
            acc = T.alloc_l0c((alloc_m, alloc_n), "float32")
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
        with T.Kernel(1):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            a_l0 = T.alloc_l0a((m, k), "float16")
            b_l0 = T.alloc_l0b((n, k), "float16")
            acc = T.alloc_l0c((alloc_m, alloc_n), "float32")
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
        with T.Kernel(1):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            a_l0 = T.alloc_l0a((m, k), "float16")
            b_l0 = T.alloc_l0b((n, k), "float16")
            acc = T.alloc_l0c((alloc_m, alloc_n), "float32")
            ub = T.alloc_shared((m // 2, n), "float16")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, acc[0:m, 0:n], transpose_B=True, clear_accum=True)
            T.dual_copy(acc[0:m, 0:n], ub)
            T.dual_copy(ub, C)

    _assert_stride(copy_fp32, "32")
    _assert_stride(dual_fp32, "32")
    _assert_stride(dual_fp16, "32")


def test_prefix_of_full_gemm_keeps_producer_pitch():
    """256x256 MAD, then a 128x128 dual_copy. Stride stays 256, not 128."""
    alloc = 256
    region = 128

    @T.prim_func
    def main(
        A: T.Buffer((alloc, alloc), "float16"),
        B: T.Buffer((alloc, alloc), "float16"),
        C: T.Buffer((region // 2, region), "float16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((alloc, alloc), "float16")
            b_l1 = T.alloc_l1((alloc, alloc), "float16")
            acc = T.alloc_l0c((alloc, alloc), "float32")
            tmp = T.alloc_shared((region // 2, region), "float16")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(acc[0:region, 0:region], tmp)
            T.copy(tmp, C)

    source = _kernel_source(main)
    assert set(_src_strides(source)) == {"256"}, source


def test_n_split_compact_second_half_uses_producer_pitch():
    """64x128 L0C, compact 32x64 MAD, N-split into 32x32. Second half is +1024."""
    alloc_m, alloc_n, m, n, k = 64, 128, 32, 64, 32

    @T.prim_func
    def main(
        A: T.Buffer((m, k), "float16"),
        B: T.Buffer((n, k), "float16"),
        C: T.Buffer((m, n), "float16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((m, k), "float16")
            b_l1 = T.alloc_l1((n, k), "float16")
            a_l0 = T.alloc_l0a((m, k), "float16")
            b_l0 = T.alloc_l0b((n, k), "float16")
            acc = T.alloc_l0c((alloc_m, alloc_n), "float32")
            ub = T.alloc_shared((m, n // 2), "float16")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, acc[0:m, 0:n], transpose_B=True, clear_accum=True)
            T.dual_copy(acc[0:m, 0:n], ub)
            T.dual_copy(ub, C)

    source = _kernel_source(main)
    args = _cc_to_ub_argument_lists(source)
    assert len(args) == 2, source
    assert set(_src_strides(source)) == {"32"}, source
    assert "1024" in args[1][1], (args[1][1], source)
    assert "2048" not in args[1][1], (args[1][1], source)
    for copy_inner, copy_rows in _cc_to_ub_sizes(source):
        assert copy_inner == "32", (copy_inner, source)
        assert copy_rows == "32", (copy_rows, source)


def test_n_split_unit_flag_clears_each_half_across_reuse():
    """Two iterations reuse one 32x64 L0C. Both N halves must CHECK_AND_CLEAR."""
    tile_m, tile_n, tile_k = 32, 64, 32

    @T.prim_func
    def main(
        A: T.Buffer((2, tile_m, tile_k), "float16"),
        B: T.Buffer((2, tile_n, tile_k), "float16"),
        C: T.Buffer((2 * tile_m, tile_n), "float16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((tile_m, tile_k), "float16")
            b_l1 = T.alloc_l1((tile_n, tile_k), "float16")
            acc = T.alloc_l0c((tile_m, tile_n), "float32")
            tmp = T.alloc_shared((tile_m, tile_n // 2), "float16")
            for step in T.serial(2):
                T.copy(A[step, :, :], a_l1)
                T.copy(B[step, :, :], b_l1)
                T.gemm(
                    a_l1,
                    b_l1,
                    acc,
                    transpose_B=True,
                    clear_accum=True,
                    unit_flag_ctrl=3,
                )
                T.dual_copy(acc, tmp, unit_flag_ctrl=3)
                T.dual_copy(tmp, C[step * tile_m : (step + 1) * tile_m, 0:tile_n])

    source = _kernel_source(main)
    tails = _cc_to_ub_tails(source)
    assert len(tails) >= 2, source
    assert all(tail[3] == "3" for tail in tails), tails


@pytest.mark.parametrize("row_start,row_end", [(0, 32), (16, 48)])
def test_m_split_uses_destination_region_not_allocation(row_start, row_end):
    """64x64 UB, dual_copy into a 32-row window. Each pipe copies 32 rows."""
    tile_m, tile_n, tile_k = 64, 64, 32

    @T.prim_func
    def main(
        A: T.Buffer((tile_m, tile_k), "float16"),
        B: T.Buffer((tile_n, tile_k), "float16"),
        C: T.Buffer((tile_m, tile_n), "float16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((tile_m, tile_k), "float16")
            b_l1 = T.alloc_l1((tile_n, tile_k), "float16")
            acc = T.alloc_l0c((tile_m, tile_n), "float32")
            ub = T.alloc_shared((tile_m, tile_n), "float16")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(acc, ub[row_start:row_end, 0:tile_n])
            T.dual_copy(ub[row_start:row_end, 0:tile_n], C)

    source = _kernel_source(main)
    sizes = _cc_to_ub_sizes(source)
    assert len(sizes) == 2, source
    for copy_inner, copy_rows in sizes:
        assert copy_inner == "64", (copy_inner, source)
        assert copy_rows == "32", (copy_rows, source)
