"""NPU accuracy for the L0C on-path dual_copy review cases.

Each case compiles, launches on the device, and checks the relative L-inf
error against a torch matmul. Shapes match the kernels Denverjin reproduced:
a compact MAD inside a larger L0C, an N-split of that compact result, reuse
of one L0C slot, and a destination window smaller than the UB allocation.
"""

import pytest
import torch

import tilelang
from tilelang.ascend import language as T


def _assert_rel(out, expected, thresh, label):
    torch.npu.synchronize()
    diff = (out.float() - expected.float()).abs()
    if not torch.isfinite(out).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{label} produced non-finite values")
    denom = expected.float().abs().max().clamp_min(1e-8)
    rel = diff.max() / denom
    print(f"  {label}: {'PASS' if rel < thresh else 'FAIL'}  rel_max={float(rel):.2e}")
    if not rel < thresh:
        raise AssertionError(f"{label} rel_max={float(rel):.2e}")
    return rel


def _randn(*shape, dtype):
    return torch.randn(*shape, dtype=dtype, device="npu") * 0.25


def _launch(program, args, expected, thresh, label):
    kernel = tilelang.compile(program, out_idx=-1)
    out = kernel(*args)
    return _assert_rel(out, expected, thresh, label)


def full_n_copy_kernel():
    """64x64 L0C, L0 GEMM writes acc[:32, :64], then a plain fp32 copy."""
    m, n, k = 32, 64, 32

    @T.prim_func
    def main(
        A: T.Buffer((m, k), "float16"),
        B: T.Buffer((n, k), "float16"),
        C: T.Buffer((m, n), "float32"),
    ):
        with T.MixedKernel(1) as (_bx, sid):
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
            # Non-dual FixPipe writes only subblock 0. The other AIV's UB is
            # empty; letting it store the same GM range clobbers the result.
            T.copy(acc[0:m, 0:n], ub)
            if sid == 0:
                T.copy(ub, C)

    return main


def full_n_dual_kernel(dst_dtype):
    """Same compact MAD, then M-split dual_copy into a half-row UB."""
    m, n, k = 32, 64, 32

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
            acc = T.alloc_l0c((64, 64), "float32")
            ub = T.alloc_shared((m // 2, n), dst_dtype)
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, acc[0:m, 0:n], transpose_B=True, clear_accum=True)
            T.dual_copy(acc[0:m, 0:n], ub)
            T.dual_copy(ub, C)

    return main


def run_full_n_compact_mad_copy_and_dual():
    m, n, k = 32, 64, 32
    a = _randn(m, k, dtype=torch.float16)
    b = _randn(n, k, dtype=torch.float16)
    ref_fp32 = a.float() @ b.float().T
    ref_fp16 = ref_fp32.to(torch.float16)
    _launch(full_n_copy_kernel(), (a, b), ref_fp32, 1e-3, "fp32 copy")
    _launch(full_n_dual_kernel("float32"), (a, b), ref_fp32, 1e-3, "fp32 dual_copy")
    _launch(full_n_dual_kernel("float16"), (a, b), ref_fp16, 2e-2, "fp16 dual_copy")


def n_split_compact_kernel(dst_dtype):
    """64x128 L0C, compact 32x64 MAD, N-split into two 32x32 UBs."""
    m, n, k = 32, 64, 32

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

    return main


def run_n_split_compact_mad_offset(dst_dtype, thresh):
    m, n, k = 32, 64, 32
    dtype = getattr(torch, dst_dtype)
    a = _randn(m, k, dtype=torch.float16)
    b = _randn(n, k, dtype=torch.float16)
    expected = (a.float() @ b.float().T).to(dtype)
    _launch(n_split_compact_kernel(dst_dtype), (a, b), expected, thresh, f"n-split {dst_dtype}")


def unit_flag_reuse_kernel():
    """Two GEMMs reuse one 32x64 L0C with unit_flag_ctrl=3."""
    m, n, k = 32, 64, 32

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

    return main


def run_n_split_unit_flag_reuses_l0c():
    m, n, k = 32, 64, 32
    a = _randn(2, m, k, dtype=torch.float16)
    b = _randn(2, n, k, dtype=torch.float16)
    expected = torch.cat([(a[i].float() @ b[i].float().T).to(torch.float16) for i in range(2)], dim=0)
    _launch(unit_flag_reuse_kernel(), (a, b), expected, 2e-2, "unit-flag reuse")


def destination_window_kernel(row_start, row_end):
    """Full 64x64 GEMM; dual_copy into a 32-row window of a 64x64 UB."""
    m, n, k = 64, 64, 32

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

    return main


def run_destination_window_m_split(row_start, row_end):
    m, n, k = 64, 64, 32
    a = _randn(m, k, dtype=torch.float16)
    b = _randn(n, k, dtype=torch.float16)
    expected = (a.float() @ b.float().T).to(torch.float16)
    _launch(
        destination_window_kernel(row_start, row_end),
        (a, b),
        expected,
        2e-2,
        f"dst window [{row_start}:{row_end}]",
    )


def prefix_kernel():
    """256x256 GEMM; dual_copy only acc[:128, :128]. Pitch must stay 256."""
    alloc, region = 256, 128

    @T.prim_func
    def main(
        A: T.Buffer((alloc, alloc), "float16"),
        B: T.Buffer((alloc, alloc), "float16"),
        C: T.Buffer((region, region), "float16"),
    ):
        with T.MixedKernel(1) as (_bx, _sid):
            a_l1 = T.alloc_l1((alloc, alloc), "float16")
            b_l1 = T.alloc_l1((alloc, alloc), "float16")
            acc = T.alloc_l0c((alloc, alloc), "float32")
            ub = T.alloc_shared((region // 2, region), "float16")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(acc[0:region, 0:region], ub)
            T.dual_copy(ub, C)

    return main


def run_prefix_of_full_gemm():
    alloc, region = 256, 128
    a = _randn(alloc, alloc, dtype=torch.float16)
    b = _randn(alloc, alloc, dtype=torch.float16)
    expected = (a[:region].float() @ b[:region].float().T).to(torch.float16)
    _launch(prefix_kernel(), (a, b), expected, 2e-2, "prefix 128 of 256")


def test_full_n_compact_mad_copy_and_dual():
    run_full_n_compact_mad_copy_and_dual()


@pytest.mark.parametrize("dst_dtype,thresh", [("float16", 2e-2), ("bfloat16", 5e-2)])
def test_n_split_compact_mad_offset(dst_dtype, thresh):
    run_n_split_compact_mad_offset(dst_dtype, thresh)


def test_n_split_unit_flag_reuses_l0c():
    run_n_split_unit_flag_reuses_l0c()


@pytest.mark.parametrize("row_start,row_end", [(0, 32), (16, 48)])
def test_destination_window_m_split(row_start, row_end):
    run_destination_window_m_split(row_start, row_end)


def test_prefix_of_full_gemm_keeps_result():
    run_prefix_of_full_gemm()


if __name__ == "__main__":
    print("=== compact MAD 32x64 inside 64x64 L0C ===")
    run_full_n_compact_mad_copy_and_dual()
    print("=== compact MAD N-split offset ===")
    run_n_split_compact_mad_offset("float16", 2e-2)
    run_n_split_compact_mad_offset("bfloat16", 5e-2)
    print("=== N-split UnitFlag reuses one L0C ===")
    run_n_split_unit_flag_reuses_l0c()
    print("=== destination window ===")
    run_destination_window_m_split(0, 32)
    run_destination_window_m_split(16, 48)
    print("=== 128 prefix of a 256 GEMM ===")
    run_prefix_of_full_gemm()
    print("PASS: review-case accuracy")
