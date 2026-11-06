"""Convert one tile of V3.2 FP8 records directly to the Hy4 NZ layout."""

import tilelang.ascend.language as T


@T.macro
def unpack_fp8_tile(raw, nz):
    scales = T.view(raw, shape=(32, 168), dtype="float32")
    words = T.view(raw, shape=(32, 336), dtype="uint16")
    with T.SimdVF():
        m8 = T.simd.pset(8)
        m32 = T.simd.pset(32)
        m16 = T.simd.pset(16)
        rope_mask = T.simd.pset(16, "PAT_VL64")
        # RoPE starts 16 bytes after the aligned 512-byte boundary.
        rope_offsets = T.reinterpret("uint16x128", T.simd.vci(8, "int16"))
        for row in T.unroll(32, unroll_factor=1):
            for group in T.unroll(4):
                scale = T.simd.vld(scales[row, 128 + group], "BRC_B32")
                # UNPK_B8 puts consecutive FP8 bytes in register lanes 0,2,4,... .
                # P0/P2 select the even/odd input elements into separate f32 vectors.
                packed = T.reinterpret("float8_e4m3fnx256", T.simd.vld(raw[row, group * 128], "UNPK_B8"))
                even = T.simd.vcvt(packed, "float32", m8, part=0)
                odd = T.simd.vcvt(packed, "float32", m8, part=2)
                even_bf16 = T.simd.vcvt(T.simd.vmul(even, scale, m32), "bfloat16", m32, part=0)
                odd_bf16 = T.simd.vcvt(T.simd.vmul(odd, scale, m32), "bfloat16", m32, part=1)
                # Narrow into complementary BF16 lanes before the NZ block store.
                bits = T.simd.vor(T.reinterpret("uint16x128", even_bf16), T.reinterpret("uint16x128", odd_bf16))
                T.simd.vsstb(T.reinterpret("bfloat16x128", bits), nz[row, group * 128], 33 << 16, m16)
            rope = T.simd.vgather2(T.access_ptr(words[row, 256], "r", extent=72), rope_offsets, rope_mask)
            T.simd.vsstb(T.reinterpret("bfloat16x128", rope), nz[row, 512], 33 << 16, rope_mask)
