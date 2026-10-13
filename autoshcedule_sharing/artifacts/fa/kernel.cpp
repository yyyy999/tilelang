#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/nd2nz_copy.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void main_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_bool init_mask = asc_create_mask_b32(PAT_ALL);
  auto zero = simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), init_mask, MODE_ZEROING);
  auto initial_m = simd_inst::vdup<float>(tl::limits::kFloatNInf, init_mask, MODE_ZEROING);
  simd_inst::vsts_norm(zero, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[57728])), 0, init_mask);
  simd_inst::vsts_norm(initial_m, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[57664])), 0, init_mask);
}

__simd_vf__ inline void main_kernel_simd_vf_1(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_bool clear_mask = asc_create_mask_b32(PAT_ALL);
  auto output_zero = simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), clear_mask, MODE_ZEROING);
  for (int32_t i = 0; i < 64; ++i) {
    for (int32_t _tmp = 0; _tmp < 2; ++_tmp) {
      simd_inst::vsts_norm(output_zero, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((i * 128) + (_tmp * 64)) + 49280)])), 0, clear_mask);
    }
  }
}

__simd_vf__ inline void main_kernel_simd_vf_3(__ubuf__ uint8_t* buf_dyn_shmem, int32_t local_mt_1, int32_t kv_1) {
  vector_bool rescale_mask = asc_create_mask_b32(PAT_ALL);
  for (int32_t i = 0; i < 64; ++i) {
    auto alpha = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((((local_mt_1 * 4) + kv_1) % 3) * 64) + i) + 57472)])), 0);
    for (int32_t _tmp = 0; _tmp < 2; ++_tmp) {
      auto prev = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((i * 128) + (_tmp * 64)) + 49280)])), 0);
      auto cur = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((((kv_1 + 1) & 1) * 8192) + (i * 128)) + (_tmp * 64)) + 24576)])), 0);
      auto updated = simd_inst::vadd(simd_inst::vmul(alpha, prev, rescale_mask, MODE_ZEROING), cur, rescale_mask, MODE_ZEROING);
      simd_inst::vsts_norm(updated, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((i * 128) + (_tmp * 64)) + 49280)])), 0, rescale_mask);
    }
  }
}

__simd_vf__ inline void main_kernel_simd_vf_2(__ubuf__ uint8_t* buf_dyn_shmem, int32_t kv_1, int32_t local_mt_1) {
  __ubuf__ void* dst_nz_ptr = nullptr;
  vector_bool softmax_full = asc_create_mask_b32(PAT_ALL);
  vector_bool softmax_one = asc_create_mask_b32(PAT_VL1);
  vector_bool softmax_b16 = asc_create_mask_b16(PAT_VL128);
  dst_nz_ptr = (&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((kv_1 & 1) * 8320) + 81920)]));
  simd_inst::vsts_norm(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[57664])), 0), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[57792])), 0, softmax_full);
  for (int32_t _tmp = 0; _tmp < 32; ++_tmp) {
    auto s00 = simd_inst::vmuls(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((((local_mt_1 * 4) + kv_1) % 3) * 8192) + (_tmp * 256))])), 0), float(0x1.6a09e667f3bccp-4f/*8.838835e-02*/), softmax_full, MODE_ZEROING);
    auto s01 = simd_inst::vmuls(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((((local_mt_1 * 4) + kv_1) % 3) * 8192) + (_tmp * 256)) + 64)])), 0), float(0x1.6a09e667f3bccp-4f/*8.838835e-02*/), softmax_full, MODE_ZEROING);
    auto s10 = simd_inst::vmuls(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((((local_mt_1 * 4) + kv_1) % 3) * 8192) + (_tmp * 256)) + 128)])), 0), float(0x1.6a09e667f3bccp-4f/*8.838835e-02*/), softmax_full, MODE_ZEROING);
    auto s11 = simd_inst::vmuls(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((((local_mt_1 * 4) + kv_1) % 3) * 8192) + (_tmp * 256)) + 192)])), 0), float(0x1.6a09e667f3bccp-4f/*8.838835e-02*/), softmax_full, MODE_ZEROING);
    simd_inst::vsts_norm(s00, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((((local_mt_1 * 4) + kv_1) % 3) * 8192) + (_tmp * 256))])), 0, softmax_full);
    simd_inst::vsts_norm(s01, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((((local_mt_1 * 4) + kv_1) % 3) * 8192) + (_tmp * 256)) + 64)])), 0, softmax_full);
    simd_inst::vsts_norm(s10, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((((local_mt_1 * 4) + kv_1) % 3) * 8192) + (_tmp * 256)) + 128)])), 0, softmax_full);
    simd_inst::vsts_norm(s11, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((((local_mt_1 * 4) + kv_1) % 3) * 8192) + (_tmp * 256)) + 192)])), 0, softmax_full);
    auto max0 = simd_inst::vcmax(simd_inst::vmax(s00, s01, softmax_full, MODE_ZEROING), softmax_full, MODE_ZEROING);
    auto max1 = simd_inst::vcmax(simd_inst::vmax(s10, s11, softmax_full, MODE_ZEROING), softmax_full, MODE_ZEROING);
    simd_inst::vsts_1st(max0, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((_tmp * 2) + 57664)])), 0, softmax_one);
    simd_inst::vsts_1st(max1, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((_tmp * 2) + 57665)])), 0, softmax_one);
  }
  simd_inst::mem_bar(VST_VLD);
  auto merged_max = simd_inst::vmax(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[57664])), 0), simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[57792])), 0), softmax_full, MODE_ZEROING);
  simd_inst::vsts_norm(merged_max, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[57664])), 0, softmax_full);
  simd_inst::mem_bar(VST_VLD);
  for (int32_t _tmp_1 = 0; _tmp_1 < 32; ++_tmp_1) {
    auto max0_1 = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((_tmp_1 * 2) + 57664)])), 0);
    auto max1_1 = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((_tmp_1 * 2) + 57665)])), 0);
    auto v = simd_inst::vld_x2<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((((local_mt_1 * 4) + kv_1) % 3) * 8192) + (_tmp_1 * 256))])), DINTLV_B32);
    auto v_1 = simd_inst::vld_x2<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((((local_mt_1 * 4) + kv_1) % 3) * 8192) + (_tmp_1 * 256)) + 128)])), DINTLV_B32);
    auto e00 = simd_inst::vexpdif(v.v0, max0_1, softmax_full);
    auto e10 = simd_inst::vexpdif(v_1.v0, max1_1, softmax_full);
    auto e01 = simd_inst::vexpdif(v.v1, max0_1, softmax_full);
    auto e11 = simd_inst::vexpdif(v_1.v1, max1_1, softmax_full);
    auto row_sum0 = simd_inst::vcadd(simd_inst::vadd(e00, e01, softmax_full, MODE_ZEROING), softmax_full, MODE_ZEROING);
    auto row_sum1 = simd_inst::vcadd(simd_inst::vadd(e10, e11, softmax_full, MODE_ZEROING), softmax_full, MODE_ZEROING);
    simd_inst::vsts_1st(row_sum0, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((_tmp_1 * 2) + 57856)])), 0, softmax_one);
    simd_inst::vsts_1st(row_sum1, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((_tmp_1 * 2) + 57857)])), 0, softmax_one);
    auto even_bf16_0 = simd_inst::vcvt<bfloat16_t>(e00, softmax_full, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
    auto even_bf16_1 = simd_inst::vcvt<bfloat16_t>(e10, softmax_full, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
    auto odd_bf16_0 = simd_inst::vcvt<bfloat16_t>(e01, softmax_full, ROUND_R, RS_DISABLE, PART_ODD, MODE_ZEROING);
    auto odd_bf16_1 = simd_inst::vcvt<bfloat16_t>(e11, softmax_full, ROUND_R, RS_DISABLE, PART_ODD, MODE_ZEROING);
    auto merged_bits0 = simd_inst::vor((*(vector_u16 *)(&(even_bf16_0))), (*(vector_u16 *)(&(odd_bf16_0))), softmax_b16, MODE_ZEROING);
    auto merged_bits1 = simd_inst::vor((*(vector_u16 *)(&(even_bf16_1))), (*(vector_u16 *)(&(odd_bf16_1))), softmax_b16, MODE_ZEROING);
    dst_nz_ptr = simd_inst::vsstb((*(vector_bf16 *)(&(merged_bits0))), (__ubuf__ bfloat16_t*)dst_nz_ptr, 4259841, softmax_b16, POST_UPDATE);
    dst_nz_ptr = simd_inst::vsstb((*(vector_bf16 *)(&(merged_bits1))), (__ubuf__ bfloat16_t*)dst_nz_ptr, 4259841, softmax_b16, POST_UPDATE);
  }
  simd_inst::mem_bar(VST_VLD);
  auto alpha_reg = simd_inst::vexpdif(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[57792])), 0), simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[57664])), 0), softmax_full);
  simd_inst::vsts_norm(alpha_reg, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((((local_mt_1 * 4) + kv_1) % 3) * 64) + 57472)])), 0, softmax_full);
  auto l_scaled = simd_inst::vmul(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[57728])), 0), alpha_reg, softmax_full, MODE_ZEROING);
  auto l_updated = simd_inst::vadd(l_scaled, simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[57856])), 0), softmax_full, MODE_ZEROING);
  simd_inst::vsts_norm(l_updated, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[57728])), 0, softmax_full);
}

__simd_vf__ inline void main_kernel_simd_vf_4(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_bool normalize_mask = asc_create_mask_b32(PAT_ALL);
  for (int32_t i = 0; i < 64; ++i) {
    auto denominator = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(i + 57728)])), 0);
    for (int32_t _tmp = 0; _tmp < 2; ++_tmp) {
      auto normalized = simd_inst::vdiv_0ulp_ftz_true(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((i * 128) + (_tmp * 64)) + 49280)])), 0), denominator, normalize_mask, MODE_ZEROING);
      simd_inst::vsts_norm(normalized, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((i * 128) + (_tmp * 64)) + 49280)])), 0, normalize_mask);
    }
  }
}

extern "C" __global__ __mix__(1, 2) void main_kernel(__gm__ bfloat16_t* K, __gm__ float* O, __gm__ bfloat16_t* Q, __gm__ bfloat16_t* V) {
  asc_init();
  __cbuf__ uint8_t *buf_dyn_l1 = (__cbuf__ uint8_t *)0;
  __cc__ uint8_t *buf_dyn_l0c = (__cc__ uint8_t *)0;
  __cb__ uint8_t *buf_dyn_l0b = (__cb__ uint8_t *)0;
  __ca__ uint8_t *buf_dyn_l0a = (__ca__ uint8_t *)0;
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  if ASC_IS_AIC {
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(3));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(4));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(5));
    asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(0));
    asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(1));
    asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(2));
    asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(3));
    asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
    asc_sync_intra_arrive(PIPE_MTE1, 5);
    asc_sync_intra_arrive(PIPE_MTE1, 21);
    asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(6));
    for (int32_t local_mt = 0; local_mt < 2; ++local_mt) {
      asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(6));
      asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(128), 0);
      asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 98304), (&(Q[(local_mt * 16384)])), 256, static_cast<asc_load_l2_cache_mode>(0), 128, 128, 0, 0);
      asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(6));
      for (int32_t kv = 0; kv < 8; ++kv) {
        if ((kv < 5) && (1 <= kv)) {
          asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(((((local_mt * 4) + kv) + 2) % 3)));
          asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(128), 0);
          asc_copy_gm2l1_dn2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((((((local_mt * 4) + kv) + 2) % 3) * 16384) + 49152)), (&(V[((kv * 16384) - 16384)])), 256, static_cast<asc_load_l2_cache_mode>(0), 128, 128, 0, 0);
          asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(((((local_mt * 4) + kv) + 2) % 3)));
        }
        if ((kv < 7) && (3 <= kv)) {
          asc_sync_intra_wait(PIPE_MTE1, 27);
          asc_sync_intra_wait(PIPE_MTE1, 11);
          asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a + 16384), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 114688), 0, 0, 8, 8, 8, 8);
          asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          asc_sync_intra_arrive(PIPE_MTE1, 5);
          asc_sync_intra_arrive(PIPE_MTE1, 21);
        }
        if (kv < 4) {
          asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(((((local_mt * 4) + kv) % 3) + 3)));
          asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(128), 0);
          asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((((local_mt * 4) + kv) % 3) * 16384)), (&(K[(kv * 16384)])), 256, static_cast<asc_load_l2_cache_mode>(0), 128, 128, 0, 0);
          asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(((((local_mt * 4) + kv) % 3) + 3)));
        }
        if ((kv < 7) && (3 <= kv)) {
          asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(((kv + 1) & 1)));
          asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
          asc_mmad(((__cc__ float*)buf_dyn_l0c + ((((kv + 1) & 1) * 16384) + 32768)), ((__ca__ bfloat16_t*)buf_dyn_l0a + 16384), ((__cb__ bfloat16_t*)buf_dyn_l0b + 16384), 128, 128, 128, 0, 1, 0, (bool)1);
          asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(((kv + 1) & 1)));
          asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
        }
        if (kv < 4) {
          if (kv == 0) {
            asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(6));
          }
          asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
          asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 98304), 0, 0, 8, 8, 8, 8);
          if (kv == 3) {
            asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(6));
          }
        }
        if (4 <= kv) {
          asc_sync_intra_wait(PIPE_FIX, ((kv & 1) + 19));
          asc_sync_intra_wait(PIPE_FIX, ((kv & 1) + 3));
          asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>((kv & 1)));
          asc_set_l0c_copy_nz_para(1, 0, 0);
          asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[(((kv & 1) * 8192) + 24576)])), ((__cc__ float*)buf_dyn_l0c + (((kv & 1) * 16384) + 32768)), 128, 128, 128, 128, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
          asc_sync_intra_arrive(PIPE_FIX, ((kv & 1) + 9));
          asc_sync_intra_arrive(PIPE_FIX, ((kv & 1) + 25));
          asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>((kv & 1)));
        }
        if (kv < 4) {
          asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(((((local_mt * 4) + kv) % 3) + 3)));
          asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((((local_mt * 4) + kv) % 3) * 16384)), 0, 0, 8, 8, 8, 8);
          asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
          asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(((((local_mt * 4) + kv) % 3) + 3)));
        }
        if ((kv < 6) && (2 <= kv)) {
          asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
          asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(((((local_mt * 4) + kv) + 1) % 3)));
          asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b + 16384), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((((((local_mt * 4) + kv) + 1) % 3) * 16384) + 49152)), 0, 0, 8, 8, 8, 8);
          asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(((((local_mt * 4) + kv) + 1) % 3)));
        }
        if (kv < 4) {
          asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(((kv & 1) + 2)));
          asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
          asc_mmad(((__cc__ float*)buf_dyn_l0c + ((kv & 1) * 16384)), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b), 128, 128, 128, 0, 1, 0, (bool)1);
          asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(((kv & 1) + 2)));
          asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
        }
        if ((kv < 5) && (1 <= kv)) {
          asc_sync_intra_wait(PIPE_FIX, (((((local_mt * 4) + kv) + 2) % 3) + 16));
          asc_sync_intra_wait(PIPE_FIX, ((((local_mt * 4) + kv) + 2) % 3));
          asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>((((kv + 1) & 1) + 2)));
          asc_set_l0c_copy_nz_para(1, 0, 0);
          asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[(((((local_mt * 4) + kv) + 2) % 3) * 8192)])), ((__cc__ float*)buf_dyn_l0c + (((kv + 1) & 1) * 16384)), 128, 128, 128, 128, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
          asc_sync_intra_arrive(PIPE_FIX, (((((local_mt * 4) + kv) + 2) % 3) + 6));
          asc_sync_intra_arrive(PIPE_FIX, (((((local_mt * 4) + kv) + 2) % 3) + 22));
          asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>((((kv + 1) & 1) + 2)));
        }
      }
    }
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
    asc_sync_intra_wait(PIPE_FIX, 0);
    asc_sync_intra_wait(PIPE_FIX, 1);
    asc_sync_intra_wait(PIPE_FIX, 2);
    asc_sync_intra_wait(PIPE_FIX, 16);
    asc_sync_intra_wait(PIPE_FIX, 17);
    asc_sync_intra_wait(PIPE_FIX, 18);
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(3));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(4));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(5));
    asc_sync_intra_wait(PIPE_FIX, 3);
    asc_sync_intra_wait(PIPE_FIX, 4);
    asc_sync_intra_wait(PIPE_FIX, 19);
    asc_sync_intra_wait(PIPE_FIX, 20);
    asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(0));
    asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(1));
    asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(2));
    asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(3));
    asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
    asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(6));
  }
  if ASC_IS_AIV {
    int32_t sid = asc_get_sub_block_id();
    asc_sync_intra_arrive(PIPE_V, 0);
    asc_sync_intra_arrive(PIPE_V, 1);
    asc_sync_intra_arrive(PIPE_V, 2);
    asc_sync_intra_arrive(PIPE_V, 3);
    asc_sync_intra_arrive(PIPE_V, 4);
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(2));
    for (int32_t local_mt_1 = 0; local_mt_1 < 2; ++local_mt_1) {
      main_kernel_simd_vf_0(buf_dyn_shmem);
      asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(2));
      main_kernel_simd_vf_1(buf_dyn_shmem);
      for (int32_t kv_1 = 0; kv_1 < 7; ++kv_1) {
        if (3 <= kv_1) {
          asc_sync_intra_wait(PIPE_V, (((kv_1 + 1) & 1) + 9));
          main_kernel_simd_vf_3(buf_dyn_shmem, local_mt_1, kv_1);
          asc_sync_intra_arrive(PIPE_V, (((kv_1 + 1) & 1) + 3));
        }
        if ((kv_1 < 5) && (1 <= kv_1)) {
          asc_sync_intra_wait(PIPE_MTE3, 5);
          asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(((kv_1 + 1) & 1)));
          asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((sid * 1024) + 114688))), (__ubuf__ void*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((((kv_1 + 1) & 1) * 8320) + 81920)]))), 8, 64, 1, 64);
          asc_sync_intra_arrive(PIPE_MTE3, 11);
          asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(((kv_1 + 1) & 1)));
        }
        if (kv_1 < 4) {
          asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>((kv_1 & 1)));
          asc_sync_intra_wait(PIPE_V, ((((local_mt_1 * 4) + kv_1) % 3) + 6));
          main_kernel_simd_vf_2(buf_dyn_shmem, kv_1, local_mt_1);
          asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>((kv_1 & 1)));
          asc_sync_intra_arrive(PIPE_V, (((local_mt_1 * 4) + kv_1) % 3));
        }
      }
      main_kernel_simd_vf_4(buf_dyn_shmem);
      asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(2));
      asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(2));
      asc_copy_ub2gm_align((__gm__ uint8_t*)((&(O[((local_mt_1 * 16384) + (sid * 8192))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[49280]))), 1, 32768, static_cast<asc_store_l2_cache_mode>(0), 32768, 32768);
      asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(2));
    }
    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
    asc_sync_intra_wait(PIPE_MTE3, 5);
    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(2));
  }
}

