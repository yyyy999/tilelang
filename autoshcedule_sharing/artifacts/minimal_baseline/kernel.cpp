#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
extern "C" __global__ __cube__ void main_kernel(__gm__ bfloat16_t* A) {
  asc_init();
  __cbuf__ bfloat16_t *l1 = (__cbuf__ bfloat16_t *)0;
  __ca__ bfloat16_t *l0 = (__ca__ bfloat16_t *)0;
  int32_t l1_version_counter_1 = 0;
  asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
  asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
  l1_version_counter_1 = 0;
  for (int32_t i = 0; i < 4; ++i) {
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(((3 & ((l1_version_counter_1 % 3) >> 31)) + (l1_version_counter_1 % 3))));
    asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0);
    asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)l1 + (((3 & ((l1_version_counter_1 % 3) >> 31)) * 4096) + ((l1_version_counter_1 % 3) * 4096))), (&(A[(i * 4096)])), 128, static_cast<asc_load_l2_cache_mode>(0), 64, 64, 0, 0);
    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(((3 & ((l1_version_counter_1 % 3) >> 31)) + (l1_version_counter_1 % 3))));
    asc_sync_pipe(PIPE_MTE1);
    asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(((3 & ((l1_version_counter_1 % 3) >> 31)) + (l1_version_counter_1 % 3))));
    asc_copy_l12l0a(((__ca__ bfloat16_t*)l0), ((__cbuf__ bfloat16_t*)l1 + (((3 & ((l1_version_counter_1 % 3) >> 31)) * 4096) + ((l1_version_counter_1 % 3) * 4096))), 0, 0, 4, 4, 4, 4);
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(((3 & ((l1_version_counter_1 % 3) >> 31)) + (l1_version_counter_1 % 3))));
    l1_version_counter_1 = (l1_version_counter_1 + 1);
  }
  asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
  asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
}

