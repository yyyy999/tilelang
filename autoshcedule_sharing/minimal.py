"""Two-copy scheduling example.

Source: testing/ascend/auto_schedule/test_schedule_integration.py,
_make_counter_l1_layout_program. The sharing variant enables offsets and uses
iteration-indexed versions, since counter mode cannot span multiple stages.
This compile-only example has no numerical output.
"""
import tilelang.ascend.language as T


def make_program(enable_offset=False, version_mode=None):
    version_mode = version_mode or ("iteration" if enable_offset else "counter")
    @T.prim_func
    def main(A: T.Tensor((256, 64), "bfloat16")):
        with T.Kernel(1):
            l1 = T.alloc_l1((64, 64), "bfloat16")
            l0 = T.alloc_l0a((64, 64), "bfloat16")
            T.annotate_buffer_versions({l1: (3, version_mode)})
            for i in T.Pipelined(
                4,
                num_stages=3,
                annotations={"multi_buffer_eligible": [l1], "enable_offset": enable_offset},
            ):
                T.copy(A[i * 64 : (i + 1) * 64, :], l1)
                T.copy(l1, l0)

    return main


if __name__ == "__main__":
    from dump_ir import main
    main(default_case="minimal")

