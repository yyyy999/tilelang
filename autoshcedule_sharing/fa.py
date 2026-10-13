"""Small automatic FlashAttention case using the local, unmodified FA core.

The core is copied from examples/ascend/flash_attention/core.py.
Keeping D=128 and 128-column KV blocks preserves its SIMD packing contract.
"""
from fa_core import flash_attention_fwd


def make_program():
    return flash_attention_fwd(
        q_len=256, kv_len=512, head_dim=128, out_dtype="float32",
        num_blocks=1, manual_schedule=False,
    )


if __name__ == "__main__":
    from dump_ir import main
    main(default_case="fa")

