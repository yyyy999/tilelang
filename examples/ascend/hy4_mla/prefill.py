"""Compile the H64 BF16 Hy4 sparse MLA core (QK576, V512)."""


def compile_bf16(nq, nk, topk=2048, *, scale, sink=True, variable_lengths=False, pipeline_stages=True):
    import tilelang

    from examples.ascend.hy4_mla.kernel import hy4_sparse_attention

    return tilelang.compile(
        hy4_sparse_attention(nq, nk, topk, scale=scale, sink=sink, variable_lengths=variable_lengths, pipeline_stages=pipeline_stages),
        target="ascend",
        out_idx=[5, 6, 7],
        compile_flags=["-Ofast", "-mllvm", "-enable-hiipu-vf-loop-unroll"],
    )
