# Hy4 preview sparse MLA on Ascend

Development implementation adapted from [FlashMLA](../flashmla/README.md).
See [TARGET.md](TARGET.md) for model evidence, parameter derivation and scope.

The BF16 core accepts contiguous Q `[nq,64,576]`, KV `[nk,576]`, int32
indices `[nq,topk]`, float32 sink logits `[64]`, and int32 lengths `[nq]`.
It returns BF16 output `[nq,64,512]` and float32 maximum/LSE `[nq,64]`.
`scale` is explicit (Hy4 preview normally uses `1/16`); topk must be a
positive multiple of 64. Lengths are consumed only with `variable_lengths=True`.
Callers must supply lengths in `[0,topk]` and contiguous tensors of these dtypes.

The QK product sums the latent512 and RoPE64 products before softmax. PV uses
only latent512. Sink contributes to the denominator with zero value;
maximum/LSE exclude sink. Invalid indices (negative or >= nk) are ignored,
duplicates retain multiplicity, and lengths mask a prefix before validity
filtering. Empty rows return zero output, -inf maximum and +inf LSE.
Indices must already encode causal/sequence eligibility; there is no additional
causal mask. No paged or quantized KV cache adapter is included yet.

## Run

Run the source-level resource and alias checks locally first (stdlib only):

```bash
python examples/ascend/hy4_mla/check_resources.py --details
python -m unittest examples.ascend.hy4_mla.test_static -v
```

See [STATIC_CHECKS.md](STATIC_CHECKS.md) for budgets, address analysis and the
boundary between source checks, compiler checks and device acceptance.

Build the current TileLang checkout with Ascend support. From the repository root:

```bash
export PYTHONPATH=.
export ASCEND_RT_VISIBLE_DEVICES=0
python -m pytest examples/ascend/hy4_mla/test_reference.py -q
python -m pytest examples/ascend/hy4_mla/test_hy4_mla.py -x -q
python -m examples.ascend.hy4_mla.run --nq 33 --nk 257 --topk 2048
```

The runner checks against the independent FP32 PyTorch reference before optional
NPU-event timing (`--repeat 20`). Timing includes kernel launches and uses warm
caches; it is not comparable to FlashMLA's L2-flushed profiler results. Concurrent
device jobs can distort results. There is no Hy4 performance claim yet.

## Files

| File | Purpose |
| --- | --- |
| `kernel.py` | Persistent sparse gather, QK512+64, softmax/sink, PV512 pipeline |
| `prefill.py` | `compile_bf16` compilation entry point |
| `reference.py` | FP32 mathematical reference, independent of TileLang/NPU |
| `testing.py` | Reproducible boundary inputs and numerical checks |
| `test_reference.py` | Hand-computable CPU semantic tests |
| `test_hy4_mla.py` | NPU correctness and persistent-query regressions |
| `run.py` | Single-case verification and optional warm-cache timing |
| `check_resources.py`, `test_static.py` | Local allocation/alias/address budget checks without TileLang/NPU |

The initial implementation is under validation. A successful syntax check alone
does not establish compilation, device correctness or high performance.
