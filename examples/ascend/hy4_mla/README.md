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

## BF16 call adapter

[ADAPTATION.md](ADAPTATION.md) defines the model-call contract and validation status.
The compatibility entry point is `interface.flash_mla_sparse_fwd`:

```python
from examples.ascend.hy4_mla.interface import flash_mla_sparse_fwd

# q [N,H,576], kv [M,1,576], indices [N,1,K]; BF16 q/kv/out.
output, maximum, lse = flash_mla_sparse_fwd(
    q, kv, indices, sm_scale,
    attn_sink=attn_sink, topk_length=topk_length, out=out,
)
```

The wrapper selects sink/length specializations, flattens single-head views and
pads H<64 queries/sinks. H64 writes directly to a supplied output, including
supported positive-pitch workspace views. Small-head calls compute a temporary
H64 output before copying the real heads to `out`. FP8 Decode is not included.
The model/framework still needs to select this Ascend backend; adding an example
entry point does not register it in vLLM or Transformers.

CPU-only adapter checks use a mock executor, without loading TileLang native libraries:

```bash
python -m unittest examples.ascend.hy4_mla.test_interface -v
```

Device acceptance for the new adapter is a separate, not-yet-validated step:

```bash
python -m pytest examples/ascend/hy4_mla/test_interface_npu.py -x -q
```

## Run

Run the source-level resource and alias checks locally first (stdlib only):

```bash
python examples/ascend/hy4_mla/check_resources.py --details
python -m unittest examples.ascend.hy4_mla.test_static -v
```

See [STATIC_CHECKS.md](STATIC_CHECKS.md) for budgets, address analysis and the
boundary between source checks, compiler checks and device acceptance.

With a matching native build, generate source without device compilation or
launching a kernel:

```bash
python -m examples.ascend.hy4_mla.check_lowering --topk 64 2048 --output /tmp/hy4-lowering
```

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
| `mfu.py` | FLOP and MFU formula; it takes a measured time and does not supply one |
| `check_resources.py`, `test_static.py` | Local allocation/alias/address budget checks without TileLang/NPU |
| `check_lowering.py` | Offline native lowering/source generation and final allocation capture |

The initial implementation is under validation. A successful syntax check alone
does not establish compilation, device correctness or high performance.
