# Compressor backward (Ascend)

Backward of the compressor's pooling stage, the same math as CANN
`aclnnCompressorGrad`. RMSNorm and RoPE stay outside the operator: `d_cmp` is
the gradient of the reduced vector.

For each valid group the forward builds a window, adds APE, applies softmax
over the window, and reduces `softmax * kv_state` across that window. With
`coff=1` the window is the group's own `cmp_ratio` tokens. With `coff=2` it is
interleaved: even rows are the previous group's first `D` projection channels
(missing, and ignored, on the first group of a segment) and odd rows are this
group's second `D` channels. `kv` is that windowed projection, not the product.

```
dV = dC * softmax
dZ = softmax * (dC * kv - sum(softmax * dC * kv))
dX = dV_token @ Wkv + dZ_token @ Wgate
dWkv = dV_token^T @ X
dWgate = dZ_token^T @ X
dApe[pos % cmp_ratio] += dZ
```

`dV_token` and `dZ_token` are scattered back to the token rows. Overlap writes
the two halves into disjoint channel ranges, so the scatter does not use
atomics. APE partials are private to each core and summed in core order.

The vector kernel rounds token gradients to the input dtype before the Cube
GEMMs, matching a cast-then-matmul pipeline. `d_ape` stays fp32. `H` is a
multiple of 256 and `D` is 128 or 512, which covers the C4A (`D=512`,
`coff=2`, `cmp_ratio=4`), C4Li (`D=128`, `coff=2`, `cmp_ratio=4`) and C128A
(`D=512`, `coff=1`, `cmp_ratio=128`) shapes. Short token counts are padded to a
multiple of 256 only inside the GEMMs.

```bash
python examples/ascend/compressor/example_compressor_bwd.py --no-bench
python examples/ascend/compressor/example_compressor_bwd.py
python -m pytest examples/ascend/compressor/test_compressor_bwd.py -q
```

The benchmark shape is `T=8192`, `H=4096`, `D=512`, `coff=2`, `cmp_ratio=4`.
Reported TFLOPS count the four projection GEMMs (`8 * T * coff * D * H`) and
the timed region includes the scatter and APE reduction.
