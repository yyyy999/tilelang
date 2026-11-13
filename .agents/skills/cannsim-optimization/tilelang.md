# TileLang cases

## Verified shape

`examples/vexp_kernel.asc` is the case that was compiled and simulated on
2026-11-05. It is a host `main` plus one AIV kernel:

- `__global__ __vector__` entry launched as `kernel<<<1, 0, stream>>>(...)`
- `__simd_vf__` loop with `#pragma clang loop unroll(disable)`
- each iteration: `vlds` of a distinct 64-lane fp32 UB tile, `vexp`, `vsts`
- `AscendC::DataCopy` GM→UB before the VF and UB→GM after it, with
  `PipeBarrier<PIPE_ALL>` around the measured VF
- ACL setup: `aclInit`, `aclrtSetDevice(0)`, `aclrtCreateStream`,
  `aclrtMalloc`, `aclrtMemcpy`, launch, `aclrtSynchronizeStream`, then teardown

Bisheng line that succeeded:

```bash
bisheng -O2 -std=c++17 --npu-arch=dav-3510 \
  -I"$ASCEND_HOME_PATH/include" \
  vexp_kernel.asc -o vexp_kernel_bench \
  -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath,"$ASCEND_HOME_PATH/lib64" -lascendcl
```

Match this shape when the question is a single Vector VF. Preserve dtype,
lane count, trip count, masks, and addressing. Give each iteration a distinct
source or destination if a repeated identical operation would be optimized
away. Initialize and copy inputs before the region you will time. Put a
required drain after the stream, not a barrier between independent issues
when the question is throughput.

`examples/parse_trace.py` prints cycles. On a 9.2.0 trace it multiplies
fractional microseconds by 1650. It only summarizes names it knows
(`RV_VEXP`, vector loads and stores, `VF`, `MOV_SRC_TO_DST_ALIGNv2`, and a
few broadcasts). For `MMAD` or another missing name, count `name` and the
`tid` pipe directly. A missing name in the script's list is not evidence
that the instruction did not run. Do not fit one issue interval across
`MOV_SRC_TO_DST_ALIGNv2` until MTE2 and MTE3 rows are separated.

## Scheduler annotations

`tilelang-ascend` owns the annotation rules. Use a measured `latency` for
output-ready delay and a measured `ii` for sustained issue spacing of that
same task body. The vexp numbers in [analysis.md](analysis.md) belong to this
microbench's fp32 64-lane loop, not to an arbitrary TileLang VF. Recompile
after changing an annotation. They do not replace an A5 correctness or
end-to-end timing run.

## Full Hy4 kernel

On 2026-11-06 `npusim record` ran the compiled `hy4_sparse_attention`
kernel. It was not a pasted `.asc` file. The program is
`examples/ascend/hy4_mla/sim/run_full.py`. The launcher is
`examples/ascend/hy4_mla/sim/run_full.sh`.

The simulation venv is `/root/cannsim-venv`:

```bash
python3 -m venv /root/cannsim-venv
/root/cannsim-venv/bin/pip install 'torch==2.7.1+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
/root/cannsim-venv/bin/pip install 'torch-npu==2.7.1.post11'
```

`torch-npu==2.7.1.post11` has an x86_64 cp312 wheel and is the CANN 9.2.0 /
Ascend 950 pair. Do not install it into `/root/tilelang-venv`, which has
`torch==2.13.0+cu130` and no `torch_npu`. Source `set_env.sh` before
`import torch_npu`.

The built tree is `/root/tilelang-hy4`. The Cursor workspace's
`3rdparty/tvm` directory is empty. Copy `/root/tilelang/3rdparty` into the
build tree first. Both gitlinks are TVM `4211874e`. Configure from that
ext4 tree, not from `/mnt/d`:

```bash
cmake -S /root/tilelang-hy4 -B /root/tilelang-hy4/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/root/cannsim-venv/bin/python \
  -DUSE_CUDA=OFF -DUSE_ASCEND=ON -DUSE_LLVM=OFF -DUSE_ROCM=OFF \
  -DTILELANG_USE_ASCEND_STUBS=ON
cmake --build /root/tilelang-hy4/build -j 12
```

`/root/tilelang` itself was not rebuilt. `cd /root/tilelang-hy4` before
importing. Otherwise Python selects `/mnt/d/codes/yyz/tilelang/tilelang` and
looks for a missing `build/lib`.

Verified launch: `nq=1`, `nk=128`, `topk=128`, `num_cores=1`,
`scale=1/16`, variable lengths off. Program line:
`hy4_full_kernel done shape=(1, 64, 512) dtype=torch.bfloat16`.
`Simulation SUCCESS` in 36.2s. Model `Ascend950PR_9589`. Numbers are in
[analysis.md](analysis.md). The output shape shows the kernel finished. It
was not checked against the reference.

`sim/gather_mte.asc` remains a copy microbench. Mode 0 issues 16 contiguous
`2×576` copies. Mode 1 issues one `32×576` copy. Neither is the strided
`asc_copy_gm2ub_align` emitted for `gather_kv` into the `32×640` UB tile.
Do not use it as the full operator.

## Not run from TileLang

These were not executed against CANNsim:

- `get_kernel_source()` after `tilelang.compile`. The 2026-11-06 run compiled
  and launched the kernel; it did not dump the generated `.asc`.
- `python -m tilelang.tools.compile_only`. The current
  `docs/tools/compile_only.md` does not document an Ascend target.
- Extracting one VF from generated TileLang source and simulating that text.
- A multi-core Hy4 launch, or `npusim report -n` beyond core 0.
- Feeding a simulated result back into the kernel and comparing with A5.

`/root/tilelang-venv` is the older CUDA checkout and was not used for this
run. The simulation venv is `/root/cannsim-venv`. The built tree is
`/root/tilelang-hy4`, not `/root/tilelang`. See [sync.md](sync.md).

Quote the Hy4 result with its shape. A vexp microbench number is not the
cost of this kernel.
