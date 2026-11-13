---
name: cannsim-optimization
description: >
  Run and interpret npusim/CANNsim cycle simulations for TileLang Ascend
  operator tuning on the local WSL CANN 9.2.0 / Ascend950 (dav-3510) install.
  Use when the task mentions CANNsim, cannsim, npusim, cycle simulation,
  instruction latency, issue interval, pipe overlap, vector or copy
  microbenchmarks, a full TileLang kernel under the simulator, or Ascend
  operator tuning before A5. Covers the verified environment, the vexp
  microbench, the Hy4 full-kernel record, Chrome-trace analysis, and the
  WSL-versus-A5 sync rule.
---

# CANNsim optimization

Use local WSL for CANNsim. Use A5 only for numerical correctness and real-device
performance. Follow [sync.md](sync.md) and the `a5-server` skill. Do not copy a
tree onto A5.

The default toolchain is official CANN 9.2.0. `cannsim` still runs, and it
prints that the command has been renamed to `npusim`. Use `npusim`.

## Check the environment

Run in WSL Ubuntu-24.04. Non-interactive shells do not load `~/.bashrc`, so
source the toolkit in the same shell that compiles or records:

```bash
source /usr/local/Ascend/cann/set_env.sh
command -v npusim
command -v bisheng
test -d "$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9589/camodel"
python3 -c "import plotly"
```

Verified on 2026-11-05 after the switch: `/usr/local/Ascend/cann` points at
`/usr/local/Ascend/cann-9.2.0`. Toolkit inner version
`V100R001C12SPC001B237`, compiler timestamp `20261015_201912253`. Bisheng is
`clang version 15.0.5` dated `2026-10-15T17:21:21+08:00`. CLI soc remains
`Ascend950`. The 4-iteration vexp run loaded model `Ascend950PR_9589`.
`cann-9.0.0` is still on disk and is not selected. Details:
[environment.md](environment.md).

A5's installed tree is the same 9.2.0 product line at inner version B236,
timestamp `20261014_000123808`. These two installs are not the same bytes.

## Run the verified case

Always pass an output directory on the Linux filesystem. The script's default
deletes `/root/vexp_kernel_bench/sim_out`.

```bash
bash /mnt/d/codes/yyz/tilelang/.agents/skills/cannsim-optimization/examples/build_vexp_kernel.sh \
  /root/cannsim_skill_from_repo 4
```

This compiles `examples/vexp_kernel.asc` with
`bisheng -O2 -std=c++17 --npu-arch=dav-3510 -lascendcl` and runs:

```bash
npusim record "$OUT/vexp_kernel_bench" -s Ascend950 -o "$OUT/sim_out" -g -u 4
```

`-u` is the program argv (`n_iters`).

On CANN 9.2.0 this command was run with `n_iters=4` on 2026-11-05, and the
same case was rerun on 2026-11-06.
The program printed `vexp_kernel done n_iters=4 lanes=64` and npusim printed
`Simulation SUCCESS` (run time 19.9s). The archive was
`sim_out/npusim_<timestamp>_vexp_kernel_bench/`. Auto-report failed until
system Python had `plotly` (`python3 -m pip install --break-system-packages plotly`).
After that, this produced the trace:

```bash
npusim report -e "$ARCHIVE" -o "$OUT/report"
```

The JSON was
`report/results/0000_std::unordered_map_reports/core_0/trace_core0.json`.
`displayTimeUnit` is `ns`. Chrome `ts` and `dur` are fractional microseconds,
not the integer cycles written by CANN 9.0.0. One cycle is `1/1650` us.
`parse_trace.py` detects fractional values and multiplies by 1650 before
printing, so its output is already in cycles. Do not divide those printed
numbers again. On 9.2.0, `cat` is null and the pipe is the `tid` prefix
before `_SLOT` (`RVECEX_SLOT00`, `MTE2_SLOT00`).

The same 4-iteration case was rerun on 2026-11-06
(`/root/cannsim_recheck_20261106`, 21.2s, model `Ascend950PR_9589`). After
conversion, `RV_VEXP` `dur` was 16 and the issue gaps were 1, 3, 1. The two
`MOV_SRC_TO_DST_ALIGNv2` events were MTE2 1327 cycles and MTE3 627 cycles.
Do not quote the 2026-11-05 copy costs for this rerun.

The 9.0.0 runs (4, 8, and the stored 64) remain in
[analysis.md](analysis.md). Their archive layout was
`cannsim_<timestamp>_vexp_kernel_bench/report/trace_core0.json`.

Keep one `npusim record` at a time. It runs the binary from the binary's
parent directory.

## Read the result

```bash
python3 /mnt/d/codes/yyz/tilelang/.agents/skills/cannsim-optimization/examples/parse_trace.py \
  /root/cannsim_skill_from_repo/sim_out
```

Use the trace, not the static "Typical operator execution times" table printed
while `npusim record` runs, and not the hardware `cycle:` lines. Full rules
and measured numbers: [analysis.md](analysis.md). Split events by `tid` pipe
before fitting an issue interval. `MOV_SRC_TO_DST_ALIGNv2` contains both the
MTE2 loads and the later MTE3 store; one histogram of that name mixes them.

Quote, for the repeated `RV_VEXP` region:

- issue count equals the intended loop count
- `dur` of one issue (16 cycles for this fp32 `vexp` in every verified trace)
- steady issue interval from a long run's span, not the median of a 1,3 alternation
- which pipes overlap (`RVECLD`, `RVECEX`, `RVECST`, `MTE2`, `MTE3`)
- copy cost only when the copy is inside the question being asked

The 64-iteration artifact gives the steady interval to cite for this kernel:
`ii_from_span = 1.984` cycles, with every `RV_VEXP` `dur = 16`. That file was
analyzed, not regenerated, on 2026-11-05. The new 4- and 8-iteration runs
reproduced `dur = 16` and the 1,3 issue spacing.

## From a TileLang operator to a case

The compiled Hy4 kernel can be recorded directly. On 2026-11-06,
`examples/ascend/hy4_mla/sim/run_full.py` compiled `hy4_sparse_attention`
(`nq=1`, `nk=128`, `topk=128`, `num_cores=1`) with `/root/cannsim-venv`
(`torch==2.7.1+cpu`, `torch-npu==2.7.1.post11`) and
`PYTHONPATH=/root/tilelang-hy4`. `npusim record` of that program printed
`hy4_full_kernel done shape=(1, 64, 512)` and `Simulation SUCCESS` in 36.2s.
The model was still `Ascend950PR_9589`. Core 0's trace had 11372 events,
including 18 `MMAD`s, MTE1/MTE2/MTE3, FIXP, and vector ops such as
`RV_VEXPDIF`. Archive:
`/root/hy4_full_sim/npusim_20261106171909_run_full.py`.

That launch is the whole operator at a one-query grid. It is not the
4096-query steady state, and its cycles are not device time. Core 0 spanned
13445 cycles. The 18 `MMAD`s occupied cycles 8921–12149 (3228 cycles, each
`dur` 89–204). The 73 MTE2 events ended at cycle 8817, immediately before
the first `MMAD`. Vector work overlapped that cube window and included 132
`RV_VEXPDIF` and 164 `RV_VMAX`. On this shape the cube is about a quarter of
the core span because the prologue copies finish first. Do not use this trace
to explain a 4096-query device MFU.

Repeat that comparison with
`bash examples/ascend/hy4_mla/sim/run_full.sh /root/hy4_full_sim`. The script
deletes the output directory. Build and import details: [tilelang.md](tilelang.md).

`sim/gather_mte.asc` is not this kernel. It is a hand-written AIV program
that copies contiguous `2×576` pairs or one `32×576` block. The lowered
`gather_kv` keeps a 640-wide UB row and a runtime gap between the two
selected rows. Do not cite the gather microbench as the operator's cost.

A hand-written AscendC launcher is still the way to isolate one instruction.
Use the same shape as `examples/vexp_kernel.asc`. Apply a measured cost with
`T.SimdVF(latency=...)` or `T.Task(latency=..., ii=...)` as described in
`tilelang-ascend`. Recompile. Correctness and whole-kernel device time still
have to be checked on A5. Simulated cycles are not device latency.

## Pitfalls found on 2026-11-06

- Source `set_env.sh` before importing `torch_npu`. Without it, import fails
  on `libhccl.so` even though
  `/usr/local/Ascend/cann-9.2.0/lib64/libhccl.so` is installed.
- Prepend the TileLang tree to `PYTHONPATH`. Replacing `PYTHONPATH` drops
  `$ASCEND_HOME_PATH/python/site-packages`, and `npusim` then raises
  `No module named cannsim`. `npusim`'s shebang is `/usr/bin/python3`, so
  the simulator process is not the venv interpreter.
- Run Python from the built tree. A shell started in
  `/mnt/d/codes/yyz/tilelang` imports that checkout first, and its `build/`
  directory does not exist.
- The workspace `3rdparty/tvm` gitlink is empty. `/root/tilelang/3rdparty`
  has the same TVM commit `4211874e`. Copy that tree before configuring.
  Leave `/root/tilelang` and `/root/tilelang-venv` alone. The simulation
  venv is `/root/cannsim-venv`.
- `-s Ascend950` always loads `Ascend950PR_9589`
  (`cannsim/core/record/paths.py`). There is no `Ascend950PR_9579` model
  directory. A5's 9579 timings are a different SKU.
- The tool banner asks for more than 32 GiB. This machine has about 15 GiB.
  The Hy4 run's resident size was 8.21 GiB and it finished. Do not move the
  simulation to A5, and do not resize WSL, only because of that banner.
- Keep one `npusim record` at a time. A second record shares the CAModel
  directory.
- From PowerShell, `>` and `$(...)` are expanded before `wsl` sees them.
  Put the bash script in a file and execute that file.
- A 4096×2048 launch is the same kernel with a much larger grid. One query
  and two tiles already took 36s. Do not start the full grid expecting a
  short iteration. No reference comparison was run. The printed bf16 shape
  shows the kernel completed; it does not show numerical correctness.

## WSL and A5

- Edit, compile, and simulate in WSL. The CANNsim binaries and sim output used
  above live on the Linux disk (`/root/...`).
- `/root/tilelang-venv` imports `/root/tilelang` and has CUDA `torch`
  2.13 without `torch_npu`. Do not install the simulator stack into it.
  Full-kernel simulation uses `/root/cannsim-venv` and
  `/root/tilelang-hy4`. Confirm the commit before trusting either tree.
  Details: [sync.md](sync.md).
- A5 receives code only by git: commit, push the workspace remote, then pull
  that branch or commit on A5. Follow `a5-server`. Do not scp, rsync, or copy
  the tree to A5. No A5 run was part of this verification.

## Not verified

Multi-core `npusim report -n`, `instr.bin` decoding, simulation with outputs
on `/mnt/d`, a 4096-query Hy4 launch, and any A5 comparison. The 2026-11-06
Hy4 run is one core of one query. Say so when a conclusion depends on one of
these. Do not fill the gap with the `npusim` commands in `tilelang-ascend`.
