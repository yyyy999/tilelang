# Reading a CANNsim trace

The performance result is `report/trace_core0.json`. It is a Chrome-trace list.
Duration events are `ph == "X"`. For those events, `ts`, `args.start`,
`args.end`, `args.duration`, and `dur` agreed on the events inspected
(`end = ts + dur`). `name` is the instruction. On CANN 9.0.0, `cat` is the
pipe. On CANN 9.2.0, `cat` is null and the pipe is the `tid` prefix before
`_SLOT`.
`args.extra_info` carries `instr_id` and, for `RV_VEXP`, `Dtype: F32`.

Do not convert these numbers to nanoseconds. `ref_period=0.40` and
`self_period=0.61` appeared in the model log; their unit was not verified.
Simulator wall time is not device time.

## Ignore these outputs

- The "Typical operator execution times" table. It is the static list in
  `typical_operator_times.py`, printed during every run.
- `[Hardware] parallel simulation finish ... cycle: N`. The stored
  64-iteration log and the 2026-11-05 8-iteration log both reported subcore
  cycles 248, 420, and 420. The trace instruction count changed with
  `n_iters`. Use the trace.

`parse_trace.py` prints a median issue interval. For this vexp kernel the
interval alternates 1 and 3, so the median is 1 and `64 / median` is not the
throughput. Prefer `ii_from_span` on a run long enough to show a stable
pattern. The script's `64 ops/cycle` lines are `64 / II` for an fp32
64-lane instruction. They are not a measured device rate.

## Checks before quoting a number

1. `cannsim.log` contains the expected program line, for example
   `vexp_kernel done n_iters=8`.
2. The trace count of the measured instruction equals the launched trip count.
   A collapsed count means the compiler removed work. The vexp source uses a
   distinct UB address per iteration because an identical `vexp(x)` is CSE'd.
3. State the boundary: one instruction's `dur`, the steady issue interval of
   that instruction, the pipe overlap, or the whole trace span. These are
   different quantities.
4. A copy that always moves the full allocated buffer is not the cost of the
   short compute loop. This kernel copies `128 * 64` fp32 elements (32 KiB)
   on MTE2 and MTE3 regardless of `n_iters`.

## Measurements

`RV_VEXP` `dur` was 16 on every issue. `RV_VLDI` `dur` was 9. `RV_VSTI` `dur`
was 9, with a few 10s in the 64-iteration trace.

| Run | How obtained | `RV_VEXP` | Steady II | VF `dur` | MTE2 / MTE3 `dur` |
|---|---|---|---|---|---|
| 64 iterations | Analyzed existing `/root/vexp_kernel_bench/sim_out/cannsim_20260909152037_vexp_kernel_bench`. Not regenerated | n=64, occupancy span 141 | `ii_from_span` 1.984; hist `{1: 32, 3: 31}` | 347 | grouped `MOV_SRC_TO_DST_ALIGNv2` durations 1057 and 631 |
| 8 iterations | Run 2026-11-05, record 36.02 s, `/root/cannsim_skill_verify/sim_out/cannsim_20261105232126_vexp_kernel_bench` | n=8, span 29, `ii_from_span` 1.857 | hist `{1: 4, 3: 3}` | 235 | 1073 / 632 |
| 4 iterations | Run 2026-11-05 from this skill's script, record 32.61 s, `/root/cannsim_skill_from_repo/sim_out/cannsim_20261105232408_vexp_kernel_bench` | n=4, span 21, `ii_from_span` 1.667 | hist `{1: 2, 3: 1}` | 227 | 1062 / 631 |
| 4 iterations | Rerun 2026-11-06, record 21.2 s, `/root/cannsim_recheck_20261106/sim_out/npusim_20261106165045_vexp_kernel_bench`. Cycles after `* 1650` | n=4, span 21, `ii_from_span` 1.667 | gaps 1, 3, 1 | 229 | MTE2 1327, MTE3 627 |

Cite the 64-iteration span for the steady interval of this particular kernel.
The short reruns confirm the latency and the 1,3 spacing; their span fit is
noisier because `n` is small. `RV_VLDI` in the 8-iteration run issued every
cycle (`ii_from_span` 1.0).

Issue order at the start of the 64-iteration compute region: `RV_VLDI` begins
at ts 6075, `RV_VEXP` at 6081, and stores follow the vexp issues. The three
RVEC pipes overlap. Load latency does not fully serialize the loop.

Pipe categories seen in that trace: `SCALAR`, `RVECEX`, `RVECLD`, `RVECST`,
`SCALARLDST`, `PUSHQ`, `MTE2`, `MTE3`, `RVECSU`, `FLOWCTRL`. The single `VF`
event is `cat=PUSHQ`. Its duration is wider than the vexp occupancy (347
versus 141 cycles at 64 iterations; 235 versus 29 at 8). It includes VF
prologue. Say which window a `latency=` annotation uses.

Trace `pid` separates pipes in `trace_core0.json` (for example `RV_VEXP` events
used pid 12). It is not a chip-wide core id. This report is core 0 only.

## CANN 9.2.0 trace unit

The 4-iteration rerun under CANN 9.2.0 wrote a Chrome trace whose
`displayTimeUnit` is `ns` and whose `ts` / `dur` are fractional. One `RV_VEXP`
event:

- `dur` = `0.009696969696969697`
- issue gaps = `0.000606060606060943`, `0.0018181818181814968`, `0.0006060606060604989`

Those gaps divided by `0.0006060606060606` are 1, 3, and 1. The duration
divided by the same step is 16. That matches the integer-cycle reading from
the 9.0.0 traces above. One cycle is `1/1650` us. `parse_trace.py` multiplies
fractional `ts` and `dur` by 1650 and then prints cycles. Quote those printed
numbers as cycles. Do not divide them again. Say that the JSON itself stores
time. Integer 9.0.0 traces are left unchanged.

The 2026-11-06 rerun of this same 4-iteration case, after that conversion,
had `RV_VEXP` `dur` 16 and gaps 1, 3, 1. Its two
`MOV_SRC_TO_DST_ALIGNv2` events were MTE2 1327 cycles and MTE3 627 cycles.
Do not quote the 1062 / 631 row for that rerun. Split the name by `tid`
before fitting an interval. The later MTE3 store is the same instruction
name as the MTE2 loads.

## Hy4 full kernel, 2026-11-06

`nq=1`, `nk=128`, `topk=128`, one core. Archive
`/root/hy4_full_sim/npusim_20261106171909_run_full.py`. The trace is
`report/results/0000_main_kernel_reports/core_0/trace_core0.json`, not the
vexp `unordered_map` directory. 11372 duration events. Core span 13445
cycles. This is one query and two 64-slot tiles, so the prologue is a large
part of the span.

| Pipe | Events | Window (cycles) | What to quote |
|---|---:|---|---|
| MTE2 | 73 | 4450–8817, span 4367 | Overlapped copies. The `dur` sum is 53231, far above the span. |
| CUBE | 18 `MMAD` | 8921–12149, span 3228 | Each `dur` is 89–204. The first issue starts as the MTE2 window ends. |
| FIXP | 12 | 5285–12787 | One event's `dur` is 4450. Do not add FIXP durations into the cube span. |
| MTE3 | 22 | 4450–13445 | Includes the final store. |
| RVECEX subset | 936 of `RV_VEXPDIF`, `RV_VMAX`, `RV_VCVT_F2F` | 6195–13002 | Those counted names overlap the cube window. Whole-trace `RV_VEXPDIF` is 132 and `RV_VMAX` is 164. |

Do not describe this as the 4096-query kernel. Compare a code change by
rerunning this same shape and these same pipes.

## Bottleneck questions

Ask one of these, and keep the others out of the quoted interval:

- **Issue cost of one vector op.** Use `dur` when the producer is already
  done. Here one fp32 `vexp` is 16 cycles.
- **Throughput of a stream.** Use the steady `ts` spacing, checked on more
  than one trip count. Dependent issue spacing for this vexp is about 2
  cycles, so a 64-lane issue moves about 32 fp32 elements per cycle inside
  that pipe. `64 / 16` would describe the single-issue latency, not the
  pipelined rate.
- **Memory.** Quote the MTE2 or MTE3 event for the actual byte count, and
  name the run. The 2026-11-05 copies are about 1060 and 630 cycles for the
  fixed 32 KiB buffer, larger than the whole 64-iteration vexp occupancy.
  The 2026-11-06 rerun measured MTE2 1327 and MTE3 627 for that same buffer.
- **Overlap.** Compare start and end across `tid` pipe prefixes. On CANN
  9.0.0 the pipe was `cat`. On 9.2.0 `cat` is null. Overlapping pipes do not
  add into one elapsed time.
- **Whole kernel.** Use the first measured instruction through the last
  completion on every participating core. Core 0's trace does not answer a
  multi-core launch. Multi-core report was not run.

After the simulation, a `T.SimdVF(latency=...)` or `T.Task(latency=..., ii=...)`
value is only a scheduler input. Device latency still comes from A5.
