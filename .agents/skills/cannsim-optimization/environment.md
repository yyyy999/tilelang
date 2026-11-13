# Environment

Checked on 2026-11-05 in WSL distro `Ubuntu-24.04` (`wsl -d Ubuntu-24.04`).
The login used by that invocation is root.

## Current install: CANN 9.2.0

`/usr/local/Ascend/cann` points at `/usr/local/Ascend/cann-9.2.0`.
`/root/.bashrc` sources that symlink, so an interactive shell gets 9.2.0.
A non-interactive shell still needs `set +u; source /usr/local/Ascend/cann/set_env.sh`.

| Item | Verified value |
|---|---|
| Toolkit | `/usr/local/Ascend/cann-9.2.0` |
| inner version | `V100R001C12SPC001B237` |
| compiler / opp timestamp | `20261015_201912253` |
| Public packages | `/root/cann-packages-9.2.0/Ascend-cann-toolkit_9.2.0_linux-x86_64.run` (1362158004 bytes), `Ascend-cann-950-ops_9.2.0_linux-x86_64.run` (2942870822 bytes) |
| `npusim` | `/usr/local/Ascend/cann-9.2.0/bin/npusim`, shebang `/usr/bin/python3`, entry `cannsim.main:main` |
| `cannsim` | same entry; prints that it is renamed to `npusim` |
| Bisheng | `clang version 15.0.5` (`2026-10-15T17:21:21+08:00`) |
| CLI soc | `-s Ascend950` |
| Model loaded by the vexp run | `Ascend950PR_9589` (`tools/simulator/Ascend950PR_9589/camodel/camodel_v100.json`) |
| Compile arch | `--npu-arch=dav-3510` |
| Report | needs `plotly` in system Python. Without it, simulation can succeed and the report step prints `No report backend succeeded` |
| Archive | `npusim_<timestamp>_<binary>/` with `npusim.log` and `record/` |
| Machine | `nproc` = 24, memory about 15 GiB. The tool asks for more than 32 GiB; the 4-iteration vexp still finished |

Checked again on 2026-11-06. The toolkit, model path, Bisheng version, and
`plotly` 7.1.0 were unchanged. `nproc` was 24 and memory was still about
15 GiB. The 4-iteration vexp rerun and the one-query Hy4 run both finished.
The Hy4 process resident size reported by `npusim` was 8.21 GiB.

Full-kernel Python is a separate venv, `/root/cannsim-venv`:

| Item | Verified value |
|---|---|
| Python | 3.12, `/root/cannsim-venv/bin/python` |
| torch | `2.7.1+cpu` from the PyTorch CPU index |
| torch_npu | `2.7.1.post11`, wheel `cp312` `manylinux_2_28_x86_64` |
| TileLang build | `/root/tilelang-hy4/build`, `USE_CUDA=OFF`, `USE_ASCEND=ON` |
| Import check | `torch.npu.is_available()` is false until the process is under `npusim` |

`/root/tilelang-venv` still has `torch==2.13.0+cu130` and no `torch_npu`.
Do not point a simulation at it.

`-s Ascend950` is mapped only to `Ascend950PR_9589` in
`cannsim/core/record/paths.py`. `Ascend950DT` maps to `Ascend950PR_9599`.
The installed simulator tree has no `Ascend950PR_9579` directory. The A5
card used for Hy4 timing was 9579. Say which model a cycle number came from.

A5 uses inner version `V100R001C12SPC001B236` and timestamp
`20261014_000123808` under
`/home/y00897153/yyz-private/cann-9.2.0-b112/cann-9.2.0`. Its package
sizes differ (toolkit 1366507894, 950-ops 2942892899). The public B112 URL
returned 403, so WSL has the public B237 packages.

## Previous install: CANN 9.0.0

The directory `/usr/local/Ascend/cann-9.0.0` is still present. It is not the
default. The table below is the 2026-11-05 record from before the switch.

## Install

| Item | Verified value |
|---|---|
| Toolkit | `/usr/local/Ascend/cann-9.0.0` |
| `set_env.sh` | `/usr/local/Ascend/cann/set_env.sh` (symlink to the 9.0.0 tree) |
| Also sourced from | `/root/.bashrc` for interactive bash only |
| `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` | `/usr/local/Ascend/cann-9.0.0` after `set_env.sh` |
| `cannsim` | `/usr/local/Ascend/cann-9.0.0/bin/cannsim` → package `cannsim` 0.1.0 |
| Package | `/usr/local/Ascend/cann-9.0.0/python/site-packages/cannsim/` |
| Bisheng | `/usr/local/Ascend/cann-9.0.0/bin/bisheng`, `clang version 15.0.5` (`2026-04-25T15:46:25+08:00`) |
| CLI soc | `Ascend950` only (`cannsim/soc_info.py`) |
| Model name | `Ascend950PR_9599` |
| CAModel | `$ASCEND_TOOLKIT_HOME/tools/simulator/Ascend950PR_9599/camodel` → `dav_3510` |
| Compile arch | `--npu-arch=dav-3510` (`ASCEND_NPU_ARCH` default in the verified script) |
| `trace_tools` | `/usr/local/Ascend/cann-9.0.0/python/site-packages/trace_tools/` |
| Machine | `nproc` = 24, memory about 15 GiB |

`cannsim --version` is not a supported flag. `cannsim --help` shows
`record` and `report`.

`SOC_INFO["Ascend950"]` also records `cube_cores: 32` and `vec_cores: 64`.
The model log printed `core count: 24` and, per core wrapper,
`cube_subcore=1, vec_subccore=2, ref_period=0.40, self_period=0.61`.
The driver stub printed `soc_version:Ascend950PR_9589`. These strings were
observed. They were not reconciled, and the period values were not used as a
clock. Report generation in `cannsim record -g` is hardcoded to core 0.

A non-interactive probe before `source set_env.sh` had `cannsim` and `bisheng`
missing and no `ASCEND_*` variables. After sourcing, both commands resolved.

## Commands that were run

Environment probe: source `set_env.sh`, `cannsim record --help`,
`cannsim report --help`, import `cannsim` and `trace_tools`.

Simulation, both successful:

```bash
bash /root/tl-src/examples/ascend/cannsim_exp/build_vexp_kernel.sh \
  /root/cannsim_skill_verify 8
bash /mnt/d/codes/yyz/tilelang/.agents/skills/cannsim-optimization/examples/build_vexp_kernel.sh \
  /root/cannsim_skill_from_repo 4
```

Record time reported by the tool: 36.02 s and 32.61 s. The tool's own note
says CPU > 16 cores and memory > 32 GB, and prints a static table from
`typical_operator_times.py` (FA, matmul, softmax, and so on). That table is
not the measured kernel. The 4- and 8-iteration vector case still finished on
this 15 GiB machine. On 2026-11-06 the same machine also finished the
4-iteration vexp rerun and the one-query Hy4 kernel. Those commands are in
[SKILL.md](SKILL.md).

## CLI

```text
cannsim record [-o OUTPUT] -s SOC_VERSION [-g] [-u USER_OPTION] user_app
cannsim report -e EXPORT [-o OUTPUT] [-n CORE_ID]
```

On CANN 9.0.0, `record` checked that `user_app` was an existing file. CANN
9.2.0 `npusim record` also accepts a quoted command. The Hy4 run used
`npusim record "/root/cannsim-venv/bin/python .../run_full.py" -s Ascend950 -o ... -g`.
The CLI soc is still only `Ascend950`. `ASCEND_TOOLKIT_HOME` is required, and
the tool prepends the CAModel directory to `LD_LIBRARY_PATH`. `-u` is
appended as the program's argument string. That flag was verified with a
single integer iteration count on the vexp binary.

`-g` calls `generate_report(..., core_id="0")`. `report -n` accepts `all`,
ranges such as `0-2,12-14`, and a single id, but a multi-core report was not
run. The report writer lives in `trace_tools.rts_sequence._gen_model_trace`.

## Artifacts

`cannsim record` creates
`<output>/cannsim_<YYYYmmddHHMMSS>_<binary name>/` and moves:

- `cannsim.log` — simulator log, including the program's stdout
- `instr.bin`
- `route_table.txt`, `rtb_debug.txt`, `.soc-version`
- `report/trace_core0.json` when `-g` succeeds

`.soc-version` on the September artifact contained
`{model_name: Ascend950PR_9599, cube_cores: 32, vec_cores: 64}`.

`log_ca` and `dirSaveDump.txt` were absent after both 2026-11-05 runs. The
tool warned and still wrote `trace_core0.json`. There is no
`cannsim.core.public.instr_decoder` in this package. Analysis uses the JSON
trace.

## Other local examples

These exist under `/root/tl-src/examples/ascend/cannsim_exp/` and were not
rebuilt on 2026-11-05: `vdup_kernel.asc`, `fill0_kernel.asc`, `main.cpp`
(aclnnExp), and their `build_*.sh` scripts. September `sim_out` directories
remain under `/root/vdup_kernel_bench`, `/root/fill0_kernel_bench`, and
`/root/vexp_bench`. Use the vexp script in this skill as the known-good run.

## Failures and non-failures

| Observation | What to do |
|---|---|
| `cannsim: command not found` | `source /usr/local/Ascend/cann/set_env.sh` in that shell |
| `soc-version ... is not supported` | Pass `-s Ascend950`. Other names are rejected by the installed `SOC_INFO` |
| `Missing environment variable: ASCEND_TOOLKIT_HOME` | The same `set_env.sh` was not sourced. Read from `record.py`; not re-triggered |
| `CA Model path does not exist` | Toolkit or soc mapping does not match this install |
| `dirSaveDump.txt not found`, `log_ca directory not found` | Non-fatal on both 2026-11-05 runs, as long as `trace_core0.json` exists and its instruction count matches the launch |
| DDR checkpoint warning, `UNKNOWN KEY 'TIMEOUT_SID_ENBALE'`, `no l3 cache` | Present in the successful September log. Do not treat them as the failure |
| Program stdout missing | It is inside `cannsim.log`. The verified kernel prints `vexp_kernel done n_iters=<n>` |
| Two records at once | Not tested. Both would write `cannsim.log` in the binary directory. Run one |
| Output on `/mnt/d` | Not tested. Verified runs used `/root/...` |

`set_env.sh` references unset variables. The verified scripts `set +u` before
sourcing it and `set -u` after.
