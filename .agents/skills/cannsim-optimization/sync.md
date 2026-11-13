# WSL and A5

Two different sync paths. Do not mix them.

## WSL: compile and simulate

Daily analysis, edits, compilation, and CANNsim stay in WSL Ubuntu-24.04.
Source `/usr/local/Ascend/cann/set_env.sh` in the shell that runs Bisheng or
`cannsim`. Put the binary and `sim_out` on the Linux filesystem, as the
verified runs did (`/root/cannsim_skill_verify`,
`/root/cannsim_skill_from_repo`).

The Cursor workspace is `/mnt/d/codes/yyz/tilelang`. Three trees matter:

| Tree | Role | Checked 2026-11-06 |
|---|---|---|
| `/mnt/d/codes/yyz/tilelang` | Cursor checkout, `hy4-mla`. `3rdparty/tvm` is an empty gitlink. No `build/`. | `d5bd66db` when `/root/tilelang-hy4` was copied |
| `/root/tilelang` | Older checkout with a populated `3rdparty` and its own `build`. TVM commit `4211874e`, the same gitlink as the workspace. Not rebuilt for simulation. | branch `hy4-mla`, commit `9c72d05d`, untracked `examples/ascend/dump_ir_dsa/` |
| `/root/tilelang-hy4` | Sources copied from the workspace plus `3rdparty` from `/root/tilelang`. This is the tree `npusim` imported. | built 2026-11-06 against `/root/cannsim-venv` |

`/root/tilelang-venv` imports `/root/tilelang` and is the CUDA environment.
`/root/cannsim-venv` is the CANNsim environment. Do not mix them.

Before a TileLang build or a source dump:

```bash
git -C /mnt/d/codes/yyz/tilelang rev-parse --abbrev-ref HEAD
git -C /mnt/d/codes/yyz/tilelang rev-parse HEAD
git -C /root/tilelang status -sb
git -C /root/tilelang rev-parse HEAD
```

If the workspace commit and `/root/tilelang-hy4` differ, the simulation is
not running the sources you are editing. Copy the intended sources, rebuild
`/root/tilelang-hy4`, and `cd` there before importing. A CANNsim microbench
built only with Bisheng, like `examples/vexp_kernel.asc`, does not use this
package.

Do not assume `/mnt/d`, `/root/tilelang`, and `/root/tilelang-hy4` are the
same working tree. The 2026-11-05 table in older notes (`pr-3323` at
`789afc96`) is stale.

## A5: correctness and device performance

Use A5 only to check numerical results and on-device performance. Connect and
run it through the `a5-server` skill (`ssh` alias `A5`, one shell for
`source ~/yyz_a5_env.sh` plus the command). Inspect the remote checkout,
branch, commit, and dirty state before using it. Do not assume it matches
either local tree.

Code reaches A5 only through git:

1. Commit the intended sources in the local checkout the user asked to publish.
2. Push that branch to the remote A5 can fetch.
3. On A5, fetch and check out that branch or commit.

Do not scp, rsync, copy, or otherwise upload the working tree onto A5.
No A5 command was run while creating this skill.

Simulated cycles stay labeled as CANNsim results. After an annotation or a
schedule change, the A5 run is what establishes device latency and
correctness. Record the commit, shape, and both results separately.
