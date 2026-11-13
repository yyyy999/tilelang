# Hy4 MLA 优化记录

只优化 `examples/ascend/hy4_mla/kernel.py` 的 `hy4_sparse_attention`。
不改 `examples/ascend/flashmla`。

## 目标形状


| 项          | 值                                             |
| ---------- | --------------------------------------------- |
| nq, nk     | 4096, 4096                                    |
| topk       | 2048                                          |
| H, Dqk, Dv | 64, 576, 512                                  |
| dtype      | BF16 输入输出，FP32 Cube 累加                        |
| scale      | 1/16                                          |
| 设备口径       | Ascend 950PR，32 AIC，Cube BF16 峰值 432 TFLOPS   |
| 目标         | 把 Cube 打满，padded MFU 90%，对应 kernel 时间 3.00 ms |


有效 MFU 使用 `make_inputs` 的保留槽 2,171,661。padded MFU 使用全部 8,388,608 个 topk 槽。分子是每个槽、每个 head 的 `2×576 + 2×512 = 2176` FLOP。

## v0 基线

提交 `9c72d05d`。A5 NPU 0，`msprof` 的 `main_kernel`，稳定样本中位数 5750.95 µs。Block Dim 32。


| 口径                | MFU    |
| ----------------- | ------ |
| padded，432 TFLOPS | 47.02% |
| 有效，432 TFLOPS     | 12.17% |


`q_prefetch` 和 `direct_query_masks` 在 `nb = topk/64 = 32` 时关闭。gather 每次最多搬 2 行、576 个 BF16。

## v1 gather 留在 L2

同一条 576 宽的 KV gather 增加 `l2_cache_ctrl="normal_lv"`。索引、softmax 和 GEMM 的数据通路不变。这个形状的 KV 只有 4.7 MB，提示 L2 保留这些行，避免每个 query 都从 HBM 重读。

仿真对照的是这个 gather 的字节数，不是 FlashMLA 的 512 维。`sim/gather_mte.asc` 在 WSL CANN 9.2.0 上跑，模型 Ascend950PR。mode 0 是 16 次连续的 `2×576` uint16 搬运，mode 1 是一次 `32×576`。程序都打印了 `hy4_gather_mte done`。下面的周期由 trace 的 `dur` 除以 `0.00060606` 得到，只用于和旧周期读数对齐，不是设备时间。


| 模式        | MTE2                          | 单次字节  | 占用              |
| --------- | ----------------------------- | ----- | --------------- |
| 16 次 pair | 每 13 个周期发出一次，单次约 1116–1192 周期 | 2304  | 16 次合起来 1311 周期 |
| 一次整块      | 一次 1287 周期                    | 36864 | 1287 周期         |


16 次短搬运在 MTE2 上已经重叠，整块时间只比一次 36 KB 搬运多大约 24 个周期。把同一个 64-token 块合成一条 DMA 打不满 Cube。仿真里的 pair 是连续 2304 字节，还不是 Hy4 运行时两个索引之间的跨行 stride。Cube 和混合核没有进这次仿真。

A5 在 `7715ad8c` 上跑了 `test_hy4_mla.py`：9 passed，40 分 30 秒。其中包含 topk=2048 的 `(1, 79)` 和 `(33, 257)`。

目标形状 `4096×4096`、topk=2048 的设备计时是 NPU 事件中位数，不是 `msprof`。预热后 20 次的中位数是 5760.65 µs。精度最大绝对误差 `9.77e-4`，相对 L2 `0.00243`。padded MFU 46.94%，有效 MFU 12.15%，对应 202.80 TFLOPS。和 v0 的 5750.95 µs 相差约 10 µs。

## BF16 TopK=2048 索引分组 512

基线 `0de046ac` 与候选 `851995f6` 在 A5 物理 2 号卡（Ascend950PR_9579，28 AIC）使用同一份 `4096×4096、TopK=2048` 输入交替运行。每版预热 5 次，记录 20 个 NPU Event 样本。两版均通过 FP32 reference 对照，输出最大绝对误差 0.0009765625、相对 L2 0.00242792。


| 分组  | 中位数（µs） | Padded MFU（378 TFLOP/s） |
| --- | ------- | ----------------------- |
| 640 | 3319.93 | 93.09%                  |
| 512 | 3317.65 | 93.15%                  |


512 每 query ++仍搬运四批索引，将固定长度 SIMD 预处理从 40 个 tile 减到 32 个，源码 UB 预算减少 1,024 B。本次中位数差 2.28 µs，样本范围重叠。日志在 A5 的++ `~/hy4-bf16-ab-851995f6-npu2.log`。

## BF16 QK 三段 192 原型

A5 物理 2 号卡（28 AIC），`851995f6`，`4096×4096、TopK=2048`，预热后采集 5 次 PipeUtilization。设备任务中位数为 3305.58 µs。AIC MAD 94.8–94.9%、MTE1 93.6–93.7%、FixPipe 52.2–52.3%，AIV Vector 78.9–79.0%。各流水可以重叠，这些占用比例无法区分具体依赖的等待。原始 CSV 在 A5 的 `/tmp/hy4-bf16-pipes.0ApuSD`。

候选仅针对 BF16 TopK=2048 流水，将 QK 分成三个 192 宽子块，L0A/L0B 改为两版。每个 64-slot tile 的 QK MAD 从 5 次降到 3 次，L1→L0 copy 从 15 次降到 9 次，矩阵算量和有效搬运字节数相同。PV 仍使用紧凑的 64×128 输入。28 核目标形状的原型已通过 WSL lowering 和 Bisheng，静态检查预算 L0A/L0B 各 49,152 B。实卡精度与延迟待验证。

A5 同卡 A/B 已完成，输入和上轮一致，每版预热 5 次，交替计时 20 次。基线 `851995f6` 中位数 3313.57 µs（93.27% padded MFU），候选 `c6ce6891` 中位数 4323.45 µs（71.48%）。两版均通过参考精度，最大绝对误差 0.0009765625，相对 L2 0.00242792。退化约 30.5%，已回退到基线。切分和 L0 版本数同时变化，无法单独归因。日志在 A5 的 `~/hy4-qk192-c6ce6891-npu2.log`。

## Q 预取，依赖搜索限制在流水窗口

Best 父版本 `f15c3fe6`。假设：AIC MTE2 只在每个 query 的第一条 tile 把 Q 从 GM 搬进 L1，这段搬运落在 QK 的关键路径上，对应大约 170 µs 的 MAC 空泡。修改：BF16 TopK=2048 打开两槽 Q，把搬运从 stage 4 挪到该 query 第一条 tile 的 stage 0，stage 编号本身不变。`q_l1` 的手工多缓冲距离从 `2*nb=64` 改为 8。64 会让每对 Q 访问都把证不到的距离交给 Z3，编译主线程在用户态打满一核，十几分钟没有日志，不是 flag 死锁。

A5 物理 2 号卡，28 AIC，`4096×4096`、TopK=2048，`make_inputs` seed 42、`variable_lengths=True`。预热 5 次，交替 20 个 NPU Event。`4f10f35c` 相对 `f15c3fe6` 的 out、max、lse 都是 `torch.equal`。


| 版本                           | 中位数（µs） | 范围（µs）          | Padded MFU（378 TFLOP/s） |
| ---------------------------- | ------- | --------------- | ----------------------- |
| Baseline / 旧 Best `f15c3fe6` | 3324.14 | 3319.11–3401.59 | 92.97%                  |
| `4f10f35c`                   | 3209.77 | 3205.41–3219.50 | 96.29%                  |


中位数少 114 µs。更新 Best 为 `4f10f35c`。97% 对应 3186 µs，还差约 24 µs。编译有一条 `MTE2->MTE1: reverse ordering too loose to bound flag reuse`，反向距离大于 8 槽 flag 环；这次输入没有踩中错序。日志在 A5 的 `/tmp/hy4_prefetch2.log`。

## 再提前两个 tile 发 Q

父版本 `4f10f35c`。假设剩下的空泡仍是第一条 tile 上的 Q 搬运，把下一条 query 的 Q 改到上一条 query 的倒数第二 tile 发出，stage 图不变。`f2f69842` 与同一次运行的 `f15c3fe6` 输出 `torch.equal`。


| 版本                     | 中位数（µs） | 范围（µs）          | Padded MFU |
| ---------------------- | ------- | --------------- | ---------- |
| 同次 Baseline `f15c3fe6` | 3321.49 | 3317.07–3401.15 | 93.05%     |
| `f2f69842`             | 3203.96 | 3201.45–3215.66 | 96.46%     |


相对 `4f10f35c` 的 3209.77 µs，中位数少 6 µs，样本范围重叠。不更新 Best。同样的 MTE2→MTE1 flag 警告还在。日志在 A5 的 `/tmp/hy4_qlead2.log`。随后把 kernel 退回 `4f10f35c`。

同一次 `f2f69842` 的 msprof PipeUtilization：AIC MAD 98.7%，MTE1 97.9%，MTE2 5.0%，FixPipe 50.7%，AIC scalar 25.4%；AIV scalar 84.4%，vector 81.7%。Q 搬运已经藏在 MAD 后面。一条 query、32 个 tile 的 CANNsim（模型 Ascend950PR_9589，不是 A5 的 9579）里 Cube 只有 77%，主因是这条 query 的 gather，不能拿来解释 4096 query 的设备 MFU。

## PV 先发第一片 V

父版本 `4f10f35c`。假设 stage 5 先等 P 进 L0A，MTE1 在 softmax 握手期间空着。CANNbot 的做法是先发第一片 V。

`6d2e6fc2` 把 P 装载放进 4 次 PV 内层流水，只在 `d == 0` 执行。编译通过，out、max、lse 与 Best 和 Baseline 都是 `torch.equal`。同一次 A/B：Baseline 3318 µs（93.14%），Best 3203 µs（96.50%），候选 6763 µs（45.70%）。flag 水位 MTE1→M 12、M→MTE1 18，超过 dav-3510 的 8，多余 flag 走 `get_buf`/`rls_buf`。

`61f0bf1b` 把第一片 V 留在内层循环外面，P 装载回到原位，循环里跳过 `d == 0` 的 V。InsertSync 报错：`uncovered multi-buffer owner exclusion at logical distance 2`。不更新 Best，kernel 退回 `4f10f35c`。

## 两条 query 的仿真，以及把 P/V0 拆出内层循环

Best 仍是 `4f10f35c`，当前 `hy4-mla` 的 kernel 与它一致。同一次 A/B 里 Best 中位数 3202.73 µs，padded MFU 96.50%（378 TFLOP/s，28 AIC）。97% 对应 3186 µs。最慢核要做 147 条 query，理想 MAC 约 3136 µs；3203 µs 里大约 42 µs 是 MAD 空闲，25 µs 是超出理想 MAC 的额外 MAD。分到每个 64 行 tile 大约 15 个周期。

`f2f69842` 的 msprof 与 Best 同级：AIC MAD 98.7%，MTE1 97.9%，MTE2 5.0%；AIV MTE2 31%，scalar 84%。Cube 是时钟。

Best 在 WSL CANN 9.2.0、模型 Ascend950PR_9589 上跑了 `nq=2, nk=2048, topk=2048`，一核。归档 `/root/hy4_nq2_sim/npusim_20261106195428_hy4_nq2.py`。`kernel_total_clocks` 89598，1.65 GHz，约 54.3 µs，不能换成设备时间。AIC Cube 82.0%，AIV0 MTE2 81.4%，AIV1 MTE2 78.5%。第二条 query 的 MMAD 窗口 40712 周期，并集忙碌 36649，空闲 4063，17 个空洞，宽度 119–428 周期。相邻 MMAD 的间隔中位数是 −24 周期，两个 Cube slot 已经叠上。这些空洞仍是仿真里的 gather 量级。设备上 AIV MTE2 只有 31%，所以这条仿真不能用来决定下一次改动。

独立 worktree `d:/codes/yyz/tilelang-hy4-pv`，分支 `hy4-pv-split-issue`，未提交。stage 5 把 P 和第一片 V 放进同一个 `T.Task`，剩下三片 V 留在 `Pipelined(3, num_stages=2)`。假设 L0A 和 L0B 两条 MTE1 可以重叠，同时避开上一次四次循环里的 flag 溢出。WSL 编译失败：`InsertSync cannot materialize an uncovered multi-buffer owner exclusion at logical distance -1 (paired distance 2)`。和 `61f0bf1b` 是同一条证明缺口。不更新 Best。仿真树 `/root/tilelang-hy4` 的 kernel 已拷回 Best。

静态预算：UB 238464 / 253952，余 15488 B。把分数加宽到 128 列会再占 12288 B，纸面上还剩约 3 KB。L0A/L0B 已经各占满 64 KB。