# Hy4 MLA 优化记录

只优化 `examples/ascend/hy4_mla/kernel.py` 的 `hy4_sparse_attention`。
不改 `examples/ascend/flashmla`。

## 目标形状

| 项 | 值 |
| --- | --- |
| nq, nk | 4096, 4096 |
| topk | 2048 |
| H, Dqk, Dv | 64, 576, 512 |
| dtype | BF16 输入输出，FP32 Cube 累加 |
| scale | 1/16 |
| 设备口径 | Ascend 950PR，32 AIC，Cube BF16 峰值 432 TFLOPS |
| 目标 | 把 Cube 打满，padded MFU 90%，对应 kernel 时间 3.00 ms |

有效 MFU 使用 `make_inputs` 的保留槽 2,171,661。padded MFU 使用全部 8,388,608 个 topk 槽。分子是每个槽、每个 head 的 `2×576 + 2×512 = 2176` FLOP。

## v0 基线

提交 `9c72d05d`。A5 NPU 0，`msprof` 的 `main_kernel`，稳定样本中位数 5750.95 µs。Block Dim 32。

| 口径 | MFU |
| --- | --- |
| padded，432 TFLOPS | 47.02% |
| 有效，432 TFLOPS | 12.17% |

`q_prefetch` 和 `direct_query_masks` 在 `nb = topk/64 = 32` 时关闭。gather 每次最多搬 2 行、576 个 BF16。

## v1 gather 留在 L2

同一条 576 宽的 KV gather 增加 `l2_cache_ctrl="normal_lv"`。索引、softmax 和 GEMM 的数据通路不变。这个形状的 KV 只有 4.7 MB，提示 L2 保留这些行，避免每个 query 都从 HBM 重读。

仿真对照的是这个 gather 的字节数，不是 FlashMLA 的 512 维。`sim/gather_mte.asc` 在 WSL CANN 9.2.0 上跑，模型 Ascend950PR。mode 0 是 16 次连续的 `2×576` uint16 搬运，mode 1 是一次 `32×576`。程序都打印了 `hy4_gather_mte done`。下面的周期由 trace 的 `dur` 除以 `0.00060606` 得到，只用于和旧周期读数对齐，不是设备时间。

| 模式 | MTE2 | 单次字节 | 占用 |
| --- | --- | --- | --- |
| 16 次 pair | 每 13 个周期发出一次，单次约 1116–1192 周期 | 2304 | 16 次合起来 1311 周期 |
| 一次整块 | 一次 1287 周期 | 36864 | 1287 周期 |

16 次短搬运在 MTE2 上已经重叠，整块时间只比一次 36 KB 搬运多大约 24 个周期。把同一个 64-token 块合成一条 DMA 打不满 Cube。仿真里的 pair 是连续 2304 字节，还不是 Hy4 运行时两个索引之间的跨行 stride。Cube、混合核和 A5 时间都还没记在这一版。
