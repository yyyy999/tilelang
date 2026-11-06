# Hy4 preview FP8 sparse Decode 算子目标

状态：待确认。核对日期：2026-11-06。

首版目标是 Hy4 preview 使用的 `flash_mla_with_kvcache` 的 FP8 sparse 子路径。验收范围为 Ascend 算子及独立用例。

## 调用依据

Hy4 的 `_fp8_flash_mla_kernel` 将 Q、按字节解释的分页 KV、物理 sparse indices、模型 scale 和每 head sink 传入 `flash_mla_with_kvcache`。在这条调用里，block table 是占位参数，调用方没有传 `topk_length`。 [Hy4 调用点](https://github.com/vllm-project/vllm/blob/db914cad5bac261dbcd7cf05ded573c1d704f650/vllm/models/hy_v4/nvidia/flashmla_sparse.py#L175)

首版选择 `fp8_ds_mla` 对应的 V3.2 型 656 B 记录。该宽度和字段偏移以固定的 FlashMLA 接口为准。vLLM 在进入 FP8 kernel 前将缓存整理为 64-token 页。 [缓存格式](https://github.com/vllm-project/FlashMLA/blob/847ef308bb7a6828c8117301f5a5a84a095a90e8/flash_mla/flash_mla_interface.py#L105) · [64-token 页视图](https://github.com/vllm-project/vllm/blob/db914cad5bac261dbcd7cf05ded573c1d704f650/vllm/v1/attention/backends/mla/flashmla_sparse.py#L136)

## 首版输入输出

| 参数 | shape / dtype | 合同 |
| --- | --- | --- |
| Q | `[B,Sq,64,576]` BF16 | 已吸收 key 投影且完成 RoPE。首版固定 H64，覆盖 Sq=1 和多 query |
| K cache | `[P,64,1,656]` uint8 | 单份缓存，首版为连续的物理页视图 |
| Indices | `[B,Sq,K]` int32 | 物理槽 `page_id * 64 + offset_in_page`。K=2048 为主要目标，小尺寸测试可取其他正的 64 倍数 |
| Sink | `[64]` FP32 或 None | 每 head 的 sink logit |
| Scale | 正有限标量 | 显式传入，Hy4 默认配置取 1/16 |
| Output | `[B,Sq,64,512]` BF16 | 加权后的 latent |
| LSE | `[B,64,Sq]` FP32 | 自然对数，统计有效数据 logits |

Q、indices 及输出首版采用连续布局。索引由调用方提供，请求隔离和因果筛选已经编码在 indices 中。负索引和超出物理页视图的索引按无效项处理，重复槽保留各次贡献。首版实现 TileLang 自己的调度，兼容范围以此表为准。

| 记录字节区间 | 内容 |
| --- | --- |
| `[0,512)` | 512 个 FP8 E4M3 latent 值 |
| `[512,528)` | 4 个 FP32 scale，每个对应 128 个 latent 值 |
| `[528,656)` | 64 个原样保存的 BF16 RoPE 值 |

## 数学语义

每个有效槽将 FP8 latent 按 128 元素一组乘对应的 FP32 scale。首版将乘积舍入为 BF16，供现有 BF16 Cube QK/PV 路径消费。QK 使用反量化 latent512 与原始 RoPE64。PV 只使用 latent512。参考实现应复现该中间精度边界。

Sink 作为零 value 的虚拟槽，仅进入输出归一化。LSE 排除 sink。空行返回零输出、LSE=+inf。Sink=-inf 等价于关闭该 head 的 sink，+inf 使该 head 输出为零。 [FlashMLA Decode 参考](https://github.com/vllm-project/FlashMLA/blob/847ef308bb7a6828c8117301f5a5a84a095a90e8/tests/ref.py#L60)

## 实现与验收顺序

1. 按 656 B 格式建立独立的 pack/unpack 和 FP32 参考。先验证四组非同值 scale、RoPE 原样读回、跨页物理槽和重复槽。
2. 复用 Hy4 的 QK576、sink/softmax 和 PV512。将 656 B gather 与反量化放进 tile 流水，核算 UB/L1/L0、DMA stride 和同步生命周期，再做离线 lowering。
3. 实卡用例覆盖 Sq=1/4、不同 B、TopK=64/2048、乱序跨页索引、全无效行、sink 开关。验收 output 和 LSE 的精度、布局及重复 launch。
4. 性能先测小 batch Decode 延迟和多 batch 吞吐。Split-KV 根据实测决定，采用时在全局合并后应用一次 sink。

后续扩展项包括 H128、可选长度、额外缓存、更多 stride，以及框架注册。BF16 Prefill 和仓库现有 V4.1 的 528 B / 288 B Decode 各自保留现有接口。

## 本地检查入口

```bash
python examples/ascend/hy4_mla/check_resources.py --kv-format fp8_ds_mla --topk 2048
python -m pytest -q examples/ascend/hy4_mla/test_decode_reference.py
python -m examples.ascend.hy4_mla.check_lowering --kv-format fp8_ds_mla --topk 64 2048 --output /tmp/hy4-decode-lowering
python -m pytest -q examples/ascend/hy4_mla/test_decode_npu.py
```

独立入口在 `decode.py`。Packed cache 按 656 B 读取，搬入 UB 时使用 672 B 行距以满足 32 B 对齐。反量化直接写入 BF16 NZ tile，QK、PV 和 sink 沿用现有计算流水。

### 2026-11-06 离线验证

本机 WSL 使用 `9c72d05d` 的原生构建和当前工作区的 Hy4 算子源码。该提交到分支基线 `1d55a786` 的差异均在 examples 下。K=64 和 K=2048 均通过 lowering，CANN 9.2.0 Bisheng 对 `dav-3510` 均生成了设备可重定位目标文件。

| K | 编译器分配的 UB | 编译器分配的 L1 |
| --- | ---: | ---: |
| 64 | 195,968 B | 458,752 B |
| 2048 | 200,576 B | 458,752 B |

两档的生成代码均使用 656 B DMA burst、672 B UB 行距。DMA 与反量化按每版 21,504 B 切换 packed tile，FP32 scale 和 BF16 RoPE 的别名使用同一版偏移。两组实卡验收用例对应的特化也通过 lowering 和 Bisheng 编译，分别为 B=1、Sq=1、K=64、无 sink，以及 B=2、Sq=4、K=2048、有 sink。生成代码和编译日志在本机 WSL 的 `/tmp/hy4-decode-lowering.xNdKu0`。实卡精度和性能待测。
