# Hy4 sparse MLA：目标合同与适配方案

记录日期：2026-11-05。开发分支：`hy4-mla`。

本文是实现前固定的目标基线，区分源码事实、实施选择和待确认项。BF16 核心已开始实现；编译、设备正确性与性能状态见 README 和后续验证记录，不由本文的目标描述推定。

BF16模型调用接口的增量要求与适配状态见[ADAPTATION.md](ADAPTATION.md)，包括可选参数、输出工作区、head padding和有序pitch布局。

## 1. 目标与范围

实现公开 **Hy4 preview** 使用的、带 attention sink 的 V3.2 式 sparse MLA 核心：**Dqk=576，Dv=512，模型 TopK=2048**。

以 Hy4 的 kernel 调用合同及数学语义为正确性参照，以当前仓库新合入的 Ascend FlashMLA 为高性能实现基础。本文不将 Hy4 正式版与 preview 的结构视为相同。

第一阶段交付给定 indices 的 BF16 核心，处理已完成投影吸收和 RoPE 的 Q/KV，返回 latent Attention 结果及统计量。Indexer、TopK 选择、跨层索引共享、模型投影、512→256 value 投影、gate 和 O 投影由上游或下游完成。

第一阶段采用 BF16 连续 KV 行；不会宣称覆盖生产量化分页 Decode。BF16 核心本身可以处理一个或多个 query，Prefill/Decode 的服务集成另行验收。

## 2. 版本与证据

### 当前实现基础

- 工作区基线：`c99577fd91962ae03e9cdb4a5c6be98632d4c899`。
- MLA 引入提交：`2031b244`，`[Ascend] Add scheduled FlashMLA kernels (#3678)`。
- 主要基础：[kernel.py::sparse_attention](../flashmla/kernel.py)。
- BF16 编译入口：[prefill.py::compile_prefill](../flashmla/prefill.py)。
- packed Decode 入口：[decode.py::compile_decode](../flashmla/decode.py)。
- 量化基础：[quantization.py::dequantize](../flashmla/quantization.py)。
- 测试结构：[test_flashmla.py](../flashmla/test_flashmla.py)、[testing.py](../flashmla/testing.py)。
- 实现来源与性能口径：[FlashMLA README](../flashmla/README.md)。其中固定上游 FlashMLA 参照为 `a123d0b0191e0da7aa1e044f8644e0989cc63220`。

### 模型语义与接口参照

1. [腾讯模型配置](https://huggingface.co/tencent/Hy4-preview/blob/main/config.json)：动态地址，调查读取日期为 2026-11-05。正式接入前固定模型 revision 并保存配置。
2. [Transformers HYV4Attention / expand_kv](https://github.com/huggingface/transformers/blob/fd3cf0140de03f7619ac95b5cfed05b711ff9e22/src/transformers/models/hy_v4/modeling_hy_v4.py#L328)：模型投影、value 宽度、sink 和 gate 的参考。
3. [Transformers eager_attention_forward](https://github.com/huggingface/transformers/blob/fd3cf0140de03f7619ac95b5cfed05b711ff9e22/src/transformers/models/hy_v4/modeling_hy_v4.py#L273)：将 sink 拼入 logits，softmax 后丢弃 sink 概率列。
4. [Hy4 的 BF16 FlashMLA 调用](https://github.com/vllm-project/vllm/blob/db914cad5bac261dbcd7cf05ded573c1d704f650/vllm/models/hy_v4/nvidia/flashmla_sparse.py#L217)：`HYV4FlashMLASparseImpl._bf16_flash_mla_kernel` 调用 `flash_mla_sparse_fwd(q, kv, indices, scale, attn_sink=..., topk_length=...)`。
5. [MLA 内部维度](https://github.com/vllm-project/vllm/blob/db914cad5bac261dbcd7cf05ded573c1d704f650/vllm/model_executor/layers/attention/mla_attention.py#L709)：query 宽度为 `kv_lora_rank + qk_rope_head_dim`，输出宽度为 `kv_lora_rank`。

这些源码支持下述逻辑合同；尚未抓取真实 Hy4 运行时张量来核实全部 stride、padding、cache 布局和分派路径。

## 3. 模型参数如何映射到 kernel

| 配置字段 | preview 值 | 含义及与核心的关系 |
| --- | --- | --- |
| `num_attention_heads` | 64 | 模型 query head 总数；第一阶段 kernel 固定 H=64 |
| `hidden_size` | 6144 | 模型隐藏状态宽度，不直接作为核心输入 |
| `q_lora_rank` | 2048 | Q 的共享中间投影宽度，不是 QK 内积宽度 |
| `kv_lora_rank` | 512 | 共享 KV latent 宽度；核心 V 和输出宽度 |
| `qk_nope_head_dim` | 192 | 展开后的非位置 Q/K 宽度；投影吸收后 Q 的此部分变成512维 |
| `qk_rope_head_dim` | 64 | 单独的位置部分；与 latent512 拼成 QK576 |
| `v_head_dim` | 256 | 模型 value head 宽度；核心输出512维之后再投影到256维 |
| `index_n_heads/index_head_dim` | 32/128 | Indexer 的评分 heads，与主 Attention 的64 heads分开 |
| `index_topk` | 2048 | 每 query 的索引数量上限；不等于 Q 压缩维度的含义 |
| `learnable_sink` | true | 每个 Attention head 有一个 sink 参数 |

令行向量 `c` 为512维 latent，展开后的非位置 key 为 `c W_K`，其中 `W_K` 为每 head 的512×192矩阵。利用

```text
q192 (c512 W_K)^T = (q192 W_K^T) c512^T
```

框架将 key 投影吸收到 query，得到 `q_latent512`。与已旋转的 `q_rope64` 拼接后，核心 Q/K 宽度为576。核心先对 latent 加权求和，value 投影在其后执行。

## 4. 第一阶段接口合同（实施选择）

builder 为 `hy4_sparse_attention`，编译入口为 `compile_bf16`。下列为第一阶段接口合同，运行覆盖范围单独记录。

静态参数：`nq > 0`、`nk > 0`、`H=64`、`Dqk=576`、`Dv=512`、`K > 0` 且 `K % 64 == 0`；主优化目标为 `K=2048`。`scale` 显式传入，支持 `variable_lengths` 和用于对照的 `sink` 开关。

| 张量 | 逻辑 shape | dtype | 合同 |
| --- | --- | --- | --- |
| Q | `[nq, 64, 576]` | BF16 | 已完成投影吸收与 RoPE，前512维 latent query，后64维位置 query |
| KV | `[nk, 576]` | BF16 | 每行前512维共享 latent，后64维已旋转位置 key |
| Indices | `[nq, K]` | int32 | 每行选择 KV 行；64 heads共享该行选择 |
| Sink | `[64]` | FP32 | 模型每 head 的可学习标量；关闭 sink 时不参与计算 |
| Lengths | `[nq]` | int32 | 每行前多少个索引槽可用；启用时要求0≤长度≤K |
| Output | `[nq, 64, 512]` | BF16 | Attention 的 latent 输出 |
| Max | `[nq, 64]` | FP32 | 有效数据 logits 的最大值，不包括 sink |
| LSE | `[nq, 64]` | FP32 | 有效数据 logits 的自然对数 logsumexp，不包括 sink |

核心默认连续输入：Q stride为`(36864,576,1)`，KV为`(576,1)`，Indices为`(K,1)`。新增GM stride参数用于有序正pitch的框架工作区view；其设备路径尚待验证，不声明支持任意stride。框架KV/Indices的单例head维由适配层无拷贝去除，具体限制见[ADAPTATION.md](ADAPTATION.md)。

preview 的默认 RoPE 配置下，模型 Attention scale 为 `1/sqrt(192+64)=1/16`。该值不因投影吸收变成576维而改成 `1/sqrt(576)`；接口以调用方传入的 scale 为准。

## 5. 数学与边界合同

对 query `t`，有效索引槽为 `u < Lengths[t]`（未启用长度时为 `u < K`），且 `0 ≤ Indices[t,u] < nk`。令 `j=Indices[t,u]`，计算：

```text
score[t,h,u] = scale * (
    dot(Q[t,h,0:512], KV[j,0:512])
  + dot(Q[t,h,512:576], KV[j,512:576])
)

Output[t,h,:] = sum_u(exp(score[t,h,u]) * KV[j,0:512])
                / (sum_u(exp(score[t,h,u])) + exp(Sink[h]))
```

实现必须使用数值稳定的 online softmax；FP32 reference 应以数据 score 与 sink 的共同最大值进行稳定归一化。关闭 sink 时去掉分母中的 sink 项。

- 负索引（包括`-1`）和`index >= nk`均无效；不得发起越界 GM 读取。
- Lengths 是索引槽数量，不是KV序列长度。有效范围内仍需检查每个 index。
- 保留输入槽顺序，不要求索引排序。重复有效索引按多次贡献处理；模型正常 TopK 无重复，reference 不得用 scatter 去重改变测试语义。
- 因果 mask、请求隔离和逻辑到物理地址转换由上游负责。核心不按展平后的 `t` 追加 `index <= t` 条件。
- 全无效行沿用当前 Ascend FlashMLA 接口约定：Output=0、Max=-inf、LSE=+inf。这是兼容性哨兵，不是数学 logsumexp 的通常定义。
- 非空行的 Max/LSE 均不包含 sink，只有输出归一化分母包含 sink；避免下游重复应用 sink。
- 第一阶段限定有限Q/KV数值、正的有限scale；Sink接口包含-inf无影响、+inf输出零的兼容要求，设备行为待专项回归。不承诺一般NaN/Inf输入传播行为。

## 6. 与现有 Ascend MLA 的差异及改法

### 保留部分

沿用 `sparse_attention` 的 persistent jobs、稀疏成对 gather、indices 分组、长度与无效项处理、sink、SIMD softmax、概率 packing、512维分块 PV、重缩放以及 `T.Stage`/自动版本管理。保留原始 FlashMLA 目录作为回归与性能基线，不直接修改成仅支持 Hy4。

### 首选适配方案

1. 保持 latent512 的 QK/PV 主路径，新增64维 RoPE 的读取和 QK 累加；PV仍只消费latent512。当前Q/KV L1和NZ各自合为576维存储，以减少独立同步通道。
2. 扩展输入 view/stride，确保 gather 能取得512+64而不误将RoPE混入PV。
3. 重新审视 UB ND→NZ packing、L1布局、`l0_pitch`、buffer版本和固定地址；用生成代码与容量检查确认合法性。
4. QK最终完成点移到RoPE累加之后，调整最后一次GEMM的UnitFlag和FixPipe交接；不能在512维结束时提前发布score。
5. 保留现有softmax/sink和四个128维PV输出块，复用其数值与流水测试结构。
6. 在实际TopK2048下重新分析调度，再决定是否扩展query预取；不强行套用原TopK640的参数。

具体资源约束：原来单个`32×512×BF16`的KV L1缓冲四版本占128 KiB，`32×576×BF16`四版本占144 KiB；不能全局替换512→576后直接沿用地址表。独立RoPE缓冲方案在首次离线lowering中遇到跨核flag分配限制；当前采用合并576维L1，重新布局固定地址。资源账本及检查状态见[STATIC_CHECKS.md](STATIC_CHECKS.md)。

TopK2048对应32个64-token块；当前`q_prefetch`仅在4≤nb≤15启用，目标默认进入单query slot路径。它不是正确性问题，但原双query预取性能不能直接外推到目标。

## 7. 后续缓存与head适配

- 当前packed Decode使用V4.1的528-byte FP8 / 288-byte FP4记录；不能作为Hy4的576维缓存格式直接沿用。
- 量化Decode需要先固定目标框架cache ABI，再适配数据、scale和64维RoPE的gather/dequant。第一阶段不选择或发明生产cache格式。
- H64是核心实现范围，不代表所有部署每卡均有64 heads。适配层将TP后H8/H16等补至H64再裁回，专用小head的SIMD、Cube tiling与任务分配优化仍是后续工作。
- 新MLA的4 GiB cache限制、物理slot索引约定等仅在采用其packed路径时继承；BF16接口以KV行索引合同为准。
- 共享索引仅共享选择结果，各层KV数据仍独立；由框架维护其生命周期。

## 8. 目录结构

按现有FlashMLA组织方式加入以下实现与验证入口：

```text
hy4_mla/
  TARGET.md          # 目标、证据、适配方案与待确认项
  kernel.py          # 576→512 sparse MLA + sink
  prefill.py         # 第一阶段BF16编译入口
  reference.py       # 独立FP32数学reference及兼容性输出约定
  testing.py         # 输入构造与接口适配
  test_hy4_mla.py    # NPU精度和边界回归
  test_reference.py # CPU手算语义测试
  run.py            # 校验后可选事件计时，输出原始记录
```

量化`decode.py`及`quantization.py`在缓存合同确定后新增，不预先复制不兼容的V4.1接口。

## 9. 验收与性能口径

### 正确性

1. 独立FP32 golden：显式gather、稳定softmax+sink，明确Max/LSE和全无效哨兵；另考虑BF16概率舍入对PV的影响。
2. 分别检查512维latent score、64维RoPE score以及合并score；测试仅RoPE非零、仅latent非零，防止尾段漏算或PV混入RoPE。
3. 覆盖nq=1/31/32/33/65、主TopK2048及64/640/704等回归点；Lengths覆盖0/1/63/64/65/639/640/641/K-1/K。
4. 覆盖全无效、混合负值/越界索引、非排序与重复索引、group尾块、跨job/launch复用、不同head重缩放、sink正负值及较大幅值。
5. 必须执行compile→launch→synchronize→数值比较，保留生成源码、环境、输入和失败证据。
6. 记录max abs、相对L2、非有限值mask及Max/LSE误差；初始BF16 allclose可用rtol=1e-2/atol=1e-2作探索门槛，最终阈值须由模型/调用方精度要求确认，不以放宽阈值替代定位错误。
7. 有NVIDIA环境时，与固定、支持576维及sink的Hy4 FlashMLA调用同输入对比；不使用仅支持512维V4.1的新FlashMLA作为目标正确性reference。

### 性能

先固定H64、BF16、K2048，以nq和nk扫描覆盖小query及prefill负载；真实生产shape待确认。每个测量case先通过精度校验，报告kernel-only延迟及基准环境，保留每次launch时长和L2策略。

有效 GEMM FLOPs 按每个保留索引槽、每个 head 的 `2×576 + 2×512 = 2176` 计算。长度前缀之外、负索引和越界索引不进入分子；重复索引按多次乘积计入。Softmax、sink 和搬运只留在时间里。MFU = 该 FLOPs / (T × Cube BF16 峰值)。T 取预热后 kernel 事件时间的中位数，单位秒。950PR 的 Cube BF16 峰值是 32 AIC 为 432 TFLOPS、28 AIC 为 378 TFLOPS；不用 Cube+Vector 总算力。按完整 topk 槽计算的是 padded MFU，必须与有效 MFU 分开报告。端到端时间另计投影、索引器、布局转换、cache 写入和输出 gate 等成本。

现有README的405/362 TFLOP/s属于Dqk=Dv512、特定V4.1 workload和Ascend950DT/CANN9.3.0的记录，不作为Hy4性能承诺。性能验收需同硬件、同dtype/cache、同shape、同计时边界的目标参照；绝对延迟和相对目标仍待确认。

## 10. 开始实现前仍需确认

- 正式目标是否确认为preview；固定权重/配置revision，并核对真实kernel调用的shape/stride/padding。
- 目标Ascend具体型号及可用设备；当前代码基础面向950，但不据此假定用户的最终硬件。
- CANN、驱动、PyTorch/torch_npu版本及所选AscendC/PTO路径；首选复用现有默认Ascend路径。
- TP、每卡head数、batch、context、Prefill chunk和MTP query数。
- 第一阶段是否接受给定indices、BF16连续cache；生产KV dtype/page/record布局及索引域。
- 同条件性能基准、延迟目标及精度要求。

上述未知项不阻碍按本文完成H64/BF16核心原型；涉及量化cache、小head及正式模型接入的结论在确认后更新。
