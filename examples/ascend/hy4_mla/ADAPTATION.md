# Hy4 preview BF16 调用适配合同

更新日期：2026-11-05。本文补充[TARGET.md](TARGET.md)的核心计算合同。

## 目标调用边界

对齐vLLM Hy4的`HYV4FlashMLASparseImpl._bf16_flash_mla_kernel`调用的底层函数：

```python
flash_mla_sparse_fwd(q, kv, indices, sm_scale,
                    d_v=512, attn_sink=None, topk_length=None, out=None)
```

固定源码参照：[Hy4 wrapper](https://github.com/vllm-project/vllm/blob/db914cad5bac261dbcd7cf05ded573c1d704f650/vllm/models/hy_v4/nvidia/flashmla_sparse.py#L217)、
[FlashMLA接口](https://github.com/vllm-project/FlashMLA/blob/847ef308bb7a6828c8117301f5a5a84a095a90e8/flash_mla/flash_mla_interface.py#L214)。
这是576→512的吸收投影后接口，不是Transformers中展开的256维QK、256维V的完整Attention接口。
FP8分页Decode、模型backend注册及indexer仍不在本次适配范围。

## 输入、输出和分派要求

| 项目 | 要求 | 实施方案 |
| --- | --- | --- |
| Q | `[N,H,576]`，BF16 | H64直接输入；1≤H<64补零到64；H>64明确拒绝 |
| KV | `[M,1,576]`，BF16 | 无拷贝去掉单例head维；M表示物理行视图范围 |
| indices | `[N,1,K]`，int32 | 无拷贝去掉单例head维；K为正的64倍数 |
| sm_scale | 调用方正的有限标量 | 原样用于QK缩放；不按576重新推导 |
| d_v | 512 | 其他值拒绝 |
| attn_sink | None或`[H]` FP32 | None编译为无sink；padding head的sink为-inf |
| topk_length | None或`[N]` int32 | 非None必须启用variable_lengths；None对应完整K |
| out | None或`[N,H,512]` BF16 | H64直接写调用方buffer；小H使用内部64-head输出后裁回并写out |
| 返回 | `(output, max_logits, lse)` | BF16 output，FP32统计量；Max/LSE不含sink |

Q/KV/indices/out均要求最后一维stride=1；允许按逻辑维顺序向外扩大的正pitch，
不接受转置、广播或内部重叠的布局。H64路径将pitch编译进StridedTensor，避免每次contiguous拷贝。
输出与输入的存储跨度不得重叠。小H padding/copy有额外开销；现有vLLM wrapper已补到H64时不重复padding。

空N返回相应空结果、不编译或launch。非空N要求M>0。
长度值范围`0≤length≤K`、物理索引与请求/因果mask的正确性由调用方保证；
适配层不通过读取设备张量来检查长度值，避免隐式同步。
负索引或index≥M仍由核心屏蔽。已展平的BF16分页cache可以接入，
但本层不接收block_table、不负责请求内索引到物理slot的转换。

KV前512维是共享latent、后64维是已旋转的key；Q也必须已经吸收key投影并完成RoPE。
输出后的value投影、gate、O投影由模型框架保留。

## 资源与生命周期

核心UB/L1/L0的shape、版本数和流水计算保持原有设计；本次只增加GM tensor pitch及显式输出入口。
编译缓存按shape、scale、sink/length开关、pitch和设备区分。
`out`只由本次调用写入，不保存在全局workspace缓存中，避免并发调用共用输出。
Max/LSE仍按调用分配；未提供的sink/length参数使用同设备的未读取占位张量。
本接口不承诺零分配或graph capture已验证；调用方out与小head fallback的成本需单独计量。

全无效行保持Output=0、Max=-inf、LSE=+inf。DCP合并如需LSE=-inf的中性元，
继续由上层执行空行修正。Sink=-inf表示无sink，+inf表示该head输出为零，统计量不变。
Q/KV限定有限值，不承诺一般NaN传播行为。

## 验收门槛

1. 本地CPU真实张量测试，使用mock executor验证接口分派与buffer绑定；不将其宣称为NPU计算通过。
2. 覆盖sink/length的None与非None、长度零、单例view、H8/H16/H32/H64、-inf补齐、预分配out和带pitch的输入输出。
3. 覆盖dtype/shape/device/非法stride及别名拒绝；H64调用方out必须直接作为kernel输出参数。
4. 本地资源扫描、语法、Ruff、文档和拼写检查通过后，再考虑集中离线lowering。
5. 新增pitch和显式输出路径仍须lowering、设备编译、launch、同步和数值验收；暂不重复上机测试。

实现及验证状态随本轮开发更新；本文件的要求不表示所有门槛已经通过。

## 本轮实现与验证记录

- 已新增`interface.py::flash_mla_sparse_fwd`，按是否提供长度/sink选择编译版本。
- `compile_bf16(explicit_outputs=True)`接受调用方O/Max/LSE；默认分配输出模式保持不变。
- 核心新增Q/KV/indices/out的GM pitch，on-chip分配和版本数保持不变。
- 本地CPU适配测试9项通过，原reference测试7项通过；适配测试还执行23个subtest。
  CPU executor是mock，不等于核心数值已在NPU通过。
- 本地22个核心资源配置扫描通过。K2048的UB保守预算仍为239488 B，L1地址终点458752 B。
- 新增`check_lowering --pitched`及`test_interface_npu.py`作为后续native/设备验收入口，本轮未运行。
- 未注册到vLLM-Ascend或Transformers，也未实现量化Decode、H128或专用小head高性能kernel。

可复现的本地检查（CPU PyTorch、pytest；无需TileLang原生库或NPU）：

```bash
python -m pytest --confcutdir=examples/ascend/hy4_mla \
  examples/ascend/hy4_mla/test_interface.py \
  examples/ascend/hy4_mla/test_reference.py -q
python examples/ascend/hy4_mla/check_resources.py
```

后续先做离线lowering，再进行设备验收：

```bash
python -m examples.ascend.hy4_mla.check_lowering \
  --topk 64 2048 --pitched --output /tmp/hy4-adapter-lowering
python -m pytest examples/ascend/hy4_mla/test_interface_npu.py -x -q
```
