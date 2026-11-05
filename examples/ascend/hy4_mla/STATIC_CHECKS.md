# 本地静态检查记录

日期：2026-11-05。对象：当前 Hy4 preview BF16 核心。

## 已修正的问题

初稿将 gather UB 从每行512扩到640（后128维SIMD load只取64维RoPE，其余为padding），仍沿用三版本。
TopK2048时，按独立存储逐个版本累加为280448 bytes（273.875 KiB），高于本后端248 KiB预算。
当前改为两版本，减少40960 bytes。此预算选择不依赖编译器额外推导出的生命周期复用；
原始总和超预算本身不能证明任意内存复用方案均不可行。

初稿另用`T.view`将64×128的L0 buffer缩为64×64，违反同bit数约束。
已删除该view，改用`aq[:, :64]`和`bq[:, :64]`作为RoPE GEMM区域。
父buffer继续使用major-K布局，RoPE copy显式使用`l0_pitch=4`。
参考后端`GemmMAD._l0_operation_extents`：MAD的M/N/K来自操作区域，且末两轴须零原点；
RoPE操作为M=N=K=64，区域满足此约束。

## 容量口径与结果

运行：

```bash
python examples/ascend/hy4_mla/check_resources.py --details
python -m unittest examples.ascend.hy4_mla.test_static -v
```

脚本直接读取`kernel.py`的分配、view、版本数和固定地址，容量来自
`src/ascend/target_utils.h`。按独立物理buffer、每版本对齐累加，alias只计一次。
`annotate_manual_multi_buffer`的分区计数不再次乘到物理大小上。
UB按每个AIV分别计算，不能把两颗AIV的UB合为一池使用。
本实现使用SIMD VF，没有SIMT的额外32 KiB context reservation。

| 池 | TopK2048保守占用 | 已扫描分支最大值 | 后端容量 |
| --- | --- | --- | --- |
| UB | 239488 B / 233.875 KiB | 241024 B / 235.375 KiB | 248 KiB |
| L1固定地址终点 | 458752 B / 448 KiB | 同左 | 512 KiB |
| L0A | 65536 B | 65536 B | 64 KiB |
| L0B | 65536 B | 65536 B | 64 KiB |
| L0C | 147456 B / 144 KiB | 同左 | 256 KiB |
| SS | 96 B（保守对齐） | 同左 | 3 KiB |

扫描K=64/128/192/256/320/512/640/704/960/1024/2048，各自pipeline开/关，共22个配置。
五项stdlib unittest通过，其中包含超容量、错误view和固定地址重叠的故障注入，
以及QK分块恰好覆盖576维、末块64维的对齐检查。
本地Ruff和Python语法检查通过。L0A/B无剩余容量，后续不得直接新增独立L0 buffer。

## 新增数据路径的地址核对

- GM gather每行读取576个BF16元素，UB行距640；一次至多读取两个所选行。
  有效pair按lo/hi排序，第二项无效时复制lo但mask其概率；双无效时n_burst=0。
  原有零burst局部assumption仍需在生成DMA指令中复核，源码预算检查不证明DMA语义。
- latent packing两个列组分别读取`row*640 + {0,128}`及`row*640 + {256,384}`，
  每次128元素，正好覆盖每行前512维。
  第二次load后的普通步长512，加上第一次的128，推进到下一行；
  最后一行的步长-19712将下一列组起点推进256。
- RoPE load从`row*640+512`读取128个BF16，末端正好落在当前行末；
  PAT_VL64只将前64维写入compact NZ。
  合并后的`(33,576)`NZ地址为`(col//16)*33*16 + row*16 + col%16`，
  row<32、512≤col<576的最大写地址为18991，低于分配19008元素。
- RoPE的L0A/B使用父major-K存储前四个C0块（K64），
  对64行每块占64×16元素，总计4096元素，落在父8192元素范围内。
  后32行B的copy通过`l0_pitch=4`保持64行父布局的panel间距。
- 四个latent QK GEMM累加512维，RoPE GEMM再累加64维并设置最后的UnitFlag=3，
  然后dual-copy score。PV仍读取latent512，未将RoPE加入value。

## 流水生命周期分析与待验证项

主目标K2048无query预取：metadata stage0 → gather stage1 → packing/copy stage2 →
QK/softmax stage4 → PV/output stage5。
gather只生产kv，两个packing VF均在stage2消费同一版本；后续QK/PV消费L1而不再读kv。
两版本要求第i+2次gather覆写前，第i次latent和RoPE packing均已结束，
由保留的真实数据依赖与owner-exclusion约束调度，未手动删除同步。
query预取分支有额外stage平移，部分分支producer与consumer的stage距离为2，
重用边界可能需要等待，不能据此宣称双缓冲与原版三缓冲性能相同。

下一道门槛是`tilelang.lower(..., enable_device_compile=False)`：检查真实Pass可达性、
调度可满足性、版本选择、最终allocation和生成copy/MAD/FixPipe指令。
该步骤不分配输入张量、不launch、不同步设备。通过后才集中进行少量NPU精度测试。
本地AST预算检查不替代这个门槛，也不代表设备正确性或高性能已经通过。

第一次离线lowering（`49e276eb`）通过前端和AutoSchedule后，在InsertSync报告跨核flag槽分配失败，
没有生成最终allocation或启动设备。检查`insert_sync.cc::FlagAllocator::Plan`可见每bank只有16槽。
分开的latent/RoPE L1存储增加了独立发布/回收通道；这是设计代价，不应靠删同步规避。
当前已将KV、Q和NZ各自合为576维存储，保留原有两条KV L1通道。
每个KV L1环变为144 KiB四版本，Q最多144 KiB，S为16 KiB。
固定区间为：KV0 `[0,147456)`、Q `[147456,294912)`（含单slot时的预留空间）、
KV1 `[294912,442368)`、S `[442368,458752)`；互不重叠。
QK合入同一五次迭代的流水循环，sub-K依次为128/128/128/128/64，
最后一次才发布UnitFlag=3；PV仍切前512维。
统一consumer循环也避免为末尾RoPE额外引入一个L1消费同步区域。
动态sub-K的copy/MAD合法性仍由离线lowering核实。
该改法已重新通过本地预算与别名检查；
跨核同步可分配性由下述离线lowering结果确认；设备执行行为仍未验证。

合并方案的离线lowering已于提交`7275bca5`通过K2048和K64（nq65、nk257、长度启用）。
已取回生成AscendC与TIR在本地审阅：

| 池 | K2048最终allocation | K64最终allocation |
| --- | --- | --- |
| UB | 239488 B | 234880 B |
| L1 | 458752 B | 458752 B |
| L0A/B各自 | 65536 B | 65536 B |
| L0C | 147456 B | 147456 B |
| SS | 64 B | 64 B |

最终arena与本地预算相符（SS实际小于保守对齐预算）。生成QK MAD使用
`min(128,576-d*128)`、d=0..4，最终UnitFlag在d4置3；PV MAD仍为64×64×128。
KV gather每行1152字节、UB行距1280字节；UB→L1一次传36个C0块。
这些只是生成代码事实，未证明设备DMA和同步实际运行正确。

本地审阅生成代码另发现：uint32行索引乘576可能在转换为int64之前溢出。
已将`safe_lo`在进入StridedTensor索引前显式提升为int64。
最终代码提交`dd476907`再次通过K2048和K64离线lowering，生成GM源地址为
`pairs[safe_lo_aiv0 * (int64_t)576]`（AIV1同样），确认乘法在64位进行。
最终离线产物位于A5的`/tmp/hy4-lowering-dd476907/`，未进行设备二进制编译、launch或性能测量。

已有执行记录：初稿CPU reference的7项测试通过；第一次NPU测试请求实际停在
`T.view`前端构建错误，没有kernel launch。此后已停止反复进行设备测试。
