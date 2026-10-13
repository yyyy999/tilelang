# AutoSchedule：任务建模与流水 IR

## 01 · AutoSchedule

**任务建模与流水 IR**

程序中的操作，怎样成为调度器处理的任务？

任务的安排，怎样成为可执行的流水循环？

**程序 → 调度模型 → 流水 IR**

*IR：WSL 构建 789afc963205 · 源码核对：fbd7cf81c700*

---

## 02 · 贯穿全程的两次搬运

**四块输入数据，每块依次经过 GM → L1 → L0A**

*本次分享变体 · 源程序参数展开节选*

```python
T.annotate_buffer_versions({l1: (3, "iteration")})
for i in T.Pipelined(
    4, num_stages=3,
    annotations={"multi_buffer_eligible": [l1], "enable_offset": True},
):
    T.copy(A[i * 64 : (i + 1) * 64, :], l1)
    T.copy(l1, l0)
```

同一块数据必须先加载再消费。不同块之间，怎样安全地重叠？

*L1 固定三个版本；本例用于观察编译过程，没有数值输出。*

来源：[minimal.py · make_program](D:/codes/yyz/tilelang/autoshcedule_sharing/minimal.py:11)

---

## 03 · 整体编译流程

**AutoSchedule 写回调度安排，后续 pass 将安排落实为流水 IR**

**准备调度输入 → AutoSchedule → 流水落地**

识别任务与控制结构

分析读写、资源，估计代价

分配执行核，插入同步

物化缓冲版本，展开流水循环

*AutoSchedule pass 内部*

```text
DecodeScheduledTIR
        ↓
ScheduleSingleKernel
        ↓
EncodeScheduledTIR
```

*布局与合法化等前置步骤，以及后续 codegen，在图中省略。*

来源：[pipeline.py · 调度前后顺序](D:/codes/yyz/tilelang/tilelang/ascend/pipeline.py:102) · [auto_schedule.cc · AutoSchedule](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule.cc:1384)

---

## 04 · IR 中的任务边界

**一层标记调度单元，一层标记语义任务**

*真实 IR 节选 · 仅将 copy 参数换行*

```python
with T.attr({"stage": -1}, "tl.schedule_unit", 1):
    T.attr({}, "tl.ascend_task", 1)
    T.ascend_copy(T.region(A[i * 64, 0], 1, 64, 64),
                  T.region(l1[0, 0], 2, 64, 64))
```

- schedule_unit 保存 stage；stage=-1 表示尚未指定。
- ascend_task 包裹这次搬运；本例未显式写 T.Task。

*MaterializeScheduleUnits 后；外层循环也有自己的 schedule_unit。*

来源：[01_MaterializeScheduleUnits.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/01_MaterializeScheduleUnits.tir:25)

---

## 05 · 任务与控制模型

**循环成为 ControlNode，语义任务成为 TaskNode**

*模型结构示意*

```text
ControlNode / for i：四次迭代
├── TaskNode / Load
│   GM → L1
└── TaskNode / Move
    L1 → L0A
```

*ScheduledTIRDecoder · 源码节选*

```cpp
const auto *loop_node =
    body.as<ForNode>();
if (loop_node == nullptr)
  return DecodeTask(marker, body,
                    std::move(guards));
```

latency：结果就绪延迟；task II：发射间隔。

本次模型代价：Load 为 latency 272 / task II 82；Move 为 37 / 35。

*模型树是源码对应示意；代价值来自 EstimateLatency 后 IR，属于估计。*

来源：[scheduled_tir.cc · DecodeNode](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule/scheduled_tir.cc:242) · [03_EstimateLatency.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/03_EstimateLatency.tir:25)

---

## 06 · 数据依赖与槽位复用

**既要等数据准备好，也要等旧数据使用完**

**Load(i) 写入 → Move(i) 消费 → Load(i+3) 复用**

*三个版本下的槽位对应示意*

```text
逻辑迭代：   0       1       2       3
L1 槽位：    0       1       2       0
```

- Load(i) → Move(i)：同一块数据的生产与消费。
- Move(i) → Load(i+3)：同一槽位的覆盖保护。

*这是依赖含义的示意，省略其他依赖；Load 使用 MTE2，Move 使用 MTE1。*

来源：[dependency_analysis.cc · AnalyzeDependencies](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule/dependency_analysis.cc:419)

---

## 07 · 调度按层推进

**先安排子循环，再把子循环的汇总信息交给父层**

*嵌套结构示意*

```text
父控制结构
├── 子循环
│   ├── 任务 A
│   └── 任务 B
└── 后续任务
```

*ScheduleRecursive · 源码节选*

```text
for (auto &child : origin_children) {
  ScheduleRecursive(child);
}
Z3SchedulePythonLoop(ctrl,
                     manual_schedule);
```

依赖保证语义；pipe、HBM 等资源限制重叠；代价影响安排。

*任务 II、内层循环 II、父层看到的子循环整体代价，需要分别理解。*

来源：[auto_schedule.cc · ScheduleRecursive](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule.cc:424)

---

## 08 · stage 表示相隔的重复窗口

**循环 II 是迭代启动间隔；start time 决定任务落在哪个窗口**

```text
II=100（示意）
[0,100)    [100,200)    [200,300)    [300,400)
 A(0):20                             A(3):320 / B(0):340
```

*示意：II=100*

```text
A：20  = 0 × 100 + 20
B：340 = 3 × 100 + 40
```

A 为 stage 0，B 为 stage 3。

窗口 3 中：A(3) 与 B(0) 并存。

因此，流水轮次 t 对应 A(t) 与 B(t−3)。

*数值仅用于换算示意，非本次求解结果；stage 不包含全部窗口内时间信息。*

来源：[auto_schedule.cc · stage 恢复](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule.cc:1129)

---

## 09 · 调度结果写回 IR

**本次结果：Load 为 stage 0，Move 为 stage 3**

*真实 stage 字段 · 换行并省略任务体*

```python
with T.attr({"stage": 3},
            "tl.schedule_unit", 1):
    # Move 的任务体省略
T.attr({"stage": 0},
       "tl.schedule_unit", 1)
# Load 的任务体省略
```

*Encoder · 控制节点写回源码*

```cpp
For loop = control->control;
loop.CopyOnWrite()->body =
    EncodeList(control->children);
return EncodeControl(control,
                     std::move(loop));
```

原语句被保留；循环体按调度结果重建。

文本中 Move 排在前面；后续执行的是 Move(t−3) 与 Load(t)。

*此时循环仍为四次，版本维度、同步和迭代偏移尚未全部落实。*

来源：[04_AutoSchedule.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/04_AutoSchedule.tir:25) · [scheduled_tir.cc · EncodeNode](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule/scheduled_tir.cc:110)

---

## 10 · 同步落实完成关系

**两种交接分别保护消费与覆盖**

**Load 完成写入 → Move 可以读取**

*数据就绪 · 非连续真实调用节选*

```python
T.ascend_set_flag("MTE2_MTE1", i % 3)
T.ascend_wait_flag("MTE2_MTE1", i % 3)
```

**Move 使用完槽位 → 后续 Load 可以覆盖**

*槽位释放 · 非连续真实调用节选*

```python
T.ascend_set_flag("MTE1_MTE2", i % 3)
T.ascend_wait_flag("MTE1_MTE2", i % 3)
```

*InsertSync 后；节选省略 stage 包裹、入口初始化、末尾等待及其他同步。*

来源：[08_InsertSync.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/08_InsertSync.tir:24)

---

## 11 · 缓冲版本落实到分配与访问

**本次 L1 分配变成三个版本，访问增加槽位索引**

*真实分配节选 · 仅换行*

```python
l1_1 = T.sblock_alloc_buffer(
    (3, 64, 64), "bfloat16", data=l1.data,
    strides=(4096, 64, 1), scope="shared.l1")
```

**slot 0 → slot 1 → slot 2**

本例使用 iteration 模式：逻辑迭代 i 选择 i % 3。

stage 3 的消费稍后展开为逻辑迭代 i−3，槽位随之成为 (i−3)%3。

*MaterializeMultiBuffer 后；三个版本是本例固定输入，不是求解器自由选出的结果。*

来源：[09_MaterializeMultiBuffer.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/09_MaterializeMultiBuffer.tir:22)

---

## 12 · 流水循环中的逻辑迭代

**四次逻辑迭代展开为七次流水循环，仍只处理四块数据**

*真实 IR · 消费端节选，仅换行，省略同步*

```python
for i in T.serial(
    7, annotations={"enable_offset": True}
):
    if i - 3 >= 0:
        T.ascend_copy(
            T.region(l1_1[(i - 3) % 3, 0, 0],
                     1, 1, 64, 64),
            T.region(l0[0, 0],
                     2, 64, 64))
```

Load 在 i<4 时执行，写入槽位 i%3。

```text
轮次    逻辑操作
0       Load(0)
1       Load(1)
2       Load(2)
3       Move(0) → Load(3)
4       Move(1)
5       Move(2)
6       Move(3)
```

*右侧由真实 guard 推导，省略同步；表示逻辑执行顺序，不是设备时间线。*

来源：[10_LowerScheduledTIR.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/10_LowerScheduledTIR.tir:31)

---

## 13 · FA 中的嵌套控制与跨核交接

**相同模型扩展到更多任务、跨迭代状态和 Cube / Vector 协作**

**QK → softmax → pack P → PV → 累积 O**

外层两个 query tile，内层四个 KV block；m、l、O 跨迭代更新。

*真实跨核调用 · 分处 CUBE / VECTOR，省略 guard 和其他协议*

```python
T.ascend_cross_core_set_flag(4, "PIPE_FIX",
    (kv - 1 + local_mt * 4) % 3 + 6)

T.ascend_cross_core_wait_flag(4, "PIPE_V",
    (kv + local_mt * 4) % 3 + 6)
```

*流程为宏调用骨架示意，宏不等于任务；事件索引按各自逻辑迭代对应。*

来源：[fa_core.py · kv_auto](D:/codes/yyz/tilelang/autoshcedule_sharing/fa_core.py:224) · [FA LowerScheduledTIR · 生产者](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/fa/10_LowerScheduledTIR.tir:173) · [消费者](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/fa/10_LowerScheduledTIR.tir:267)

---

## 14 · 程序与流水 IR 的对应

**从两次搬运出发，每一步都保留程序语义**

**操作与循环 → 任务 / 控制树 → 顺序与 stage → 同步与槽位 → 流水循环**

- Decode / Encode：在标记化 TIR 与内部模型之间转换。
- 调度：在依赖和资源限制下形成任务安排。
- 落地：生成完成交接、版本访问和迭代边界。

stage、缓冲版本数、任务 II、循环 II 各有不同含义。

*本次完成编译与 IR 采集；模型代价不等于设备实测时间。*

---

## 15 · 接下来：求解器细节

**任务模型与流水生成之间的求解环节**

- C++ 怎样组织求解器输入，怎样读取输出？
- 依赖、资源和缓冲要求怎样写成约束？
- 怎样得到任务时间、版本数和循环 II？

接下来由同事展开输入输出协议、数学约束与求解过程。

*本部分已完成程序 → 调度模型 → 流水 IR 的链路。*

---

## 附录 1 · 任务边界与 guard

**作用域表示方法与节点类型是两个问题**

*叶任务结构示意*

```text
schedule_unit
└── if enabled          ← guard
    └── ascend_task
        └── copy
```

*控制结构示意*

```text
schedule_unit
└── if enabled          ← guard
    └── For
        └── 子调度单元
```

- Decoder 先记录外层 guard，再识别任务标记或直接的循环。
- 独立 T.attr 是作用域简写，范围延续到当前语句块末尾。
- 任务分组内部包含循环，仍可能是一个 TaskNode。

*附录 · with 的有无不决定节点类型；guard 在写回时恢复。*

来源：[scheduled_tir.cc · DecodeGuardedNode](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule/scheduled_tir.cc:209)

---

## 附录 2 · 用例与证据来源

**真实产物、源码对应与示意说明分开使用**

- 最小主线：offset=True、iteration 索引、固定三个 L1 版本。
- 保留原例 counter 基线；counter 只开启 offset 的拒绝记录另存。
- FA：256 / 512 / 128，自动调度，多个版本数由用户显式固定。

真实 IR 使用 WSL 现有构建 checkout 789afc963205；Windows 源码核对版本为 fbd7cf81c700。

未进行设备编译、NPU 数值验证或性能测试。

*附录 · 完整产物、运行配置和日志见 artifacts 下的 manifest.json。*

来源：[IR 阅读索引](D:/codes/yyz/tilelang/autoshcedule_sharing/IR-reading-guide.md:1) · [最小用例 manifest](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/manifest.json:1) · [FA manifest](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/fa/manifest.json:1)

---


