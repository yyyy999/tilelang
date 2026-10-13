# AutoSchedule：从程序到流水 IR

> 分享日期：2026-10-13，19:00 开始  
> 本文覆盖：我的分享部分，建议约 25 分钟，问答另计  
> 当前状态：第一至第七节材料已完成，待对照阅读、提问与试讲修订；最小原版、流水变体和 FA 的真实 IR 已采集  
> 源码核对基线：fbd7cf81c700dc3f5b2d54fa9ef1b5d7cc8c7907；实际 WSL 运行编译器：789afc9632059c7404ffd10e188c6a073be40e15

## 阅读与维护约定

本文是分享材料的主文件，先完成 Markdown 内容，再决定是否制作 PPT。

技术内容按“机制 → 真实 IR → 关键实现”的顺序补充。代码只保留支撑解释的关键逻辑，注明文件、函数及核对时的位置。IR 只截取必要部分，注明 pass 阶段和运行配置。

统一使用以下证据标记：

- **[代码确认]**：由当前源码直接确认。
- **[实跑结果]**：来自指定版本、配置和用例的实际输出。
- **[推测]**：尚待验证的解释。
- **[示意]**：用于理解的简化结构或 IR，不是实际运行输出。

本次真实 IR 与观察结果见 [IR-reading-guide.md](IR-reading-guide.md)。运行使用 WSL 现成构建；stage 等事实引用对应产物，未采集到的循环 II 和设备性能不预填。

建议第一次按第一至第五节顺序阅读，先看机制解释和真实 IR，再沿代码入口核对；第二遍补读 FA 与实现细节。各节正文用于理解和查阅，台上优先展示指定片段并参考“台上讲述”，不逐字朗读全部正文。预定 25 分钟仍需通过试讲校准。

## 分工与叙述边界

我的部分先独立讲完完整链路：整体框架、任务与控制结构、依赖和资源概念、分层调度、调度结果写回，以及同步、缓冲版本和流水展开。

同事随后展开 Z3 输入输出协议、数学约束与求解细节。我的部分只说明接口信息代表什么、结果如何被后续编译流程消费，不展开字段协议、约束公式和搜索算法。

我的部分结束时，听众应能回答：

1. 原始程序如何转换成调度器处理的任务与控制结构？
2. 调度器主要受到哪些依赖、资源和代价信息影响？
3. 调度结果如何转换成带同步、版本索引和边界保护的流水 IR？

## 讲述路线与时间

1. 目标与全景：2 分钟。
2. 程序成为任务与控制模型：5 分钟。
3. 调度机制：4 分钟。
4. 调度结果写回 IR：4 分钟。
5. 同步、缓冲版本与流水展开：6 分钟。
6. FA 补充：3 分钟，可选。
7. 回顾与交接：1 分钟。

总计约 25 分钟。FA 放到备份材料时，仍有约 22 分钟的完整主线。

---

## 1. 目标与全景：我们在编译流程的哪里？

**建议时长：2 分钟；台上展示两页，代码入口和 IR 节选作为备查。**

### 1.1 本节的核心结论

**AutoSchedule 这条编译链路，把程序中的操作组织成调度任务，安排任务的顺序与跨迭代关系，再由后续 pass 将这些安排落实为带同步和缓冲版本的流水 IR。**

其中，名为 AutoSchedule 的 pass 负责读入内部模型、完成调度并写回带调度信息的 IR；同步、版本物化和流水循环展开还有各自的后续 pass。

本节结束时，听众只需要建立三个位置关系：原始程序在入口，求解器在调度环节，真正的流水 IR 在后续落地环节。

### 1.2 台上展示第一页：用最小程序提出问题

主线用例：[minimal.py / make_program](D:/codes/yyz/tilelang/autoshcedule_sharing/minimal.py:11)。

**[代码确认] 源程序节选**，来自函数体，省略参数定义、分配和返回语句。本次分享变体的实参为 enable_offset=True、version_mode="iteration"；L1 版本数固定为 3。

~~~python
T.annotate_buffer_versions({l1: (3, version_mode)})
for i in T.Pipelined(
    4,
    num_stages=3,
    annotations={"multi_buffer_eligible": [l1], "enable_offset": enable_offset},
):
    T.copy(A[i * 64 : (i + 1) * 64, :], l1)
    T.copy(l1, l0)
~~~

先只解释操作本身：A 是 GM 中的输入，每次搬一个 64×64 的块；先从 GM 搬到 L1，再从 L1 搬到 L0A，共重复四次。这里没有 GEMM，也没有数值输出，用来观察编译变化。

然后提出贯穿分享的问题：

> 每一块数据仍然必须先加载、再消费。如果想让不同迭代的搬运重叠，编译器需要知道什么？它又怎样保证消费时数据已经准备好、下一次加载不会覆盖还在使用的槽位？

此处不展开三个版本的选取、iteration 模式、stage 或依赖公式；相关注解的含义在后续章节解释。原例和分享变体的完整区别见 [IR 阅读索引](D:/codes/yyz/tilelang/autoshcedule_sharing/IR-reading-guide.md)。

### 1.3 台上展示第二页：从程序走到流水 IR

**[示意] 根据已核对的 pass 顺序绘制的概念流程图，不是真实 IR dump。** 图中只展开本次讲述的主线，省略前置合法化和后续具体指令、布局及代码生成细节。

~~~mermaid
flowchart TD
    A["TileLang 程序 / PrimFunc<br/>描述操作、循环和缓冲区"]
    B["准备调度输入<br/>任务化、缓冲资格和代价估计"]
    subgraph S["AutoSchedule pass"]
        C["DecodeScheduledTIR<br/>读取任务与控制模型"]
        D["分层调度<br/>分析依赖与资源，调用求解器"]
        E["EncodeScheduledTIR<br/>写回任务顺序、stage 和缓冲版本信息"]
        C --> D --> E
    end
    F["核分配与同步准备<br/>AssignCore → PrepareMultiBuffer<br/>→ ResolveCore → InsertSync"]
    G["MaterializeMultiBuffer<br/>将版本计划落实到分配与访问"]
    H["LowerScheduledTIR<br/>展开迭代偏移与边界保护"]
    I["后续 lowering / codegen<br/>生成 Ascend 源码"]
    A --> B --> C
    E --> F --> G --> H --> I
~~~

讲图时按三段走，先让听众看清每段的输入与输出：

1. **准备：程序成为可调度的描述。** 操作被识别为任务，循环等控制信息得到保留，任务补上后续调度需要的代价信息。
2. **调度：决定怎样安排任务。** 内部模型包含任务和控制结构；调度器根据依赖、资源和代价安排任务，再把结果写回 IR。求解器的具体协议和数学细节由同事随后展开。
3. **落地：安排成为具体 IR。** 后续 pass 生成核归属、同步和版本访问，并把 stage 关系展开成带逻辑迭代偏移和边界保护的循环。

需要指明的关键边界：**经过 AutoSchedule 后，IR 已带有调度安排；走完后续同步、版本物化和流水展开，才看到本次要解释的流水循环。**

### 1.4 两分钟讲稿

以下为试讲草稿，可按自己的表达习惯调整；这段是台上口述，后面的实现和 IR 节选不逐行讲解。

> 今天我先讲 AutoSchedule 的整体编译过程，重点回答：一个描述操作和循环的程序，怎样变成能够流水执行的 IR。我的部分会先把这条链路讲完整，之后由同事展开求解器的协议、约束和求解细节。
>
> 我们先看一个很小的程序。循环执行四次，每次只有两项搬运：从全局内存把一块数据搬到 L1，再从 L1 搬到 L0A。数据当然要先加载才能消费，但不同迭代之间能否重叠？使用多个缓冲槽位之后，什么时候可以安全地再次写入？我们就用这两项操作追踪编译器的处理过程。
>
> 整体流程可以分成三段。第一段准备调度输入：识别任务、保留循环结构，并补充代价信息。第二段是 AutoSchedule：将 IR 读成内部模型，根据依赖和资源安排任务，再把顺序、stage 和缓冲版本信息写回 IR。求解器就在这一段。
>
> 第三段把这些安排落实下来：分配执行核，生成同步，把缓冲版本变成具体的分配和访问，最后展开流水循环。所以，调度结果和最终流水 IR 之间，还有一段需要解释的编译过程。
>
> 后面我们始终追踪这个小程序：先看两次 copy 怎样成为任务，再看调度怎样安排它们，最后看这些安排如何体现在真实 IR 中。

### 1.5 实现备查：在代码中确认全景的两个位置

**[代码确认] pipeline 负责串联阶段。**

位置：[tilelang/ascend/pipeline.py / AscendPassPipelineBody](D:/codes/yyz/tilelang/tilelang/ascend/pipeline.py:106)。以下是连续源码节选，前面的控制流规范化等 pass 未展示。

~~~python
mod = ascend_transform.MaterializeScheduleUnits()(mod)
mod = ascend_transform.AnnotateMultiBufferEligible()(mod)
mod = ascend_transform.EstimateLatency()(mod)
mod = ascend_transform.AutoSchedule()(mod)
mod = ascend_transform.AssignCore()(mod)
mod = ascend_transform.PrepareMultiBuffer()(mod)
mod = ascend_transform.ResolveCore()(mod)
mod = ascend_transform.InsertSync()(mod)
mod = ascend_transform.MaterializeMultiBuffer()(mod)
mod = ascend_transform.LowerScheduledTIR()(mod)
~~~

只圈出 AutoSchedule 的位置和前后边界，不在第一节解释每个 pass 的内部实现。

**[代码确认] AutoSchedule 内部的模型读写边界。**

位置：[src/ascend/transform/auto_schedule.cc / AutoSchedule](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule.cc:1384)。以下是源码摘录，删除了版本覆盖记录与告警检查，保留核心调用。

~~~cpp
ScheduledTIR scheduled_tir = DecodeScheduledTIR(
    context.root, context.outer_ctx, MakeAutoScheduleExtraInfo);
ScheduleSingleKernel(context.root->body, target, &scheduled_tir);
return EncodeScheduledTIR(std::move(scheduled_tir));
~~~

这里的 Decode 是从标记化 TIR 读取内部调度模型，Encode 是把模型写回 TIR。暂不展开树结构或 Z3 调用，它们分别放到第二、三节。

### 1.6 真实 IR 备查：提前看清本次链路的终点

**[实跑结果]** 使用 WSL 编译器 checkout 789afc963205；对应 minimal_offset，启用 offset、采用 iteration 索引并固定三个 L1 版本。以下两段来自同一轮成功运行。

入口：[00_frontend.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/00_frontend.tir:18)。这个前端 IR 的循环仍然执行四次，循环体是两次语义搬运；当前只观察循环头。

~~~python
for i in T.serial(4, annotations={"enable_offset": True, "multi_buffer_eligible": [l1_1], "num_stages": 3}):
~~~

流水展开后：[10_LowerScheduledTIR.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/10_LowerScheduledTIR.tir:31)。只摘录循环头和相邻的第一项边界保护，完整搬运与同步仍在原文件中。

~~~python
for i in T.serial(7, annotations={"enable_offset": True}):
    if i - 3 >= 0:
        T.ascend_pipe_barrier("PIPE_MTE1")
~~~

这两段只用于预告：编译器已经把四次逻辑迭代转换为七次流水循环，其中加入了边界保护和同步。为什么是七次、为什么出现 i-3，留到第五节结合完整 IR 解释。七次循环不意味着处理七块输入。

**证据边界：** 本节代码入口来自 Windows checkout fbd7cf81c700；真实 IR 来自 WSL 现有构建所用 checkout 789afc963205。这里使用已核对的共同主线建立对应，不声称 Windows 当前 checkout 已重新构建运行；也没有设备时延或数值正确性结论。

### 1.7 第一节完成检查与过渡

- [x] 最小源程序与贯穿问题已准备。
- [x] 全景流程图已准备。
- [x] 两分钟讲稿已准备。
- [x] 关键实现入口与真实 IR 位置已核对。
- [ ] 完成口头试讲，确认两分钟内能讲清三个阶段。

过渡到第二节：

> 现在我们知道这条编译链路从哪里开始、在哪里结束。接下来先看第一步：编译器怎样把这两次 copy 和外层循环，读成调度器能够处理的任务与控制结构？

## 2. 程序成为任务与控制模型

**建议时长：5 分钟。主线：两次 copy → 两个任务；外层循环 → 一个控制节点。**

### 2.1 先回答“调度器到底在安排什么”

调度器要安排的是具有读写、资源和代价信息的任务，而程序本身还包含循环、条件与缓冲区等结构。把这些信息读成统一的内部模型，才能在保留语义的前提下讨论重排和跨迭代重叠。

**[代码确认]** TaskNode 保存任务的原始语句、访问区域、资源信息和代价；ControlNode 保存原始 For、子节点列表以及循环相关信息。控制节点还有自己的辅助 task，记录循环边界等表达式的访问；它不是另一次实际搬运。

对最小用例，先给两个主要任务起讲解用的名字：

- **Load(i)**：读 A 的第 i 块，写 l1；数据路径是 GM→L1。
- **Move(i)**：读 l1，写 l0；数据路径是 L1→L0A。

Load、Move 是本文中的讲解名称，不是打印 IR 中的 task ID。

### 2.2 真实 IR：先有任务边界，再有任务代价

**[实跑结果]** [01_MaterializeScheduleUnits.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/01_MaterializeScheduleUnits.tir:25) 的循环体节选。省略外部循环头，按原顺序保留两个任务：

~~~python
with T.attr({"stage": -1}, "tl.schedule_unit", 1):
    T.attr({}, "tl.ascend_task", 1)
    T.ascend_copy(T.region(A[i * 64, 0], 1, 64, 64), T.region(l1[0, 0], 2, 64, 64))
T.attr({"stage": -1}, "tl.schedule_unit", 1)
T.attr({}, "tl.ascend_task", 1)
T.ascend_copy(T.region(l1[0, 0], 1, 64, 64), T.region(l0[0, 0], 2, 64, 64))
~~~

这里有两层不同职责的标记：

- tl.ascend_task 描述一个语义任务；任务可以由编译器推断，也可以由用户显式分组。
- tl.schedule_unit 是调度结构的包裹，保存 stage；循环也会被它包裹，不能把每个 schedule_unit 都数成叶任务。
- stage=-1 表示尚未指定调度 stage，不是负数流水级，也不是非法数据。
- IR 打印器会把包裹剩余语句的 AttrStmt 打印成独立的 T.attr 行；第二个任务没有 with 字样，不代表任务标记失效。

最小源程序没有显式写 T.Task。**[代码确认]** MaterializeScheduleUnits 先通过 TaskNormalizer 规范任务边界，再创建 schedule unit，因此基本搬运操作也能成为任务。

接着看 [03_EstimateLatency.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/03_EstimateLatency.tir:25)，同一循环体变为：

~~~python
with T.attr({"stage": -1}, "tl.schedule_unit", 1):
    T.attr({"ii": T.int64(82), "latency": T.int64(272)}, "tl.ascend_task", 1)
    T.ascend_copy(T.region(A[i * 64, 0], 1, 64, 64), T.region(l1[0, 0], 2, 64, 64))
T.attr({"stage": -1}, "tl.schedule_unit", 1)
T.attr({"ii": T.int64(35), "latency": T.int64(37)}, "tl.ascend_task", 1)
T.ascend_copy(T.region(l1[0, 0], 1, 64, 64), T.region(l0[0, 0], 2, 64, 64))
~~~

读法是：Load 的模型 latency 为 272、任务 II 为 82；Move 分别为 37、35。这些是调度代价，不是本次 NPU 实测时间。为什么是这些数，由 EstimateLatency 的模型决定；第一轮分享不展开带宽和指令代价公式。

#### 怎样识别一个任务单元？

把本例的打印形式展开后，可以按下面的结构理解。**[示意] 这是节点嵌套关系，不是真实 IR dump：**

~~~text
循环体
├── schedule_unit
│   └── ascend_task
│       └── GM→L1 copy
└── schedule_unit
    └── ascend_task
        └── L1→L0A copy
~~~

有 with 时，缩进明确限定包裹范围；独立 T.attr 是作用域的简写，包裹它后面直到当前语句块末尾的内容。两种打印方式不改变任务结构。

循环本身还有一层 schedule_unit，它直接包着 For，对应 ControlNode；上面两个直接包着 ascend_task 的单元对应 TaskNode。因此这里有一个循环单元和两个叶任务单元。

#### 外层 guard 是什么？

guard 就是“这项操作在什么条件下执行”。例如条件包在任务外面时：

~~~text
schedule_unit
└── if enabled
    └── ascend_task
        └── copy
~~~

**[示意]** enabled 是该任务的外层 guard。Decoder 先把条件记录在节点上，再向内判断是 ascend_task 还是直接的 For；Encode 时会恢复这个条件。

这里的“剥离 guard”是将条件与主体分开保存，条件没有丢弃。显式任务内部的 For 仍是任务实现的一部分，不能只因为某处出现 For 就把整个单元识别为 ControlNode。


### 2.3 模型如何保留“操作、控制和上下文”

**[示意] 下图由源码和 IR 建立对应，不是实际打印的内部对象：**

~~~text
ScheduledTIR
├── metadata：kernel 原有外壳、版本注解等
└── tree：根级有序节点列表
    └── ControlNode：for i，范围 0…3
        ├── TaskNode / Load
        │   原始语句：GM→L1 copy
        │   读：A 的第 i 个 64×64 区域；写：l1
        │   资源：MTE2；代价：latency=272、task II=82
        └── TaskNode / Move
            原始语句：L1→L0A copy
            读：l1；写：l0
            资源：MTE1；代价：latency=37、task II=35
~~~

这里的关键是“原始语句仍然保留”。模型不是将程序替换为一组数字后丢弃 IR；后面 Encode 会复用这些语句，并重建控制结构。

对于条件和符号环境，**[代码确认]** Decoder 将 guard 与节点关联，并保存外层约束上下文，例如循环变量范围、已有绑定和条件事实。这样分析两个访问区域时，才知道它们在什么条件下可能相交。最小用例没有复杂条件，FA 的 VF 内部循环也不会全部变成顶层可独立调度的任务。

读版本 map 时还要区分 Buffer 与 storage：Buffer 描述形状、类型、作用域等视图信息，buffer.data 标识相关存储。不同视图可以共享 data，因此分析不能只比较名字；本例版本 map 中的 l1.data 就是这种存储身份。后续独立分配是否允许物理复用则是另一个问题。

### 2.4 关键实现：读循环，还是读叶任务？

位置：[scheduled_tir.cc / ScheduledTIRDecoder::DecodeNode](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule/scheduled_tir.cc:242)。

**[代码确认] 核心分派的连续节选：**

~~~cpp
const auto *loop_node = body.as<ForNode>();
if (loop_node == nullptr)
  return DecodeTask(marker, body, std::move(guards));
~~~

遇到 For 时，代码保留原循环，再递归读取 body：

~~~cpp
control->control = GetRef<For>(loop_node);
// 省略循环边界访问分析和约束作用域。
control->children = DecodeList(loop_node->body, control.get());
~~~

上段为同一函数中的非连续摘录，注释标明省略内容。

叶任务的主要信息在 [DecodeTask](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule/scheduled_tir.cc:282) 中恢复。以下为非连续源码摘录：

~~~cpp
task->stmt = body;
task->SetStage(GetScheduleUnitStage(marker));
// 省略 core、guard 和 outer_ctx 的恢复。
ApplyTaskResourceUsage(AnalyzeTaskResourceUsage(body), task.get());
ApplyTaskAccesses(AnalyzeTaskAccesses(body), task.get());
~~~

**读代码时要抓住的关系：** schedule_unit 恢复结构与 stage，task marker 恢复语义任务及代价，语句分析恢复读写和资源。所有信息并不都印在一份 TIR 文本里。

补充入口：[TaskNode / ControlNode](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule/ir_structure.h:494)、[ScheduleUnitMaterializer](D:/codes/yyz/tilelang/src/ascend/transform/materialize_schedule_units.cc:423)、[TaskCostEstimator](D:/codes/yyz/tilelang/src/ascend/transform/estimate_latency.cc:269)。

### 2.5 台上讲述与过渡

> 两次 copy 首先得到语义任务边界和调度单元标记；代价估计再补上 latency 与 II。Decoder 把循环读成控制节点，把两次搬运读成叶任务，并保留访问、资源、条件和原始语句。这样调度器既能讨论任务之间的限制，也有足够的信息在调度后还原程序。

本节重点展示任务化和代价估计两份 IR，任务树用示意图解释。下一节转入：**有了任务模型，什么关系限制它们的执行安排？**

## 3. 调度机制：怎样安排这些任务？

**建议时长：4 分钟。主线：数据要求、资源限制、分层求解。**

### 3.1 依赖来自读写关系，不只来自语句相邻

Load(i) 写 l1，Move(i) 读 l1，因此同一块数据必须先生产、再消费。这是 RAW：写后读。

同一逻辑缓冲还会带来覆盖保护：后续写入不能破坏尚未完成的读取，这是 WAR；两次写入同一存储的关系属于 WAW。分析需要考虑存储身份、访问区域、guard 和迭代关系，不能只按 buffer 名字或源码行号连边。

**[代码确认]** [AnalyzeDependencies](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule/dependency_analysis.cc:419) 明确考察三类访问组合：

~~~cpp
consider(accesses[i].writes, accesses[j].reads);
consider(accesses[i].writes, accesses[j].writes);
consider(accesses[i].reads, accesses[j].writes);
~~~

这些组合是分析入口，不代表任意两个任务一定产生边：是否真正冲突还要经过区域及上下文判断。

### 3.2 多缓冲怎样改变“后续覆盖”的含义

**[示意] 本次固定三个版本，按逻辑迭代选择槽位：**

~~~text
逻辑迭代 0 → slot 0
逻辑迭代 1 → slot 1
逻辑迭代 2 → slot 2
逻辑迭代 3 → slot 0
~~~

所以对于 L1，Move(0) 完成对 slot 0 的使用后，Load(3) 才能安全重用它。不是每次下一轮加载都必须覆盖同一个物理位置。

概念上需要同时表达：

~~~text
Load(i) → Move(i)             同一块数据的生产与消费
Move(i) → Load(i+3)           三个版本下，同一物理槽位的复用保护
~~~

**这些箭头是语义示意，不是导出的完整依赖图。** 内部模型会将存储版本与跨迭代依赖联系起来，具体编码和约束由同事说明。最小用例还有 L0A 的写入关系与资源约束，不应把这两条箭头当作全部模型。

### 3.3 数据允许重叠，还要看硬件资源

Load 使用 MTE2，Move 使用 MTE1，因而存在利用不同执行资源重叠搬运的机会。两个任务没有数据冲突，也不自动意味着可以任意同时发射：同 pipe、HBM 通道等资源仍可能限制安排。

这里需要区分三个代价概念：

- **任务 latency**：模型中，从该任务开始到相关结果可供后续使用的延迟。
- **任务 II**：模型中，该任务占用相关发射/执行资源而形成的最小发射间隔。
- **循环 II**：循环稳态相邻逻辑迭代启动之间的模型间隔，由整个循环的依赖和资源共同决定。

最小用例中 82 和 35 是两个任务的 II。**不能据此直接宣布循环 II 是 82，也不能用 272÷82 推出 task stage。** 当前普通 TIR 快照没有完整保存本次循环 II 和 start times。

### 3.4 嵌套程序怎样分层处理？

**[代码确认]** 调度按树自底向上推进。先完成子控制结构，再将子结构的代价、访问和资源信息提供给父层。它不是把所有循环的所有叶任务一次性放进一个全 kernel 联合模型。

位置：[auto_schedule.cc / ScheduleRecursive](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule.cc:424)。以下为控制节点分支中的连续关键代码：

~~~cpp
for (auto &child : origin_children) {
  ScheduleRecursive(child);
}
Z3SchedulePythonLoop(ctrl, manual_schedule);
~~~

根级任务列表则在子节点处理后由 ScheduleList 进行列表调度。循环关注迭代之间的重复安排，普通列表关注这组节点的整体完成安排；具体目标函数和搜索方法留给同事。

父层看到的循环代价也需要与内层循环 II 区分。以下为 [循环代价汇总](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule.cc:1272) 的非连续源码摘录：

~~~cpp
overall_latency += ii * (ctrl->GetTripCount() - 1);
ctrl->SetII(overall_latency);
ctrl->SetLatency(overall_latency);
GetAutoScheduleExtraInfo(ctrl)->SetIIperIter(ii);
~~~

代码中的 ii 是内层循环的每迭代 II；父层节点的 GetII 则被设为循环整体代价。这里不是说这两个量定义相同，而是该分层实现把子循环整体作为父层的一项安排对象。理解此处即可，不展开代价汇总公式的推导。

### 3.5 本部分与求解器讲解的接口边界

用概念描述即可：输入提供任务代价、依赖、资源、缓冲大小和版本要求等；循环求解返回任务时间、版本选择和循环 II，C++ 再消费结果。字段格式、负数 distance 编码、容量约束与求解过程不在这里展开。

**[代码确认]** 本例 L1 三个版本由用户固定，L0A 在调度后版本 map 中为 1。任务代价属于模型估计；stage 和 IR 是实跑产物；整体性能收益尚未测量。

### 3.6 台上讲述与过渡

> 调度首先要保证数据生产、消费和覆盖安全，再满足执行资源的限制。多个缓冲版本让不同迭代有机会使用不同槽位，但不会消除依赖。对于嵌套循环，编译器从内向外安排，把子循环的汇总信息交给父层。接下来，我们看这次求出的安排如何写回 IR。

## 4. 调度结果如何写回 IR？

**建议时长：4 分钟。主线：时间安排 → stage 与顺序 → 可继续处理的 TIR。**

### 4.1 从模型时间到流水 stage

**先分别理解两个时间量，再看 stage。** 循环 II 表示模型中相邻逻辑迭代相隔多久启动；start time 表示某项任务在基准逻辑迭代中的开始位置。不同任务可以在相同逻辑迭代开始之后很久才执行，甚至跨过多个迭代启动间隔。

**[示意] 以下数值仅用于解释换算，不是最小用例的求解结果。** 假设循环 II=100，任务 A 的 start time=20，任务 B 的 start time=340。将模型时间每 100 划成一个窗口：

~~~text
窗口 0：[  0, 100)   A 位于 20
窗口 1：[100, 200)
窗口 2：[200, 300)
窗口 3：[300, 400)   B 位于 340
~~~

A 落在窗口 0，B 落在窗口 3，所以得到 stage(A)=0、stage(B)=3。窗口是对求解模型时间的分组，不是硬件里额外存在的执行阶段。

换算可以理解为除法的商与余数：

~~~text
A：20  = 0 × 100 + 20    stage=0，窗口内位置=20
B：340 = 3 × 100 + 40    stage=3，窗口内位置=40
~~~

stage 只保留“隔了几个窗口”的粗粒度信息。这里 A、B 的开始时间差是 320，并不恰好等于 3×100；窗口内的位置仍影响同一流水轮次中任务的排列。因此 stage 不能完整表达 start time。

**为什么这会成为跨迭代流水？** 每项任务都按循环 II 重复出现。延续上面的示意数值：

~~~text
A(0) 在 20；A(1) 在 120；A(2) 在 220；A(3) 在 320
B(0) 在 340；B(1) 在 440；B(2) 在 540
~~~

于是窗口 3 中既有 A(3)，又有 B(0)：它们处于同一个重复窗口，但属于不同逻辑迭代。这就是把 stage 转换成迭代偏移的依据。对于 stage 0 与 stage 3，流水轮次 t 中对应 A(t) 与 B(t−3)，再用 guard 排除越界迭代。这个展开由后续 LowerScheduledTIR 完成。

**“把最小窗口编号归一化到 0”是什么意思？** 如果求解器返回 A=220、B=540，II 仍为 100，它们的原始窗口编号就是 2、5。编译器将两者共同的最小编号 2 减掉，得到 0、3。编号从 0 开始，两个任务的窗口差保持为 3。这是 stage 编号的归一化；源码还会另行对 start time 做相对化处理。

**[代码确认]** 循环求解返回任务的 start times、存储版本和循环 II。自动调度且 enable_offset=True 时，C++ 先以 start time / II 的整数商恢复窗口编号，再减去最小编号，得到 stage。手动调度则保留请求的 stage。

关闭 enable_offset 时，自动调度任务的 stage 都设为 0，后续不再按这些时间窗口差展开跨迭代偏移。这不表示两个任务在同一时刻执行；任务顺序和同步仍需保证依赖。

联系本次最小用例：真实结果只有 Load stage=0、Move stage=3 已从 IR 确认，因此能解释为后续执行 Load(t) 与 Move(t−3)。本次普通 TIR 快照没有完整保存 start times 和循环 II，不能将上面的 100、20、340 当作真实值，也不能从 Load 的任务 II=82 推出循环 II。

位置：[auto_schedule.cc / Z3SchedulePythonLoop](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule.cc:1129)，以下为非连续源码摘录：

~~~cpp
int k = (ii > 0) ? static_cast<int>(start_times[i] / ii) : 0;
// 省略 min_k_raw 的统计。
nodes[i]->SetStage(manual_schedule
                       ? static_cast<int>(manual_stages[i])
                       : (enable_offset ? (raw_k[i] - min_k_raw) : 0));
~~~

循环体的排列还考虑任务在重复时间窗口内的位置；源码使用 start time 减去 stage×II 计算排序键。**[代码确认]** 同一个位置出现并列时，更大 stage 的任务排在前面，因此文本顺序不能直接读作“同一逻辑迭代的执行顺序”。

### 4.2 真实 IR：两项任务得到不同的 stage

调度前见第二节的 EstimateLatency 节选：Load、Move 都是 stage=-1。

**[实跑结果]** [04_AutoSchedule.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/04_AutoSchedule.tir:25)，省略循环头：

~~~python
with T.attr({"stage": 3}, "tl.schedule_unit", 1):
    T.attr({"ii": T.int64(35), "latency": T.int64(37)}, "tl.ascend_task", 1)
    T.ascend_copy(T.region(l1[0, 0], 1, 64, 64), T.region(l0[0, 0], 2, 64, 64))
T.attr({"stage": 0}, "tl.schedule_unit", 1)
T.attr({"ii": T.int64(82), "latency": T.int64(272)}, "tl.ascend_task", 1)
T.ascend_copy(T.region(A[i * 64, 0], 1, 64, 64), T.region(l1[0, 0], 2, 64, 64))
~~~

这次结果是 Move stage=3、Load stage=0，Move 在文本中排在前面。此时循环仍然是四次，访问里也还没有物理版本维度。

正确的阅读方式是：后续流水循环的第 t 次，将执行 Move(t−3) 和 Load(t)，并分别保护它们的合法范围。**这里是基于 stage 的解释，不是说上述 IR 已经完成了迭代替换。**

根级的版本 map 还记录 l1=3、l0=1。l1=3 是本例固定输入，不是自由优化得到的三个版本。

### 4.3 Encode 如何把树还原为 TIR？

**[代码确认]** 任务节点复用原语句；控制节点复用原 For 并替换 body；guard 被重新包回去；stage 写入 schedule_unit；缓冲版本等 metadata 写回 kernel 根。

位置：[scheduled_tir.cc / ScheduledTIREncoder::EncodeNode](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule/scheduled_tir.cc:110)，连续源码节选：

~~~cpp
if (node->IsTask())
  return EncodeTask(static_cast<const TaskNode *>(node));

const auto *control = static_cast<const ControlNode *>(node);
For loop = control->control;
loop.CopyOnWrite()->body = EncodeList(control->children);
return EncodeControl(control, std::move(loop));
~~~

写入 stage 的位置见 [MakeScheduleUnitMetadata](D:/codes/yyz/tilelang/src/ascend/transform/auto_schedule/scheduled_tir.cc:82)：

~~~cpp
ScheduleUnitMetadata metadata;
metadata.Set(kScheduleUnitStage, Integer(node->GetStage()));
return metadata;
~~~

这就是“IR 编码/解码抽象”的两个方向：

~~~text
标记化 TIR
  -- Decode --> 任务/控制树 + metadata
  -- 调度 -->   更新节点顺序、stage、版本等
  -- Encode --> 重新组织的标记化 TIR
~~~

**[示意] 上图是表示转换，不是另一个求解器的输入输出协议。** 后续多个 scheduled-TIR pass 也会复用这套内部结构和读写方法；它不是 AutoSchedule 独占的一次性对象。

### 4.4 这里还没有完成哪些事情？

- core_mask 的最终分配、核归属解析在后续阶段完成。
- copy 周围的同步尚未物化。
- l1 的三个版本尚未变成 (3,64,64) 的实际分配与访问。
- stage 3 尚未变成 i−3，循环也尚未扩展为七次。

因此，AutoSchedule 后的 IR 是后续 pass 可以消费的“调度安排”。不能把它当作已可直接照文本顺序执行的最终设备代码。

### 4.5 台上讲述与过渡

> 调度结果先变成 stage、循环体排列和缓冲版本信息。Encoder 用原有语句重建 TIR，保留控制和条件。这里即使 Move 排在 Load 前面，也要结合 stage 理解：它们将属于不同逻辑迭代。下一步，我们看编译器如何生成真正的槽位、同步和边界。

## 5. 从调度 IR 到流水 IR

**建议时长：6 分钟。主线：核归属 → 版本计划 → 同步 → 物理槽位 → 迭代展开。**

### 5.1 先区分版本数量、版本时钟与物理访问

l1=3 只回答“需要几个版本”。还需要回答两件事：本次访问使用哪一版，以及什么时候允许再次覆盖这一版。

AssignCore、ResolveCore 处理任务的执行核归属；PrepareMultiBuffer 确定版本的 owner 和时钟方案；InsertSync 将依赖落实为同步；MaterializeMultiBuffer 改写分配和访问；LowerScheduledTIR 最后展开 stage。

**[实跑结果]** [05_AssignCore.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/05_AssignCore.tir) 中，两次搬运得到 core_mask=2，对应 Cube 侧；这不代表两者使用同一个 pipe，Load 仍是 MTE2 路径，Move 仍是 MTE1 路径。


### core_mask：这项任务在哪一类核侧执行？

这里的字段名是 core_mask。可以直接把它理解成“这项任务属于 Vector 侧、Cube 侧，还是两侧都有相关执行”的标记。

它用二进制的不同位表示核侧。**[代码确认]** 定义见 [core_mask.h](D:/codes/yyz/tilelang/src/ascend/transform/core_mask.h:28)：

~~~cpp
constexpr CoreMask kCoreUnassigned = 0;
constexpr CoreMask kCoreVector = 1U << 0;
constexpr CoreMask kCoreCube = 1U << 1;
constexpr CoreMask kCoreBroadcast = kCoreVector | kCoreCube;
~~~

对应关系：

- 0，即二进制 00：尚未分配。
- 1，即二进制 01：Vector 侧，AIV。
- 2，即二进制 10：Cube 侧，AIC。
- 3，即二进制 11：两位都置位，内部称为 broadcast；具体保留哪些执行流，还要结合后续核解析理解。

因此，本例 core_mask=2 表示两次搬运属于 Cube 侧。这里的 2 不是“第二颗物理核”，也不是“使用两个核”，更不是 MTE2 的编号。

还要把核侧与 pipe 分开：同样属于 Cube 侧，GM→L1 搬运走 MTE2，L1→L0A 搬运走 MTE1。可以理解为“都在同一类执行核侧，但使用其中不同的搬运通路”。


### 5.2 PrepareMultiBuffer：这轮该用哪一个槽位？

已经决定给 L1 三个版本之后，可以把它想象成三个可以轮流使用的格子：

~~~text
L1 的三个槽位：slot 0、slot 1、slot 2
~~~

接下来仍有一个问题：**这次搬运应该使用哪一个格子？** iteration 和 counter 就是两种“算槽位编号”的方法。它们决定怎样轮换三个槽位，不改变已经固定的版本数量。

#### iteration：根据“这是第几轮逻辑迭代”选槽位

本例的循环变量 i 从 0 开始，每次加 1，所以第 i 轮使用 i%3：

~~~text
第 0 轮 → 0%3=0 → slot 0
第 1 轮 → 1%3=1 → slot 1
第 2 轮 → 2%3=2 → slot 2
第 3 轮 → 3%3=0 → slot 0
~~~

这里所说的 owner，可以先理解成“负责组织这个缓冲区一轮使用的循环”。本例就是包着两次 copy 的 for i；不是两次 copy 各有一个 owner。

**iteration 模式不需要额外放一个变量来记轮换到了哪里，编译器根据逻辑迭代次数计算。** 若循环不是从 0 开始、步长不是 1，或需要计算嵌套循环的总次数，就要先求逻辑迭代计数；因此一般形式并不总是原始循环变量直接取模。

流水展开后，每个任务使用自己的逻辑迭代：

~~~text
流水轮次 t 的 Load：处理第 t 块，写 slot(t%3)
流水轮次 t 的 Move：处理第 t−3 块，读 slot((t−3)%3)
~~~

同一块数据在哪一轮加载、在哪一轮消费，都由它自己的逻辑迭代编号选到对应槽位。

#### counter：另外放一个“轮换计数器”

counter 模式直接创建一个变量，记住已经推进了多少轮缓冲使用。每轮相关访问使用 counter%3，随后在编译器安排的位置将 counter 加 1。

**[示意] 以下只说明原版单 stage 用例的槽位轮换，省略真实同步和任务标记：**

~~~text
counter = 0
每一轮：
    slot = counter%3
    将本轮数据加载到 l1[slot]
    从同一个 l1[slot] 搬运到 l0
    counter = counter+1
~~~

注意，是一轮相关访问共用同一个槽位之后再推进，不能理解成“每执行一条 copy 就加 1”。计数器加一也不是硬件完成信号；搬运完成与覆盖安全仍由后续同步协议保证。

**[实跑结果]** [基线 PrepareMultiBuffer](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_baseline/06_PrepareMultiBuffer.tir) 中确实创建了这个变量：

~~~python
l1_version_counter_1 = T.sblock_alloc_buffer((1,), "int32", scope="local.var")
~~~

初始化和更新分别为：

~~~python
l1_version_counter_1[0] = 0
l1_version_counter_1[0] = l1_version_counter_1[0] + 1
~~~

以上是非连续真实 IR 节选；循环注解 tl.multi_buffer_counter_map 将 L1 与该计数器关联。后续物化的访问及同步索引会使用它的值取模。

#### 为什么需要两种方式？

对于本例原版这种每轮都执行、从 0 开始且步长为 1 的简单循环，counter 会跟着 i 一起增长，两种方式都能形成 0、1、2、0 的轮换。

复杂程序中，缓冲区可能只在部分循环迭代里实际使用，或由多个不重叠的 owner 接力使用。这时 counter 可以按实际发生的使用轮次推进，而不完全跟着某一个外层循环变量走。

**[示意] 如果只在外层 i=0、2、4 时使用缓冲区：**

~~~text
按外层逻辑迭代选槽位：0%3、2%3、4%3 → 0、2、1
按实际使用轮次选槽位：0%3、1%3、2%3 → 0、1、2
~~~

这个例子只说明两种计数方式的区别，不表示第一种选择必然错误。真正采用哪种方式，还需要满足程序的 owner、条件和同步要求。

#### 为什么分享主线改用 iteration？

原版两项访问都在 stage 0。只开启 offset 后，Load 在 stage 0，Move 在 stage 3：同一流水轮次内，它们处理不同逻辑迭代。

**[代码确认] 当前实现对 counter 模式有一个限制：同一 owner 内，访问同一个缓冲区的相关任务必须位于一致的 stage。** PrepareMultiBuffer 检查到多 stage 访问会拒绝继续；不能仅凭一个共用 counter 就假定各 stage 的版本索引已经正确对齐。

所以这次分享用例同时进行了两项配置调整：

- 开启 enable_offset=True，让调度结果可以跨 stage 展开。
- 改为 iteration，让各任务按自己的逻辑迭代计算槽位。

这属于当前编译器支持范围的要求，不代表所有 counter 方案在理论上都不能做流水。原版和失败实验保留为备查。

源码入口：[MultiBufferPlanBuilder::Build](D:/codes/yyz/tilelang/src/ascend/transform/prepare_multi_buffer.cc:470)、[CounterStorageStage](D:/codes/yyz/tilelang/src/ascend/transform/prepare_multi_buffer.cc:614)、[VersionIndex](D:/codes/yyz/tilelang/src/ascend/transform/materialize_multi_buffer.cc:221)。


### 5.3 InsertSync：分别保护“数据就绪”和“槽位可复用”

主线中需要两类交接：

1. Load 完成写入后，Move 才能读取同一块数据。
2. Move 完成对槽位的使用后，后续 Load 才能重新写入该槽位。

**[实跑结果]** 在 [08_InsertSync.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/08_InsertSync.tir) 中，可以找到 MTE2_MTE1 和 MTE1_MTE2 两个方向的协议。下面按语义分组展示真实调用；它们在原文件中并不相邻，guard/stage 包裹未抄入本段。

~~~python
# 数据就绪：Load 所在 stage 中通知，Move 所在 stage 中等待
T.ascend_set_flag("MTE2_MTE1", i % 3)
T.ascend_wait_flag("MTE2_MTE1", i % 3)

# 槽位复用：Move 所在 stage 中通知，Load 所在 stage 中等待
T.ascend_set_flag("MTE1_MTE2", i % 3)
T.ascend_wait_flag("MTE1_MTE2", i % 3)
~~~

入口有三个 MTE1_MTE2 初始 set，表示初始槽位可进入使用协议；末尾对应 wait，完成最终收尾。实际 IR 还包含 MTE1 pipe barrier，不能把本段四行当作完整同步实现。

要向听众说明：IR 中 copy 和 set 的先后位置承载的是硬件 pipe 的完成交接协议；调度代价中的“预计数据就绪时间”仍须通过实际同步保障。模型时间不能替代真实的等待关系。

源码入口：[insert_sync.cc / SyncMaterializer](D:/codes/yyz/tilelang/src/ascend/transform/insert_sync.cc:2344)。第一轮只沿这两类交接解释，不展开 flag 分配算法。

### 5.4 MaterializeMultiBuffer：版本计划变成槽位

**[实跑结果]** [09_MaterializeMultiBuffer.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/09_MaterializeMultiBuffer.tir:22) 中，l1 的分配变成：

~~~python
l1_1 = T.sblock_alloc_buffer((3, 64, 64), "bfloat16", data=l1.data, strides=(4096, 64, 1), scope="shared.l1")
~~~

Move 和 Load 的实际访问分别包含：

~~~python
T.ascend_copy(T.region(l1_1[i % 3, 0, 0], 1, 1, 64, 64), T.region(l0[0, 0], 2, 64, 64))
T.ascend_copy(T.region(A[i * 64, 0], 1, 64, 64), T.region(l1_1[i % 3, 0, 0], 2, 1, 64, 64))
~~~

上段为非连续真实 IR 节选。物理版本维度已经出现，但此时两个任务的 i 仍要结合各自 stage 理解，尚未统一替换成流水循环中的不同逻辑迭代。

**[代码确认]** [CreateMultiVersionBuffer](D:/codes/yyz/tilelang/src/ascend/transform/materialize_multi_buffer.cc:99) 在 shape 前插入版本维度；[VersionIndex](D:/codes/yyz/tilelang/src/ascend/transform/materialize_multi_buffer.cc:221) 先取得 iteration 或 counter，再按版本数取模。以下为核心源码节选：

~~~cpp
PrimExpr iteration =
    info.UsesCounter()
        ? PrimExpr(BufferLoad(info.counter, {IntImm(DataType::Int(32), 0)}))
        : CalculateIterationCount(info.owners.front().loop);
PrimExpr version =
    indexmod(iteration, IntImm(iteration.dtype(), info.num_versions));
~~~

物理版本维度的物化与最后的存储 arena 布局/复用仍属于不同问题；本次主线到流水 IR 即闭环，不继续展开分配合并。

### 5.5 LowerScheduledTIR：将 stage 变成逻辑迭代偏移

对于本例，stage 0 的 Load 使用当前 i，stage 3 的 Move 使用 i−3。同步、copy 和相关槽位访问随各自 stage 一起替换，保证等待与实际访问仍对应同一逻辑数据。

**[代码确认]** [PipelineLowerer](D:/codes/yyz/tilelang/src/ascend/transform/lower_scheduled_tir.cc:433) 的核心迭代替换：

~~~cpp
substitution.Set(op->loop_var,
                 op->loop_var -
                     loop_step * (static_cast<int>(stage) - min_stage));
~~~

同一函数将 extent 增加 step×(max_stage−min_stage)，并在替换前构造合法范围 guard。本例 step=1、min_stage=0、max_stage=3，所以四次逻辑迭代需要七次流水循环。一般循环要按源码中的 min、extent 和 step 处理，不能只记住“所有循环加三”。

**[实跑结果]** [10_LowerScheduledTIR.tir](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/minimal_offset/10_LowerScheduledTIR.tir:31) 的连续真实 IR 节选：

~~~python
for i in T.serial(7, annotations={"enable_offset": True}):
    if i - 3 >= 0:
        T.ascend_pipe_barrier("PIPE_MTE1")
    if i - 3 >= 0:
        T.ascend_wait_flag("MTE2_MTE1", (i - 3) % 3)
    if i - 3 >= 0:
        T.ascend_copy(T.region(l1_1[(i - 3) % 3, 0, 0], 1, 1, 64, 64), T.region(l0[0, 0], 2, 64, 64))
    if i - 3 >= 0:
        T.ascend_set_flag("MTE1_MTE2", (i - 3) % 3)
    if i < 4:
        T.ascend_wait_flag("MTE1_MTE2", i % 3)
    if i < 4:
        T.ascend_copy(T.region(A[i * 64, 0], 1, 64, 64), T.region(l1_1[i % 3, 0, 0], 2, 1, 64, 64))
    if i < 4:
        T.ascend_set_flag("MTE2_MTE1", i % 3)
~~~

这个片段可以直接解释四件事：

- Move 只在 i≥3 时执行，读取逻辑迭代 i−3。
- Load 只在 i<4 时执行，保证只读取原来的四块输入。
- 两者的槽位及同步 ID 都跟随逻辑迭代分别取模。
- 扩展的是流水循环次数，程序仍然只处理四块数据。

### 5.6 沿七次循环手工检查一次

**[示意] 从真实 guard 推导出的执行序列，省略所有同步，不是设备时间线：**

~~~text
流水 i=0：Load(0)，slot 0
流水 i=1：Load(1)，slot 1
流水 i=2：Load(2)，slot 2
流水 i=3：Move(0)，slot 0；然后 Load(3)，slot 0
流水 i=4：Move(1)，slot 1
流水 i=5：Move(2)，slot 2
流水 i=6：Move(3)，slot 0
~~~

现在可以解释第四节的“Move 为什么排在 Load 前”：在 i=3，它先消费旧数据，再由同步协议保护同一槽位的复用。源程序的生产—消费语义没有变成“先读未初始化数据”。

这是展示填充、重叠和排空的教学用小循环；只有很短的重叠区间，不适合用来声称长循环稳态性能或实际加速比。

### 5.7 台上讲述与过渡

> 调度写回的是安排，后续 pass 才补齐执行细节。版本时钟说明使用哪个槽位，同步保证数据就绪和安全覆盖，版本物化改写分配与访问，流水展开再把 stage 变成不同逻辑迭代。最终我们看到 i−3、取模索引和边界 guard；它们一起保证原来的四块数据按正确语义通过流水。

最小用例的链路到这里已经完整。接下来只用 FA 验证：任务更多、控制更深、跨核交接更复杂时，仍然使用相同的表示与落地流程。

## 6. FA：主线框架如何扩展到复杂程序？

**建议时长：3 分钟，可选。只补充嵌套控制、跨核交接和循环状态。**

### 6.1 从完整算子中只取一段主线

本次入口：[fa.py](D:/codes/yyz/tilelang/autoshcedule_sharing/fa.py)。参数为 q_len=256、kv_len=512、head_dim=128、num_blocks=1、out_dtype=float32，使用自动调度。

**[代码确认]** 本次 manual_schedule=False，kv_step 绑定到 kv_auto。见 [宏选择位置](D:/codes/yyz/tilelang/autoshcedule_sharing/fa_core.py:334)；[kv_auto 末尾](D:/codes/yyz/tilelang/autoshcedule_sharing/fa_core.py:224) 是连续五次宏调用：

~~~python
qk(kv, K, Q_shared, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub)
softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub)
pack_p(P_nz_ub, P_shared)
pv(kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub)
accumulate_output(O_tmp_ub, O_ub, alpha_ub)
~~~

宏调用方便阅读程序结构，但五个宏不等于五个调度任务：例如 qk 内部含多次搬运与 GEMM；VF 则作为相应任务整体处理。宏会展开，真正任务边界要看任务化后的 IR。

### 6.2 第一项补充：嵌套控制与循环状态

**[实跑结果]** [FA AutoSchedule 后 IR](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/fa/04_AutoSchedule.tir:59) 中，外层 local_mt 执行两次，内层 kv 执行四次。

**[示意] 按程序和 IR 建立的控制树，省略实际任务列表：**

~~~text
ControlNode / local_mt：两个 query tile
├── Q 加载、状态初始化
├── ControlNode / kv：四个 KV block
│   └── QK、softmax、pack、PV、输出累积等展开后的任务
└── 最终归一化与输出
~~~

**[代码确认]** softmax 读取并更新 m_ub、l_ub；accumulate_output 读取旧 O_ub 并写回更新结果。因此不同 kv 之间既有可流水的数据搬运，也有需要维持的状态依赖。调度器不能将每个 KV block 当成完全无关的独立计算。

不在此节推导 online softmax 公式；只圈出状态的读写，说明依赖分析为什么需要访问区域与循环上下文。

### 6.3 第二项补充：Cube 与 Vector 之间的完成交接

QK 在 Cube 侧生成结果，再经 dual copy 写入 S_ub，Vector 的 softmax 消费它。这比最小用例的单核内部 MTE2/MTE1 交接多了一层核间关系。

**[实跑结果]** [FA LowerScheduledTIR](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/fa/10_LowerScheduledTIR.tir:173) 中，Cube 侧在相应 copy 后包含：

~~~python
if kv - 1 < 4 and kv - 1 >= 0:
    T.ascend_cross_core_set_flag(4, "PIPE_FIX", (kv - 1 + local_mt * 4) % 3 + 6)
~~~

Vector 侧在 softmax 任务前包含：

~~~python
if kv < 4:
    T.ascend_cross_core_wait_flag(4, "PIPE_V", (kv + local_mt * 4) % 3 + 6)
~~~

第二段见 [对应消费者位置](D:/codes/yyz/tilelang/autoshcedule_sharing/artifacts/fa/10_LowerScheduledTIR.tir:267)。这两段是分处 CUBE、VECTOR 块的真实节选，不是连续 IR。

重点观察：双方在各自循环中使用本任务的逻辑迭代，事件索引还包含 local_mt*4，协议跨外层 query tile 保持相应的版本进度。不要把 Cube 循环的同一个 kv 数字与 Vector 循环的同一个 kv 数字直接当成同一时刻。完整 IR 还有另一 AIV 侧的事件和槽位释放协议，本段只展示一条数据就绪路径。

flag 编号、首个参数及分配细节不在这三分钟内展开。

### 6.4 第三项补充：核拆分后分别形成流水循环

**[实跑结果]** 同一份 LowerScheduledTIR IR 中，CUBE 块的 kv 循环为八次，VECTOR 块为七次；两者仍然处理每个 query tile 的四个逻辑 KV block。

这说明不能把一个 kernel 的所有任务 stage 简单取最大值，再给所有执行核套相同的循环范围。LowerScheduledTIR 中会先提取相应核的结构，再对各自的 stage 与控制关系进行展开。与最小用例一样，实际工作量由逻辑迭代和 guard 保证。

源码入口：[lower_scheduled_tir.cc / CoreExtractor 与 LowerScheduledTIR](D:/codes/yyz/tilelang/src/ascend/transform/lower_scheduled_tir.cc:194)。这里不再逐行走一遍算法。

FA 还显式固定 K/V、L0C 和若干 UB 缓冲的版本数；引用最终版本 map 时，需要区分用户固定数量与其他缓冲的选择，不能统称“求解器自动决定了所有版本”。

### 6.5 台上讲述与范围控制

> FA 没有换一套模型。宏展开后仍然是任务和控制树，只是多了嵌套循环、跨迭代状态和 Cube/Vector 交接。我们用同样的依赖与资源信息安排任务，再由同步和版本处理保证跨核消费安全。最后，不同执行核各自得到带边界保护的流水循环。

本节只展示五行宏调用、两层控制树和一对跨核 set/wait。完整 FA IR 留作问题查阅，SIMD 指令、性能调优和物理分配复用留在本次主线之外。

## 7. 回顾与交接

**建议时长：1 分钟。先收束自己的完整叙述，再交给同事。**

### 7.1 将七步串成一条完整链路

1. 源程序表达操作、缓冲和控制结构。
2. 任务化、代价估计补出调度所需的边界与信息。
3. Decode 将标记化 TIR 读成任务/控制树，保留原语句及 metadata。
4. 调度器按层处理依赖和资源，形成任务安排。
5. Encode 将顺序、stage 和版本等结果写回 TIR。
6. 核归属、同步和版本物化把安排落实为执行协议与槽位访问。
7. 流水展开将 stage 转为逻辑迭代偏移、guard 和循环范围，后续继续 lowering/codegen。

这里再指出两条边界：模型代价不是设备实测时间；调度安排需要后续同步与访问改写才能落地。

### 7.2 交接讲稿

> 我们已经沿着两次搬运走完了从程序到流水 IR 的全过程：任务与控制结构如何建立，依赖和资源如何影响安排，调度结果又如何变成具体的同步、槽位和迭代偏移。FA 只是把同一套机制扩展到更多任务、嵌套控制和跨核交接。
>
> 接下来聚焦流程中间的求解器环节：C++ 怎样组织输入，哪些信息成为求解变量和约束，求解器怎样得到任务时间、版本数与循环 II，以及结果如何返回。具体协议、数学约束和求解细节由同事展开。

### 7.3 分享结束前自检

- 原始两次 copy 能否沿 IR 找到最终搬运？
- stage、版本数量和循环 II 是否一直分开解释？
- 是否将示意图、源码事实和真实产物明确区分？
- 是否先完成最小用例闭环，再使用 FA 补充？
- 是否明确真实产物使用的 WSL 版本，避免声称已经验证当前 Windows checkout？

---


## 附录 A：用例与运行计划

### 最小用例

来源：[test_schedule_integration.py](D:/codes/yyz/tilelang/testing/ascend/auto_schedule/test_schedule_integration.py) 中的 _make_counter_l1_layout_program。

- 四次循环，每轮 GM → L1、L1 → L0A 两次搬运。
- 原例将 L1 固定为三个 counter 版本，不是求解器自动选出三个版本。
- 它用于展示编译过程，没有计算结果输出，不承担数值正确性演示。
- 保留原例基线；分享变体同时启用 enable_offset=True，并将固定三个版本的索引模式改为 iteration。原例 counter 模式跨 stage 访问会被 PrepareMultiBuffer 拒绝，失败记录已保留。
- **[代码确认]** 当前实现默认关闭 enable_offset，会将自动调度任务的 stage 压到同一层。
- **[实跑结果]** 基线两个主要任务均为 stage 0；分享变体加载 stage 0、搬运 stage 3，循环展开为七次。详见 IR 阅读索引；不是 Windows 当前 checkout 的实跑验证。

### FA 用例

来源：examples/ascend/flash_attention/core.py 中的 flash_attention_fwd，也是 example_mha.py 使用的共享实现。

建议配置：

~~~python
q_len = 256
kv_len = 512
head_dim = 128
num_blocks = 1
out_dtype = "float32"
manual_schedule = False
~~~

该例显式固定了多个缓冲版本，讲述时须说明。优先截取嵌套控制结构、跨核交接和跨迭代状态。

### 采集命令

采集入口：[dump_ir.py](dump_ir.py)；批量脚本：[run_wsl.sh](run_wsl.sh)。源程序：[minimal.py](minimal.py)、[fa.py](fa.py)；完整 FA 实现：[fa_core.py](fa_core.py)。

在 Windows 仓库根目录执行：

~~~powershell
wsl.exe -e bash /mnt/d/codes/yyz/tilelang/autoshcedule_sharing/run_wsl.sh
~~~

脚本启动 /root/tilelang-venv，在 /root/tilelang 中运行，将三个成功用例的产物写入本目录 artifacts。仅 lower 到 Ascend 源码，不启动 NPU。运行版本、输入 SHA256、快照统计与状态见各用例 manifest.json。

单独采集命令和失败实验复现见 [README.md](README.md) 与 [IR-reading-guide.md](IR-reading-guide.md)。完整 pass trace 可通过 dump_ir.py 的 --trace 选项启用。

## 附录 B：IR 清单与观察点

下面各阶段均已采集，产物位于 artifacts。勾选表示采集完成，技术讲述正文仍逐节补充。

- [x] **原始 PrimFunc**：循环、copy、buffer、num_stages、enable_offset 和固定版本注解。
- [x] **MaterializeScheduleUnits 后**：tl.schedule_unit、tl.ascend_task，以及循环与叶任务的边界。
- [x] **EstimateLatency 后**：任务 latency 和 II；说明它们属于模型代价。
- [x] **AutoSchedule 后**：任务顺序、stage、tl.buffer_versions_map。
- [x] **PrepareMultiBuffer 后**：版本时钟、owner 与 counter/iteration 方案。
- [x] **InsertSync 后**：数据就绪、跨迭代覆盖和最终排空对应的同步。
- [x] **MaterializeMultiBuffer 后**：版本维度、槽位索引和 counter 更新。
- [x] **LowerScheduledTIR 后**：逻辑迭代替换、guard、extent 和任务标记消除。
- [x] **可选：最终 kernel.cpp**：确认搬运、同步和版本索引的最终对应。

AssignCore、ResolveCore 快照保留作核对，不单独展开讲述。

任务树、完整读写依赖、start times 和循环 II 不保证全部写入普通 TIR dump。先用源码建立对应并标注证据类型；只有确有必要时再增加只读日志。

### 结果记录模板

每轮运行补充：源码版本、用例及变体、参数与 pass 配置、产物目录、是否成功、关键观察，以及哪些结论仍未确认。

IR 裁剪时保留必要的循环边界、guard 和版本注解；省略内容须说明，不能让节选改变原始语义。

## 附录 C：今天的准备时间表

以下为 11:28 讨论时确定的计划。19:00 开始，扣除午休和晚饭，净准备时间约四小时；按 17:30 前完成安排。

- **11:28–12:00，必须**：确定全景和最小任务树，熟悉源程序与采集脚本；方便时提前采集。
- **12:00–14:00**：午休。
- **14:00–14:40，必须**：理解关键调用链，选取短代码片段。
- **14:40–15:25，必须**：采集和读取最小用例原版、offset 变体的 IR。
- **15:25–16:10，必须**：完成最小用例完整主线材料。
- **16:10–16:40，必须**：第一次完整试讲，记录卡点。
- **16:40–17:05，可选**：补 FA；主线未完成时改为修补主线。
- **17:05–17:30，必须**：定向复讲、修订和离线保存。
- **17:30–19:00**：晚饭和会前准备，不预支这段时间。

检查点：15:25 最小链路 IR 到齐；16:10 主线材料完整；16:40 第一次试讲完成。

延误时，优先删去 FA 实跑扩展和额外日志，保留最小用例闭环与试讲。不将主要时间投入环境配置或 Z3 内部分析。

## 附录 D：内容推进记录

- [x] 明确听众、分工和个人分享时长。
- [x] 建立整体框架、用例建议和 IR 采集清单。
- [x] 建立本 Markdown 主文件。
- [x] 将材料和用例集中到 autoshcedule_sharing。
- [x] 通过 WSL 现有环境采集最小原版、流水变体和 FA 的关键 IR。
- [x] 建立真实 IR 阅读索引，记录 offset/counter 失败实验。
- [x] 完成第 1 节全景材料与讲稿；口头试讲待完成。
- [x] 完成第 2 节任务抽象及真实 IR 对应。
- [x] 完成第 3–4 节调度与结果写回。
- [x] 完成第 5 节流水落地，形成最小用例完整链路。
- [ ] 完成第一次试讲和主线修订。
- [x] 完成第 6 节 FA 补充材料。
- [x] 完成第 7 节回顾与交接话术。
- [ ] 对照材料阅读，记录疑问与需补充内容。
- [ ] 完成最终复讲。

