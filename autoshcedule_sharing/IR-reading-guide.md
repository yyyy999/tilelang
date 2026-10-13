# IR 阅读索引与本次运行结果

## 从哪里开始

主线使用 minimal_offset。minimal_baseline 用来对照“没有跨 stage 展开”的情况。先读最小用例，再读 FA，不按文件大小或 pass 数量平均分配时间。

本次真实运行使用 WSL /root/tilelang-venv，在 /root/tilelang 中调用编译器，输出写回当前分享目录。

- 实际编译器 checkout：789afc9632059c7404ffd10e188c6a073be40e15。
- 分享文件所在 checkout：fbd7cf81c700dc3f5b2d54fa9ef1b5d7cc8c7907。
- Python：/root/tilelang-venv/bin/python。
- TileLang Python：/root/tilelang/tilelang/__init__.py。
- 加载的开发构建目录：/root/tilelang/build。
- Z3：4.15.4。
- 仅执行 lower 到 Ascend 源码，不执行设备编译、NPU 运行或性能测试。

每个 artifacts 子目录的 manifest.json 记录运行时间、配置、输入文件 SHA256、各阶段统计和运行状态。源码 checkout 版本不单独证明构建新鲜度，也不代表 Windows 当前 checkout 已经实跑验证。

## 已有产物

- artifacts/minimal_baseline：成功，前端 IR + 十个 pass 快照 + 27 行最终源码。
- artifacts/minimal_offset：成功，前端 IR + 十个 pass 快照 + 28 行最终源码。
- artifacts/fa：成功，前端 IR + 十个 pass 快照 + 280 行最终源码。
- artifacts/minimal_counter_offset_rejected：保留的失败实验，到 AssignCore 后为止；不是分享主线。

默认 TIR 打印省略 layout metadata 的完整定义，文件尾部会明确提示。这些快照用于阅读编译变化，不宣称是能独立重新解析的完整序列化 IR。

## 最小用例的阅读顺序

1. [源程序](minimal.py)：四次循环、GM→L1→L0A；基线固定三个 counter 版本，流水变体固定三个 iteration 版本。
2. [00_frontend.tir](artifacts/minimal_offset/00_frontend.tir)：前端循环注解、固定版本和两次 copy。
3. [01_MaterializeScheduleUnits.tir](artifacts/minimal_offset/01_MaterializeScheduleUnits.tir)：寻找 tl.schedule_unit、tl.ascend_task；此处尚未得到自动 stage。
4. [03_EstimateLatency.tir](artifacts/minimal_offset/03_EstimateLatency.tir)：两项主要任务的模型代价；加载 latency=272、ii=82，L1→L0A 搬运 latency=37、ii=35。它们不是设备实测时间。
5. [04_AutoSchedule.tir](artifacts/minimal_offset/04_AutoSchedule.tir)：加载 stage=0、搬运 stage=3；L1 三个版本、L0A 一个版本。文本顺序中搬运在前，不能仅据此认为消费发生在生产之前。
6. [06_PrepareMultiBuffer.tir](artifacts/minimal_offset/06_PrepareMultiBuffer.tir)：观察版本计划及 indexing mode 的消费；此变体使用 iteration，没有基线的共享 counter 时钟。
7. [08_InsertSync.tir](artifacts/minimal_offset/08_InsertSync.tir)：寻找 MTE2_MTE1、MTE1_MTE2 的 set/wait，以及入口初始化和末尾等待。
8. [09_MaterializeMultiBuffer.tir](artifacts/minimal_offset/09_MaterializeMultiBuffer.tir)：L1 的物理版本形状成为 (3,64,64)，访问使用 i%3。
9. [10_LowerScheduledTIR.tir](artifacts/minimal_offset/10_LowerScheduledTIR.tir)：循环扩展为 7 次，消费使用 i-3，对应槽位 (i-3)%3；加载受 i<4 保护，消费受 i-3>=0 保护。
10. [kernel.cpp](artifacts/minimal_offset/kernel.cpp)：最终搬运、同步和版本寻址。源码短，适合收尾核对。

第 4、5、8、9 项是分享中最值得展示的四次变化，其余快照作为回答问题和核对机制的依据。

[实跑结果] num_stages=3 的例子出现了 task stage=3，且缓冲版本数也是 3。这里三个数属于不同概念，不能由相同数值推出含义相同。

## 基线与流水变体的真实区别

- 基线：enable_offset=False，version_mode=counter；两个主要任务都是 stage 0。见 [04_AutoSchedule.tir](artifacts/minimal_baseline/04_AutoSchedule.tir)。
- 分享变体：enable_offset=True，version_mode=iteration；两个主要任务分别为 stage 0、3。
- 两者都固定 L1 三个版本；不能声称这些版本数量全部由求解器自由选择。
- 分享变体相对原例有两项配置变化，不再描述为“仅打开 enable_offset”。

## 失败实验：为何不能只打开 offset？

最初保留原例 counter 模式并开启 enable_offset。AutoSchedule 得到不同 stage 后，PrepareMultiBuffer 拒绝继续：

> Counter multi-buffer storage l1 is accessed at multiple schedule stages in one owner loop.

完整记录位于 [失败 manifest](artifacts/minimal_counter_offset_rejected/manifest.json) 和 [error.txt](artifacts/minimal_counter_offset_rejected/error.txt)。

这说明原先的分享方案必须调整。它不是本轮要修的编译器问题，也不应为绕过检查而改动编译器。已改用一个 owner 下的 iteration 版本索引，成功完成整条链路。

需要复现这次失败时执行：

~~~bash
source /root/tilelang-venv/bin/activate
cd /root/tilelang
python /mnt/d/codes/yyz/tilelang/autoshcedule_sharing/minimal.py \
  --offset --version-mode counter \
  --out /mnt/d/codes/yyz/tilelang/autoshcedule_sharing/artifacts/counter_offset_reproduction
~~~

## FA 的阅读范围

先读 [fa.py](fa.py) 的参数，再读 [fa_core.py](fa_core.py) 的 kv_step 宏调用链：

QK → softmax → pack_p → PV → accumulate_output。

- [00_frontend.tir](artifacts/fa/00_frontend.tir)：宏展开后的源程序结构。
- [03_EstimateLatency.tir](artifacts/fa/03_EstimateLatency.tir)：Cube、搬运和 VF 的任务边界与代价。
- [04_AutoSchedule.tir](artifacts/fa/04_AutoSchedule.tir)：外层 local_mt 两次、内层 kv 四次；不同 stage 的主要任务与跨迭代状态。
- [08_InsertSync.tir](artifacts/fa/08_InsertSync.tir)：截取一个跨核交接，保留必要的生产者和消费者。
- [09_MaterializeMultiBuffer.tir](artifacts/fa/09_MaterializeMultiBuffer.tir)：多类缓冲的版本形状与槽位访问。
- [10_LowerScheduledTIR.tir](artifacts/fa/10_LowerScheduledTIR.tir)：查看 CUBE/VECTOR 各自的流水循环、guard 和同步。
- [kernel.cpp](artifacts/fa/kernel.cpp)：最终混合核代码。

FA 不展示整份 IR，也不展开 softmax SIMD 指令。该例显式固定多个缓冲版本，不能将所有版本数归因于求解器自动选择。

## 尚未从这些产物确认的内容

- 完整依赖图、start times 和 loop II 并未全部序列化到普通 TIR dump；不能从文件缺失字段推断调度器没有这些信息。
- 没有测量设备时延或验证 NPU 数值结果。
- 没有用 Windows 当前 checkout 的新构建复跑。

