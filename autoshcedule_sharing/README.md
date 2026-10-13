# AutoSchedule 分享材料

## 当前分享主文档

**[AutoSchedule-talk.md](AutoSchedule-talk.md)** 是当前用于连续讲解的文档。大量保留原版机制说明、真实 IR、关键源码和 stage 数值例子，去掉准备时间表、进度勾选及重复的口述稿，按章节滚动讲解。

第 2 节补充 task/control、with 简写与 guard 的解释；第 5 节用“在哪类核侧执行”和“这轮该用哪个槽位”解释 core_mask、iteration 与 counter。

## 保留的备查与其他形式

- [AutoSchedule-sharing.md](AutoSchedule-sharing.md)：原始准备材料，保留时间表与准备记录；核心问答同步补充。
- [IR-reading-guide.md](IR-reading-guide.md)：原始 IR 阅读索引。
- [AutoSchedule-audience.md](AutoSchedule-audience.md) 与 [HTML 翻页版](AutoSchedule-audience.html)：先前制作的简版，作为备选，不再作为本次分享的主文档。

所有数值示意和裁剪 IR 均有说明；完整原始 IR 留在 artifacts。

- minimal.py：两个搬运任务的最小源程序；原例基线使用 counter，offset 分享变体同时改用 iteration 索引。
- fa.py：小尺寸 FA 入口，使用本目录的 fa_core.py。
- fa_core.py：从 examples/ascend/flash_attention/core.py 原样复制的 FA 实现。
- dump_ir.py：保存前端 IR、十个关键 pass 后的 IR、最终源码及运行信息。
- run_wsl.sh：激活 WSL 虚拟环境，在配套 TileLang checkout 中依次运行三个用例。
- artifacts/：真实运行产物，包括 manifest.json 与日志。
- IR-reading-guide.md：运行后补充的 IR 阅读索引与观察结果。

## 本次采集结果

最小原版、最小流水变体和 FA 均已成功生成前端 IR、十个关键 pass 快照和最终源码。

- [主线 IR 阅读索引](IR-reading-guide.md)
- [最小原版](artifacts/minimal_baseline/manifest.json)
- [最小流水变体](artifacts/minimal_offset/manifest.json)
- [FA](artifacts/fa/manifest.json)

原例 counter 模式只开启 offset 时在 PrepareMultiBuffer 被拒绝，完整失败记录另存于 artifacts/minimal_counter_offset_rejected；该记录不作为成功的主线产物。

## WSL 运行

在 Windows 仓库根目录执行：

~~~powershell
wsl.exe -e bash /mnt/d/codes/yyz/tilelang/autoshcedule_sharing/run_wsl.sh
~~~

脚本默认使用 /root/tilelang-venv 和 /root/tilelang。单独运行：

~~~bash
source /root/tilelang-venv/bin/activate
cd /root/tilelang
python /mnt/d/codes/yyz/tilelang/autoshcedule_sharing/minimal.py
python /mnt/d/codes/yyz/tilelang/autoshcedule_sharing/minimal.py --offset
python /mnt/d/codes/yyz/tilelang/autoshcedule_sharing/fa.py
~~~

直接 lower 到 Ascend 源码，不进行设备编译或 NPU 执行。运行成功仅表示编译链路完成，不表示数值正确性或设备性能已验证。

## 版本边界

创建材料时 Windows checkout 为 fbd7cf81c700，WSL 现有构建 checkout 为 789afc963205。实际 Python 来源、两份源码版本、输入文件 SHA256 和运行状态保存在每份 manifest.json 中。

两份源码的相关差异涉及 NormalizeGemm、AnnotateMultiBufferEligible 与前置流水。这里先复用现成 WSL 构建采集，不声称验证了 Windows 当前 checkout。最小用例无需 GEMM 展开，FA 使用显式 L0 GEMM；这不等于可以忽略所有版本差异。

版本号描述的是 checkout，不能独立证明原生库构建的新鲜度；运行信息须与加载的库及构建记录一起理解。

