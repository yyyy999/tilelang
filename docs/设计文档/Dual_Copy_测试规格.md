# T.dual_copy 测试规格

## 1. op 概述

`T.dual_copy` 用一条语句完成「双目标搬运」：把一个完整的逻辑 tile 沿 M（行）或 N（列）方向对半切开，由两个 AIV 子核（即 Vector 侧的两个并行小核）各处理其中一半；`L0C → UB` 走向量核的硬件双目标搬运，其余路径由两个子核各搬一半。

## 2. 接口

```python
T.dual_copy(
    src,
    dst,
    *,
    unit_flag_ctrl=None,
    l2_cache_ctrl=0,
)
```

| 参数 | 类型 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `src` | Buffer / BufferRegion / BufferLoad | 必填 | 源区域。按路径为 GM、L0C 或 UB；可用整块缓冲，也可用切片区域（如 `a[0:m, 0:n]`）。 |
| `dst` | Buffer / BufferRegion / BufferLoad | 必填 | 目的区域。按路径为 UB、GM 或 L1；同样支持整块缓冲或切片区域。 |
| `unit_flag_ctrl` | int / 整型表达式 / None | 可选 | Cube 单元标志控制（0 = 手工同步，3 = 与前序矩阵乘流水）。不传等价于 0，只影响同步方式，不改变搬运的数据。 |
| `l2_cache_ctrl` | int / str | 可选 | 仅对 `UB → GM` 写回想生效的 L2 缓存策略；默认 0（`normal_fv`）。可用不区分大小写的策略名，如 `"notalloc_clean"`。 |

- **返回值**：返回一条搬运语句的句柄（`tirx.PrimExpr | tirx.Stmt`），与普通 `T.copy` 的返回值性质相同，一般不接收、也不参与数值计算。
- **调用前提**：必须在**混合核**（同时包含 Cube 与 Vector 工作的 kernel，例如 `T.Kernel` 或 `T.MixedKernel(..., sids=2)`）里使用。`T.dual_copy` 不会替用户生成 Cube 侧工作；纯 Vector 核不会因为调用它而变成双 AIV 启动。
- 后端限定：仅 AscendC 后端有效；`l2_cache_ctrl` 在其他后端被忽略。

## 3. 支持的内存路径与切分方式

| 路径 | 是否支持 | 由谁搬运 | 切分方向 | 说明 |
| --- | --- | --- | --- | --- |
| L0C → UB | 支持 | 硬件双目标：一条搬运指令同时写入两个子核各自的 UB | M 或 N | 源是完整 tile、目的是每个子核的半 tile；一条指令完成两个目标。 |
| GM → UB | 支持 | 两个子核各搬一半（软件切分） | M 或 N | 源是完整 GM tile，两个子核各自把对应的一半搬进自己的 UB。 |
| UB → GM | 支持 | 两个子核各搬一半（软件切分） | M 或 N | 源是每个子核的半 tile，两个子核各自写回完整 GM tile 的对应位置。 |
| UB → L1 | 支持 | 两个子核各搬一半（软件切分），并自动完成 ND→NZ 格式转换 | M 或 N | 目的必须是 NZ（分形）布局；用户不需要手工转换格式。 |
| 其他组合（如 GM → L1、L1 → UB 等） | 不支持 | — | — | 编译期直接报错，不生成代码。 |

**切分方向怎么推断**：只看源/目的区域的最后两维 —— 倒数第二维视为 M（行）方向，最后一维视为 N（列）方向。若 M 方向恰好是 2:1（大的一侧 = 2 × 小的一侧）就是 M 切分；若 N 方向恰好是 2:1 就是 N 切分；必须**恰好一个**方向成立。一维区域只有一个维度，它必须满足 2:1，按该方向切分。

**动态形状的回退规则**：当区域大小在编译期无法确定（动态 shape，例如尾部 tile）导致无法从区域形状判定 2:1 时，编译器改为按缓冲区**分配时的形状**来判定切分方向；实际搬运仍然使用运行期的真实区域大小，因此不会把尾部区域放大或缩小。

## 4. 支持的形状与 dtype

可按下列断言逐条验证：

- **秩一致**：源与目的区域要么都是一维，要么都是至少二维；一个一维、一个多维则不合法。
- **恰好一个 2:1 维度**：最后两维中恰好一维满足「大侧 = 2 × 小侧」；两个都满足或都不满足都不合法。
- **静态 extent 必须为偶数**：当完整一侧的 extent 在编译期可知时必须为偶数，且正好等于半侧的两倍；奇数一律报错，不做不等分取整。
- **一维区域**：只有软件路径（GM→UB、UB→GM、UB→L1）允许一维；L0C→UB 一维不合法。
- **L0C → UB 的秩**：区域秩必须 ≥ 2。
- **L0C → UB 的 N 对齐**：N 切分时，完整一侧的 N extent 必须是 32 的倍数。
- **dtype 转换**：`L0C → UB` **不允许**类型转换，源与目的 dtype 必须一致；不一致在前端调用时即报错。
- **GM ↔ UB**：直接按字节搬运，不改变 dtype，源与目的应使用相同 dtype。
- **UB → L1**：区域秩 ≥ 2；源与目的的逻辑行数、列数必须一致；ND→NZ 只对 16/32 位浮点（fp16 / bf16 / fp32）有实现；源区域起始地址需 32 字节对齐；分形填充产生的空洞**不会被初始化**（这些位置的内容不确定，测试不应断言其数值）。
- **L0C → UB 对齐**：传输宽度按 32 字节对齐；编译器在能证明的前提下自动补齐，无法证明时给出编译告警并保留原区域（由调用方自行保证对齐）。

## 5. 约束与限制

1. **必须是混合核**：`T.dual_copy` 只表达 Vector 侧的对半处理，不会合成 Cube 工作。展开：纯 Vector 核中调用它不会获得双 AIV 并行，需同时提供 Cube 工作或使用 `T.MixedKernel(..., sids=2)`。
2. **只支持四条路径**：`L0C→UB`、`GM→UB`、`UB→GM`、`UB→L1`。展开：其他源/目的组合（如 `GM→L1`）在编译期报错，不会静默降级。
3. **一维区域只走软件路径**：一维区域仅允许 `GM→UB`、`UB→GM`、`UB→L1`。展开：一维的 `L0C→UB` 在编译期报错。
4. **L0C→UB 的形状要求**：区域秩 ≥ 2，且 N 切分时完整 N 必须是 32 的倍数。展开：不满足时编译期报错；M 切分无此 N 限制。
5. **静态 extent 必须偶数**：完整一侧的 extent 编译期可知时必须为偶数且恰为半侧的 2 倍。展开：奇数直接报错，不会取整成不等分。
6. **L0C→UB 不做类型转换**：源与目的 dtype 必须一致。展开：dtype 不一致在前端调用 `T.dual_copy` 时抛错，需改用独立 `T.copy` 或在向量计算里自行转换。
7. **manual 模式的作用域要求**：关闭自动调度时必须写在显式 `T.Vector()` 块内，且该子核线程范围（`cthread` extent）必须为 2。展开：写在 `T.Vector()` 之外会编译报错；不在任何子核作用域内也会报错。
8. **manual 模式不支持 `T.MixedKernel`**：手工流水下只接受 `T.Kernel` 内显式 `T.Cube()` + `T.Vector()` 的写法。展开：manual 模式下用 `T.MixedKernel` 承载软件双拷贝会编译报错。
9. **32 字节对齐**：`L0C→UB` 的搬运宽度按 32 字节对齐。展开：能证明时自动补齐；不能证明时编译期给出告警并保留原区域，此时需调用方自行保证对齐，否则可能读到/写越界数据。
10. **UB→L1 的目的布局**：目的必须是 NZ（分形）布局，且源/目的逻辑行列一致。展开：目的不是 NZ 布局或行列不一致时编译期报错；通常由布局推断自动给出，也可显式指定 NZ 布局。
11. **UB→L1 的源地址对齐**：源区域起始地址需 32 字节对齐。展开：不满足时编译期报错，可通过补齐行首或调整分配避免。
12. **UB→L1 的填充空洞**：ND→NZ 转换产生的填充位置不初始化。展开：不要对这些位置做数值断言，只校验有效逻辑区域。

## 6. 错误与诊断

| 触发条件 | 报错时机 | 报错类型或关键字 | 说明 |
| --- | --- | --- | --- |
| 源与目的秩不一致（一个一维、一个多维） | 前端调用 | `ValueError`，关键字 `both one-dimensional or both have rank at least two` | 统一成同秩区域。 |
| 找不到唯一的 2:1 方向（两个尾维都 2:1，或都不满足） | 前端调用 | `ValueError`，关键字 `Cannot infer dual_copy split direction` | 只让一个尾维呈 2:1。 |
| L0C 源与目的 dtype 不一致 | 前端调用 | `ValueError`，关键字 `Type conversion during cc->ub dual-destination copy is NOT supported on dav-950` | 改用独立 `T.copy` 或向量内转换。 |
| `l2_cache_ctrl` 传入未知的策略名 | 前端调用 | `ValueError`，关键字 `Unknown l2_cache_ctrl string` | 使用文档列出的策略名或直接传整数。 |
| 不支持的源/目的组合 | 编译期 | 编译失败，关键字 `supports only L0C->UB` | 例如 `GM→L1`。 |
| 切分方向上比例不是精确 2:1（含奇数完整 extent） | 编译期 | 编译失败，关键字 `exact 2:1 extent ratio` | 仅当 extent 在编译期可知时检查。 |
| L0C→UB 的 N 切分且完整 N 不是 32 的倍数 | 编译期 | 编译失败，关键字 `multiple of 32` | 补齐 N 或改用 M 切分。 |
| L0C→UB 使用一维区域 | 编译期 | 编译失败，关键字 `at least two-dimensional` | 硬件路径要求秩 ≥ 2。 |
| manual 模式下软件双拷贝不在 `T.Vector()` 内（或用了 `T.MixedKernel`） | 编译期 | 编译失败，关键字 `explicit T.Vector` | 消息中同时说明 manual 模式不支持混合核。 |
| 软件双拷贝所在的子核作用域 extent 不是 2 | 编译期 | 编译失败，关键字 `cthread extent of 2` | 使用 `T.Vector()`（范围固定为 2）。 |
| 软件双拷贝不在任何子核作用域内，且无法自动建立 | 编译期 | 编译失败，关键字 `outside a cthread scope` | 自动调度模式下一般会自动建立，手工模式需显式作用域。 |
| UB→L1 目的不是 NZ 布局 | 编译期 | 编译失败，关键字 `MajorK destination layout` | 使用布局推断给出的 NZ 布局或显式 NZ 布局。 |
| UB→L1 区域秩 < 2 | 编译期 | 编译失败，关键字 `rank >= 2 regions` | 至少提供二维区域。 |
| UB→L1 源与目的逻辑行列不一致 | 编译期 | 编译失败，关键字 `logical regions must match` | 半 tile 与完整 tile 的非切分维必须一致。 |
| UB→L1 源地址未 32 字节对齐 | 编译期 | 编译失败，关键字 `32-byte-aligned source address` | 调整分配或行首偏移。 |
| UB→L1 的 dtype 不是 16/32 位浮点 | 编译期 | 编译失败，关键字 `ND2NZ dtype must be half, bfloat16_t, or float` | 仅 fp16 / bf16 / fp32 支持自动 ND→NZ。 |
| L0C→UB 无法证明 32 字节对齐 | 编译期 | **告警**（不中断），关键字 `Ascend L0C->UB copy` 与 `32-byte boundary` | 保留原区域，由调用方保证对齐。 |

## 7. 使用示例

**示例 1：L0C → UB（M 切分，硬件双目标）**

```python
BR, BC = 128, 128
ROWS = BR // 2

with T.Kernel(NUM_BLOCKS) as bx:
    qk_acc_l0c = T.alloc_l0c((BR, BC), "float32")     # 完整累加结果
    S_ub = T.alloc_shared((ROWS, BC), "float32")      # 每个子核的半 tile
    # ... Cube 侧工作：把 BR×BC 写满 qk_acc_l0c ...
    T.dual_copy(qk_acc_l0c, S_ub[0:ROWS, 0:BC])
```

预期结果：两个子核各拿到 `BR/2 × BC` 的一半，两个子核的 `S_ub` 拼起来逐元素等于 `qk_acc_l0c` 的完整内容。

**示例 2：UB → L1（半 tile 源 → 完整 NZ tile，自动 ND→NZ）**

```python
P_nz_ub = T.alloc_shared((ROWS, BC), "bfloat16")      # 每个子核的半 tile
P_shared = T.alloc_l1((BR, BC), "bfloat16")           # 完整 tile，目的为 NZ 布局
# ...
T.dual_copy(P_nz_ub[0:ROWS, 0:BC], P_shared[0:BR, 0:BC])
```

预期结果：两个子核各写完整 L1 tile 的一半，`P_shared` 得到完整的 `BR × BC` NZ 数据；再次用它做矩阵乘的结果与用普通完整拷贝得到的结果一致。

**示例 3：UB → GM（半 tile 源 → 完整 GM 区域）**

```python
O_ub = T.alloc_shared((ROWS, D), "float32")
# ...
T.dual_copy(O_ub[0:ROWS, 0:D], O[mt * BR : (mt + 1) * BR, 0:D])
```

预期结果：`O` 的第 `mt*BR` 到 `(mt+1)*BR` 行被两个子核各写一半完整覆盖，其他行不受影响。
