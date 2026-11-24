# T.StridedTensor 测试规格

## 1. op 概述

`T.StridedTensor` 用逻辑 `shape` 描述"要访问哪些元素"，用 `strides`（每一维在内存里隔多远，单位是元素个数）描述"这些元素在存储上的实际位置"，用于在一块更大的存储上描述视图；声明本身不复制数据，只影响之后 `T.copy` 和标量/向量访问的地址计算。

术语约定：

- **GM**：全局内存，NPU 外部的大容量存储。
- **UB**：片上统一缓冲区，向量核直接操作的工作内存。
- **L1**：Cube 核的片上输入缓存。
- **L0C**：Cube 矩阵乘的累加结果缓冲区。
- **pitch / gap**：一行有效数据之后、下一行开始之前空出来的元素位置。
- **行内连续**：一行内部的数据在内存里一个挨一个存放，中间没有空洞。

## 2. 接口

### 2.1 T.StridedTensor

```python
T.StridedTensor(shape, strides, dtype="float32", scope=None, *, data=None) -> Buffer
```

| 参数 | 类型 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `shape` | `list` / `tuple`，元素为 `int` 或 `PrimExpr` | 必填 | 逻辑形状；维数可以与物理存储不同，允许动态维 |
| `strides` | `list` / `tuple`，元素为 `int` 或 `PrimExpr` | 必填 | 每一维的元素跨度；长度必须等于 `len(shape)`；可以是 Python 整数（静态），也可以是 `T.dynamic(...)` 产生的标量（动态，由 host 按实际张量行距填入） |
| `dtype` | `str` | 可选 | 元素类型，默认 `"float32"` |
| `scope` | `str` | 可选 | 默认 `"global"`（GM）；在 UB 上声明带行距的别名时传 `"shared"` / `"shared.dyn"` |
| `data` | `Var` | 可选（仅关键字） | 复用已有张量的存储（通常传 `buf.data`），用于在同一块存储上声明多个视图 |

**返回值**：一个张量句柄（Buffer），持有形状、跨度和存储地址；声明不搬数据、不做类型转换、不做物理合法性检查。

### 2.2 T.view（带 strides）

```python
T.view(src, shape=None, dtype=None, strides=None) -> Buffer
```

| 参数 | 类型 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `src` | `Buffer` | 必填 | 源张量 |
| `shape` | `list` / `tuple` | 可选 | 新形状，默认沿用源的形状 |
| `dtype` | `str` | 可选 | 新元素类型，默认沿用源的类型 |
| `strides` | `list` / `tuple` | 可选 | 新视图的每一维元素跨度；不传时按紧凑布局生成，且要求源本身紧凑 |
| 返回值 | `Buffer` | — | 与 `src` 共享同一存储的新视图，不复制数据 |

关系：`T.view` 返回共享同一存储的新视图，可同时改 shape、dtype、strides；不传 `strides` 时得到紧凑视图，传 `strides` 时得到带行距的视图（此时代价是必须自己保证跨度对底层存储合法）。

补充两点（均可在 Python 侧观察到）：

- 改 shape 和 dtype 时，新旧视图的总位数必须相等（例如 `int8 (12, 448)` 可变成 `int32 (12, 112)`），否则直接报错。
- 传 `strides` 时，不允许负值；也不允许源张量的起始偏移非 0。

## 3. 支持的场景

| 场景 | 用哪个接口表达 | 说明 |
| --- | --- | --- |
| 分页 KV Cache 按页取数 | `T.StridedTensor`（跨度可为动态）或 `T.view(..., strides=...)` | 每页有效列少于页宽、页与页之间还有空隙时，逻辑形状只描述有效区域，页间距由 strides 表达，host 传入实际视图即可，无需先做连续化拷贝 |
| 带填充的矩阵输入输出 | `T.StridedTensor` / `T.view(..., strides=...)` | 逻辑形状只写有效列数，strides 写每行实际宽度，读写都只落有效区域，不碰填充区 |
| Cube 结果按行距写回 | `T.StridedTensor` 作为 `T.copy` 的目标 | 输出存储每行留有空隙时，矩阵乘结果按目标行距逐行写回，空隙不被写 |
| 子核各写一个列区间 | `T.StridedTensor(..., data=某个 UB 张量的 data, scope="shared.dyn")` | 同一块片上存储上为不同子核声明不同行距的视图，各子核只写自己的列区间 |
| 同存储按不同位宽解释 | `T.view(src, shape=..., dtype=..., strides=...)` | 同一段字节既按 8 位读、也按 32 位读；总位数守恒，两个视图互相可见 |

## 4. 支持的搬运路径与 stride 用法

| 路径 | 是否支持带 stride | stride 出现在哪一侧 | 形态要求 |
| --- | --- | --- | --- |
| GM → UB | 支持 | 源、目的任一侧或两侧 | 两侧各自合并后最多 2 种跨度（行内一种、行间一种）；行内必须连续；两侧都带 stride 时，行宽与行数必须一致 |
| UB → GM | 支持 | 源、目的任一侧或两侧 | 同上 |
| GM → L1 | 支持 | 源（GM）侧 | 源必须能表示为"一个矩阵"或"一批矩阵"（批量形态下批量维的跨度也由 strides 给出），目的 L1 必须是分形布局；不支持再拆外层循环 |
| L0C → GM | 支持 | 目的（GM）侧（作为行距） | L0C 源尾部 2 维必须从 0 起点、行内连续；目的区域必须落在源 tile 范围内 |
| L0C → UB | 支持 | 目的（UB）侧（作为物理行距） | L0C 源要求同上；若目的 UB 是分形（NZ）布局，则目标视图的跨度必须是稠密的，列起点与列宽按 16 对齐 |
| UB → L1 | 不支持（原始搬运） | — | 只接受合并后单段连续的数据，且字节数是 32 的整数倍；带 stride 的 UB 源需要先写回紧凑 UB |

标量读写与向量化 load/store 也按 stride 计算地址（平坦偏移 = Σ 下标 × 该维 stride），结果与 NumPy 的视图语义一致：同一逻辑下标在两个视图里指向同一块存储位置。

## 5. 约束与限制

1. **strides 必须与 shape 等长**：`len(shape) == len(strides)`，否则声明时立即失败；每一维都要给出跨度，即使该维大小为 1。
2. **stride 的单位是元素且非负**：strides 表示"隔多少个元素"，不是字节；`T.view` 会显式拒绝负值，直接用 `T.StridedTensor` 构造时不检查负值、别名或重叠，合法性由调用方保证。
3. **GM↔UB 合并后最多 2 种跨度**：访问区域按最内层向外的连续维合并后，每侧只能剩"行内宽度 + 行间距"两种跨度；更多独立跨度直接报不支持，需要调用方显式分块。
4. **两侧同时带 stride 时行宽与行数必须一致**：源和目的都非紧凑时，一次二维搬运要求两者的行边界（每行元素数）和行数完全相同，否则报错（不支持自动拆外层循环）。
5. **行内必须连续**：最内层跨度必须为 1；若最内层跨度大于 1，会被当作"每行 1 个元素"的退化形态，功能上仍正确，但每行只搬一个元素、带宽极低。
6. **GM→L1 形态受限**：源必须能归一化为"一个矩阵"（末尾 2 维为行/内维、其余维大小为 1）或"一批矩阵"（≥3 维、最内层连续、第 3 维以外的维大小为 1），且目的必须是分形布局的 L1。
7. **L0C 尾部 2 维需从 0 起点且行内连续**：源 L0C 的取用区域必须在最后两维上从下标 0 开始，且最后一维连续；L0C→GM 还要求目的区域不超出源 tile 范围。
8. **UB→L1 原始搬运不支持 stride**：该路径只接受两侧都连续的一段数据，长度需为 32 字节的整数倍；带行距的 UB 数据要先变成紧凑数据再搬。
9. **亚字节/打包类型需按字节行对齐**：4bit 之类打包类型的每行元素位数必须是 8 的整数倍，否则报错（避免按行向上取整后越界写）。
10. **带填充尾部要求静态行字节且位宽 8/16/32**：使用 `pad_value` / `data_select` 做行尾填充时，行字节数必须是编译期常量；填充值只支持 8/16/32 位的标量类型（含 fp16/bf16/fp32）。
11. **大张量需 64 位 stride 防溢出**：元素跨度可能超出 32 位整型范围时，动态 stride 请用 `T.int64` 声明，否则地址计算可能溢出、结果不可预期。

## 6. 错误与诊断

| 触发条件 | 报错时机 | 报错类型或关键字 | 说明 |
| --- | --- | --- | --- |
| `len(shape) != len(strides)` | Python 调用时（声明张量） | `ValueError`，关键字 `Invalid shape/strides' dimensions` | 不生成张量 |
| `T.view` 改 shape/dtype 后总位数不等 | Python 调用时 | `ValueError`，关键字 `must preserve the logical bit count` | 提示两边的 bit 数 |
| `T.view` 传入负 stride | Python 调用时 | `ValueError`，关键字 `must be non-negative` | 直接用 `T.StridedTensor` 时不检查负值 |
| `T.view` 的 strides 长度与 shape 不符 | Python 调用时 | `ValueError`，关键字 `T.view expected` … `strides for shape` | 提示期望与实际的个数 |
| `T.view` 不传 strides，但源不紧凑 | Python 调用时 | `ValueError`，关键字 `requires explicit strides for non-contiguous buffer` | 需显式给出跨度 |
| `T.view` 的源起始偏移非 0 | Python 调用时 | `ValueError`，关键字 `non-zero elem_offset` | 视图必须从同一地址开始 |
| GM↔UB 合并后超过 2 种跨度 | 编译期 | 关键字 `>2D strided copy not yet supported` | 需调用方显式分块 |
| GM↔UB 两侧都带 stride，行宽不一致 | 编译期 | 关键字 `incompatible 2D row boundaries require outer-loop lowering` | 不支持自动拆外层循环 |
| GM↔UB 两侧都带 stride，行数不一致 | 编译期 | 关键字 `source and destination row counts differ` | 同上 |
| 源与目的区域元素总数不等 | 编译期 | 关键字 `must contain the same number of elements` | — |
| 打包亚字节类型的行不是整字节 | 编译期 | 关键字 `packed sub-byte row must be byte-aligned` | 调整列数或行距 |
| GM→L1 源/目的无法表示为矩阵或批量矩阵 | 编译期 | 关键字 `cannot represent the source and destination as a single matrix or a batch of matrices` | 需改写搬运区域 |
| GM→L1 的 L1 目的不是分形布局 | 编译期 | 关键字 `expects an Ascend fractal layout on the L1 dst` | 目的需带分形布局 |
| L0C 源尾部 2 维不从 0 起点 | 编译期 | 关键字 `requires zero-origin trailing matrix dimensions` | 只能整块取用 |
| L0C→GM 目的区域超出源 tile | 编译期 | 关键字 `destination region must fit inside the source tile` | — |
| L0C→UB 的 NZ 目的视图跨度非稠密 | 编译期 | 关键字 `requires dense logical strides` | 改用扩展 buffer 形状表达填充 |
| L0C→UB 的 NZ 目的列起点/宽度未按 16 对齐 | 编译期 | 关键字 `aligned to 16 elements` | 补齐列起点与列宽 |
| GM→UB 做行尾填充但行字节为符号值 | 编译期 | 关键字 `requires a statically-known row byte length` | 动态行距下不支持填充语义 |
| UB→L1 源或目的带 stride | 编译期 | 关键字 `Strided UB->L1 copies are not yet supported` | 先转成紧凑数据 |
| UB→L1 搬运字节数不是 32 的倍数 | 编译期 | 关键字 `must be a multiple of 32 bytes` | — |
| 填充值类型不受支持 | 编译期 | 编译失败（无固定文案） | 填充值只支持 8/16/32 位标量类型 |
| 静态 stride 与传入张量实际 stride 不符（Cython 执行后端） | 调用 kernel 时 | `ValueError`，关键字 `Static stride mismatch for parameter` | 默认执行后端不做逐维校验，靠调用方传对视图 |

## 7. 使用示例

### 7.1 静态 stride 声明

```python
ROWS, COLS, PITCH = 4, 8, 16

@tilelang.jit(out_idx=None, target="ascend")
def kernel_pair():
    @T.prim_func
    def main(src: T.Tensor((ROWS, COLS), "float32"),
             out: T.StridedTensor((ROWS, COLS), (PITCH, 1), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((ROWS, COLS), "float32")
            T.copy(src, ub)
            T.copy(ub, out)
    return main
```

预期结果：`out` 视图逐元素等于 `src`；每行 `[COLS:PITCH]` 的 gap 保持调用前的值。

### 7.2 动态 stride 声明

```python
ROWS, COLS = 12, 448
s0 = T.dynamic("s0", dtype=T.int64)

@tilelang.jit(out_idx=None, target="ascend")
def kernel_dynamic():
    @T.prim_func
    def main(src: T.Tensor((ROWS, COLS), T.int8),
             out: T.StridedTensor((ROWS, COLS), (s0, 1), T.int8)):
        with T.Kernel(1):
            ub = T.alloc_shared((ROWS, COLS), T.int8)
            T.copy(src, ub)
            T.copy(ub, out)
    return main
```

预期结果：宿主传入 `storage[:, :COLS]` 这类视图，行距取该张量的实际 stride；pitch 为 448/512/520 等不同值时，每次都写到自己那一份存储，gap 不被写。

### 7.3 T.view 同时改 dtype 与 stride

```python
ROWS, COLS_I8 = 12, 448
COLS_I32 = COLS_I8 // 4
stride0 = T.dynamic("stride0", dtype=T.int64)

@tilelang.jit(out_idx=None, target="ascend")
def kernel_view():
    @T.prim_func
    def main(src: T.Tensor((ROWS, COLS_I8), T.int8),
             out: T.StridedTensor((ROWS, COLS_I8), (stride0, 1), T.int8)):
        src_i32 = T.view(src, shape=(ROWS, COLS_I32), dtype=T.int32)
        out_i32 = T.view(out, shape=(ROWS, COLS_I32), dtype=T.int32,
                         strides=(stride0 // 4, 1))
        with T.Kernel(1):
            staging = T.alloc_shared((ROWS, COLS_I32), T.int32)
            T.copy(src_i32, staging)
            T.copy(staging, out_i32)
    return main
```

预期结果：按 int32 视角比较，输出视图逐元素等于源；每行末尾的字节 gap（总 pitch 减 448 字节）保持 0。

### 7.4 带填充输出回写（矩阵乘结果按行距写）

```python
ldd = T.dynamic("ldd", dtype="int64")
m, n, k = 16, 16, 64

@T.prim_func
def gemm_kernel(a: T.Tensor((m, k), "bfloat16"),
                b: T.Tensor((n, k), "bfloat16"),
                out: T.StridedTensor((m, n), (ldd, 1), "float32")):
    with T.Kernel(1):
        a_l1 = T.alloc_l1((m, k), "bfloat16")
        b_l1 = T.alloc_l1((n, k), "bfloat16")
        acc = T.alloc_l0c((m, n), "float32")
        T.copy(a, a_l1)
        T.copy(b, b_l1)
        T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True, unit_flag_ctrl=3)
        T.copy(acc, out[0, 0], unit_flag_ctrl=3)
```

预期结果：`out` 视图等于 `A @ Bᵀ`（按 float32 比较）；宿主传入 `storage[:, :16]`（storage 宽 32）时，每行 `[16:32]` 的 pitch 区域保持原值。
