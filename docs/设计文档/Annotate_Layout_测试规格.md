# T.annotate_layout 测试规格

## 1. op 概述

`T.annotate_layout` 让用户在 kernel 里为某个 Buffer（一个"视图"）指定它的物理布局，也就是"这块片上内存里的元素按什么顺序摆放"。它只在编译期改变编译器对这个 Buffer 的寻址解释，不搬运数据、不转置数据、也不改变任何数值。调用后编译出来的程序如果用同一套解释去读写，数值结果应与不加注解时完全一致。

## 2. 接口

**签名**

```python
T.annotate_layout(layout_map: dict)
```

**参数**

| 参数 | 类型 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `layout_map` | `dict` | 是 | 键是 Buffer 对象（不是 `buffer.data`），值是该 Buffer 要使用的布局 |

**返回值**

返回一条编译期注解语句，在 `@T.prim_func` 的函数体里当作普通语句调用即可，不需要接收返回值，也没有运行期返回值。

**作用对象（键）**

- 在 Ascend 后端（`from tilelang.ascend import language as T`，编译时 `target="ascend"`）下，键必须是 **Buffer 对象本身**：`T.alloc_shared` / `T.alloc_l1` / `T.alloc_l0a` / `T.alloc_l0b` / `T.alloc_l0c` 的返回值，或由 `T.view` / `T.reshape` 派生出的新 Buffer，或 `T.alloc_fragment` / `T.alloc_reducer` 的返回值。
- 不能传 `buffer.data`（那是另一个对象，在 Python 调用阶段就会报类型错误）。
- 注解按"视图"生效：同一块 storage（一块实际分配的片上内存）通过不同视图访问时，每个视图需要单独注解。

**取值（值）**

| 取值类型 | 说明 |
| --- | --- |
| `Layout` | 显式给出"逻辑下标 → 物理位置"的映射，通常由仓库提供的分形布局构造器生成 |
| `Callable`（函数/lambda） | 会被自动包装成 `Layout(buffer.shape, 函数)`；函数参数个数必须等于 buffer 的维数 |
| `Fragment` | 仅用于 Fragment 类 Buffer（`T.alloc_fragment`），描述哪个线程拥有哪个元素 |
| `PartialFragment` | 仅用于 reducer 类 Buffer（`T.alloc_reducer`，归约累加器），描述每线程的部分和 |

**生效阶段**

编译期。编译器在生成代码时据此决定元素的物理地址，不会生成任何额外的运行期指令。

## 3. 支持的布局取值与构造器

**取值写法**

| 取值类型 | 写法示例 | 说明 |
| --- | --- | --- |
| `Layout` | `Layout((32, 32), lambda i, j: (j, i))` | 第一个参数是逻辑形状（必须与 Buffer 形状一致），第二个参数是下标映射函数 |
| Callable | `lambda i, j: (j, i)` | 等价于上一条，自动按 `buffer.shape` 包装 |
| 分形构造器返回值 | `make_ascend_major_k_layout(a_l1)` | 也是 `Layout`，但带有硬件分形语义，推荐用这种方式 |
| `Fragment` | `Fragment((16, 16), forward_thread_fn=..., forward_index_fn=...)` | 只能注解 `T.alloc_fragment` 出来的 Buffer |
| `PartialFragment` | `PartialFragment((32,), forward_thread_fn=..., replicate=8)` | 只能注解 `T.alloc_reducer` 出来的 Buffer |

**仓库提供的布局构造器**（均可从 `tilelang.layout` 导入）

| 名称 | 作用 | 典型使用场景 |
| --- | --- | --- |
| `make_ascend_nz_layout(buffer)` | 生成标准的 NZ 分形布局 | Cube 输入（L1/L0）按 16×C0 分形块摆放 |
| `make_ascend_major_k_layout(buffer, k_align=1)` | 生成 K 轴为主序的分形布局（与 NZ 同源） | L1/L0 上的矩阵操作数 |
| `make_ascend_major_mn_layout(buffer, k_align=1)` | 生成 M/N 轴为主序的分形布局 | L1/L0 上转置后的操作数 |
| `make_ascend_compact_nz_layout(buffer)` | 生成紧凑 NZ 布局，不把行数补齐到 16 的倍数 | UB 上已按 NZ 顺序准备好的数据，用于降低占用 |
| `make_ascend_l0c_layout(buffer)` | 生成 L0C 累加器专用的 16×16 分形布局，也可用于 UB | L0C 输出；L0C→UB 搬运时保留 NZ |
| `make_ascend_sf_layout(buffer)` | 生成 scale-factor 布局 | MX 量化的 scale 张量 |

补充说明：

- 构造器会自动按元素宽度推算分形宽度 `C0 = 32 字节 / 元素字节数`。例如 fp16/bf16 为 16，int8/fp8 为 32，fp32 为 8。
- 构造器接收 Buffer 或 Buffer 的切片区域，并读取其形状；通常就是用被注解的那个 Buffer 去构造。
- `T.alloc_l0c` 默认已经带上了 L0C 布局，一般不需要再手动注解。
- `make_ascend_sf_layout` 只支持位宽能整除 16 的元素类型（4/8/16 位），32 位及以上不支持。

## 4. 约束与限制

- **键必须是 Buffer，且在同一个函数内声明。** 传 `buffer.data` 或其他非 Buffer 对象不会被接受，会在编译期失败；传其他函数里的 Buffer 不属于合法输入。
- **同一 storage 的多个视图必须分别注解。** 布局只挂在被注解的那个视图上，不会自动传给共享同一块内存的其他视图；需要特殊解释的每个视图都要单独写一次。
- **注解不会替你搬动或转置数据。** 它只改变"读写这块内存时怎么算地址"；如果生产者按一种顺序写、消费者按另一种解释读，得到的是被重新解释的值，不是自动转置后的正确结果。
- **共享 storage 的 Cube 视图必须 dtype 相同、补齐后物理尺寸相同。** 同一块存储被多个视图以分形方式使用时，元素类型和补齐后的总大小必须一致，否则编译期失败。
- **Fragment 别名必须线程归属一致。** 同一块 Fragment 存储的多个视图，如果都注解了 `Fragment`，它们的线程映射必须相同，否则编译期报错。
- **`T.view` / `T.reshape` 派生出的视图不继承布局。** 派生出来的是新的 Buffer，如果它需要非默认解释，必须再单独注解一次。
- **`Layout` 的输入形状必须等于被注解 Buffer 的逻辑形状。** 形状不一致属于编译期错误，不会自动裁剪或补齐。
- **分形布局会带来补齐，搬运时只应覆盖有效区域。** 例如 CompactNZ 的补齐行不初始化，`T.copy` 只应覆盖有效行；L0C 布局用在 UB 上时，拷贝区域要覆盖完整的 16 列面板。
- **注解不会被自动放宽。** 如果注解与算子本身要求的布局冲突，编译直接报错，而不会悄悄改用其中一种。

## 5. 错误与诊断

| 触发条件 | 报错时机 | 报错类型或关键字 | 说明 |
| --- | --- | --- | --- |
| 注解值不是 `Layout` / Callable，且不是 Fragment/reducer Buffer | Python 调用时 | `ValueError`，关键字 `Invalid layout` | 例如传 `123`、`"nz"`、`None` |
| reducer Buffer 注解了非 `PartialFragment` | Python 调用时 | `AssertionError`，关键字 `must be a PartialFragment` | 提示必须描述每线程部分和 |
| Fragment Buffer 注解了非 `Fragment` | Python 调用时 | `AssertionError`，关键字 `layout must be a Fragment` | 必须用 `Fragment` 及其子类 |
| 键传了 `buffer.data`（或任何非 Buffer 对象） | 编译推断时 | 编译失败（键类型不匹配），无固定文案 | 必须传 Buffer 对象本身；测试只断言编译失败，不要匹配具体关键字 |
| Callable 的参数个数与 Buffer 维数不一致 | Python 调用时 | `TypeError` | Python 的普通参数个数错误 |
| 构造器收到 rank < 2 的 Buffer | Python 调用时 | `InternalError`，关键字 `rank >= 2` | 分形布局只作用于有行列两维的对象 |
| CompactNZ 的最后一维不是静态值或不能被 C0 整除 | Python 调用时 | `ValueError`，关键字 `static column extent` / `columns divisible by C0` | 需要静态可算的列宽 |
| SF 布局用于 32 位及以上元素 | Python 调用时 | `InternalError`，关键字 `bit width divides 16` | SF 只支持 4/8/16 位存储 |
| 构造器传了 `k_align > 1`，而对应轴静态不可整除 | Python 调用时 | `InternalError`，关键字 `requires a static extent to verify divisibility` / `requires K-axis extent to be divisible by` | 需把该轴补齐到对齐倍数 |
| 布局输入形状与 Buffer 逻辑形状不一致 | 编译推断时 | `InternalError`，关键字 `Layout input shape must match buffer view` | 消息中会带上两个形状 |
| 共享 buffer 的布局前向映射非单射（两个逻辑元素落到同一物理位置） | 编译推断时 | `ValueError`，关键字 `must be injective` | 消息提示检查传给 `T.annotate_layout` 的布局 |
| reducer 的 `PartialFragment` 非单射 | 编译推断时 | `ValueError`，关键字 `must be injective over (thread, index)` | 线程与索引的组合必须一一对应 |
| reducer 注解与更新循环实际推导出的部分和计划不一致 | 编译推断时 | `InternalError`，关键字 `different from the annotated one` | 注解不会被自动放宽 |
| 同一 Fragment 存储的多个视图线程归属不一致 | 编译推断时 | `InternalError`，关键字 `Conflicting fragment layouts for aliases` | 消息会带上两个视图名 |
| 用户注解与算子推导出的布局冲突 | 编译推断时 | `InternalError`，关键字 `Get different layout for` | 消息会打印两种布局 |
| 不同位置对同一 Buffer 给出了不同布局 | 编译推断时 | `InternalError`，关键字 `Conflicting layouts for` | 消息带 Buffer 名 |
| 共享 storage 的视图在补齐后物理尺寸不同 | 编译推断时 | `InternalError`，关键字 `equal physical sizes including padding` | 消息带存储名 |
| 共享 storage 的视图元素类型不同 | 编译推断时 | `InternalError`，关键字 `canonical operand dtype` | Cube 视图必须同 dtype |
| 需要非默认布局的 Cube 视图自己没被注解 | 编译推断时 | `InternalError`，关键字 `needs its own layout` | 提示显式注解该视图 |
| 标注后仍无法为某个 Buffer 定出布局 | 编译推断时 | `InternalError`，关键字 `no available layout found` | 常见于注解与算子要求互相矛盾 |

> 说明："Python 调用时"指写 `@T.prim_func` 的函数体被解析、`T.annotate_layout` 或布局构造器被真正执行的时刻；"编译推断时"指随后调用编译流程（如 `tilelang.compile`）时的检查阶段。`InternalError` 即 TVM 的编译期内部错误，测试里用 `pytest.raises(tvm.error.InternalError, match="关键字")` 断言；`ValueError` / `TypeError` / `AssertionError` 为 Python 异常。

## 6. 使用示例

**示例 1：普通 Buffer 注解（L1 输入）**

```python
from tilelang.ascend import language as T
from tilelang.layout import make_ascend_major_k_layout

with T.Kernel(grid) as bx:
    a_l1 = T.alloc_l1((block_M, block_K), "float16")
    b_l1 = T.alloc_l1((block_N, block_K), "float16")
    c_l0c = T.alloc_l0c((block_M, block_N), "float32")

    T.annotate_layout({
        a_l1: make_ascend_major_k_layout(a_l1),
        b_l1: make_ascend_major_k_layout(b_l1),
    })
    # 后续 T.copy / T.gemm 正常使用
```

预期现象：编译通过；`c_l0c` 不需要注解（`alloc_l0c` 已自带 L0C 布局）；计算结果与不加这两条注解时一致。

**示例 2：两个 view 分别注解**

```python
from tilelang.layout import Layout

with T.Kernel(1):
    storage = T.alloc_shared((8, 16), "float32")
    view = T.view(storage, (16, 8))
    T.annotate_layout({
        storage: Layout((8, 16), lambda i, j: (i, j)),
        view: Layout((16, 8), lambda i, j: (j, i)),
    })
    with T.SimtVF(threads=128):
        for i, j in T.Parallel(8, 16):
            storage[i, j] = A[i, j]
        for i, j in T.Parallel(16, 8):
            B[i, j] = view[i, j]
```

预期现象：编译通过；输出 `B` 等于 `A` 的转置。若把两个注解都去掉，`B` 等于 `A` 按行主序重排成 16×8 后的数据（而不是转置）；只保留其中一个注解时，结果仍是 `A` 的转置，因为被注解的视图参与了读写。

**示例 3：Callable 写法**

```python
T.annotate_layout({buf: lambda i, j: (j, i)})
```

预期现象：编译通过，等价于 `T.annotate_layout({buf: Layout(buf.shape, lambda i, j: (j, i))})`，数值结果完全一致。

**示例 4：Fragment 与 PartialFragment**

```python
from tilelang.layout import Fragment, PartialFragment

with T.Kernel(1):
    with T.SimtVF(threads=32):
        frag = T.alloc_fragment((16, 16), "float32")
        T.annotate_layout({
            frag: Fragment(
                (16, 16),
                forward_thread_fn=lambda i, j: (j * 16 + i) % 32,
                forward_index_fn=lambda i, j: (j * 16 + i) // 32,
            )
        })

with T.Kernel(1) as _, T.SimtVF(threads=128):
    acc = T.alloc_reducer((128,), "float32", op="sum")
    T.annotate_layout({
        acc: PartialFragment((128,), forward_thread_fn=lambda i, rep: rep, replicate=128),
    })
```

预期现象：Fragment 示例编译通过，每个元素由映射函数指定的线程处理；`PartialFragment` 示例要求该映射与 reducer 更新循环实际产生的部分和一致，不一致时编译报错（关键字 `different from the annotated one`）。若给 Fragment Buffer 传普通 `Layout`，调用 `T.annotate_layout` 时立即断言失败。
