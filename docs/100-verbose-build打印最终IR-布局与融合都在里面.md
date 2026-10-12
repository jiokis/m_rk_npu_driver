# 100 · ★★★ `rknn.build(verbose=True)` **打印最终 IR**：布局、融合、内存全在里面

> 接 [docs/99](99-tiling规则的proc可调用但需要准备好的IR.md)（tiling proc 需要准备好的 IR）。
> 本轮按计划「在真实流水线里挂钩」——**发现根本不需要挂钩：verbose 模式直接把最终 IR 打出来了。**

---

## 1. ★ 最终 IR 表（probe6，实测输出）

```
ID  OpType          Tensor              DataType  Format    OrigShape       [Start   End)     Size
1   ConvRelu        input               INT8      NC1HWC2   (1,3,32,32)     ...
2   Reshape         r                   INT8      NC1HWC2   (1,16,32,32)    ...
3   Split           f_tp_rs             INT8      NC1HWC2   (1,16384,1,1)   ...
4   Conv            f_tp_rs_split_o1    INT8      NC1HWC2   (1,8192,1,1)    ...
5   ConvAdd         f_tp_rs_split       INT8      NC1HWC2   (1,8192,1,1)    ...
5   ConvAdd         output_mm_tile1_conv
6   Reshape         output_mm           INT8      NC1HWC2   (1,2,1,1)       ...
7   OutputOperator  output              INT8      UNDEFINED (1,2)
```

### ★★ 三个关键信息

**① `Format = NC1HWC2`** —— **这就是内存布局的答案！**

`(1,16,32,32)` → `(1,2,32,32,16)` ⇒ **C1=2、C2=16** ⇒ **`C2 = 16` 就是通道对齐宽度**
—— **与我实测的 `align(in_ch, 16)` 完全一致**（docs/68/70）。
`(1,3,32,32)` → `(1,1,32,32,3)` ⇒ 3 个真实通道，对齐到 C2 里的前 3 个槽。

**② `ConvRelu`** —— **融合后的算子类型**（不是 Conv + Relu 两个节点）
⇒ **直接印证 docs/98 的 `can_fuse` 规则表**。

**③ 常量表给出逻辑形状**

```
ID  User      Tensor   DataType  OrigShape      [Start    End)     Size
1   ConvRelu  W        INT8      (16,3,3,3)     0x0       0x900     0x900 = 2304
1   ConvRelu  B        INT32     (16)           ...
4   Conv      fc#2_tile1_i1  INT8 (2,8192,1,1) 0x4bc0    0x8bc0    0x4000
```

**⇒ `W` 的逻辑形状是 `(16,3,3,3)`（16 输出通道 × 3 输入 × 3×3），大小 `0x900 = 2304`
—— 与寄存器 `WEIGHT_SIZE0 = 2304` 完全一致**（docs/68）。

**⇒ `Gemm` 被转成了 `Conv`**（`fc#2_tile1_i1`，形状 `(2,8192,1,1)`）——
与我在寄存器侧看到的「FC 被实现成 1×1 卷积」一致（docs/63）。

---

## 2. 为什么这是重大进展

| 问题 | 之前 | 现在 |
|---|---|---|
| **内存布局** | 盲探 6 轮、否掉 7 条路径、判定「不可得」（docs/79） | **`NC1HWC2`，C2=16** —— verbose 日志直接给出 |
| **融合规则** | 20 轮探针 → 3 条不完整规则 | `can_fuse` 枚举出 4 条 100% 规则（docs/98）+ **最终 IR 里的 `ConvRelu`** 印证 |
| **算子替换** | 从寄存器反推（GAP→7×7池化、FC→1×1卷积） | **IR 表里直接列出**（`Conv` / `ConvAdd` / `Split` / `Reshape`） |
| **内存分配** | 无 | **每个张量的 `[Start, End)` 与大小** —— 可与抓到的 DRAM 镜像逐一对上 |
| **权重内存总量** | 无 | `Total Weight Memory Size: 37.1875KB`；`Total Internal Memory Size: 67KB` |

**⇒ 一份日志同时回答了布局、融合、算子替换、内存分配四件事。**

---

## 3. 立刻可做（把日志变成结论）

| # | 动作 | 预期 |
|---|---|---|
| 1 | 对 **9 个真实模型**各跑一次 `verbose=True` build，抓 IR 表 | 得到每模型的 **OpType 序列** ⇒ 与实测提交数对照 |
| 2 | 用 IR 表的 **`[Start,End)`** 与抓到的 DRAM 镜像对齐 | **验证内存分配**，并定位每个张量 |
| 3 | 数 IR 表里的 **`tile` 字样**（如 `fc#2_tile1_i1`） | **tile 划分直接写在名字里** |
| 4 | 对比 `ConvRelu` / `ConvAdd` / `Conv` 的**数量**与提交数 | 检验「融合后的算子数是否预测提交数」 |

**#1 最直接**：**IR 表的 OpType 序列就是提交打包的输入** ——
9 个模型的 IR 表与实测提交数一对照，**预测模型就有依据了**。

---

## 4. 教训（本轮新增）

1. **★★★ 「挂钩」之前先试 `verbose=True`。** 我准备去 monkeypatch 编译流水线，
   而**编译器自己就把中间/最终 IR 完整打印出来了**。
   **在改造流程之前，先看它自带的日志能给你什么。**
2. **★★ 一个「格式名」胜过 6 轮盲探。** `NC1HWC2` 这四个字直接给出了布局维度顺序，
   而我从字节里反推了 6 轮、否掉 7 条路径、最后判定「不可得」。
   **黑盒的日志里往往有它的内部表示 —— 那是逆向的最短路径。**
3. **★ 日志里的**名字**也是信息。** `fc#2_tile1_i1` 这个名字同时告诉我
   「Gemm 变成了 Conv」「被切成 tile」「tile 编号是 1」。
   **读名字比读数值更快。**
