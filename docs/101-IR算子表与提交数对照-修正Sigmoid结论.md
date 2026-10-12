# 101 · IR 算子表与提交数对照：**修正 docs/98 的 Sigmoid 结论**

> 接 [docs/100](100-verbose-build打印最终IR-布局与融合都在里面.md)（verbose build 打印 IR）。
> 本轮对 9 个真实模型跑 verbose build 并解析 IR 表。

---

## 1. 结果（9 个里解析成功 5 个；4 个的日志格式需修正则）

```
模型               IR算子数  最大ID  算子类型数  Conv*  实测提交  IR/提交
resnet18               49     41       6      28      5     9.80
squeezenet1_0         119     89       8      60     22     5.41
googlenet             879    499       8     453     35    25.11
regnet_x_400mf        156    131       7     122    113     1.38
efficientnet_b0       783    248       8     122     88     8.90
```

**⇒ `IR算子数/提交` 在 1.38 ~ 25.11 之间，单靠算子数**不能**预测提交数。**

---

## 2. ★★ 算子类型分布 —— 这里才有信息

```
regnet_x_400mf:
  [('ConvExSwish',49), ('Conv',39), ('Mul',32), ('ConvAdd',18), ('ConvSigmoid',16), ('Reshape',1)]

efficientnet_b0:
  [('Concat',593), ('Conv',63), ('BatchNormalizationRelu',62), ('ConvRelu',59),
   ('AveragePool',3), ('MaxPool',1)]
```

### ★ 修正一：`ConvSigmoid` / `ConvExSwish` **是存在的融合算子**

**docs/98 我写「`Conv→Sigmoid` = 0% 不融合」** ——
但最终 IR 里出现了 **`ConvSigmoid`（16 个）** 和 **`ConvExSwish`（49 个）**。

**⇒ 那个 `can_fuse` 的 0% 结论需要收窄**：它测的是**原始 ONNX 节点对**，
而**编译器的后续 pass 会做更深的融合**（或 `exSwish` 是特殊算子被单独处理）。

**⇒ 所以 docs/98 的「`Conv→Sigmoid` 不融合解释了 efficientnet 的碎片化」这个因果链是错的。**

### ★★ 修正二：`efficientnet_b0` 的碎片化来自 **593 个 `Concat`**

```
efficientnet_b0 的算子分布：Concat 593 个（占 783 的 76%！）
```

**⇒ 编译器**插入了大量 `Concat`**（该模型原图里只有 62 个 Concat）。
`Concat` 是**布局/内存搬运类算子** ⇒ 每个都可能造成提交边界
⇒ **这才是 efficientnet 88 次提交（`tile/提交` 仅 3.6，最低）的真正原因。**

**⇒ 而 `Concat` 在 docs/98 的 `can_fuse` 表里是 0% 可融合** —— 这一条**方向是对的**。

---

## 3. 修正后的结论

| 项 | docs/98 的说法 | 修正 |
|---|---|---|
| `Conv→Sigmoid` 不融合 | 0%（原始节点对） | ⚠️ **收窄**：最终 IR 里存在 `ConvSigmoid`/`ConvExSwish` ⇒ 后续 pass 会融合 |
| efficientnet 碎片化的原因 | 「`Conv→Sigmoid` 不融合」 | ❌ **改为**：**编译器插入了 593 个 `Concat`** |
| `Concat` 打断融合 | 0% | ✅ **方向正确**（且是 efficientnet 的主因） |
| IR 算子数预测提交数 | — | ❌ 比值 1.38~25.11，**不能单独预测** |

---

## 4. 下一步

| # | 动作 | 说明 |
|---|---|---|
| 1 | **修正则**，把另外 4 个模型也解析出来 | 现在只有 5/9 |
| 2 | **统计「编译器插入的 `Concat` 数量」**与提交数对照 | efficientnet 593、googlenet 呢？ |
| 3 | 看 `Concat` 的来源（原始图 vs 编译器插入） | 区分「模型结构」与「编译器行为」 |
| 4 | 用 IR 表的 `[Start,End)` 与 DRAM 镜像对齐 | 验证内存分配 |

**#2 最直接**：如果「插入的 `Concat` 数」与提交数强相关，**提交数的驱动就找到了** ——
而且这是**编译器行为**，不是模型结构。

---

## 5. 教训（本轮新增）

1. **★★ 我的 `can_fuse` 结论适用范围比我想的窄。** 它测的是**原始 ONNX 节点对**，
   而**最终 IR 里存在 `ConvSigmoid`** ⇒ 编译器的后续 pass 会做更深的融合。
   **⇒ 用「某一层的判定」去解释「最终产物」，必须验证中间没有别的 pass 改变结论。**
2. **★★ 「编译器插入的算子」是一个新维度。** efficientnet 原图 62 个 Concat，
   最终 IR 里 **593 个** —— 多出的 531 个是**编译器插入的**（布局/搬运）。
   **我一直在分析「模型结构」，而碎片化可能主要由「编译器行为」决定。**
3. **★ 分布比总数有信息。** 「783 个算子」没用；
   「其中 593 个是 Concat」直接指向原因。**看分布，不看总数。**
