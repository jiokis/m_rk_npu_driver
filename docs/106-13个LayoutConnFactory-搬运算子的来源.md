# 106 · ★★★ 找到 **13 个 `*LayoutConnFactory`**：搬运算子（593 个 Concat）的来源

> 接 [docs/105](105-编译器本体librknnc未strip-tiling到lowering的符号链.md)（编译器本体未 strip）。
> 本轮枚举 `librknnc.so` 的 RTTI，**找到搬运算子的产生机制**。

---

## 1. ★★ 13 个 LayoutConnFactory（每个算子类型一个）

```
OpLayoutConnFactory          LNLayoutConnFactory         RNNLayoutConnFactory
ConvLayoutConnFactory        FullLayoutConnFactory       ConcatLayoutConnFactory    ★
MatMulLayoutConnFactory      ReshapeLayoutConnFactory    SoftmaxLayoutConnFactory
BroadcastLayoutConnFactory   ShapeHoldLayoutConnFactory  TransposeLayoutConnFactory
NativeOnlyLayoutConnFactory
```

**⇒ `RKNNLayoutMatchPass`（流水线第 20 步）用这些工厂，
为「布局不同的相邻算子」插入**搬运算子**。**

### ★ 这接上了 docs/101 的因果链

docs/101 发现：**`efficientnet_b0` 的 IR 里有 593 个 `Concat`**，而原图只有 62 个
—— **多出的 531 个是编译器插入的**。

**⇒ `ConcatLayoutConnFactory` 就是它们的来源。**
**⇒ 每个插入点都是一个潜在的提交边界 ⇒ 碎片化。**

**⇒ 从「编译器插入了很多搬运算子」（现象）到「哪个类插的」（机制），因果链完整了。**

---

## 2. 其他新符号

```
rknn::RKNNAddConvWeightSparse      ← ★ 稀疏权重（docs/64 的嫌疑 pass）
rknn::RKNNDuplicateWeightPass      ← 重复权重（与 docs/47/50 的 81 遍拷贝相关?）
rknn::ReshapeTranspose / TransposeReshape
onnc::Tile / onnc::TileLower / onnc::TransposeLower / onnc::ConvTransposeLower
onnx::optimization::EliminateNopTranspose / FuseTransposeIntoGemm / FuseConsecutiveTransposes
```

### 日志字符串（会泄露结构）

```
"BroadcastLayoutConnFactory: op(%s) has undefined broadcast type, please check this op."
"Failed to config layer: '%s', illegal tiling method %d"
```

---

## 3. 更新后的完整因果图

```
模型结构（Concat/Add/SE…）
   ⇒ RKNNLayoutMatchPass + 13 个 LayoutConnFactory
       ⇒ **插入搬运算子**（efficientnet: +531 个 Concat）
           ⇒ RKNNTilingPass / RKNNTileChannel / RKNNTileGroupConv
               ⇒ onnc::Tile ⇒ **onnc::TileLower**（下降成硬件算子）
                   ⇒ UnpackTile / UnpackTileCopy
                       ⇒ RKNNModelRegCmdbuildPass（生成 regcmd = 我一直在分析的命令流）
```

**⇒ 链条上的每一环现在都有名字。**

---

## 4. 下一步

| # | 动作 | 预期 |
|---|---|---|
| 1 | 找 `ConcatLayoutConnFactory` 的实现/字符串 | 看它**按什么条件**插入 Concat |
| 2 | 用 RTTI 名在 `.so` 里定位这些工厂的代码 | 拿到判定逻辑 |
| 3 | **检验：13 个工厂的触发次数 vs 实测提交数** | 如果相关，**提交数的驱动就找到了** |

**#3 最直接**：如果「插入的搬运算子数」与提交数相关 ——
但 docs/102 已经否掉了「搬运算子数 ⇒ 提交数」（R²=0.057）。
**⇒ 所以 #3 要改成「按算子类型分开统计」**（Concat 插入 vs Transpose 插入可能效应不同）。

---

## 5. 教训（本轮新增）

1. **★★★ 一个「工厂族」的名字列表 = 一张功能地图。**
   13 个 `*LayoutConnFactory` 的**名字本身就说明了系统怎么工作**：
   **每个算子类型有一个工厂，负责把它与相邻算子的布局接起来。**
   **不需要反汇编 —— 名字够了。**
2. **★ 因果链要「两头对齐」才算完整。** docs/101 有**现象**（593 个 Concat）、
   本轮有**机制**（`ConcatLayoutConnFactory`）—— **两头对齐后这条链才可用**。
   **只找到现象或只找到机制，都不足以支撑结论。**
3. **★ 否证要带着「分类粒度」重看。** docs/102 否掉「搬运算子数 ⇒ 提交数」（R²=0.057），
   但那是**把所有搬运算子混在一起**。**有了 13 个工厂的分类，应该按类型分开再试一次。**
