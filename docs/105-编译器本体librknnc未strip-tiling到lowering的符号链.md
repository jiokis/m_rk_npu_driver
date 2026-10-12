# 105 · ★★★ 编译器本体 `librknnc.so` **未 strip**：tiling → lowering 的符号链

> 接 [docs/104](104-IR表含编译器自己的性能模型-Cycles与RW.md)（IR 表只末尾打印一次 ⇒ 需 hook）。
> 本轮去找 hook 的入口，**找到了编译器本体**，而且它**带 C++ RTTI 符号（未 strip）**。

---

## 1. ★ 编译器本体

```
rknn/api/lib/linux-x86_64/librknnc.so     31.5 MB     ELF 64-bit LSB shared object
```

**⇒ 33 个 pass 的实现都在这里，而且**符号没被 strip**（C++ RTTI 名字都在）。**

---

## 2. ★★ tiling → lowering 的符号链（实测提取）

```
rknn::RKNNTileChannel            ← 按通道切 tile
rknn::RKNNTileGroupConv          ← 分组卷积切 tile
rknn::RKNNTileFcBatchFuse        ← ★ FC 的 batch 融合（与 docs/63「FC 变 1×1 conv」相关）
rknn::RKNNTilingPass             ← 主 tiling pass
rknn::RKNNOperatorTilingInfo     ← ★★ **承载 tiling 结果的数据结构**
rknn::f2::DeconvTiling           ← 反卷积的 tiling

onnc::Tile                       ← ★ 中间表示层（onnc = ONNC 编译器框架）
onnc::TileLower                  ← ★★★ **「tile 下降」——把 tile 变成硬件算子**
UnpackTile / UnpackTileCopy      ← ★ 解包（与搬运/拷贝相关）
ChannelTile / TileH
```

### ★★★ 关键发现：`onnc::TileLower`

**「Tile Lower」= 把 tile 下降成硬件算子** ——
**这很可能就是「提交打包」真正发生的地方**（tile → 硬件算子 → regcmd 条目）。

**⇒ 完整链路：**

```
RKNNTileChannel / RKNNTileGroupConv / RKNNTilingPass
   ⇒ onnc::Tile（中间表示）
   ⇒ onnc::TileLower（★ 下降成硬件算子）
   ⇒ UnpackTile / UnpackTileCopy（解包/搬运）
   ⇒ RKNNModelRegCmdbuildPass（生成 regcmd —— 我一直在分析它的输出）
```

**⇒ `UnpackTileCopy` 与 docs/47/50 的「81 遍冗余拷贝」很可能直接相关。**

---

## 3. 另一条有价值的字符串

```
"Failed to config layer: '%s', illegal tiling method %d"
"Tiling failed"
```

**⇒ 「illegal **tiling method** %d」说明 tiling 是**从一组「方法」里选的**，
而且有个 `%d` 编号** —— 与 docs/99 的 `hardware_rules`（12 条规则）对应。
**⇒ 如果能找到「tiling method 枚举」的名字，就等于拿到了切分策略的完整清单。**

---

## 4. 为什么这是重大进展

| 问题 | 之前 | 现在 |
|---|---|---|
| 切分在哪 | 否掉 12 条假设 → 「tiling 层」（docs/102） | **`RKNNTilingPass` → `onnc::TileLower`**（符号链） |
| tiling 结果存在哪 | 不知道 | **`rknn::RKNNOperatorTilingInfo`**（有类型名） |
| 切分策略有哪些 | `hardware_rules` 12 条（Python 侧） | 编译器侧有 **「tiling method %d」** 编号 |
| 编译器本体 | 不知道在哪 | **`librknnc.so`（31.5 MB，未 strip）** |

**⇒ 从「知道 pass 名字」进到「知道实现位置、数据结构名、和子 pass 链」。**

---

## 5. 下一步

| # | 动作 | 预期 |
|---|---|---|
| 1 | **在 `librknnc.so` 里搜更多 RTTI**（`RKNNOperatorTilingInfo` 的方法、tiling method 枚举名） | 拿到切分策略清单 |
| 2 | **用 `LD_PRELOAD` 或 gdb 在 `librknnc.so` 的 `RKNNTilingPass::run` 下断点** | 直接观察切分前后 |
| 3 | 找 `onnc::TileLower` 的实现 | **看「tile → 硬件算子」的规则** |
| 4 | 搜 `UnpackTileCopy` | 解释 81 遍冗余拷贝 |

**#1 最省事**：符号没 strip ⇒ **继续枚举 RTTI 名字**就能拿到更多结构名。

---

## 6. 教训（本轮新增）

1. **★★★ 「未 strip 的 .so」是逆向的金矿。** 我用了几轮去 hook、去猜机制，
   而**编译器本体的 C++ 类名、方法名、日志字符串全都可读**。
   **在动手 hook 之前，先 `strings` 一遍目标库。**
2. **★ 符号链本身就是流程图。** `RKNNTileChannel → onnc::Tile → onnc::TileLower → UnpackTile`
   —— **光看名字就能排出流水线的形状**，不需要反汇编。
3. **★ 日志字符串会泄露「枚举」。** `"illegal tiling method %d"` 这句话告诉我
   **tiling 是一组编号的方法** —— 这是**从一句报错里读出的结构信息**。
