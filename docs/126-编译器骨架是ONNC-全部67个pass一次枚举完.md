# 126 · ★★★★★ 编译器骨架是 **ONNC**（开源）—— 全部 67 个 pass 一次枚举完

> 接 [docs/125](125-预测被确认-两个类型同源.md)。
> 本轮从「盯住一个 pass」升级为「**看清整个编译器**」——
> 起因是**一个字符串枚举**，收获远超预期。

---

## 1. ★★★ 起因：枚举 `.rodata` 里的字符串，撞见真相

我想确认 `rknn::RKNNOperatorTilingInfo` 到底有没有自己的 RTTI 名，于是把
`.rodata` 里所有含 `Tiling` / `ModulePass` 的字符串全打出来（1862 个串里的 60 个）。
结果第一条就命中了架构：

```
N4onnc10CustomPassIN4rknn14RKNNTilingPassENS_10ModulePassEvEE
  = onnc::CustomPass< rknn::RKNNTilingPass, onnc::ModulePass, void >
```

**⇒ RKNN 的编译器骨架是 ONNC（Open Neural Network Compiler，开源），
   RKNN 只提供 `rknn::XxxPass` 这些叶子类。**

**⇒ 「pass 清单」是可以静态完整枚举的** —— 不需要 gdb，不需要跑编译。

---

## 2. ★★★ 全部 67 个 pass（新工具 `src/compiler/pass_list.py`）

| 来源 | 数量 | 说明 |
|---|---|---|
| `rknn::*` | **38** | RKNN 自己的 pass |
| `onnc::*` | **29** | ONNC 框架的 pass（含 5 个 `GraphBuildingPass`） |

**tiling 家族（本轮新发现，之前完全不知道它们存在）**：

```
RKNNTilingPass          ← 通用 tiling
RKNNTileChannel         ← 按通道切
RKNNTileGroupConv       ← 切 group conv
RKNNTileFcBatchFuse     ← 切 FC / batch fuse
```

**其它与两个未解问题直接相关的 pass**：

```
RKNNSetOpTargetPass          设置算子目标（哪部分跑 NPU / CPU）
RKNNMultiSurfacePass         多 surface 切分  ← 可能就是「一个算子分到几个 core」
RKNNAllocateConvCachePass    conv cache（CBUF）分配  ← 直接对应 docs 里的 CBUF 复用分析
RKNNSubGraphMemoryPlanPass   子图内存规划
RKNNProfileAnalysisPass      性能分析（编译器自己的 Cycles/RW 模型，docs/104）
RKNNModelRegCmdbuildPass     生成寄存器命令流  ← 提交数的真正来源
RKNNFlatcModelBuildPass      flatbuffer 打包
RKNNModelExportPass          导出
RKNNOperatorIdGenPass        给算子编号
```

完整表：`data/analysis/rknn_passes_full.txt`。

---

## 3. ★★ 三条**修正**（都是本轮反汇编推翻的旧结论）

### ① `rknn::RKNNOperatorTilingInfo` **没有自己的 RTTI/vtable**

`.rodata` 里唯一含 `RKNNOperatorTilingInfo` 的串是

```
St23_Sp_counted_ptr_inplaceIN4rknn22RKNNOperatorTilingInfoESaIS1_ELN9__gnu_cxx12_Lock_policyE2EE
```

即 `std::_Sp_counted_ptr_inplace<rknn::RKNNOperatorTilingInfo, std::allocator<...>, _Lock_policy 2>`。
**裸的 `N4rknn22RKNNOperatorTilingInfoE` 只作为它的前缀存在，不是独立名字。**

**⇒ 该类型是**非多态**的（没有虚函数）**
**⇒ [docs/125](125-预测被确认-两个类型同源.md) 说的「vtable `RKNNOperatorTilingInfo` @ `0x1dc4670`」是**错的**：
   那是它的 **shared_ptr 控制块**的 vtable。**

**反汇编证明**（`0x1dc4670` 的 5 个槽，正好等于 `_Sp_counted_ptr_inplace` 的 5 个虚函数）：

| 槽 | 地址 | 反汇编 | 判定 |
|---|---|---|---|
| `[0]` | `0x3913e0` | 裸 `ret` | 完整析构（无需清理） |
| `[1]` | `0x391400` | `jmp _ZdlPv` | deleting dtor |
| `[2]` | `0x391510` | 见 §4 | **`_M_dispose`（销毁 `_Tp`）** |
| `[3]` | `0x391410` | `jmp _ZdlPv` | `_M_destroy` |
| `[4]` | `0x391480` | 比对 `_ZTSSt19_Sp_make_shared_tag` | **`_M_get_deleter`** ✓ |

### ② [docs/122](122-观测通道端到端验证-断点处对象的vtable正是静态提取的那个.md) 的「`rdi = rsi = this`」是**残留寄存器**，不是参数

反汇编 `0x391420`：

```asm
391420: push %r12
        xor  %ecx,%ecx            ; 第 4 参 = 0
        mov  %rdi,%r12
        xor  %edx,%edx            ; 第 3 参 = 0
        xor  %esi,%esi            ; 第 2 参 = 0  ← rsi 被**清零**，入口值无意义
        lea  0x1a8d760,%rdi       ; rdi = "N4rknn14RKNNTilingPassE"（自己的 RTTI 名）
        call __cxa_demangle
        mov  %r12,%rdi
        mov  %rax,%rsi            ; rsi = demangle 结果
        call 0x2b4c50             ; → this->set_name(name)
        mov  %r12,%rax
        ret
```

**⇒ 这是「取自己的 RTTI 名 → demangle → 设名字」的虚方法。**
**⇒ 它**不读**入口的 `rsi`** ⇒ docs/122 里那个「第 1 参数」是**上一个调用留下的残留**。
**教训：寄存器入口值只有在**确认函数会读它**之后才有意义 —— 否则是在读上一个函数的现场。**

（同一段的 `0x3913f0` 是 `lea 0x1a8d718(%rip),%rax; ret`，返回一个**空字符串**的地址 —— 另一个 name getter。）

### ③ `objdump` 的 `符号+偏移` 标注**只在符号 size 内有效**

析构函数里出现 `lea 0x1e05e00`，objdump 标成 `_ZTVN4rknn12RKNNCompilerE@@Base+0xa0`。
但 `readelf -sW` 显示该符号在 `0x1e05d60`、**size 只有 32**（到 `0x1e05d80` 为止）。
**⇒ `0x1e05e00` 与 `RKNNCompiler` 无关**，objdump 只是标注了「最近的**前面**那个符号」。

**⇒ 规则：`symbol+offset` 必须**同时核对符号 size**，否则会凭空造出一个错误归属。**
（本项目里「猜结构」栽过多次；这次是「工具输出被当成结论」。）

---

## 4. ★★★★ 决定性的方法转变：**用 vptr 在堆里找回非多态对象**

`RKNNOperatorTilingInfo` 非多态 ⇒ 没有虚函数可下断点（docs/120-125 那套失效）。
但它由 `std::make_shared` 创建 ⇒ 堆上有控制块，**控制块的 vptr 是静态可求的常量**。

**⇒ 跑完编译后扫描可写内存，搜这个常量，就能找回对象。**

### 踩到的坑（本轮修的）

| 坑 | 症状 | 修正 |
|---|---|---|
| `self.stop = stop` 覆盖了 `Breakpoint.stop()` 方法 | `attribute of type 'bool' is not callable` | 改名 `self.do_stop` |
| `break 符号名` 在 `dlopen` 之前执行 | `Function "RKNNCompiler_build" not defined` | `catch load librknnc.so` 后再下断 |
| 命中汇总正则要求地址后紧跟 `===` | 明明有命中却报 0 | 去掉尾部锚定 |
| **搜 `0x1dc4670` 命中 0 处** | 见 §5 | **要搜 `0x1dc4678`** |
| 迭代名解析用了贪婪 `[A-Za-z0-9_]+` | pass 名带 `ENS_10ModulePassEv` 尾巴 | 用 Itanium **长度前缀**切 |

---

## 5. ★★★ 修正：对象的 vptr = `rtti_find` 报的地址 **+8**

`rtti_find.py` 报的「vtable @ `0x1dc4670`」是 **typeinfo 槽**的地址。
按 Itanium ABI，对象里存的 vptr 指向**第一个虚函数槽**：

```
typeinfo 槽   0x1dc4670   → 指向 typeinfo 0x1e090a8
第一个虚函数  0x1dc4678   → 0x3913e0
对象里的 vptr = 0x1dc4678        ← 实测（本轮 dump 到的 [0] 槽）
```

**⇒ 搜内存必须搜 `vtable+8`。**（这是「搜 0 处」的根因，不是对象不存在。）

---

## 6. 本轮新增工具

| 工具 | 作用 |
|---|---|
| `src/compiler/pass_list.py` | 从 RTTI 名**静态枚举全部 pass**（可 `--vtable` 顺带解析虚函数地址） |
| `tools/gdb/build_min.py` | 最小 ONNX→RKNN 构建，专供 gdb `run`（可用 `BUILD_MIN_ONNX` 换模型） |
| `tools/gdb/probe.py` | 在任意库内偏移下断，dump `this`/参数寄存器/栈回溯，**可把某个偏移当 `std::vector` 头解析** |
| `tools/gdb/scan_heap.py` | 编译跑完后扫可写内存，按 vptr 找回对象 |

**⇒ 这三个 gdb 工具把 docs/121-125 里散在 `/tmp` 的做法固化了**（`/tmp` 是 tmpfs，重启就没了 —— 本轮开头就重装了 gdb）。

---

## 7. 教训（本轮新增）

1. **★★★ 一次「便宜的枚举」胜过十次「精巧的推断」。**
   我只是想确认一个 RTTI 名，把 `.rodata` 里的相关串全打了出来 ——
   **结果直接看清了编译器的骨架（ONNC）和全部 67 个 pass。**
   **当你要猜「有没有 X」时，先把候选全集列出来。**
2. **★★★ 反汇编能终结猜测。** 本轮三个「修正」全部来自反汇编：
   vtable 归属、`rdi=rsi` 的含义、`symbol+offset` 的归属。
   **gdb 给出「发生了什么」，反汇编给出「为什么」。**
3. **★★ 修 bug 的过程本身就是方法记录。** 本轮 5 个坑全是**工具自身的坑**，
   而且症状（`bool is not callable` / `not defined` / 命中报 0）都**指向错误的方向**。
   **工具报「没有」时，先怀疑工具。**（这条本项目已付出多次代价。）
4. **★ 结论要写清「修正了哪一条」。** docs/125 是我上一轮刚写的，
   本轮就推翻了它的一个具体论断 —— **写清被推翻的是哪句，比笼统说「有修正」有用。**

---

## 8. 现在的状态

| 问题 | 状态 |
|---|---|
| 提交数驱动 | **tiling 结果已可读**（见 [docs/127](127-tiling结果已读出-每个算子一张tile表.md)） |
| 通用权重布局 | 仍待观测；`RKNNLayoutMatchPass` 地址已可求（同 §2 工具） |
| 编译器骨架 | **已明确是 ONNC** ⇒ 框架语义可查公开资料 |

**⇒ 两个未解问题里，第一个（tiling/提交数）本轮从「可达」推进到「已读出数据结构」。**
