# 139 · ★★★★★ 换目标：**运行库 `librknnrt` 同样可逆向**，调度器类全在里面

> 接 [docs/138](138-负结果-tensor地址链不能揭示段边界.md)。
> docs/138 的结论是「切段是 `librknnrt` 的调度器做的」。
> 本轮验证：**`librknnrt` 和 `librknnc` 一样能逆**，而且**调度相关的类名全在里面**。

---

## 1. ★★★★★ `librknnrt.so`（板子 `/usr/lib/librknnrt.so`）的可逆性

| 项 | 值 |
|---|---|
| 架构 | **aarch64** |
| 大小 | 7.26 MB |
| 符号表 | **stripped**（与 `librknnc` 一样） |
| `.dynsym` | **643 个 FUNC**（可用） |
| `.rodata` RTTI 名 | **161 个**（可用） |
| `.rela.dyn` 重定位 | **25117 条**（完整） |

**⇒ 与编译器**完全一样的情况**：`.symtab` 被剥离，但 `.dynsym` + RTTI + 重定位都在。**
**⇒ 所以 `src/compiler/rtti_find.py` 那套（RTTI → typeinfo → vtable → 虚函数地址）**直接可用**。**

---

## 2. ★★★★★ 调度相关的类**全在**（35 个 `rknn::` 类）

```
★ rknn::RKNNSubGraph          ← 「段」本身！
★ rknn::RKNNExecutor          ← 执行器（调度）
★ rknn::RKNNJob / RKNNJobV2   ← 作业
★ rknn::RKNNRegisterTask      ← 任务列表构造
  rknn::RKNNRegisterTask_f2 / _lite / _t3 / _t4 / _v2 / _w1 / _w2   ← 各平台的实现
★ rknn::RKNNOperatorTilingInfo ← 与**编译器里同名**的类（docs/126~131）
★ rknn::RKNNModel / RKNNOperator / RKNNTensor / RKNNConfig
  rknn::RKNNTarget / F2 / Lite / T3 / T4 / V2 / W1 / W2
  rknn::RKNNDynCtrlInfo / RKNNNormalize / RKNNProfiler / CLContext
```

**⇒ 这正是我要找的东西：**
* **`RKNNSubGraph`** = 「段」的载体 ⇒ 段边界规则应该在它的构造/划分逻辑里
* **`RKNNRegisterTask`** = 把算子编成 task 列表 ⇒ `task_number` 的来源
* **`RKNNOperatorTilingInfo`** 在运行库里**也有** ⇒ 运行期也会做 tiling

---

## 3. ★★ 已提取的 vtable（可直接用于板上 gdb / objdump）

| 类 | vtable（库内 vaddr） | 部分虚函数地址 |
|---|---|---|
| `RKNNSubGraph` | `0x6c85d8` | `0xe1698`, `0xe16a0`, **`0xe5848`**, `0xe16b8`, `0xe3108`, `0xe8cf0` |
| `RKNNRegisterTask` | `0x6c86b8` | `0xf2720`, `0xf2728`, …, `0xf2770`（间隔 8，多为 thunk） |
| `RKNNJob` | `0x6f5ff8` | `0x2b8d60`, `0x2b8fd0`, `0x2ba970`, `0x2b5f38`, … |

**⇒ 板子上 `objdump` **原生支持 aarch64**（笔记本上的 objdump 不行 —— `can't disassemble for architecture UNKNOWN`），
   所以反汇编要在板子上做（或用 `apt-get download binutils-aarch64-linux-gnu` 装一份）。**

### 板上反汇编 `RKNNSubGraph` 的结果（已做）

`0xe5848` = **析构函数**：释放 `+728`/`+712` 处的 shared_ptr，清理 `+0xc8`，
再遍历 `+168..+176` 的容器；`0xe3108` = 另一个清理路径（`+0x2e8` 的 vector、`+0x2a8` 的树）。
**⇒ `RKNNSubGraph` 至少有 0x2e0 字节，成员分布在 +0x168 / +0x2a8 / +0x2e8 / +696 / +712 / +728。**

---

## 4. ★★★ 为了做这件事修的两个 bug（都在 `rtti_find.py`）

| 坑 | 症状 | 修法 |
|---|---|---|
| **只认 `R_X86_64_*` 重定位** | aarch64 库上一条重定位都读不到，报 **「没有重定位指向 …（可能该 RTTI 未被使用）」** | 加上 `R_AARCH64_*` 前缀 |
| 交替用普通组 `(?:A\|B)` | `IndexError: no such group`（组号整体偏移） | 改用**命名组** |
| 段偏移硬编码 `KNOWN`（x86-64 实测值） | 换库/换架构必错 | **优先用 `readelf` 解析**，`KNOWN` 只在确认是 librknnc 时兜底 |

**⚠️ 第一条特别值得记：那个错误信息会把「架构不匹配」误导成「这个类型没被使用」。**
（和 docs/126 的教训同源：**工具报「没有」时，先怀疑工具。**）

---

## 5. 现在能说什么（边界）

**已验证**：

1. **`librknnrt.so` 与 `librknnc.so` 一样可逆**（stripped 但有 `.dynsym` + RTTI + 重定位）
2. **调度相关的类全在里面**：`RKNNSubGraph` / `RKNNExecutor` / `RKNNJob` / `RKNNRegisterTask`
3. **`RKNNOperatorTilingInfo` 在运行库里也存在**（运行期也做 tiling）
4. **板子的 `objdump` 原生支持 aarch64**；vtable 已提取
5. `rtti_find.py` 已支持 aarch64（并且修了会把架构问题误报成「类型未使用」的坑）

**未做**：

* **板上 gdb 还没有**（下一步要装；`apt-get download gdb` + `dpkg-deb -x` 不需要 root，
  但板子是 aarch64 ⇒ 要拿 **arm64 的 deb**）
* `RKNNSubGraph` 的**划分逻辑**还没读（只读了析构函数）
* `RKNNRegisterTask` 的 task 生成逻辑还没读

---

## 6. 下一步（可检验）

| # | 动作 | 判据 |
|---|---|---|
| 1 | **在板子上装 aarch64 gdb**（`apt-get download gdb:arm64` 或板子自己的 apt） | `gdb --version` 能跑 |
| 2 | **在 `RKNNSubGraph` 的构造函数/划分函数上下断** | 直接看「一个模型被切成几个 SubGraph」 |
| 3 | 板上 `objdump` 反汇编 `RKNNSubGraph` 的**非析构**方法（`0xe1698` 一族、`0xe8cf0`） | 找到「加算子到子图」的逻辑 |
| 4 | 反汇编 `RKNNRegisterTask`（`0xf2720` 一族） | 看 `task_number` 怎么算出来 |
| 5 | 交叉：把 `RKNNSubGraph` 的成员偏移与 regcmd 里的字段对照 | 把「段」与「提交」对上 |

**#1+#2 是最直接的路径**：既然类名和 vtable 都有，**下断点就能看划分决策**。

---

## 7. 教训（本轮新增）

1. **★★★★★ 「这个库不行」和「这个库我没用对手段」是两件事。**
   我原以为运行库是另一个世界（板子上、aarch64、可能完全不同）；
   实际一查：**和编译器库一样是 stripped + RTTI 完整**，
   而且**类名比编译器库还直白**（`RKNNSubGraph` / `RKNNRegisterTask` 直接写着答案的方向）。
   **⇒ 换目标时，先花 5 分钟验证「新目标能不能用老手段」，再决定要不要发明新手段。**
2. **★★★ 工具的错误信息可能指向完全错误的方向。**
   「没有重定位指向 …（可能该 RTTI 未被使用）」让我一度以为这些类没被用；
   **真相是我的正则只认 x86-64 重定位名。**
   **⇒ 修法已写进代码注释，因为下次换库还会踩。**
3. **★★ 板子本身就是最好的工具机。** 笔记本上的 `objdump` 不认 aarch64，
   而**板子上的 `objdump` 原生认** —— 一条 SSH 就解决了，不用装交叉工具链。
   **⇒ 「在目标机上做目标机的分析」往往比在主机上搭环境便宜。**
