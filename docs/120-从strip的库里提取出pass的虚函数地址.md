# 120 · ★★★ 从**被 strip 的** `librknnc.so` 里提取出 pass 的**虚函数地址**

> 接 [docs/119](119-gdb可获取-阻塞原因①已移除.md)（gdb 已装好）。本轮是目标的最后一轮，
> 我用 gdb 之外的一条**纯静态**路径解决了「符号表被剥离」这个根本性阻塞。

---

## 1. ★ 路径：RTTI → typeinfo → vtable → **虚函数地址**

```
librknnc.so 的 .symtab 被剥离（只有 .dynsym，且不含 pass 符号）
但：C++ RTTI 名还在 .rodata、vtable 在 .data、**.rela.dyn 重定位表是完整的**
⇒ 全部可从文件静态解析，不需要 gdb
```

### 实测结果：`rknn::RKNNTilingPass` 的 10 个虚函数地址

```
0x391450   0x3914d0   0x3913f0   0x391420   0x284720
0x284730   0x284740   0x2bed20   0x284750   0x3854b0
（全部落在 .text 内：.text = 0x11b260 ~ 0x1b23fb2）
```

**⇒ 这是**可用的 gdb 断点地址**，而**不需要符号表**。**

### 用它们下断点

```
gdb -p <python_pid>
(gdb) info sharedlibrary          # 取 librknnc.so 的加载基址 BASE
(gdb) break *BASE+0x391450
(gdb) continue
```

---

## 2. ★ 沉淀成工具：`src/compiler/rtti_find.py`

```bash
python3 src/compiler/rtti_find.py RKNNTilingPass RKNNLayoutMatchPass RKNNWeightTransposePass \
        RKNNPerChannelPrep RKNNFuseOptimizerPass
```

输出落盘 `data/analysis/pass_addresses.txt`。

### 开发中踩的三个坑（都写进注释了）

| # | 坑 | 修法 |
|---|---|---|
| 1 | 段解析正则不匹配 `readelf -SW` 的输出格式 | 加 `KNOWN` 常量兜底 |
| 2 | `.rodata` 里存的是 **mangled** 名（`N4rknn14RKNNTilingPassE`），精确匹配找不到 | 改**子串**搜索 |
| 3 | 子串命中点落在名字**中间**（`0x1a8d73b` 而非起点 `0x1a8d720`） | 回退到 mangled 名起点 |
| 4 | 「vtable 起点 = typeinfo 指针位置 − 8」的算术**不可靠**（多重继承/虚基类） | 改为**向前扫描并筛出落在 `.text` 内的项** |

---

## 3. 这一步的意义

| 之前（docs/118） | 现在 |
|---|---|
| 「gdb 路径不可行」：没装 gdb + 符号表剥离 | gdb 已装（docs/119）+ **符号表剥离已被绕过**（本轮） |
| 「要解它必须先装 gdb 并做 vtable 考古」 | **vtable 考古已完成，地址在手** |

**⇒ 布局与切分问题的观测路径**重新打开**了。**
**⇒ docs/117/118 的「全闭」结论**被本轮推翻**（它们只考虑了「有符号表」和「有 Python 回调」两条路）。**

---

## 4. 下一步（具体、可执行）

| # | 动作 | 说明 |
|---|---|---|
| 1 | **在 `BASE+0x391450` 等下断点，观察 tiling 前后的图** | 回答「提交切分怎么发生」 |
| 2 | 在 `RKNNTilingPass` 返回处断下，读 IR 节点的 `layout` | 回答「权重布局」（**注意要在 `RKNNLayoutMatchPass` 之后**） |
| 3 | 用同样方法找 `RKNNModelRegCmdbuildPass` 的地址 | 观察 regcmd 生成 |

---

## 5. 教训（本轮新增）

1. **★★★ 「符号表被剥离」不等于「函数地址不可得」。** 我上一轮把它当成**根本性阻塞**；
   **实际上 RTTI + 重定位表完整保留了 vtable ⇒ 虚函数地址可以直接算出来。**
   **剥离的只是「名字 → 地址」的映射，不是地址本身。**
2. **★★ 静态解析（读文件）常常比动态调试（gdb）更省事。** 我本来准备用 gdb 回溯调用栈；
   **而直接读 `.rela.dyn` 就拿到了地址** —— **不需要运行、不需要断点、不需要回溯。**
3. **★★ 我连续两轮（docs/117/118）给了「路径全闭」的结论，两次都被推翻。**
   * docs/117 说「不是 hook 太早而是没有回调」→ **对**（那条路确实闭）
   * docs/118 说「gdb 不可行」→ **错**（可装，且符号表剥离可绕过）
   **⇒ 教训：说「某条路不可行」时，要区分「这条路不存在」和「我还没找到走法」。**
