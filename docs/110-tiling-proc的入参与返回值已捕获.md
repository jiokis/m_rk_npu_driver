# 110 · ★★★ tiling proc 的**入参与返回值已捕获**：它返回「创建的节点」

> 接 [docs/109](109-规则触发可观测-tiling规则确实触发.md)（monkeypatch 成功、tiling 规则确实触发）。
> 本轮把 tiling proc 包一层，**同时捕获入参（哪些节点）与返回值（产出什么）**。

---

## 1. ★ 观测结果

```
TILING convert_conv_to_pad_conv   nodes=['conv']  => ([], [])
TILING tiling_depthwise_conv      nodes=['conv']  => ([], [])
TILING tiling_conv_channelin      nodes=['conv']  => ([], [])
TILING convert_exmatmul_to_conv   nodes=['fc#1']  => ([node…], [node…])    ← ★ 返回了节点
TILING convert_conv_to_pad_conv   nodes=['fc#2']  => ([], [])
TILING tiling_conv_channelin      nodes=['fc#2']  => ([node…], [node…])    ← ★
```

### ★ 关键：返回值是**它创建的节点**

`tiling_conv_channelin` 对 `fc#2` 返回的节点长这样：

```
input: "f_tp_rs"
input: "f_tp_rs_split_i1"
output: "f_tp_rs…"
```

**⇒ 它把 `fc#2` 的输入沿**输入通道**切成了 `f_tp_rs_split_i1` 之类的新节点。**

**⇒ `tiling_conv_channelin` 的语义 = 「把一个 conv 沿输入通道切开」，返回值是切出来的节点。**

---

## 2. ★★ 一个重要的观察：三个规则**都**在每个 conv 上触发

```
对 'conv'：convert_conv_to_pad_conv → tiling_depthwise_conv → tiling_conv_channelin  （都返回空）
对 'fc#2'：convert_conv_to_pad_conv → tiling_depthwise_conv → tiling_conv_channelin  （最后一个返回节点）
```

**⇒ 这三条规则是**依次尝试**的（前两条不适用就返回空，第三条适用就产出）。**
**⇒ 这就是「tiling 方法」的**选择机制** —— 与 docs/105 的 `"illegal tiling method %d"` 对应。**

**⇒ `nodes` 入参始终是**单个节点**（`['conv']` / `['fc#2']`），
说明这些规则是**逐算子**应用的。**

---

## 3. 这条通道能回答什么

| 问题 | 怎么用这条通道回答 |
|---|---|
| **为什么 `regnet` 的 `tile/conv=44.3` 而 `mnasnet` 只有 3.2** | 对同一算子，比较两模型的 tiling 返回值（切出几个节点） |
| **哪些算子会被切** | 统计「返回非空的规则 + 节点类型」 |
| **切分粒度由什么决定** | 比较同一规则在不同 shape/量化参数下的返回值 |
| **`tile` 总数 vs 提交数** | 累计所有返回的节点数，与提交数对照 |

**⇒ 全部是**直接观测**，不再需要相关性。**

---

## 4. 下一步（具体）

| # | 动作 | 预期 |
|---|---|---|
| 1 | **对 `regnet_x_400mf` 与 `mnasnet0_5` 各跑一次，比较 tiling 返回的节点数** | **直接看出切分粒度的差异** |
| 2 | 统计每个模型「tiling 返回的节点总数」 | 与实测提交数对照（**这是切分的直接产物**） |
| 3 | 对同一算子改 shape/量化参数，看返回值变化 | **切分规则的条件** |

**#1 最直接**：`regnet` 的 `tile/conv=44.3` 是 `mnasnet`（3.2）的 14 倍
—— **如果 tiling 返回的节点数也差 14 倍，那「切分粒度」就定位到了。**

---

## 5. 教训（本轮新增）

1. **★★★ 捕获「返回值」比捕获「触发」有用得多。**
   docs/109 我只数了「谁触发了多少次」——那只能做相关性（而且失败了）。
   本轮捕获**返回的节点** ⇒ **直接看到它切出了什么**。
   **hook 一个函数时，返回值比调用次数重要。**
2. **★ 「依次尝试、不适用返回空」是一个可识别的模式。**
   三条规则在每个 conv 上依次触发、前两条返回空 ⇒ **这就是「方法选择」**。
   **返回值是否为空，本身就是判定的结果** —— 不需要读代码。
3. **★ 入参是「单个节点」这个细节定了规则的作用域。**
   `nodes=['conv']` ⇒ **逐算子应用**（不是整图）。
   **一个参数的形式，往往就说明了函数的语义边界。**
