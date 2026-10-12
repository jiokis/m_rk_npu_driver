# 99 · `hardware_rules` 的 tiling 规则：**proc 可调用，但需要「准备好的 IR」**

> 接 [docs/98](98-融合规则表已枚举-只有4种算子对能融合.md)（IR 层融合规则已枚举）。
> 本轮去调 tiling 层的 12 条规则，拿到了签名，也定位了它们的**前置条件**。

---

## 1. ✅ 签名已拿到

```
每条规则： {'ops': [...], 'proc': <cyfunction>}
proc 签名： (ir, nodes, quant_tab={})

convert_conv_to_pad_conv      ops=['Conv']
tiling_maxpool                ops=['MaxPool']
tiling_global_maxpool         ops=['GlobalMaxPool']
tiling_depthwise_conv         ops=['Conv']
tiling_group_conv_act         ops=['Conv','exSwish','Conv']     ← ★ 三个算子一组
tiling_conv_channelin         ops=['Conv']
tiling_deconv_channelin       ops=['ConvTranspose']
convert_reducesum_to_einsum   ops=['ReduceSum']
convert_exmatmul_to_conv      ops=['exMatMul']
p_tiling_exmatmul_channel     ops=['exMatMul']
gather_to_conv                ops=['Gather']
fix_exsoftmax_axis            ops=['exSoftmax13']
```

**注意 `tiling_group_conv_act` 的 `ops` 是 `['Conv','exSwish','Conv']`** ——
**它描述的是一个「三算子模式」**（分组卷积 + Swish + 卷积）。
**⇒ 这印证了 docs/98 的发现：`exSwish` 相关结构在 tiling 层被特殊处理**
（而 IR 层 `Conv→Sigmoid` 不融合）。

---

## 2. ⚠️ 调用返回空 ⇒ 需要「准备好的 IR」

```python
pr = F.hardware_rules['tiling_depthwise_conv']['proc']
pr(m, dw_nodes)          # ⇒ ([], [])
pr(m, ow_nodes)          # ⇒ ([], [])
```

（`mobilenet_v2`：17 个 depthwise conv、35 个普通 conv，调用都返回 `([], [])`）

**⇒ proc 能跑（不报错），但不产出任何结果。**
**⇒ 前置条件没满足** —— 最可能是：
* 节点还没有**量化信息**（`quant_tab` 是空 `{}`，而签名里它是个参数）
* IR 还没有**目标/形状推断**（`infer_shapes` 需要 ≥2 个参数，我没跑过）

---

## 3. 下一步（很具体）

| # | 动作 | 说明 |
|---|---|---|
| 1 | **先跑 `m.infer_shapes(...)`**（补上它需要的参数） | 形状是 tiling 的输入 |
| 2 | **把量化信息填进 IR**（`ir_utils.add_quant_to_ir`） | 签名里的 `quant_tab` 参数 |
| 3 | 再调 `proc(ir, nodes, quant_tab=...)` | 应该就有输出了 |
| 4 | 或**直接在完整编译流水线里挂钩**（`rknn_base.build` 的中间步骤） | 绕过「手工准备 IR」 |

**#4 可能更省事**：`rknn_base` 里的 `build` 会按顺序跑所有 pass，
**在其中 hook 一下（或直接读它的日志）就能看到每一步的 IR 状态** ——
比手工准备 IR 更贴近真实流程。

---

## 4. 目前这条线的完整状态

| 层 | 工具 | 状态 |
|---|---|---|
| **IR 层：融合** | `can_fuse(node, node2, relu=)` | ✅ **可用，规则已枚举**（4 种对 100%，其余 0%） |
| **tiling 层：切分** | `hardware_rules[*]['proc'](ir, nodes, quant_tab)` | 🟡 **可调用，但需准备好的 IR** |
| 量化层 | `get_quant_ann(name, attr)` | 🟡 签名已知，量化前返回空 |
| 编译流水线 | `rknn_base.build` / `rknn_compiler` | ⬜ 未接 |

**⇒ 三层里**一层已通、两层已拿到接口**。这比 20 轮探针时（完全靠外部观测）好得多。**

---

## 5. 教训（本轮新增）

1. **★ 「可调用但返回空」是一个明确的状态，要写清楚。**
   它不是「失败」也不是「成功」—— 它说明**接口对了、前置条件没满足**。
   **把卡点写成「缺什么」，比写成「没成功」有用得多。**
2. **★ 规则表里的 `ops` 字段本身就是信息。**
   `tiling_group_conv_act` 的 `ops=['Conv','exSwish','Conv']` 告诉我
   **tiling 层是按「算子模式」匹配的**，而且 **`exSwish` 是被单独识别出来的**
   —— 这与 docs/98 的 `Conv→Sigmoid` 不融合是**同一件事的两层表现**。
   **读规则表的字段，往往比调用它更快得到信息。**
3. **★ 手工准备 IR 可能不如「在真实流水线里挂钩」。**
   我需要形状 + 量化信息才能调 tiling proc —— 而这些正是 `build` 会依次产生的。
   **与其复现前置步骤，不如在真实流程里观察。**
