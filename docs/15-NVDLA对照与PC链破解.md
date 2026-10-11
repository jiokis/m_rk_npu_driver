# NVDLA 参考模型对照 + PC 链机制完全破解

> 原始调研报告：[15a-NVDLA对照报告-原始.md](15a-NVDLA对照报告-原始.md)（子代理产出，含全表与来源）
> 本文件是**我方复核与结论**。

---

## 一、最反直觉的发现：NVDLA 官方没有 "CNA" 这个块

| | |
|---|---|
| 官方块名 | **CSC**（0x6000–0x6FFF，25 个寄存器） |
| 证据 | hwarch 规格里 "CNA" 出现 **0 次**；`nvdla/hw/spec/manual/` 只有 `NVDLA_CSC.rdl`，无 `NVDLA_CNA.rdl` |

**"CNA" 是 Rockchip 自己的命名**，Mesa 与 Linux 内核沿用了它。

而我之前想找的 `D_DATAIN_ADDR_LOW` 在 NVDLA 里**根本不存在**（全仓库 0 命中），
真实名是 `CDMA_D_DAIN_ADDR_LOW_0_0` = **0x5034**。

> **教训**：不要拿"我以为的 NVDLA 寄存器名"去对照，那个名字可能是社区叫法而非官方名。

---

## 二、RK 的块 = NVDLA 块的合并

| RK 窗口 | RK 名 | NVDLA 对应物 |
|---|---|---|
| 0x0000 | PC | **无 —— Rockchip 新增** |
| 0x1000 | CNA | **CDMA(0x5000) + CSC(0x6000) 合并** |
| 0x3000 | CORE | **CMAC_A/B(0x7000/0x8000) + CACC(0x9000) 合并** |
| 0x4000 | DPU | SDP(0xB000) |
| 0x5000 | DPU_RDMA | SDP_RDMA(0xA000) |
| 0x6000 | PPU | PDP(0xD000) |
| 0x7000 | PPU_RDMA | PDP_RDMA(0xC000) |

**"CNA 是合并块"有三重独立证据**：
1. 它同时含 feature 地址(`0x1070`) + 权重地址(`0x1110`) + CBUF 配置 + CVT + PAD + conv stride
   —— 在 NVDLA 里这些**分属 CDMA 与 CSC 两个块**
2. CLK_GATE 位域写成 `CNA_FEATURE / CNA_WEIGHT / CSC / CBUF`（自己就承认拆成了几部分）
3. PC 中断位图把 CNA 拆成 `CNA_FEATURE_0/1`、`CNA_WEIGHT_0/1`、`CNA_CSC_0/1`

### block_id 的结构（顺手确认）

`block_id = target | 写操作位`，target 是**2 的幂**（每个块一个 bit）：

```
0x0041 → 0x40 (bit6)    0x0081 → 0x80 (bit7, GLOBAL)
0x0101 → 0x100 (PC)     0x0201 → 0x200 (CNA)
0x0801 → 0x400 (CORE)   0x1001 → 0x800 (DPU)
0x2001 → 0x1000         0x4001 → 0x2000 (PPU)   0x8001 → 0x4000
```

---

## 三、★★ PC 链机制完全破解（本文件最重要的结果）

### 3.1 NVDLA 没有命令流 —— 这是 Rockchip 加的

NVDLA 是**纯 CSB 寄存器编程**：CPU 写寄存器 → 写该块 `D_OP_ENABLE` 的 bit0 →
硬件完成后自清并切乒乓组。官方地址表 16 个块里**没有 PC**，
而且规格里有一句 "The NVDLA hardware does not have intrinsic support for
dependency tracking"。

内核 `rocket_job.c` 则明确有：
```c
rocket_pc_writel(core, BASE_ADDRESS, task->regcmd);
PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT((regcmd_count + 1) / 2 - 1);
```

**⇒ 命令流 + PC 链是 Rockchip 的增量，不是 NVDLA 的。**

### 3.2 `0x0014` 的语义：全量 610/610 验证通过

我一开始用子代理给的公式直接验，只有 329/615 命中。查下来发现差的是**一个标志位**：

```
0x1003E = 0x10000 + 0x3E = 65536 + 62    ← 62 正是公式算出的值
0x1000E = 0x10000 + 0x0E = 65536 + 14    ← 14 正是公式算出的值
```

**把 bit16 剥离后，610/610 全部命中，零例外：**

```
0x0014 = ((目标段 regcfg_amount + RKNPU_PC_DATA_EXTRA_AMOUNT + scale − 1) / scale − 1)
         | (flag << 16)
         其中 EXTRA = 4, scale = 2（RK3588）
```

这条公式把**三样东西**串起来了：
1. 内核驱动代码里的公式（字面一致）
2. 我们独立测出的「`regcfg_amount` 是指令条数」
3. 命令流里的链结构（`0x0014` 描述的是**目标段**的长度，不是自己那一段）

**⇒ `0x0014` 就是"下一段该写进 `PC_DATA_AMOUNT` 的值"。**
链指令是一个 **(下一段地址, 下一段DATA_AMOUNT)** 的二元组 —— 两这条指令。

### 3.3 段与链的精确布局

```
段 = amount 条「体」 + 2 条链指令
      [0 .. amount-1]  体（由 regcfg_amount 计数）
      [amount]         0x0010 = 下一段地址
      [amount+1]       0x0014 = 下一段的 PC_DATA_AMOUNT（低16位）+ 相位标志（bit16）
```

### 3.4 相位标志的分布

| flag | 覆盖的 task index | 数量 |
|---|---|---|
| 0 | 1..205 与 587 | 329 |
| 1 | 206..586 | 281 |

**两者 index 零重叠**（可分），说明它标记的是执行相位，不是算子边界。
确切含义待定，但不影响链的构造。

---

## 四、复核子代理提出的两处疑问

### 4.1 ⚠️ CNA `0x1184`：子代理是对的，**我的判据错了**

| 取值 | 次数 | 我的旧判读 |
|---|---|---|
| `0xffffff80`（= -128，符号扩展） | 415 | ❌ 判成"张量地址" |
| `0xffff8080`（= -32640） | 4 | ❌ 同上 |

Mesa/内核把它定义为 `PAD_CON1 / PAD_VALUE`（填充值）—— **子代理正确**。

**我的判据错在**：只要"值落在某缓冲区区间内"就算地址。
`0xffffff80` 恰好落在高位缓冲区区间里，属**巧合命中**。

**已修正**：地址字段还要排除「符号扩展的小负数」
（`v >= 0xFF000000` 且 `0x100000000 − v < 0x400000`）。
修正后**张量地址寄存器从 10 个降为 8 个**。

对照修正后的 8 个（全部与 NVDLA 对得上）：

| RK 寄存器 | NVDLA 对应 | 偏移关系 |
|---|---|---|
| CNA `0x1070` | CDMA `D_DAIN_ADDR_LOW` | 语义同 |
| CNA `0x1110` | （Rockchip 权重解压引擎的地址） | NVDLA 无解压硬件 |
| DPU `0x4020` | SDP `D_DST_BASE_ADDR_LOW` | 语义同 |
| DPU_RDMA `0x5018` | SDP_RDMA `0xa018` | **偏移完全相同** |
| DPU_RDMA `0x5020` | SDP_RDMA `0xa02c` | 语义同 |
| DPU_RDMA `0x5038` | SDP_RDMA `0xa05c` | 语义同 |
| PPU `0x6070` | PDP `0xd070` | **偏移完全相同** |
| PPU_RDMA `0x701c` | PDP_RDMA `0xc01c` | **偏移完全相同** |

### 4.2 CNA `0x100c/0x1010/0x1014`

采纳子代理建议：**不要用 NVDLA CSC 的位域去解**。位置（+0x0c/+0x10/+0x14）与 CSC 的
`D_MISC_CFG`/`D_DATAIN_FORMAT`/`D_DATAIN_SIZE_EXT_0` 一致，但内容对不上 ——
只能用 Mesa/内核的位域。

---

## 五、`OP_ENABLE` 位图获得外部确证

我观测的 `block 0x0081 offset 0x0008 = [24, 29, 96]`，子代理在 Mesa/内核里找到了定义：
**GLOBAL `0xf008` OPERATION_ENABLE**，而且是 **NVDLA 完全没有的寄存器**
（NVDLA GLB 只有 HW_VERSION / INTR_MASK / INTR_SET / INTR_STATUS 四个）。

位域：bit0 CNA_OP_EN、bit1 RESERVED、bit2 CORE、bit3 DPU、bit4 DPU_RDMA、bit5 PPU、bit6 PPU_RDMA

| 观测值 | 二进制 | 解释 | 对应层类型 |
|---|---|---|---|
| 24 | `0b0011000` | DPU \| DPU_RDMA | 纯激活/逐元素 |
| 29 | `0b0011101` | CNA \| CORE \| DPU \| DPU_RDMA | 卷积 |
| 96 | `0b1100000` | PPU \| PPU_RDMA | pooling |

保留位 bit1 **从未置位**，三例全中 ⇒ 可作"一层涉及哪些块"的算子分类特征。

---

## 六、Rockchip 相对 NVDLA 的增量（解释不了的部分）

| 增量 | 说明 |
|---|---|
| **PC 块 + 命令流 + PC 链** | NVDLA 完全没有，见 §3 |
| **GLOBAL 使能位图** | NVDLA GLB 无此寄存器 |
| **CNA 权重解压引擎**（`DCOMP_CTRL/REGNUM/ADDR0` + 16 个 AMOUNT，`0x1100-0x117c`） | NVDLA 只有一个压缩标志位，**没有解压硬件** |
| FC_CON / FC_DATA_SIZE | NVDLA 无 |
| DMA_CON0 burst | NVDLA 无 |
| DPU BS_OW / WDMA_SIZE | NVDLA 无 |
| PPU MISC_CTRL、QoS/ARB 字段 | NVDLA 无 |
| `0x0041`（block 0x40 / reg 0x0000） | 仍未解释（Mesa/内核未定义） |

---

## 七、对模拟器的影响（已落地）

| 项 | 变化 |
|---|---|
| 段布局 | `amount + 1` → **`amount + 2`**（链是两条） |
| 链长度语义 | 新增 `Machine.pc_chain_amount()`，**全量 610/610 验证** |
| 张量地址寄存器 | 10 → **8**（剔除假阳性 `0x1184`） |
| 测试 | 加到 13 项，全通过 |

`src/sim/rknpu/regmap.py` 已把 NVDLA 对照结论写进注释，
每个地址寄存器标注了对应的 NVDLA 名与偏移关系。

---

## 八、下一步

- [ ] **试着构造一个能跑通的链**：用 `RegcmdBuilder` 生成 (体 + 链) 并让模拟器执行，
      检查是否与真实样本结构一致 —— 这是"自己造算子配置"的第一块拼图
- [ ] 用 NVDLA 位域给 CNA `0x100c/0x1010/0x1014` 之外的寄存器逐个赋语义
- [ ] 查清 `0x0014` bit16 相位标志的确切含义
- [ ] 查清 `0x0041` 是什么
- [ ] 功能级模拟：等 DMA 格式确定后实现内存搬运与算子计算
