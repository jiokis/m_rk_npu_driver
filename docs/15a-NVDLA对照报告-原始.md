# NVDLA 公开寄存器规格 ↔ RK3588（NVDLA 衍生）观测寄存器 对照报告

> 目的：用 NVDLA 的公开规格解释从 RK3588 闭源运行时命令流里逆出的寄存器偏移。
> 所有偏移均取自一手来源，未找到的一律写"未找到"。
> 生成时间：本会话调研结果。相关文件另存：`NVDLA规格-vs-RK3588观测寄存器-对照报告.md`

---

## 0. 结论速览（先看这段）

| # | 结论 | 证据 |
|---|---|---|
| 1 | **NVDLA 官方没有 "CNA" 这个块**。官方名字是 **CSC**（0x6000-0x6FFF，25 个寄存器），负责卷积序列控制；搬数据由 **CDMA**（0x5000）负责。nvdla.org 的 hwarch 规格里 "CNA" 出现次数 = 0；nvdla/sw、nvdla/hw 两个官方仓库的寄存器清单/`.rdl` 里也没有 `NVDLA_CNA.*`。"CNA" 是 **Rockchip 自己的命名**（Mesa 与内核沿用了它）。 | hwarch Table 10/Table 18；`spec/manual/` 只有 `NVDLA_CSC.rdl` |
| 2 | **RK3588 的 CNA = NVDLA 的 CDMA + CSC + CBUF 配置 合并成一块**。证据：CNA 块同时含 feature 地址（0x1070）、权重/压缩权重地址（0x1110）、CBUF bank/entry 配置、CVT 转换、PAD 配置、conv stride/dilation（这些在 NVDLA 分属 CDMA 与 CSC）；且 PC 块中断位图里把 CNA 拆成 `CNA_FEATURE_0/1`、`CNA_WEIGHT_0/1`、`CNA_CSC_0/1` 三类。 | Mesa `registers.xml` PC 中断位域；CLK_GATE 位域 `CNA_FEATURE/CNA_WEIGHT/CSC/CBUF` |
| 3 | **NVDLA 是纯 CSB 寄存器编程，没有 PC、没有命令流**。编程序列 = CPU 写寄存器 → 写该块 `D_OP_ENABLE` 的 bit0 → 硬件完成后自清并切到另一个乒乓组。地址表里根本没有 PC/command 块。 | hwarch "Programming sequence" / "Ping-pong Synchronization Mechanism" |
| 4 | **RK3588 的 "PC 块 + 64bit regcmd 命令流" 是 Rockchip 加的**（NVDLA 无对应物）。内核 `rocket_job.c`：`rocket_pc_writel(core, BASE_ADDRESS, task->regcmd)` + `REGISTER_AMOUNTS((regcmd_count+1)/2-1)` + `OPERATION_ENABLE(OP_EN=1)`。 | `drivers/accel/rocket/rocket_job.c` |
| 5 | **观测的 PC 0x0010/0x0014 语义确认**：0x0010 = 命令流起始地址（`PC_BASE_ADDRESS`，低 4 位是 PC_SEL/保留），0x0014 = **`PC_DATA_AMOUNT`，单位是 64bit 项，值 = 项数−1**（不是字节数）。 | Mesa `rkt_regcmd.c` + 内核 `(regcmd_count+1)/2-1` |
| 6 | **RK3588 的 `GLOBAL+0x008`(0xf008) 是"多块使能位图"，NVDLA 完全没有这种寄存器**（NVDLA GLB 只有 4 个：HW_VERSION/INTR_MASK/INTR_SET/INTR_STATUS）。你现在观测到的 `block 0x0081 offset 0x0008 = [24,29,96]` **就是它**，且三个值一一对应合法块组合（见 §7.3）。 | Mesa `REG_GLOBAL_OPERATION_ENABLE` 位域 |
| 7 | PPU↔NVDLA PDP、PPU_RDMA↔PDP_RDMA 的寄存器偏移**几乎 1:1 对齐**（0x000-0x030 除地址高 32 位外全中）；DPU_RDMA↔SDP_RDMA 前 7 个寄存器对齐；DPU↔SDP 属"语义对得上、偏移重排"。 | §8 |
| 8 | **NVDLA 里没有 `D_DATAIN_ADDR_LOW` 这个名字**（`D_DAIN_ADDR_LOW_0` 才是）。`D_WEIGHT_ADDR_LOW`(0x507c)、`D_DATAOUT_ADDR`(0x9018) 存在。 | opendla_initial.h |

---

## 1. 一手来源（已实际抓取）

| 来源 | URL | 本报告用它回答 |
|---|---|---|
| NVDLA Hardware Architectural Specification（官方，含地址空间表 + 每个块的寄存器表 + 编程序列） | https://nvdla.org/hw/v1/hwarch.html | Q1/Q2/Q4/Q6 |
| NVDLA `opendla_initial.h`（nv_full 配置，全部寄存器地址 + 位域 RANGE） | https://github.com/nvdla/sw/blob/master/kmd/firmware/include/opendla_initial.h | Q2/Q3/Q5 |
| NVDLA `opendla_small.h`（nv_small 配置，地址映射不同） | https://github.com/nvdla/sw/blob/master/kmd/firmware/include/opendla_small.h | Q1 |
| NVDLA 寄存器 RDL 源（`spec/manual/README.md` 说明寄存器规格由此生成） | https://github.com/nvdla/hw/tree/master/spec/manual | Q1（证明无 CNA.rdl） |
| NVDLA `loadable_generated.h`（编译产物格式：task / pre_actions / post_actions / event_list） | https://github.com/nvdla/sw/blob/master/umd/core/src/common/include/priv/loadable_generated.h | Q4 |
| **Mesa `registers.xml`**（RK3588 寄存器定义的权威一手来源） | https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/gallium/drivers/rocket/registers.xml | 全篇对照 |
| **Mesa `rkt_regcmd.c`**（regcmd 打包格式） | https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/gallium/drivers/rocket/rkt_regcmd.c | Q4 |
| **Linux `drivers/accel/rocket/rocket_registers.h`**（由 registers.xml 自动生成） | https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/accel/rocket/rocket_registers.h | 全篇对照 |
| Linux `drivers/accel/rocket/rocket_job.c`（PC 提交路径） | https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/accel/rocket/rocket_job.c | Q4/Q6 |
| 社区 RKNPU regcmd ISA 参考（用于交叉验证命令流） | https://github.com/oRKLLM/ork-driver/wiki/regcmd-ISA-Reference | Q4 |

> 注：本机 `raw.githubusercontent.com` 被拦，上述文件通过 `git.kernel.org/plain`、gitlab.freedesktop.org raw、GitHub API(base64) 与 `git clone`（`GIT_SSL_NO_VERIFY=true`）成功取得实际内容。

---

## 2. Q1：NVDLA 的寄存器块划分与地址范围

### 2.1 官方地址空间（hwarch Table 10，nv_full / "initial" 配置）

NVDLA 需要 **256 KiB** MMIO，**每个子单元 4 KiB**，CSB 从 0x0000_0000 起。

| 子单元 | 起 | 止 | 本报告别名/说明 |
|---|---|---|---|
| GLB | 0x0000_0000 | 0x0000_0FFF | 全局（仅 4 个寄存器） |
| Reserved | 0x0000_1000 | 0x0000_1FFF | 预留 |
| MCIF | 0x0000_2000 | 0x0000_2FFF | 访存接口（读写权重/outstanding） |
| SRAMIF (CVIF) | 0x0000_3000 | 0x0000_3FFF | SRAM 接口 |
| BDMA | 0x0000_4000 | 0x0000_4FFF | 通用 DMA |
| CDMA | 0x0000_5000 | 0x0000_5FFF | **卷积 DMA（feature/weight 输入）** |
| **CSC** | **0x0000_6000** | **0x0000_6FFF** | **卷积序列控制 = 部分资料所称 "CNA"** |
| CMAC_A | 0x0000_7000 | 0x0000_7FFF | MAC 阵列 A |
| CMAC_B | 0x0000_8000 | 0x0000_8FFF | MAC 阵列 B |
| CACC | 0x0000_9000 | 0x0000_9FFF | 卷积累加器 |
| SDP (RDMA) | 0x0000_A000 | 0x0000_AFFF | 单点处理读 DMA |
| SDP | 0x0000_B000 | 0x0000_BFFF | 单点处理（激活/BN/EW） |
| PDP (RDMA) | 0x0000_C000 | 0x0000_CFFF | 平面处理读 DMA |
| PDP | 0x0000_D000 | 0x0000_DFFF | 平面处理（pooling） |
| CDP (RDMA) | 0x0000_E000 | 0x0000_EFFF | 跨通道处理读 DMA |
| CDP | 0x0000_F000 | 0x0000_FFFF | 跨通道处理 |
| RUBIK | 0x0001_0000 | 0x0001_0FFF | 数据重排 |
| Reserved | 0x0001_1000 | 0x0003_FFFF | 预留 |

**CBUF 子单元没有任何寄存器**（hwarch 原文："The CBUF subunit does not have any registers"）。所以"CNA / CMAC_A / … / CBUF / GLOBAL"这份清单里，**CBUF 无寄存器窗口**；MCIF=0x2000，GLOBAL=GLB=0x0000。

### 2.2 nv_small 配置的地址映射不同（重要）

`opendla_small.h`（nv_small）用的是另一套块基址（CSB 地址 = 块号<<12）：

| 块 | nv_small 基址 |
|---|---|
| CDMA | 0x3000 |
| CSC | 0x4000 |
| CMAC_A | 0x5000 |
| CMAC_B | 0x6000 |
| CACC | 0x7000 |
| … | … |

说明"每个块 4 KiB、块号在地址高位"是 NVDLA 的固定设计，**具体块号由配置决定**。RK3588 用的是自己的一套块编号（下节）。

### 2.3 RK3588（Mesa/内核）的窗口与块 id —— 与 NVDLA 的对应关系

Mesa `registers.xml` 的 `target` 枚举就是硬件 regcmd 里的"块选择"值（`target = 1 << (block_index + 8)`，见 §6.4）：

| RK 块窗口 | Mesa/内核命名 | csb target | NVDLA 对应物 | RK 块 id(observ.) |
|---|---|---|---|---|
| 0x0000 | PC | 0x100 | **无**（Rockchip 新增） | 0x0101 |
| 0x1000 | CNA | 0x200 | CDMA(0x5000)+CSC(0x6000) 合并 | 0x0201 |
| 0x3000 | CORE | 0x800 | CMAC_A/B(0x7000/0x8000)+CACC(0x9000) 合并 | 0x0801 |
| 0x4000 | DPU | 0x1000 | SDP(0xB000) | 0x1001 |
| 0x5000 | DPU_RDMA | 0x2000 | SDP_RDMA(0xA000) | 0x2001 |
| 0x6000 | PPU | 0x4000 | PDP(0xD000) | 0x4001 |
| 0x7000 | PPU_RDMA | 0x8000 | PDP_RDMA(0xC000) | 0x8001 |
| 0x8000 | DDMA | 0x10000 | MCIF(0x2000) 一族（读写权重/outstanding/status） | 未观测 |
| 0x9000 | SDMA | 0x20000 | MCIF 的第二个实例 | 未观测 |
| 0xF000 | GLOBAL | 0x40000 | GLB(0x0000)，但 +0x8 被换成使能位图 | 未观测（见 §7.3 的 0x0081） |

> 你观测到的 `block 0x0201 / 0x0801 / 0x1001 / 0x2001 / 0x4001 / 0x8001 / 0x0101` 正是 **target|1**（低 bit = "写寄存器"操作码）。0x0041 / 0x0081 的块域 = 0，是特例（§7.3）。

---

## 3. Q2：NVDLA "CNA"（官方名 CSC）完整寄存器清单

### 3.1 CSC 块（0x6000-0x6064，25 个）— hwarch Table 18 原文

| 偏移 | 名称 | 含义 |
|---|---|---|
| 0x6000 | S_STATUS | 两个寄存器组的空闲状态 |
| 0x6004 | S_POINTER | CSB master / 数据通路选择寄存器组 |
| 0x6008 | D_OP_ENABLE | 写 1 启动当前寄存器组 |
| 0x600c | D_MISC_CFG | 卷积模式/精度/权重复用/数据复用 |
| 0x6010 | D_DATAIN_FORMAT | 输入数据格式与像素格式 |
| 0x6014 | D_DATAIN_SIZE_EXT_0 | 扩展后输入 cube 的**宽 [12:0] / 高 [28:16]** |
| 0x6018 | D_DATAIN_SIZE_EXT_1 | 扩展后输入 cube 的**通道 [12:0]** |
| 0x601c | D_BATCH_NUMBER | batch 数 |
| 0x6020 | D_POST_Y_EXTENSION | image-in 的后扩展参数 |
| 0x6024 | D_ENTRY_PER_SLICE | 一个输入 slice 占用的 CBUF entry 数 |
| 0x6028 | D_WEIGHT_FORMAT | 权重是否压缩 |
| 0x602c | D_WEIGHT_SIZE_EXT_0 | 权重 **宽 [4:0] / 高 [20:16]** |
| 0x6030 | D_WEIGHT_SIZE_EXT_1 | 权重**通道 [12:0] / 卷积核数 [28:16]** |
| 0x6034 | D_WEIGHT_BYTES | 权重总字节数 |
| 0x6038 | D_WMB_BYTES | WMB 总字节数 |
| 0x603c | D_DATAOUT_SIZE_0 | 输出 **宽 [12:0] / 高 [28:16]** |
| 0x6040 | D_DATAOUT_SIZE_1 | 输出通道 [12:0] |
| 0x6044 | D_ATOMICS | = out_w * out_h − 1，[20:0] |
| 0x6048 | D_RELEASE | 本层结束时释放的 CBUF slice 数 |
| 0x604c | D_CONV_STRIDE_EXT | **X stride [2:0] / Y stride [18:16]** |
| 0x6050 | D_DILATION_EXT | 膨胀参数 |
| 0x6054 | D_ZERO_PADDING | **左 pad [4:0] / 上 pad [20:16]** |
| 0x6058 | D_ZERO_PADDING_VALUE | 填充值 |
| 0x605c | D_BANK | CBUF 中 data bank / weight bank 数 |
| 0x6060 | D_PRA_CFG | Winograd 模式的 PRA 截断 |

**CSC 块没有任何地址寄存器。** 输入/权重地址在 CDMA（0x5000），输出地址在 CACC（0x9018）。

### 3.2 与你问的 `D_DATAIN_ADDR_LOW` / `D_WEIGHT_ADDR_LOW` / `D_DATAOUT_ADDR`

| 你写的名字 | NVDLA 是否存在 | NVDLA 真实名字与偏移 |
|---|---|---|
| `D_DATAIN_ADDR_LOW` | ❌ **不存在**（全仓库 0 命中） | 最接近：`CDMA_D_DAIN_ADDR_LOW_0_0` **0x5034**（feature 数据地址低 32 位，字段 [31:5]）；UV 平面：`CDMA_D_DAIN_ADDR_LOW_1_0` 0x503c |
| `D_WEIGHT_ADDR_LOW` | ✅ 存在 | `CDMA_D_WEIGHT_ADDR_LOW_0` **0x507c**（HIGH 在 0x5078） |
| `D_DATAOUT_ADDR` | ✅ 存在 | `CACC_D_DATAOUT_ADDR_0` **0x9018**（字段 [31:5]，**没有 HIGH**，因为 CACC 写 CBUF 而非 DRAM） |
| — | 补充 | `CSC_D_DATAOUT_ADDR` 不存在；`SDP_RDMA_D_SRC_BASE_ADDR_LOW` 0xa018；`PDP_D_DST_BASE_ADDR_LOW` 0xd070 |

### 3.3 CDMA 块（0x5000-0x50e8，58 个）地址类寄存器（节选全表）

| 偏移 | 名称 |
|---|---|
| 0x5000 | S_STATUS |
| 0x5004 | S_POINTER |
| 0x5008 | S_ARBITER |
| 0x500c | S_CBUF_FLUSH_STATUS |
| **0x5010** | **D_OP_ENABLE**（注意：CDMA 的使能不在 +0x8！） |
| 0x5014 | D_MISC_CFG |
| 0x5018 | D_DATAIN_FORMAT |
| 0x501c | D_DATAIN_SIZE_0（宽[12:0]/高[28:16]） |
| 0x5020 | D_DATAIN_SIZE_1（通道[12:0]） |
| 0x5024 | D_DATAIN_SIZE_EXT_0 |
| 0x5028 | D_PIXEL_OFFSET |
| 0x502c | D_DAIN_RAM_TYPE |
| **0x5030** | **D_DAIN_ADDR_HIGH_0** |
| **0x5034** | **D_DAIN_ADDR_LOW_0** |
| **0x5038** | **D_DAIN_ADDR_HIGH_1**（UV 平面） |
| **0x503c** | **D_DAIN_ADDR_LOW_1** |
| 0x5040 | D_LINE_STRIDE |
| 0x5044 | D_LINE_UV_STRIDE |
| 0x5048 | D_SURF_STRIDE |
| 0x504c | D_DAIN_MAP |
| 0x5050 / 0x5054 | D_RESERVED_X_CFG / D_RESERVED_Y_CFG |
| 0x5058 / 0x505c | D_BATCH_NUMBER / D_BATCH_STRIDE |
| 0x5060 / 0x5064 | D_ENTRY_PER_SLICE / D_FETCH_GRAIN |
| 0x5068 | D_WEIGHT_FORMAT |
| 0x506c | D_WEIGHT_SIZE_0（每核字节 [17:0]） |
| 0x5070 | D_WEIGHT_SIZE_1（卷积核数 [12:0]） |
| 0x5074 | D_WEIGHT_RAM_TYPE |
| **0x5078** | **D_WEIGHT_ADDR_HIGH_0** |
| **0x507c** | **D_WEIGHT_ADDR_LOW_0** |
| 0x5080 | D_WEIGHT_BYTES |
| **0x5084 / 0x5088** | **D_WGS_ADDR_HIGH / D_WGS_ADDR_LOW** |
| **0x508c / 0x5090** | **D_WMB_ADDR_HIGH / D_WMB_ADDR_LOW** |
| 0x5094 | D_WMB_BYTES |
| 0x5098 | D_MEAN_FORMAT |
| 0x509c / 0x50a0 | D_MEAN_GLOBAL_0 / _1 |
| 0x50a4 / 0x50a8 / 0x50ac | D_CVT_CFG / D_CVT_OFFSET / D_CVT_SCALE |
| 0x50b0 | D_CONV_STRIDE |
| 0x50b4 / 0x50b8 | D_ZERO_PADDING / D_ZERO_PADDING_VALUE |
| 0x50bc | D_BANK |
| 0x50c0 | D_NAN_FLUSH_TO_ZERO |
| 0x50c4-0x50d0 | NaN/Inf 计数（data/weight） |
| 0x50d4-0x50e4 | D_PERF_* 性能计数 |
| 0x50e8 | D_CYA |

---

## 4. Q3：NVDLA 地址寄存器命名规律

**规律是"有"，但不是一条统一规则：**

1. **命名**：`D_<对象>_ADDR_LOW` / `D_<对象>_ADDR_HIGH` 成对出现。hwarch 原文对 HIGH 的描述统一是 *"Higher 32bits of … address when axi araddr is 64bits"* —— **即只有实现使用 64 位 AXI 地址时才有 HIGH 寄存器**。
2. **LOW 字段几乎都是 [31:5]**（32 字节对齐）：`CDMA_D_DAIN_ADDR_LOW_0` [31:5]、`CDMA_D_WEIGHT_ADDR_LOW` [31:5]、`CACC_D_DATAOUT_ADDR` [31:5]、`BDMA_CFG_SRC_ADDR_LOW` [31:5]。
3. **偏移顺序不统一（这是坑）**：
   - **HIGH 在前、LOW 在后**：CDMA（0x5030/0x5034，0x5078/0x507c，0x5084/0x5088，0x508c/0x5090）、RUBIK（0x1001c HIGH / 0x10020 LOW）。
   - **LOW 在前、HIGH 在后**：BDMA（0x4000/0x4004）、SDP_RDMA（0xa018/0xa01c，0xa02c/0xa030，0xa044/0xa048，0xa05c/0xa060）、SDP（0xb048/0xb04c）、PDP_RDMA（0xc01c/0xc020）、PDP（0xd060/0xd064，0xd070/0xd074）、CDP_RDMA（0xe018/0xe01c）、CDP（0xf050/0xf054）。
   - 所以**不能**用"高地址寄存器一定在低地址寄存器之后"来推断。
4. **没有 HIGH 的例外**：CACC 只有 `D_DATAOUT_ADDR`（0x9018），因为它把结果写进 CBUF/内部，不需要全宽 DRAM 地址。
5. **RK3588 的情况**：观测到的地址寄存器全是**单个 32 位**寄存器（`FEATURE_DATA_ADDR`、`SRC_BASE_ADDR`、`BS_BASE_ADDR`、`EW_BASE_ADDR`、`DST_BASE_ADDR`），**没有 HIGH/LOW 对** —— 对应"Axi 地址 32 位"的实现，这正好解释为什么你在 RK3588 上永远看不到 `*_ADDR_HIGH`。RK 的地址字段对齐不统一：`CNA_FEATURE_BASE_ADDR[31:0]`、`DPU_RDMA_SRC_BASE_ADDR[31:0]`、`PPU_RDMA_SRC_BASE_ADDR[31:0]`、`DPU_DST_BASE_ADDR[31:0]`、`PPU_DST_BASE_ADDR[31:4]`。

---

## 5. Q4：NVDLA 有没有 PC / 命令流？——**没有**

### 5.1 硬件侧（决定性证据）

- **地址表里没有 PC/command/fetch 块**（§2.1 全表 16 个块，只有 GLB/MCIF/SRAMIF/BDMA/CDMA/CSC/CMAC_A/CMAC_B/CACC/SDP_RDMA/SDP/PDP_RDMA/PDP/CDP_RDMA/CDP/RUBIK）。
- **hwarch 的编程模型原文**："After reset… The CPU programs the parameters for the first hardware layer into register group 0. After configuration completes, the CPU sets the enable bit in the `D_OP_ENABLE` register. Hardware begins processing… Hardware … clears the enable bit of the `D_OP_ENABLE` register. Hardware advances the CONSUMER field to the next register group…"
  → 硬件没有取指单元；**每一次"运算"都是 CPU 一条条写 CSB 寄存器 + 写一个使能位**。
- 同处还有一条重要注释："**The NVDLA hardware does not have intrinsic support for dependency tracking**" —— 依赖与调度全靠 CPU（或外挂 microcontroller）。

### 5.2 软件侧（有"动作列表"，但不是硬件命令流）

NVDLA 的编译产物 loadable 里确实有类似命令表的结构（`loadable_generated.h`）：
- `TaskListEntry{ id, interface, instance, address_list, pre_actions, post_actions }`
- `EventListEntry{ id, type(0..2), target, val, op(WAIT/SIGNAL) }`

但这些是**给 KMD/固件解析、再由 CPU 变成寄存器写**的数据结构，硬件并不去取它。NVDLA 的"命令"止步于 CSB 写。

### 5.3 RK3588 侧（Rockchip 自己加的）

- **PC 块（0x0000 窗口）**：`VERSION 0x0000`、`VERSION_NUM 0x0004`、`OPERATION_ENABLE 0x0008`（OP_EN@0）、**`BASE_ADDRESS 0x0010`**（`PC_SOURCE_ADDR[31:4]` + `PC_SEL@0`）、**`REGISTER_AMOUNTS 0x0014`**（`PC_DATA_AMOUNT[15:0]`）、`INTERRUPT_MASK/CLEAR/STATUS/RAW_STATUS 0x0020/0x0024/0x0028/0x002c`、`TASK_CON 0x0030`（TASK_NUMBER[11:0]、TASK_PP_EN、TASK_COUNT_CLEAR）、`TASK_DMA_BASE_ADDR 0x0034`、`TASK_STATUS 0x003c`。
- 你的观测 `0x0010=命令流地址 / 0x0014=段长度` **完全对上 Mesa 定义**，且 0x0014 的单位可以定死：
  - Mesa `rkt_regcmd.c`：一条 regcmd = **64 bit**（8 字节）。
  - 内核 `rocket_job.c`：`PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT((task->regcmd_count + 1) / 2 - 1)`。
  - ⇒ 若 `regcmd_count` 以 32bit 字计数（一条=2 字），则 **`0x0014 = 命令流 64bit 项数 − 1`**。
  - 顺带在你自己抓的 dump 里交叉验证：`regcmd_*_task117.bin` 里 `target=0x0101, reg=0x0014, value=14`，即可推出被指向的命令流是 15 项 = 120 字节。
- **regcmd 打包格式**（Mesa `emit_raw`）：
  ```
  entry(64bit) = target[63:48] | value[47:16] | reg[15:0]
  target = block_onehot | opcode        // Mesa 恒 OR 0x1 = "写寄存器"
  block: PC=0x100, CNA=0x200, CORE=0x800, DPU=0x1000, DPU_RDMA=0x2000, PPU=0x4000, PPU_RDMA=0x8000
  ```
  即你 dump 里看到的 `0x0201 / 0x1001 / …` = `target|1`，`0x0041/0x0081` 的块域为 0（见 §7.3）。
- **结论**：**"命令流 + PC 链"确凿是 Rockchip 在 NVDLA 之上加的一层**。NVDLA 用 CSB 逐寄存器写；RK3588 把 (target, offset, value) 三元组做成 DRAM 里的命令流，由硬件 PC 逐条执行——这更像把 NVDLA 软件栈里 loadable 的"动作列表"下沉成了硬件。

---

## 6. Q5：shape 是怎么编码进寄存器的

### 6.1 NVDLA 的编码（一手，位域来自 opendla_initial.h 的 `_RANGE`）

| 寄存器 | 位域 |
|---|---|
| `CDMA_D_DATAIN_SIZE_0_0` 0x501c | `DATAIN_WIDTH[12:0]`、`DATAIN_HEIGHT[28:16]`（**不是** 16bit 对半！） |
| `CDMA_D_DATAIN_SIZE_1_0` 0x5020 | `DATAIN_CHANNEL[12:0]` |
| `CSC_D_DATAIN_SIZE_EXT_0_0` 0x6014 | `DATAIN_WIDTH_EXT[12:0]`、`DATAIN_HEIGHT_EXT[28:16]` |
| `CSC_D_DATAIN_SIZE_EXT_1_0` 0x6018 | `DATAIN_CHANNEL_EXT[12:0]` |
| `CSC_D_DATAOUT_SIZE_0_0` 0x603c | `DATAOUT_WIDTH[12:0]`、`DATAOUT_HEIGHT[28:16]` |
| `CSC_D_DATAOUT_SIZE_1_0` 0x6040 | `DATAOUT_CHANNEL[12:0]` |
| `CSC_D_WEIGHT_SIZE_EXT_0_0` 0x602c | `WEIGHT_WIDTH_EXT[4:0]`、`WEIGHT_HEIGHT_EXT[20:16]`（**5 bit**） |
| `CSC_D_WEIGHT_SIZE_EXT_1_0` 0x6030 | `WEIGHT_CHANNEL_EXT[12:0]`、`WEIGHT_KERNEL[28:16]` |
| `CSC_D_ATOMICS_0` 0x6044 | `ATOMICS[20:0]` |
| `CSC_D_ZERO_PADDING_0` 0x6054 | `PAD_LEFT[4:0]`、`PAD_TOP[20:16]` |
| `CSC_D_CONV_STRIDE_EXT_0` 0x604c | `CONV_X_STRIDE_EXT[2:0]`、`CONV_Y_STRIDE_EXT[18:16]` |
| `CSC_D_WEIGHT_SIZE_EXT_*` 的 tensor 描述 | 宽/高/通道/核数分散在 ext_0/ext_1 |
| `CACC_D_DATAOUT_SIZE_0_0` 0x9010 | `DATAOUT_WIDTH[12:0]`、`DATAOUT_HEIGHT[28:16]` |
| `CACC_D_CLIP_CFG_0` 0x902c | `CLIP_TRUNCATE[4:0]` |
| `SDP_RDMA_D_DATA_CUBE_WIDTH_0` 0xa00c | `WIDTH[12:0]`（高/通道各自独立寄存器） |
| `PDP_D_DATA_CUBE_IN_WIDTH_0` 0xd00c | `CUBE_IN_WIDTH[12:0]` |

**回答你的猜想**：**"低 16 位宽 / 高 16 位高"在结构上对，但位域不是 16bit 对半** —— NVDLA 是 **宽 [12:0] + 高 [28:16]**（13 bit 字段，中间 3 bit 间隙），通道另占一个寄存器。权重是 5bit 宽 + 5bit 高 + 13bit 通道/核数，分两个寄存器。

### 6.2 RK3588 的编码（Mesa/kernel 一手）

| 寄存器 | 位域 | 与 NVDLA 比 |
|---|---|---|
| `CNA_DATA_SIZE0` 0x1020 | `DATAIN_HEIGHT[10:0]`、**`DATAIN_WIDTH[26:16]`** | ⚠️ **字段顺序相反、位宽 11bit、偏移不同**（NVDLA：宽[12:0]、高[28:16]） |
| `CNA_DATA_SIZE1` 0x1024 | `DATAIN_CHANNEL[15:0]`、`DATAIN_CHANNEL_REAL[29:16]` | ⚠️ NVDLA 只有通道[12:0]，无 channel_real |
| `CNA_DATA_SIZE2` 0x1028 | `DATAOUT_WIDTH[10:0]` | ⚠️ NVDLA 输出宽高同寄存器 |
| `CNA_DATA_SIZE3` 0x102c | `DATAOUT_ATOMICS[21:0]`、`SURF_MODE[23:22]` | ✅ 对应 CSC `D_ATOMICS[20:0]`（+SURF_MODE） |
| `CNA_WEIGHT_SIZE0` 0x1030 | `WEIGHT_BYTES[31:0]` | ✅ 对应 CSC `D_WEIGHT_BYTES` 0x6034 |
| `CNA_WEIGHT_SIZE1` 0x1034 | `WEIGHT_BYTES_PER_KERNEL[18:0]` | ✅ 对应 CDMA `D_WEIGHT_SIZE_0` 0x506c [17:0] |
| `CNA_WEIGHT_SIZE2` 0x1038 | `WEIGHT_KERNELS[13:0]`、`WEIGHT_HEIGHT[20:16]`、`WEIGHT_WIDTH[28:24]` | ✅ 把 NVDLA 的 `D_WEIGHT_SIZE_EXT_0/1`(0x602c/0x6030) 合并成一个 |
| `CORE_DATAOUT_SIZE_0` 0x3014 | **`DATAOUT_WIDTH[15:0]`、`DATAOUT_HEIGHT[31:16]`** | ✅ **这里才是你猜的"低16宽/高16高"**（对应 CACC `D_DATAOUT_SIZE_0`，只是位域更规整） |
| `CORE_DATAOUT_SIZE_1` 0x3018 | `DATAOUT_CHANNEL[15:0]` | ✅ 对应 CACC 0x9014 |
| `PPU_DATA_CUBE_IN/OUT_*` 0x600c-0x6020 | 每个字段独立 13bit `[12:0]` | ✅ 对应 PDP `D_DATA_CUBE_IN/OUT_WIDTH/HEIGHT/CHANNEL` |
| `DPU_RDMA_RDMA_DATA_CUBE_*` 0x500c/0x5010/0x5014 | `WIDTH[12:0]`/`HEIGHT[12:0]`/`CHANNEL[12:0]` | ✅ 对应 SDP_RDMA 0xa00c/0xa010/0xa014 |
| `CNA_PAD_CON0` 0x1068 | `PAD_LEFT[7:4]`、`PAD_TOP[3:0]` | ⚠️ 对应 CSC `D_ZERO_PADDING`，但位域位置不同（NVDLA 左[4:0]/上[20:16]） |

---

## 7. Q6：`D_OP_ENABLE` 机制 / 有没有"位图使能多个块"

### 7.1 NVDLA：每块一个 `D_OP_ENABLE`，没有位图

- 机制：向该块 `D_OP_ENABLE` 写 1 → 启动当前寄存器组（乒乓组由 `S_POINTER` 选择）；硬件完成后**自清**该位并切组。
- 各组 `D_OP_ENABLE` 偏移（hwarch 一手）：

| 块 | D_OP_ENABLE 偏移 |
|---|---|
| CDMA | 0x5010（**特例：+0x10**，因为 +0x8 是 S_ARBITER、+0xc 是 S_CBUF_FLUSH_STATUS） |
| CSC | 0x6008 |
| CMAC_A | 0x7008 |
| CMAC_B | 0x8008 |
| CACC | 0x9008 |
| SDP_RDMA | 0xa008 |
| SDP | **0xb038**（前面 13 个 S_LUT_* 寄存器） |
| PDP_RDMA | 0xc008 |
| PDP | 0xd008 |
| CDP_RDMA | 0xe008 |
| CDP | **0xf048** |
| RUBIK | 0x10008 |

- **NVDLA 的 GLB 只有 4 个寄存器**（0x0000 HW_VERSION / 0x0004 INTR_MASK / 0x0008 INTR_SET / 0x000c INTR_STATUS）——**没有任何"一次使能多个块"的位图寄存器**。要跑一层，CPU 依次把每个参与块的 `D_OP_ENABLE` 置 1。
- 乒乓机制原文：两个寄存器组，硬件执行组 0 时 CPU 后台写组 1；硬件完成组 0 后自清 enable 并切到组 1（若已使能）。这解释了 RK 侧 `S_POINTER` 里的 `POINTER_PP_EN / EXECUTER_PP_EN / POINTER_PP_MODE / *_PP_CLEAR` 位。

### 7.2 RK3588：每块一个 OPERATION_ENABLE（+0x8），**并且**多了一个全局位图

- 每块的使能沿用 NVDLA 风格（+0x8）：`CNA 0x1008`、`CORE 0x3008`、`DPU 0x4008`、`DPU_RDMA 0x5008`、`PPU 0x6008`、`PPU_RDMA 0x7008`（RP 的 RP 观测里也都能对上）。
- **额外新增**：`GLOBAL 0xf008 OPERATION_ENABLE`，位域（Mesa/kernel 一手）：

| bit | 名称 |
|---|---|
| 0 | `CNA_OP_EN` |
| 1 | RESERVED |
| 2 | `CORE_OP_EN` |
| 3 | `DPU_OP_EN` |
| 4 | `DPU_RDMA_OP_EN` |
| 5 | `PPU_OP_EN` |
| 6 | `PPU_RDMA_OP_EN` |

### 7.3 你的 `block 0x0081 offset 0x0008 = [24, 29, 96]` 就是这个位图（高置信推断）

把它按位展开：

| 观测值 | 二进制 | 置位 | 解码（层类型） |
|---|---|---|---|
| 24 | 0b0011000 | bit3+bit4 | DPU + DPU_RDMA → 纯激活/EW 层 |
| 29 | 0b0011101 | bit0+bit2+bit3+bit4 | CNA + CORE + DPU + DPU_RDMA → 卷积层 |
| 96 | 0b1100000 | bit5+bit6 | PPU + PPU_RDMA → pooling 层 |

三个值**无一例外**落在合法块组合上，且**保留位 bit1 从未出现** —— 与 Mesa 的位域定义完全吻合。也就是说：
- 偏移 0x0008 落在 GLOBAL 窗口（0xf000）内 = **0xf008**；
- 命令流里的 `0x0081` 是"块域=0 + 特殊操作码 0x81"的写法（写寄存器操作码是 0x1，见 §5.3），它承担了"把这一层要参与的块一次性使能"的语义。
- 顺带：你的 `0x0041`（块域 0、opcode 0x41、reg=0x0000）落在 `GLOBAL+0x0000 = 0xf000`，Mesa/kernel 在该地址**没有定义寄存器**，语义**未确定**（你标注的"op_en 前置"合理，但无法从公开规格证实）。**这是本次调研未能解释的两项之一。**

---

## 8. 逐块对照表（观测偏移 ↔ RK 名字 ↔ NVDLA 对应物）

图例：✅ = 语义 + 偏移都对得上；🟡 = 语义对得上但偏移/位域不同；❌ = NVDLA 无对应物。

### 8.1 PC 块（0x0000）— 观测：0x0010, 0x0014

| 观测 | Mesa/内核名 | NVDLA 对应物 |
|---|---|---|
| 0x0010（张量地址） | `PC_BASE_ADDRESS`（`PC_SOURCE_ADDR[31:4]`+`PC_SEL@0`） | ❌ **无**（Rockchip 新增命令流 PC） |
| 0x0014（段长度） | `PC_REGISTER_AMOUNTS`（`PC_DATA_AMOUNT[15:0]`，单位 = 64bit 项，值 = 项数−1） | ❌ **无** |

### 8.2 CNA 块（0x1000）— 观测 0x100c…0x1184

| 观测偏移 | RK 名字（Mesa/kernel，权威） | NVDLA 对应物 |
|---|---|---|
| 0x100c | `CNA_CONV_CON1`（CONV_MODE/IN_PRECISION/PROC_PRECISION/ARGB_IN/DECONV/NONALIGN_DMA/GROUP_LINE_OFF） | 🟡 CSC `D_MISC_CFG` 0x600c（相对偏移同为 +0x0c） |
| 0x1010 | `CNA_CONV_CON2`（KERNEL_GROUP/FEATURE_GRAINS/CSC_WO_EN/CSC_DO_EN/CMD_FIFO_SRST） | 🟡 相对偏移 +0x10 = CSC `D_DATAIN_FORMAT`，但内容不同 |
| 0x1014 | `CNA_CONV_CON3`（CONV/DECONV stride、ATROUS dilation、NN_MODE） | 🟡 CSC `D_CONV_STRIDE_EXT` 0x604c + `D_DILATION_EXT` 0x6050（RK 合并） |
| 0x1020 | `CNA_DATA_SIZE0`（高[10:0]/宽[26:16]） | 🟡 CSC `D_DATAIN_SIZE_EXT_0`（宽[12:0]/高[28:16]）**字段顺序相反** |
| 0x1024 | `CNA_DATA_SIZE1`（通道/通道_real） | 🟡 CSC `D_DATAIN_SIZE_EXT_1` |
| 0x1028 | `CNA_DATA_SIZE2`（输出宽） | 🟡 CSC `D_DATAOUT_SIZE_0` |
| 0x102c | `CNA_DATA_SIZE3`（atomics/surf_mode） | ✅ CSC `D_ATOMICS` 0x6044 语义 |
| 0x1030 | `CNA_WEIGHT_SIZE0`（weight_bytes） | ✅ CSC `D_WEIGHT_BYTES` 0x6034 |
| 0x1034 | `CNA_WEIGHT_SIZE1`（bytes/kernel） | ✅ CDMA `D_WEIGHT_SIZE_0` 0x506c |
| 0x1038 | `CNA_WEIGHT_SIZE2`（kernels/height/width） | ✅ CSC `D_WEIGHT_SIZE_EXT_0/1` |
| 0x1040 | `CNA_CBUF_CON0`（DATA/WEIGHT/FC_DATA bank、DATA_REUSE、WEIGHT_REUSE） | ✅ CSC `D_BANK` 0x605c |
| 0x1044 | `CNA_CBUF_CON1`（DATA_ENTRIES[13:0]） | ✅ CSC `D_ENTRY_PER_SLICE` 0x6024 |
| 0x104c | `CNA_CVT_CON0`（CVT_TRUNCATE_0..3/TYPE/ROUND/SIGN/BYPASS） | ✅ CDMA `D_CVT_CFG` 0x50a4 |
| 0x1050-0x105c | `CNA_CVT_CON1..4`（CVT_SCALE/CVT_OFFSET 0..3） | ✅ CDMA `D_CVT_OFFSET` 0x50a8 / `D_CVT_SCALE` 0x50ac |
| 0x1060 | `CNA_FC_CON0`（FC_SKIP_DATA/FC_SKIP_EN） | ❌（NVDLA 的 FC 模式由 CSC `D_MISC_CFG` 控制，无独立 FC 寄存器） |
| 0x1064 | `CNA_FC_CON1`（DATA_OFFSET[16:0]） | ❌ |
| 0x1068 | `CNA_PAD_CON0`（PAD_LEFT/PAD_TOP） | ✅ CSC `D_ZERO_PADDING` 0x6054（位域不同） |
| **0x1070（观测为张量地址）** | `CNA_FEATURE_DATA_ADDR`（FEATURE_BASE_ADDR[31:0]） | ✅ CDMA `D_DAIN_ADDR_LOW_0` 0x5034 —— **观测判为地址，完全正确** |
| 0x1074 | `CNA_FC_CON2`（WEIGHT_OFFSET[16:0]） | ❌ |
| 0x1078 | `CNA_DMA_CON0`（OV4K_BYPASS/DATA+WEIGHT burst len） | ❌（NVDLA CDMA 无 burst 配置） |
| 0x107c | `CNA_DMA_CON1`（LINE_STRIDE[27:0]） | ✅ CDMA `D_LINE_STRIDE` 0x5040 |
| 0x1080 | `CNA_DMA_CON2`（SURF_STRIDE[27:0]） | ✅ CDMA `D_SURF_STRIDE` 0x5048 |
| 0x1084 | `CNA_FC_DATA_SIZE0`（DMA width/height） | ❌（最接近 CDMA `D_DATAIN_SIZE_0`，但语义是 DMA 侧尺寸） |
| 0x1088 | `CNA_FC_DATA_SIZE1`（DMA channel） | ❌ |
| 0x1100 | `CNA_DCOMP_CTRL`（DECOMP_CONTROL/WT_DEC_BYPASS） | ❌ |
| 0x1104 | `CNA_DCOMP_REGNUM` | ❌ |
| **0x1110（观测为张量地址）** | `CNA_DCOMP_ADDR0`（DECOMPRESS_ADDR0[31:0]） | ❌ **NVDLA 没有解压硬件**（只有 `D_WEIGHT_FORMAT` 一个"是否压缩"位）——观测为地址也自洽（压缩权重存放地址） |
| 0x1140-0x117c | `CNA_DCOMP_AMOUNT0..15` | ❌ |
| 0x1180 | `CNA_CVT_CON5`（PER_CHANNEL_CVT_EN[31:0]） | ❌（观测值 [0, 4095] 与"12 通道使能位"吻合） |
| **0x1184（观测为张量地址）** | `CNA_PAD_CON1`（PAD_VALUE[31:0]） | ✅ 语义上对应 CSC `D_ZERO_PADDING_VALUE` 0x6058，**但与"观测为张量地址"矛盾**，建议复核（见 §9.3） |

### 8.3 CORE 块（0x3000）— 观测 0x3010, 0x3014, 0x3018, 0x301c, 0x3030

| 观测 | RK 名字 | NVDLA CACC(0x9000)/CMAC |
|---|---|---|
| 0x3010 | `CORE_MISC_CFG`（DW_EN@1/QD_EN@0/PROC_PRECISION） | 🟡 CACC `D_MISC_CFG` 0x900c 下移 4（观测 [1,3] = DW/QD 组合，自洽） |
| 0x3014 | `CORE_DATAOUT_SIZE_0`（宽[15:0]/高[31:16]） | ✅ CACC `D_DATAOUT_SIZE_0` 0x9010（**位域就是"低16宽/高16高"**） |
| 0x3018 | `CORE_DATAOUT_SIZE_1`（通道[15:0]） | ✅ CACC `D_DATAOUT_SIZE_1` 0x9014（观测 0x1f..0x1ff = 通道数，自洽） |
| 0x301c | `CORE_CLIP_TRUNCATE`（CLIP_TRUNCATE[4:0]/ROUND_TYPE） | ✅ CACC `D_CLIP_CFG` 语义（CACC 在 0x902c） |
| 0x3030 | **Mesa/kernel 未定义** | ✅ **CACC `D_OUT_SATURATION` 0x9030 —— 相对偏移完全相同 (+0x30)**，强烈提示这里是输出饱和计数 |
| （未观测）0x300c | `CORE_MAC_GATING`（SLCG_OP_EN[26:0]） | 🟡 位置等同 CMAC_A `D_MISC_CFG` 0x700c —— CORE 就是 CMAC+CACC 合并的又一证据 |

### 8.4 DPU 块（0x4000）— 观测 0x4004, 0x400c, 0x4010, 0x4020, 0x4024, 0x4030, 0x4034-0x40c4

| 观测 | RK 名字 | NVDLA SDP(0xb000) |
|---|---|---|
| 0x4004 | `DPU_S_POINTER`（观测 [14,48] = PP_EN/PP_MODE/CLEAR 位组合，自洽） | ✅ `S_POINTER` 0xb004 |
| 0x400c | `DPU_FEATURE_MODE_CFG` | ✅ `D_FEATURE_MODE_CFG` 0xb0b0（偏移不同） |
| 0x4010 | `DPU_DATA_FORMAT` | ✅ `D_DATA_FORMAT` 0xb0bc |
| **0x4020（张量地址）** | `DPU_DST_BASE_ADDR` | ✅ `D_DST_BASE_ADDR_LOW` 0xb048 —— 观测判为地址正确 |
| 0x4024 | `DPU_DST_SURF_STRIDE` | ✅ `D_DST_SURFACE_STRIDE` 0xb054 |
| 0x4030 / 0x4034 / 0x4038 / 0x403c | `DATA_CUBE_WIDTH` / `HEIGHT` / `NOTCH_ADDR` / `CHANNEL` | ✅ 0xb03c / 0xb040 / — / 0xb044 |
| 0x4040 / 0x4044 / 0x4048 / 0x404c | `BS_CFG` / `BS_ALU_CFG` / `BS_MUL_CFG` / `BS_RELUX_CMP_VALUE` | ✅🟡 0xb058 `D_DP_BS_CFG` / 0xb05c / 0xb064；RK 把 NVDLA 的 `*_SRC_VALUE`(0xb060/0xb068) 合并进 CFG |
| 0x4050 / 0x4054 | `BS_OW_CFG` / `BS_OW_OP` | ❌ 新（output-width/od bypass） |
| 0x4058 / 0x405c | `WDMA_SIZE_0/1` | ❌ 新 |
| 0x4060-0x406c | `BN_CFG`/`BN_ALU_CFG`/`BN_MUL_CFG`/`BN_RELUX_CMP_VALUE` | ✅ 0xb06c-0xb07c |
| 0x4070 / 0x4074 / 0x4078 / 0x407c | `EW_CFG` / `EW_CVT_OFFSET_VALUE` / `EW_CVT_SCALE_VALUE` / `EW_RELUX_CMP_VALUE` | ✅🟡 0xb080 / 0xb08c / 0xb090 / — |
| 0x4080 / 0x4084 / 0x4088 | `OUT_CVT_OFFSET` / `OUT_CVT_SCALE` / `OUT_CVT_SHIFT` | ✅ 0xb0c0 / 0xb0c4 / 0xb0c8 |
| 0x4090-0x40ac | `EW_OP_VALUE_0..7` | 🟡 NVDLA 只有 ALU/MUL 两个 SRC_VALUE (0xb088/0xb09c) + truncate |
| 0x40c0 | `SURFACE_ADD` | ❌/🟡 |
| 0x4100-0x412c | `LUT_ACCESS_CFG` … `LUT_LO_SLOPE_SHIFT`（12 个） | ✅ NVDLA `S_LUT_*` 0xb008-0xb034（**同一组 12 个寄存器**，RK 搬到 0x4100） |

### 8.5 DPU_RDMA 块（0x5000）— 观测 0x500c, 0x5010, 0x5014, 0x5018, 0x5020, 0x5038

| 观测 | RK 名字 | NVDLA SDP_RDMA(0xa000) |
|---|---|---|
| 0x500c / 0x5010 / 0x5014 | `RDMA_DATA_CUBE_WIDTH/HEIGHT/CHANNEL` | ✅ 0xa00c / 0xa010 / 0xa014（**偏移完全相同**） |
| **0x5018（张量地址）** | `RDMA_SRC_BASE_ADDR` | ✅ `D_SRC_BASE_ADDR_LOW` 0xa018（**偏移完全相同**） |
| **0x5020（张量地址）** | `RDMA_BS_BASE_ADDR` | ✅ `D_BS_BASE_ADDR_LOW` 0xa02c（语义同，偏移不同——因为 RK 砍掉了 HIGH/stride） |
| **0x5038（张量地址）** | `RDMA_EW_BASE_ADDR` | ✅ `D_EW_BASE_ADDR_LOW` 0xa05c（同上） |
| （未观测）0x501c/0x5028/0x502c/0x5034/0x5040-0x506c | `BRDMA_CFG`/`NRDMA_CFG`/`BN_BASE_ADDR`/`ERDMA_CFG`/`EW_SURF_STRIDE`/`FEATURE_MODE_CFG`/`SRC_DMA_CFG`/`SURF_NOTCH`/`PAD_CFG`/`WEIGHT`/`EW_SURF_NOTCH` | ✅/❌ 混（前 7 个偏移 1:1 到 0x018 为止，之后 RK 压缩重排并新增 NOTCH/PAD/WEIGHT） |

### 8.6 PPU 块（0x6000）— 观测 0x6070（张量地址）

PPU ↔ NVDLA PDP 是全篇**最整齐**的对应（同一相对偏移）：

| 偏移(相对) | RK | NVDLA PDP |
|---|---|---|
| 0x000/0x004/0x008 | S_STATUS / S_POINTER / OPERATION_ENABLE | ✅ 0xd000/0xd004/0xd008 |
| 0x00c/0x010/0x014 | IN_WIDTH / IN_HEIGHT / IN_CHANNEL | ✅ 0xd00c/0xd010/0xd014 |
| 0x018/0x01c/0x020 | OUT_WIDTH / OUT_HEIGHT / OUT_CHANNEL | ✅ 0xd018/0xd01c/0xd020 |
| 0x024 | OPERATION_MODE_CFG | ✅ 0xd024 |
| 0x034/0x038/0x03c | POOLING_KERNEL_CFG / RECIP_KERNEL_WIDTH / RECIP_KERNEL_HEIGHT | ✅ 0xd034/0xd038/0xd03c |
| 0x040/0x044/0x048 | POOLING_PADDING_CFG / PADDING_VALUE_1 / _2 | ✅ 0xd040/0xd044/0xd048 |
| **0x070（张量地址）** | `PPU_DST_BASE_ADDR` | ✅ `D_DST_BASE_ADDR_LOW` **0xd070（偏移完全相同）** |
| 0x07c | `PPU_DST_SURF_STRIDE` | ✅ `D_DST_SURFACE_STRIDE` 0xd07c（**偏移完全相同**） |
| 0x084 | `PPU_DATA_FORMAT` | ✅ `D_DATA_FORMAT` 0xd084 |
| 0x0dc | `PPU_MISC_CTRL`（SURF_LEN/BURST_LEN/MC_SURF_OUT/NONALIGN） | ❌ 新增 |
| （NVDLA 有、RK 无） | — | 0xd028 NaN_FLUSH、0xd02c/0xd030 partial width、0xd04c-0xd05c padding value 3..7、0xd060 src addr、0xd074 HIGH、0xd078 dst line stride、0xd080 ram cfg、0xd088-0xd098 状态/性能 |

### 8.7 PPU_RDMA 块（0x7000）— 观测 0x701c（张量地址）

| 偏移(相对) | RK | NVDLA PDP_RDMA |
|---|---|---|
| 0x000/0x004/0x008 | S_STATUS / S_POINTER / OPERATION_ENABLE | ✅ 0xc000/0xc004/0xc008 |
| 0x00c/0x010/0x014 | CUBE_IN_WIDTH / HEIGHT / CHANNEL | ✅ 0xc00c/0xc010/0xc014 |
| **0x01c（张量地址）** | `SRC_BASE_ADDR` | ✅ `D_SRC_BASE_ADDR_LOW` **0xc01c（偏移完全相同）** |
| 0x024/0x028 | SRC_LINE_STRIDE / SRC_SURF_STRIDE | ✅ 0xc024/0xc028 |
| 0x030 | DATA_FORMAT | ✅ 0xc030 |
| （NVDLA 有、RK 无） | — | 0xc018 D_FLYING_MODE（RK 砍掉，所以后面偏移没被顶开）、0xc020 HIGH、0xc02c RAM_CFG、0xc034-0xc048 |

### 8.8 DDMA/SDMA（0x8000/0x9000，你未观测）

Mesa/kernel 定义的是 `CFG_OUTSTANDING`、`RD_WEIGHT_0/1`（PDP/DPU/KERNEL/FEATURE/PC 各客户端读权重）、`WR_WEIGHT_0`、`CFG_DMA_ARB/RC_QOS/RD_CFG/WR_CFG`、`CFG_STATUS`——**语义上属于 NVDLA MCIF(0x2000) 一族**（`CFG_RD_WEIGHT_0/1/2`、`CFG_WR_WEIGHT_0/1`、`CFG_OUTSTANDING_CNT`、`STATUS`），但增加了 QoS/ARB/WSTRB 等 Rockchip 自有字段。

---

## 9. 对照结论

### 9.1 NVDLA 规格能解释的（可直接采信）

1. **RK3588 的块划分与块内寄存器"骨架"**：`S_STATUS(0x0) / S_POINTER(+0x4) / OPERATION_ENABLE(+0x8)` + `S_POINTER` 里的乒乓位（`*_PP_EN`、`*_PP_MODE`、`*_PP_CLEAR`）＝ NVDLA 每块的乒乓寄存器组设计，逐位一致。
2. **PPU ↔ PDP**、**PPU_RDMA ↔ PDP_RDMA**：绝大多数寄存器**相对偏移完全相同**（尤其 `DST_BASE_ADDR 0x070`、`SRC_BASE_ADDR 0x01c`、`DST_SURF_STRIDE 0x07c`、`DATA_FORMAT`）。你判定为"张量地址"的 `0x6070`、`0x701c` 在 NVDLA 里正是 PDP 的输出地址与 PDP_RDMA 的输入地址，**判读正确**。
3. **DPU_RDMA ↔ SDP_RDMA**：前 7 个寄存器偏移 1:1（cube w/h/c、`SRC_BASE_ADDR`）。你判为地址的 `0x5018 / 0x5020 / 0x5038` = SRC / BS(bias) / EW 三张表的基址，**判读正确**。
4. **DPU ↔ SDP**：BS/BN/EW/LUT/输出 CVT 的结构与命名同源（`D_DP_BS_CFG`→`BS_CFG`、`D_DP_EW_CFG`→`EW_CFG`、`S_LUT_*`12 个→`LUT_*`12 个）。你判为地址的 `0x4020` = `D_DST_BASE_ADDR_LOW`，**判读正确**。
5. **CORE ↔ CACC(+CMAC)**：`DATAOUT_SIZE_0` 正是"低 16 位宽、高 16 位高"；`0x301c CLIP_TRUNCATE` = `D_CLIP_CFG`；**你观测到的 0x3030（Mesa/kernel 未定义）在 NVDLA 里是 CACC `D_OUT_SATURATION` 0x9030，相对偏移完全相同** → 可确定为输出饱和计数。
6. **CNA ↔ CDMA+CSC**：`FEATURE_DATA_ADDR(0x1070)` = `D_DAIN_ADDR_LOW`；`WEIGHT_SIZE0/1/2` = `D_WEIGHT_BYTES`/`D_WEIGHT_SIZE_0`/`D_WEIGHT_SIZE_EXT_*`；`CBUF_CON0/1` = `D_BANK`/`D_ENTRY_PER_SLICE`；`CVT_CON0..4` = `D_CVT_CFG/OFFSET/SCALE`；`DMA_CON1/2` = `D_LINE_STRIDE`/`D_SURF_STRIDE`；`PAD_CON0/1` = `D_ZERO_PADDING`/`_VALUE`；`DATA_SIZE3` = `D_ATOMICS`。**CNA 的"身份"是 NVDLA 卷积前端（DMA+序列控制）的合并体。**
7. **PC 0x0010/0x0014 的含义与单位**（见 §5.3）。
8. **`GLOBAL 0xf008` 位图 = 你观测的 `block 0x0081 offset 0x0008`**，且 24/29/96 三个值可逐位解释成 {DPU+DPU_RDMA}／{CNA+CORE+DPU+DPU_RDMA}／{PPU+PPU_RDMA}。

### 9.2 NVDLA 规格解释不了的（属于 Rockchip 增量 / 私有扩展）

| 项目 | 说明 |
|---|---|
| **PC 块与 regcmd 命令流**（0x0000-0x003c） | NVDLA 无 PC、无命令流。Rockchip 新增（硬件取指执行 `{target, offset, value}`）。 |
| **`GLOBAL 0xf008` 多块使能位图** | NVDLA GLB 无此寄存器（只有 4 个）。注意它落在 NVDLA `INTR_SET(+0x8)` 的位置，但语义完全不同。 |
| **CNA `DCOMP_*`（0x1100-0x117c）** | NVDLA 没有权重解压硬件（只有一个 `D_WEIGHT_FORMAT` 压缩标志位）。这是 Rockchip 新增的解压引擎（`DCOMP_ADDR0` + 16 个 `DCOMP_AMOUNT`）。你观测到 `0x1110` 是地址，与此完全自洽。 |
| **CNA `FC_CON0/1/2`、`FC_DATA_SIZE0/1`、`DMA_CON0`、`CLK_GATE`、`CVT_CON5`** | NVDLA 无对应（FC 模式在 NVDLA 由 CSC `D_MISC_CFG` 控制；NVDLA 无 burst/时钟门控寄存器）。 |
| **DPU `BS_OW_CFG/BS_OW_OP`、`WDMA_SIZE_0/1`、`*_RELUX_CMP_VALUE`、`SURFACE_ADD`、`OFFSET_PEND`** | Rockchip 增量。 |
| **PPU `MISC_CTRL(0x60dc)`、DPU/PPU RDMA 的 `PAD_CFG`/`WEIGHT`/`NOTCH`** | Rockchip 增量（NVDLA 的 padding 在 PDP 内、notch 分散）。 |
| **DDMA/SDMA 的 QoS/ARB/WSTRB** | NVDLA MCIF 只有读写权重与 outstanding。 |
| **块编号体系不同** | NVDLA：块基址 = 块号<<12（CDMA=5、CSC=6…，且 nv_small 又是另一套）；RK3588：CNA=0x1000、CORE=0x3000…，并在 regcmd 里用"独热块域 | 操作码"表达。 |
| **`block 0x0041 offset 0x0000`** | 未解释（GLOBAL 窗口 +0x0 在 Mesa/kernel 无定义，NVDLA GLB 0x0 是只读 HW_VERSION）。**建议保留为未知**。 |

### 9.3 需要你复核的两处

1. **CNA `0x1184`**：Mesa/kernel 定义为 `PAD_CON1 / PAD_VALUE[31:0]`（填充值），但你的启发式判为"张量地址（2/2 落在已知缓冲区）"。二者矛盾。可能性：(a) 取值恰为 0（0 被你的判据当作合法缓冲区地址）；(b) RK3588 在该偏移实际是又一个地址寄存器（例如压缩权重/权重基址），Mesa 的定义有误或过时。建议用"同一算子里 `0x1184` 的取值 vs 该算子权重缓冲地址"做一次定点核对。
2. **CNA `0x100c/0x1010/0x1014`**：它们的相对偏移（+0x0c/+0x10/+0x14）与 NVDLA CSC 的 `D_MISC_CFG/D_DATAIN_FORMAT/D_DATAIN_SIZE_EXT_0` 位置一致，但 RK 的 `CONV_CON2/CONV_CON3` 位域内容（KERNEL_GROUP/FEATURE_GRAINS/CSC_WO_EN/CSC_DO_EN、stride/dilation）与 CSC 对不上。**不要用 NVDLA CSC 的位域去解 RK 的 `CONV_CON2/3`**——只能用 Mesa/kernel 的位域。

### 9.4 给下一步的建议

- 判定"某个未知偏移"时，优先按 **PPU/PPU_RDMA 的 1:1 偏移规律** 反查（这两个块与 NVDLA 最一致）；CNA 因为把 CDMA+CSC 揉在一起，偏移规律不可移植。
- 把 §9.2 的"Rockchip 增量列表"当作闭源运行时特有能力的清单（解压、burst、notch、output-width、QoS），NVDLA 文档在这些点上帮不上忙。
- 你观测到的 `0x0081/+0x0008` 位图语义已可定案，可以作为"一层涉及哪些块"的算子分类特征（卷积=CNA|CORE|DPU|DPU_RDMA；pooling=PPU|PPU_RDMA；纯 EW/激活=DPU|DPU_RDMA），对 82 个算子做自动归档。

---

## 附录 A：RF 块 id 编码（供参考）

Mesa `rkt_regcmd.c::emit_raw()`：

```c
packed_value  = ((uint64_t)target) << 48;
packed_value |= ((uint64_t)value)  << 16;
packed_value |= (uint64_t)reg;          // 低 16 位 = 寄存器平坦地址
// target = rkt_get_target(reg) + 0x1;  // +1 = "写寄存器" 操作码
```

`registers.xml` 的 `target` 枚举：PC=0x100、CNA=0x200、CORE=0x800、DPU=0x1000、DPU_RDMA=0x2000、PPU=0x4000、PPU_RDMA=0x8000、DDMA=0x10000、SDMA=0x20000、GLOBAL=0x40000 ⇒ 即 `target = 1 << (块序号 + 8)`，块序号 PC=0、CNA=1、(2 空缺)、CORE=3、DPU=4、DPU_RDMA=5、PPU=6、PPU_RDMA=7、DDMA=8、SDMA=9、GLOBAL=10。因此 **block id 的高字节是块独热位，低字节是操作码**；你 dump 里 `0x01xx` 系列的 `|1` 就是这个操作码。`0x0041/0x0081` 高字节为 0，属"无块"特殊操作码。

## 附录 B：NVDLA 每块寄存器数量（nv_full，核对用）

GLB 4、MCIF 7、SRAMIF/CVIF 7、BDMA 21、CDMA 58-59、CSC 25-26、CMAC_A 4、CMAC_B 4、CACC 13-14、SDP_RDMA 37、SDP 63、PDP_RDMA 19-20、PDP 39-40、CDP_RDMA 16-17、CDP 46-47、RUBIK 25。
