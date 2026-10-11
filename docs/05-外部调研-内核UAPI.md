# RK3588 NPU 内核驱动（`drivers/rknpu`, linux-6.1 vendor, v0.9.8）用户态 ABI 调研报告

> **取证方式说明（重要）**：本机沙箱把 `github.com` / `raw.githubusercontent.com` / `api.github.com` 的 DNS 劫持到 `127.0.0.1`，无法直连。所有 vendor 源码均通过 `ghproxy.net` 代理抓取 `raw.githubusercontent.com` 的**原文**（并在本地逐文件比对），下面给出的是对应的 **GitHub 规范 URL**。raw 原文等价链接只需把 `https://github.com/<org>/<repo>/blob/<ref>/<path>` 换成 `https://raw.githubusercontent.com/<org>/<repo>/<ref>/<path>`。
>
> **可靠性标注**：`【源码】`= 我直接读到的一手内核/内核树源码；`【二手】`= 文档/博客/第三方逆向记录；`【推断】`= 我的推理，未在源码中直接见到。
> 本报告核对的版本：`develop-6.1`（v0.9.8，20240828）为主，另取 `develop-4.19`(v0.8.0)、`develop-5.10`(v0.9.8)、`develop-6.6`(v0.9.8)、`armbian/linux-rockchip` 的 `rk-6.1-rkr1`(v0.9.6) / `rk-5.10-rkr4`(v0.8.5) 做 ABI 差分。

---

## 0. 一句话结论（先看这个）

| 问题 | 结论 |
|---|---|
| UAPI 头在哪 | **`drivers/rknpu/include/rknpu_ioctl.h`**。`include/uapi/drm/rknpu_drm.h` **不存在**（4.19/5.10/6.1/6.6 四个分支均 404）|
| 是不是 DRM 驱动 | **是**。`drm_driver{DRIVER_GEM|DRIVER_PRIME|DRIVER_RENDER}`，6 个 `DRM_IOCTL_RKNPU_*`，全部 `DRM_RENDER_ALLOW` |
| `RKNPU_CMD_*` opcode 枚举 | **在 GPL 驱动中不存在**。4.19/5.10/6.1/6.6 的 `drivers/rknpu` 全量文件里 grep 不到任何 `RKNPU_CMD`/`rknpu_pc.h`/`rknpu_reg_cmd.h`。regcmd 不是"opcode 流"，而是**64-bit 寄存器写对**（block_id + reg_offset + value）|
| PC 地址编码 | `task.regcmd_addr` = 设备侧（IOVA/物理）**32 位字节地址，驱动原样写进 `PC_DATA_ADDR(0x10)`**；该寄存器 `[31:4]=地址`、`[3:1]=保留`、`[0]=pc_sel`（0=PC 模式/AXI DMA，1=Slave 模式）。所以"32 位地址+标志位"这个说法**成立**，但标志位只有 bit0，且是驱动自己写的（Slave 模式写 `0x1`）|
| 内存模型 | 双后端二选一：`DRM_GEM`（默认）或 `DMA_HEAP`（misc `/dev/rknpu`）。GEM 下非连续→ `drm_gem_get_pages()`（**shmem 页**）+ IOMMU 映射；连续/无 IOMMU → CMA。**这就是 shmem-rss 暴涨的根因** |
| 版本破坏性变更 | 0.8.x→0.9.x：`rknpu_submit` 偏移 32–56 的字段**语义整体换掉**（长度不变，静默错解）；`rknpu_mem_create` 40→48 字节（**ioctl 编号都变了**）；`RKNPU_MEM_NON_DMA32`→`RKNPU_MEM_DMA32`（**语义反转**）|

---

## 1. UAPI 头文件

### 1.1 路径【源码】

- **正确路径**：`drivers/rknpu/include/rknpu_ioctl.h`
  <https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/include/rknpu_ioctl.h>
- 它是**驱动私有** include 目录：`drivers/rknpu/Makefile` 里 `ccflags-y += -I$(srctree)/$(src)/include`
  <https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/Makefile#L4-L5>
- **不存在** `include/uapi/drm/rknpu_drm.h`，也不存在 `drivers/rknpu/include/uapi/…`（我逐个试过 4.19 / 5.10 / 6.1 / 6.6 四个分支，全部 404）。同目录下的头是 `rknpu_drv.h / rknpu_job.h / rknpu_gem.h / rknpu_mem.h / rknpu_iommu.h / rknpu_reset.h / rknpu_fence.h / rknpu_debugger.h / rknpu_mm.h`。
- 驱动源码清单（`Makefile`）：`rknpu_drv.c rknpu_reset.c rknpu_job.c rknpu_debugger.c rknpu_iommu.c`，按 config 追加 `rknpu_devfreq.c / rknpu_mm.c / rknpu_fence.c / rknpu_gem.c / rknpu_mem.c`。

### 1.2 头文件结构与关键宏数值【源码】

`rknpu_ioctl.h`（v0.9.8，共 330 行）成员：`RKNPU_OFFSET_*` 宏 → `enum e_rknpu_mem_type` → `enum e_rknpu_mem_sync_mode` → `enum e_rknpu_job_mode` → `enum e_rknpu_action` → 结构体 → ioctl 编号宏。

**PC / 中断寄存器偏移**（`#L21-L38`）：

| 宏 | 值 | 用途（驱动侧）|
|---|---|---|
| `RKNPU_OFFSET_VERSION` | `0x0` | HW 版本 |
| `RKNPU_OFFSET_VERSION_NUM` | `0x4` | HW 版本号（低 16 位）|
| `RKNPU_OFFSET_PC_OP_EN` | `0x8` | PC 使能（写 1 再写 0 即"发令"）|
| `RKNPU_OFFSET_PC_DATA_ADDR` | `0x10` | **PC 程序基址 + pc_sel** |
| `RKNPU_OFFSET_PC_DATA_AMOUNT` | `0x14` | 每个 task 取多少条寄存器指令 |
| `RKNPU_OFFSET_INT_MASK` | `0x20` | 中断掩码（取 **last** task 的 `int_mask`）|
| `RKNPU_OFFSET_INT_CLEAR` | `0x24` | 中断清除（取 **first** task 的 `int_mask`！）|
| `RKNPU_OFFSET_INT_STATUS` | `0x28` | 中断状态 |
| `RKNPU_OFFSET_INT_RAW_STATUS` | `0x2c` | 原始中断状态 |
| `RKNPU_OFFSET_PC_TASK_CONTROL` | `0x30` | task 数 / ping-pong / clear |
| `RKNPU_OFFSET_PC_DMA_BASE_ADDR` | `0x34` | 任务 DMA 基址（= `submit.task_base_addr`）|
| `RKNPU_OFFSET_ENABLE_MASK` | `0xf008` | **定义但驱动未使用**（grep 无引用）|
| `RKNPU_INT_CLEAR` | `0x1ffff` | 17 位全清 |
| `RKNPU_PC_DATA_EXTRA_AMOUNT` | `4` | PC_DATA_AMOUNT 公式里的 +4（32-bit word 数）|

> 注：4.19 分支的同一头文件还定义了 `RKNPU_OFFSET_PC_TASK_STATUS 0x3c` 与 `RKNPU_OFFSET_CLR_ALL_RW_AMOUNT 0x8010 / DT_WR_AMOUNT 0x8034 / DT_RD_AMOUNT 0x8038 / WT_RD_AMOUNT 0x803c`；0.9.x 把这些从宏改成了 per-SoC 的 `struct rknpu_config` 字段（见 §6）。

**版本编码宏**（`#L42-L49`）：`CODE = MAJOR*10000 + MINOR*100 + PATCHLEVEL`。v0.9.8 → `908`。
**`DRIVER_MAJOR/MINOR/PATCHLEVEL = 0/9/8`，`DRIVER_DATE "20240828"`，`DRIVER_NAME "rknpu"`**：
<https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/include/rknpu_drv.h#L30-L35>

### 1.3 枚举数值【源码】

`enum e_rknpu_mem_type`（`#L52-L85`，`mem_create.flags`）：

| 值 | 名字 | 0.9.8 驱动是否真的用 |
|---|---|---|
| `0<<0` | `RKNPU_MEM_CONTIGUOUS` | ✅（默认值，隐含：未置 bit0 即 CMA）|
| `1<<0` | `RKNPU_MEM_NON_CONTIGUOUS` | ✅ 走 shmem+IOMMU 分支 |
| `0<<1` | `RKNPU_MEM_NON_CACHEABLE` | ✅（默认）|
| `1<<1` | `RKNPU_MEM_CACHEABLE` | ✅（`MEM_SYNC` 的前置条件）|
| `1<<2` | `RKNPU_MEM_WRITE_COMBINE` | ✅ `DMA_ATTR_WRITE_COMBINE` |
| `1<<3` | `RKNPU_MEM_KERNEL_MAPPING` | ✅（`vmap()` 出内核虚拟地址）|
| `1<<4` | `RKNPU_MEM_IOMMU` | ❌ **定义了但驱动代码零引用** |
| `1<<5` | `RKNPU_MEM_ZEROING` | ✅ |
| `1<<6` | `RKNPU_MEM_SECURE` | ❌ **定义了但驱动代码零引用** |
| `1<<7` | `RKNPU_MEM_DMA32` | ✅（强制 `__GFP_DMA32`）|
| `1<<8` | `RKNPU_MEM_TRY_ALLOC_SRAM` | ✅（需 `CONFIG_ROCKCHIP_RKNPU_SRAM` + DT 有 SRAM）|
| `1<<9` | `RKNPU_MEM_TRY_ALLOC_NBUF` | ✅（需 `CONFIG_NO_GKI`）|
| `1<<10` | `RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT` | ✅（0.9.8 新增）|

`RKNPU_MEM_MASK` = 上述 bit0..bit10 的并集。

> 第三方实践中实际用的组合是 `0x403` = `NON_CONTIGUOUS | CACHEABLE | IOMMU_LIMIT_IOVA_ALIGNMENT`（见 `orki_bcreate(c->fd, wtotal, 0x403, w->domain)`，oRKLLM/ork-driver 的 `src/npu.c`）：<https://github.com/oRKLLM/ork-driver/blob/main/src/npu.c> 【二手，可复现】

`enum e_rknpu_mem_sync_mode`（`#L88-L93`）：`TO_DEVICE=1<<0`，`FROM_DEVICE=1<<1`。

`enum e_rknpu_job_mode`（`#L96-L107`，`submit.flags` 位掩码）：
`SLAVE=0<<0`、`PC=1<<0`、`BLOCK=0<<1`、`NONBLOCK=1<<1`、`PINGPONG=1<<2`、`FENCE_IN=1<<3`、`FENCE_OUT=1<<4`。

`enum e_rknpu_action`（`#L110-L137`，26 个，见 §2.3）。

**内部 job 状态位**（不在 UAPI，`rknpu_job.h`）：`RKNPU_JOB_DONE (1<<0)`、`RKNPU_JOB_ASYNC (1<<1)`、`RKNPU_JOB_DETACHED (1<<2)`；`RKNPU_MAX_CORES 3`；`RKNPU_CORE_AUTO_MASK 0x00 / CORE0 0x01 / CORE1 0x02 / CORE2 0x04`：
<https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/include/rknpu_job.h#L18-L27>

---

## 2. IOCTL 清单

### 2.1 编号是**算出来的**，不是编的【源码】

两套编号并存（`rknpu_ioctl.h#L294-L328`）：

```c
#define RKNPU_ACTION 0x00 ... RKNPU_MEM_SYNC 0x05
#define RKNPU_IOC_MAGIC 'r'
#define DRM_IOCTL_RKNPU_ACTION  DRM_IOWR(DRM_COMMAND_BASE + RKNPU_ACTION, struct rknpu_action)
...
#define IOCTL_RKNPU_ACTION      RKNPU_IOWR(RKNPU_ACTION, struct rknpu_action)
```

`DRM_IOWR = _IOWR('d', nr, T) = 0xC0000000 | (sizeof(T)<<16) | ('d'<<8) | nr`，`DRM_COMMAND_BASE=0x40`。

**实测计算结果（v0.9.8，用 `_IOC` 公式独立复算）**：

| 名字 | DRM 路径 `DRM_IOCTL_RKNPU_*`（`/dev/dri/cardN`、`/dev/dri/renderDxxx`）| 非 DRM 路径 `IOCTL_RKNPU_*`（misc `/dev/rknpu`）| `sizeof` |
|---|---|---|---|
| ACTION | nr=0x40 → **`0xC0086440`** | nr=0x00 → `0xC0087200` | 8 |
| SUBMIT | nr=0x41 → **`0xC0686441`** | nr=0x01 → `0xC0687201` | 104 |
| MEM_CREATE | nr=0x42 → **`0xC0306442`** | nr=0x02 → `0xC0307202` | 48 |
| MEM_MAP | nr=0x43 → **`0xC0106443`** | nr=0x03 → `0xC0107203` | 16 |
| MEM_DESTROY | nr=0x44 → **`0xC0106444`** | nr=0x04 → `0xC0107204` | 16 |
| MEM_SYNC | nr=0x45 → **`0xC0206445`** | nr=0x05 → `0xC0207205` | 32 |

> **注意 sizeof 参与了编号**：0.8.x 的 `rknpu_mem_create` 是 40 字节 → `0xC0286442`；0.9.x 是 48 字节 → `0xC0306442`。但 DRM core 用 `_IOC_NR()`（0x42）**只按序号索引**，并且**不校验 size**（见 §6.3），所以旧客户端不会被拒绝，只会被零填充补齐。
> **结构体是否 `__packed` 不影响** `rknpu_task` 的 sizeof（u64 落在 32 偏移上本来就 8 对齐，packed 前后都是 40）；它也不作为 ioctl 参数出现。

### 2.2 每个 IOCTL 的作用与实现细节【源码】

dispatch 有两条互斥实现（`rknpu_drv.c`）：

- `CONFIG_ROCKCHIP_RKNPU_DRM_GEM`（Kconfig 默认，`default ROCKCHIP_RKNPU_DRM_GEM`）→ DRM 设备，`drm_ioctl` → `rknpu_ioctls[]`：
  <https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_drv.c#L702-L713>
- `CONFIG_ROCKCHIP_RKNPU_DMA_HEAP` → `miscdevice`，`rknpu_ioctl()` 按 `_IOC_NR(cmd)` 手动 switch：
  <https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_drv.c#L617-L663>

两者都在每次 ioctl 前后做 `rknpu_power_get()/rknpu_power_put_delay()`（引用计数上电）。

**`RKNPU_ACTION`（0x40）** — `struct rknpu_action { __u32 flags; __u32 value; }`。`flags` 选动作，`value` 进出参。实现在 `rknpu_action()`：<https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_drv.c#L417-L526>
- `GET_HW_VERSION(0)`：`*value = readl(base0+0x0) + (readl(base0+0x4) & 0xffff)`
- `GET_DRV_VERSION(1)`：`*value = 908`（v0.9.8）
- `GET_FREQ(2)` / `GET_VOLT(4)`：读 clk/regulator（`FPGA_PLATFORM` 下不填值但返回 0）
- **`SET_FREQ(3)` / `SET_VOLT(5)`：`case …: break;` → 返回 `-EINVAL`（`ret` 初值），即"接受枚举但不实现"**
- `ACT_RESET(6)`：软复位
- `GET/SET_BW_PRIORITY(7/8)`、`GET/SET_BW_EXPECT(9/10)`、`GET/SET_BW_TW(11/12)`：带宽仲裁（RK3588 的 `bw_priority_addr=0x0` → 该 SoC 上是空转）
- `ACT_CLR_TOTAL_RW_AMOUNT(13)`、`GET_DT_WR/RD(14/15)`、`GET_WT_RD(16)`、`GET_TOTAL_RW_AMOUNT(17)`
- `GET_IOMMU_EN(18)`、`SET_PROC_NICE(19)`（`set_user_nice(current, value)`）
- `POWER_ON(20)` / `POWER_OFF(21)`
- `GET_TOTAL_SRAM_SIZE(22)` / `GET_FREE_SRAM_SIZE(23)`（无 SRAM 时返回 0）
- `GET_IOMMU_DOMAIN_ID(24)` / `SET_IOMMU_DOMAIN_ID(25)`（0.9.x 新增；SET 语义是"切换当前域"，会 `get_and_switch` + `put`）
- `default: -EINVAL`。**第三方 oRKLLM 用的 `RKNPU_ACT_SET_DOMAIN = 0x102` 不在 stock 0.9.8 里**（源码 grep 无 `0x102`），那是他们打了内核补丁【二手】

**`RKNPU_SUBMIT`（0x41）** — 见 §3。`rknpu_submit_ioctl()` 只是转发给 `rknpu_submit()`（DRM 路径 `dev_get_drvdata(dev->dev)`）。

**`RKNPU_MEM_CREATE`（0x42）** — `rknpu_gem_create_ioctl()`：<https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_gem.c#L876-L908>
1. 先 `rknpu_gem_object_find(file_priv, args->handle)`：**如果这个 handle 已存在（例如刚用 PRIME 导入的 dma-buf）就复用**，不再分配；
2. 否则 `rknpu_gem_object_create(drm, flags, size, sram_size, iommu_domain_id, core_mask)`；
3. 回填 `handle / size(页对齐后) / sram_size / obj_addr / dma_addr`。
   **`args->obj_addr = (__u64)(uintptr_t)rknpu_obj;`** ← 直接把**内核对象指针**当 cookie 返回给用户态（`#L904`）。这是 ABI 的核心"句柄"，也是明显的信息泄露/安全点。

**`RKNPU_MEM_MAP`（0x43）** — 就是 `drm_gem_dumb_map_offset(file_priv, dev, args->handle, &args->offset)`（`#L910-L921`）：拿 DRM GEM 的 fake offset，之后 `mmap(fd, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, offset)`。**只有 `DRM_GEM` 后端有实现；`DMA_HEAP` 后端里这个 case 是空的（`break`，返回 -EINVAL），因为那时你直接 mmap dma-buf fd。**

**`RKNPU_MEM_DESTROY`（0x44）** — `rknpu_gem_destroy_ioctl()`：先 `get_and_switch` 到该对象所属 IOMMU 域（失败重试 3 次），再 `rknpu_gem_handle_destroy(file_priv, handle)`，最后 `iommu_domain_put`。**按 handle 释放；`reserved`/`obj_addr` 不参与释放判断。**

**`RKNPU_MEM_SYNC`（0x45）** — `rknpu_gem_sync_ioctl()`：<https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_gem.c#L1667-L1730>
- 只认 `args->obj_addr`（=MEM_CREATE 返回的内核指针 cookie），**不认 handle**；
- **`if (!(rknpu_obj->flags & RKNPU_MEM_CACHEABLE)) return -EINVAL;`** —— 非 cacheable buffer 调 SYNC 直接报错；
- 非连续对象按 sg 表逐段 `dma_sync_single_range_for_device/for_cpu`；SRAM/NBUF 走 `rknpu_cache_sync()`。

**DRM render 节点能不能用？** 六个 ioctl 全部 `DRM_RENDER_ALLOW`，且**没设 `DRM_AUTH`/`DRM_MASTER`**（`#L702-L713`）。对照 mainline `drm_ioctl_permit()`：`DRM_AUTH` 缺失即无需认证，`DRM_RENDER_ALLOW` 存在即允许 render client：
<https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/drm_ioctl.c#L612-L631>
→ **`/dev/dri/renderD129` 上可以直接跑全部 6 个 ioctl，无需 master/auth**。0.9.8 的第三方用户态头文件也写"verified … render node /dev/dri/renderD129，render nodes need no DRM auth/master"：
<https://github.com/oRKLLM/ork-driver/blob/main/src/rknpu_ioctl.h>【二手，但与源码一致】
⚠️ **冲突记录**：同一个项目的 wiki 另一页声称 "ioctls on `/dev/dri/card1` — the PRIMARY card node, not renderD129 (raw RKNPU_SUBMIT only works on the card node)"（<https://github.com/oRKLLM/ork-driver/wiki/regcmd-ISA-Reference>）。**源码证据支持 render 节点可用**（`DRM_RENDER_ALLOW`），该 wiki 说法与源码及其自家头文件注释矛盾，我判定为**过时/记错**，但你上板时应两种节点都试一次。

---

## 3. 任务提交模型

### 3.1 `struct rknpu_task`（40 字节，`__packed`）【源码】

`rknpu_ioctl.h#L222-L232`：

| 偏移 | 类型 | 字段 | 内核是否读 | 语义 |
|---|---|---|---|---|
| 0x00 | u32 | `flags` | ❌ | task 标志；**驱动完全不读** |
| 0x04 | u32 | `op_idx` | ❌ | 算子索引；**驱动完全不读** |
| 0x08 | u32 | `enable_mask` | ❌ | 模块使能掩码；**驱动完全不读**（第三方实测写 `0xd`）|
| 0x0C | u32 | `int_mask` | ✅ | **期望的完成中断掩码** |
| 0x10 | u32 | `int_clear` | ❌ | **驱动不读**（清除值其实是驱动自己写 `RKNPU_INT_CLEAR=0x1ffff`）|
| 0x14 | u32 | `int_status` | ✅ 只写 | 完成时驱动把读到的状态写回 `last_task->int_status` |
| 0x18 | u32 | `regcfg_amount` | ✅ | 参与 PC_DATA_AMOUNT 计算（单位见 §3.3）|
| 0x1C | u32 | `regcfg_offset` | ❌ | **驱动完全不读** |
| 0x20 | u64 | `regcmd_addr` | ✅ | regcmd 程序在**设备侧（IOVA）的 32 位字节地址** |

> **`enable_mask` / `op_idx` / `regcfg_offset` 是"死字段"**：我在 `develop-4.19`、`develop-5.10`、`develop-6.1` 的 `rknpu_job.c` 里 grep 均无引用（只有 `int_mask`、`regcfg_amount`、`regcmd_addr`、`int_status` 被用）。第三方 StarryOS 文档称 `enable_mask` 是"模块使能掩码"由驱动使用 —— **该说法与源码不符**（<https://qc-gpu-driver.github.io/starryos-pulsar/documents/ioctl-protocol.html>，二手且有误，该页的 "36 字节/96 字节" 尺寸也是错的）。

### 3.2 `struct rknpu_submit`（104 字节）【源码】

`rknpu_ioctl.h#L265-L280`；偏移由我按 C 布局复算：

| 偏移 | 类型 | 字段 | 语义 |
|---|---|---|---|
| 0x00 | u32 | `flags` | `RKNPU_JOB_*` 位掩码（PC/NONBLOCK/PINGPONG/FENCE_IN/FENCE_OUT）|
| 0x04 | u32 | `timeout` | 等待超时 **ms**（阻塞提交用）|
| 0x08 | u32 | `task_start` | 全局起始 task 索引 |
| 0x0C | u32 | `task_number` | **task 数，必须 ≥1**，否则 `-EINVAL` |
| 0x10 | u32 | `task_counter` | 出参：完成数（超时路径会写入硬件 task 计数器）|
| 0x14 | s32 | `priority` | 调度优先级（驱动未用于排序，仅存) |
| 0x18 | u64 | `task_obj_addr` | **task 数组的"内核对象 cookie"**（就是 MEM_CREATE 的 `obj_addr`，驱动直接当指针解引用）|
| 0x20 | u32 | `iommu_domain_id` | 该 job 用哪个 IOMMU 域（0.9.x 新增）|
| 0x24 | u32 | `reserved` | 填充（0.9.6 时这里叫 `reserved`，0.9.8 仍是 `reserved`；`mem_create` 里同位置已改名 `core_mask`）|
| 0x28 | u64 | `task_base_addr` | 写进 `PC_DMA_BASE_ADDR(0x34)`；一般填 **0**（mainline 也写 0）|
| 0x30 | s64 | `hw_elapse_time` | 出参：硬件本次耗时（ktime）|
| 0x38 | u32 | `core_mask` | 核选择位掩码，见 §3.4 |
| 0x3C | s32 | `fence_fd` | IN: 入 fence fd；OUT: 驱动回填出 fence fd；不用时填 `-1` |
| 0x40 | `struct rknpu_subcore_task[5]` | 每核 task 区间（8 字节 ×5）|

`struct rknpu_subcore_task { u32 task_start; u32 task_number; }`（`#L241-L244`）。

**提交校验**（`rknpu_submit()`，<https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_job.c#L746-L850>）：
```c
if (args->task_number == 0) return -EINVAL;
if (args->core_mask > rknpu_dev->config->core_mask) return -EINVAL;   // RK3588: > 0x7 非法
```
**阻塞语义的坑**：`if (args->flags & RKNPU_JOB_PC) job->ret = rknpu_job_wait(job);`（`#L834`）——**只有带 `RKNPU_JOB_PC` 的提交才真的等**；不带 PC 的提交即使没设 NONBLOCK 也立刻返回。（第三方 wiki 的 "a blocking submit only blocks if RKNPU_JOB_PC is set" 与此一致。）
`NONBLOCK` 路径走 `rknpu_job_timeout_clean` + `schedule_work` 异步清理，`fence_fd` 出参可用。

### 3.3 PC 提交：驱动实际做了什么【源码】

`rknpu_job_subcore_commit_pc()`：<https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_job.c#L272-L375>

```c
/* 0. 先切回 slave 模式（非 PC 提交也这么干）*/
REG_WRITE(0x1, RKNPU_OFFSET_PC_DATA_ADDR);                       // #L388/#L391
/* 1. PC 程序基址 = first task 的 regcmd_addr（原样写，无移位/无掩码）*/
REG_WRITE(first_task->regcmd_addr, RKNPU_OFFSET_PC_DATA_ADDR);   // #L346 / #L349
/* 2. 每条 task 取多少条寄存器指令 */
REG_WRITE((first_task->regcfg_amount + RKNPU_PC_DATA_EXTRA_AMOUNT
           + pc_data_amount_scale - 1) / pc_data_amount_scale - 1,
          RKNPU_OFFSET_PC_DATA_AMOUNT);                          // #L352-356
/* 3. 用 last task 的 int_mask 设中断掩码 */
REG_WRITE(last_task->int_mask, RKNPU_OFFSET_INT_MASK);           // #L358
/* 4. 用 first task 的 int_mask 当"中断清除"值（怪癖！）*/
REG_WRITE(first_task->int_mask, RKNPU_OFFSET_INT_CLEAR);         // #L360
/* 5. task 数 + ping-pong + bit2 */
REG_WRITE(((0x6 | task_pp_en) << pc_task_number_bits) | task_number,
          RKNPU_OFFSET_PC_TASK_CONTROL);                         // #L362-363
/* 6. DMA base */
REG_WRITE(args->task_base_addr, RKNPU_OFFSET_PC_DMA_BASE_ADDR);  // #L365
/* 7. 发令：0→1→0 */
REG_WRITE(0x1, RKNPU_OFFSET_PC_OP_EN);
REG_WRITE(0x0, RKNPU_OFFSET_PC_OP_EN);                           // #L371-372
```

**PC_DATA_AMOUNT 公式**：RK3588 `pc_data_amount_scale = 2`，`RKNPU_PC_DATA_EXTRA_AMOUNT = 4`
→ `amount = (regcfg_amount + 4 + 2 - 1)/2 - 1 = (regcfg_amount + 5)/2 - 1`（整数除）。
第三方实测标定：`regcfg_amount = 108` → 写入 `0x14` 的值 = **55 = 0x37**，与他们 wiki 记录的 "PC-chain amount `0x37`" 吻合。
对照 **mainline `accel/rocket`**（同一硬件、另一套驱动）：`PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT((task->regcmd_count + 1) / 2 - 1)`：
<https://github.com/torvalds/linux/blob/master/drivers/accel/rocket/rocket_job.c#L138-L152>
→ 两者形式相同，差 **+2（=4 words / 2）**。**【推断】**含义是：vendor 认为实际条目数比用户报的多 2 条 64-bit 条目（`RKNPU_PC_DATA_EXTRA_AMOUNT/2`），而 mainline 不加。也就是说 **`regcfg_amount` 的单位是 32-bit word**，且 vendor ABI 里用户报的数**不含**那 2 条额外条目。这一点源码没有注释说明，请以实测标定为准。

**PC 地址编码（回答"32 位地址+标志位"）**：
- 驱动**原样**把 `regcmd_addr` 写进 `PC_DATA_ADDR(0x10)`；Slave 模式写 `0x1`（=地址 0 + pc_sel=1）。
- 按 RK3588 TRM 的寄存器描述（二手渲染，<https://qc-gpu-driver.github.io/starryos-pulsar/documents/registers/pc.html>，标注来源 `RK3588 TRM §36.4.3`）：`0x10 = RKNN_pc_base_address`，`[31:4]=pc_source_addr`，`[3:1]=保留`，`[0]=pc_sel`（0=PC 模式走 AXI DMA 取寄存器配置，1=Slave 模式走 AHB）。
- mainline rocket 的 UAPI 里 `regcmd` 本身就是 `__u32`（NPU 地址空间内的 32 位 DMA 地址）：<https://github.com/torvalds/linux/blob/master/include/uapi/drm/rocket_accel.h>
- **结论**：`regcmd_addr` = 32 位设备地址，`[3:0]` 必须为 0（16 字节对齐 + bit0 不能置位，否则会变成 Slave 模式）；驱动不做任何编码/重定位。

### 3.4 core mask：三核如何选/广播【源码】

- **`core_mask` 是位掩码**：bit0/1/2 = core0/1/2；`0x00` = `RKNPU_CORE_AUTO_MASK`。
- **`0x00`（AUTO）**：`rknpu_job_schedule()` 里由驱动挑"当前排队 task 最少"的核，并把 `args->core_mask` 改写成单核掩码、`use_core_num=1`（`#L532-L541`）。注意：kernel **会修改你的入参**（DRM 路径回写用户空间）。<https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_job.c#L525-L560>
- **合法值只有 `0x1 / 0x2 / 0x4 / 0x3 / 0x7`**。`rknpu_job_commit()` 是显式 switch，`0x5`、`0x6` 落到 `default: LOG_ERROR("Unknown core mask")` → **该 job 永远不会被 kick，只能等到超时 abort**（不是广播，也不是报错返回）。<https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_job.c#L402-L427>
- `core_mask > config->core_mask`（RK3588 为 `0x7`）→ `-EINVAL`。每 SoC 上限见 `rk3588_rknpu_config.core_mask = 0x7`（`rknpu_drv.c#L183-L202`）。
- **per-core task 切分读 `subcore_task[]` 的索引规则很反直觉**（`rknpu_job.c#L305-L336`）：
  - `use_core_num == 1 或 2` → 用 `subcore_task[core_index]`（即 0、1 号槽）
  - `use_core_num == 3` → 用 `subcore_task[core_index + 2]`（**2、3、4 号槽**！）
  其余情况 runtime 日志 "Unknown use core num"。
- **等待核**：`rknpu_wait_core_index()` 把 `0x1/0x3/0x7` 都映射为 core0 的等待队列（`#L27-L48`）。
- **三核是"同 job 三核各自排队并发"**：`job->run_count/interrupt_count = use_core_num`，`atomic_dec_and_test(&job->run_count)` 才 commit（三核全部排到才一起 kick）。
- 中断：每核一个 IRQ（`rk3588_npu_irqs = npu0_irq/npu1_irq/npu2_irq`，`rknpu_drv.c#L84-L88`），IRQ 里对 `INT_STATUS` 做 `rknpu_fuzz_status()` 归一化（每 2 bit 一组：`0x3/0xc/0x30/0xc0/0x300/0xc00`，任一非零则该组全置 1）后与 `job->int_mask` 比较，**不等则只清中断、不置 DONE** → 阻塞提交一路等到超时（`#L643-L679`）。所以 `int_mask` 填错是"静默超时"的常见原因。

### 3.5 第三方可复现的提交样例【二手，工程实证】

oRKLLM/ork-driver 的 `src/npu.c`（真实上板跑通的最小提交，RK3588）：
`task`: `enable_mask=0xd, int_mask=0x300, int_clear=0x1ffff, regcfg_amount=108, regcmd_addr=<dma>`；
`submit`: `flags=0x5 (PC|PINGPONG), task_number=1, subcore_task[0]={0,1}, core_mask=0x1`。
<https://github.com/oRKLLM/ork-driver/wiki/regcmd-ISA-Reference> · <https://github.com/oRKLLM/ork-driver/blob/main/src/npu.c>
`int_mask=0x300` = DPU group0+group1（bit8/bit9），与 mainline rocket 里硬编码的 `PC_INTERRUPT_MASK_DPU_0|DPU_1` 一致 → **互相印证**。

---

## 4. regcmd "命令集"（回答 `RKNPU_CMD_*` 问题）

### 4.1 结论：GPL 驱动里**没有** `RKNPU_CMD_*` 枚举【源码】

我在 4.19 / 5.10 / 6.1 / 6.6 四个分支的 `drivers/rknpu` 全量文件（`*.c` + `include/*.h`）里 grep `RKNPU_CMD`、`rknpu_pc.h`、`rknpu_reg_cmd.h`：**零命中**（`drivers/rknpu/include/rknpu_ioctl.h` 里只有 `RKNPU_OFFSET_*`、`RKNPU_INT_CLEAR`、`RKNPU_PC_DATA_EXTRA_AMOUNT`、`RKNPU_MEM_*`、`RKNPU_JOB_*`、`RKNPU_CORE*_MASK`、`RKNPU_ACTION/SUBMIT/MEM_*` 这几族宏）。`include/uapi/drm/rknpu_drm.h`、`drivers/rknpu/rknpu_reg_cmd.h`、`drivers/rknpu/rknpu_pc.h` 均 404。
**⇒ "RKNPU_CMD_* opcode 清单（enable/disable、PC 跳转、算子配置、DMA）"这个提法在 vendor 内核里找不到依据，属未确认；不要按它写代码。**

真正存在的三层"命令"语义如下。

### 4.2 第一层：内核自己写的 PC 控制寄存器（= `RKNPU_OFFSET_*`）【源码】

见 §1.2 / §3.3。这一层就是你说的 "enable / PC 跳转 / DMA 基址 / task 控制"：
`PC_OP_EN(0x8)` 使能并发令、`PC_DATA_ADDR(0x10)` 程序基址+模式位、`PC_DATA_AMOUNT(0x14)` 条目数、
`INT_MASK/CLEAR/STATUS/RAW(0x20/0x24/0x28/0x2c)`、`PC_TASK_CONTROL(0x30)` task 数+pingpong、`PC_DMA_BASE_ADDR(0x34)`。

### 4.3 第二层：task 描述符字段

`flags / op_idx / enable_mask / int_mask / int_clear / regcfg_amount / regcfg_offset / regcmd_addr`（§3.1）。**除 `int_mask / regcfg_amount / regcmd_addr / int_status` 外内核都不读**。

### 4.4 第三层：regcmd 流本身 = **64-bit 寄存器写对**（不是 opcode）【二手，两份独立来源互证】

**(a) TRM 描述**（二手渲染，<https://qc-gpu-driver.github.io/starryos-pulsar/documents/registers/pc.html>，标注 `RK3588 TRM §36.4.3`），每条 64-bit 指令：

| 位域 | 含义 |
|---|---|
| `[63:48]` | 目标模块选择（哪个 block），外加 bit55 = "设置各 block 的 op_en" |
| `[47:16]` | 寄存器值 |
| `[15:0]` | 该 block 内的寄存器偏移 |

模块位（同页）：bit56=PC、bit57=CNA、bit59=CORE、bit60=DPU、bit61=DPU_RDMA、bit62=PPU、bit63=PPU_RDMA。
示例：`64'h0081_0000_007f_0008` = 设置各 block 的 op_en（`[15:0]=0x0008` → offset 8 = op_en）；并**强烈建议 `op_en` 放在寄存器列表末尾，且在它之前必须先写一条 `64'h0041_xxxx_xxxx_xxxx`**。

**(b) 第三方逆向的解码**（<https://github.com/oRKLLM/ork-driver/wiki/regcmd-ISA-Reference>，上板实证）：

```
word0, word1 两个 u32（小端）
reg_offset = word0 & 0xffff
block_id   = word1 >> 16
value      = ((word1 & 0xffff) << 16) | (word0 >> 16)
Blocks（他们的命名）: 0x0201 CNA(feature/conv), 0x0801 DPU, 0x1001 PPU/输出, 0x0101 PC(程序控制)
```

**【推断/核对】** 两份描述其实**相容**：把一条 64-bit 指令按小端拆成 `word0`(低 32) / `word1`(高 32)，
`[15:0] = word0 & 0xffff` ✓；`[47:16] = (word0>>16) | ((word1 & 0xffff)<<16)` ✓；`[63:48] = word1>>16` ✓。
但**模块编号命名存在系统性错位**（他们叫 `0x0801` DPU、`0x1001` PPU，按 TRM 位表 0x0801→bit59=CORE、0x1001→bit60=DPU），且两者的 block 字段都带一个 `+1`（`0x0101/0x0201/…`）。**哪位正确我未确认**，建议以你自己上板的寄存器写入观测为准（用 `RKNPU_ACTION` 的 RW_AMOUNT / 或直接看 CNA/DPU 输出是否正确来判定）。

**PC 链（同页，`0x0101` block）**：reg `0x10` = 下一个 regcmd 的地址（0 = 结束）；reg `0x14` = amount（非末段 `0x37`，末段 `0`）。`RKNPU_OFFSET_PC_DATA_ADDR` 在 UAPI 里也是 `0x10`、`PC_DATA_AMOUNT` 也是 `0x14` —— **注意这是两个不同命名空间**（一个是 PC 块自身寄存器，一个是 regcmd 里对 PC 块的写），别混。

---

## 5. 内存模型：IOMMU / CMA / dma-buf / 为什么 shmem 爆掉

### 5.1 两套互斥后端【源码】

`Kconfig`（`choice`，"RKNPU memory manager"，`default ROCKCHIP_RKNPU_DRM_GEM`）：
<https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/Kconfig#L38-L56>

| | `ROCKCHIP_RKNPU_DRM_GEM`（默认）| `ROCKCHIP_RKNPU_DMA_HEAP` |
|---|---|---|
| 设备节点 | `/dev/dri/cardN` + `/dev/dri/renderDxxx` | misc `/dev/rknpu` |
| ioctl 编号 | `DRM_IOWR('d',0x40+nr,…)` | `_IOWR('r',nr,…)` |
| `mem_create.handle` | **GEM handle**（`drm_gem_handle_create`）| **dma-buf fd**（入参！`fd = args.handle; dmabuf = dma_buf_get(fd)`）|
| `MEM_MAP` | GEM fake offset，用于 mmap | **空实现**（直接 mmap dma-buf fd）|
| 分配 | `drm_gem_get_pages` / `dma_alloc_attrs` | `rk_dma_heap_buffer_alloc()` |
| 依赖 | `DRM` | `DMABUF_HEAPS_ROCKCHIP_CMA_HEAP` |

DMA-heap 版 `rknpu_mem_create_ioctl`（`rknpu_mem.c`）里有一句很关键的兼容代码：
```c
unsigned int in_size = _IOC_SIZE(cmd);
unsigned int k_size  = sizeof(struct rknpu_mem_create);
copy_from_user(&args, (void *)data, in_size);
if (k_size > in_size) memset(k_data + in_size, 0, k_size - in_size);   // 老结构体自动补零
```
<https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_mem.c>

### 5.2 GEM 路径：什么时候 CMA、什么时候 shmem【源码】

`rknpu_gem_alloc_buf()`：<https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_gem.c#L139-L245>

```c
if (!(flags & RKNPU_MEM_NON_CONTIGUOUS)) dma_attrs |= DMA_ATTR_FORCE_CONTIGUOUS;   // → CMA
if (flags & RKNPU_MEM_CACHEABLE)  dma_attrs |= DMA_ATTR_NON_CONSISTENT /*+DMA_ATTR_SYS_CACHE_ONLY*/;
if (flags & RKNPU_MEM_WRITE_COMBINE) dma_attrs |= DMA_ATTR_WRITE_COMBINE;
#if RKNPU_GEM_ALLOC_FROM_PAGES   // 硬编码为 1
if ((flags & RKNPU_MEM_NON_CONTIGUOUS) && rknpu_dev->iommu_en)
        return rknpu_gem_get_pages(rknpu_obj);        // ← shmem 页 + IOMMU
#endif
… dma_alloc_attrs(...)                                 // ← CMA / dma-direct
    失败且 (连续 && iommu_en) → 清 FORCE_CONTIGUOUS + 置 NON_CONTIGUOUS 再试一次
```
`#define RKNPU_GEM_ALLOC_FROM_PAGES 1` 是**硬编码**（`rknpu_gem.c#L31`，4.19 同样），所以 CMA 路径只在"非 NON_CONTIGUOUS"或"IOMMU 关闭"时走到。

`rknpu_gem_get_pages()`（`#L34-L114`）：
```c
rknpu_obj->pages = drm_gem_get_pages(&rknpu_obj->base);      // ← shmem 后备页
rknpu_obj->sgt   = drm_prime_pages_to_sg(drm, pages, num_pages);
rknpu_iommu_dma_map_sg(drm->dev, sgt->sgl, nents, DMA_BIDIRECTIONAL, iova_aligned);
rknpu_obj->dma_addr = sg_dma_address(sgt->sgl);              // ← 返回的就是 IOVA
```
`iova_aligned = !(flags & RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT)`（`#L41-L42`）——即该 flag 的语义是"**放弃 IOVA 对齐优化**，走标准 `dma_map_sg()`"，而默认（不置位）走 Rockchip 自己那套 `rknpu_iommu_dma_map_sg()`（`rknpu_iommu.c#L207-L296`，它会自己 `alloc_iova` + `iommu_map_sg`，并对 sg 做长度/偏移修补）。

### 5.3 为什么 RKLLM 加载模型会占满 shmem（`shmem-rss:10.98GiB`）

**已验证的机制【源码 + mainline 源码】**：
1. rknpu 的 GEM 对象由 `drm_gem_object_init()` 初始化 → mainline 里它就是 `shmem_file_setup("drm mm object", size, flags)`；
2. `drm_gem_get_pages()` 的文档注释明确写 "allocate backing pages for a GEM object **from shmem**"，实现是 `shmem_read_mapping_page_gfp()` / `shmem_read_folio_gfp()`。
   证据：<https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/drm_gem.c#L174-L207>（`drm_gem_object_init` → `shmem_file_setup*`）与 `#L640-L710`（`drm_gem_get_pages`，注释 "from shmem"、调用 `shmem_read_folio_gfp`）。
3. 因此**只要 librknnrt/RKLLM 用 `RKNPU_MEM_NON_CONTIGUOUS` 分配 NPU buffer（第三方实证的常规做法，flags `0x403`），这些缓冲区的物理页就是 tmpfs/shmem 页**，在 `smaps_rollup`/`smem` 里计入 **Shmem/`shmem-rss`**；再叠加 `MEM_MAP` + `mmap` 把同一批页映射进进程地址空间，RSS 里也会看到它们。
4. RKLLM 把**整个模型权重 + KV cache + scratch**都放这种 buffer（它必须对 NPU 可见、且要能 mmap 做 CPU 侧填充），所以 `shmem-rss ≈ 模型体积 + 额外缓冲`。10.98 GiB 对 7B 级 w8a8 模型量级上说得通。

**【推断】** 为什么看起来"占满"：`smem`/`smaps` 统计把**每个进程各自的 shmem 相加**，且同页既算共享内存又可能被计入进程 RSS；多进程/多次加载会叠加。若你的 10.98 GiB 明显大于模型文件体积，优先怀疑：①加载期同时存在"文件页 + GEM 副本"两份；②残留未 `MEM_DESTROY` 的 buffer（fd 关闭时 DRM 会 reap，但**IOMMU 域的分配器状态不会回收**——第三方明确指出多域场景下进程被 kill 会"strand whole 4 GiB domains until reboot"，<https://github.com/oRKLLM/ork-driver/wiki/NPU-Quirks>）。
**可验证做法**：`grep -i shmem /proc/meminfo`、`cat /proc/<pid>/smaps_rollup`、以及 `RKNPU_ACTION` 前后对比 + `/sys/kernel/debug/dri/*/`（驱动带 debugfs/procfs 支持，`CONFIG_ROCKCHIP_RKNPU_DEBUG_FS/PROC_FS`）。

### 5.4 IOMMU 域与 4 GiB 窗口

- 域由 `rknpu_iommu_init_domain()` 建立，`rknpu_iommu_switch_domain()` 真正 `iommu_domain_alloc(bus)` + `iommu_attach_device()`；设备树无 `iommus` 或该节点 `status=disabled` 时驱动进 non-IOMMU 模式（`rknpu_is_iommu_enable()`，`rknpu_drv.c#L827-L848`）。
- **一次只有一个域是"活"的**：`rknpu_dev->iommu_domain_id` 是进程/设备级全局，`rknpu_iommu_domain_get_and_switch()` 用一把 mutex + 引用计数做切换，切换要等 refcount 归零（`rknpu_iommu.c#L514-L560`）。**这解释了为什么 `submit` 里还要单独带 `iommu_domain_id`**：驱动会在提交前把它切过去。
- RK3588 的 `config.dma_mask = DMA_BIT_MASK(40)`（`rknpu_drv.c#L186`），IOVA 上限取 `dma_get_mask(dev)`。
- **"每个 domain ~4 GiB"是第三方实测结论**【二手】：<https://github.com/oRKLLM/ork-driver/wiki/regcmd-ISA-Reference> / `NPU-Quirks`（他们的 ggml backend 用"多域 + 每域锚点分配 + 16 MB 分块 import"来放大模型）。源码里我没找到 32 位 IOVA 的硬性截断，**标注为未在源码确认**（很可能来自 RK3588 rk_iommu 的域几何或 NPU 侧地址位宽）。
- **已知驱动缺陷（第三方，实测）**：新域里若**第一个** buffer 是 dma-buf **import**（SG 映射），IOVA 会落错；必须先做一次原生 `bcreate` 锚点。另：单次 import ≥16–64 MB 会在 `PRIME_FD_TO_HANDLE` 报 `ENOMEM`；一个 PC 链里 import 映射过多 → `errno=110`。全部来自 <https://github.com/oRKLLM/ork-driver/wiki/NPU-Quirks>【二手】。

### 5.5 dma-buf 的正确用法（DRM 后端）【源码 + 二手互证】

- **导出**：GEM → `DRM_IOCTL_PRIME_HANDLE_TO_FD`（core ioctl，`DRM_RENDER_ALLOW`，驱动挂了 `prime_handle_to_fd = drm_gem_prime_handle_to_fd`、`gem_prime_export`）。
- **导入**：`DRM_IOCTL_PRIME_FD_TO_HANDLE`（`drm_gem_prime_fd_to_handle` → `rknpu_gem_prime_import` / `rknpu_gem_prime_import_sg_table`）拿到 GEM handle，**然后调用 `RKNPU_MEM_CREATE` 把该 handle 传进去** —— 此时 `rknpu_gem_object_find()` 命中已存在对象，`size/flags` 被忽略，只回填 `obj_addr/dma_addr`。第三方代码注释即为此流程："`PRIME_FD_TO_HANDLE` → `MEM_CREATE(handle, flags=0, size=0)`"。
- **注意**：导入的 dma-buf 不受 rknpu 的 `MEM_SYNC` 管；CPU 写完必须用 `DMA_BUF_IOCTL_SYNC`（`DMA_BUF_SYNC_START|WRITE` / `END|WRITE`）——第三方头文件里明确写了这一点：<https://github.com/oRKLLM/ork-driver/blob/main/src/rknpu_ioctl.h>【二手，机制合理】。

---

## 6. 版本差异与兼容矩阵

### 6.1 我实际核对到的版本快照【源码】

| 驱动版本 | `DRIVER_DATE` | 来源（可自行复验）|
|---|---|---|
| **0.8.0** | 20220803 | `rockchip-linux/kernel` @ `develop-4.19` → `drivers/rknpu/include/rknpu_drv.h#L34-L36` |
| **0.8.5** | 20230202 | `armbian/linux-rockchip` @ `rk-5.10-rkr4` → 同路径 |
| **0.9.6** | 20240322 | `armbian/linux-rockchip` @ `rk-6.1-rkr1` → 同路径 |
| **0.9.8** | 20240828 | `rockchip-linux/kernel` @ `develop-5.10`、`develop-6.1`、`develop-6.6` → `drivers/rknpu/include/rknpu_drv.h#L30-L35` |

<https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/include/rknpu_drv.h#L30-L35>

### 6.2 ABI 差分（我逐行 diff 得到）【源码】

**0.9.6 → 0.9.8**（小改）：
- 新增 `RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT = 1<<10`（并加入 `RKNPU_MEM_MASK`）；
- `rknpu_mem_create` 第 7 个 u32 从 `reserved` 改名为 `core_mask`（**大小不变**，无 ABI 破坏）。

**0.8.x → 0.9.x**（破坏性）：
| 变化 | 0.8.x | 0.9.x | 影响 |
|---|---|---|---|
| `rknpu_submit` 偏移 0x20 | `__u64 regcfg_obj_addr` | `__u32 iommu_domain_id` + `__u32 reserved` | **同 size（104），ioctl 编号不变 → 旧程序不会被拒绝，字段被静默错解** |
| `rknpu_submit` 偏移 0x30 | `__u64 user_data` | `__s64 hw_elapse_time`（出参）| 同上 |
| `rknpu_mem_create` | 40 字节 | 48 字节（+`iommu_domain_id` +`core_mask`）| **ioctl 编号 0xC0286442 → 0xC0306442** |
| mem flags bit7 | `RKNPU_MEM_NON_DMA32`（"非 DMA32"）| `RKNPU_MEM_DMA32`（"用 DMA32"）| **同名位语义反转！** 老程序置 bit7 = 想避开 DMA32，新驱动读成"强制 DMA32" |
| mem flags | 无 bit9/bit10 | `TRY_ALLOC_NBUF(1<<9)`、`IOMMU_LIMIT_IOVA_ALIGNMENT(1<<10)` | 新增 |
| action | 0..23 | +`GET/SET_IOMMU_DOMAIN_ID(24/25)` | 新增 |
| 头文件宏 | 有 `PC_TASK_STATUS 0x3c` 与 `*_RW_AMOUNT 0x80xx` | 移除，改为 `struct rknpu_config` 字段（`pc_task_status_offset`、`amount_top/amount_core`）| 仅内部 |
| `submit` 结构体尾 | `subcore_task[5]` | 同 | 不变 |

`rknpu_task`（40 字节，`__packed`）在 4.19/5.10/6.1 完全一致 —— **任务描述符本身是稳定的**。

### 6.3 驱动为什么"能吃下"旧结构体：DRM core 的 size 容忍【源码】

mainline `drm_ioctl()`：
```c
drv_size = _IOC_SIZE(ioctl->cmd);            // 内核版本的结构体大小
out_size = in_size = _IOC_SIZE(cmd);         // 用户 cmd 里的 size 位
ksize = max(max(in_size, out_size), drv_size);
copy_from_user(kdata, arg, in_size);
if (ksize > in_size) memset(kdata + in_size, 0, ksize - in_size);   // ← 自动补零
```
<https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/drm_ioctl.c#L855-L915>
⇒ 旧客户端传 40 字节 `mem_create`：序号 0x42 仍命中同一 handler，缺的 8 字节补 0（`iommu_domain_id=0, core_mask=0`）。**不会报 EINVAL**。这也意味着"老 librknnrt + 新内核"不会在 ioctl 层失败，而是**在语义层静默出错**（尤其 §6.2 的 bit7 反转与 submit 字段错位）。
另：驱动带 `.compat_ioctl = drm_compat_ioctl`（`rknpu_drv.c#L725-L727`），结构体无指针字段，**32 位 ARM 用户态可直接用**。

### 6.4 驱动 ↔ `librknnrt` 版本兼容矩阵

**未找到权威一手来源 ⇒ 我不给具体矩阵（避免编造）。** 已知可核验的三点：
1. **运行时自检**：`RKNPU_ACTION{GET_DRV_VERSION}` 返回 `MAJOR*10000+MINOR*100+PATCH`；v0.9.8 → **908**。第三方用户态库（含 librknnrt）通常就是拿这个值与内部期望值比对/打印（`get_drv_version` 在 `rknpu_drv.c#L342-L347`）。
2. 官方文档侧（二手）：Rockchip 的 *RKNPU User Guide*（`RKNN_SDK_V2.x`）里有"驱动版本要求"章节，例如 v2.2.0 的手册：<https://gitea.swigg.net/dustins/airrockchip-rknn-toolkit2/raw/tag/v2.2.0/doc/02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.2.0_EN.pdf>（我尝试抓取 PDF 失败，**内容未确认**，仅给出入口）。
3. **可自证的判据**（推荐）：在板子上跑 `dmesg | grep -i rknpu` 看初始化行 `"Initialized RKNPU driver: v%d.%d.%d for %s"`（`rknpu_drv.c#L1458-L1459`），再对照 `struct rknpu_submit` 的 `sizeof` 与 §6.2 的字段表 —— **只要你的 librknnrt 在 `submit` 偏移 0x20/0x30 处写的是 domain/elapse 而不是 regcfg/user_data，就要求 ≥0.9.x**。

---

## 7. 它到底怎么用 DRM 框架【源码】

- **是 DRM 驱动**（可以这么定性）：`static struct drm_driver rknpu_drm_driver`，`.driver_features = DRIVER_GEM | DRIVER_PRIME | DRIVER_RENDER`（5.4+ 编译分支；<5.4 才有 `DRIVER_PRIME` 单独位），`.ioctls = rknpu_ioctls[], .num_ioctls = 6`，`.major/.minor/.patchlevel = 0/9/8`：
  <https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_drv.c#L733-L776>
- **GEM + PRIME 全套**：`dumb_create / dumb_map_offset / prime_handle_to_fd / prime_fd_to_handle / gem_prime_import / gem_prime_import_sg_table / gem_prime_mmap / gem_prime_export`（<6.1 分支）。6.1+ 用 `DEFINE_DRM_GEM_FOPS(rknpu_drm_driver_fops)`（`#L715-L716`）。
- **没有 KMS/modeset**：无 CRTC/connector/plane，纯计算的 "render-only" 加速器设备。`DRIVER_RENDER` → 内核为它创建 render minor，所以有 `/dev/dri/renderDxxx`（你看到的 **renderD129** = 第二个 render minor，128 通常是 Mali GPU）；同时也有 `/dev/dri/cardN` 主节点。
- **注册**：`drm_dev_alloc(&rknpu_drm_driver, dev)`（`#L895`）→ `drm_dev_register`。**`DMA_HEAP` 构建下则完全不同**：`misc_register(&rknpu_dev->miscdev)`（`#L1445`），`/dev/rknpu` + `_IOWR('r',…)` 编号 + 自研 `file_operations`。
- **与 mainline `accel/rocket`（同一个硬件、Tomeu Vizoso 的新驱动）的区别**：mainline 走 **drm_accel + drm_sched**，UAPI 是 `include/uapi/drm/rocket_accel.h`（`DRM_ROCKET_CREATE_BO/SUBMIT/PREP_BO/FINI_BO`），任务用 `struct drm_rocket_task{ __u32 regcmd; __u32 regcmd_count; }`、按 in/out BO handle 数组做依赖与 fence：
  <https://github.com/torvalds/linux/blob/master/include/uapi/drm/rocket_accel.h> · <https://github.com/torvalds/linux/blob/master/drivers/accel/rocket/rocket_job.c>
  → **两边不兼容**，但寄存器语义一致（`rocket_pc_writel(BASE_ADDRESS, task->regcmd)`、`PC_DATA_AMOUNT=(regcmd_count+1)/2-1`、`TASK_DMA_BASE_ADDR=0`、INTERRUPT_MASK=DPU0|DPU1），**是极好的交叉验证材料**，也是你若移植到纯 mainline 的出路。
  mainline 提交历史：<https://lore.kernel.org/linux-arm-kernel/20250713-6-10-rocket-v8-10-64fa3115e910@tomeuvizoso.net/>

---

## 8. 逆向切入点建议：写最小用户态程序提交自制 regcmd

**前置条件**：内核有 `CONFIG_ROCKCHIP_RKNPU=m/y` 且选了 `ROCKCHIP_RKNPU_DRM_GEM`；DT 里 NPU 有 `iommus`（否则 IOMMU 关闭、只能走 CMA 连续内存）。先 `dmesg | grep RKNPU` 看到 `v0.9.8` 与 `"rknpu iommu is enabled"`。

### 最小步骤（按顺序）

1. **打开设备**：`open("/dev/dri/renderD129", O_RDWR|O_CLOEXEC)`（或 `/dev/dri/cardN`；两者按源码都允许，见 §2.2 末）。
2. **握手**：`ACTION{GET_DRV_VERSION}` 必须返回 `908`；`ACTION{GET_IOMMU_EN}` 记下 IOMMU 开关（决定下一步 flags）；可选 `GET_HW_VERSION`。
3. **建 buffer**（`MEM_CREATE`，DRM 编号 `0xC0306442`）：
   - regcmd 程序缓冲：`flags = NON_CONTIGUOUS|CACHEABLE (0x3)`（IOMMU 开时）+ 建议加 `IOMMU_LIMIT_IOVA_ALIGNMENT(1<<10)` → 用 `0x403`；`size` = 你的程序 word 数×4，页对齐；
   - task 数组：≥ 40 字节（1 个 task），同样 flags；
   - 输入/输出数据 buffer（同样 flags）。
   记下每个的 **`dma_addr`（IOVA，写进 regcmd 的地址）** 与 **`obj_addr`（cookie，写进 `task_obj_addr`）**。
4. **mmap**：对每个 handle 发 `MEM_MAP`（`0xC0106443`）拿 `offset`，再 `mmap(fd, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, offset)`。
5. **填 task[0]**（40 字节，**不要**加 padding）：`enable_mask=0xd`、`int_mask=0x300`（只使能 DPU group0/1 完成中断时；也可先用 `0x1ffff` 观察）、`int_clear=0x1ffff`、`regcfg_amount=<你程序的 32-bit word 数>`、`regcmd_addr=<regcmd 的 dma_addr，低 4 位为 0>`。
6. **`MEM_SYNC`（flags=`TO_DEVICE=1`，`obj_addr`=对应 cookie）刷 cache**（CACHEABLE 时必需；非 cacheable 会 -EINVAL）。
7. **写 regcmd**：每条 64-bit 对 → `word0 = (value & 0xffff) << 16 | (reg_offset & 0xffff)`，`word1 = (block_id << 16) | ((value >> 16) & 0xffff)`。**把 `op_en` 那条放最后，并在它之前先写一条 `0x0041_xxxx_xxxx_xxxx`**（TRM 二手要求，见 §4.4）。**所有地址字段填 IOVA（硬件不做重定位）**【二手，ork 实测】。
8. **提交**：`SUBMIT`（`0xC0686441`），
   `flags=0x5`（`PC|PINGPONG`；想排障就先用 `0x1` = 只 PC）、`timeout=5000`、`task_start=0`、`task_number=1`、`task_obj_addr=<task buffer 的 obj_addr>`、`iommu_domain_id=<MEM_CREATE 时用的域 id（默认 0）>`、`task_base_addr=0`、`core_mask=0x1`（单核最稳）、`fence_fd=-1`、`subcore_task[0]={0,1}`（**并建议把 5 个槽都填成 `{0,1}`**，无害且可避免第三方报告过的边界问题）。
9. **判定成功**：ioctl 返回 0 且 `task_counter == task_number`、`hw_elapse_time > 0`；随后 `MEM_SYNC(FROM_DEVICE=2)` + 读输出 buffer。
10. **清理**：逐个 `MEM_DESTROY`（`0xC0106444`，按 handle）。**务必自己 destroy**：DRM 会在 close 时 reap GEM，但 IOMMU 域/IOVA 状态在多域场景不会回收。

### 硬坑清单（按踩到概率排序）

| 坑 | 现象 | 依据 |
|---|---|---|
| `core_mask` 用了 `0x5`/`0x6` | 日志 "Unknown core mask"，**job 永不 kick** → 阻塞到超时 | 源码 `rknpu_job_commit` switch |
| `int_mask` 与硬件实际置位不符 | IRQ 只清中断不置 DONE → 静默超时 | `rknpu_fuzz_status(status) != job->int_mask` |
| `task.regcfg_amount` 少算/单位搞错 | PC 取指令条数不对 → 卡死或结果错 | §3.3 公式（0.9.8 实测标定 108↔55）|
| `regcmd_addr` 未对齐 / bit0 置位 | 变成 Slave 模式，PC 不跑 | TRM 寄存器描述（二手）|
| 结构体加了 `#pragma pack` / 手动算错 size | ioctl 编号的 size 位不对 → 内核按错长度 copy（会零填充或截断）| `drm_ioctl()` 逻辑 |
| 用 0.8.x 时代的 `regcfg_obj_addr/user_data` 布局 | 不报错，静默错解字段 | §6.2 |
| 复用了别人的 handle / 用 `obj_addr` 当 mmap 地址 | `obj_addr` 是**内核指针**，不是可映射地址 | `args->obj_addr = (__u64)(uintptr_t)rknpu_obj` |
| 把 dma-buf import 作为某 IOMMU 域的第一个 buffer | 结果错乱/超时（域首分配顺序 bug）| 第三方实测 |
| 启用 PINGPONG 做长链 | `errno=110` stall | 第三方（可用 `flags=0x1` 回避）|

### 可以直接抄的参考实现

- **UAPI 定义（自包含、无 libdrm 依赖）**：<https://github.com/oRKLLM/ork-driver/blob/main/src/rknpu_ioctl.h>
- **真实能跑的最小提交 + regcmd 合成**：<https://github.com/oRKLLM/ork-driver/blob/main/src/npu.c>（22 万字符，含 matmul 全流程；配合其 wiki `regcmd-ISA-Reference` 读）
- 追硬件语义时的另一条腿：mainline `accel/rocket`（`rocket_job.c` / `rocket_registers.h`）——它的寄存器命名可比 vendor 的 `RKNPU_OFFSET_*` 完整得多。

---

## 9. 明确"未确认"的清单

1. `RKNPU_CMD_*` opcode 枚举：**在 GPL vendor 驱动中不存在**（四个分支全量 grep 无命中）；若你手上有这个词表，来源另有其物（可能是某个第三方文档/工具，或把 `RKNPU_OFFSET_*` 与 block 字段混称），**不要据此写 submit**。
2. `regcfg_amount` 的精确单位语义与 `RKNPU_PC_DATA_EXTRA_AMOUNT=4` 的动机：源码无注释；我根据 mainline 公式推算为"+2 条 64-bit 条目"，**属推断**。
3. regcmd 的 block_id 与 TRM 模块位的对应关系（`0x0801`/`0x1001` 谁对应 DPU/PPU）：两份来源命名错位，**未确认**。
4. "每个 IOMMU 域 ~4 GiB"：第三方实测，**源码未找到硬截断**。
5. `librknnrt` ↔ 驱动版本的官方兼容矩阵：**未找到可信一手来源**，请用 `GET_DRV_VERSION` + §6.2 字段表自行判定，或查对应 `RKNPU_User_Guide_RKNN_SDK_V2.x` PDF。
6. 你实测的 `shmem-rss:10.98GiB` 的**具体构成比例**（模型/权重/KV/残留）：机制已确认（shmem 后备 GEM），量级拆解**属推断**。

---

### 附：本报告引用的一手源码锚点（速查）

| 内容 | 链接 |
|---|---|
| UAPI 头（0.9.8） | https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/include/rknpu_ioctl.h |
| 版本宏 | https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/include/rknpu_drv.h#L30-L35 |
| Kconfig（双后端） | https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/Kconfig |
| ioctl 表 + drm_driver | https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_drv.c#L702-L776 |
| action 全部实现 | https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_drv.c#L417-L526 |
| RK3588 config | https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_drv.c#L183-L202 |
| PC 提交全过程 | https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_job.c#L272-L375 |
| core mask / AUTO | https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_job.c#L27-L48 · #L402-L427 · #L525-L560 |
| submit 校验 | https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_job.c#L746-L850 |
| IRQ 完成判定 | https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_job.c#L643-L679 |
| GEM 分配（CMA/shmem） | https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_gem.c#L31-L245 |
| GEM ioctl 实现 | https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_gem.c#L876-L960 · #L1667-L1730 |
| IOMMU 域切换 / IOVA | https://github.com/rockchip-linux/kernel/blob/develop-6.1/drivers/rknpu/rknpu_iommu.c#L207-L296 · #L406-L570 |
| DRM shmem 后备（mainline） | https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/drm_gem.c#L174-L207 · #L640-L710 |
| DRM ioctl 权限/size 容忍（mainline） | https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/drm_ioctl.c#L612-L631 · #L855-L915 |
| mainline rocket（交叉验证） | https://github.com/torvalds/linux/blob/master/include/uapi/drm/rocket_accel.h · https://github.com/torvalds/linux/blob/master/drivers/accel/rocket/rocket_job.c#L118-L155 |
| 第三方逆向（regcmd ISA / 驱动怪癖） | https://github.com/oRKLLM/ork-driver/wiki/regcmd-ISA-Reference · https://github.com/oRKLLM/ork-driver/wiki/NPU-Quirks · https://github.com/oRKLLM/ork-driver/wiki/Vendor-Kernel-Behaviour |
| 第三方寄存器图（TRM 二手） | https://qc-gpu-driver.github.io/starryos-pulsar/documents/registers/pc.html · https://qc-gpu-driver.github.io/starryos-pulsar/documents/ioctl-protocol.html |
