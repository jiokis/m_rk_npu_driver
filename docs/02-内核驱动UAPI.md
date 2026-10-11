# 内核驱动 UAPI 分析（一手材料）

> **来源**：Rockchip vendor 内核 `linux-6.1` 的 `drivers/rknpu/`，GPL-2.0，本地副本在 `vendor/rknpu-6.1/`
> **驱动版本**：v0.9.8（板子实测 `/sys/kernel/debug/rknpu/version`）
> **设备节点**：`/dev/dri/renderD129`（DRM render node）
> **状态**：以下全部为**逐行读源码所得**，非二手资料

---

## 一、最重要的结论：内核驱动不解析 regcmd

这是整个逆向工作的战略前提。

`struct rknpu_task` 里跟命令相关的字段只有两个地址：

```c
__u32 regcfg_amount;   /* 寄存器配置的条数 */
__u32 regcfg_offset;   /* 寄存器配置的偏移 */
__u64 regcmd_addr;     /* 命令流在设备地址空间里的地址 */
```

驱动拿到的就是这个结构体数组，它做的事是：
1. 把 task 结构拷进 DMA 缓冲
2. 写几个控制寄存器（PC 模式、中断掩码、使能位）
3. 等中断

**它从头到尾没有一行代码去解析 `regcmd_addr` 指向的内容。**

推论（对逆向路线有决定性影响）：

| | 含义 |
|---|---|
| ✅ **好消息** | 硬件接口对内核是透明的，意味着我们**不需要改内核**就能提交自制命令流 |
| ❌ **坏消息** | `regcmd` 的 opcode 格式**不可能**从 GPL 源码得到，它只存在于闭源 `librknnrt.so` 和硬件里 |
| 🎯 **切入点** | 内核驱动给了我们**控制面**（怎么提交），缺的是**数据面**（提交什么）。后者只能靠 dump + 试错逆出来 |

所以正确路线是：**用开源驱动当"发射器"，把 librknnrt 的真实命令流抓下来当"弹药样本"，然后反推格式、自己造弹药。**

---

## 二、IOCTL 全清单

驱动是 **DRM 驱动**，所以 IOCTL 编号走 `DRM_COMMAND_BASE + nr`：

```c
#define RKNPU_ACTION       0x00
#define RKNPU_SUBMIT       0x01
#define RKNPU_MEM_CREATE   0x02
#define RKNPU_MEM_MAP      0x03
#define RKNPU_MEM_DESTROY  0x04
#define RKNPU_MEM_SYNC     0x05
#define RKNPU_IOC_MAGIC    'r'
```

| IOCTL | 作用 |
|---|---|
| `DRM_IOCTL_RKNPU_ACTION` | 杂项控制：查版本、调频调压、复位、上下电、查 SRAM、IOMMU domain |
| `DRM_IOCTL_RKNPU_SUBMIT` | **提交任务**（核心） |
| `DRM_IOCTL_RKNPU_MEM_CREATE` | 分配设备可见内存，返回 handle + dma_addr |
| `DRM_IOCTL_RKNPU_MEM_MAP` | 拿一个可 mmap 的伪偏移 |
| `DRM_IOCTL_RKNPU_MEM_DESTROY` | 释放 |
| `DRM_IOCTL_RKNPU_MEM_SYNC` | cache 一致性同步（TO/FROM device） |

最小用户态程序只需要这 6 个就能跑起来。

---

## 三、关键结构体（可直接用于编译用户态程序）

### 3.1 任务提交

```c
struct rknpu_submit {
	__u32 flags;            /* e_rknpu_job_mode 位或 */
	__u32 timeout;          /* 超时（ms？待确认） */
	__u32 task_start;       /* 起始 task 下标 */
	__u32 task_number;      /* task 个数 */
	__u32 task_counter;     /* 累计计数器 */
	__s32 priority;
	__u64 task_obj_addr;    /* task 数组的设备地址 */
	__u32 iommu_domain_id;
	__u32 reserved;
	__u64 task_base_addr;
	__s64 hw_elapse_time;   /* 出参：硬件耗时 */
	__u32 core_mask;        /* 三核选择 */
	__s32 fence_fd;         /* dma-fence 同步 */
	struct rknpu_subcore_task subcore_task[5];
};
```

> `subcore_task[5]`：**注意是 5 而不是 3**。RK3588 的 NPU 有 3 核，这个数组留到 5 说明驱动为更多核的 SoC（如 RK3576/RK3562 之类）预留了。

### 3.2 单个任务（**packed**，注意对齐）

```c
struct rknpu_task {
	__u32 flags;
	__u32 op_idx;           /* 算子序号 */
	__u32 enable_mask;
	__u32 int_mask;         /* 中断掩码 */
	__u32 int_clear;        /* 中断清除 */
	__u32 int_status;       /* 出参：中断状态 */
	__u32 regcfg_amount;    /* 寄存器配置条数 */
	__u32 regcfg_offset;
	__u64 regcmd_addr;      /* ← 命令流地址（逆向的核心目标） */
} __packed;
```

`__packed` 意味着**没有隐式填充**：`4*7 + 8 = 36` 字节。写用户态程序时结构体必须也 packed，否则 offset 全错。

### 3.3 内存创建

```c
struct rknpu_mem_create {
	__u32 handle;           /* 出参：GEM handle */
	__u32 flags;            /* e_rknpu_mem_type 位或 */
	__u64 size;
	__u64 obj_addr;         /* 出参 */
	__u64 dma_addr;         /* 出参：★ 设备侧地址，regcmd 里要用这个 */
	__u64 sram_size;
	__s32 iommu_domain_id;
	__u32 core_mask;
};
```

**`dma_addr` 是关键出参** —— 所有写进 `regcmd` 的地址必须是设备地址（IOVA），不是 CPU 虚拟地址。

### 3.4 内存同步

```c
struct rknpu_mem_sync {
	__u32 flags;            /* TO_DEVICE=1<<0  FROM_DEVICE=1<<1 */
	__u32 reserved;
	__u64 obj_addr;
	__u64 offset;
	__u64 size;
};

struct rknpu_mem_map   { __u32 handle; __u32 reserved; __u64 offset; };
struct rknpu_mem_destroy { __u32 handle; __u32 reserved; __u64 obj_addr; };
struct rknpu_action    { __u32 flags; __u32 value; };
```

---

## 四、ACTION 子命令全表（26 个）

`struct rknpu_action { flags; value; }` —— `flags` 选子命令，`value` 读写值。

| # | 名称 | 用途 |
|---|---|---|
| 0 | `RKNPU_GET_HW_VERSION` | 查硬件版本 |
| 1 | `RKNPU_GET_DRV_VERSION` | 查驱动版本 |
| 2/3 | `GET/SET_FREQ` | 调频（实测 300–1000 MHz 8 档） |
| 4/5 | `GET/SET_VOLT` | 调压 |
| 6 | `RKNPU_ACT_RESET` | **软件复位**（逆向搞挂硬件时救命） |
| 7/8 | `GET/SET_BW_PRIORITY` | 带宽优先级 |
| 9/10 | `GET/SET_BW_EXPECT` | 期望带宽 |
| 11/12 | `GET/SET_BW_TW` | 带宽时间窗 |
| 13 | `ACT_CLR_TOTAL_RW_AMOUNT` | 清累计读写量 |
| 14/15/16 | `GET_DT_WR/RD_AMOUNT`, `GET_WT_RD_AMOUNT` | 读写量统计（**做性能分析用**） |
| 17 | `GET_TOTAL_RW_AMOUNT` | 总量 |
| 18 | `GET_IOMMU_EN` | IOMMU 是否启用 |
| 19 | `SET_PROC_NICE` | 进程优先级 |
| 20/21 | `POWER_ON` / `POWER_OFF` | **手动上下电**（逆向调试用，可省去反复加载模型） |
| 22/23 | `GET_TOTAL/FREE_SRAM_SIZE` | SRAM 容量（**SRAM 比 DRAM 快，是优化空间**） |
| 24/25 | `GET/SET_IOMMU_DOMAIN_ID` | 多进程地址空间隔离 |

> 🔧 **对逆向最有用的三个**：`ACT_RESET`（搞挂了能复位）、`POWER_ON/OFF`（不重载模型就能反复试）、`GET_TOTAL/FREE_SRAM_SIZE`（找更快的暂存区）。

---

## 五、内存类型标志（优化空间所在）

```c
enum e_rknpu_mem_type {
	RKNPU_MEM_CONTIGUOUS            = 0 << 0,  /* 物理连续，默认 */
	RKNPU_MEM_NON_CONTIGUOUS        = 1 << 0,
	RKNPU_MEM_NON_CACHEABLE         = 0 << 1,  /* 默认不可 cache */
	RKNPU_MEM_CACHEABLE             = 1 << 1,
	RKNPU_MEM_WRITE_COMBINE         = 1 << 2,
	RKNPU_MEM_KERNEL_MAPPING        = 1 << 3,
	RKNPU_MEM_IOMMU                 = 1 << 4,
	RKNPU_MEM_ZEROING               = 1 << 5,
	RKNPU_MEM_SECURE                = 1 << 6,
	RKNPU_MEM_DMA32                 = 1 << 7,
	RKNPU_MEM_TRY_ALLOC_SRAM        = 1 << 8,   /* ★ 尝试放 SRAM */
	RKNPU_MEM_TRY_ALLOC_NBUF        = 1 << 9,   /* ★ NBUF，疑似片上专用缓冲 */
	RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT = 1 << 10,
};
```

> 💡 **`TRY_ALLOC_SRAM` 和 `TRY_ALLOC_NBUF` 是重点**。RKLLM 把整个模型塞进 DRAM 靠带宽硬扛
> （我们实测有效带宽上限 25.9 GB/s），而 **SRAM 的带宽和延迟都远优于 DRAM**。
> 如果能把热点权重/激活值钉在 SRAM 里，是绕过 DRAM 带宽墙的**唯一**硬件级手段。
> 这需要先知道 SRAM 有多大 —— 用 `GET_TOTAL_SRAM_SIZE` 查。

---

## 六、任务模式标志

```c
enum e_rknpu_job_mode {
	RKNPU_JOB_SLAVE     = 0 << 0,   /* 从模式 */
	RKNPU_JOB_PC        = 1 << 0,   /* ★ PC 模式：硬件从内存取命令流执行 */
	RKNPU_JOB_BLOCK     = 0 << 1,
	RKNPU_JOB_NONBLOCK  = 1 << 1,
	RKNPU_JOB_PINGPONG  = 1 << 2,   /* 乒乓缓冲 */
	RKNPU_JOB_FENCE_IN  = 1 << 3,
	RKNPU_JOB_FENCE_OUT = 1 << 4,
};
```

**`RKNPU_JOB_PC` 是理解 regcmd 的关键**：PC = Program Counter。硬件把 `regcmd_addr`
当作**指令流入口**，逐条取指执行。所以 regcmd 是一段**由硬件解释的字节码**，
而不是一堆被动写入的寄存器值。这解释了为什么格式必须靠逆向。

`PINGPONG` 说明硬件支持双缓冲流水 —— 也是优化空间（重叠计算与搬运）。

---

## 七、寄存器偏移（regcmd 的"坐标系"）

**这是全文件最有价值的一段**，它泄露了硬件寄存器布局：

```c
#define RKNPU_OFFSET_VERSION          0x0
#define RKNPU_OFFSET_VERSION_NUM      0x4
#define RKNPU_OFFSET_PC_OP_EN         0x8    /* PC 操作使能 */
#define RKNPU_OFFSET_PC_DATA_ADDR     0x10   /* ★ PC 数据地址 */
#define RKNPU_OFFSET_PC_DATA_AMOUNT   0x14   /* ★ PC 数据长度 */
#define RKNPU_OFFSET_INT_MASK         0x20
#define RKNPU_OFFSET_INT_CLEAR        0x24
#define RKNPU_OFFSET_INT_STATUS       0x28
#define RKNPU_OFFSET_INT_RAW_STATUS   0x2c
#define RKNPU_OFFSET_PC_TASK_CONTROL  0x30   /* 任务控制 */
#define RKNPU_OFFSET_PC_DMA_BASE_ADDR 0x34   /* ★ DMA 基址 */
#define RKNPU_OFFSET_ENABLE_MASK      0xf008
#define RKNPU_INT_CLEAR               0x1ffff  /* 17 个中断位 */
#define RKNPU_PC_DATA_EXTRA_AMOUNT    4
```

可以读出几件事：

1. **寄存器空间至少到 `0xf008`**（`ENABLE_MASK`），说明地址空间不小，不是几十个寄存器的小外设。
2. **`PC_DATA_ADDR` / `PC_DATA_AMOUNT` 成对出现**（0x10/0x14）—— 硬件自己能做"按地址+长度取数据"，
   这暗示 regcmd 里有 **DMA 类指令**（不是纯计算）。
3. **`PC_DMA_BASE_ADDR`（0x34）** 说明 DMA 地址有一个基址寄存器 —— 很可能是**地址压缩**机制。
   如果 regcmd 里存的是相对基址的偏移（16/24 位），就能大幅缩小命令流体积。
4. **`INT_CLEAR = 0x1ffff`** → **17 个中断源**。可以据此推断硬件内部有 17 类事件
   （算完、DMA 完、出错、各子模块就绪……），反推模块划分。
5. **`PC_DATA_EXTRA_AMOUNT = 4`** 含义未确认，疑似取数据时的额外字节数（对齐或头部）。

---

## 八、对逆向路线的直接指导

基于以上，最小可行路径是：

```
① 查底座
   ACTION: GET_HW_VERSION / GET_DRV_VERSION / GET_IOMMU_EN
   ACTION: GET_TOTAL_SRAM_SIZE / GET_FREE_SRAM_SIZE     ← 找 SRAM 有多大

② 建内存
   MEM_CREATE(size, flags=CONTIGUOUS|IOMMU) → handle, dma_addr
   MEM_MAP(handle) → offset ; mmap(offset) → CPU 指针
   （此时 CPU 指针和 dma_addr 指向同一块物理内存，两个视角）

③ 抓弹药
   strace -e ioctl -x -s 4096 跑 librknnrt 的一次推理
   → 拿到真实的 rknpu_submit / rknpu_task 参数
   → 顺着 regcmd_addr 把那段内存 dump 出来  ← 逆向的真正素材

④ 反推格式
   对比 dump 出来的字节流与 librknnrt 里字符串/常量
   找出 opcode 编码规律（很可能是"定长指令 + 变长载荷"）

⑤ 自造弹药
   先不做计算，只做「能让硬件不报错地跑完」的最小命令流（空转 / 触发一次中断）
   成功后再逐步加 DMA、加计算

⑥ 兜底
   任何时候硬件挂了 → ACTION: ACT_RESET
   反复调试 → ACTION: POWER_ON / POWER_OFF 省去重载模型
```

**第 ③ 步的 dump 是整个项目的转折点** —— 在那之前都是准备工作，在那之后才有真实素材。

---

## 九、待确认项

| 问题 | 怎么查 |
|---|---|
| `regcfg_amount` / `regcfg_offset` 的确切语义 | 读 `rknpu_job.c` 里怎么用它们 |
| `timeout` 单位 | 读 `rknpu_job.c` |
| `task_base_addr` 与 `task_obj_addr` 的区别 | 同上 |
| `RKNPU_PC_DATA_EXTRA_AMOUNT = 4` 是什么 | 需结合硬件行为或 librknnrt 反汇编 |
| SRAM 实际容量 | 板上跑 `ACTION: GET_TOTAL_SRAM_SIZE` |
| regcmd opcode 表 | **只能从 librknnrt dump 逆** |
