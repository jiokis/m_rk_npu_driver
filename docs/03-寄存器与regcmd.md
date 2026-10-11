# 寄存器布局与 regcmd 提交机制

> **来源**：`rknpu_job.c` / `rknpu_drv.c` 逐行阅读（GPL 源码，一手材料）
> 配套阅读：[02-内核驱动UAPI.md](02-内核驱动UAPI.md)

---

## 一、硬件被这样编程（提交一次任务的完整序列）

`rknpu_job_subcore_commit_pc()` 是全部秘密所在。它就是驱动对 NPU 说的全部话：

```c
/* ① 命令流入口地址 */
REG_WRITE(first_task->regcmd_addr, RKNPU_OFFSET_PC_DATA_ADDR);

/* ② 命令流长度（推导见下节） */
REG_WRITE((first_task->regcfg_amount + RKNPU_PC_DATA_EXTRA_AMOUNT +
           pc_data_amount_scale - 1) / pc_data_amount_scale - 1,
          RKNPU_OFFSET_PC_DATA_AMOUNT);

/* ③ 中断掩码 / 清除 */
REG_WRITE(last_task->int_mask,  RKNPU_OFFSET_INT_MASK);
REG_WRITE(first_task->int_mask, RKNPU_OFFSET_INT_CLEAR);

/* ④ 任务控制：模式字段 + 本批任务数 */
REG_WRITE(((0x6 | task_pp_en) << pc_task_number_bits) | task_number,
          RKNPU_OFFSET_PC_TASK_CONTROL);

/* ⑤ DMA 基址（regcmd 内 DMA 指令的地址基准） */
REG_WRITE(args->task_base_addr, RKNPU_OFFSET_PC_DMA_BASE_ADDR);

/* ⑥ 触发：先置 1 再置 0（上升沿启动） */
REG_WRITE(0x1, RKNPU_OFFSET_PC_OP_EN);
REG_WRITE(0x0, RKNPU_OFFSET_PC_OP_EN);
```

**就这么多。** 写完这 7 个寄存器，硬件自己跑去执行命令流，算完发中断。
内核驱动到此为止 —— 这也是为什么不需要改内核就能做逆向。

---

## 二、RK3588 的硬件常量（`rk3588_rknpu_config`）

```c
static const struct rknpu_config rk3588_rknpu_config = {
	.dma_mask               = DMA_BIT_MASK(40),   /* 40 位地址空间 */
	.pc_data_amount_scale   = 2,                  /* 长度以 16 位为单位 */
	.pc_task_number_bits    = 12,                 /* 任务数字段占 12 位 */
	.pc_task_number_mask    = 0xfff,
	.pc_task_status_offset  = 0x3c,               /* 任务状态寄存器 */
	.pc_dma_ctrl            = 0,                  /* 不走 spinlock 分支 */
	.num_irqs               = 3,                  /* 三核各一个中断 */
	.nbuf_phyaddr           = 0,  .nbuf_size = 0, /* 本 SoC 无 NBUF */
	.max_submit_number      = (1 << 12) - 1,      /* 4095 */
	.core_mask              = 0x7,                /* 三位，三核 */
};
```

### 2.1 推导 `PC_DATA_AMOUNT` 的真实含义

代入 `scale = 2`、`EXTRA = 4`：

```
寄存器值 = ⌈(regcfg_amount + 4) / 2⌉ - 1
```

反推：硬件读的字节数 = `regcfg_amount + 4`，寄存器存的是「**16 位单位数 − 1**」。

**结论：`regcfg_amount` 是 regcmd 命令流的字节长度。**
那多出来的 `+4` 字节（`RKNPU_PC_DATA_EXTRA_AMOUNT`）是硬件额外读的 4 字节尾部 ——
很可能是**命令流终止符**或状态回写槽。这一条要靠 dump 验证。

> 这意味着：只要我们在用户态填好 `regcfg_amount`（长度）和 `regcmd_addr`（地址），
> 硬件就会老老实实执行我们自己造的命令流。**这就是突破口。**

### 2.2 每个 NPU 核占 256 MB 寄存器窗口

`rknpu_job_subcore_commit_pc()` 里有这么一段：

```c
for (i = 0; i < rknpu_dev->config->num_irqs; i++) {
	if (i == core_index) {
		REG_WRITE((0xe + 0x10000000 * i), 0x1004);
		REG_WRITE((0xe + 0x10000000 * i), 0x3004);
	}
}
```

`0x10000000 * i` → **核间距 = 0x10000000 = 256 MB**：

| 核 | 寄存器窗口基址 |
|---|---|
| core0 | `0x0000_0000` |
| core1 | `0x1000_0000` |
| core2 | `0x2000_0000` |

而写的是 `0x1004` / `0x3004` 两个偏移，值 `0xe` —— 疑似**核间同步/中断路由**寄存器。
这两个偏移不在 `rknpu_ioctl.h` 的偏移表里，属于**未文档化的控制寄存器**，值得单独试。

### 2.3 中断模型

`RKNPU_INT_CLEAR = 0x1ffff` → **17 个中断源**（bit0..bit16）。
结合 `pc_task_status_offset = 0x3c`，可见 `0x20–0x3c` 这段是**中断与状态区**：

```
0x20  INT_MASK
0x24  INT_CLEAR
0x28  INT_STATUS
0x2c  INT_RAW_STATUS
0x30  PC_TASK_CONTROL
0x34  PC_DMA_BASE_ADDR
0x3c  PC_TASK_STATUS      ← 从 config 得到，不在头文件里
```

17 个中断源大致对应：计算完成、DMA 完成、各子模块就绪/错误等。
**具体哪一位是什么，只能靠触发实验枚举。**

---

## 三、当前的知识缺口（这是逆向的真正战场）

| 已知 ✅ | 未知 ❌ |
|---|---|
| 怎么分配设备内存（6 个 ioctl） | **regcmd 的指令编码** |
| 怎么提交（7 个寄存器写） | 有哪些 opcode、各占几字节 |
| 命令流的地址与**长度**怎么给 | 长度字段之外是否还有头部 |
| 三核怎么选、怎么分任务 | 算子怎么描述（卷积/矩阵乘的参数格式） |
| 内存类型与 SRAM/NBUF 标志位 | SRAM 实际多大、NBUF 是什么 |
| 中断有 17 个源 | 每个中断位的确切含义 |
| 每核 256MB 寄存器窗口 | `0x1004`/`0x3004` 两个魔数寄存器干什么的 |

**所有 ❌ 都只能从 `librknnrt.so` 的运行时行为逆出来** —— 因为是闭源二进制，
它构造命令流的逻辑不在任何可读源码里。

---

## 四、下一步：抓真实命令流

计划（工具在 `src/probes/`）：

```
1. 在板上 strace 一次真实的 rknn 推理，抓全部 ioctl
   → 得到 rknpu_submit / rknpu_task 的真实参数
   → 特别是 regcmd_addr 和 regcfg_amount

2. 顺着 regcmd_addr，把那段设备内存 dump 出来
   → 得到真实的命令流字节流（第一手素材）

3. 对比多次不同算子的推理
   → 找出哪些字节随算子类型变化（opcode）
   → 哪些随张量形状变化（参数）

4. 小步试错
   → 先提交空命令流看是否报错
   → 再加单条指令
   → 用 ACTION: ACT_RESET / POWER_ON/OFF 兜底
```

**第 2 步是整个项目的转折点。** 在拿到真实命令流字节之前，一切都是推测。
