/*
 * rknpu_poc.c — RK3588 NPU 最小用户态程序（不依赖 libdrm）
 *
 * 目的：验证我们能绕开 rknn-toolkit2 / librknnrt，直接跟 NPU 内核驱动对话。
 * 分阶段执行，每阶段独立可验证：
 *
 *   ./rknpu_poc query          ① 取硬件/驱动版本、IOMMU 状态、SRAM 容量
 *   ./rknpu_poc mem            ② 分配内存 → MEM_MAP → mmap → 写模式 → MEM_SYNC → 读回
 *   ./rknpu_poc counter        ③ 读**硬件带宽计数器**（"真算了没"的判据）
 *   ./rknpu_poc submit-safe    ④ 提交自制命令流（仅 OP_NONE，安全）
 *   ./rknpu_poc submit-enable  ⑤ 含 OP_ENABLE 的流（⚠️ 危险，曾搞挂板子，见 docs/07）
 *
 * ⚠️ 事故教训：曾经在没有任何算子描述符时就写 OP_ENABLE，硬件拿未初始化的
 *    描述符做 DMA，写坏内存导致内核崩溃、板子需物理断电。所以：
 *    · 第一发实验永远只用 OP_NONE，目标是"让 ioctl 返回"而非"让硬件干活"
 *    · 使能位必须在所有描述符配好之后才能写
 *
 * 结构体定义复刻自 drivers/rknpu/include/rknpu_ioctl.h (GPL-2.0)，
 * 交叉核对了 mainline accel/rocket 与第三方 oRKLLM/ork-driver 的资料。
 *
 * 编译: gcc -O2 -Wall -o rknpu_poc rknpu_poc.c
 * 运行: sudo ./rknpu_poc query     （需要能访问 /dev/dri/renderD129）
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

/* ─────────── UAPI（复刻自 rknpu_ioctl.h）─────────── */
struct rknpu_action      { uint32_t flags; uint32_t value; };
struct rknpu_mem_map     { uint32_t handle; uint32_t reserved; uint64_t offset; };
struct rknpu_mem_destroy { uint32_t handle; uint32_t reserved; uint64_t obj_addr; };
struct rknpu_mem_sync    { uint32_t flags; uint32_t reserved; uint64_t obj_addr;
                           uint64_t offset; uint64_t size; };
struct rknpu_mem_create  { uint32_t handle; uint32_t flags; uint64_t size;
                           uint64_t obj_addr; uint64_t dma_addr; uint64_t sram_size;
                           int32_t iommu_domain_id; uint32_t core_mask; };
struct rknpu_task        { uint32_t flags, op_idx, enable_mask, int_mask, int_clear,
                           int_status, regcfg_amount, regcfg_offset;
                           uint64_t regcmd_addr; } __attribute__((packed));
struct rknpu_subcore_task { uint32_t task_start, task_number; };
struct rknpu_submit      { uint32_t flags, timeout, task_start, task_number, task_counter;
                           int32_t priority; uint64_t task_obj_addr;
                           uint32_t iommu_domain_id, reserved; uint64_t task_base_addr;
                           int64_t hw_elapse_time; uint32_t core_mask; int32_t fence_fd;
                           struct rknpu_subcore_task subcore_task[5]; };

/* 枚举 */
#define ACT_GET_HW_VERSION        0
#define ACT_GET_DRV_VERSION       1
#define ACT_GET_FREQ              2
#define ACT_ACT_RESET             6
#define ACT_GET_DT_WR_AMOUNT      14
#define ACT_GET_DT_RD_AMOUNT      15
#define ACT_GET_TOTAL_RW_AMOUNT   17
#define ACT_GET_IOMMU_EN          18
#define ACT_POWER_ON              20
#define ACT_POWER_OFF             21
#define ACT_GET_TOTAL_SRAM_SIZE   22
#define ACT_GET_FREE_SRAM_SIZE    23
#define ACT_GET_IOMMU_DOMAIN_ID   24

/* 内存标志 */
#define MEM_NON_CONTIGUOUS   (1u << 0)
#define MEM_CACHEABLE        (1u << 1)
#define MEM_WRITE_COMBINE    (1u << 2)
#define MEM_KERNEL_MAPPING   (1u << 3)
#define MEM_IOMMU            (1u << 4)
#define MEM_ZEROING          (1u << 5)
#define MEM_DMA32            (1u << 7)
#define MEM_TRY_ALLOC_SRAM   (1u << 8)
#define MEM_TRY_ALLOC_NBUF   (1u << 9)
#define MEM_IOMMU_LIMIT_IOVA_ALIGNMENT (1u << 10)

/* Job 模式 */
#define JOB_PC         (1u << 0)
#define JOB_NONBLOCK   (1u << 1)
#define JOB_PINGPONG   (1u << 2)

/* DRM ioctl 编号 */
#define DRM_IOCTL_RKNPU_ACTION      _IOWR('d', 0x40, struct rknpu_action)
#define DRM_IOCTL_RKNPU_SUBMIT      _IOWR('d', 0x41, struct rknpu_submit)
#define DRM_IOCTL_RKNPU_MEM_CREATE  _IOWR('d', 0x42, struct rknpu_mem_create)
#define DRM_IOCTL_RKNPU_MEM_MAP     _IOWR('d', 0x43, struct rknpu_mem_map)
#define DRM_IOCTL_RKNPU_MEM_DESTROY _IOWR('d', 0x44, struct rknpu_mem_destroy)
#define DRM_IOCTL_RKNPU_MEM_SYNC    _IOWR('d', 0x45, struct rknpu_mem_sync)

static const char *DEV = "/dev/dri/renderD129";
static int fd = -1;

static int action(uint32_t f, uint32_t *v)
{
	struct rknpu_action a = { .flags = f, .value = v ? *v : 0 };
	int r = ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a);
	if (v) *v = a.value;
	return r;
}

static int mem_create(uint32_t flags, uint64_t size, struct rknpu_mem_create *out)
{
	struct rknpu_mem_create m;
	memset(&m, 0, sizeof(m));
	m.flags = flags;
	m.size = size;
	if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &m) < 0) return -errno;
	*out = m;
	return 0;
}

static void *mem_map_to_cpu(struct rknpu_mem_create *m)
{
	struct rknpu_mem_map mm = { .handle = m->handle, .offset = 0 };
	if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mm) < 0) return NULL;
	void *p = mmap(NULL, m->size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mm.offset);
	return (p == MAP_FAILED) ? NULL : p;
}

static int mem_sync(uint64_t obj_addr, uint32_t flags, uint64_t off, uint64_t size)
{
	struct rknpu_mem_sync s = { .flags = flags, .obj_addr = obj_addr,
	                            .offset = off, .size = size };
	return ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &s);
}

/* ─────────── ① query ─────────── */
static int stage_query(void)
{
	printf("═══ ① UAPI 自检与能力查询 ═══\n\n");

	printf("  ── ioctl 编号（应与调研报告一致）──\n");
	printf("    ACTION      = 0x%08lx   (期望 0xc0086440)\n", (unsigned long)DRM_IOCTL_RKNPU_ACTION);
	printf("    SUBMIT      = 0x%08lx   (期望 0xc0686441)\n", (unsigned long)DRM_IOCTL_RKNPU_SUBMIT);
	printf("    MEM_CREATE  = 0x%08lx   (期望 0xc0306442)\n", (unsigned long)DRM_IOCTL_RKNPU_MEM_CREATE);
	printf("    MEM_MAP     = 0x%08lx   (期望 0xc0106443)\n", (unsigned long)DRM_IOCTL_RKNPU_MEM_MAP);
	printf("    MEM_DESTROY = 0x%08lx   (期望 0xc0106444)\n", (unsigned long)DRM_IOCTL_RKNPU_MEM_DESTROY);
	printf("    MEM_SYNC    = 0x%08lx   (期望 0xc0206445)\n", (unsigned long)DRM_IOCTL_RKNPU_MEM_SYNC);
	printf("    sizeof: action=%zu submit=%zu mem_create=%zu task=%zu\n\n",
	       sizeof(struct rknpu_action), sizeof(struct rknpu_submit),
	       sizeof(struct rknpu_mem_create), sizeof(struct rknpu_task));

	struct { uint32_t f; const char *n; } qs[] = {
		{ ACT_GET_HW_VERSION,      "GET_HW_VERSION" },
		{ ACT_GET_DRV_VERSION,     "GET_DRV_VERSION" },
		{ ACT_GET_IOMMU_EN,        "GET_IOMMU_EN" },
		{ ACT_GET_IOMMU_DOMAIN_ID, "GET_IOMMU_DOMAIN_ID" },
		{ ACT_GET_TOTAL_SRAM_SIZE, "GET_TOTAL_SRAM_SIZE" },
		{ ACT_GET_FREE_SRAM_SIZE,  "GET_FREE_SRAM_SIZE" },
		{ ACT_GET_TOTAL_RW_AMOUNT,"GET_TOTAL_RW_AMOUNT" },
	};
	printf("  ── ACTION 查询 ──\n");
	for (size_t i = 0; i < sizeof(qs) / sizeof(qs[0]); i++) {
		uint32_t v = 0;
		int r = action(qs[i].f, &v);
		if (r < 0) {
			printf("    %-24s ❌ errno=%d (%s)\n", qs[i].n, errno, strerror(errno));
			continue;
		}
		if (qs[i].f == ACT_GET_DRV_VERSION) {
			printf("    %-24s = %u  → v%u.%u.%u\n", qs[i].n, v,
			       v / 10000, (v % 10000) / 100, v % 100);
		} else if (qs[i].f == ACT_GET_HW_VERSION) {
			char s[5] = { (char)(v >> 24), (char)(v >> 16), (char)(v >> 8), (char)v, 0 };
			printf("    %-24s = 0x%08x  ASCII=\"%s\"\n", qs[i].n, v, s);
		} else {
			printf("    %-24s = %u (0x%x)\n", qs[i].n, v, v);
		}
	}
	return 0;
}

/* ─────────── ② mem ─────────── */
static int stage_mem(void)
{
	printf("═══ ② 内存通路验证 ═══\n\n");

	struct {
		const char *n; uint32_t flags;
	} cases[] = {
		{ "常规 0x403（运行时用的组合）", MEM_NON_CONTIGUOUS | MEM_CACHEABLE | MEM_IOMMU_LIMIT_IOVA_ALIGNMENT },
		{ "+ TRY_ALLOC_SRAM",            MEM_NON_CONTIGUOUS | MEM_CACHEABLE | MEM_IOMMU_LIMIT_IOVA_ALIGNMENT | MEM_TRY_ALLOC_SRAM },
		{ "+ TRY_ALLOC_NBUF",            MEM_NON_CONTIGUOUS | MEM_CACHEABLE | MEM_IOMMU_LIMIT_IOVA_ALIGNMENT | MEM_TRY_ALLOC_NBUF },
		{ "+ SRAM + NBUF 同时",          MEM_NON_CONTIGUOUS | MEM_CACHEABLE | MEM_IOMMU_LIMIT_IOVA_ALIGNMENT | MEM_TRY_ALLOC_SRAM | MEM_TRY_ALLOC_NBUF },
		{ "连续 + ZEROING（走 CMA）",     MEM_ZEROING },
	};

	const uint64_t SZ = 1 << 20;   /* 1 MiB */
	for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		struct rknpu_mem_create m;
		int r = mem_create(cases[i].flags, SZ, &m);
		if (r < 0) {
			printf("  %-32s ❌ errno=%d (%s)\n", cases[i].n, -r, strerror(-r));
			continue;
		}
		printf("  %-32s ✅ handle=%u dma=%#llx sram=%llu\n",
		       cases[i].n, m.handle, (unsigned long long)m.dma_addr,
		       (unsigned long long)m.sram_size);

		/* 对常规那条做完整的 mmap + 读写 + sync 验证 */
		if (i == 0) {
			void *p = mem_map_to_cpu(&m);
			if (!p) {
				printf("      ⚠️ mmap 失败\n");
			} else {
				printf("      mmap → %p，写入模式并回读\n", p);
				memset(p, 0xA5, SZ);
				if (mem_sync(m.obj_addr, 1 /*TO_DEVICE*/, 0, SZ) < 0)
					printf("      ⚠️ MEM_SYNC(TO_DEVICE) 失败 errno=%d\n", errno);
				uint8_t *b = p;
				int ok = 1;
				for (uint64_t k = 0; k < SZ; k += 4096)
					if (b[k] != 0xA5) { ok = 0; break; }
				printf("      回读校验: %s\n", ok ? "✅ 一致（CPU 侧映射有效）" : "❌ 不一致");
				munmap(p, SZ);
			}
		}
		struct rknpu_mem_destroy d = { .handle = m.handle, .obj_addr = m.obj_addr };
		ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &d);
	}
	return 0;
}

/* ─────────── 硬件带宽计数器：判断"真算了没有"的关键判据 ───────────
 *
 * ⚠️ 为什么这个必须有（血泪教训，见 docs/07）：
 *   外部调研记录了一个案例：252 个 job 全部报"完成"，但硬件带宽计数器
 *   读出来 dt_wr=0、wt_rd=0 —— 硬件根本没算，却报告成功。当事人因为缺少
 *   这个判据浪费了两个月，中间还发表过错误结论。
 *
 *   所以：**在写任何优化之前，必须先能区分「真算了」和「没算却报完成」。**
 *
 * 这三个 ACTION 读的是硬件自己的读写字节计数器，不依赖任何驱动推测：
 *   GET_DT_WR_AMOUNT(14) / GET_DT_RD_AMOUNT(15) / GET_WT_RD_AMOUNT(16)
 *   ACT_CLR_TOTAL_RW_AMOUNT(13) 清零，便于"归零→跑→读"的干净测量
 */
static int read_counters(uint32_t *dt_wr, uint32_t *dt_rd, uint32_t *wt_rd, uint32_t *total)
{
	uint32_t a = 0, b = 0, c = 0, d = 0;
	if (action(ACT_GET_DT_WR_AMOUNT,    &a) < 0) return -1;
	if (action(ACT_GET_DT_RD_AMOUNT,    &b) < 0) return -1;
	if (action(ACT_GET_WT_RD_AMOUNT,    &c) < 0) return -1;
	if (action(ACT_GET_TOTAL_RW_AMOUNT, &d) < 0) return -1;
	if (dt_wr) *dt_wr = a;
	if (dt_rd) *dt_rd = b;
	if (wt_rd) *wt_rd = c;
	if (total) *total = d;
	return 0;
}

static int stage_counter(void)
{
	printf("═══ ④ 硬件带宽计数器（\"真算了没\"判据）═══\n\n");
	uint32_t w, r, wt, tot;
	if (read_counters(&w, &r, &wt, &tot) < 0) {
		printf("  ❌ 读计数器失败 errno=%d (%s)\n", errno, strerror(errno));
		return -1;
	}
	printf("  DT_WR (数据通路 写)   = %u 0x%x\n", w, w);
	printf("  DT_RD (数据通路 读)   = %u 0x%x\n", r, r);
	printf("  WT_RD (权重通路 读)   = %u 0x%x\n", wt, wt);
	printf("  TOTAL_RW              = %u 0x%x\n", tot, tot);
	printf("\n  用法：清零(ACT_CLR_TOTAL_RW_AMOUNT) → 跑一次推理 → 再读。\n");
	printf("        总增量≈0 就说明硬件没真干活，哪怕 ioctl 报成功。\n");

	uint32_t t2;
	if (action(ACT_ACT_CLR_TOTAL_RW_AMOUNT, &t2) == 0) {
		printf("\n  已清零。清零后立即复读：\n");
		read_counters(&w, &r, &wt, &tot);
		printf("    DT_WR=%u DT_RD=%u WT_RD=%u TOTAL=%u\n", w, r, wt, tot);
	} else {
		printf("\n  ⚠️ 清零失败 errno=%d（某些硬件版本不支持）\n", errno);
	}
	return 0;
}

/* ─────────── ③ submit ─────────── */
/*
 * ★ 安全第一：默认只提交 OP_NONE
 *
 * 曾经的错误（已造成板子失联，见 docs/07 事故记录）：
 *   在**没有任何有效算子描述符**的情况下就写了 OP_ENABLE(block 0x0081)，
 *   硬件于是拿着未初始化的描述符去做 DMA，写坏内存 → 内核崩溃 → 需物理断电。
 *
 * 正确的最小实验应该是：
 *   ① 只发 OP_NONE —— 目标仅仅是"让 ioctl 正常返回"，不追求硬件干活
 *   ② 确认通路打通后，才逐步加指令，且**每加一条都先确认上一次正常**
 *   ③ 任何时候不要在没有描述符的情况下写使能位
 *
 * 危险实验（写 OP_ENABLE）被移到 submit-enable，且需要显式环境变量确认。
 */
static void put_cmd(uint8_t *buf, int idx, uint16_t reg_off, uint32_t val, uint16_t blk)
{
	uint8_t *p = buf + idx * 8;
	*(uint16_t *)(p + 0) = reg_off;        /* word0 低 16 位 = 寄存器偏移 */
	*(uint16_t *)(p + 2) = val & 0xffff;   /* word0 高 16 位 = 值低半 */
	*(uint16_t *)(p + 4) = val >> 16;      /* word1 低 16 位 = 值高半 */
	*(uint16_t *)(p + 6) = blk;            /* word1 高 16 位 = block_id */
}

/* 通用的提交骨架。cmds 为 NULL 表示只填 OP_NONE。 */
static int do_submit(int n_cmds, const uint32_t (*cmds)[3], const char *label)
{
	printf("  实验: %s  (%d 条指令)\n", label, n_cmds);

	/* 读基线计数器 —— 用于事后判断硬件有没有真干活 */
	uint32_t w0, r0, wt0, t0;
	read_counters(&w0, &r0, &wt0, &t0);
	printf("  基线计数器: DT_WR=%u DT_RD=%u WT_RD=%u TOTAL=%u\n", w0, r0, wt0, t0);

	struct rknpu_mem_create rc;
	int r = mem_create(MEM_NON_CONTIGUOUS | MEM_CACHEABLE | MEM_IOMMU_LIMIT_IOVA_ALIGNMENT, 4096, &rc);
	if (r < 0) { printf("  regcmd 缓冲分配失败 errno=%d\n", -r); return -1; }
	uint8_t *rc_cpu = mem_map_to_cpu(&rc);
	if (!rc_cpu) { printf("  regcmd mmap 失败\n"); return -1; }

	for (int i = 0; i < n_cmds; i++)
		put_cmd(rc_cpu, i, (uint16_t)cmds[i][0], cmds[i][1], (uint16_t)cmds[i][2]);
	uint32_t amount = n_cmds * 8;

	printf("  regcmd: %u 字节 @ dma=%#llx\n", amount, (unsigned long long)rc.dma_addr);
	for (int i = 0; i < n_cmds; i++) {
		uint8_t *p = rc_cpu + i * 8;
		printf("    [%d] reg=0x%04x val=0x%08x blk=0x%04x\n", i,
		       *(uint16_t *)p,
		       (*(uint16_t *)(p + 2)) | ((uint32_t)(*(uint16_t *)(p + 4)) << 16),
		       *(uint16_t *)(p + 6));
	}
	mem_sync(rc.obj_addr, 1, 0, amount);

	struct rknpu_mem_create tc;
	r = mem_create(MEM_NON_CONTIGUOUS | MEM_CACHEABLE | MEM_IOMMU_LIMIT_IOVA_ALIGNMENT, 4096, &tc);
	if (r < 0) { printf("  task 缓冲分配失败 errno=%d\n", -r); return -1; }
	struct rknpu_task *tasks = mem_map_to_cpu(&tc);
	if (!tasks) { printf("  task mmap 失败\n"); return -1; }

	memset(tasks, 0, 4096);
	tasks[0].int_mask      = 0x1ffff;   /* 全开，观察硬件到底置哪些位 */
	tasks[0].int_clear     = 0x1ffff;
	tasks[0].regcfg_amount = amount;
	tasks[0].regcmd_addr   = rc.dma_addr;
	mem_sync(tc.obj_addr, 1, 0, sizeof(struct rknpu_task));

	struct rknpu_submit s;
	memset(&s, 0, sizeof(s));
	s.flags = JOB_PC; s.timeout = 3000; s.task_start = 0; s.task_number = 1;
	s.task_obj_addr = tc.obj_addr; s.core_mask = 0x1; s.fence_fd = -1;
	s.subcore_task[0].task_number = 1;

	printf("\n  提交中 (core_mask=0x1, timeout=3000ms)…\n");
	fflush(stdout);
	r = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &s);
	int saved = errno;
	printf("  ioctl 返回 %d  errno=%d (%s)\n", r, r < 0 ? saved : 0, r < 0 ? strerror(saved) : "-");
	printf("  task_counter=%u  hw_elapse_time=%lld us\n", s.task_counter, (long long)s.hw_elapse_time);

	mem_sync(tc.obj_addr, 2, 0, sizeof(struct rknpu_task));
	printf("  task[0].int_status = %#x\n", tasks[0].int_status);

	/* ★ 关键：判断硬件有没有真干活 */
	uint32_t w1, r1, wt1, t1;
	read_counters(&w1, &r1, &wt1, &t1);
	printf("\n  ── 带宽计数增量（\"真算了没\"判据）──\n");
	printf("    DT_WR  %u → %u  (Δ%+d)\n", w0, w1, (int)(w1 - w0));
	printf("    DT_RD  %u → %u  (Δ%+d)\n", r0, r1, (int)(r1 - r0));
	printf("    WT_RD  %u → %u  (Δ%+d)\n", wt0, wt1, (int)(wt1 - wt0));

	long d = (long)(w1 - w0) + (long)(r1 - r0) + (long)(wt1 - wt0);
	if (r == 0 && s.task_counter == s.task_number && d == 0)
		printf("\n  ⚠️ **报成功但计数零增量 —— 这正是"假完成"的特征**（见 docs/07）\n");
	else if (d > 0)
		printf("\n  ✅ 有实际数据搬运，硬件真干活了\n");
	else
		printf("\n  ℹ️ 未正常完成，但拿到了诊断信息\n");

	struct rknpu_mem_destroy dd;
	dd = (struct rknpu_mem_destroy){ .handle = tc.handle, .obj_addr = tc.obj_addr };
	ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &dd);
	dd = (struct rknpu_mem_destroy){ .handle = rc.handle, .obj_addr = rc.obj_addr };
	ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &dd);
	return (r == 0 && s.task_counter == s.task_number) ? 0 : 1;
}

/* 最安全：整条流只有 OP_NONE。目标仅仅是"ioctl 能正常返回"。 */
static int stage_submit_safe(void)
{
	printf("═══ ③ 提交自制命令流（安全版：仅 OP_NONE）═══\n\n");
	printf("  目的：验证"我们构造的命令流能被硬件接受并让 ioctl 返回"。\n");
	printf("  不做任何计算、不使能任何模块、不引用任何数据地址。\n\n");
	static const uint32_t cmds[4][3] = {
		{ 0x0000, 0x00000000, 0x0000 },   /* OP_NONE ×4 */
		{ 0x0000, 0x00000000, 0x0000 },
		{ 0x0000, 0x00000000, 0x0000 },
		{ 0x0000, 0x00000000, 0x0000 },
	};
	int r = do_submit(4, cmds, "纯 OP_NONE");
	printf("\n  软复位兜底…\n");
	if (action(ACT_ACT_RESET, NULL) == 0) printf("    ✅ ACT_RESET ok\n");
	else printf("    ⚠️ ACT_RESET errno=%d\n", errno);
	return r;
}

/* 危险版：写 OP_ENABLE。必须显式确认，因为它能把板子搞挂。 */
static int stage_submit_enable(void)
{
	if (!getenv("RKNPU_I_KNOW_THE_RISK")) {
		printf("⛔ 这个实验写过 OP_ENABLE，曾在无描述符时让硬件拿垃圾地址 DMA，\n");
		printf("   把板子搞到需要物理断电（见 docs/07 事故记录）。\n\n");
		printf("   如果确实要继续，请：\n");
		printf("     1) 先确认已建立可观测性（能读带宽计数器）\n");
		printf("     2) 确认已配好所有算子描述符（不是全零！）\n");
		printf("     3) 用 RKNPU_I_KNOW_THE_RISK=1 重新运行\n");
		return 2;
	}
	printf("═══ ③ 提交自制命令流（危险版：含 OP_ENABLE）═══\n\n");
	printf("  ⚠️ 注意：本流**没有配置任何算子描述符**，仍然是不完整的实验。\n");
	printf("     仅用于验证"使能位本身"的行为，不要指望它算出东西。\n\n");
	static const uint32_t cmds[4][3] = {
		{ 0x0000, 0x00000000, 0x0041 },   /* OP_40 前置 */
		{ 0x0000, 0x00000000, 0x0000 },   /* OP_NONE */
		{ 0x0008, 0x00000060, 0x0081 },   /* OP_ENABLE —— 就是这条出的事 */
		{ 0x0000, 0x00000000, 0x0000 },
	};
	int r = do_submit(4, cmds, "OP_40 + OP_NONE + OP_ENABLE");
	printf("\n  软复位兜底…\n");
	if (action(ACT_ACT_RESET, NULL) == 0) printf("    ✅ ACT_RESET ok\n");
	else printf("    ⚠️ ACT_RESET errno=%d\n", errno);
	return r;
}

int main(int argc, char **argv)
{
	const char *mode = argc > 1 ? argv[1] : "query";

	fd = open(DEV, O_RDWR | O_CLOEXEC);
	if (fd < 0) {
		fprintf(stderr, "打不开 %s: %s\n（需要 root 或 video/render 组权限）\n",
		        DEV, strerror(errno));
		/* 退而求其次试 card 节点 */
		fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
		if (fd < 0) { fprintf(stderr, "card0 也不行: %s\n", strerror(errno)); return 2; }
		printf("（renderD129 打不开，改用 /dev/dri/card0）\n");
	}
	printf("已打开 %s (fd=%d)\n\n", DEV, fd);

	int r = 0;
	if      (!strcmp(mode, "query"))          r = stage_query();
	else if (!strcmp(mode, "mem"))            r = stage_mem();
	else if (!strcmp(mode, "counter"))        r = stage_counter();
	else if (!strcmp(mode, "submit-safe"))    r = stage_submit_safe();
	else if (!strcmp(mode, "submit-enable"))  r = stage_submit_enable();
	else if (!strcmp(mode, "submit")) {
		/* 兼容旧名字：现在指向安全版 */
		fprintf(stderr, "ℹ️  'submit' 已更名为 'submit-safe'（只发 OP_NONE）。\n"
		                "    危险实验请用 'submit-enable'。\n\n");
		r = stage_submit_safe();
	} else {
		fprintf(stderr,
		    "用法: %s <stage>\n"
		    "  query         取硬件/驱动版本、IOMMU、SRAM 容量\n"
		    "  mem           内存分配→mmap→读写→sync 通路验证\n"
		    "  counter       读硬件带宽计数器（判断\"真算了没\"的判据）\n"
		    "  submit-safe   提交自制命令流（仅 OP_NONE，安全）\n"
		    "  submit-enable 提交含 OP_ENABLE 的流（危险，需 RKNPU_I_KNOW_THE_RISK=1）\n",
		    argv[0]);
		r = 2;
	}

	close(fd);
	return r;
}
