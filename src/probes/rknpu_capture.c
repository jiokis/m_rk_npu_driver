/*
 * rknpu_capture.c — 用 LD_PRELOAD 拦截 rknpu 的 ioctl，把闭源运行时的 regcmd 抓下来
 *
 * 用法:
 *   gcc -shared -fPIC -O2 -o rknpu_capture.so rknpu_capture.c -ldl -lpthread
 *   RKNPU_DUMP_DIR=/tmp/rknpu_dump \
 *   LD_PRELOAD=./rknpu_capture.so python3 your_rknn_script.py
 *
 * 原理（细节见 docs/04-实验记录.md）:
 *   内核驱动把 GEM 对象的内核虚拟地址通过 rknpu_mem_create.obj_addr 泄露给用户态，
 *   用户态再把它当 rknpu_submit.task_obj_addr 传回来。于是:
 *     ① 记下每次 MEM_CREATE 的 {handle, obj_addr, dma_addr, size}
 *     ② 记下每次 MEM_MAP 的 {handle → offset}，以及随后 mmap(offset) 得到的 cpu 指针
 *        → 于是有了 obj_addr → cpu_ptr 和 dma_addr → cpu_ptr 两张表
 *     ③ SUBMIT 时，task_obj_addr 必然等于某个 obj_addr → 反查到 task 数组的 cpu 指针
 *     ④ 遍历 task 数组，每个 task 的 regcmd_addr 是 dma_addr → 反查到 regcmd 的 cpu 指针
 *     ⑤ 按 regcfg_amount 的字节数把 regcmd 原样 dump 出来  ← 逆向的真正素材
 *
 * 本文件不依赖 vendor 头文件，结构体按 uapi 定义复刻（见注释里的出处）。
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>

/* ─────────── 以下结构体复刻自 drivers/rknpu/include/rknpu_ioctl.h (GPL-2.0) ─────────── */

struct rknpu_mem_create {
	uint32_t handle;
	uint32_t flags;
	uint64_t size;
	uint64_t obj_addr;
	uint64_t dma_addr;
	uint64_t sram_size;
	int32_t  iommu_domain_id;
	uint32_t core_mask;
};

struct rknpu_mem_map {
	uint32_t handle;
	uint32_t reserved;
	uint64_t offset;
};

struct rknpu_mem_destroy {
	uint32_t handle;
	uint32_t reserved;
	uint64_t obj_addr;
};

struct rknpu_mem_sync {
	uint32_t flags;
	uint32_t reserved;
	uint64_t obj_addr;
	uint64_t offset;
	uint64_t size;
};

struct rknpu_task {
	uint32_t flags;
	uint32_t op_idx;
	uint32_t enable_mask;
	uint32_t int_mask;
	uint32_t int_clear;
	uint32_t int_status;
	uint32_t regcfg_amount;   /* ← 命令流字节长度 */
	uint32_t regcfg_offset;
	uint64_t regcmd_addr;     /* ← 命令流设备地址 */
} __attribute__((packed));

struct rknpu_subcore_task { uint32_t task_start, task_number; };

struct rknpu_submit {
	uint32_t flags;
	uint32_t timeout;
	uint32_t task_start;
	uint32_t task_number;
	uint32_t task_counter;
	int32_t  priority;
	uint64_t task_obj_addr;
	uint32_t iommu_domain_id;
	uint32_t reserved;
	uint64_t task_base_addr;
	int64_t  hw_elapse_time;
	uint32_t core_mask;
	int32_t  fence_fd;
	struct rknpu_subcore_task subcore_task[5];
};

struct rknpu_action { uint32_t flags; uint32_t value; };

/* ─────────── ioctl 编号解析：只认 type 和 nr，不依赖结构体大小 ─────────── */
#define DRM_IOCTL_BASE   'd'
#define DRM_COMMAND_BASE 0x40
#define RKNPU_NR_ACTION       0x00
#define RKNPU_NR_SUBMIT       0x01
#define RKNPU_NR_MEM_CREATE   0x02
#define RKNPU_NR_MEM_MAP      0x03
#define RKNPU_NR_MEM_DESTROY  0x04
#define RKNPU_NR_MEM_SYNC     0x05

#define IOC_TYPE(nr) (((nr) >> 8) & 0xff)
#define IOC_NR(nr)   ((nr) & 0xff)

static int is_rknpu_ioctl(unsigned long req, int *nr_out)
{
	if (IOC_TYPE(req) != DRM_IOCTL_BASE) return 0;
	int n = IOC_NR(req);
	if (n < DRM_COMMAND_BASE || n > DRM_COMMAND_BASE + 5) return 0;
	*nr_out = n - DRM_COMMAND_BASE;
	return 1;
}

/* ─────────── 全局状态 ─────────── */
#define MAX_OBJ 256

/*
 * ★ 设备地址归一化（踩过的坑）
 *
 * 同一块内存，不同运行时给的地址**符号扩展位不一样**：
 *   librknnrt (yolov5s)  : regcmd_addr = 0x0000000080c48580
 *   librkllmrt (Qwen2.5) : regcmd_addr = 0xfffffffffed1e000
 * 而 MEM_CREATE 报的 dma_addr 是 0x00000000fed1e000。
 *
 * ⚠️ 第一次修的时候我掩成 40 位（驱动 dma_mask = BIT_MASK(40)），
 *    结果 0xfffffffffed1e000 & 0xFFFFFFFFFF = 0xfffed1e000，仍然对不上！
 *    因为它是**从 32 位符号扩展**来的，bit32-39 也是 1。
 *
 * 正解：先看高 32 位是不是全 1（= 32 位有符号扩展），是就取低 32 位；
 *       否则按 40 位 IOVA 取。
 */
static uint64_t norm_dma(uint64_t v)
{
	if ((v >> 32) == 0xFFFFFFFFULL)
		return v & 0xFFFFFFFFULL;      /* 32 位符号扩展 */
	return v & 0xFFFFFFFFFFULL;            /* 40 位 IOVA */
}
#define DMA_MASK40 0xFFFFFFFFFFULL
struct objent {
	int      used;
	uint32_t handle;
	uint64_t obj_addr;    /* 内核对象地址（驱动泄露出来的） */
	uint64_t dma_addr;    /* 设备地址 */
	uint64_t size;
	void    *cpu;         /* 用户态映射得到的指针 */
	uint32_t flags;
};
static struct objent g_obj[MAX_OBJ];
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

/* MEM_MAP 的 offset → handle，等 mmap 来配对 */
#define MAX_MAP 256
static struct { int used; uint64_t offset; uint32_t handle; } g_map[MAX_MAP];

static char g_dir[512];
static int  g_dump_seq = 0;
static int  g_dump_bufs = 0;          /* RKNPU_DUMP_BUFS: 0=关 1=仅首个提交 2=每次提交 */
static size_t g_bufs_max = 32u << 20; /* RKNPU_DUMP_BUFS_MAX: 单缓冲区上限 */
static int  g_buf_dumped_first = 0;
static int  g_verbose = 0;
static uint32_t g_slack = 0;
static FILE *g_log;

static void logf_(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	if (g_log) { vfprintf(g_log, fmt, ap); fflush(g_log); }
	va_end(ap);
}

__attribute__((constructor))
static void cap_init(void)
{
	const char *d = getenv("RKNPU_DUMP_DIR");
	snprintf(g_dir, sizeof(g_dir), "%s", d ? d : "/tmp/rknpu_dump");
	mkdir(g_dir, 0755);

	char path[600];
	snprintf(path, sizeof(path), "%s/capture.log", g_dir);
	g_log = fopen(path, "w");
	g_verbose = getenv("RKNPU_CAPTURE_VERBOSE") != NULL;
	{ const char *s = getenv("RKNPU_DUMP_SLACK"); if (s) g_slack = (uint32_t)strtoul(s, NULL, 0); }
	{ const char *s = getenv("RKNPU_DUMP_BUFS"); if (s) g_dump_bufs = atoi(s); }
	{ const char *s = getenv("RKNPU_DUMP_BUFS_MAX"); if (s) g_bufs_max = (size_t)strtoull(s, NULL, 0); }
	logf_("# 缓冲区镜像 dump: mode=%d 单缓冲区上限=%zu\n", g_dump_bufs, g_bufs_max);

	logf_("# rknpu_capture 启动\n");
	logf_("# pid=%d dump_dir=%s\n", getpid(), g_dir);
	logf_("# sizeof(rknpu_task)=%zu sizeof(rknpu_submit)=%zu sizeof(rknpu_mem_create)=%zu\n",
	      sizeof(struct rknpu_task), sizeof(struct rknpu_submit),
	      sizeof(struct rknpu_mem_create));

	/* 自检：算出六个 ioctl 的编号，便于和 strace 对照 */
	unsigned long i_action = _IOWR('d', DRM_COMMAND_BASE + 0, struct rknpu_action);
	unsigned long i_submit = _IOWR('d', DRM_COMMAND_BASE + 1, struct rknpu_submit);
	logf_("# ioctl 编号: ACTION=%#lx SUBMIT=%#lx (低16位=%#lx)\n",
	      i_action, i_submit, i_submit & 0xffff);
}

static struct objent *find_by_obj(uint64_t obj)
{
	for (int i = 0; i < MAX_OBJ; i++)
		if (g_obj[i].used && g_obj[i].obj_addr == obj) return &g_obj[i];
	return NULL;
}
/*
 * ⚠️ 必须按区间查找，不能精确匹配。
 * 实测：regcmd_addr 不是某个独立缓冲区的首地址，而是**落在某个大缓冲区内部**——
 * 例如 regcmd_addr=0x80c48580 落在 handle2 的 [0x8054d000, +8282112) 里，
 * 偏移 7320960。也就是说运行时把命令流塞在权重缓冲区的尾部，
 * 用精确匹配会一条都找不到（第一次实验就是这么空手的）。
 */
static struct objent *find_by_dma(uint64_t dma)
{
	/* ★ 掩到 40 位再比：不同运行时的符号扩展位不一致
	   （librknnrt 给 0x0000000080c48580，librkllmrt 给 0xfffffffffed1e000） */
	uint64_t t = norm_dma(dma);
	for (int i = 0; i < MAX_OBJ; i++) {
		uint64_t base;
		if (!g_obj[i].used) continue;
		base = norm_dma(g_obj[i].dma_addr);
		if (t >= base && t < base + g_obj[i].size)
			return &g_obj[i];
	}
	return NULL;
}
static struct objent *find_by_handle(uint32_t h)
{
	for (int i = 0; i < MAX_OBJ; i++)
		if (g_obj[i].used && g_obj[i].handle == h) return &g_obj[i];
	return NULL;
}

static void hexdump_file(const char *path, const void *data, size_t len)
{
	FILE *f = fopen(path, "wb");
	if (!f) return;
	fwrite(data, 1, len, f);
	fclose(f);
}

/* ─────────── 处理各 ioctl ─────────── */
static void on_mem_create(struct rknpu_mem_create *a, long ret)
{
	if (ret < 0) return;
	pthread_mutex_lock(&g_lock);
	for (int i = 0; i < MAX_OBJ; i++) {
		if (g_obj[i].used) continue;
		g_obj[i].used = 1;
		g_obj[i].handle   = a->handle;
		g_obj[i].obj_addr = a->obj_addr;
		g_obj[i].dma_addr = norm_dma(a->dma_addr);
		g_obj[i].size     = a->size;
		g_obj[i].flags    = a->flags;
		g_obj[i].cpu      = NULL;
		pthread_mutex_unlock(&g_lock);

		logf_("[MEM_CREATE] handle=%u flags=%#x size=%llu obj_addr=%#llx dma_addr=%#llx sram_size=%llu core_mask=%#x\n",
		      a->handle, a->flags, (unsigned long long)a->size,
		      (unsigned long long)a->obj_addr, (unsigned long long)a->dma_addr,
		      (unsigned long long)a->sram_size, a->core_mask);
		return;
	}
	pthread_mutex_unlock(&g_lock);
	logf_("[MEM_CREATE] ⚠️ 表满，丢弃\n");
}

static void on_mem_map(struct rknpu_mem_map *a, long ret)
{
	if (ret < 0) return;
	pthread_mutex_lock(&g_lock);
	for (int i = 0; i < MAX_MAP; i++) {
		if (g_map[i].used) continue;
		g_map[i].used = 1;
		g_map[i].offset = a->offset;
		g_map[i].handle = a->handle;
		break;
	}
	pthread_mutex_unlock(&g_lock);
	logf_("[MEM_MAP] handle=%u → offset=%#llx\n", a->handle,
	      (unsigned long long)a->offset);
}

static void on_mem_destroy(struct rknpu_mem_destroy *a)
{
	pthread_mutex_lock(&g_lock);
	struct objent *o = find_by_handle(a->handle);
	if (o) {
		logf_("[MEM_DESTROY] handle=%u obj_addr=%#llx size=%llu\n",
		      a->handle, (unsigned long long)o->obj_addr,
		      (unsigned long long)o->size);
		o->used = 0;
	}
	pthread_mutex_unlock(&g_lock);
}

static void on_action(struct rknpu_action *a, long ret)
{
	static const char *names[] = {
		"GET_HW_VERSION","GET_DRV_VERSION","GET_FREQ","SET_FREQ","GET_VOLT","SET_VOLT",
		"ACT_RESET","GET_BW_PRIORITY","SET_BW_PRIORITY","GET_BW_EXPECT","SET_BW_EXPECT",
		"GET_BW_TW","SET_BW_TW","ACT_CLR_TOTAL_RW_AMOUNT","GET_DT_WR_AMOUNT",
		"GET_DT_RD_AMOUNT","GET_WT_RD_AMOUNT","GET_TOTAL_RW_AMOUNT","GET_IOMMU_EN",
		"SET_PROC_NICE","POWER_ON","POWER_OFF","GET_TOTAL_SRAM_SIZE","GET_FREE_SRAM_SIZE",
		"GET_IOMMU_DOMAIN_ID","SET_IOMMU_DOMAIN_ID"
	};
	const char *nm = a->flags < 26 ? names[a->flags] : "?";
	logf_("[ACTION] %s (flags=%u) value=%#x ret=%ld\n", nm, a->flags, a->value, ret);
}

/* SUBMIT：核心。把 task 数组和每条 regcmd 都 dump 出来 */
static void on_submit(struct rknpu_submit *a, long ret)
{
	pthread_mutex_lock(&g_lock);

	struct objent *tobj = find_by_obj(a->task_obj_addr);
	logf_("\n[SUBMIT] flags=%#x core_mask=%#x timeout=%u task_start=%u task_number=%u "
	      "task_counter=%u priority=%d task_obj_addr=%#llx task_base_addr=%#llx iommu_dom=%u fence_fd=%d ret=%ld\n",
	      a->flags, a->core_mask, a->timeout, a->task_start, a->task_number,
	      a->task_counter, a->priority,
	      (unsigned long long)a->task_obj_addr,
	      (unsigned long long)a->task_base_addr,
	      a->iommu_domain_id, a->fence_fd, ret);

	for (int i = 0; i < 5; i++)
		if (a->subcore_task[i].task_number)
			logf_("           subcore[%d]: start=%u number=%u\n", i,
			      a->subcore_task[i].task_start, a->subcore_task[i].task_number);

	if (!tobj || !tobj->cpu) {
		logf_("           ⚠️ 找不到 task 数组的 cpu 映射（obj_addr=%#llx）"
		      "—— 若 libc 走的不是 ioctl() 请检查\n",
		      (unsigned long long)a->task_obj_addr);
		pthread_mutex_unlock(&g_lock);
		return;
	}

	/* ★ 缓冲区镜像：把 NPU 将要读/写的真实 DRAM 内容落盘。
	 *   这是「全仿真」的前置 —— 没有精确的量化权重与布局，执行器无法验证。
	 *   obj_addr 就是内核对象地址，用户态可直接解引用（见本文件顶部注释）。 */
	if (g_dump_bufs && (g_dump_bufs == 2 || !g_buf_dumped_first)) {
		size_t total = 0;
		int nb = 0;
		for (int i = 0; i < MAX_OBJ; i++) {
			if (!g_obj[i].used || !g_obj[i].size) continue;
			if (g_obj[i].size > g_bufs_max) continue;
			/* cpu 只在进程 mmap 过该缓冲区时才有；否则用 obj_addr
			 * （内核对象地址，本文件顶部注释说明用户态可直接解引用） */
			void *ptr = g_obj[i].cpu ? g_obj[i].cpu : (void *)g_obj[i].obj_addr;
			if (!ptr) continue;
			char path[512];
			snprintf(path, sizeof(path), "%s/buf_%03d_dma%llx.bin", g_dir,
			         g_dump_seq, (unsigned long long)g_obj[i].dma_addr);
			hexdump_file(path, ptr, (size_t)g_obj[i].size);
			total += g_obj[i].size; nb++;
		}
		g_buf_dumped_first = 1;
		logf_("           [BUFS] 落盘 %d 个缓冲区，共 %zu 字节（seq=%d）\n",
		      nb, total, g_dump_seq);
	}

	logf_("           task 数组: cpu=%p size=%llu\n", tobj->cpu,
	      (unsigned long long)tobj->size);

	struct rknpu_task *tasks = (struct rknpu_task *)tobj->cpu;
	uint32_t n = a->task_number ? a->task_number : 1;
	uint32_t start = a->task_start;

	/* 兜底：task_number 有时为 0（由 subcore_task 指定），用 subcore 汇总 */
	if (!a->task_number) {
		uint32_t mx = 0;
		for (int i = 0; i < 5; i++) {
			uint32_t e = a->subcore_task[i].task_start + a->subcore_task[i].task_number;
			if (e > mx) mx = e;
		}
		if (mx) { start = 0; n = mx; }
	}
	if ((uint64_t)(start + n) * sizeof(struct rknpu_task) > tobj->size) {
		logf_("           ⚠️ task 范围越界，截断到缓冲区内\n");
		uint64_t cap = tobj->size / sizeof(struct rknpu_task);
		if (start >= cap) { pthread_mutex_unlock(&g_lock); return; }
		n = cap - start;
	}

	char path[700];
	snprintf(path, sizeof(path), "%s/tasks_%03d.bin", g_dir, g_dump_seq);
	hexdump_file(path, tasks + start, (size_t)n * sizeof(struct rknpu_task));
	logf_("           → 原始 task 数组已存 %s (%zu 字节)\n", path,
	      (size_t)n * sizeof(struct rknpu_task));

	for (uint32_t k = 0; k < n; k++) {
		struct rknpu_task *t = &tasks[start + k];
		logf_("           task[%u] flags=%#x op_idx=%u enable_mask=%#x int_mask=%#x "
		      "int_clear=%#x regcfg_amount=%u regcfg_offset=%u regcmd_addr=%#llx\n",
		      start + k, t->flags, t->op_idx, t->enable_mask, t->int_mask,
		      t->int_clear, t->regcfg_amount, t->regcfg_offset,
		      (unsigned long long)t->regcmd_addr);

		struct objent *rob = find_by_dma(t->regcmd_addr);
		/* ★★ 重大修正：regcfg_amount 是**指令条数**，不是字节数！★★
		 *
		 * 用软件模拟器（src/analysis/regcmd_sim.py）在 615 条真实样本上验证：
		 *   regcfg_amount 精确等于「PC 链指令」在流中的索引，109/109 全中。
		 * 所以它是**本段命令流的指令条数**（每条 8 字节）。
		 *
		 * 之前的错误：当成字节用 → dump 长度 = amt + slack，
		 * 导致 82% 的样本被截断（平均只看到 41.4%，
		 * op80 声明 611 条却只看到 108 条）。
		 * 寄存器表因此只覆盖了每条流的开头。
		 *
		 * 正确做法：amt * 8 字节。
		 */
		uint32_t amt = t->regcfg_amount * 8 + g_slack;

		if (rob && rob->cpu && amt) {
			uint64_t off = norm_dma(t->regcmd_addr)
			             - norm_dma(rob->dma_addr);
			if (off + amt <= rob->size) {
				if (off + amt > rob->size) amt = rob->size - off;
				snprintf(path, sizeof(path), "%s/regcmd_%03d_task%u.bin",
					 g_dir, g_dump_seq, start + k);
				hexdump_file(path, (uint8_t *)rob->cpu + off, amt);
				logf_("             ★ regcmd %u 字节 (regcfg=%u+slack=%u) → %s\n",
				      amt, t->regcfg_amount, g_slack, path);

				if (g_verbose) {
					uint32_t show = amt < 256 ? amt : 256;
					for (uint32_t b = 0; b < show; b += 16) {
						logf_("               %04x:", b);
						for (uint32_t j = 0; j < 16 && b + j < show; j++)
							logf_(" %02x", ((uint8_t *)rob->cpu + off)[b + j]);
						logf_("\n");
					}
				}
			} else {
				logf_("             ⚠️ regcmd 越界 (off=%llu amt=%u size=%llu)\n",
				      (unsigned long long)off, amt,
				      (unsigned long long)rob->size);
			}
		} else if (!rob) {
			logf_("             ⚠️ regcmd_addr 不在已知缓冲区内\n");
		}
	}
	g_dump_seq++;
	pthread_mutex_unlock(&g_lock);
}

/* ─────────── 真正的拦截 ─────────── */
typedef int (*ioctl_fn)(int, unsigned long, ...);
static ioctl_fn real_ioctl;

int ioctl(int fd, unsigned long req, ...)
{
	va_list ap;
	va_start(ap, req);
	void *arg = va_arg(ap, void *);
	va_end(ap);

	if (!real_ioctl) real_ioctl = (ioctl_fn)dlsym(RTLD_NEXT, "ioctl");

	int nr;
	if (!is_rknpu_ioctl(req, &nr))
		return real_ioctl(fd, req, arg);

	long ret = real_ioctl(fd, req, arg);

	switch (nr) {
	case RKNPU_NR_ACTION:      on_action((struct rknpu_action *)arg, ret); break;
	case RKNPU_NR_MEM_CREATE:  on_mem_create((struct rknpu_mem_create *)arg, ret); break;
	case RKNPU_NR_MEM_MAP:     on_mem_map((struct rknpu_mem_map *)arg, ret); break;
	case RKNPU_NR_MEM_DESTROY: on_mem_destroy((struct rknpu_mem_destroy *)arg); break;
	case RKNPU_NR_SUBMIT:      on_submit((struct rknpu_submit *)arg, ret); break;
	case RKNPU_NR_MEM_SYNC: {
		struct rknpu_mem_sync *s = (struct rknpu_mem_sync *)arg;
		logf_("[MEM_SYNC] flags=%#x obj_addr=%#llx offset=%llu size=%llu\n",
		      s->flags, (unsigned long long)s->obj_addr,
		      (unsigned long long)s->offset, (unsigned long long)s->size);
		break;
	}
	}
	return ret;
}

/* mmap 用来把 MEM_MAP 的 offset 和真实 cpu 指针配上
 *
 * ⚠️ 坑：librknnrt 同时导入 mmap 和 mmap64，而它实际走的是 mmap64 的 PLT 入口。
 * 只挂 mmap 会一条都拦不到（第一次实验就是这么失败的）。两个都要挂。
 * 另外 offset 做一次页对齐兜底，防止驱动给的 offset 被调用方做过变换。 */
typedef void *(*mmap_fn)(void *, size_t, int, int, int, off_t);
static mmap_fn real_mmap;
static mmap_fn real_mmap64;

static void bind_mmap(const char *tag, void *p, size_t len, off_t off)
{
	if (p == MAP_FAILED) return;
	pthread_mutex_lock(&g_lock);
	for (int i = 0; i < MAX_MAP; i++) {
		if (!g_map[i].used) continue;
		uint64_t want = g_map[i].offset;
		if (want == (uint64_t)off ||
		    (want & ~(uint64_t)0xfff) == ((uint64_t)off & ~(uint64_t)0xfff)) {
			struct objent *o = find_by_handle(g_map[i].handle);
			if (o) {
				o->cpu = p;
				logf_("[%s] handle=%u offset=%#llx → cpu=%p len=%zu"
				      "  ⇒ dma_addr=%#llx 现在可从用户态读写\n",
				      tag, o->handle, (unsigned long long)off, p, len,
				      (unsigned long long)o->dma_addr);
			}
			g_map[i].used = 0;
			break;
		}
	}
	pthread_mutex_unlock(&g_lock);
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off)
{
	if (!real_mmap) real_mmap = (mmap_fn)dlsym(RTLD_NEXT, "mmap");
	if (!real_mmap) real_mmap = (mmap_fn)dlsym(RTLD_NEXT, "mmap64");
	void *p = real_mmap(addr, len, prot, flags, fd, off);
	bind_mmap("mmap", p, len, off);
	return p;
}

void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off64_t off)
{
	if (!real_mmap64) real_mmap64 = (mmap_fn)dlsym(RTLD_NEXT, "mmap64");
	if (!real_mmap64) real_mmap64 = real_mmap;
	if (!real_mmap64) real_mmap64 = (mmap_fn)dlsym(RTLD_NEXT, "mmap");
	void *p = real_mmap64(addr, len, prot, flags, fd, (off_t)off);
	bind_mmap("mmap64", p, len, (off_t)off);
	return p;
}
