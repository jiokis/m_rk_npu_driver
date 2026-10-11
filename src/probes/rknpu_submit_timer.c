/*
 * rknpu_submit_timer.c — 量出每次提交的「硬件耗时」，从而分离调度开销
 *
 * ## 为什么需要它
 *
 * docs/24 分析出多核只有 1.76×（而非 3×），并估算"同步屏障开销占 42%~62%"
 * —— 但那个估算只有两个数据点，**是欠定的**。
 *
 * 驱动其实早就把每次提交的硬件耗时写回用户态了：
 *     args->hw_elapse_time = job->hw_elapse_time;      // rknpu_job.c:267
 *     job->hw_elapse_time  = ktime_sub(now, job->hw_commit_time);  // 完成中断里
 *
 * 所以：**拦截 ioctl，在它返回后读 hw_elapse_time，就能把
 *   「硬件真正干活的时间」与「墙钟时间」分开** —— 差值就是提交/调度开销。
 *
 * ## 用法
 *     LD_PRELOAD=./rknpu_submit_timer.so python3 your_infer.py
 * 输出（stderr 与文件）：
 *     SUBMIT <序号> start=<> n=<> core_mask=<> wall_us=<> hw_us=<> overhead_us=<>
 *
 * ## 与 rknpu_capture 的区别
 * 那个抓命令流（重），这个只量时间（轻、可长期挂着跑）。
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/ioctl.h>

/*
 * ★ 驱动暴露【两套】ioctl 路径，两者都要拦：
 *   RKNPU_IOWR(nr,type) = _IOWR('r', nr, type)              ← 直接路径
 *   DRM_IOCTL_RKNPU_*(…)= DRM_IOWR(DRM_COMMAND_BASE + nr)   ← DRM 路径（magic 'd'）
 * 实测 librknnrt / librkllmrt 走的是【DRM 路径】（设备是 /dev/dri/renderD129）：
 *     DRM_IOWR(0x40 + 1, struct rknpu_submit) = 0xc0686441
 * 一开始我按 _IOWR('d',1,…) 算成 0xc0686401，差在 nr 少了 DRM_COMMAND_BASE(0x40)。
 */
#define RKNPU_SUBMIT_NR     0x01
#define DRM_COMMAND_BASE    0x40
#define RKNPU_IOC_MAGIC     'r'      /* 0x72 */
#define DRM_IOCTL_MAGIC     'd'      /* 0x64 */

/* 与内核 rknpu_ioctl.h 一致（sizeof=104） */
struct rknpu_subcore_task { uint32_t task_start, task_number; };

struct rknpu_submit {
	uint32_t flags, timeout;
	uint32_t task_start, task_number, task_counter;
	int32_t  priority;
	uint64_t task_obj_addr;
	uint32_t iommu_domain_id, reserved;
	uint64_t task_base_addr;
	int64_t  hw_elapse_time;
	uint32_t core_mask;
	int32_t  fence_fd;
	struct rknpu_subcore_task subcore_task[5];
};

static long (*real_ioctl)(int, unsigned long, void *) = NULL;
static FILE *g_fp = NULL;
static unsigned long g_n = 0;
static unsigned long long g_wall_ns = 0, g_hw_ns = 0;

/* 两条路径的 SUBMIT ioctl 编号 */
#define DRM_SUBMIT_IOCTL \
	_IOWR(DRM_IOCTL_MAGIC, DRM_COMMAND_BASE + RKNPU_SUBMIT_NR, struct rknpu_submit)
#define RKNPU_SUBMIT_IOCTL \
	_IOWR(RKNPU_IOC_MAGIC, RKNPU_SUBMIT_NR, struct rknpu_submit)

static inline int is_submit(unsigned long req)
{
	return req == (unsigned long)DRM_SUBMIT_IOCTL ||
	       req == (unsigned long)RKNPU_SUBMIT_IOCTL;
}

static void init(void)
{
	const char *p;
	if (real_ioctl) return;
	real_ioctl = (long (*)(int, unsigned long, void *))dlsym(RTLD_NEXT, "ioctl");
	p = getenv("RKNPU_TIMER_OUT");
	if (!p) p = "/tmp/rknpu_timer.log";
	g_fp = fopen(p, "w");
	if (g_fp) {
		fprintf(g_fp, "# 序号 task_start task_number core_mask wall_us hw_us overhead_us\n");
		fflush(g_fp);
	}
}

static unsigned long long now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (unsigned long long)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* ⚠️ 签名必须与 glibc 一致：int ioctl(int, unsigned long, ...) */
int ioctl(int fd, unsigned long req, ...)
{
	va_list ap;
	void *arg;
	unsigned long long t0, t1;
	long ret;

	if (!real_ioctl) init();

	va_start(ap, req);
	arg = va_arg(ap, void *);
	va_end(ap);

	if (!is_submit(req) || !arg)
		return real_ioctl(fd, req, arg);

	t0 = now_ns();
	ret = real_ioctl(fd, req, arg);
	t1 = now_ns();

	if (ret == 0) {
		struct rknpu_submit *s = (struct rknpu_submit *)arg;
		unsigned long long wall = t1 - t0;
		long long hw = s->hw_elapse_time;
		long long ov = (long long)wall - hw;

		g_wall_ns += wall;
		if (hw > 0) g_hw_ns += (unsigned long long)hw;
		g_n++;

		if (g_fp) {
			fprintf(g_fp, "%lu %u %u %#x %llu %lld %lld\n",
				g_n, s->task_start, s->task_number, s->core_mask,
				wall, hw, ov);
			if ((g_n % 64) == 0) {
				fflush(g_fp);
				fprintf(stderr,
					"[rknpu_timer] 已 %lu 次提交 · 墙钟 %.1fms · 硬件 %.1fms (%.0f%%)\n",
					g_n, g_wall_ns / 1e6, g_hw_ns / 1e6,
					g_wall_ns ? (double)g_hw_ns * 100.0 / g_wall_ns : 0.0);
			}
		}
	}
	return ret;
}

__attribute__((destructor)) static void fini(void)
{
	if (!g_fp) return;
	fprintf(g_fp, "# TOTAL submits=%lu wall_us=%.0f hw_us=%.0f hw_ratio=%.1f%%\n",
		g_n, g_wall_ns / 1e3, g_hw_ns / 1e3,
		g_wall_ns ? (double)g_hw_ns * 100.0 / g_wall_ns : 0.0);
	fclose(g_fp);
	fprintf(stderr,
		"\n[rknpu_timer] ═══ 汇总 ═══\n"
		"  提交次数      : %lu\n"
		"  墙钟总计      : %.2f ms\n"
		"  硬件总计      : %.2f ms  (%.1f%%)\n"
		"  提交/调度开销 : %.2f ms  (%.1f%%)\n",
		g_n, g_wall_ns / 1e6, g_hw_ns / 1e6,
		g_wall_ns ? (double)g_hw_ns * 100.0 / g_wall_ns : 0.0,
		(g_wall_ns - g_hw_ns) / 1e6,
		g_wall_ns ? (double)(g_wall_ns - g_hw_ns) * 100.0 / g_wall_ns : 0.0);
}
