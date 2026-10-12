/*
 * rknpu_ioctl_cpu.c — 按 ioctl 类型统计 RKNPU 的内核 CPU 时间（LD_PRELOAD）
 *
 * ═══ 为什么需要它 ═══
 *
 * docs/33 量出一次推理的时间构成是「NPU 硬忙 1/3 + CPU 2/3」。进一步拆开 CPU：
 *
 *     wall = 25.92 ms   user = 7.50 ms   sys = 9.50 ms
 *     ioctl 墙钟 9.68 ms，其中 NPU 硬忙 9.00 ms
 *
 * 也就是说：**ioctl 里等 NPU 只占 9.00 ms，而内核态 CPU 却烧了 9.50 ms。**
 * 那 9.5 ms 是内核在真干活（245 次 ioctl/推理，多半是 MEM_SYNC 的缓存维护、
 * IOMMU 映射、提交路径簿记）—— 而这部分**是驱动侧能改的**。
 *
 * 但"能改"之前必须先知道花在**哪一类** ioctl 上。所以本探针：
 *   · 按 RKNPU 的 ioctl 请求码分类（ACTION/SUBMIT/MEM_CREATE/MAP/DESTROY/SYNC/其他）
 *   · 每类统计 次数 / 墙钟 / **线程 CPU 时间**
 *
 * ★ 用 CLOCK_THREAD_CPUTIME_ID 而不是墙钟，是关键：
 *   阻塞在等待 NPU 完成上的时间**不计入** CPU 时间，于是能把
 *   "等硬件" 和 "内核真在算" 干净地区分开。这正是 docs/33 里
 *   ioctl 墙钟(9.68) 与 sys CPU(9.50) 看起来矛盾的原因。
 *
 * 编译（板子上）：
 *   gcc -shared -fPIC -O2 -o rknpu_ioctl_cpu.so rknpu_ioctl_cpu.c -ldl -lpthread
 * 用法：
 *   LD_PRELOAD=./rknpu_ioctl_cpu.so python3 your_infer.py
 *   # 退出时把统计表写到 $RKNPU_IOCTL_CPU_LOG（默认 /tmp/rknpu_ioctl_cpu.log）
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ── RKNPU 的 ioctl 请求码（来自 rknpu_ioctl.h，已实测确证）──
 * DRM 路径：DRM_IOWR(DRM_COMMAND_BASE(0x40) + nr, type)
 *   DRM_IOWR(nr,type) = _IOWR('d', nr, type)，'d'=0x64
 * _IOWR(base,nr,size) = (3<<30)|(size<<16)|(base<<8)|nr
 */
#define RK_ACTION      0xc0086440UL   /* _IOWR('d',0x40, 8)  */
#define RK_SUBMIT      0xc0686441UL   /* _IOWR('d',0x41,104) */
#define RK_MEM_CREATE  0xc0306442UL   /* _IOWR('d',0x42, 48) */
#define RK_MEM_MAP     0xc0106443UL   /* _IOWR('d',0x43, 16) */
#define RK_MEM_DESTROY 0xc0106444UL   /* _IOWR('d',0x44, 16) */
#define RK_MEM_SYNC    0xc0206445UL   /* _IOWR('d',0x45, 32) */

/* 旧的非 DRM 路径 `_IOWR('r', nr, type)` 也一并认（'r'=0x72）。
 * 实测 librknnrt/librkllmrt 走 DRM 路径，但留着以防换版本。 */
#define RK_IO(nr, size) ((unsigned long)((3UL << 30) | ((size) << 16) | (0x72UL << 8) | (nr)))

#define NCLASS 12

struct cls {
    const char *name;
    unsigned long code;      /* 0 = 通配（other） */
};

static const struct cls CLASSES[NCLASS] = {
    { "ACTION",       RK_ACTION },
    { "SUBMIT",       RK_SUBMIT },
    { "MEM_CREATE",   RK_MEM_CREATE },
    { "MEM_MAP",      RK_MEM_MAP },
    { "MEM_DESTROY",  RK_MEM_DESTROY },
    { "MEM_SYNC",     RK_MEM_SYNC },
    { "r:ACTION",     RK_IO(0, 8) },
    { "r:SUBMIT",     RK_IO(1, 104) },
    { "r:MEM_CREATE", RK_IO(2, 48) },
    { "r:MEM_MAP",    RK_IO(3, 16) },
    { "r:MEM_SYNC",   RK_IO(5, 32) },
    { "OTHER(DRM等)", 0 },
};

struct stat_row {
    unsigned long n;
    unsigned long long wall_ns;
    unsigned long long cpu_ns;
    unsigned long err;
};

static struct stat_row g_row[NCLASS];
static unsigned long long g_total_wall, g_total_cpu;
static unsigned long g_total_n;
static int g_inited;
static int (*real_ioctl)(int, unsigned long, ...);

static long long now_wall_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* ★ 只算本线程真正烧掉的 CPU 时间，阻塞等待不计入 */
static long long now_cpu_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static int classify(unsigned long code)
{
    int i;
    for (i = 0; i < NCLASS; i++)
        if (CLASSES[i].code && CLASSES[i].code == code)
            return i;
    return NCLASS - 1;
}

static void init(void)
{
    if (g_inited) return;
    real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
    g_inited = 1;
}

int ioctl(int fd, unsigned long req, ...)
{
    va_list ap;
    void *arg;
    long long w0, c0, w1, c1;
    int idx, ret;

    va_start(ap, req);
    arg = va_arg(ap, void *);
    va_end(ap);

    init();
    if (!real_ioctl) { errno = ENOSYS; return -1; }

    w0 = now_wall_ns();
    c0 = now_cpu_ns();
    ret = real_ioctl(fd, req, arg);
    c1 = now_cpu_ns();
    w1 = now_wall_ns();

    idx = classify(req);
    g_row[idx].n++;
    g_row[idx].wall_ns += (unsigned long long)(w1 - w0);
    g_row[idx].cpu_ns += (unsigned long long)(c1 - c0);
    if (ret < 0) g_row[idx].err++;
    g_total_n++;
    g_total_wall += (unsigned long long)(w1 - w0);
    g_total_cpu += (unsigned long long)(c1 - c0);
    return ret;
}

__attribute__((destructor)) static void fini(void)
{
    const char *path = getenv("RKNPU_IOCTL_CPU_LOG");
    FILE *f;
    int i;
    if (!g_total_n) return;
    f = fopen(path ? path : "/tmp/rknpu_ioctl_cpu.log", "w");
    if (!f) return;

    fprintf(f, "# RKNPU ioctl 按类型统计（cpu 列用 CLOCK_THREAD_CPUTIME_ID，不含阻塞等待）\n");
    fprintf(f, "# %-14s %8s %14s %14s %8s %14s\n",
            "类型", "次数", "墙钟ms", "CPU_ms", "错误", "CPU/次_us");
    for (i = 0; i < NCLASS; i++) {
        if (!g_row[i].n) continue;
        fprintf(f, "  %-14s %8lu %14.2f %14.2f %8lu %14.1f\n",
                CLASSES[i].name, g_row[i].n,
                g_row[i].wall_ns / 1e6, g_row[i].cpu_ns / 1e6, g_row[i].err,
                g_row[i].cpu_ns / 1e3 / (double)g_row[i].n);
    }
    fprintf(f, "  %-14s %8lu %14.2f %14.2f\n", "合计", g_total_n,
            g_total_wall / 1e6, g_total_cpu / 1e6);

    /* 再算一个"CPU 占比"：哪一类吃掉了最多内核 CPU */
    fprintf(f, "\n# 内核 CPU 时间占比\n");
    for (i = 0; i < NCLASS; i++) {
        if (!g_row[i].n || !g_total_cpu) continue;
        fprintf(f, "  %-14s %6.1f%%\n", CLASSES[i].name,
                g_row[i].cpu_ns * 100.0 / (double)g_total_cpu);
    }
    fclose(f);
}
