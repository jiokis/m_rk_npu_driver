// bench1280.c — 1280 YOLO11 的极限吞吐：分块 + 多核并发（C 运行时）
//
// 三种模式，同一台账本：
//   native : 1280 模型单实例（绑 3 核）        —— 基线
//   tile1  : 4×640 分块，单实例（绑 3 核），串行
//   tile3  : 4×640 分块，3 实例各绑一核，并发  —— 目标方案
//
// 编译：
//   gcc -O2 -o bench1280 bench1280.c -I<inc> /usr/lib/librknnrt.so -lpthread -lm
// 用法：
//   ./bench1280 m640.rknn m1280.rknn native|tile1|tile3 <frames>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include "rknn_api.h"

#define TILE 640
#define NT 16

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static unsigned char *read_file(const char *p, int *sz) {
    FILE *f = fopen(p, "rb"); if (!f) { perror("open"); return NULL; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(n);
    if (fread(b, 1, n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    fclose(f); *sz = (int)n; return b;
}

typedef struct {
    rknn_context ctx;
    rknn_tensor_mem *in_mem;
    rknn_tensor_attr in_attr;
    uint32_t n_out;
    long done;
} wk_t;

static int ctx_init(rknn_context *ctx, unsigned char *model, int len) {
    int r = rknn_init(ctx, model, len, 0, NULL);
    if (r != RKNN_SUCC) { printf("init fail %d\n", r); return -1; }
    return 0;
}
static int bind_in(wk_t *w) {
    rknn_input_output_num n;
    if (rknn_query(w->ctx, RKNN_QUERY_IN_OUT_NUM, &n, sizeof(n)) != RKNN_SUCC) return -1;
    w->n_out = n.n_output;
    memset(&w->in_attr, 0, sizeof(w->in_attr)); w->in_attr.index = 0;
    if (rknn_query(w->ctx, RKNN_QUERY_INPUT_ATTR, &w->in_attr, sizeof(w->in_attr)) != RKNN_SUCC) return -1;
    w->in_mem = rknn_create_mem(w->ctx, w->in_attr.size_with_stride);
    memset(w->in_mem->virt_addr, 0, w->in_attr.size_with_stride);
    return 0;
}

// ---- 并发分块：全局 tile 队列 ----
static volatile int g_next = 0, g_total = 0;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static unsigned char *g_tile_src = NULL;
static int g_use_copy = 0;   // 1 = 用 rknn_inputs_set（缓存路径）

static void *worker(void *arg) {
    wk_t *w = (wk_t *)arg;
    // 预热
    for (int i = 0; i < 6; i++) {
        memcpy(w->in_mem->virt_addr, g_tile_src, TILE * TILE * 3);
        rknn_run(w->ctx, NULL);
    }
    for (;;) {
        pthread_mutex_lock(&g_mu);
        int idx = g_next++;
        pthread_mutex_unlock(&g_mu);
        if (idx >= g_total) break;
        if (g_use_copy) {
            rknn_input in; memset(&in, 0, sizeof(in));
            in.index = 0; in.type = RKNN_TENSOR_UINT8;
            in.size = w->in_attr.size; in.fmt = RKNN_TENSOR_NHWC;
            in.buf = g_tile_src;
            rknn_inputs_set(w->ctx, 1, &in);
        } else {
            memcpy(w->in_mem->virt_addr, g_tile_src, TILE * TILE * 3);
        }
        rknn_run(w->ctx, NULL);
        w->done++;
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 5) { printf("usage: %s m640.rknn m1280.rknn native|tile1|tile3 frames\n", argv[0]); return 1; }
    const char *m640p = argv[1], *m1280p = argv[2], *mode = argv[3];
    int frames = atoi(argv[4]);
    int l640 = 0, l1280 = 0;
    unsigned char *m640 = read_file(m640p, &l640);
    unsigned char *m1280 = read_file(m1280p, &l1280);
    if (!m640 || !m1280) return 1;
    g_tile_src = malloc(TILE * TILE * 3);
    memset(g_tile_src, 128, TILE * TILE * 3);

    if (strcmp(mode, "native") == 0) {
        rknn_context ctx; wk_t w; memset(&w, 0, sizeof(w));
        if (ctx_init(&ctx, m1280, l1280)) return 1;
        rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
        w.ctx = ctx; if (bind_in(&w)) { printf("bind fail\n"); return 1; }
        int in_sz = w.in_attr.size_with_stride;
        rknn_tensor_mem *mem = rknn_create_mem(ctx, in_sz);
        memset(mem->virt_addr, 0, in_sz);
        rknn_input in; memset(&in, 0, sizeof(in));
        in.index = 0; in.type = RKNN_TENSOR_UINT8; in.size = w.in_attr.size;
        in.fmt = RKNN_TENSOR_NHWC; in.buf = mem->virt_addr;
        for (int i = 0; i < 8; i++) { rknn_inputs_set(ctx, 1, &in); rknn_run(ctx, NULL); }
        double t0 = now_ms();
        for (int f = 0; f < frames; f++) {
            rknn_inputs_set(ctx, 1, &in);
            int r = rknn_run(ctx, NULL);
            if (f == 0) printf("  [check] run=%d\n", r);
            rknn_output o; memset(&o, 0, sizeof(o)); o.index = 0; o.want_float = 0;
            rknn_outputs_get(ctx, 1, &o, NULL);
            rknn_outputs_release(ctx, 1, &o);
        }
        double ms = (now_ms() - t0) / frames;
        printf("  native 1280 (3核, 单实例): %.2f ms/帧  →  %.2f FPS\n", ms, 1000.0 / ms);
    } else if (strcmp(mode, "tile1") == 0) {
        rknn_context ctx; wk_t w; memset(&w, 0, sizeof(w));
        if (ctx_init(&ctx, m640, l640)) return 1;
        rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
        w.ctx = ctx; if (bind_in(&w)) return 1;
        rknn_input in; memset(&in, 0, sizeof(in));
        in.index = 0; in.type = RKNN_TENSOR_UINT8; in.size = w.in_attr.size;
        in.fmt = RKNN_TENSOR_NHWC; in.buf = w.in_mem->virt_addr;
        for (int i = 0; i < 8; i++) {
            memcpy(w.in_mem->virt_addr, g_tile_src, TILE * TILE * 3);
            rknn_inputs_set(ctx, 1, &in); rknn_run(ctx, NULL);
        }
        double t0 = now_ms();
        for (int f = 0; f < frames; f++)
            for (int t = 0; t < 4; t++) {
                memcpy(w.in_mem->virt_addr, g_tile_src, TILE * TILE * 3);
                rknn_inputs_set(ctx, 1, &in);
                rknn_run(ctx, NULL);
                rknn_output o; memset(&o, 0, sizeof(o)); o.index = 0; o.want_float = 0;
                rknn_outputs_get(ctx, 1, &o, NULL);
                rknn_outputs_release(ctx, 1, &o);
            }
        double ms = (now_ms() - t0) / frames;
        printf("  tile1 (4×640串行, 3核): %.2f ms/帧  →  %.2f FPS\n", ms, 1000.0 / ms);
    } else {
        g_use_copy = (strcmp(mode, "tile3c") == 0);
        int nw = getenv("NW") ? atoi(getenv("NW")) : 3;
        if (nw > NT) nw = NT;
        rknn_context ctx[NT]; wk_t w[NT];
        if (ctx_init(&ctx[0], m640, l640)) return 1;
        rknn_set_core_mask(ctx[0], RKNN_NPU_CORE_0);
        for (int i = 1; i < nw; i++) {
            if (rknn_dup_context(&ctx[0], &ctx[i]) != RKNN_SUCC) { printf("dup fail %d\n", i); return 1; }
            rknn_set_core_mask(ctx[i], (rknn_core_mask)((i % 3 == 1) ? RKNN_NPU_CORE_1 :
                                                        (i % 3 == 2) ? RKNN_NPU_CORE_2 : RKNN_NPU_CORE_0));
        }
        for (int i = 0; i < nw; i++) { memset(&w[i], 0, sizeof(wk_t)); w[i].ctx = ctx[i]; if (bind_in(&w[i])) return 1; }
        g_total = frames * 4;
        g_next = 0;
        pthread_t th[NT];
        double t0 = now_ms();
        for (int i = 0; i < nw; i++) pthread_create(&th[i], NULL, worker, &w[i]);
        for (int i = 0; i < nw; i++) pthread_join(th[i], NULL);
        printf("  (worker=%d) ", nw);
        double wall = (now_ms() - t0) / 1000.0;
        double fps = frames / wall;
        printf("  tile3 (4×640, 3实例各绑一核): %.2f ms/帧  →  %.2f FPS   (tile 吞吐 %.1f/s)\n",
               wall * 1000.0 / frames, fps, g_total / wall);
        for (int i = 0; i < nw; i++) printf("    实例%d 完成 %ld 个 tile\n", i, w[i].done);
    }
    free(m640); free(m1280); free(g_tile_src);
    return 0;
}
