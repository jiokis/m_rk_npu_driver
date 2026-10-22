// rknn_bench.c — RK3588 上 YOLO11 的 C 级基准：零拷贝 I/O + 上下文复用
//
// 对比三条路径：
//   A) 单线程 + rknn_inputs_set（传统拷贝路径）
//   B) 单线程 + rknn_set_io_mem（零拷贝）
//   C) 3 线程 + rknn_dup_context + 零拷贝（每线程绑一个核）
//
// 编译（板上）：
//   gcc -O2 -o rknn_bench rknn_bench.c -I<include> -L<libdir> -lrknnrt -lpthread -lm
//
// 用法：
//   ./rknn_bench <model.rknn> <mode> [iters]
//     mode: copy | zero | multi
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include "rknn_api.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static unsigned char *read_file(const char *path, int *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)malloc(n);
    if (fread(buf, 1, n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *size = (int)n;
    return buf;
}

typedef struct {
    rknn_context ctx;
    rknn_tensor_mem *in_mem;
    rknn_tensor_mem **out_mems;
    rknn_tensor_attr *out_attrs;
    uint32_t n_out;
    rknn_tensor_attr in_attr;
    int iters;
    double *lat;
    int core_mask;
} worker_t;

// 在已有 ctx 上建立零拷贝绑定
static int setup_io(worker_t *w) {
    rknn_input_output_num io_num;
    if (rknn_query(w->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) != RKNN_SUCC) return -1;
    w->n_out = io_num.n_output;

    memset(&w->in_attr, 0, sizeof(w->in_attr));
    w->in_attr.index = 0;
    if (rknn_query(w->ctx, RKNN_QUERY_INPUT_ATTR, &w->in_attr, sizeof(w->in_attr)) != RKNN_SUCC) return -1;

    w->in_mem = rknn_create_mem(w->ctx, w->in_attr.size_with_stride);
    if (!w->in_mem) return -1;
    if (rknn_set_io_mem(w->ctx, w->in_mem, &w->in_attr) != RKNN_SUCC) return -1;

    w->out_mems = calloc(w->n_out, sizeof(rknn_tensor_mem *));
    w->out_attrs = calloc(w->n_out, sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < w->n_out; i++) {
        w->out_attrs[i].index = i;
        if (rknn_query(w->ctx, RKNN_QUERY_OUTPUT_ATTR, &w->out_attrs[i], sizeof(rknn_tensor_attr)) != RKNN_SUCC)
            return -1;
        w->out_mems[i] = rknn_create_mem(w->ctx, w->out_attrs[i].size_with_stride);
        if (!w->out_mems[i]) return -1;
        if (rknn_set_io_mem(w->ctx, w->out_mems[i], &w->out_attrs[i]) != RKNN_SUCC) return -1;
    }
    return 0;
}

static int init_ctx(rknn_context *ctx, const unsigned char *model, int model_len, int core_mask) {
    int ret = rknn_init(ctx, (void *)model, model_len, 0, NULL);
    if (ret != RKNN_SUCC) { printf("rknn_init failed %d\n", ret); return -1; }
    rknn_core_mask mask = (core_mask == 7) ? RKNN_NPU_CORE_0_1_2 :
                          (core_mask == 1) ? RKNN_NPU_CORE_0 :
                          (core_mask == 2) ? RKNN_NPU_CORE_1 : RKNN_NPU_CORE_2;
    ret = rknn_set_core_mask(*ctx, mask);
    if (ret != RKNN_SUCC) { printf("set_core_mask failed %d\n", ret); return -1; }
    return 0;
}

static int g_use_copy = 0;   // 1 = 走 rknn_inputs_set / outputs_get

static void *thread_body(void *arg) {
    worker_t *w = (worker_t *)arg;
    for (int i = 0; i < 8; i++) rknn_run(w->ctx, NULL);
    for (int i = 0; i < w->iters; i++) {
        double t0 = now_ms();
        if (g_use_copy) {
            rknn_input in; memset(&in, 0, sizeof(in));
            in.index = 0; in.type = RKNN_TENSOR_UINT8;
            in.size = w->in_attr.size; in.fmt = RKNN_TENSOR_NHWC;
            in.buf = w->in_mem->virt_addr;
            rknn_inputs_set(w->ctx, 1, &in);
            rknn_run(w->ctx, NULL);
            rknn_output outs[8]; memset(outs, 0, sizeof(outs));
            for (uint32_t k = 0; k < w->n_out; k++) { outs[k].index = k; outs[k].want_float = 0; }
            rknn_outputs_get(w->ctx, w->n_out, outs, NULL);
            rknn_outputs_release(w->ctx, w->n_out, outs);
        } else {
            rknn_run(w->ctx, NULL);
        }
        w->lat[i] = now_ms() - t0;
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) { printf("usage: %s model.rknn copy|zero|multi [iters]\n", argv[0]); return 1; }
    const char *mp = argv[1];
    const char *mode = argv[2];
    int iters = (argc > 3) ? atoi(argv[3]) : 100;

    int mlen = 0;
    unsigned char *model = read_file(mp, &mlen);
    if (!model) return 1;
    printf("model %s  %.2f MB  mode=%s iters=%d\n", mp, mlen / 1048576.0, mode, iters);

    if (strcmp(mode, "copy") == 0 || strcmp(mode, "zero") == 0) {
        rknn_context ctx;
        if (init_ctx(&ctx, model, mlen, 7) != 0) return 1;

        rknn_input_output_num io_num;
        rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        rknn_tensor_attr in;
        memset(&in, 0, sizeof(in)); in.index = 0;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in, sizeof(in));
        printf("  input: n_dims=%u dims=[%u,%u,%u,%u] size=%u fmt=%d type=%d qnt=%d\n",
               in.n_dims, in.dims[0], in.dims[1], in.dims[2], in.dims[3], in.size,
               in.fmt, in.type, in.qnt_type);
        rknn_tensor_attr oa; memset(&oa,0,sizeof(oa)); oa.index=0;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &oa, sizeof(oa));
        printf("  output0: dims=[%u,%u,%u,%u] size=%u fmt=%d\n",
               oa.dims[0], oa.dims[1], oa.dims[2], oa.dims[3], oa.size, oa.fmt);

        worker_t w; memset(&w, 0, sizeof(w));
        w.ctx = ctx; w.n_out = io_num.n_output;

        rknn_tensor_mem *in_mem = NULL;
        if (strcmp(mode, "zero") == 0) {
            if (setup_io(&w) != 0) { printf("setup_io failed\n"); return 1; }
            in_mem = w.in_mem;
        } else {
            in_mem = rknn_create_mem(ctx, in.size_with_stride);
            memset(in_mem->virt_addr, 0, in.size_with_stride);
        }

        // 预热
        for (int i = 0; i < 8; i++) rknn_run(ctx, NULL);

        double *lat = malloc(sizeof(double) * iters);
        double *all = malloc(sizeof(double) * iters);
        for (int i = 0; i < iters; i++) {
            double t0 = now_ms();
            if (strcmp(mode, "zero") == 0) {
                rknn_run(ctx, NULL);          // 数据已在绑定的内存里，直接跑
            } else {
                rknn_input inputs[1];
                memset(inputs, 0, sizeof(inputs));
                inputs[0].index = 0;
                inputs[0].type = RKNN_TENSOR_UINT8;
                inputs[0].size = in.size;
                inputs[0].fmt = RKNN_TENSOR_NHWC;
                inputs[0].buf = in_mem->virt_addr;
                int rs = rknn_inputs_set(ctx, 1, inputs);
                int rr = rknn_run(ctx, NULL);
                if (i == 0) printf("  [check] inputs_set=%d run=%d\n", rs, rr);
                rknn_output outputs[8];
                memset(outputs, 0, sizeof(outputs));
                for (uint32_t k = 0; k < io_num.n_output; k++) {
                    outputs[k].index = k;
                    outputs[k].want_float = 0;   // ★ 不反量化，避免拷贝
                    outputs[k].is_prealloc = 0;
                }
                rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
                rknn_outputs_release(ctx, io_num.n_output, outputs);
            }
            all[i] = now_ms() - t0;
            lat[i] = all[i];
        }
        qsort(lat, iters, sizeof(double), cmp_double);
        double med = lat[iters / 2], p90 = lat[(int)(iters * 0.9)];
        printf("  %s: 中位 %.2f ms  p90 %.2f ms  FPS %.2f\n", mode, med, p90, 1000.0 / med);
    } else if (strcmp(mode, "multi") == 0 || strcmp(mode, "multi_copy") == 0) {
        g_use_copy = (strcmp(mode, "multi_copy") == 0);
        const int NT = 3;
        rknn_context ctx[NT];
        if (init_ctx(&ctx[0], model, mlen, 1) != 0) return 1;
        for (int i = 1; i < NT; i++) {
            int ret = rknn_dup_context(&ctx[0], &ctx[i]);
            if (ret != RKNN_SUCC) { printf("dup_context failed %d\n", ret); return 1; }
            rknn_core_mask m = (i == 1) ? RKNN_NPU_CORE_1 : RKNN_NPU_CORE_2;
            rknn_set_core_mask(ctx[i], m);
        }
        worker_t w[NT];
        for (int i = 0; i < NT; i++) {
            memset(&w[i], 0, sizeof(worker_t));
            w[i].ctx = ctx[i];
            if (g_use_copy) {
                rknn_input_output_num io_num;
                rknn_query(ctx[i], RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
                w[i].n_out = io_num.n_output;
                memset(&w[i].in_attr, 0, sizeof(w[i].in_attr));
                w[i].in_attr.index = 0;
                rknn_query(ctx[i], RKNN_QUERY_INPUT_ATTR, &w[i].in_attr, sizeof(w[i].in_attr));
                w[i].in_mem = rknn_create_mem(ctx[i], w[i].in_attr.size_with_stride);
                memset(w[i].in_mem->virt_addr, 0, w[i].in_attr.size_with_stride);
            } else if (setup_io(&w[i]) != 0) { printf("setup_io %d failed\n", i); return 1; }
            w[i].iters = iters;
            w[i].lat = malloc(sizeof(double) * iters);
        }
        pthread_t th[NT];
        double t0 = now_ms();
        for (int i = 0; i < NT; i++) pthread_create(&th[i], NULL, thread_body, &w[i]);
        for (int i = 0; i < NT; i++) pthread_join(th[i], NULL);
        double wall = (now_ms() - t0) / 1000.0;
        int total = NT * iters;
        printf("  multi(%d 实例, dup_context, 零拷贝): 总 %d 帧 / %.2f s = %.2f FPS\n",
               NT, total, wall, total / wall);
        for (int i = 0; i < NT; i++) {
            qsort(w[i].lat, iters, sizeof(double), cmp_double);
            printf("    实例%d 中位 %.2f ms\n", i, w[i].lat[iters / 2]);
        }
    }
    free(model);
    return 0;
}
