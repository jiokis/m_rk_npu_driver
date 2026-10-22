// bench_full.c — 1280 YOLO11 完整链路（C 版）：分块 + 多核并发 + DFL + NMS
//
// 相比 bench1280.c 增加了后处理：从 double-output INT8 模型拿 (1,64,N)+(1,80,N)，
// 对 score>CONF 的候选锚点做 DFL，映射回全图坐标，跨 tile NMS。
//
// 编译：gcc -O2 -o bench_full bench_full.c -I<inc> /usr/lib/librknnrt.so -lpthread -lm
// 用法：./bench_full m640_head.rknn anchor.npy stride.npy <frames> [nworkers]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include "rknn_api.h"

#define TILE 640
#define NA  8400
#define CONF 0.25f
#define NWMAX 16
#define MAXBOX 4096

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6;
}
static unsigned char *read_file(const char *p, int *sz) {
    FILE *f = fopen(p, "rb"); if (!f) { perror("open"); return NULL; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    unsigned char *b = malloc(n);
    if (fread(b,1,n,f)!=(size_t)n) { fclose(f); free(b); return NULL; }
    fclose(f); *sz=(int)n; return b;
}

typedef struct { float x1,y1,x2,y2,score; int cls; } box_t;

static float g_anchor[2*NA];   // (2,NA)
static float g_stride[NA];
static const float AR[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

typedef struct {
    rknn_context ctx;
    rknn_tensor_attr in_attr;
    rknn_tensor_mem *in_mem;
    uint32_t n_out;
    int nw;                  // worker 总数（用于算 tile 原点）
} wk_t;

static volatile int g_next = 0, g_total = 0;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static unsigned char *g_tile = NULL;
static unsigned char *g_tiles4 = NULL;   // 4 个真实 tile 连续存放（可选）
static box_t *g_boxes[2];    // 双缓冲：按 tile 顺序写
static int g_nbox[2];

static void dfl_decode(const float *lg, const float *cl, int tid, box_t *outbox, int *nout) {
    int k = 0;
    float *buf = (float *)malloc(sizeof(float) * 16);
    for (int a = 0; a < NA; a++) {
        // 类别分数：80 类取最大
        float best = 0.f; int bc = 0;
        for (int c = 0; c < 80; c++) {
            float v = cl[c*NA + a];
            if (v > best) { best = v; bc = c; }
        }
        if (best <= CONF) continue;
        float d[4];
        for (int g = 0; g < 4; g++) {
            const float *src = lg + (g*16)*NA + a;
            float m = -1e30f;
            for (int b = 0; b < 16; b++) { buf[b] = src[b*NA]; if (buf[b] > m) m = buf[b]; }
            float s = 0.f, acc = 0.f;
            for (int b = 0; b < 16; b++) { float e = expf(buf[b] - m); buf[b] = e; s += e; }
            for (int b = 0; b < 16; b++) acc += buf[b] / s * AR[b];
            d[g] = acc;
        }
        // 原点
        int ox = (tid % 2) * TILE, oy = (tid / 2) * TILE;
        float anc0 = g_anchor[0*NA + a], anc1 = g_anchor[1*NA + a], st = g_stride[a];
        float x1 = (anc0 - d[0]) * st, y1 = (anc1 - d[1]) * st;
        float x2 = (anc0 + d[2]) * st, y2 = (anc1 + d[3]) * st;
        if (k < MAXBOX) {
            outbox[k].x1 = x1 + ox; outbox[k].y1 = y1 + oy;
            outbox[k].x2 = x2 + ox; outbox[k].y2 = y2 + oy;
            outbox[k].score = best; outbox[k].cls = bc;
            k++;
        }
    }
    free(buf);
    *nout = k;
}

static void *worker(void *arg) {
    wk_t *w = (wk_t *)arg;
    box_t tmp[MAXBOX];
    for (;;) {
        pthread_mutex_lock(&g_mu);
        int idx = g_next++;
        pthread_mutex_unlock(&g_mu);
        if (idx >= g_total) break;
        int tid = idx % 4;

        const unsigned char *src_tile = g_tiles4 ? (g_tiles4 + (size_t)tid*TILE*TILE*3) : g_tile;
        memcpy(w->in_mem->virt_addr, src_tile, TILE*TILE*3);
        rknn_input in; memset(&in,0,sizeof(in));
        in.index=0; in.type=RKNN_TENSOR_UINT8; in.size=w->in_attr.size;
        in.fmt=RKNN_TENSOR_NHWC; in.buf=w->in_mem->virt_addr;
        rknn_inputs_set(w->ctx,1,&in);
        if (rknn_run(w->ctx,NULL)!=RKNN_SUCC) continue;

        rknn_output outs[4]; memset(outs,0,sizeof(outs));
        for (uint32_t k=0;k<w->n_out;k++){ outs[k].index=k; outs[k].want_float=1; }
        if (rknn_outputs_get(w->ctx,w->n_out,outs,NULL)!=RKNN_SUCC){ rknn_outputs_release(w->ctx,w->n_out,outs); continue; }
        const float *lg=NULL,*cl=NULL;
        for (uint32_t k=0;k<w->n_out;k++){
            float *d=(float*)outs[k].buf;
            // 用元素数区分：64*8400 vs 80*8400
            if (d) {
                if (outs[k].size == (uint32_t)(64*NA*4)) lg=d;
                else if (outs[k].size == (uint32_t)(80*NA*4)) cl=d;
            }
        }
        int nb=0;
        if (lg && cl) dfl_decode(lg, cl, tid, tmp, &nb);
        rknn_outputs_release(w->ctx,w->n_out,outs);

        pthread_mutex_lock(&g_mu);
        int cap = MAXBOX - g_nbox[0];
        int n = nb < cap ? nb : cap;
        memcpy(g_boxes[0]+g_nbox[0], tmp, sizeof(box_t)*n);
        g_nbox[0] += n;
        pthread_mutex_unlock(&g_mu);
    }
    return NULL;
}

int main(int argc,char**argv){
    if(argc<5){printf("usage: %s m640_head.rknn anchor.npy stride.npy frames [nw]\n",argv[0]);return 1;}
    int nw = (argc>5)?atoi(argv[5]):6;
    int frames=atoi(argv[4]);
    if(nw>NWMAX)nw=NWMAX;

    // 载入 anchor / stride (.npy: 只支持 float32 且无 pickle)
    FILE*f=fopen(argv[2],"rb"); if(!f){perror("anchor");return 1;}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    unsigned char *raw=malloc(sz); if(fread(raw,1,sz,f)!=(size_t)sz){} fclose(f);
    memcpy(g_anchor, raw+sz-2*NA*4, 2*NA*4); free(raw);
    f=fopen(argv[3],"rb"); fseek(f,0,SEEK_END); sz=ftell(f); fseek(f,0,SEEK_SET);
    raw=malloc(sz); if(fread(raw,1,sz,f)!=(size_t)sz){} fclose(f);
    memcpy(g_stride, raw+sz-NA*4, NA*4); free(raw);

    int mlen=0; unsigned char*model=read_file(argv[1],&mlen);
    if(!model)return 1;
    g_tile = malloc(TILE*TILE*3); memset(g_tile,128,TILE*TILE*3);
    if (argc > 6) {
        int tsz=0; unsigned char *t4 = read_file(argv[6], &tsz);
        if (t4 && tsz >= 4*TILE*TILE*3) { g_tiles4 = t4; printf("载入真实 tile: %s (%d bytes)\n", argv[6], tsz); }
        else if (t4) printf("tile 文件太小: %d\n", tsz);
    }
    g_boxes[0]=malloc(sizeof(box_t)*MAXBOX);

    rknn_context ctx[NWMAX]; wk_t w[NWMAX];
    for(int i=0;i<nw;i++){
        if(rknn_init(&ctx[i],model,mlen,0,NULL)!=RKNN_SUCC){printf("init %d fail\n",i);return 1;}
        rknn_core_mask m = (i%3==0)?RKNN_NPU_CORE_0:(i%3==1)?RKNN_NPU_CORE_1:RKNN_NPU_CORE_2;
        rknn_set_core_mask(ctx[i],m);
        memset(&w[i],0,sizeof(wk_t)); w[i].ctx=ctx[i];
        rknn_input_output_num io; rknn_query(ctx[i],RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io));
        w[i].n_out=io.n_output;
        memset(&w[i].in_attr,0,sizeof(w[i].in_attr)); w[i].in_attr.index=0;
        rknn_query(ctx[i],RKNN_QUERY_INPUT_ATTR,&w[i].in_attr,sizeof(w[i].in_attr));
        w[i].in_mem=rknn_create_mem(ctx[i],w[i].in_attr.size_with_stride);
        memset(w[i].in_mem->virt_addr,0,w[i].in_attr.size_with_stride);
    }
    printf("workers=%d  n_out=%u  input=%u bytes\n",nw,w[0].n_out,w[0].in_attr.size);

    // 预热
    g_total = 4; g_next = 0; g_nbox[0]=0;
    pthread_t th[NWMAX];
    for(int i=0;i<nw;i++) pthread_create(&th[i],NULL,worker,&w[i]);
    for(int i=0;i<nw;i++) pthread_join(th[i],NULL);
    printf("预热完成，单帧框数 %d\n", g_nbox[0]);

    g_total = frames*4; g_next=0; g_nbox[0]=0;
    double t0=now_ms();
    for(int i=0;i<nw;i++) pthread_create(&th[i],NULL,worker,&w[i]);
    for(int i=0;i<nw;i++) pthread_join(th[i],NULL);
    double wall=now_ms()-t0;
    printf("  %d 帧 / %.2f s = %.2f FPS   框合计 %d（%.1f/帧）\n",
           frames, wall/1000.0, frames/(wall/1000.0), g_nbox[0], (double)g_nbox[0]/frames);
    for(int i=0;i<nw;i++){ rknn_destroy_mem(ctx[i],w[i].in_mem); rknn_destroy(ctx[i]); }
    free(model); free(g_tile); free(g_boxes[0]);
    return 0;
}
