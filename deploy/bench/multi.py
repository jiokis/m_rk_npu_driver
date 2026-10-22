"""多实例并行：单实例(3核) vs 2/3 实例各绑一核 —— 直接利用「56% 提交只用 1 核」这个事实"""
import threading, time, statistics
import numpy as np
from rknnlite.api import RKNNLite

M="/home/orangepi/y11/yolo11n_640.rknn"
S=640
x=np.zeros((1,S,S,3),np.uint8)

def make(mask):
    r=RKNNLite(verbose=False); r.load_rknn(M); r.init_runtime(core_mask=mask); return r

def solo(mask, n=60):
    r=make(mask)
    for _ in range(8): r.inference(inputs=[x])
    ts=[]
    for _ in range(n):
        t0=time.perf_counter(); r.inference(inputs=[x]); ts.append((time.perf_counter()-t0)*1000)
    r.release(); ts.sort()
    return statistics.median(ts)

def concurrent(masks, n=60):
    rs=[make(m) for m in masks]
    for r in rs:
        for _ in range(8): r.inference(inputs=[x])
    lat=[[] for _ in rs]; bar=threading.Barrier(len(rs))
    def work(i,r):
        bar.wait(); t=[]
        for _ in range(n):
            t0=time.perf_counter(); r.inference(inputs=[x]); t.append((time.perf_counter()-t0)*1000)
        lat[i]=t
    ths=[threading.Thread(target=work,args=(i,r)) for i,r in enumerate(rs)]
    t0=time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    wall=(time.perf_counter()-t0)
    for r in rs: r.release()
    total=sum(len(t) for t in lat)
    return wall, total/wall, [statistics.median(t) for t in lat]

print("=== 单实例 ===")
for name,mask in (("3核",RKNNLite.NPU_CORE_0_1_2),("1核",RKNNLite.NPU_CORE_0)):
    t=solo(mask)
    print(f"  {name:<4} 中位 {t:6.2f} ms   FPS {1000/t:5.2f}")

print("\n=== 多实例并发（每个绑不同核）===")
for masks,label in (((RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1),"2实例 0|1"),
                    ((RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2),"3实例 0|1|2")):
    try:
        wall, fps, meds = concurrent(masks)
        print(f"  {label:<10} 总吞吐 {fps:5.2f} FPS   (墙钟 {wall:.2f}s, 各实例中位 {[round(m,1) for m in meds]})")
    except Exception as e:
        print(f"  {label}: 失败 {type(e).__name__}: {str(e)[:60]}")
