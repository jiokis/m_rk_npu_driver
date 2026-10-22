"""交替多轮测：1核/3核单实例 vs 多实例并发（消除热漂移）"""
import threading, time, statistics
import numpy as np
from rknnlite.api import RKNNLite
M="/home/orangepi/y11/yolo11n_640.rknn"; S=640
x=np.zeros((1,S,S,3),np.uint8)
def make(mask):
    r=RKNNLite(verbose=False); r.load_rknn(M); r.init_runtime(core_mask=mask); return r
def solo(mask,n=80):
    r=make(mask)
    for _ in range(8): r.inference(inputs=[x])
    ts=[]
    for _ in range(n):
        t0=time.perf_counter(); r.inference(inputs=[x]); ts.append((time.perf_counter()-t0)*1000)
    r.release(); ts.sort(); return statistics.median(ts)
def conc(masks,n=80):
    rs=[make(m) for m in masks]
    for r in rs:
        for _ in range(8): r.inference(inputs=[x])
    bar=threading.Barrier(len(rs))
    def work(r):
        bar.wait()
        for _ in range(n): r.inference(inputs=[x])
    ths=[threading.Thread(target=work,args=(r,)) for r in rs]
    t0=time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    wall=time.perf_counter()-t0
    for r in rs: r.release()
    return n*len(rs)/wall

res={"1核单实例":[], "3核单实例":[], "2实例并发":[], "3实例并发":[]}
for rnd in range(3):
    res["1核单实例"].append(1000/solo(RKNNLite.NPU_CORE_0))
    res["3核单实例"].append(1000/solo(RKNNLite.NPU_CORE_0_1_2))
    res["2实例并发"].append(conc((RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1)))
    res["3实例并发"].append(conc((RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2)))
print(f"{'配置':<12}{'第1轮':>9}{'第2轮':>9}{'第3轮':>9}{'中位FPS':>10}{'相对':>8}")
base=statistics.median(res["3核单实例"])
for k,v in res.items():
    m=statistics.median(v)
    print(f"{k:<12}{v[0]:>9.2f}{v[1]:>9.2f}{v[2]:>9.2f}{m:>10.2f}{m/base:>7.2f}x")
