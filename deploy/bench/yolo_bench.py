#!/usr/bin/env python3
"""yolo_bench.py — YOLOv5s 在 RK3588 NPU 上的实测速度（多种 core_mask）"""
import time, statistics
import numpy as np
from rknnlite.api import RKNNLite

MODEL = "/opt/rknn-models/yolov5s-640-640.rknn"
SHAPE = (1, 640, 640, 3)

def bench(cm, name, iters=100, warm=10):
    r = RKNNLite(verbose=False)
    assert r.load_rknn(MODEL) == 0
    assert r.init_runtime(core_mask=cm) == 0
    x = np.zeros(SHAPE, dtype=np.uint8)
    for _ in range(warm):
        r.inference(inputs=[x])
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = r.inference(inputs=[x])
        ts.append((time.perf_counter() - t0) * 1000)
    r.release()
    ts.sort()
    n = len(ts)
    print(f"{name:<14} n={n}  中位 {statistics.median(ts):7.2f} ms   "
          f"均值 {statistics.mean(ts):7.2f}   min {ts[0]:6.2f}   "
          f"p90 {ts[int(n*0.9)]:6.2f}   FPS(中位) {1000/statistics.median(ts):6.2f}")
    return statistics.median(ts)

print(f"模型: {MODEL}")
print(f"输入: {SHAPE} (NHWC uint8)")
print()
res = {}
res["1核 (core0)"]   = bench(RKNNLite.NPU_CORE_0,      "1核 core0")
res["2核 (core0_1)"] = bench(RKNNLite.NPU_CORE_0_1,    "2核 core0_1")
res["3核 (core0_1_2)"]= bench(RKNNLite.NPU_CORE_0_1_2, "3核 core0_1_2")
res["AUTO"]          = bench(RKNNLite.NPU_CORE_AUTO,   "AUTO")
print()
best = min(res.items(), key=lambda kv: kv[1])
print(f"最快: {best[0]}  {best[1]:.2f} ms  ({1000/best[1]:.1f} FPS)")
