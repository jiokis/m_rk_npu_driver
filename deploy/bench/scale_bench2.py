#!/usr/bin/env python3
"""C=16 k=3 卷积在不同分辨率下的 NPU 延迟（3核）"""
import statistics, time
import numpy as np
from rknnlite.api import RKNNLite

print("  尺寸    MAC量(G)   中位(ms)   min(ms)   相对320")
base = None
for S in (320, 640, 1280):
    p = "/home/orangepi/conv-sweep2/c16_%d.rknn" % S
    r = RKNNLite(verbose=False)
    assert r.load_rknn(p) == 0
    assert r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0
    x = np.zeros((1, S, S, 16), dtype=np.uint8)
    for _ in range(10):
        r.inference(inputs=[x])
    ts = []
    for _ in range(100):
        t0 = time.perf_counter(); r.inference(inputs=[x]); ts.append((time.perf_counter()-t0)*1000)
    r.release(); ts.sort()
    med = statistics.median(ts)
    if base is None: base = med
    mac = 16*16*9*S*S/1e9
    print("  %4d2  %8.2f  %8.3f  %8.3f   %5.2fx" % (S, mac, med, ts[0], med/base))
