#!/usr/bin/env python3
"""稳态抓取：warmup W 次 + 测量 M 次，全部抓取；用于取稳态每趟耗时与地板。"""
import os, sys, time, statistics
from rknnlite.api import RKNNLite
import numpy as np

model = sys.argv[1]; cm = int(sys.argv[2]); W = int(sys.argv[3]); M = int(sys.argv[4])
CMAP = {1: RKNNLite.NPU_CORE_0, 3: RKNNLite.NPU_CORE_0_1, 7: RKNNLite.NPU_CORE_0_1_2}
r = RKNNLite(verbose=False)
assert r.load_rknn(model) == 0, "load"
assert r.init_runtime(core_mask=CMAP[cm]) == 0, "init"
x = np.zeros((1, 640, 640, 3), dtype=np.uint8)
for _ in range(W):
    r.inference(inputs=[x])
print("WARMUP_DONE", flush=True)
ts = []
for _ in range(M):
    t0 = time.perf_counter(); r.inference(inputs=[x]); ts.append((time.perf_counter()-t0)*1000)
r.release()
print("MEASURED " + " ".join(f"{v:.2f}" for v in ts))
print(f"median={statistics.median(ts):.2f} mean={statistics.mean(ts):.2f} ms")
