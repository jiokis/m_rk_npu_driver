import sys, time, statistics
import numpy as np
from rknnlite.api import RKNNLite
p, S = sys.argv[1], int(sys.argv[2])
r = RKNNLite(verbose=False)
assert r.load_rknn(p) == 0, "load 失败"
assert r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0, "init 失败"
x = np.zeros((1, S, S, 3), dtype=np.uint8)
o = r.inference(inputs=[x])
print("首次推理输出:", None if o is None else [a.shape for a in o])
if o is None:
    print("INFER_FAILED"); sys.exit(2)
for _ in range(10): r.inference(inputs=[x])
ts=[]
for _ in range(100):
    t0=time.perf_counter(); r.inference(inputs=[x]); ts.append((time.perf_counter()-t0)*1000)
r.release(); ts.sort()
print(f"中位 {statistics.median(ts):.2f} ms  min {ts[0]:.2f}  p90 {ts[int(len(ts)*0.9)]:.2f}  FPS {1000/statistics.median(ts):.1f}")
