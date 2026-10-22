import time, statistics
import numpy as np
from rknnlite.api import RKNNLite
x=np.zeros((1,640,640,3),np.uint8)
models=[("det(基准)","yolo11n_640_head_real.rknn"),
        ("pose","yolo11n-pose-split-int8.rknn"),
        ("seg","yolo11n-seg-split-int8.rknn"),
        ("obb","yolo11n-obb-split-int8.rknn"),
        ("cls","yolo11n-cls-int8.rknn")]
print(f"  {'任务':<12}{'中位(ms)':>10}{'FPS':>8}{'模型(MB)':>10}")
import os
for name,f in models:
    p=f"/home/orangepi/y11/{f}"
    if not os.path.exists(p): print(f"  {name:<12}  缺文件"); continue
    r=RKNNLite(verbose=False); r.load_rknn(p); r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    for _ in range(8): r.inference(inputs=[x])
    ts=[]
    for _ in range(60):
        t0=time.perf_counter(); r.inference(inputs=[x]); ts.append((time.perf_counter()-t0)*1000)
    r.release(); ts.sort()
    med=ts[len(ts)//2]
    print(f"  {name:<12}{med:>10.2f}{1000/med:>8.1f}{os.path.getsize(p)/1048576:>10.2f}")
