import time, statistics
import numpy as np
from rknnlite.api import RKNNLite

def load(p):
    r=RKNNLite(verbose=False); r.load_rknn(p); r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2); return r

r6=load("/home/orangepi/y11/yolo11n_640.rknn")
r12=load("/home/orangepi/y11/yolo11n_1280.rknn")
x6=np.zeros((1,640,640,3),np.uint8); x12=np.zeros((1,1280,1280,3),np.uint8)
for _ in range(8):
    r6.inference(inputs=[x6]); r12.inference(inputs=[x12])

def bench(r,x,n=40):
    ts=[]
    for _ in range(n):
        t0=time.perf_counter(); r.inference(inputs=[x]); ts.append((time.perf_counter()-t0)*1000)
    ts.sort(); return statistics.median(ts), ts[0]

t6,_=bench(r6,x6); t12,_=bench(r12,x12)
print(f"  640  x1 : {t6:7.2f} ms")
print(f"  1280 x1 : {t12:7.2f} ms   (比值 {t12/t6:.2f}x)")
print(f"  4x640   : {t6*4:7.2f} ms   (对 1280 的比值 {t6*4/t12:.2f}x)")
r6.release(); r12.release()
