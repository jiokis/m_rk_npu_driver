import time, statistics
import numpy as np, cv2
from rknnlite.api import RKNNLite

r = RKNNLite(verbose=False)
r.load_rknn("/home/orangepi/y11/yolo11n_640.rknn")
r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
x = np.zeros((1,640,640,3), np.uint8)
for _ in range(5): r.inference(inputs=[x])

# 1) 连续 4 次推理的耗时
ts=[]
for _ in range(10):
    t0=time.perf_counter()
    for _ in range(4): r.inference(inputs=[x])
    ts.append((time.perf_counter()-t0)*1000)
ts.sort()
print(f"  4 次推理纯耗时: 中位 {ts[len(ts)//2]:.2f} ms  (每次 {ts[len(ts)//2]/4:.2f} ms)")

# 2) 单次推理 + decode + NMS
o = r.inference(inputs=[x]); r.release()
a = np.squeeze(o[0]).T
print(f"  输出 {np.shape(a)}")
sc = a[:,4:].max(1); print(f"  分数>0.25 的框: {(sc>0.25).sum()} 个  >0.5: {(sc>0.5).sum()}  最大分数 {sc.max():.3f}")
m = sc>0.25
box = a[m,:4]; s = sc[m]; c = a[m,4:].argmax(1)
xyxy = np.concatenate([box[:,:2]-box[:,2:]/2, box[:,:2]+box[:,2:]/2],1)
t0=time.perf_counter()
out=[]
for k in np.unique(c):
    b = xyxy[c==k]; ss=s[c==k]
    xywh = np.concatenate([b[:,:2], b[:,2:4]-b[:,:2]],1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), ss.tolist(), 0.25, 0.45)
    out.extend(idx)
print(f"  NMS 耗时 {((time.perf_counter()-t0)*1000):.2f} ms  → 保留 {len(out)} 个")
