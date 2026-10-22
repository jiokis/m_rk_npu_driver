import time
import numpy as np, cv2
from rknnlite.api import RKNNLite

def tm(fn,n=20):
    fn(); ts=[]
    for _ in range(n):
        t0=time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000)
    ts.sort(); return ts[len(ts)//2]

img = np.random.randint(0,255,(1280,1280,3),np.uint8)
def lb(im,S=640):
    h,w=im.shape[:2]; r=min(S/h,S/w); nh,nw=int(round(h*r)),int(round(w*r))
    rz=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)
    c=np.full((S,S,3),114,np.uint8); t,l=(S-nh)//2,(S-nw)//2; c[t:t+nh,l:l+nw]=rz
    return c

crop_view = img[0:640, 0:640]          # 非连续视图
crop_copy = np.ascontiguousarray(crop_view)
print(f"  letterbox(非连续切片) : {tm(lambda: lb(crop_view)):6.2f} ms")
print(f"  letterbox(连续拷贝)   : {tm(lambda: lb(crop_copy)):6.2f} ms")
print(f"  ascontiguousarray     : {tm(lambda: np.ascontiguousarray(crop_view)):6.2f} ms")

r=RKNNLite(verbose=False); r.load_rknn("/home/orangepi/y11/yolo11n_640.rknn")
r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
x=np.zeros((1,640,640,3),np.uint8)
for _ in range(5): r.inference(inputs=[x])
out = r.inference(inputs=[x])
def dec():
    p=np.squeeze(out[0]).T
    s=p[:,4:]; return s.max(1)
print(f"  真实输出的 decode      : {tm(dec):6.2f} ms")
def dec2():
    a=out[0][0]            # (84,8400) 连续
    s=a[4:,:]; return s.max(0)
print(f"  真实输出 无转置 decode : {tm(dec2):6.2f} ms")
r.release()
