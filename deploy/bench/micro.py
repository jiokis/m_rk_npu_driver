import time, statistics
import numpy as np, cv2
from rknnlite.api import RKNNLite

def tm(fn, n=20):
    fn(); ts=[]
    for _ in range(n):
        t0=time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000)
    ts.sort(); return ts[len(ts)//2]

img1280 = np.random.randint(0,255,(1280,1280,3),np.uint8)
img640  = np.random.randint(0,255,(640,640,3),np.uint8)

def lb(im, S):
    h,w = im.shape[:2]; r = min(S/h, S/w)
    nh,nw = int(round(h*r)), int(round(w*r))
    rz = cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)
    c = np.full((S,S,3),114,np.uint8)
    t,l = (S-nh)//2,(S-nw)//2
    c[t:t+nh,l:l+nw]=rz
    return c

print(f"  letterbox 1280->1280 : {tm(lambda: lb(img1280,1280)):6.2f} ms")
print(f"  letterbox 640->640   : {tm(lambda: lb(img640,640)):6.2f} ms")
print(f"  letterbox 640->640 纯 resize: {tm(lambda: cv2.resize(img640,(640,640))):6.2f} ms")
print(f"  np.full 1280         : {tm(lambda: np.full((1280,1280,3),114,np.uint8)):6.2f} ms")
print(f"  np.full 640          : {tm(lambda: np.full((640,640,3),114,np.uint8)):6.2f} ms")

o = np.random.rand(1,84,8400).astype(np.float32)*0.01
def dec_squeeze():
    p=np.squeeze(o).T; s=p[:,4:]; return s.max(1)
def dec_direct():
    a=o[0]; s=a[4:,:]; return s.max(0)
print(f"  decode squeeze+transpose : {tm(dec_squeeze):6.2f} ms")
print(f"  decode 直接切片(无转置)  : {tm(dec_direct):6.2f} ms")

r=RKNNLite(verbose=False); r.load_rknn("/home/orangepi/y11/yolo11n_640.rknn")
r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
x=np.zeros((1,640,640,3),np.uint8)
for _ in range(5): r.inference(inputs=[x])
print(f"  推理(复用同一输入数组)   : {tm(lambda: r.inference(inputs=[x]),10):6.2f} ms")
def fresh():
    y=np.zeros((1,640,640,3),np.uint8); return r.inference(inputs=[y])
print(f"  推理(每次新建输入数组)   : {tm(fresh,10):6.2f} ms")
r.release()
