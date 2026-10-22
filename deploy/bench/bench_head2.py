"""去 DFL 方案 v2：尾部只对候选锚点计算"""
import time, statistics
import numpy as np
from rknnlite.api import RKNNLite

S=640; CONF=0.25
anchor=np.load("/home/orangepi/y11/anchor640.npy"); stride=np.load("/home/orangepi/y11/stride640.npy")
AR=np.arange(16,dtype=np.float32)
x=np.zeros((1,S,S,3),np.uint8)

def load(p):
    r=RKNNLite(verbose=False); r.load_rknn(p); r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2); return r
r_full=load("/home/orangepi/y11/yolo11n_640.rknn")
r_head=load("/home/orangepi/y11/yolo11n_640_head.rknn")
for _ in range(8):
    r_full.inference(inputs=[x]); r_head.inference(inputs=[x])

def tail_all(lg, cl):
    p=lg.reshape(1,4,16,lg.shape[2]).transpose(0,3,1,2)
    e=np.exp(p-p.max(-1,keepdims=True)); pr=e/e.sum(-1,keepdims=True)
    d=(pr.transpose(0,3,2,1)*AR.reshape(1,16,1,1)).sum(1)
    x1=anchor-d[:,0:2]; x2=anchor+d[:,2:4]
    return np.concatenate([np.concatenate([(x1+x2)*0.5, x2-x1],1)*stride, cl],1)

def tail_sel(lg, cl, conf=CONF):
    """★ 只对候选锚点算 DFL"""
    sc=cl[0].max(0)
    idx=np.flatnonzero(sc>conf)
    if idx.size==0:
        return np.zeros((1,84,0),np.float32)
    L=lg[0][:,idx]                         # (64,k) gather
    d=np.empty((4,idx.size),np.float32)
    for g in range(4):
        s=L[g*16:(g+1)*16]
        m=s.max(0); e=np.exp(s-m); e/=e.sum(0)
        d[g]=AR@e
    d=d[None]
    a=anchor[:,:,idx]; st=stride[:,idx]
    x1=a-d[:,0:2]; x2=a+d[:,2:4]
    box=np.concatenate([(x1+x2)*0.5, x2-x1],1)*st
    return np.concatenate([box, cl[:,:,idx]],1), idx

def bench(fn,n=40):
    ts=[]; o=None
    for _ in range(n):
        t0=time.perf_counter(); o=fn(); ts.append((time.perf_counter()-t0)*1000)
    ts.sort(); return statistics.median(ts), o

t_full,_=bench(lambda: r_full.inference(inputs=[x])[0])
def ht_all():
    a=r_head.inference(inputs=[x]); return tail_all(a[0],a[1])
def ht_sel():
    a=r_head.inference(inputs=[x]); return tail_sel(a[0],a[1])
t_all,_=bench(ht_all); t_sel,_=bench(ht_sel)
t_head,_=bench(lambda: r_head.inference(inputs=[x])[0])

print(f"  完整模型                : {t_full:7.2f} ms")
print(f"  head + 全量DFL          : {t_all:7.2f} ms  ({(t_full/t_all-1)*100:+.1f}%)")
print(f"  head + 候选DFL          : {t_sel:7.2f} ms  ({(t_full/t_sel-1)*100:+.1f}%)")
print(f"    head 推理 {t_head:.2f} ms / 尾部 {t_sel-t_head:.2f} ms")
print(f"  候选数（全零输入）: {len(tail_sel(*[np.asarray(z) for z in r_head.inference(inputs=[x])])[1]) if False else '见下'}")
a=r_head.inference(inputs=[x]); r,tmp=tail_sel(a[0],a[1])
print(f"  候选项 {tmp.size} 个")
r_full.release(); r_head.release()
