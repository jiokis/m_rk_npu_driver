import time, statistics
import numpy as np
anchor=np.load("/home/orangepi/y11/anchor640.npy"); stride=np.load("/home/orangepi/y11/stride640.npy")
N=8400
logits=np.random.rand(1,64,N).astype(np.float32)
cls=np.random.rand(1,80,N).astype(np.float32)
AR=np.arange(16,dtype=np.float32)

def v1(logits, cls):
    p=logits.reshape(1,4,16,N).transpose(0,3,1,2)
    e=np.exp(p-p.max(-1,keepdims=True)); pr=e/e.sum(-1,keepdims=True)
    pr=pr.transpose(0,3,2,1)
    d=(pr*AR.reshape(1,16,1,1)).sum(1)
    x1=anchor-d[:,0:2]; x2=anchor+d[:,2:4]
    return np.concatenate([np.concatenate([(x1+x2)*0.5, x2-x1],1)*stride, cls],1)

def v2(logits, cls):
    """按 4 组切片 + BLAS 求期望，避免任何转置"""
    L=logits[0]
    d=np.empty((4,N),np.float32)
    for g in range(4):
        s=L[g*16:(g+1)*16]           # (16,N) 连续
        m=s.max(0)
        e=np.exp(s-m)
        e/=e.sum(0)
        d[g]=AR@e                    # BLAS
    d=d[None]
    x1=anchor-d[:,0:2]; x2=anchor+d[:,2:4]
    return np.concatenate([np.concatenate([(x1+x2)*0.5, x2-x1],1)*stride, cls],1)

for name,fn in (("v1 转置版",v1),("v2 切片+BLAS",v2)):
    fn(logits,cls); ts=[]
    for _ in range(30):
        t0=time.perf_counter(); fn(logits,cls); ts.append((time.perf_counter()-t0)*1000)
    ts.sort(); print(f"  {name:<14} {ts[len(ts)//2]:6.2f} ms")
a=v1(logits,cls); b=v2(logits,cls)
print(f"  两者最大差 {np.abs(a-b).max():.2e}")
