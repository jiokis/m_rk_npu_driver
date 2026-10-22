"""head(INT8, 双输出) + numpy DFL 尾部 → 最终 (84,8400)"""
import numpy as np
from rknnlite.api import RKNNLite
anchor=np.load("/home/orangepi/y11/anchor640.npy"); stride=np.load("/home/orangepi/y11/stride640.npy")
AR=np.arange(16,dtype=np.float32).reshape(1,16,1,1)
r=RKNNLite(verbose=False); r.load_rknn("/home/orangepi/y11/yolo11n_640_head_real.rknn")
r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
z=np.load("/home/orangepi/y11/in_u8.npz")
keys=sorted(z.files,key=lambda s:int(s.replace("arr_","")))
outs=[]
for k in keys:
    a=r.inference(inputs=[z[k][None]])
    lg=cl=None
    for t in a:
        t=np.asarray(t)
        if t.shape[1]==64: lg=t
        elif t.shape[1]==80: cl=t
    N=lg.shape[2]
    p=lg.reshape(1,4,16,N).transpose(0,3,1,2)
    e=np.exp(p-p.max(-1,keepdims=True)); pr=e/e.sum(-1,keepdims=True)
    d=(pr.transpose(0,3,2,1)*AR).sum(1)
    x1=anchor-d[:,0:2]; x2=anchor+d[:,2:4]
    box=np.concatenate([(x1+x2)*0.5, x2-x1],1)*stride
    outs.append(np.concatenate([box,cl],1)[0])
r.release()
np.savez_compressed("/home/orangepi/y11/out_head.npz", *outs)
print("完成", len(outs), np.shape(outs[0]))
