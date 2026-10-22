"""接缝问题验证：非重叠 2x2 vs 重叠分块，对比 native 1280 FP32"""
import numpy as np, cv2, onnxruntime as ort, glob

S640=640
def nms(b, iou=0.45, conf=0.25):
    if len(b)==0: return b
    out=[]
    for c in np.unique(b[:,5]):
        q=b[b[:,5]==c]
        xywh=np.concatenate([q[:,:2], q[:,2:4]-q[:,:2]],1)
        k=cv2.dnn.NMSBoxes(xywh.tolist(), q[:,4].tolist(), conf, iou)
        if len(k): out.append(q[np.asarray(k).ravel()])
    return np.concatenate(out,0) if out else b[:0]

def decode(o, conf=0.25):
    sc=o[4:].max(0); idx=np.flatnonzero(sc>conf)
    if idx.size==0: return np.zeros((0,6),np.float32)
    d=o[:4,idx].T   # cx,cy,w,h
    xyxy=np.stack([d[:,0]-d[:,2]/2, d[:,1]-d[:,3]/2, d[:,0]+d[:,2]/2, d[:,1]+d[:,3]/2],1)
    return np.concatenate([xyxy, sc[idx][:,None], o[4:,idx].argmax(0)[:,None].astype(np.float32)],1)

s1280=ort.InferenceSession("yolo11n_1280.onnx",providers=["CPUExecutionProvider"])
s640 =ort.InferenceSession("yolo11n_640.onnx", providers=["CPUExecutionProvider"])

def run_native(im):          # im: BGR 1280
    x=im[:,:,::-1].astype(np.float32).transpose(2,0,1)[None]/255.0
    return nms(decode(s1280.run(None,{"images":x})[0][0]))

def run_tiled(im, tile_px, stride_px):
    """tile_px: 每块在原图上的边长; stride_px: 网格步长"""
    H=W=im.shape[0]; allb=[]
    origins=[]
    ys=list(range(0, H-tile_px+1, stride_px)); xs=list(range(0, W-tile_px+1, stride_px))
    if ys[-1]+tile_px < H: ys.append(H-tile_px)
    if xs[-1]+tile_px < W: xs.append(W-tile_px)
    for y0 in ys:
        for x0 in xs:
            tl=im[y0:y0+tile_px, x0:x0+tile_px]
            rz=cv2.resize(tl,(S640,S640))
            x=rz[:,:,::-1].astype(np.float32).transpose(2,0,1)[None]/255.0
            b=decode(s640.run(None,{"images":x})[0][0])
            if len(b):
                k=tile_px/S640
                b[:,:4]=b[:,:4]*k
                b[:,[0,2]]+=x0; b[:,[1,3]]+=y0
                allb.append(b); origins.append((x0,y0))
    A=np.concatenate(allb,0) if allb else np.zeros((0,6),np.float32)
    return nms(A)

def iou1(a,b):
    iw=max(0,min(a[2],b[2])-max(a[0],b[0])); ih=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=iw*ih
    aa=(a[2]-a[0])*(a[3]-a[1]); bb=(b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+bb-inter+1e-9)

files=sorted(glob.glob("/home/dunaandone/rknn-work/calib/real/*.jpg"))
cfgs=[("非重叠 640/步640",640,640), ("重叠 768/步512",768,512), ("重叠 896/步384",896,384)]
res={c[0]:[0,0,0] for c in cfgs}   # [总数, 跨缝目标数, 跨缝命中数]
ref_total=0; seam_total=0
for f in files:
    im=cv2.imread(f); h,w=im.shape[:2]; r=min(1280/h,1280/w)
    rz=cv2.resize(im,(int(round(w*r)),int(round(h*r))))
    c=np.full((1280,1280,3),114,np.uint8); t,l=(1280-rz.shape[0])//2,(1280-rz.shape[1])//2
    c[t:t+rz.shape[0], l:l+rz.shape[1]]=rz
    ref=run_native(c)
    ref_total+=len(ref)
    # 跨缝目标：框跨过 x=640 或 y=640
    seam=[b for b in ref if (b[0]<640<b[2]) or (b[1]<640<b[3])]
    seam_total+=len(seam)
    for name,tp,st in cfgs:
        det=run_tiled(c,tp,st)
        res[name][0]+=len(det)
        res[name][1]+=len(seam)
        hit=sum(1 for s_ in seam if any(iou1(s_,d)>0.5 for d in det))
        res[name][2]+=hit
print(f"  参考(native 1280 FP32): {len(files)} 图，检出 {ref_total}，其中跨缝(跨 x=640/y=640) {seam_total}")
print(f"  {'方案':<20}{'检出总数':>9}{'跨缝命中':>10}{'命中率':>8}")
for name,tp,st in cfgs:
    tot,ns,hit=res[name]
    print(f"  {name:<20}{tot:>9}{hit:>10}/{ns:<5}{hit/max(ns,1)*100:>6.1f}%")
