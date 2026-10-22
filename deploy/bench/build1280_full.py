"""1280 双输出 INT8 模型：出口径照 docs/152，避开 Concat 共享量化 scale"""
import os, glob
import numpy as np, cv2, onnx
from rknn.api import RKNN

# 1) 导出 1280 ONNX
if not os.path.exists("yolo11n_1280.onnx") or os.path.getsize("yolo11n_1280.onnx") < 1e7:
    from ultralytics import YOLO
    YOLO("yolo11n.pt").export(format="onnx", imgsz=1280, opset=12, simplify=False, dynamic=False)
    os.replace("yolo11n.onnx", "yolo11n_1280.onnx")
print("onnx", os.path.getsize("yolo11n_1280.onnx")/1048576, "MB")

# 2) 切出双输出 head
onnx.utils.extract_model("yolo11n_1280.onnx", "yolo11n_1280_head.onnx", ["images"],
                         ["/model.23/Concat_output_0", "/model.23/Sigmoid_output_0"])
print("head onnx ok")

# 3) 标定图 letterbox 到 1280
SRC="/home/dunaandone/rknn-work/calib/real"; OUT="/home/dunaandone/rknn-work/calib/calib1280"
os.makedirs(OUT, exist_ok=True); paths=[]
for p in sorted(glob.glob(os.path.join(SRC,"*.jpg"))):
    im=cv2.imread(p)
    if im is None: continue
    h,w=im.shape[:2]; r=min(1280/h,1280/w)
    nh,nw=int(round(h*r)),int(round(w*r))
    rz=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)
    c=np.full((1280,1280,3),114,np.uint8); t,l=(1280-nh)//2,(1280-nw)//2
    c[t:t+nh,l:l+nw]=rz
    o=os.path.join(OUT, os.path.basename(p).replace(".jpg",".png")); cv2.imwrite(o,c); paths.append(o)
txt="/home/dunaandone/rknn-work/calib/dataset1280.txt"
open(txt,"w").write("\n".join(paths)+"\n")
print("标定图", len(paths))

# 4) 双输出 INT8
r=RKNN(verbose=False)
r.config(target_platform="rk3588", mean_values=[[0,0,0]], std_values=[[255,255,255]],
         quantized_dtype="w8a8", optimization_level=3)
r.load_onnx(model="yolo11n_1280_head.onnx")
r.build(do_quantization=True, dataset=txt)
r.export_rknn("yolo11n_1280_head_real.rknn"); r.release()
print("BUILD_OK", os.path.getsize("yolo11n_1280_head_real.rknn")/1048576, "MB")

# 5) 抽 anchor / stride 给 CPU 尾部
from onnx import numpy_helper
m=onnx.load("yolo11n_1280.onnx"); g=m.graph
prod={o:n for n in g.node for o in n.output}
def cval(t):
    n=prod.get(t)
    for a in n.attribute:
        if a.name=="value": return numpy_helper.to_array(a.t)
np.save("anchor1280.npy", cval("/model.23/Constant_12_output_0"))
np.save("stride1280.npy", cval("/model.23/Constant_15_output_0"))
print("anchor/stride saved")
