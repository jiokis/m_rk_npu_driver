"""用真实 COCO 图做标定集，重建 640 模型"""
import os, glob
import numpy as np, cv2

SRC = "/home/dunaandone/rknn-work/calib/real"
OUT = "/home/dunaandone/rknn-work/calib/calib640"
os.makedirs(OUT, exist_ok=True)
paths = []
for p in sorted(glob.glob(os.path.join(SRC, "*.jpg"))):
    im = cv2.imread(p)
    if im is None: continue
    h, w = im.shape[:2]
    r = min(640/h, 640/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    rz = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    c = np.full((640, 640, 3), 114, np.uint8)
    t, l = (640-nh)//2, (640-nw)//2
    c[t:t+nh, l:l+nw] = rz
    o = os.path.join(OUT, os.path.basename(p).replace(".jpg", ".png"))
    cv2.imwrite(o, c); paths.append(o)
print(f"标定图 {len(paths)} 张 → {OUT}")
txt = "/home/dunaandone/rknn-work/calib/dataset640.txt"
open(txt, "w").write("\n".join(paths) + "\n")

from rknn.api import RKNN
r = RKNN(verbose=False)
r.config(target_platform="rk3588", mean_values=[[0,0,0]], std_values=[[255,255,255]],
         quantized_dtype="w8a8", optimization_level=3)
r.load_onnx(model="yolo11n_640.onnx")
r.build(do_quantization=True, dataset=txt)
r.export_rknn("yolo11n_640_real.rknn")
r.release()
print("BUILD_OK", os.path.getsize("yolo11n_640_real.rknn")/1048576, "MB")
