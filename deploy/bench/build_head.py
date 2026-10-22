import os
import numpy as np, cv2
from rknn.api import RKNN
txt="/home/dunaandone/rknn-work/yolo11/ds_head.txt"
open(txt,"w").write("\n".join(f"calib_640/c{i}.png" for i in range(8))+"\n")
r=RKNN(verbose=True)
r.config(target_platform="rk3588", mean_values=[[0,0,0]], std_values=[[255,255,255]],
         quantized_dtype="w8a8", optimization_level=3)
r.load_onnx(model="yolo11n_640_head.onnx")
r.build(do_quantization=True, dataset=txt)
out="yolo11n_640_head.rknn"
r.export_rknn(out); r.release()
print("BUILD_OK", os.path.getsize(out)/1048576, "MB")
