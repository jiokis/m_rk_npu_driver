"""op_target 实验：把指定的 ONNX 算子放到 CPU"""
import os, sys, json
import numpy as np
from rknn.api import RKNN
name = sys.argv[1]; targets = json.loads(sys.argv[2])
cfg = dict(target_platform="rk3588", mean_values=[[0,0,0]], std_values=[[255,255,255]],
           quantized_dtype="w8a8", optimization_level=3, op_target=targets)
txt="/home/dunaandone/rknn-work/yolo11/ds_t.txt"
open(txt,"w").write("\n".join(f"calib_640/c{i}.png" for i in range(8))+"\n")
r=RKNN(verbose=True); r.config(**cfg)
r.load_onnx(model="yolo11n_640.onnx")
r.build(do_quantization=True, dataset=txt)
out=f"/home/dunaandone/rknn-work/yolo11/{name}.rknn"
r.export_rknn(out); r.release()
print(f"BUILD_OK {name} {os.path.getsize(out)/1048576:.2f}MB targets={targets}")
