import os
from rknn.api import RKNN
txt="/home/dunaandone/rknn-work/calib/dataset640.txt"
r=RKNN(verbose=False)
r.config(target_platform="rk3588", mean_values=[[0,0,0]], std_values=[[255,255,255]],
         quantized_dtype="w8a8", optimization_level=3)
r.load_onnx(model="yolo11n_640_head.onnx")
r.build(do_quantization=True, dataset=txt)
r.export_rknn("yolo11n_640_head_real.rknn"); r.release()
print("BUILD_OK", os.path.getsize("yolo11n_640_head_real.rknn")/1048576, "MB")
