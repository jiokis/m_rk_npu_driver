import os, sys
from rknn.api import RKNN
src, out, sparse = sys.argv[1], sys.argv[2], sys.argv[3].lower()=="true"
txt="/home/dunaandone/rknn-work/yolo11/ds_sp.txt"
open(txt,"w").write("\n".join(f"calib_640/c{i}.png" for i in range(8))+"\n")
r=RKNN(verbose=True)
r.config(target_platform="rk3588", mean_values=[[0,0,0]], std_values=[[255,255,255]],
         quantized_dtype="w8a8", optimization_level=3, sparse_infer=sparse)
r.load_onnx(model=src)
r.build(do_quantization=True, dataset=txt)
r.export_rknn(out); r.release()
print("BUILD_OK", out, os.path.getsize(out)/1048576, "MB sparse=", sparse)
