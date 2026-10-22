"""用编译器的 IR 表（verbose=True）抓 1280 的每算子成本剖面"""
import os, sys
import numpy as np, cv2
from rknn.api import RKNN

onnx_p, S = sys.argv[1], int(sys.argv[2])
cal = os.path.join("/home/dunaandone/rknn-work/yolo11", f"calib_{S}")
os.makedirs(cal, exist_ok=True)
rng = np.random.default_rng(7); paths = []
for i in range(4):
    img = rng.integers(0, 256, (S, S, 3), dtype=np.uint8)
    p = os.path.join(cal, f"c{i}.png"); cv2.imwrite(p, img); paths.append(p)
txt = f"/home/dunaandone/rknn-work/yolo11/prof_{S}.txt"
open(txt, "w").write("\n".join(os.path.join(f"calib_{S}", os.path.basename(p)) for p in paths) + "\n")

r = RKNN(verbose=True)
r.config(target_platform="rk3588", mean_values=[[0,0,0]], std_values=[[255,255,255]],
         quantized_dtype="w8a8", optimization_level=3)
r.load_onnx(model=onnx_p)
r.build(do_quantization=True, dataset=txt)
r.export_rknn(f"/home/dunaandone/rknn-work/yolo11/prof_{S}.rknn")
r.release()
print("PROFILE_DONE")
