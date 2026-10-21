import os, sys
import numpy as np, cv2
from rknn.api import RKNN

onnx_p, rknn_p, S = sys.argv[1], sys.argv[2], int(sys.argv[3])
cal = os.path.join(os.path.dirname(rknn_p), f"calib_{S}")
os.makedirs(cal, exist_ok=True)
rng = np.random.default_rng(7)
paths = []
for i in range(8):
    img = rng.integers(0, 256, (S, S, 3), dtype=np.uint8)
    p = os.path.join(cal, f"c{i}.png"); cv2.imwrite(p, img); paths.append(p)
txt = os.path.join(os.path.dirname(rknn_p), f"dataset_{S}.txt"); open(txt, "w").write("\n".join(os.path.join(os.path.basename(cal), os.path.basename(p)) for p in paths) + "\n")

r = RKNN(verbose=False)
assert r.config(target_platform="rk3588", mean_values=[[0,0,0]], std_values=[[255,255,255]],
                quantized_dtype="w8a8", optimization_level=3) == 0
assert r.load_onnx(model=onnx_p) == 0, "load_onnx failed"
assert r.build(do_quantization=True, dataset=txt) == 0, "build failed"
assert r.export_rknn(rknn_p) == 0, "export failed"
r.release()
print("OK", rknn_p, os.path.getsize(rknn_p)/1048576, "MB")
