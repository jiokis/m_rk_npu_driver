#!/usr/bin/env python3
"""C=16 k=3 卷积 @ 320/640/1280 —— 测「输入变大」的 NPU 标度"""
import os
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/home/dunaandone/rknn-work/sweep2"
os.makedirs(OUT, exist_ok=True)
C = 16

def build(path, S, k=3, p=1):
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((C, C, k, k)) * 0.05).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[k,k], strides=[1,1],
                          pads=[p,p,p,p], name="conv")],
        "conv_scale",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, S, S])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, C, S, S])],
        [numpy_helper.from_array(w, "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
    onnx.checker.check_model(m); onnx.save(m, path)

def convert(op, rp, S):
    from rknn.api import RKNN
    d = os.path.join(OUT, "calib"); os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(7); paths = []
    for i in range(8):
        a = rng.integers(0, 256, (1, C, S, S), dtype=np.uint8)   # ★ 标定必须是 NCHW
        pth = os.path.join(d, f"c16_{S}_{i}.npy"); np.save(pth, a); paths.append(pth)
    txt = os.path.join(OUT, f"dataset_{S}.txt")
    open(txt, "w").write("\n".join(paths) + "\n")
    r = RKNN(verbose=False)
    r.config(target_platform="rk3588", mean_values=[[0]*C], std_values=[[1]*C],
             quantized_dtype="w8a8")
    r.load_onnx(model=op); r.build(do_quantization=True, dataset=txt)
    r.export_rknn(rp); r.release()

for S in (320, 640, 1280):
    op = os.path.join(OUT, f"c16_{S}.onnx"); rp = os.path.join(OUT, f"c16_{S}.rknn")
    build(op, S); convert(op, rp, S)
    print(f"  {S}: {os.path.getsize(rp)/1024:.0f} KB", flush=True)
print("DONE")
