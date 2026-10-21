#!/usr/bin/env python3
"""make_conv_sweep.py — 生成「单变量 Conv 扫描」用于反解寄存器语义

## 为什么要这套东西

`docs/146` 之后，剩下的硬骨头是**形状/步长寄存器**：
CNA 的 48 个、DPU 的 51 个寄存器，大部分还只是「常量 vs 数值字段」的粗分类。

要定语义，唯一靠得住的办法是**单变量对照**：
固定其他一切，只改一个参数，看哪个寄存器的值跟着变。

所以这里生成一组「只有一个 Conv」的模型，每个只差一个参数：

    h32/h64/h128/h224  改输入空间尺寸
    k1/k3/k5/k7        改卷积核
    s1/s2/s4           改步长

## 用法
  /home/dunaandone/rknn-venv/bin/python deploy/make_conv_sweep.py --outdir /path/to/sweep
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# (名字, H, W, Cin, Cout, k, s, p)
SPECS = [
    ("h32",  32,  32,  3, 3, 3, 1, 1),
    ("h64",  64,  64,  3, 3, 3, 1, 1),   # 基准
    ("h128", 128, 128, 3, 3, 3, 1, 1),
    ("h224", 224, 224, 3, 3, 3, 1, 1),
    ("k1",   64,  64,  3, 3, 1, 1, 0),
    ("k5",   64,  64,  3, 3, 5, 1, 2),
    ("k7",   64,  64,  3, 3, 7, 1, 3),
    ("s2",   64,  64,  3, 3, 3, 2, 1),
    ("s4",   64,  64,  3, 3, 3, 4, 1),
]


def build_conv(path, H, W, Cin, Cout, k, s, p, seed=0):
    rng = np.random.default_rng(seed)
    ho = (H + 2 * p - k) // s + 1
    wo = (W + 2 * p - k) // s + 1
    w = (rng.standard_normal((Cout, Cin, k, k)) * 0.1).astype(np.float32)
    inits = [numpy_helper.from_array(w, "W")]
    nodes = [helper.make_node("Conv", ["x", "W"], ["y"],
                              kernel_shape=[k, k], strides=[s, s],
                              pads=[p, p, p, p], name="conv")]
    graph = helper.make_graph(
        nodes, "conv_single",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, Cin, H, W])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, Cout, ho, wo])],
        inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return ho, wo


def make_calib(workdir, H, W, n=8, seed=7):
    d = os.path.join(workdir, "calib")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
        p = os.path.join(d, f"cal_{i:02d}.png")
        try:
            import cv2
            cv2.imwrite(p, img)
        except Exception:
            from PIL import Image
            Image.fromarray(img).save(p)
        paths.append(p)
    txt = os.path.join(workdir, "dataset.txt")
    with open(txt, "w") as f:
        f.write("\n".join(paths) + "\n")
    return txt


def convert(onnx_path, rknn_path, H, W):
    from rknn.api import RKNN
    r = RKNN(verbose=False)
    assert r.config(target_platform="rk3588", mean_values=[[0, 0, 0]],
                    std_values=[[255, 255, 255]], quantized_dtype="w8a8") == 0
    assert r.load_onnx(model=onnx_path) == 0
    cal = make_calib(os.path.dirname(rknn_path), H, W)
    assert r.build(do_quantization=True, dataset=cal) == 0
    assert r.export_rknn(rknn_path) == 0
    r.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--skip-build", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for name, H, W, Cin, Cout, k, s, p in SPECS:
        op = os.path.join(a.outdir, f"{name}.onnx")
        rp = os.path.join(a.outdir, f"{name}.rknn")
        ho, wo = build_conv(op, H, W, Cin, Cout, k, s, p)
        print(f"[{name}] onnx: {H}x{W} k{k} s{s} p{p} -> {ho}x{wo}", flush=True)
        if not a.skip_build:
            convert(op, rp, H, W)
            sz = os.path.getsize(rp)
            print(f"[{name}] rknn: {sz/1024:.0f} KB", flush=True)
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
