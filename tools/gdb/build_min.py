#!/usr/bin/env python3
"""build_min.py — 最小 ONNX → RKNN 构建，**专供 gdb 观察编译流水线用**

## 为什么单独写一个

gdb 要 `run` 一个程序，那个程序必须**只做一次编译**、**尽量短**、
**不依赖命令行参数**（gdb 的 `run` 传参很别扭）。
所以这里把 `deploy/convert_rknn.py` 的构建流程抽成一个固定入口。

## 为什么默认用 ab_dw.onnx

docs/99 的 `hardware_rules` 里 `tiling_depthwise_conv` 的 `ops = ['Conv']`；
`ab_dw.onnx` 是单 depthwise conv 模型 —— **最小、且必然走 tiling 规则**。

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V tools/gdb/build_min.py                       # 默认 ab_dw.onnx
  $V tools/gdb/build_min.py --onnx deploy/onnx/tiny_cnn.onnx

## 环境变量
  BUILD_MIN_ONNX    覆盖 ONNX 路径（gdb 脚本里不好传 argv，用这个）
  BUILD_MIN_NOQUANT 非空则不做量化

## 两条踩过的接口约定（合成模型必读）
  ① 工具链把 ONNX 输入当 **NCHW** 读 ⇒ 造模型用 [1,C,H,W]（用 NHWC 会被读成 H）
  ② `mean_values`/`std_values` 只在**输入是 3 通道**时才能给：
     - 非 3 通道给了 ⇒ "The channel of r_shape [...] must be 3!"
     - 而且**标定图是 3 通道** ⇒ 非 3 通道模型必须配 `BUILD_MIN_NOQUANT=1`
"""
from __future__ import annotations

import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ONNX = os.path.join(REPO, "deploy", "onnx", "ab_dw.onnx")
CAL = "/home/dunaandone/rknn-work/calib"
# ⚠️ 标定集路径不能含空格：工具链按空白切分 dataset.txt（踩过）
CAL_TXT = os.path.join(CAL, "dataset.txt")


def main() -> int:
    onnx = os.environ.get("BUILD_MIN_ONNX") or DEFAULT_ONNX
    if not os.path.isfile(onnx):
        print(f"[build_min] ONNX 不存在: {onnx}", file=sys.stderr)
        return 2
    quant = not os.environ.get("BUILD_MIN_NOQUANT")

    from rknn.api import RKNN

    # ⚠️ mean_values/std_values 的长度必须等于输入的**通道数**，
    #    否则 load_onnx 直接报 "The len of mean_values ([0,0,0]) is wrong, expect 112!"
    #    （踩过一次：合成单 conv 模型输入是 64/112 通道，不是 3）
    nch = 3
    try:
        # ⚠️ 不能写 `import onnx` —— 那会把**路径变量** onnx 覆盖成模块（踩过一次，
        #    报错是 "stat: path should be string, ... not module"）
        import onnx as _onnx_mod
        m = _onnx_mod.load(onnx)
        dims = m.graph.input[0].type.tensor_type.shape.dim
        # ⚠️ 工具链把 ONNX 输入当 **NCHW** 读 ⇒ 通道是 dims[1]（不是 dims[3]）
        if len(dims) == 4 and dims[1].dim_value:
            nch = dims[1].dim_value
    except Exception:
        pass

    # ⚠️ mean_values/std_values 只有**输入是 3 通道**时才能给：
    #    给非 3 通道会报 "The channel of r_shape [...] must be 3!"（踩过一次）
    cfg = dict(
        target_platform="rk3588",
        quantized_dtype="w8a8",
        quantized_algorithm="normal",
        optimization_level=3,
    )
    if nch == 3:
        cfg["mean_values"] = [[0, 0, 0]]
        cfg["std_values"] = [[255, 255, 255]]
    r = RKNN(verbose=False)
    ret = r.config(**cfg)
    if ret != 0:
        print(f"[build_min] config ret={ret}", file=sys.stderr)
        return 3
    ret = r.load_onnx(model=onnx)
    if ret != 0:
        print(f"[build_min] load_onnx ret={ret}", file=sys.stderr)
        return 4
    t0 = time.time()
    ret = r.build(do_quantization=quant, dataset=CAL_TXT if quant else None)
    if ret != 0:
        print(f"[build_min] build ret={ret}", file=sys.stderr)
        return 5
    print(f"[build_min] build OK  {time.time() - t0:.2f}s  onnx={os.path.basename(onnx)}")
    r.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
