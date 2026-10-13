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
  BUILD_MIN_NOQUANT 非空则不做量化（更快，但 tiling 行为可能不同）
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

    r = RKNN(verbose=False)
    ret = r.config(
        target_platform="rk3588",
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        quantized_dtype="w8a8",
        quantized_algorithm="normal",
        optimization_level=3,
    )
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
