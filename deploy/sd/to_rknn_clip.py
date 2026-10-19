#!/usr/bin/env python3
"""to_rknn_clip.py — CLIP 文本塔 ONNX → RKNN（**用 rknn-venv 跑**）

⚠️ RKNN 工具链在 `rknn-venv`（torch 侧在 conda `sddiff`）——
   导出用 sddiff，转换用 rknn-venv，**两个环境不能混**（踩过：`No module named 'rknn.api'`）。
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "rknn")
CALIB = "/home/dunaandone/rknn-work/sdcalib_clip"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", action="store_true")
    ap.add_argument("--opt-level", type=int, default=3)
    a = ap.parse_args()
    onnx_p = os.path.join(OUTDIR, "clip_text.onnx")
    if not os.path.exists(onnx_p):
        raise SystemExit(f"缺 {onnx_p}")
    print(f"[in] {onnx_p}  {os.path.getsize(onnx_p)/1e6:.0f} MB")
    cal = None
    if a.quant:
        os.makedirs(CALIB, exist_ok=True)
        rng = np.random.RandomState(0); lines = []
        for i in range(8):
            p = os.path.join(CALIB, f"ids_{i:02d}.npy")
            v = np.zeros((1, 77), np.float32); n = int(rng.randint(4, 20))
            v[0, 0] = 49406; v[0, 1:1 + n] = rng.randint(100, 40000, n); v[0, 1 + n] = 49407
            np.save(p, v); lines.append(p)
        cal = os.path.join(CALIB, "dataset.txt"); open(cal, "w").write("\n".join(lines) + "\n")
        print(f"[calib] {len(lines)} 个样本")
    from rknn.api import RKNN
    r = RKNN(verbose=False)
    r.config(target_platform="rk3588", optimization_level=a.opt_level,
             quantized_dtype="w8a8", disable_rules=["fuse_mul_into_sdpa"])
    assert r.load_onnx(model=onnx_p) == 0
    ret = r.build(do_quantization=cal is not None, dataset=cal)
    print(f"[rknn] build ret={ret}")
    rk = os.path.join(OUTDIR, "clip_text.rknn")
    if ret == 0:
        r.export_rknn(rk)
        print(f"[rknn] ✅ {rk}  {os.path.getsize(rk)/1e6:.0f} MB")
    r.release()
    return 0 if ret == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
