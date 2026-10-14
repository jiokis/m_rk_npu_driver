#!/usr/bin/env python3
"""unet_to_rknn.py — SD 的 UNet → RKNN（**整条链路最大的风险点**）

## 为什么风险最大

SD 的 UNet 每层都有 **self-attention 与 cross-attention**（`Attention`：
`to_q/to_k/to_v → scaled_dot_product_attention → to_out`），
外加 `LayerNormalization`、`GELU`、`Upsample`、`Concat`、`GroupNorm`。
**这些在 RK3588 NPU 上的支持与精度都要实测。**

## 已知要处理的

* attention ⇒ `disable_rules=['fuse_mul_into_sdpa']`（否则融合规则报 TypeError）
* 权重在**外部数据文件** `.onnx.data` 里（torch 新版导出器的默认行为）⇒ 要和 `.onnx` 放一起
* 标定集：三输入，`dataset.txt` 每行要给**三个**文件

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/sd/unet_to_rknn.py --latent 32 --no-quant     # 先验支持性
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = "/home/dunaandone/rknn-work/sdcalib_unet"   # ⚠️ 不能含空格


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--calib-n", type=int, default=4)
    ap.add_argument("--opt-level", type=int, default=3,
                    help="RKNN optimization_level。**本机只有 7 GB 内存**，"
                         "865 M 参数的 UNet 在 level=3 下会被 OOM kill（实测 anon-rss 6.7 GB）"
                         "⇒ 降到 0 可显著降低峰值")
    a = ap.parse_args()
    name = a.name or f"sdunet_l{a.latent}"
    onnx_p = os.path.join(HERE, "rknn", f"sdunet_l{a.latent}_{a.dtype}.onnx")
    rknn_p = os.path.join(HERE, "rknn", name + ".rknn")
    if not os.path.exists(onnx_p):
        raise SystemExit(f"缺 {onnx_p}（先跑 export_unet_gpu.py）")
    print(f"[in] {onnx_p}  {os.path.getsize(onnx_p)/1e6:.1f} MB"
          f" + data {os.path.getsize(onnx_p + '.data')/1e6:.1f} MB"
          if os.path.exists(onnx_p + ".data") else f"[in] {onnx_p}")

    cal = None
    if not a.no_quant:
        d = os.path.join(CALIB, name)
        os.makedirs(d, exist_ok=True)
        rng = np.random.RandomState(0)
        lines = []
        for i in range(a.calib_n):
            s = os.path.join(d, f"s_{i:02d}.npy")
            t = os.path.join(d, f"t_{i:02d}.npy")
            e = os.path.join(d, f"e_{i:02d}.npy")
            np.save(s, (rng.randn(1, 4, a.latent, a.latent) * 2.0).astype(np.float32))
            np.save(t, np.array([500.0], np.float32))
            np.save(e, (rng.randn(1, 77, 1024) * 0.5).astype(np.float32))
            lines.append(f"{s} {t} {e}")
        cal = os.path.join(d, "dataset.txt")
        open(cal, "w").write("\n".join(lines) + "\n")
        print(f"[calib] {a.calib_n} 个样本（三输入，每行三个文件）")

    from rknn.api import RKNN
    r = RKNN(verbose=False)
    r.config(target_platform="rk3588", optimization_level=a.opt_level, quantized_dtype="w8a8",
             disable_rules=["fuse_mul_into_sdpa"])
    ret = r.load_onnx(model=onnx_p)
    print(f"[rknn] load_onnx ret={ret}")
    if ret != 0:
        r.release(); return 1
    ret = r.build(do_quantization=cal is not None, dataset=cal)
    print(f"[rknn] build ret={ret}")
    if ret == 0:
        r.export_rknn(rknn_p)
        print(f"[rknn] ✅ {rknn_p}  {os.path.getsize(rknn_p)/1e6:.1f} MB")
    r.release()
    return 0 if ret == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
