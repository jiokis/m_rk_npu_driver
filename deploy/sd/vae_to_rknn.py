#!/usr/bin/env python3
"""vae_to_rknn.py — SD 的 VAE 解码器 → ONNX → RKNN

## 关键点

* **只导解码器**（生成只需要「潜空间 → 图像」）
* SD 的 `scaling_factor = 0.18215`：潜空间要先除以它再解码（**烘焙进图里**，省一次 CPU 运算）
* 潜空间边长 = 图像边长 / 8（SD 的 8× 下采样）

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/sd/vae_to_rknn.py --latent 64 --name sd15_vae512
  $V deploy/sd/vae_to_rknn.py --latent 32 --name sd15_vae256 --no-quant   # 快速试
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

VAE_PT = "/home/dunaandone/rknn-work/sd/vae_sd.pt"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rknn")
CALIB = "/home/dunaandone/rknn-work/sdcalib"     # ⚠️ 不能含空格


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--calib-n", type=int, default=8)
    a = ap.parse_args()
    name = a.name or f"sdvae_l{a.latent}"
    os.makedirs(OUTDIR, exist_ok=True)

    from diffusers import AutoencoderKL
    ck = torch.load(VAE_PT, map_location="cpu", weights_only=False)
    vae = AutoencoderKL.from_config(ck["config"])
    vae.load_state_dict(ck["state_dict"])
    vae.eval()
    sf = float(vae.config.scaling_factor)

    class Dec(torch.nn.Module):
        """z_scaled → image；把 1/scaling_factor 烘焙进图里"""
        def __init__(self, v, sf):
            super().__init__()
            self.post = v.post_quant_conv
            self.dec = v.decoder
            self.sf = sf
        def forward(self, z):
            return self.dec(self.post(z / self.sf))

    m = Dec(vae, sf).eval()
    nparam = sum(p.numel() for p in m.parameters())
    print(f"[vae] 解码器参数 {nparam/1e6:.2f} M  scaling_factor={sf}")
    z = torch.zeros(1, 4, a.latent, a.latent)
    with torch.no_grad():
        y = m(z)
    print(f"[shape] 潜空间 {tuple(z.shape)} → 图像 {tuple(y.shape)}")
    onnx_p = os.path.join(OUTDIR, name + ".onnx")
    torch.onnx.export(m, (z,), onnx_p, input_names=["z"], output_names=["img"],
                      opset_version=14, do_constant_folding=True)
    print(f"[onnx] {onnx_p}  {os.path.getsize(onnx_p)/1e6:.1f} MB")

    cal = None
    if not a.no_quant:
        d = os.path.join(CALIB, name)
        os.makedirs(d, exist_ok=True)
        rng = np.random.RandomState(0)
        lines = []
        for i in range(a.calib_n):
            p = os.path.join(d, f"z_{i:02d}.npy")
            np.save(p, (rng.randn(1, 4, a.latent, a.latent) * 0.6).astype(np.float32))
            lines.append(p)
        cal = os.path.join(d, "dataset.txt")
        open(cal, "w").write("\n".join(lines) + "\n")

    from rknn.api import RKNN
    r = RKNN(verbose=False)
    # ⚠️ SD 的 VAE 里有**自注意力**（mid.attn_1）。RKNN 的 `fuse_mul_into_sdpa` 规则
    #    在含 attention 的图上会报
    #    `TypeError: only 0-dimensional arrays can be converted to Python scalars`，
    #    且工具链自己提示 `You can add disable_rules=['fuse_mul_into_sdpa']`（踩过）。
    r.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8",
             disable_rules=["fuse_mul_into_sdpa"])
    assert r.load_onnx(model=onnx_p) == 0
    ret = r.build(do_quantization=cal is not None, dataset=cal)
    rk = os.path.join(OUTDIR, name + ".rknn")
    if ret == 0:
        r.export_rknn(rk)
        print(f"[rknn] ✅ {rk}  {os.path.getsize(rk)/1e6:.1f} MB")
    else:
        print(f"[rknn] ❌ build ret={ret}")
    r.release()
    return 0 if ret == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
