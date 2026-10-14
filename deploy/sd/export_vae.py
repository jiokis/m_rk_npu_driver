#!/usr/bin/env python3
"""export_vae.py — 从 SD-Turbo 单文件权重里导出 **VAE 解码器** 到 ONNX

## 为什么只导解码器

生成时只需要「潜空间 → 图像」，编码器（图像 → 潜空间）只有图生图才用。
⇒ 部署到 NPU 的只有解码器。

## SD 的 VAE 解码器结构（会遇到的算子）

* `Conv2d` + `SiLU` —— RKNN 友好（`exSwish` 可融进 Conv）
* **`GroupNorm`** —— 会变成 `exNorm` + 2 个 `Reshape`（实测，见 docs/131/140）
* **`Attention`**（`mid.attn_1`，自注意力 + softmax）—— **RKNN 支持存疑，这是主要风险**
* `Upsample(nearest)` + `Conv2d`

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/sd/export_vae.py --out deploy/sd/vae_dec.onnx
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/dunaandone/rknn-work/sd/sd_turbo.safetensors")
    ap.add_argument("--out", default="deploy/sd/vae_dec.onnx")
    ap.add_argument("--latent", type=int, default=64, help="潜空间边长（512/8=64）")
    a = ap.parse_args()

    from diffusers import AutoencoderKL
    print("[load] 正在从单文件加载 VAE …", flush=True)
    vae = AutoencoderKL.from_single_file(a.ckpt)
    vae.eval()
    print(f"[load] 完成  latent_channels={vae.config.latent_channels} "
          f"block_out_channels={vae.config.block_out_channels} "
          f"scaling_factor={vae.config.scaling_factor}")

    class Dec(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.d = v.decoder
            self.post = v.post_quant_conv
        def forward(self, z):
            return self.d(self.post(z))

    m = Dec(vae).eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"[model] 解码器参数 {n/1e6:.2f} M")
    z = torch.zeros(1, 4, a.latent, a.latent)
    with torch.no_grad():
        y = m(z)
    print(f"[shape] 潜空间 {tuple(z.shape)} → 图像 {tuple(y.shape)}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.onnx.export(m, (z,), a.out, input_names=["z"], output_names=["img"],
                      opset_version=14, do_constant_folding=True)
    print(f"[onnx] {a.out}  {os.path.getsize(a.out)/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
