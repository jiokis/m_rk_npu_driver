#!/usr/bin/env python3
"""export_unet.py — 从 SD-Turbo 单文件权重里导出 **UNet** 到 ONNX（RKNN 静态形状）

## 输入设计（RKNN 只吃静态形状）

| 输入 | 形状 | 说明 |
|---|---|---|
| `sample` | `[1,4,L,L]` | 潜空间（L = 图/8） |
| `timestep` | `[1]` | 时间步（SD-Turbo 只用 1~4 步） |
| `encoder_hidden_states` | `[1,77,768]` | CLIP 文本嵌入（**可以预先算好固定住**） |

## 主要风险：**attention**

SD 的 UNet 每层都有 **self-attention** 与 **cross-attention**（`Attention` 模块：
`to_q/to_k/to_v → scaled_dot_product_attention → to_out`）。
RKNN 对 `Softmax`/`MatMul`/`SDPA` 的支持与精度都要实测 —— 这是本轮的核心验证点。

## 内存（本机只有 7 GB）

safetensors 有 5.2 GB，**绝不能整文件读入**。这里用 `safe_open` **只取 `model.*`**
（UNet 那部分，860 M 参数 ≈ 3.4 GB fp16）—— 仍然偏大，所以**边读边转 fp32 并立刻存盘**。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

CKPT = "/home/dunaandone/rknn-work/sd/sd_turbo.safetensors"
UNET_PT = "/home/dunaandone/rknn-work/sd/unet_sd.pt"


def load_unet():
    """懒加载 UNet（只读 `model.*`）并做 CompVis → diffusers 的名字转换"""
    from safetensors import safe_open
    from diffusers import UNet2DConditionModel
    from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint

    if os.path.exists(UNET_PT):
        print(f"[load] 复用已存盘的 {UNET_PT}")
        ck = torch.load(UNET_PT, map_location="cpu", weights_only=False)
        u = UNet2DConditionModel.from_config(ck["config"])
        u.load_state_dict(ck["state_dict"])
        return u.eval()

    cfg = dict(sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
               block_out_channels=[320, 640, 1280, 1280],
               down_block_types=["CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                                 "CrossAttnDownBlock2D", "DownBlock2D"],
               up_block_types=["UpBlock2D", "CrossAttnUpBlock2D",
                               "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
               cross_attention_dim=1024, attention_head_dim=8,
               norm_num_groups=32)
    sd = {}
    with safe_open(CKPT, framework="pt") as f:
        for k in f.keys():
            if k.startswith("model."):
                sd[k[len("model."):]] = f.get_tensor(k).to(torch.float32)
    print(f"[load] UNet 张量 {len(sd)} 个  {sum(v.numel() for v in sd.values())/1e6:.1f} M 参数")
    conv = convert_ldm_unet_checkpoint(sd, cfg, path=None, extract_ema=False)
    u = UNet2DConditionModel.from_config(cfg)
    missing, unexpected = u.load_state_dict(conv, strict=False)
    print(f"[load] missing {len(missing)}  unexpected {len(unexpected)}")
    if missing[:3]: print("   missing 例:", missing[:3])
    if unexpected[:3]: print("   unexpected 例:", unexpected[:3])
    torch.save({"state_dict": u.state_dict(), "config": cfg}, UNET_PT)
    print(f"[load] 已存盘 {UNET_PT}")
    return u.eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"deploy/sd/rknn/sdunet_l{a.latent}.onnx"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    u = load_unet()
    n = sum(p.numel() for p in u.parameters())
    print(f"[unet] 参数 {n/1e6:.1f} M")

    class W(torch.nn.Module):
        def __init__(self, u):
            super().__init__()
            self.u = u
        def forward(self, sample, timestep, ehs):
            return self.u(sample, timestep, encoder_hidden_states=ehs).sample

    m = W(u).eval()
    s = torch.zeros(1, 4, a.latent, a.latent)
    t = torch.zeros(1)
    e = torch.zeros(1, 77, 1024)
    with torch.no_grad():
        y = m(s, t, e)
    print(f"[shape] {tuple(s.shape)} + t{tuple(t.shape)} + ehs{tuple(e.shape)} → {tuple(y.shape)}")
    torch.onnx.export(m, (s, t, e), out, input_names=["sample", "timestep", "ehs"],
                      output_names=["out"], opset_version=14, do_constant_folding=True)
    print(f"[onnx] {out}  {os.path.getsize(out)/1e6:.1f} MB")
    json.dump({"latent": a.latent, "params": n}, open(out + ".json", "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
