#!/usr/bin/env python3
"""export_onnx.py — 把 TinyUNet 导出成 **RKNN 友好的 ONNX**

要点：
* opset 13、**静态形状**（batch=1、32×32、temb [1,D,1,1]）
* `do_constant_folding=True`
* 输入名固定为 `x` / `temb`，输出名 `eps`（采样脚本按名字喂数据）
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet import TinyUNet  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default="group", choices=["none", "batch", "group"])
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--temb-dim", type=int, default=128)
    ap.add_argument("--in-ch", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--weights", default=None, help="训练好的 .pt（不给就用随机权重，只用于转换测试）")
    args = ap.parse_args()

    out = args.out or os.path.join(HERE, "onnx", f"unet_{args.norm}_{args.size}.onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    m = TinyUNet(in_ch=args.in_ch, base=args.base, temb_dim=args.temb_dim, norm=args.norm).eval()
    if args.weights:
        m.load_state_dict(torch.load(args.weights, map_location="cpu"))
        print(f"[export] 载入权重 {args.weights}")

    x = torch.zeros(1, args.in_ch, args.size, args.size)
    temb = torch.zeros(1, args.temb_dim, 1, 1)
    with torch.no_grad():
        y = m(x, temb)
    torch.onnx.export(
        m, (x, temb), out,
        input_names=["x", "temb"], output_names=["eps"],
        opset_version=13, do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[export] {out}  参数 {sum(p.numel() for p in m.parameters())/1e6:.3f} M  输出 {tuple(y.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
