#!/usr/bin/env python3
"""check_numeric.py — 用**同一个输入**跑一次 UNet，比对 torch 与 RKNN 的输出

这是部署里最该先做的一步：**先把单次前向对齐，再谈采样**。
采样循环会把任何前向误差放大 200 步，出问题时分不清是模型、量化还是循环。

用法
  笔记本：$V deploy/diffusion/check_numeric.py --backend torch --ckpt <ckpt> --in /tmp/probe.npz --out /tmp/probe_torch.npy
  板子  ：python3 check_numeric.py --backend rknn --model x.rknn --meta x.json --in /tmp/probe.npz --out /tmp/probe_rknn.npy
  然后  ：$V deploy/diffusion/check_numeric.py --compare /tmp/probe_torch.npy /tmp/probe_rknn.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "rknn"])
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--ckpt"); ap.add_argument("--model"); ap.add_argument("--meta")
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--make-probe", default=None, help="生成一个测试输入 .npz 后退出")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.make_probe:
        rng = np.random.RandomState(args.seed)
        x = rng.randn(1, 1, 32, 32).astype(np.float32) * 0.8
        t = np.array([500], dtype=np.int64)
        import math
        half = 64
        f = np.exp(-math.log(10000) * np.arange(half, dtype=np.float32) / half)
        a = t.astype(np.float32)[:, None] * f[None, :]
        tb = np.concatenate([np.sin(a), np.cos(a)], 1)[:, :, None, None].astype(np.float32)
        np.savez(args.make_probe, x=x, temb=tb)
        print(f"[probe] {args.make_probe}  x{x.shape} temb{tb.shape}")
        return 0

    if args.compare:
        a, b = np.load(args.compare[0]), np.load(args.compare[1])
        d = a - b
        print(f"torch: 均值 {a.mean():+.4f} 标准差 {a.std():.4f} min {a.min():+.3f} max {a.max():+.3f}")
        print(f"rknn : 均值 {b.mean():+.4f} 标准差 {b.std():.4f} min {b.min():+.3f} max {b.max():+.3f}")
        print(f"差值 : 均值 {d.mean():+.4f} 标准差 {d.std():.4f} |max| {np.abs(d).max():.4f}")
        cos = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"余弦相似度 {cos:.4f}   相对误差 {np.abs(d).mean()/(np.abs(a).mean()+1e-12):.2%}")
        return 0

    z = np.load(args.inp)
    x, tb = z["x"], z["temb"]
    if args.backend == "torch":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from sample import TorchBackend
        be = TorchBackend(args.ckpt)
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from sample import RknnBackend
        be = RknnBackend(args.model)
    y = be(x, tb)
    np.save(args.out, y)
    print(f"[{args.backend}] 输出 均值 {y.mean():+.4f} 标准差 {y.std():.4f} "
          f"min {y.min():+.3f} max {y.max():+.3f}  → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
