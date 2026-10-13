#!/usr/bin/env python3
"""synth_data.py — **本地程序化生成的彩色数据集**（不依赖任何下载）

## 为什么不用 CIFAR-10 / STL-10

实测所有可达的源都被限速到 **10~12 KB/s**（CIFAR-10 170 MB 要 ~4.7 小时，
STL-10 2.6 GB 更不可能），HuggingFace 与 hf-mirror **完全不可达**（0 字节）。
⇒ 与其等，不如**自己造一个结构清晰、可复现的彩色数据集**。

## 这个数据集长什么样

每张 32x32 RGB，含 1~3 个随机**形状**（圆 / 方 / 三角 / 环），
随机颜色、大小、位置、旋转，背景是随机**双色渐变**，再叠一点噪声。

**为什么这个设计适合验证「SD 架构」**：
* 结构清晰 ⇒ **一眼能看出模型有没有真的在生成**（形状是否成形、颜色是否合理）
* 多样性足够 ⇒ VAE 与扩散都必须真的学到东西（不是退化成复制）
* **完全确定性可复现**（固定种子），不需要任何外部依赖

## 用法
  python3 deploy/diffusion/synth_data.py --n 60000 --out /home/dunaandone/rknn-work/synth32.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def _disc(h, w, cx, cy, r):
    y, x = np.ogrid[:h, :w]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= r * r


def _square(h, w, cx, cy, r, ang):
    y, x = np.ogrid[:h, :w]
    dx, dy = x - cx, y - cy
    c, s = np.cos(-ang), np.sin(-ang)
    u, v = c * dx - s * dy, s * dx + c * dy
    return (np.abs(u) <= r) & (np.abs(v) <= r)


def _tri(h, w, cx, cy, r, ang):
    y, x = np.ogrid[:h, :w]
    dx, dy = x - cx, y - cy
    c, s = np.cos(-ang), np.sin(-ang)
    u, v = c * dx - s * dy, s * dx + c * dy
    # 等边三角形：三条半平面交集
    return (v >= -r) & (v <= r - np.sqrt(3) * np.abs(u)) & (np.sqrt(3) * np.abs(u) <= 2 * r)


def _ring(h, w, cx, cy, r, t):
    d = _disc(h, w, cx, cy, r) & ~_disc(h, w, cx, cy, max(r - t, 0.5))
    return d


def make(n: int, size: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.empty((n, 3, h, w), dtype=np.float32)
    shapes = [_disc, _square, _tri, _ring]
    for i in range(n):
        # 背景：随机双色线性渐变
        c1 = rng.uniform(0, 1, 3).astype(np.float32)
        c2 = rng.uniform(0, 1, 3).astype(np.float32)
        ang = rng.uniform(0, 2 * np.pi)
        t = (np.cos(ang) * xx + np.sin(ang) * yy) / (h + w)
        t = (t - t.min()) / (t.max() - t.min() + 1e-6)
        img = c1[None, None, :] * (1 - t)[..., None] + c2[None, None, :] * t[..., None]
        # 1~3 个形状
        for _ in range(rng.randint(1, 4)):
            f = shapes[rng.randint(len(shapes))]
            cx, cy = rng.uniform(0.25, 0.75, 2) * h
            r = rng.uniform(0.10, 0.28) * h
            col = rng.uniform(0, 1, 3).astype(np.float32)
            a = rng.uniform(0, 2 * np.pi)
            if f is _ring:
                m = f(h, w, cx, cy, r, rng.uniform(0.15, 0.45) * r)
            elif f is _disc:
                m = f(h, w, cx, cy, r)
            else:
                m = f(h, w, cx, cy, r, a)
            img[m] = col
        # 噪声 + 轻微模糊（用 3x3 均值代替，避免依赖 scipy）
        img += rng.normal(0, 0.02, img.shape).astype(np.float32)
        k = img.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                k += np.roll(np.roll(img, dy, 0), dx, 1)
        img = (k / 10.0).clip(0, 1)
        out[i] = np.transpose(img, (2, 0, 1)) * 2 - 1      # → [-1,1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    x = make(a.n, a.size, a.seed)
    np.savez_compressed(a.out, x=x.astype(np.float16))
    print(f"[synth] {a.out}  {x.shape}  均值 {x.mean():.3f} 标准差 {x.std():.3f}  "
          f"{os.path.getsize(a.out)/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
