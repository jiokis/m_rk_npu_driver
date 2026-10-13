#!/usr/bin/env python3
"""unet.py — 为 **RK3588 NPU** 设计的极小扩散 UNet（RKNN 友好优先）

## 设计取舍（每条都是为了「能转成 RKNN」）

| 选择 | 原因 |
|---|---|
| **输入静态**：`x[1,C,H,W]` + `temb[1,D,1,1]` | RKNN 只吃静态形状（本项目已反复验证） |
| **时间步嵌入由 CPU 算好、当张量喂进来** | 省掉 MLP/MatMul，避免 RKNN 不支持或很慢的算子 |
| **每块用 1×1 conv 把 `temb` 投到通道维再相加** | 只有 Conv + Add，融合友好 |
| **`norm` 可切换** `none/group/batch` | GroupNorm 是 UNet 标配，但 RKNN 支持存疑 ⇒ 先试，不行就换 |
| **上采样用 nearest `Upsample`** | 最容易被 RKNN 接受（`Resize`/`Upsample` 都试过） |
| **不做 attention** | Softmax/MatMul 在 NPU 上又慢又不一定支持；小模型不需要 |
| **激活用 SiLU** | IR 表里见过 `ConvExSwish` ⇒ **Swish 能被融进 Conv**（docs/132） |

## 与 RKNN 的接口约定（踩过的坑，别再踩）

* ONNX 输入按 **NCHW** 造（工具链按 NCHW 读；用 NHWC 会被读成 H，见 docs/128）
* 非 3 通道模型做量化时，标定集不能用图片 ⇒ 要喂 **.npy**（或不做量化）
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_layer(kind: str, ch: int) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "group":
        g = 8 if ch % 8 == 0 else (4 if ch % 4 == 0 else 1)
        return nn.GroupNorm(g, ch)
    raise ValueError(f"未知 norm: {kind}")


class ResBlock(nn.Module):
    """conv3x3 → norm → SiLU → conv3x3 → norm，加上时间步嵌入（1×1 conv 投影）"""

    def __init__(self, cin: int, cout: int, temb_dim: int, norm: str = "group"):
        super().__init__()
        self.n1 = norm_layer(norm, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Conv2d(temb_dim, cout, 1)
        self.n2 = norm_layer(norm, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(temb)
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, cin, cout, temb_dim, norm="group"):
        super().__init__()
        self.r1 = ResBlock(cin, cout, temb_dim, norm)
        self.r2 = ResBlock(cout, cout, temb_dim, norm)
        self.down = nn.Conv2d(cout, cout, 3, stride=2, padding=1)

    def forward(self, x, temb):
        x = self.r2(self.r1(x, temb), temb)
        return x, self.down(x)


class Up(nn.Module):
    def __init__(self, cin, cout, temb_dim, norm="group"):
        super().__init__()
        # nearest 上采样（RKNN 友好），通道先由 concat 变成 cin
        self.r1 = ResBlock(cin, cout, temb_dim, norm)
        self.r2 = ResBlock(cout, cout, temb_dim, norm)

    def forward(self, x, skip, temb):
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.r2(self.r1(x, temb), temb)


class TinyUNet(nn.Module):
    """极小 UNet。默认 32×32 单通道（MNIST 尺度），约 1~3 M 参数"""

    def __init__(self, in_ch: int = 1, base: int = 32, mult=(1, 2, 4),
                 temb_dim: int = 128, norm: str = "group", out_ch: int | None = None):
        super().__init__()
        self.temb_dim = temb_dim
        chs = [base * m for m in mult]
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.downs = nn.ModuleList()
        prev = chs[0]
        for c in chs:
            self.downs.append(Down(prev, c, temb_dim, norm))
            prev = c
        self.mid = ResBlock(prev, prev, temb_dim, norm)
        self.ups = nn.ModuleList()
        for c in reversed(chs):
            self.ups.append(Up(prev + c, c, temb_dim, norm))
            prev = c
        self.out_norm = norm_layer(norm, prev)
        self.out_conv = nn.Conv2d(prev, out_ch or in_ch, 3, padding=1)

    def forward(self, x, temb):
        h = self.in_conv(x)
        skips = []
        for d in self.downs:
            h, hd = d(h, temb)
            skips.append(h)
            h = hd
        h = self.mid(h, temb)
        for u, s in zip(self.ups, reversed(skips)):
            h = u(h, s, temb)
        return self.out_conv(F.silu(self.out_norm(h)))


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: [B] 整数步 → [B, dim, 1, 1]（CPU 侧算好再喂给 NPU）"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None, :]
    e = torch.cat([torch.sin(a), torch.cos(a)], dim=1)
    return e[:, :, None, None]


if __name__ == "__main__":
    for norm in ("none", "batch", "group"):
        m = TinyUNet(norm=norm).eval()
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            y = m(torch.zeros(1, 1, 32, 32), torch.zeros(1, 128, 1, 1))
        print(f"norm={norm:6s} 参数 {n/1e6:.3f} M  输出 {tuple(y.shape)}")
