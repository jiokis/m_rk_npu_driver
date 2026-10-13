#!/usr/bin/env python3
"""latent_arch.py — **SD 架构**（VAE 潜空间扩散）的 RKNN 友好实现 + FLOPs 预算

## 预算算术（实测值，见 data/analysis/npu_budget.txt）

| 量 | 实测 | 来源 |
|---|---|---|
| NPU 有效算力 | **≈ 1.0 TFLOP/s**（INT8） | resnet18 1.82 GFLOPs / 1.87 ms；yolov5s 16.5 GFLOPs / 28 ms |
| DDR 带宽 | **≈ 20.8 GB/s**（memcpy） | 板上 numpy 实测 |
| **算力/带宽比** | **24 MAC/byte** | 1e12 / 20.8e9 / 2 |

**⇒ 一个算子「算力受限」的条件是算术强度 > 24 MAC/byte。**
3x3 卷积（Cin=Cout=C，空间 SxS，INT8）的强度 ≈ `9·S²·C² / (2·S²·C + 9·C²)`：

| S | C | 强度 | 判定 |
|---|---|---|---|
| 32 | 256 | **544** | 算力受限 ✓ |
| 8 | 128 | **58** | 算力受限 ✓ |
| 8 | 32 | **44** | 算力受限（勉强） |
| 4 | 16 | **13** | **带宽受限** ✗ |

**⇒ 设计原则：通道 ≥ 64、空间 ≥ 8×8 才不浪费算力。**
**⇒ 这也说明为什么「8× 下采样的 VAE + 潜空间 UNet」是甜点：
   潜空间小 ⇒ FLOPs 低；但潜空间不能小到让通道/空间掉进带宽受限区。**

## 结构（SD 的简化版）

```
图像 64x64x3 ──Encoder──▶ 潜空间 8x8x4 ──Diffusion UNet──▶ 潜空间 8x8x4 ──Decoder──▶ 64x64x3
             (8x 下采样)                    (在潜空间去噪)        (只在最后跑一次)
```
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_layer(kind: str, ch: int) -> nn.Module:
    """`batch` 会在导出时折进 Conv（实测 .rknn 大小与 none 相同）⇒ 推理零成本"""
    if kind == "none":
        return nn.Identity()
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    g = 8 if ch % 8 == 0 else (4 if ch % 4 == 0 else 1)
    return nn.GroupNorm(g, ch)


class Res(nn.Module):
    def __init__(self, cin, cout, temb=None, norm="batch"):
        super().__init__()
        self.n1 = norm_layer(norm, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Conv2d(temb, cout, 1) if temb else None
        self.n2 = norm_layer(norm, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb=None):
        h = self.c1(F.silu(self.n1(x)))
        if self.emb is not None and temb is not None:
            h = h + self.emb(temb)
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


# ----------------------------------------------------------------- VAE

#: 下采样级数（每级 stride-2 ⇒ 空间减半）
#:
#: ⚠️ **不要盲目照搬 SD 的 8×**：8× 是为 512x512 设计的（512/8 = 64x64 潜空间）。
#:    在 32x32 上用 8× 只剩 **4x4x4 = 64 个数**去表示 32x32x3 = 3072 个值（48 倍压缩），
#:    实测**重构明显发糊、形状边界丢失**（见 docs/img/vae_recon_8x_too_lossy.png）。
#:    ⇒ 判据是**潜空间的绝对大小**（≥ 8x8），不是压缩比。
DEFAULT_DOWNS = 2


class Encoder(nn.Module):
    """HxWx3 → (H/2^downs)x(W/2^downs)x(2*latent_ch)（均值 + logvar）"""

    def __init__(self, in_ch=3, base=32, latent_ch=4, norm="batch", downs=DEFAULT_DOWNS):
        super().__init__()
        self.downs = downs
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        blocks = []
        prev = base
        for i in range(downs):
            cout = base * (2 ** min(i + 1, 2))     # 通道最多翻到 4x，避免过深
            blocks.append(nn.Sequential(Res(prev, prev, norm=norm),
                                        nn.Conv2d(prev, cout, 3, stride=2, padding=1)))
            prev = cout
        self.blocks = nn.ModuleList(blocks)
        self.mid = Res(prev, prev, norm=norm)
        self.out = nn.Conv2d(prev, latent_ch * 2, 3, padding=1)

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return self.out(self.mid(h))


class Decoder(nn.Module):
    """潜空间 → 图像（nearest 上采样，RKNN 友好）"""

    def __init__(self, out_ch=3, base=32, latent_ch=4, norm="batch", downs=DEFAULT_DOWNS):
        super().__init__()
        self.downs = downs
        c0 = base * (2 ** min(downs, 2))
        self.stem = nn.Conv2d(latent_ch, c0, 3, padding=1)
        self.mid = Res(c0, c0, norm=norm)
        ups = []
        prev = c0
        for i in range(downs):
            cout = base * (2 ** max(downs - 1 - i, 0))
            ups.append(nn.Sequential(Res(prev, prev, norm=norm),
                                     nn.Conv2d(prev, cout, 3, padding=1)))
            prev = cout
        self.ups = nn.ModuleList(ups)
        self.out = nn.Conv2d(prev, out_ch, 3, padding=1)

    def forward(self, z):
        h = self.mid(self.stem(z))
        for u in self.ups:
            h = u(F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(h)


# ----------------------------------------------------------------- 潜空间 UNet

class LatentUNet(nn.Module):
    """潜空间 UNet：z[1,lc,h,w] + temb[1,D,1,1] → eps[1,lc,h,w]

    默认 h=w=8（64x64 图像 8x 下采样），三级：8 → 4 → 2
    """

    def __init__(self, latent_ch=4, base=128, mult=(1, 2, 4), temb_dim=128, norm="batch"):
        super().__init__()
        chs = [base * m for m in mult]
        self.temb_dim = temb_dim
        self.in_conv = nn.Conv2d(latent_ch, chs[0], 3, padding=1)
        self.down = nn.ModuleList()
        prev = chs[0]
        for c in chs:
            self.down.append(nn.ModuleDict({
                "r1": Res(prev, c, temb_dim, norm),
                "r2": Res(c, c, temb_dim, norm),
                "ds": nn.Conv2d(c, c, 3, stride=2, padding=1)}))
            prev = c
        self.mid = Res(prev, prev, temb_dim, norm)
        self.up = nn.ModuleList()
        for c in reversed(chs):
            self.up.append(nn.ModuleDict({
                "r1": Res(prev + c, c, temb_dim, norm),
                "r2": Res(c, c, temb_dim, norm)}))
            prev = c
        self.out = nn.Conv2d(prev, latent_ch, 3, padding=1)

    def forward(self, z, temb):
        h = self.in_conv(z)
        skips = []
        for d in self.down:
            h = d["r2"](d["r1"](h, temb), temb)
            skips.append(h)
            h = d["ds"](h)
        h = self.mid(h, temb)
        for u, s in zip(self.up, reversed(skips)):
            h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = torch.cat([h, s], 1)
            h = u["r2"](u["r1"](h, temb), temb)
        return self.out(h)


# ----------------------------------------------------------------- FLOPs

def count_macs(model: nn.Module, *shapes) -> int:
    """用 forward hook 数 MACs（conv/linear 为主；nearest 上采样与 concat 不计）"""
    total = [0]
    hooks = []

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            o = out.shape
            k = m.kernel_size[0] * m.kernel_size[1]
            total[0] += o[0] * o[1] * o[2] * o[3] * (m.in_channels // m.groups) * k

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(*[torch.zeros(*s) for s in shapes])
    for h in hooks:
        h.remove()
    return total[0]
