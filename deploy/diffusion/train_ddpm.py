#!/usr/bin/env python3
"""train_ddpm.py — 在 **CPU** 上训练一个「能在 RK3588 NPU 上跑」的极小 DDPM

## 为什么用 CPU 训练

本机没有 NVIDIA GPU；模型只有 ~2.4 M 参数、32×32 单通道，
CPU 训练是可行的（比追 GPU 环境便宜得多）。

## 关键设计（都是为了**部署到 NPU**）

| 选择 | 原因 |
|---|---|
| **`norm=batch`** | 导出时会**折进 Conv**（实测 .rknn 大小与 `norm=none` 完全相同）⇒ 推理零成本 |
| `norm=group` 不用 | 会多出 **54 个 `Reshape`**（IR 表实测）⇒ 慢 |
| 时间步嵌入**当张量喂**（`temb[1,D,1,1]`） | 省掉 MLP/MatMul，且 CPU 侧算很便宜 |
| 输出 = 预测 **ε**（噪声） | 最标准的 DDPM 参数化，且输出范围有界，量化友好 |
| 数据 pad 28→32 | 32 是 2 的幂，UNet 三次下采样后正好 4×4 |

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/diffusion/train_ddpm.py --steps 3000 --out deploy/diffusion/ckpt/ddpm.pt
  $V deploy/diffusion/train_ddpm.py --steps 200 --subset 2000 --out /tmp/quick.pt   # 快速验证链路
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unet import TinyUNet, sinusoidal_embedding  # noqa: E402

DATA = "/home/dunaandone/rknn-work/mnist"   # ⚠️ 不能放在含空格的仓库路径下


def get_data(subset: int | None = None):
    from torchvision import datasets, transforms
    os.makedirs(DATA, exist_ok=True)
    ds = datasets.MNIST(DATA, train=True, download=True,
                        transform=transforms.ToTensor())
    x = ds.data.float() / 255.0            # [N,28,28]
    x = (x - 0.5) / 0.5                    # → [-1,1]
    x = F.pad(x[:, None], (2, 2, 2, 2))    # 28→32
    if subset:
        x = x[:subset]
    return x


def make_schedule(T: int, kind: str = "cosine"):
    if kind == "cosine":
        s = 0.008
        t = torch.linspace(0, T, T + 1) / T
        ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = ab / ab[0]
        betas = torch.clip(1 - ab[1:] / ab[:-1], 1e-4, 0.999)
    else:
        betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, 0)
    return betas, alphas, abar


@torch.no_grad()
def sample(model, T, alphas, abar, n=16, size=32, temb_dim=128, steps=None, seed=0):
    """DDPM 采样（CPU 版，用于训练中途看效果）"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, size, size, generator=g)
    idxs = list(range(T - 1, -1, -1))
    if steps:
        idxs = list(range(T - 1, -1, -1))[:: max(1, T // steps)]
    for i in idxs:
        tb = torch.full((n,), i, dtype=torch.long)
        e = model(x, sinusoidal_embedding(tb, temb_dim))
        a, ab = alphas[i], abar[i]
        mean = (x - (1 - a) / torch.sqrt(1 - ab) * e) / torch.sqrt(a)
        if i > 0:
            ab_prev = abar[i - 1]
            var = (1 - a) * (1 - ab_prev) / (1 - ab)
            x = mean + torch.sqrt(var) * torch.randn(x.shape, generator=g)
        else:
            x = mean
    return x.clamp(-1, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--temb-dim", type=int, default=128)
    ap.add_argument("--norm", default="batch", choices=["none", "batch", "group"])
    ap.add_argument("--out", default=os.path.join(HERE, "ckpt", "ddpm.pt"))
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--preview-every", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0,
                    help="每 N 步存一次中间 checkpoint（**踩过：不存就只能等全部跑完**）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 4))

    x = get_data(args.subset)
    N = x.shape[0]
    print(f"[data] {N} 张 32x32 单通道  (MNIST pad 28→32)")

    model = TinyUNet(in_ch=1, base=args.base, temb_dim=args.temb_dim, norm=args.norm)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] TinyUNet base={args.base} norm={args.norm} 参数 {nparam/1e6:.3f} M")

    betas, alphas, abar = make_schedule(args.T)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch,))
        x0 = x[idx]
        t = torch.randint(0, args.T, (args.batch,))
        e = torch.randn_like(x0)
        ab = abar[t][:, None, None, None]
        xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * e
        pred = model(xt, sinusoidal_embedding(t, args.temb_dim))
        loss = F.mse_loss(pred, e)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t0
            print(f"  step {step:6d}/{args.steps}  loss {np.mean(losses[-args.log_every:]):.4f}  "
                  f"{dt/step*1000:.0f} ms/step  已用 {dt/60:.1f} min", flush=True)
        if args.save_every and step % args.save_every == 0:
            ck = os.path.splitext(args.out)
            mid = f"{ck[0]}_s{step}{ck[1]}"
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "nparam": nparam, "loss_final": float(np.mean(losses[-200:])),
                        "step": step}, mid)
            print(f"  [ckpt] {mid}   loss {np.mean(losses[-200:]):.4f}", flush=True)
        if args.preview_every and step % args.preview_every == 0:
            model.eval()
            s = sample(model, args.T, alphas, abar, n=16, temb_dim=args.temb_dim, steps=100)
            print(f"  [preview@{step}] 采样 16 张: 均值 {s.mean():.3f} 标准差 {s.std():.3f}", flush=True)
            model.train()

    model.eval()
    torch.save({"model": model.state_dict(), "args": vars(args),
                "nparam": nparam, "loss_final": float(np.mean(losses[-200:]))}, args.out)
    print(f"[save] {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)  "
          f"最终 loss {np.mean(losses[-200:]):.4f}  总耗时 {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
