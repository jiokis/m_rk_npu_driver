#!/usr/bin/env python3
"""train_latent.py — 训练 **SD 架构**（VAE 潜空间扩散）的两个阶段

## 两个阶段（分开训，各自便宜）

**阶段 1：VAE**（编码器 + 解码器）
  * 目标：重构 + KL（β 小一点，先保重构）
  * **编码器只在训练时需要** —— 部署到 NPU 的**只有解码器**（重要简化）

**阶段 2：潜空间扩散 UNet**（VAE 冻结）
  * 在 VAE 的潜空间上做标准 DDPM
  * 潜空间按数据统计**缩放**（SD 的做法：latent × scale ≈ 单位方差）⇒ 训练更稳

## 为什么这样切

* VAE 在 32×32 上训练便宜（编码器/解码器都是小网络）
* 潜空间只有 8×8，扩散 UNet 的 FLOPs 极低（实测 0.14 GFLOPs/步）
* **只有解码器要上 NPU** ⇒ RKNN 侧只需要转两个模型（UNet + Decoder）

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/diffusion/train_latent.py --stage vae    --steps 6000
  $V deploy/diffusion/train_latent.py --stage unet   --steps 12000 --vae <vae.pt>
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
from latent_arch import Decoder, Encoder, LatentUNet  # noqa: E402

DATA = "/home/dunaandone/rknn-work/cifar"
CIFAR_TAR = os.path.join(DATA, "cifar-10-python.tar.gz")
CIFAR_DIR = os.path.join(DATA, "cifar-10-batches-py")


SYNTH_NPZ = "/home/dunaandone/rknn-work/synth32.npz"
SYNTH64_NPZ = "/home/dunaandone/rknn-work/synth64.npz"


def load_data(path: str | None = None) -> torch.Tensor:
    """数据源优先级：本地合成集 > CIFAR-10（后者实测被限速到 ~10 KB/s，基本拿不到）

    ⚠️ 实测：CIFAR-10 / STL-10 的源都只有 10~12 KB/s（170 MB 要 ~4.7 小时），
       HuggingFace 与 hf-mirror **完全不可达**。所以默认用本地程序化生成的数据集
       （见 synth_data.py），它完全可复现、结构清晰、能一眼看出生成是否有效。
    """
    path = path or SYNTH_NPZ
    if os.path.exists(path):
        x = np.load(path)["x"].astype(np.float32)
        print(f"[data] 本地合成集 {path}  {tuple(x.shape)}")
        return torch.from_numpy(x)
    import pickle
    if not os.path.isdir(CIFAR_DIR):
        raise SystemExit(f"既没有 {SYNTH_NPZ}，也没有 {CIFAR_DIR}")
    xs = []
    for i in range(1, 6):
        with open(os.path.join(CIFAR_DIR, f"data_batch_{i}"), "rb") as f:
            d = pickle.load(f, encoding="bytes")
            xs.append(d[b"data"])
    x = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return torch.from_numpy((x - 0.5) / 0.5)


def make_schedule(T: int, kind: str = "cosine"):
    if kind == "cosine":
        s = 0.008
        t = torch.linspace(0, T, T + 1) / T
        ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = ab / ab[0]
        betas = torch.clip(1 - ab[1:] / ab[:-1], 1e-4, 0.999)
    else:
        betas = torch.linspace(1e-4, 0.02, T)
    return betas, 1 - betas, torch.cumprod(1 - betas, 0)


def sin_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    a = t.float()[:, None] * f[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], 1)[:, :, None, None]


# ------------------------------------------------------------- 阶段 1：VAE

def train_vae(a, x):
    base = a.base
    enc = Encoder(base=base, downs=a.downs)
    dec = Decoder(base=base, downs=a.downs)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    nparam = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(f"[vae] base={base} 参数 {nparam/1e6:.3f} M  batch={a.batch}")
    t0, losses, kls = time.time(), [], []
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, x.shape[0], (a.batch,))
        xb = x[idx]
        h = enc(xb)
        mu, logvar = h.chunk(2, 1)
        logvar = logvar.clamp(-6, 2)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        rec = dec(z)
        # 重构用 L1+L2 混合（比纯 MSE 少糊一点），KL 权重小（先保重构）
        l_rec = F.l1_loss(rec, xb) + 0.5 * F.mse_loss(rec, xb)
        l_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = l_rec + a.kl * l_kl
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
        opt.step(); sched.step()
        losses.append(l_rec.item()); kls.append(l_kl.item())
        if step % a.log_every == 0 or step == 1:
            print(f"  step {step:6d}/{a.steps}  rec {np.mean(losses[-a.log_every:]):.4f}  "
                  f"kl {np.mean(kls[-a.log_every:]):.3f}  { (time.time()-t0)/step*1000:.0f} ms/step  "
                  f"已用 {(time.time()-t0)/60:.1f} min", flush=True)
        if a.save_every and step % a.save_every == 0:
            torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(),
                        "args": vars(a), "nparam": nparam, "step": step},
                       f"{os.path.splitext(a.out)[0]}_s{step}.pt")
    # ★ 潜空间缩放：让 latent 的每个通道接近单位方差（SD 的 0.18215 就是这么来的）
    enc.eval()
    with torch.no_grad():
        h = enc(x[:2000])
        mu = h.chunk(2, 1)[0]
        scale = 1.0 / mu.std().item()
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "args": vars(a),
                "nparam": nparam, "latent_scale": scale}, a.out)
    print(f"[vae] 保存 {a.out}  潜空间缩放 scale={scale:.4f}  "
          f"最终 rec {np.mean(losses[-200:]):.4f}  用时 {(time.time()-t0)/60:.1f} min")
    return scale


# ------------------------------------------------------------- 阶段 2：潜空间扩散

def train_unet(a, x):
    v = torch.load(a.vae, map_location="cpu")
    va = v["args"]
    scale = v["latent_scale"]
    enc = Encoder(base=va["base"], downs=va.get("downs", 2)).eval()
    enc.load_state_dict(v["enc"])
    for p in enc.parameters():
        p.requires_grad_(False)
    lat = x.shape[-1] // (2 ** va.get("downs", 2))
    print(f"[unet] 潜空间边长 {lat}")
    model = LatentUNet(latent_ch=4, base=a.base, mult=(1, 2, 4), temb_dim=a.temb_dim)
    if a.init_unet and os.path.exists(a.init_unet):
        old = torch.load(a.init_unet, map_location="cpu", weights_only=False)
        model.load_state_dict(old["model"])
        print(f"[unet] 微调自 {a.init_unet}（原 loss {old.get('loss_final', float('nan')):.4f}）")
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[unet] base={a.base} 参数 {nparam/1e6:.3f} M  潜空间缩放 {scale:.4f}")

    # 预先把全部潜空间算好（CIFAR 只有 5 万张，编码很快）⇒ 训练时不用再跑编码器
    zs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 512):
            h = enc(x[i:i + 512])
            zs.append(h.chunk(2, 1)[0] * scale)
    z = torch.cat(zs)
    print(f"[unet] 潜空间张量 {tuple(z.shape)}  均值 {z.mean():.3f} 标准差 {z.std():.3f}")

    T = a.T
    _, _, abar = make_schedule(T)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    t0, losses = time.time(), []
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, z.shape[0], (a.batch,))
        z0 = z[idx]
        t = torch.randint(0, T, (a.batch,))
        e = torch.randn_like(z0)
        ab = abar[t][:, None, None, None]
        zt = torch.sqrt(ab) * z0 + torch.sqrt(1 - ab) * e
        pred = model(zt, sin_emb(t, a.temb_dim))
        loss = F.mse_loss(pred, e)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        if step % a.log_every == 0 or step == 1:
            print(f"  step {step:6d}/{a.steps}  loss {np.mean(losses[-a.log_every:]):.4f}  "
                  f"{(time.time()-t0)/step*1000:.0f} ms/step  已用 {(time.time()-t0)/60:.1f} min",
                  flush=True)
        if a.save_every and step % a.save_every == 0:
            torch.save({"model": model.state_dict(), "args": vars(a), "nparam": nparam,
                        "latent_scale": scale, "vae_args": va, "step": step},
                       f"{os.path.splitext(a.out)[0]}_s{step}.pt")
    torch.save({"model": model.state_dict(), "args": vars(a), "nparam": nparam,
                "latent_scale": scale, "vae_args": va}, a.out)
    print(f"[unet] 保存 {a.out}  最终 loss {np.mean(losses[-200:]):.4f}  "
          f"用时 {(time.time()-t0)/60:.1f} min")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["vae", "unet"], required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--kl", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--temb-dim", type=int, default=128)
    ap.add_argument("--vae", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--downs", type=int, default=2, help="VAE 下采样级数（2=4x, 3=8x）")
    ap.add_argument("--data", default=None, help="数据集 .npz（缺省 synth32）")
    ap.add_argument("--init-unet", default=None,
                    help="用已有 UNet 权重做**微调**（换 VAE 后潜空间分布变了，微调比从零训快得多）")
    a = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    x = load_data(a.data)
    if a.subset:
        x = x[:a.subset]
    print(f"[data] {tuple(x.shape)}  范围 [{x.min():.2f},{x.max():.2f}]")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if a.stage == "vae":
        train_vae(a, x)
    else:
        if not a.vae:
            raise SystemExit("--stage unet 需要 --vae")
        train_unet(a, x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
