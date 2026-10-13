#!/usr/bin/env python3
"""sample_latent.py — **SD 架构**的完整采样：潜空间 DDIM → VAE 解码 → 彩色图

## 与 sample.py 的区别

| | sample.py | sample_latent.py |
|---|---|---|
| 扩散在哪 | **像素空间**（32x32 单通道） | **潜空间**（8x8x4）—— SD 的做法 |
| 解码 | 不需要 | **VAE 解码器**（8x8x4 → 32x32x3） |
| 后端 | UNet 可切 torch/rknn | **UNet 与解码器都可切 torch/rknn** |

## 两个后端各自的意义

* `--backend torch`：笔记本上验证模型本身
* `--backend rknn` ：板子上跑 **两个 NPU 模型**（UNet + Decoder），
  CPU 只做 DDIM 的算术（纯 numpy）

## 用法
  # 笔记本
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/diffusion/sample_latent.py --backend torch --unet <unet.pt> --vae <vae.pt> \
      --n 64 --steps 50 --out /tmp/latent_torch.png
  # 板子
  python3 sample_latent.py --backend rknn --unet unet.rknn --unet-meta unet.json \
      --dec dec.rknn --n 64 --steps 50 --out /tmp/latent_npu.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np


def make_schedule(T: int, kind: str = "cosine"):
    if kind == "cosine":
        s = 0.008
        t = np.linspace(0, T, T + 1) / T
        ab = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = ab / ab[0]
        betas = np.clip(1 - ab[1:] / ab[:-1], 1e-4, 0.999)
    else:
        betas = np.linspace(1e-4, 0.02, T)
    return betas.astype(np.float32), np.cumprod(1.0 - betas).astype(np.float32)


def sin_emb(t: np.ndarray, dim: int) -> np.ndarray:
    half = dim // 2
    f = np.exp(-math.log(10000) * np.arange(half, dtype=np.float32) / half)
    a = t.astype(np.float32)[:, None] * f[None, :]
    return np.concatenate([np.sin(a), np.cos(a)], 1)[:, :, None, None]


# ----------------------------------------------------------------- 后端

class TorchUNet:
    def __init__(self, ckpt):
        import torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from latent_arch import LatentUNet
        ck = torch.load(ckpt, map_location="cpu")
        a = ck["args"]
        self.torch = torch
        self.m = LatentUNet(latent_ch=4, base=a["base"], mult=(1, 2, 4),
                            temb_dim=a["temb_dim"]).eval()
        self.m.load_state_dict(ck["model"])
        self.dim = a["temb_dim"]
        self.scale = ck["latent_scale"]
        self.name = f"torch UNet({ck['nparam']/1e6:.2f}M)"

    def __call__(self, z, tb):
        with self.torch.no_grad():
            return self.m(self.torch.from_numpy(z), self.torch.from_numpy(tb)).numpy()


class TorchDec:
    def __init__(self, ckpt):
        import torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from latent_arch import Decoder
        v = torch.load(ckpt, map_location="cpu")
        self.torch = torch
        self.m = Decoder(out_ch=3, base=v["args"]["base"], latent_ch=4).eval()
        self.m.load_state_dict(v["dec"])
        self.name = "torch Decoder"

    def __call__(self, z):
        with self.torch.no_grad():
            return self.m(self.torch.from_numpy(z)).numpy()


class RknnModel:
    """RKNN 后端

    ## ★★★★★ 两个必须记住的坑（都踩过）

    **① 输入要按 NHWC 顺序喂（输出仍然是 NCHW）**
       即使 ONNX 是 NCHW 导出的，`RKNNLite.inference()` 的**输入**按 **NHWC** 解释。
       多通道模型必须把 [N,C,H,W] 转成 [N,H,W,C] 再喂，否则**数值范围对、内容完全不相关**
       （余弦≈0）。实测：转置后 **余弦 = 1.0000**。
       **输出不用转**：实测 Decoder 返回的 shape 是 `(1,3,32,32)` ⇒ 已经是 NCHW。

       ⚠️ **为什么这个 bug 会藏起来**：单通道时 `[1,1,H,W]` 与 `[1,H,W,1]`
          **内存布局完全相同**（都是 H*W 个连续值）⇒ 单通道模型不转置也对。
          本项目之前那个像素空间 DDPM 正是单通道（余弦 0.9996），所以没暴露。
          **⇒ 换到多通道模型时，这个坑一定会浮出来。**

       ⚠️ 空间尺寸为 1×1 的张量（如 `temb[1,D,1,1]`）**不需要转置**：
          `[1,D,1,1]` 与 `[1,1,1,D]` 的内存布局也相同。

    **② batch 固定为 1** ⇒ 逐张循环（见 deploy/diffusion/README.md）
    """

    def __init__(self, model, n_in, label):
        from rknnlite.api import RKNNLite
        self.r = RKNNLite(verbose=False)
        assert self.r.load_rknn(model) == 0, f"load {model} 失败"
        assert self.r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0, "init_runtime 失败"
        self.n_in = n_in
        self.label = label
        self.name = f"rknn {label}"
        self._first = True
        self.calls = 0

    @staticmethod
    def _to_nhwc(a):
        return a.transpose(0, 2, 3, 1).astype(np.float32) if a.ndim == 4 and a.shape[2] > 1 else a.astype(np.float32)

    # ⚠️ 输出**不需要**转置：实测 Decoder 返回 (1,3,32,32)（NCHW）。
    #    只有输入需要 NHWC。

    def _one(self, *arrs):
        out = self.r.inference(inputs=[self._to_nhwc(a) for a in arrs])
        o = out[0]
        if self._first:
            print(f"[rknn:{self.label}] 原始输出 shape={o.shape} dtype={o.dtype}")
            self._first = False
        if o.dtype != np.float32:
            o = o.astype(np.float32) / 128.0
        return o

    def __call__(self, *arrs):
        n = arrs[0].shape[0]
        if n == 1:
            self.calls += 1
            return self._one(*arrs)
        outs = []
        for i in range(n):
            self.calls += 1
            outs.append(self._one(*[a[i:i + 1] for a in arrs]))
        return np.concatenate(outs, 0)

    def release(self):
        self.r.release()


# ----------------------------------------------------------------- 采样

def sample(unet, T, abar, dim, scale, n, lat, steps, seed=0):
    rng = np.random.RandomState(seed)
    z = rng.randn(n, 4, lat, lat).astype(np.float32)
    idxs = [int(round(v)) for v in np.linspace(T - 1, 0, steps)] if steps < T else list(range(T - 1, -1, -1))
    times = []
    for k, i in enumerate(idxs):
        tb = sin_emb(np.full((n,), i, dtype=np.int64), dim)
        t0 = time.perf_counter()
        e = unet(z, tb)
        times.append(time.perf_counter() - t0)
        ab = float(abar[i])
        z0 = (z - math.sqrt(1 - ab) * e) / math.sqrt(ab)
        z0 = np.clip(z0, -4.0, 4.0)
        ab_next = float(abar[idxs[k + 1]]) if k + 1 < len(idxs) else 1.0
        z = math.sqrt(ab_next) * z0 + math.sqrt(max(1 - ab_next, 0.0)) * e
    return z / scale, np.array(times)


def to_grid(imgs: np.ndarray, cols=8, pad=2) -> np.ndarray:
    n, c, h, w = imgs.shape
    rows = (n + cols - 1) // cols
    out = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 255, np.uint8)
    for k in range(n):
        r, cc = divmod(k, cols)
        v = ((imgs[k].transpose(1, 2, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        y, x0 = pad + r * (h + pad), pad + cc * (w + pad)
        out[y:y + h, x0:x0 + w] = v
    return out


def save_png(arr, path, scale=4):
    from PIL import Image
    im = Image.fromarray(arr).resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
    im.save(path)
    print(f"[png] {path}  {im.size}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "rknn"], required=True)
    ap.add_argument("--unet", required=True)
    ap.add_argument("--vae", default=None, help="torch 后端的 VAE ckpt")
    ap.add_argument("--unet-meta", default=None, help="rknn 后端的 unet.json")
    ap.add_argument("--dec", default=None, help="rknn 后端的解码器 .rknn")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lat", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--out", default="/tmp/latent_grid.png")
    ap.add_argument("--save-raw", default=None)
    a = ap.parse_args()

    if a.backend == "torch":
        import torch
        ck = torch.load(a.unet, map_location="cpu")
        T, dim, scale = ck["args"]["T"], ck["args"]["temb_dim"], ck["latent_scale"]
        unet = TorchUNet(a.unet)
        dec = TorchDec(a.vae)
    else:
        meta = json.load(open(a.unet_meta))
        T, dim, scale = meta["T"], meta["temb_dim"], meta["latent_scale"]
        unet = RknnModel(a.unet, 2, "UNet")
        dec = RknnModel(a.dec, 1, "Decoder")

    _, abar = make_schedule(T)
    print(f"[latent] UNet={unet.name}  Decoder={dec.name}  T={T} steps={a.steps} n={a.n} lat={a.lat}")
    t0 = time.time()
    z, times = sample(unet, T, abar, dim, scale, a.n, a.lat, a.steps, a.seed)
    t_s = time.time() - t0
    t1 = time.time()
    img = dec(z)
    t_d = time.time() - t1
    print(f"[latent] 采样 {len(times)} 步  总 {t_s:.2f} s  每步 {times.mean()*1000:.1f} ms（中位 {np.median(times)*1000:.1f}）")
    print(f"[latent] 解码 {t_d*1000:.0f} ms   图像 {img.shape}  均值 {img.mean():.3f} 标准差 {img.std():.3f}")
    print(f"[latent] ⇒ 单张 {a.n} 张耗时 {t_s+t_d:.2f} s ⇒ **{(t_s+t_d)/a.n*1000:.0f} ms/张**")
    if a.save_raw:
        np.save(a.save_raw, img)
    save_png(to_grid(np.clip(img, -1, 1), a.cols), a.out)
    if hasattr(unet, "release"):
        unet.release()
    if hasattr(dec, "release"):
        dec.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
