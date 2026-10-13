#!/usr/bin/env python3
"""sample.py — 跑完整的 DDPM 采样循环，**UNet 可以在 CPU(torch) 或 NPU(rknn)** 上

## 为什么一个脚本两种后端

1. **先在笔记本上用 torch 采样**，确认模型训出来了（不然板子上出问题分不清是模型还是部署）
2. 再在板子上用 rknn 采样，**同一份随机种子**下与 torch 结果对比 —— 量化误差可量化
3. 出图、计时、存 PNG 一套代码

## 用法（笔记本）
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/diffusion/sample.py --backend torch --ckpt deploy/diffusion/ckpt/ddpm_mnist.pt \
      --n 64 --steps 200 --out /tmp/torch_grid.png

## 用法（板子）
  python3 sample.py --backend rknn --model ddpm_mnist.rknn --meta ddpm_mnist.json \
      --n 64 --steps 200 --out /tmp/npu_grid.png
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
    """与训练完全一致（cosine，s=0.008）"""
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
    """与训练一致：t[B] → [B,dim,1,1]"""
    half = dim // 2
    f = np.exp(-math.log(10000) * np.arange(half, dtype=np.float32) / half)
    a = t.astype(np.float32)[:, None] * f[None, :]
    return np.concatenate([np.sin(a), np.cos(a)], 1)[:, :, None, None]


# ----------------------------------------------------------------- 后端

class TorchBackend:
    def __init__(self, ckpt: str):
        import torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from unet import TinyUNet
        ck = torch.load(ckpt, map_location="cpu")
        a = ck["args"]
        self.torch = torch
        self.m = TinyUNet(in_ch=1, base=a["base"], temb_dim=a["temb_dim"], norm=a["norm"]).eval()
        self.m.load_state_dict(ck["model"])
        self.dim = a["temb_dim"]
        self.name = f"torch({ck['nparam']/1e6:.3f}M)"

    def __call__(self, x, tb):
        with self.torch.no_grad():
            return self.m(self.torch.from_numpy(x), self.torch.from_numpy(tb)).numpy()


class RknnBackend:
    """RKNN 后端

    ⚠️ **RKNN 模型的 batch 是固定为 1 的**（静态形状）——
       喂 [16,1,32,32] 会返回 1024 个元素（只算了第一张）。
       所以这里**逐张循环**（踩过一次，报错是
       `cannot reshape array of size 1024 into shape (16,1,32,32)`）。
    """

    def __init__(self, model: str):
        from rknnlite.api import RKNNLite
        self.r = RKNNLite(verbose=False)
        assert self.r.load_rknn(model) == 0, "load_rknn 失败"
        assert self.r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0, "init_runtime 失败"
        self.name = "rknn(NPU 0_1_2)"
        self._first = True
        self.calls = 0
        self.batch = 1

    def _one(self, x, tb):
        out = self.r.inference(inputs=[x.astype(np.float32), tb.astype(np.float32)])
        o = out[0]
        if self._first:
            print(f"[rknn] 首次输出: dtype={o.dtype} shape={o.shape} "
                  f"（输入 x{x.shape} temb{tb.shape}）")
            self._first = False
        if o.dtype != np.float32:
            # 输出被量化成整型 ⇒ 用输出 scale 反量化。
            # RKNNLite 不直接暴露 scale，这里先按 /128 试，并在打印里给出数值范围供核对。
            print(f"[rknn] ⚠️ 输出 dtype={o.dtype} 范围[{o.min()},{o.max()}] ⇒ 按 /128 反量化")
            o = o.astype(np.float32) / 128.0
        return o.reshape(x.shape)

    def __call__(self, x, tb):
        if x.shape[0] == 1:
            self.calls += 1
            return self._one(x, tb)
        # 逐张跑（batch 固定 1）
        outs = []
        for i in range(x.shape[0]):
            self.calls += 1
            outs.append(self._one(x[i:i + 1], tb[i:i + 1]))
        return np.concatenate(outs, 0)

    def release(self):
        self.r.release()


# ----------------------------------------------------------------- 采样

def sample(backend, T, alphas, abar, dim, n, size, steps=None, seed=0, snapshots=False):
    """★ 用 **DDIM（eta=0，确定性）** 采样，而不是 DDPM 祖先采样。

    为什么换掉祖先采样（踩过）：
      * 跳步采样时，祖先采样的方差公式要用「**下一个实际采样时刻**」的 alpha_bar，
        我原来写的是 `abar[i-1]`，跳步时完全不对 ⇒ 输出直接变 **NaN**。
      * 祖先采样每步注入随机噪声 ⇒ **torch 与 NPU 的结果无法逐点对比**。
    DDIM(eta=0) 既是确定性的（同种子两侧应逐点一致，差异只剩量化误差），
    又能正确处理跳步。
    """
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 1, size, size).astype(np.float32)
    if steps and steps < T:
        idxs = [int(round(v)) for v in np.linspace(T - 1, 0, steps)]
    else:
        idxs = list(range(T - 1, -1, -1))
    snaps, times = [], []
    for k, i in enumerate(idxs):
        tb = sin_emb(np.full((n,), i, dtype=np.int64), dim)
        t0 = time.perf_counter()
        e = backend(x, tb)
        times.append(time.perf_counter() - t0)
        ab = float(abar[i])
        # 估计 x0，再跳到下一个采样时刻
        x0 = (x - math.sqrt(1 - ab) * e) / math.sqrt(ab)
        x0 = np.clip(x0, -3.0, 3.0)          # 防爆（未训练好的模型会给出离谱的 x0）
        ab_next = float(abar[idxs[k + 1]]) if k + 1 < len(idxs) else 1.0
        # DDIM eta=0：确定性，只保留 x0 与当前噪声的方向
        x = math.sqrt(ab_next) * x0 + math.sqrt(max(1 - ab_next, 0.0)) * e
        if snapshots and k % max(1, len(idxs) // 6) == 0:
            snaps.append((i, x.copy()))
    return np.clip(x, -1, 1), np.array(times), snaps


def to_grid(imgs: np.ndarray, cols: int = 8, pad: int = 2) -> np.ndarray:
    """[N,1,H,W] in [-1,1] → uint8 网格图 [H',W']"""
    n, _, h, w = imgs.shape
    rows = (n + cols - 1) // cols
    out = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad), 255, np.uint8)
    for k in range(n):
        r, c = divmod(k, cols)
        v = ((imgs[k, 0] + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        y, x0 = pad + r * (h + pad), pad + c * (w + pad)
        out[y:y + h, x0:x0 + w] = v
    return out


def save_png(arr: np.ndarray, path: str, scale: int = 4):
    try:
        from PIL import Image
        im = Image.fromarray(arr).resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
        im.save(path)
        print(f"[png] {path}  {im.size}")
    except Exception as e:
        np.save(path + ".npy", arr)
        print(f"[png] PIL 不可用({e})，改存 {path}.npy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "rknn"], required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/ddpm_grid.png")
    ap.add_argument("--save-raw", default=None, help="把 [-1,1] 的原始张量存成 .npy（用于跨后端对比）")
    ap.add_argument("--trajectory", default=None,
                    help="存「去噪过程」图：同一张图在若干个 t 上的样子（很直观的交付物）")
    args = ap.parse_args()

    if args.backend == "torch":
        import torch
        ck = torch.load(args.ckpt, map_location="cpu")
        T, dim, size = ck["args"]["T"], ck["args"]["temb_dim"], 32
        backend = TorchBackend(args.ckpt)
    else:
        meta = json.load(open(args.meta))
        T, dim, size = meta["T"], meta["temb_dim"], meta["size"]
        backend = RknnBackend(args.model)

    alphas, abar = make_schedule(T)
    print(f"[sample] backend={backend.name}  T={T}  steps={args.steps}  n={args.n}")
    t0 = time.time()
    imgs, times, snaps = sample(backend, T, alphas, abar, dim, args.n, size,
                                steps=args.steps, seed=args.seed,
                                snapshots=bool(args.trajectory))
    dt = time.time() - t0
    print(f"[sample] {len(times)} 次 UNet 调用  总计 {dt:.2f} s  "
          f"每次 {times.mean()*1000:.1f} ms (中位 {np.median(times)*1000:.1f} ms)")
    print(f"[sample] 输出统计: 均值 {imgs.mean():.3f} 标准差 {imgs.std():.3f} "
          f"min {imgs.min():.3f} max {imgs.max():.3f}")
    if args.save_raw:
        np.save(args.save_raw, imgs)
        print(f"[raw] {args.save_raw}")
    save_png(to_grid(imgs, args.cols), args.out)
    if args.trajectory and snaps:
        # 每行 = 一个采样时刻；每列 = 同一张图（取前 8 张）
        rows = []
        for t_i, arr in snaps:
            rows.append(arr[:8])
        strip = np.concatenate(rows, 0)          # [k*8, 1, H, W]
        save_png(to_grid(strip, cols=8), args.trajectory)
        print(f"[traj] 去噪过程 {len(snaps)} 个时刻 × 8 张 → {args.trajectory}")
    if hasattr(backend, "release"):
        backend.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
