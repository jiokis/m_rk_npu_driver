#!/usr/bin/env python3
"""to_rknn.py — 训练好的 DDPM → ONNX → **RK3588 NPU 的 .rknn**

## 这个脚本把「踩过的坑」都固化了

| 坑 | 处理 |
|---|---|
| **标定集路径不能含空格** | 强制写到 `CALIB_DIR`（仓库外，`/home/dunaandone/rknn-work/...`） |
| **多输入模型的标定集**：`dataset.txt` 每行要给**每个输入**一个文件 | 生成 `x_*.npy` + `t_*.npy`，每行两个路径 |
| **标定分布要匹配真实输入** | 不用随机噪声，而是**真数据 + 随机 t** 算出真实的 `x_t` 与 `temb`（扩散模型的输入分布随 t 剧烈变化） |
| **非 3 通道模型不能喂图片** | 全部用 `.npy` |
| ONNX 要 **NCHW + 静态形状** | `unet.py` 已经是这样 |

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/diffusion/to_rknn.py --ckpt deploy/diffusion/ckpt/ddpm_mnist.pt \
      --out deploy/diffusion/rknn/ddpm_mnist.rknn --calib-n 32
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unet import TinyUNet, sinusoidal_embedding  # noqa: E402

# ⚠️ 标定集**必须**在无空格路径下：工具链按空白切分 dataset.txt 的每一行
CALIB_DIR = "/home/dunaandone/rknn-work/diffcalib"


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


def build_calib(n: int, T: int, abar, size: int, temb_dim: int, seed: int = 0):
    """★ 用**真数据 + 随机 t** 造标定集：这才是模型真实见到的输入分布"""
    from train_ddpm import get_data
    d = CALIB_DIR
    os.makedirs(d, exist_ok=True)
    x_all = get_data()[:2000]
    g = torch.Generator().manual_seed(seed)
    lines = []
    for i in range(n):
        x0 = x_all[torch.randint(0, x_all.shape[0], (1,), generator=g)]
        t = int(torch.randint(0, T, (1,), generator=g))
        e = torch.randn(x0.shape, generator=g)
        ab = abar[t]
        xt = (torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * e).numpy().astype(np.float32)
        tb = sinusoidal_embedding(torch.tensor([t]), temb_dim).numpy().astype(np.float32)
        px, pt = os.path.join(d, f"x_{i:03d}.npy"), os.path.join(d, f"t_{i:03d}.npy")
        np.save(px, xt)
        np.save(pt, tb)
        lines.append(f"{px} {pt}")
    txt = os.path.join(d, "dataset.txt")
    open(txt, "w").write("\n".join(lines) + "\n")
    print(f"[calib] {n} 个样本 → {txt}  (每行两个文件：x 与 temb)")
    return txt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--calib-n", type=int, default=32)
    ap.add_argument("--no-quant", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    T, temb_dim, size = a["T"], a["temb_dim"], 32
    model = TinyUNet(in_ch=1, base=a["base"], temb_dim=temb_dim, norm=a["norm"]).eval()
    model.load_state_dict(ck["model"])
    print(f"[ckpt] {args.ckpt}  参数 {ck['nparam']/1e6:.3f} M  loss {ck.get('loss_final', float('nan')):.4f}")

    onnx_path = args.onnx or os.path.join(HERE, "onnx", "ddpm_deploy.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    x = torch.zeros(1, 1, size, size)
    tb = torch.zeros(1, temb_dim, 1, 1)
    torch.onnx.export(model, (x, tb), onnx_path, input_names=["x", "temb"],
                      output_names=["eps"], opset_version=13, do_constant_folding=True)
    print(f"[onnx] {onnx_path}  {os.path.getsize(onnx_path)/1e6:.2f} MB")

    from rknn.api import RKNN
    r = RKNN(verbose=False)
    assert r.config(target_platform="rk3588", optimization_level=3,
                    quantized_dtype="w8a8") == 0
    assert r.load_onnx(model=onnx_path) == 0
    cal = None
    if not args.no_quant:
        _, _, abar = make_schedule(T)
        cal = build_calib(args.calib_n, T, abar, size, temb_dim)
    ret = r.build(do_quantization=cal is not None, dataset=cal)
    print(f"[rknn] build ret={ret}")
    if ret != 0:
        r.release(); return 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ret = r.export_rknn(args.out)
    print(f"[rknn] export ret={ret}  {args.out}  {os.path.getsize(args.out)/1e6:.2f} MB")
    r.release()

    # ★ 把采样循环需要的元信息一起存下来（板子上只有 numpy，不需要 torch）
    import json
    meta = {"T": T, "temb_dim": temb_dim, "size": size, "schedule": "cosine",
            "base": a["base"], "norm": a["norm"], "nparam": ck["nparam"]}
    mpath = os.path.splitext(args.out)[0] + ".json"
    json.dump(meta, open(mpath, "w"), indent=1)
    print(f"[meta] {mpath}  {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
