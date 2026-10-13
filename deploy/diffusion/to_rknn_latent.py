#!/usr/bin/env python3
"""to_rknn_latent.py — 把**潜空间扩散 UNet** 与 **VAE 解码器** 转成 RKNN

## 只需要转两个模型

* **UNet**：`z[1,4,8,8]` + `temb[1,128,1,1]` → `eps[1,4,8,8]`（**两输入**）
* **Decoder**：`z[1,4,8,8]` → `img[1,3,32,32]`（**单输入**）
* **编码器不用转** —— 它只在训练时用（生成时不需要）

## 标定集的两种格式（都在这里固化）

* 两输入：`dataset.txt` 每行**两个文件**（空格分隔）
* 单输入：每行一个文件
* 路径**不能含空格**（仓库路径是 "orange pi"）⇒ 统一写到 `/home/dunaandone/rknn-work/difcalib3`

## 标定分布要贴近真实
UNet 的输入是「VAE 潜空间 + 随机 t 的加噪结果」⇒ 用**真实潜空间**按随机 t 加噪来造，
而不是随机高斯噪声（后者与真实分布差很远，量化会偏）。

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/diffusion/to_rknn_latent.py --unet <unet.pt> --vae <vae.pt> --name synth32
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from latent_arch import Decoder, Encoder, LatentUNet  # noqa: E402

CALIB = "/home/dunaandone/rknn-work/difcalib3"     # ⚠️ 不能含空格
SYNTH = "/home/dunaandone/rknn-work/synth32.npz"


def make_schedule(T):
    s = 0.008
    t = torch.linspace(0, T, T + 1) / T
    ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    betas = torch.clip(1 - ab[1:] / ab[:-1], 1e-4, 0.999)
    return 1 - betas, torch.cumprod(1 - betas, 0)


def build_calib(name, kind, lat, T, temb_dim, abar, vae_path, n=24, seed=0):
    """kind='unet' 两输入；kind='dec' 单输入。用**真实潜空间**造。"""
    d = os.path.join(CALIB, name)
    os.makedirs(d, exist_ok=True)
    rng = np.random.RandomState(seed)
    lines = []
    if kind == "dec":
        for i in range(n):
            p = os.path.join(d, f"z_{i:03d}.npy")
            np.save(p, (rng.randn(1, 4, lat, lat) * 1.0).astype(np.float32))
            lines.append(p)
    else:
        # 真实潜空间：用 VAE 编码真数据
        v = torch.load(vae_path, map_location="cpu", weights_only=False)
        enc = Encoder(base=v["args"]["base"], downs=v["args"].get("downs", 2)).eval()
        enc.load_state_dict(v["enc"])
        x = torch.from_numpy(np.load(SYNTH)["x"][:200].astype(np.float32))
        with torch.no_grad():
            mu = enc(x).chunk(2, 1)[0] * v["latent_scale"]
        for i in range(n):
            z0 = mu[rng.randint(0, mu.shape[0])]
            t = int(rng.randint(0, T))
            ab = abar[t]
            zt = (torch.sqrt(ab) * z0 + torch.sqrt(1 - ab) * torch.randn_like(z0))
            pz = os.path.join(d, f"z_{i:03d}.npy")
            pt = os.path.join(d, f"t_{i:03d}.npy")
            np.save(pz, zt.numpy()[None].astype(np.float32))
            tb = np.zeros((1, temb_dim, 1, 1), np.float32)
            half = temb_dim // 2
            f = np.exp(-math.log(10000) * np.arange(half, dtype=np.float32) / half)
            a = np.array([t], np.float32)[:, None] * f[None, :]
            tb[0, :half, 0, 0] = np.sin(a)[0]
            tb[0, half:, 0, 0] = np.cos(a)[0]
            np.save(pt, tb)
            lines.append(f"{pz} {pt}")
    txt = os.path.join(d, "dataset.txt")
    open(txt, "w").write("\n".join(lines) + "\n")
    print(f"[calib] {name}: {n} 个样本 → {txt}")
    return txt


def convert(onnx_p, rknn_p, calib, verbose=False):
    from rknn.api import RKNN
    r = RKNN(verbose=verbose)
    r.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
    assert r.load_onnx(model=onnx_p) == 0
    ret = r.build(do_quantization=True, dataset=calib)
    if ret != 0:
        r.release(); return ret, 0
    r.export_rknn(rknn_p)
    r.release()
    return 0, os.path.getsize(rknn_p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", default=os.path.join(HERE, "rknn"))
    ap.add_argument("--lat", type=int, default=8)
    ap.add_argument("--downs", type=int, default=2)
    ap.add_argument("--calib-n", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    ck = torch.load(a.unet, map_location="cpu", weights_only=False)
    ua = ck["args"]
    T, temb_dim, scale = ua["T"], ua["temb_dim"], ck["latent_scale"]
    v = torch.load(a.vae, map_location="cpu", weights_only=False)
    base = v["args"]["base"]
    print(f"[meta] T={T} temb_dim={temb_dim} latent_scale={scale:.4f} vae_base={base}")

    unet = LatentUNet(latent_ch=4, base=ua["base"], mult=(1, 2, 4), temb_dim=temb_dim).eval()
    unet.load_state_dict(ck["model"])
    dec = Decoder(out_ch=3, base=base, latent_ch=4, downs=a.downs).eval()
    dec.load_state_dict(v["dec"])

    _, abar = make_schedule(T)
    z = torch.zeros(1, 4, a.lat, a.lat); tb = torch.zeros(1, temb_dim, 1, 1)
    onnx_u = os.path.join(a.outdir, f"{a.name}_unet.onnx")
    onnx_d = os.path.join(a.outdir, f"{a.name}_dec.onnx")
    torch.onnx.export(unet, (z, tb), onnx_u, input_names=["z", "temb"], output_names=["eps"],
                      opset_version=13, do_constant_folding=True)
    torch.onnx.export(dec, (z,), onnx_d, input_names=["z"], output_names=["img"],
                      opset_version=13, do_constant_folding=True)

    rk_u = os.path.join(a.outdir, f"{a.name}_unet.rknn")
    rk_d = os.path.join(a.outdir, f"{a.name}_dec.rknn")
    cal_u = build_calib(a.name + "_unet", "unet", a.lat, T, temb_dim, abar, a.vae, a.calib_n)
    cal_d = build_calib(a.name + "_dec", "dec", a.lat, T, temb_dim, abar, a.vae, a.calib_n)
    ru, su = convert(onnx_u, rk_u, cal_u)
    rd, sd = convert(onnx_d, rk_d, cal_d)
    print(f"[rknn] UNet ret={ru} {su/1e6:.2f} MB     Decoder ret={rd} {sd/1e6:.2f} MB")

    meta = {"T": T, "temb_dim": temb_dim, "latent_scale": scale, "lat": a.lat,
            "vae_base": base, "unet_base": ua["base"],
            "unet_params": ck["nparam"], "dec_params": sum(p.numel() for p in dec.parameters()),
            "unet_mb": su / 1e6, "dec_mb": sd / 1e6}
    json.dump(meta, open(os.path.join(a.outdir, f"{a.name}.json"), "w"), indent=1)
    print(f"[meta] {os.path.join(a.outdir, a.name + '.json')}  {meta}")
    return 0 if (ru == 0 and rd == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
