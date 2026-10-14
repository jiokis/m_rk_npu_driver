#!/usr/bin/env python3
"""pressure_test.py — 逐级加压：**端到端**（潜空间 DDIM + VAE 解码）的实时性边界

## 为什么可以不用训练

计时只取决于**算子图**。⇒ 随机权重转成的 RKNN，耗时与训练好的几乎一样。
⇒ 于是可以在**不花几小时训练**的前提下，量出「板上实时性的边界」。

## 加压维度

* **分辨率**：64 / 128 / 256（潜空间 = 分辨率 / 8）
* **通道数**：UNet base=128 / 256

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/diffusion/pressure_test.py --build      # 生成 + 转换（笔记本）
  bash deploy/diffusion/pressure_board.sh           # 板上端到端计时
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from latent_arch import Decoder, LatentUNet, count_macs  # noqa: E402

OUT = os.path.join(HERE, "pressure")
CALIB = "/home/dunaandone/rknn-work/difcalib4"     # ⚠️ 不能含空格

CONFIGS = []
for res in (64, 128, 256):
    for base in (128, 256):
        CONFIGS.append(dict(name=f"p_unet_{res}_b{base}", kind="unet", res=res,
                            lat=res // 8, base=base))
for res in (64, 128, 256):
    for base in (32, 64):
        CONFIGS.append(dict(name=f"p_dec_{res}_b{base}", kind="dec", res=res,
                            lat=res // 8, base=base))


def calib_for(c, n=16):
    d = os.path.join(CALIB, c["name"])
    os.makedirs(d, exist_ok=True)
    rng = np.random.RandomState(0)
    lines = []
    for i in range(n):
        z = os.path.join(d, f"z_{i:02d}.npy")
        np.save(z, rng.randn(1, 4, c["lat"], c["lat"]).astype(np.float32))
        if c["kind"] == "unet":
            t = os.path.join(d, f"t_{i:02d}.npy")
            np.save(t, rng.randn(1, 128, 1, 1).astype(np.float32))
            lines.append(f"{z} {t}")
        else:
            lines.append(z)
    txt = os.path.join(d, "dataset.txt")
    open(txt, "w").write("\n".join(lines) + "\n")
    return txt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        t = json.load(open(os.path.join(OUT, "table.json")))
        for k, v in t.items():
            print(f"{k:22s} {v['params']/1e6:7.2f}M {v['gflops']:8.3f} GFLOPs")
        return 0
    if not a.build:
        print(__doc__); return 1

    from rknn.api import RKNN
    os.makedirs(OUT, exist_ok=True)
    table = {}
    for c in CONFIGS:
        m = (LatentUNet(latent_ch=4, base=c["base"], mult=(1, 2, 4), temb_dim=128)
             if c["kind"] == "unet" else
             Decoder(out_ch=3, base=c["base"], latent_ch=4, downs=3)).eval()
        nparam = sum(p.numel() for p in m.parameters())
        if c["kind"] == "unet":
            args = (torch.zeros(1, 4, c["lat"], c["lat"]), torch.zeros(1, 128, 1, 1))
            names = ["z", "temb"]
        else:
            args = (torch.zeros(1, 4, c["lat"], c["lat"]),)
            names = ["z"]
        macs = count_macs(m, *[tuple(x.shape) for x in args])
        onnx_p = os.path.join(OUT, c["name"] + ".onnx")
        torch.onnx.export(m, args, onnx_p, input_names=names, output_names=["y"],
                          opset_version=13, do_constant_folding=True)
        cal = calib_for(c)
        r = RKNN(verbose=False)
        r.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
        r.load_onnx(model=onnx_p)
        ret = r.build(do_quantization=True, dataset=cal)
        rk = os.path.join(OUT, c["name"] + ".rknn")
        if ret == 0:
            r.export_rknn(rk)
        r.release()
        table[c["name"]] = {**{k: v for k, v in c.items() if k != "name"},
                            "params": nparam, "gflops": 2 * macs / 1e9,
                            "mb": (os.path.getsize(rk) / 1e6) if os.path.exists(rk) else 0,
                            "ret": ret}
        print(f"  {c['name']:22s} {nparam/1e6:7.2f}M {2*macs/1e9:8.3f} GFLOPs "
              f"{table[c['name']]['mb']:7.2f} MB ret={ret}", flush=True)
    json.dump(table, open(os.path.join(OUT, "table.json"), "w"), indent=1)
    print(f"[pressure] {len(table)} 个配置 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
