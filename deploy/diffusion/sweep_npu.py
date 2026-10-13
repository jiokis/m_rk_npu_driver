#!/usr/bin/env python3
"""sweep_npu.py — 扫描「SD 架构各组件在不同规模下」的 **NPU 实测耗时**，找甜点

## 为什么可以不训练就扫

**计时只需要结构，不需要权重。** 随机初始化的模型转成 RKNN 后的耗时与训练好的
几乎一样（同样的算子图）。⇒ **可以在不花几小时训练的前提下，量出 NPU 的规模上限。**

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/diffusion/sweep_npu.py --build          # 生成并转换所有配置
  $V deploy/diffusion/sweep_npu.py --report         # 打印本地 FLOPs 表
  # 然后 bash deploy/diffusion/sweep_board.sh       # 拷到板子统一计时
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

OUT = os.path.join(HERE, "sweep")
CALIB = "/home/dunaandone/rknn-work/diffcalib2"   # ⚠️ 不能含空格

#: (名字, 图像边长, 潜空间边长, 类型, 构造参数)
CONFIGS = []
for res, lat in ((64, 8), (128, 16), (256, 32), (512, 64)):
    for base in (128, 256):
        CONFIGS.append((f"unet_{res}_b{base}", res, lat, "unet",
                        dict(latent_ch=4, base=base, mult=(1, 2, 4), temb_dim=128)))
for res, lat in ((64, 8), (128, 16), (256, 32), (512, 64)):
    for base in (32, 64):
        CONFIGS.append((f"dec_{res}_b{base}", res, lat, "dec",
                        dict(out_ch=3, base=base, latent_ch=4)))


def build_model(kind: str, cfg: dict):
    return LatentUNet(**cfg) if kind == "unet" else Decoder(**cfg)


def make_calib(name: str, kind: str, lat: int, temb_dim: int = 128, n: int = 16):
    d = os.path.join(CALIB, name)
    os.makedirs(d, exist_ok=True)
    rng = np.random.RandomState(0)
    lines = []
    for i in range(n):
        if kind == "unet":
            z = os.path.join(d, f"z_{i:02d}.npy"); t = os.path.join(d, f"t_{i:02d}.npy")
            np.save(z, rng.randn(1, 4, lat, lat).astype(np.float32))
            np.save(t, rng.randn(1, temb_dim, 1, 1).astype(np.float32))
            lines.append(f"{z} {t}")
        else:
            z = os.path.join(d, f"z_{i:02d}.npy")
            np.save(z, rng.randn(1, 4, lat, lat).astype(np.float32))
            lines.append(z)
    txt = os.path.join(d, "dataset.txt")
    open(txt, "w").write("\n".join(lines) + "\n")
    return txt


def do_build(quant: bool):
    from rknn.api import RKNN
    os.makedirs(OUT, exist_ok=True)
    table = {}
    for name, res, lat, kind, cfg in CONFIGS:
        m = build_model(kind, cfg).eval()
        nparam = sum(p.numel() for p in m.parameters())
        if kind == "unet":
            args = (torch.zeros(1, 4, lat, lat), torch.zeros(1, cfg["temb_dim"], 1, 1))
            names = ["z", "temb"]
        else:
            args = (torch.zeros(1, 4, lat, lat),)
            names = ["z"]
        macs = count_macs(m, *[tuple(a.shape) for a in args])
        onnx_p = os.path.join(OUT, name + ".onnx")
        torch.onnx.export(m, args, onnx_p, input_names=names, output_names=["y"],
                          opset_version=13, do_constant_folding=True)
        cal = make_calib(name, kind, lat, cfg.get("temb_dim", 128)) if quant else None
        r = RKNN(verbose=False)
        r.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8")
        r.load_onnx(model=onnx_p)
        ret = r.build(do_quantization=bool(cal), dataset=cal)
        rk = os.path.join(OUT, name + ".rknn")
        if ret == 0:
            r.export_rknn(rk)
        r.release()
        table[name] = {"res": res, "lat": lat, "kind": kind, "base": cfg["base"],
                       "params": nparam, "macs": macs, "gflops": 2 * macs / 1e9,
                       "rknn_bytes": os.path.getsize(rk) if os.path.exists(rk) else 0,
                       "build_ret": ret}
        print(f"  {name:20s} {nparam/1e6:7.2f}M  {2*macs/1e9:7.3f} GFLOPs  "
              f"rknn {table[name]['rknn_bytes']/1e6:6.2f} MB  ret={ret}", flush=True)
    json.dump(table, open(os.path.join(OUT, "table.json"), "w"), indent=1)
    print(f"[sweep] {len(table)} 个配置 → {OUT}/table.json")


def report():
    import json as J
    t = J.load(open(os.path.join(OUT, "table.json")))
    print(f"{'name':20s} {'params':>9s} {'GFLOPs':>8s} {'rknn MB':>8s}")
    for k, v in t.items():
        print(f"{k:20s} {v['params']/1e6:8.2f}M {v['gflops']:8.3f} {v['rknn_bytes']/1e6:8.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-quant", action="store_true")
    a = ap.parse_args()
    if a.build:
        do_build(not a.no_quant)
    elif a.report:
        report()
    else:
        print(__doc__)
