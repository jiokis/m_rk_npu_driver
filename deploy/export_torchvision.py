#!/usr/bin/env python3
"""
export_torchvision.py — 把 torchvision 预训练模型导成 ONNX（真实权重，不是合成的）

## 为什么这一步重要

`make_onnx_models.py` 造的是**随机权重**的模型 —— 能验证「转换链路通不通」，
但验证不了「真实模型的权重分布会不会让量化出问题」。
本脚本用 `download.pytorch.org` 上的**真实预训练权重**（该域名本机可达）。

## 选模型的思路：覆盖不同的算子组合

| 模型 | 想覆盖的结构 |
|---|---|
| resnet18 | 标准残差 |
| mobilenet_v2 | 深度可分离（真实权重） |
| squeezenet1_0 | Fire 模块（1×1 + Concat） |
| shufflenet_v2_x0_5 | channel shuffle |
| mnasnet0_5 | NAS 搜索出来的混合结构 |
| efficientnet_b0 | SE + Swish |
| densenet121 | 密集连接（大量 Concat） |
| googlenet | Inception 多分支 |
| regnet_x_400mf | 分组卷积 |
| vit_b_16 | Transformer（**预期失败**，作为负面对照） |

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/export_torchvision.py                    # 默认一组
  $V deploy/export_torchvision.py --only resnet18
  $V deploy/export_torchvision.py --list             # 看默认清单
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "onnx_tv")

# (名字, 输入尺寸)  —— 尺寸按各模型的标准输入
DEFAULT_SET = [
    ("resnet18", 224),
    ("mobilenet_v2", 224),
    ("squeezenet1_0", 224),
    ("shufflenet_v2_x0_5", 224),
    ("mnasnet0_5", 224),
    ("efficientnet_b0", 224),
    ("densenet121", 224),
    ("googlenet", 224),
    ("regnet_x_400mf", 224),
    ("vit_b_16", 224),          # 预期失败：Transformer，作为负面对照
]


def export(name, size, outdir, opset=13):
    import torch
    import torchvision.models as tvm
    fn = getattr(tvm, name, None)
    if fn is None:
        raise ValueError(f"torchvision 里没有 {name}")
    # 取默认预训练权重（会从 download.pytorch.org 下载）
    weights = None
    try:
        wcls = tvm.get_model_weights(name) if hasattr(tvm, "get_model_weights") else None
        if wcls is not None:
            weights = wcls.DEFAULT
    except Exception:
        pass
    m = fn(weights=weights) if weights is not None else fn(pretrained=True)
    m.eval()
    dummy = torch.randn(1, 3, size, size)
    path = os.path.join(outdir, f"{name}.onnx")
    t0 = time.time()
    torch.onnx.export(
        m, dummy, path,
        input_names=["input"], output_names=["output"],
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    dt = time.time() - t0
    # 参数
    nparam = sum(p.numel() for p in m.parameters())
    return dt, nparam, os.path.getsize(path), (weights is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--opset", type=int, default=13)
    a = ap.parse_args()
    if a.list:
        for n, s in DEFAULT_SET:
            print(f"  {n:<22} {s}×{s}")
        return 0
    os.makedirs(a.outdir, exist_ok=True)
    todo = [(n, s) for n, s in DEFAULT_SET if not a.only or n == a.only]
    print(f"{'模型':<22} {'导出s':>7} {'参数量':>12} {'ONNX KB':>9} {'预训练':>7}")
    print("─" * 62)
    ok = 0
    for name, size in todo:
        try:
            dt, nparam, sz, pret = export(name, size, a.outdir, a.opset)
            print(f"{name:<22} {dt:>7.1f} {nparam:>12,} {sz/1024:>9.1f} "
                  f"{('是' if pret else '否'):>7}")
            ok += 1
        except Exception as e:
            print(f"{name:<22} ❌ {type(e).__name__}: {str(e)[:60]}")
    print(f"\n成功 {ok}/{len(todo)}，输出 {a.outdir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
