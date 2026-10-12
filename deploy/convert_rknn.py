#!/usr/bin/env python3
"""
convert_rknn.py — ONNX → RKNN（RK3588, w8a8）批量转换

## 环境
  /home/dunaandone/rknn-venv/bin/python     （rknn-toolkit2 2.3.2 + torch 2.4.0+cpu）
  pip 走清华源；`import RKNN` 需要 torch（即使只转 ONNX）—— 实测过。

## 为什么用「合成标定集」而不是真实图片

本机拿不到公开数据集（huggingface / github API 不通）。
对**我们自己构造的模型**（权重是随机的）来说，标定集只需要**统计分布合理**即可 ——
所以用固定种子的随机 uint8 图。这不是「随便糊弄」：转换的可复现性反而更强，
而且标定集的分布是**已知且可控**的，便于把「精度差异」归因到架构而不是数据。

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/convert_rknn.py                     # 转 deploy/onnx 下全部
  $V deploy/convert_rknn.py --only tiny_cnn     # 只转一个
  $V deploy/convert_rknn.py --no-quant          # 不量化（冒烟测试）
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ONNX = os.path.join(HERE, "onnx")
DEFAULT_OUT = os.path.join(HERE, "rknn")
# ⚠️ 标定集路径**不能有空格**：工具链按空白切分 dataset.txt 的每一行，
#    本仓库路径含空格（"orange pi"），踩过一次 —— 报错是 "Unsupport file /home/dunaandone/orange!"
DEFAULT_CAL = "/home/dunaandone/rknn-work/calib"


def make_calib(dataset_txt, size, n=20, seed=7):
    """生成标定集：n 张 size×size 的随机 uint8 图（固定种子 ⇒ 可复现）"""
    import cv2
    d = os.path.dirname(dataset_txt)
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        p = os.path.join(d, f"cal_{size}_{i:02d}.png")
        cv2.imwrite(p, img)
        paths.append(p)
    with open(dataset_txt, "w") as f:
        f.write("\n".join(paths) + "\n")
    return dataset_txt


def convert(onnx_path, out_path, cal_txt, size, quant=True, verbose=False,
            opt_level=3, target="rk3588"):
    from rknn.api import RKNN
    r = RKNN(verbose=verbose)
    ret = r.config(
        target_platform=target,
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        quantized_dtype="w8a8",
        quantized_algorithm="normal",
        optimization_level=opt_level,
    )
    if ret != 0:
        raise RuntimeError(f"config 失败 ret={ret}")
    ret = r.load_onnx(model=onnx_path)
    if ret != 0:
        raise RuntimeError(f"load_onnx 失败 ret={ret}")
    t0 = time.time()
    ret = r.build(do_quantization=quant, dataset=cal_txt if quant else None)
    if ret != 0:
        raise RuntimeError(f"build 失败 ret={ret}")
    dt = time.time() - t0
    ret = r.export_rknn(out_path)
    if ret != 0:
        raise RuntimeError(f"export_rknn 失败 ret={ret}")
    r.release()
    return dt, os.path.getsize(out_path)


def rknn_header(path):
    """读 RKNN 文件头：magic + version（v6 = 当前运行时接受的格式，docs 里 mobilenet_ssd 是 v1）"""
    with open(path, "rb") as f:
        b = f.read(16)
    magic = b[:4]
    # ⚠️ 版本号在 offset 8（不是 4）：yolov5s 是 06 00 00 00，mobilenet_ssd 是 01 00 00 00
    ver = int.from_bytes(b[8:12], "little")
    return magic, ver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--opt-level", type=int, default=3)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cal = make_calib(os.path.join(DEFAULT_CAL, "dataset.txt"), a.size)
    files = sorted(f for f in os.listdir(a.onnx) if f.endswith(".onnx"))
    if a.only:
        files = [f for f in files if f[:-5] == a.only]
    if not files:
        print("没找到 .onnx"); return 1

    print(f"目标平台 rk3588 · 量化 {'关' if a.no_quant else 'w8a8'} · 标定集 {cal}")
    print(f"{'模型':<12} {'build(s)':>9} {'ONNX KB':>9} {'RKNN KB':>9} {'头':>6} {'版本':>5}")
    print("─" * 60)
    ok = 0
    for f in files:
        name = f[:-5]
        src = os.path.join(a.onnx, f)
        dst = os.path.join(a.out, name + ".rknn")
        try:
            dt, sz = convert(src, dst, cal, a.size, quant=not a.no_quant,
                             verbose=a.verbose, opt_level=a.opt_level)
            magic, ver = rknn_header(dst)
            print(f"{name:<12} {dt:>9.1f} {os.path.getsize(src)/1024:>9.1f} "
                  f"{sz/1024:>9.1f} {magic.decode('latin1'):>6} {ver:>5}")
            ok += 1
        except Exception as e:
            print(f"{name:<12} ❌ {type(e).__name__}: {e}")
    print(f"\n成功 {ok}/{len(files)}，输出目录 {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
