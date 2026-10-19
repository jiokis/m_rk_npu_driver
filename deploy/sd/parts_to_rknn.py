#!/usr/bin/env python3
"""parts_to_rknn.py — 把拆分后的 UNet 子图逐个转 RKNN

## 为什么这样能绕开 OOM

`rknn-toolkit2` 转换时的峰值内存约是 ONNX 体积的数倍。
整只 UNet 1652 MB ⇒ 峰值 ~6.5 GB ⇒ 在 7 GB 的 WSL 上被 OOM kill。
拆成 96 / 575 / 981 MB 三段后，**最大的一段峰值 ~4 GB**，能过。

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/sd/parts_to_rknn.py --manifest deploy/sd/parts/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# ★★★★★ **`fuse_mul_into_sdpa` 不需要再禁了**（2026-10 的重要发现）
#
#   这条规则以前必须禁，否则含 attention 的模型 build 会抛
#     `TypeError: only 0-dimensional arrays can be converted to Python scalars`
#   位置在 `rknn/api/rules/matmul.py:1633`。
#
#   **根因不是规则的逻辑，而是 numpy 版本**：rknn-toolkit2 2.3.2 内部做了
#   `float(np.array([x]))`（ndim>0 的数组转标量），这在 **numpy 1.x 只是 DeprecationWarning，
#   numpy 2.x 变成硬错误**。而 venv 里当时是 numpy **2.5.4**。
#
#   ⇒ 把 numpy 钉到 **1.26.4** 后，规则**正常工作**，含 attention 的模型
#     （unet_up1 / clipn_p0 / vae512_v1 …）带规则全部 `build=0` ✅
#   ⇒ 于是 **SDPA 融合可以打开**（attention 效率的关键），不再需要这个 workaround。
#
#   实证：最小 attention 图在 numpy 2.x 下**带规则也能过**（因为形状刚好是标量），
#   真实模型里 attention 的 scale 是 1 元素数组 ⇒ 才触发。这个「最小图不复现」
#   正是它长期没被识破的原因。
DISABLE_RULES: list[str] = []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "parts", "manifest.json"))
    ap.add_argument("--only", default=None, help="只转某一个 part")
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--no-quant", action="store_true")
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    from rknn.api import RKNN
    for part in man["parts"]:
        if a.only and part["name"] != a.only:
            continue
        src = part["onnx"]
        if not os.path.isabs(src) and not os.path.exists(src):
            src = os.path.join(os.path.dirname(os.path.dirname(HERE)), src)
        dst = os.path.join(HERE, "rknn", part["name"] + ".rknn")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"\n=== {part['name']}  {os.path.getsize(src)/1e6:.0f} MB  "
              f"输入 {len(part.get('inputs', []))} 输出 {len(part.get('outputs', []))} ===", flush=True)
        r = RKNN(verbose=False)
        r.config(target_platform="rk3588", optimization_level=a.opt_level,
                 quantized_dtype="w8a8", disable_rules=DISABLE_RULES)
        ret = r.load_onnx(model=src)
        print(f"[{part['name']}] load_onnx ret={ret}", flush=True)
        if ret != 0:
            r.release(); continue
        t0 = time.time()
        ret = r.build(do_quantization=False)          # 先验支持性（不量化）
        print(f"[{part['name']}] build ret={ret}  {time.time()-t0:.0f}s", flush=True)
        if ret == 0:
            r.export_rknn(dst)
            print(f"[{part['name']}] ✅ {dst}  {os.path.getsize(dst)/1e6:.0f} MB", flush=True)
        r.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
