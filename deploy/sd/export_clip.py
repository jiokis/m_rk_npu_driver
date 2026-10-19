#!/usr/bin/env python3
"""export_clip.py — SD 的 **CLIP 文本塔** → ONNX → RKNN（**只建文本塔**）

## 为什么不能直接用 `open_clip.create_model('ViT-H-14')`

它会**同时建视觉塔**（32 层 × 1280 宽，~600 M 参数）⇒ 本机 7 GB 内存直接 OOM
（实测 `anon-rss:6740192kB` 被 kill）。
**⇒ 改用 `TextTransformer` 只建文本塔**（24 层 × 1024，~123 M）—— 内存减到 1/5。

## 参数名与检查点**逐字对应**（不需要任何名字转换）

`conditioner.embedders.0.model.` 后面就是：`transformer.*` / `token_embedding.*` /
`ln_final.*` / `positional_embedding` / `text_projection` / `logit_scale`

## 输出什么

SD 的 cross-attention 要 **`last_hidden_state` [1,77,1024]**，
**不是** `encode_text` 的池化投影 ⇒ forward 手写到 `ln_final` 为止。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

CKPT = "/home/dunaandone/rknn-work/sd/sd_turbo.safetensors"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rknn")
CALIB = "/home/dunaandone/rknn-work/sdcalib_clip"
PRE = "conditioner.embedders.0.model."

TCFG = dict(context_length=77, vocab_size=49408, width=1024, heads=16, layers=24)


class TextTower(nn.Module):
    """★ **直接用 `TextTransformer`** —— 它自带 `token_embedding` / `positional_embedding` /
    `ln_final` / `transformer.resblocks.*`，与检查点命名**逐字对应**。

    ⚠️ 不要再外面再包一层 `token_embedding`/`ln_final`：那样会变成
    `TextTransformer.token_embedding.*`，与检查点的 `token_embedding.*` 对不上，
    报 `missing 293 / unexpected 288`（踩过）。

    输出：`TextTransformer` 返回 `[77,B,1024]`（序列在前）⇒ 转成 `[B,77,1024]`
    —— 这是 SD cross-attention 要的 `encoder_hidden_states`。
    """

    def __init__(self):
        super().__init__()
        from open_clip.transformer import TextTransformer
        # ★ 必须 `output_tokens=True`：否则 `TextTransformer` 返回**池化后**的
        #   [B,1024]，而 SD 的 cross-attention 要的是**整条序列** [B,77,1024]（踩过）
        self.t = TextTransformer(**TCFG, mlp_ratio=4.0, output_dim=1024,
                                 pool_type="last", proj_type="linear",
                                 act_layer=nn.GELU, output_tokens=True)

    def forward(self, ids):
        out = self.t(ids.long())
        # ★ `output_tokens=True` 时 `TextTransformer` 返回 **(tokens, pooled) 元组**
        #   （踩过：直接 `.dim()` 会报 `'tuple' object has no attribute 'dim'`）
        # ★ 实测元组是 **(pooled [B,1024], tokens [B,77,1024])** —— 要的是**后者**
        #   （踩过：取 out[0] 会得到池化结果，cross-attention 的维度就错了）
        if isinstance(out, (tuple, list)):
            x = next((o for o in out if getattr(o, "dim", lambda: 0)() == 3), out[0])
        elif isinstance(out, dict):
            x = out["x"]
        else:
            x = out
        if x.dim() == 3 and x.shape[0] == TCFG["context_length"]:
            x = x.permute(1, 0, 2)                   # [77,B,1024] → [B,77,1024]
        return x


def build():
    from safetensors import safe_open
    m = TextTower()
    sd = {}
    with safe_open(CKPT, framework="pt") as f:
        for k in f.keys():
            if k.startswith(PRE):
                kk = k[len(PRE):]
                # 只要文本塔的键（丢掉 text_projection / logit_scale）
                if kk.startswith(("transformer.", "token_embedding.",
                                  "positional_embedding", "ln_final.")):
                    sd["t." + kk] = f.get_tensor(k).to(torch.float32)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[clip] 载入 {len(sd)} 项  参数 {sum(p.numel() for p in m.parameters())/1e6:.1f} M")
    print(f"[clip] missing {len(missing)}  unexpected {len(unexpected)}")
    if missing[:4]: print("   missing:", missing[:4])
    if unexpected[:4]: print("   unexpected:", unexpected[:4])
    return m.eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    m = build()
    ids = torch.zeros(1, 77, dtype=torch.float32)
    with torch.no_grad():
        y = m(ids)
    print(f"[clip] 输入 {tuple(ids.shape)} → 输出 {tuple(y.shape)}")

    # 与 open_clip 的参考实现对照（只跑文本塔，不建视觉塔）
    onnx_p = os.path.join(OUTDIR, "clip_text.onnx")
    torch.onnx.export(m, (ids,), onnx_p, input_names=["input_ids"], output_names=["ehs"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"[onnx] {onnx_p}  {os.path.getsize(onnx_p)/1e6:.0f} MB")

    import json
    json.dump({"onnx": onnx_p, "quant": a.quant}, open(os.path.join(OUTDIR, "clip_export.json"), "w"))
    print("[onnx] 导出完成；RKNN 转换请用 rknn-venv 跑 to_rknn_clip.py")
    return 0

    from rknn.api import RKNN
    r = RKNN(verbose=False)
    r.config(target_platform="rk3588", optimization_level=3, quantized_dtype="w8a8",
             disable_rules=["fuse_mul_into_sdpa"])
    assert r.load_onnx(model=onnx_p) == 0
    cal = None
    if a.quant:
        d = CALIB; os.makedirs(d, exist_ok=True)
        rng = np.random.RandomState(0); lines = []
        for i in range(8):
            p = os.path.join(d, f"ids_{i:02d}.npy")
            v = np.zeros((1, 77), np.float32)
            n = int(rng.randint(4, 20))
            v[0, 0] = 49406
            v[0, 1:1 + n] = rng.randint(100, 40000, n)
            v[0, 1 + n] = 49407
            np.save(p, v); lines.append(p)
        cal = os.path.join(d, "dataset.txt"); open(cal, "w").write("\n".join(lines) + "\n")
    ret = r.build(do_quantization=cal is not None, dataset=cal)
    rk = os.path.join(OUTDIR, "clip_text.rknn")
    if ret == 0:
        r.export_rknn(rk)
        print(f"[rknn] ✅ {rk}  {os.path.getsize(rk)/1e6:.0f} MB")
    else:
        print(f"[rknn] ❌ build ret={ret}")
    r.release()
    return 0 if ret == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
