#!/usr/bin/env python3
"""export_clip_parts.py — 把 CLIP 文本塔**按层切成 3 段**分别导出（绕开 7 GB 内存）

## 为什么拆

整只文本塔（354 M 参数，ONNX 1413 MB）转 RKNN 峰值 **6.59 GB** ⇒ 7 GB 的 WSL 上 OOM。
**切成 3 段（每段 8 层，~120 M）⇒ 每段峰值 ~2 GB。**

## 比 UNet 简单得多

文本塔**没有 skip 连接**，接口就是 `[B,77,1024] → [B,77,1024]`，串起来即可。

```
P1: ids            → x    (词嵌入 + 位置嵌入 + 层 0..7)
P2: x              → x    (层 8..15)
P3: x              → ehs  (层 16..23 + ln_final)
```
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

# ★★★★★ 必须关掉 MHA 的「快速路径」：
#   open_clip 的 `resblocks` 用的是 `nn.MultiheadAttention`，在 torch 2.x 下会走
#   `aten::_native_multi_head_attention`，而**该算子无法导出到 ONNX**：
#     `UnsupportedOperatorError: Exporting the operator 'aten::_native_multi_head_attention'
#      to ONNX opset version 17 is not supported`（踩过）
#   （完整塔那次没触发，是因为走的是 `TextTransformer.forward` 的另一条分支；
#     这里直接调 resblock 就暴露了。）
try:
    torch.backends.mha.set_fastpath_enabled(False)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "/home/dunaandone/rknn-work/sd/sd_turbo.safetensors"
OUTDIR = os.path.join(HERE, "parts")
PRE = "conditioner.embedders.0.model."
TCFG = dict(context_length=77, vocab_size=49408, width=1024, heads=16, layers=24)


def build():
    from open_clip.transformer import TextTransformer
    from safetensors import safe_open
    t = TextTransformer(**TCFG, mlp_ratio=4.0, output_dim=1024, pool_type="last",
                        proj_type="linear", act_layer=nn.GELU, output_tokens=True)
    sd = {}
    with safe_open(CKPT, framework="pt") as f:
        for k in f.keys():
            if k.startswith(PRE):
                kk = k[len(PRE):]
                if kk.startswith(("transformer.", "token_embedding.",
                                  "positional_embedding", "ln_final.")):
                    sd[kk] = f.get_tensor(k).to(torch.float32)
    missing, unexpected = t.load_state_dict(sd, strict=False)
    miss = [x for x in missing if "text_projection" not in x]
    print(f"[clip] 载入 {len(sd)} 项  非 projection 的 missing {len(miss)}  unexpected {len(unexpected)}")
    return t.eval()


class Part(nn.Module):
    """第 lo..hi 层（含 lo，不含 hi）；`head=True` 时前面加词/位置嵌入，`tail=True` 时后面加 ln_final"""

    def __init__(self, t, lo, hi, head=False, tail=False):
        super().__init__()
        self.t = t; self.lo, self.hi = lo, hi
        self.head, self.tail = head, tail

    def forward(self, x):
        t = self.t
        if self.head:                                   # x 此时是 ids
            x = t.token_embedding(x.long()) + t.positional_embedding
        x = x.permute(1, 0, 2)                          # [B,77,C] → [77,B,C]
        # ★★★★★ **必须传因果掩码**！CLIP 的文本塔是**因果注意力**，
        #   `TextTransformer.forward` 会算好 `attn_mask` 并传给每个 resblock；
        #   手写循环漏了它 ⇒ 输出与真正 CLIP 完全不同
        #   （实测：串接结果与 torch 手写版余弦 0.9999，但与**真正的 ehs** 只有 **0.2657**）。
        am = getattr(t, "attn_mask", None)
        for blk in t.transformer.resblocks[self.lo:self.hi]:
            x = blk(x, attn_mask=am) if am is not None else blk(x)
        x = x.permute(1, 0, 2)                          # → [B,77,C]
        if self.tail:
            x = t.ln_final(x)
        return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    t = build()
    n = len(t.transformer.resblocks)
    step = (n + a.parts - 1) // a.parts
    bounds = [(i, min(i + step, n)) for i in range(0, n, step)]
    print(f"[clip] 共 {n} 层 ⇒ {len(bounds)} 段 {bounds}")

    ids = torch.zeros(1, 77, dtype=torch.float32)
    x = ids
    man = []
    with torch.no_grad():
        for i, (lo, hi) in enumerate(bounds):
            m = Part(t, lo, hi, head=(i == 0), tail=(i == len(bounds) - 1)).eval()
            y = m(x)
            print(f"[p{i}] 层 {lo}..{hi-1}  输入 {tuple(x.shape)} → 输出 {tuple(y.shape)}")
            p = os.path.join(OUTDIR, f"clip_p{i}.onnx")
            torch.onnx.export(m, (x,), p, input_names=["x"], output_names=["y"],
                              opset_version=17, do_constant_folding=True, dynamo=False)
            print(f"[onnx] clip_p{i}  {os.path.getsize(p)/1e6:.0f} MB")
            man.append({"name": f"clip_p{i}", "onnx": p, "layers": [lo, hi]})
            x = y
    # 与完整塔对照
    with torch.no_grad():
        ref = t(ids)
        ref = ref[1] if isinstance(ref, (tuple, list)) else ref
    cos = float((x.ravel() @ ref.ravel()) / (x.ravel().norm() * ref.ravel().norm() + 1e-12))
    print(f"[check] 三段串起来 vs 完整文本塔 余弦 = {cos:.6f}")
    json.dump({"parts": man, "layers": n}, open(os.path.join(OUTDIR, "clip_manifest.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
