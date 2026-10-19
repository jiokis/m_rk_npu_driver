#!/usr/bin/env python3
"""export_clip_noembed.py — CLIP 文本塔，**词嵌入查表挪到 numpy 做**

## 为什么

`clip_p0` 里有目标类型 **INT64** 的 `Cast`（来自 `x.long()`），
RKNN 处理不了 int64 索引的 Gather ⇒ **查表结果全错** ⇒ 整条 CLIP 输出余弦只有 **0.26**。
（实测：`clip_p0.onnx` 里 `/Cast` 的 target 就是 INT64）

## 做法

* 词嵌入表 + 位置嵌入表**导出成 .npy**（板上一行 numpy 就能查表）
* 三段 CLIP 都只吃 **已嵌入的 `[1,77,1024]`**，接口统一
* ⇒ 模型里**不再有整数 Cast**，也不再带 49408×1024 的大表

## 产物

```
parts/clipn_p0.onnx   [1,77,1024] → [1,77,1024]   层 0..7
parts/clipn_p1.onnx   层 8..15
parts/clipn_p2.onnx   层 16..23 + ln_final
clip_tok_emb.npy      词嵌入表 (49408,1024) fp16
clip_pos_emb.npy      位置嵌入 (77,1024) fp16
```
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_unet_parts import OUTDIR   # noqa: E402
from export_clip_parts import build, TCFG  # noqa: E402

# ⚠️ 必须关掉 MHA 快速路径（否则 `aten::_native_multi_head_attention` 无法导出）
try:
    torch.backends.mha.set_fastpath_enabled(False)
except Exception:
    pass


class Part(nn.Module):
    """吃**已嵌入的** `[B,77,C]`，跑第 lo..hi 层（tail 时接 ln_final）

    ★★★★★ 两个必须做对的点（实测判据）：

    1. **不要转置**：`open_clip` 的 `Transformer.batch_first = True`
       ⇒ 张量保持 `[B,77,C]`。我原来按 `[L,N,D]` 做 `permute(1,0,2)`
       ⇒ 报 `attn_mask ... torch.Size([77,77]) but should be (1,1)`，或静默算错（余弦 0.266）
    2. **必须传因果掩码**：CLIP 文本塔是**因果注意力**。
       实测：原样+掩码 **1.000000**，原样无掩码 **0.253506**。

    掩码是固定的 77×77，作为常量 buffer 烘进子模块。
    """

    def __init__(self, t, lo, hi, tail=False):
        super().__init__()
        self.t = t; self.lo, self.hi, self.tail = lo, hi, tail
        am = getattr(t, "attn_mask", None)
        if am is not None:
            self.register_buffer("am", am.clone())

    def forward(self, x):
        am = getattr(self, "am", None)
        for blk in self.t.transformer.resblocks[self.lo:self.hi]:
            x = blk(x, attn_mask=am) if am is not None else blk(x)
        if self.tail:
            x = self.t.ln_final(x)
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

    # ① 嵌入表存成 .npy（板上查表）
    tok = t.token_embedding.weight.detach().float().numpy().astype(np.float16)
    pos = t.positional_embedding.detach().float().numpy().astype(np.float16)
    np.save(os.path.join(OUTDIR, "clip_tok_emb.npy"), tok)
    np.save(os.path.join(OUTDIR, "clip_pos_emb.npy"), pos)
    print(f"[emb] 词嵌入 {tok.shape} 位置嵌入 {pos.shape} → .npy（板上一行 numpy 查表）")

    x = torch.zeros(1, 77, 1024)
    man = []
    with torch.no_grad():
        for i, (lo, hi) in enumerate(bounds):
            m = Part(t, lo, hi, tail=(i == len(bounds) - 1)).eval()
            y = m(x)
            p = os.path.join(OUTDIR, f"clipn_p{i}.onnx")
            torch.onnx.export(m, (x,), p, input_names=["x"], output_names=["y"],
                              opset_version=17, do_constant_folding=True, dynamo=False)
            print(f"[onnx] clipn_p{i} 层 {lo}..{hi-1}  {os.path.getsize(p)/1e6:.0f} MB")
            man.append({"name": f"clipn_p{i}", "onnx": p, "inputs": ["x"], "outputs": ["y"]})
            x = y
    json.dump({"parts": man, "layers": n}, open(os.path.join(OUTDIR, "clipn_manifest.json"), "w"), indent=1)
    print("[done] 清单 → clipn_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
