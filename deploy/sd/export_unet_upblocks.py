#!/usr/bin/env python3
"""export_unet_upblocks.py — 把 UNet 的 **up_blocks 逐个拆开**导出（每块一段）

## 依据（来自逆向出来的工具链内存统计）

`RKNNMemStatisticsPass` 在 build 日志里报：

```
Total Internal Memory Size: 17046.8KB     ← 激活只有 17 MB
Total Weight Memory Size:  825668KB       ← **权重 825 MB**
```

⇒ 512 档那笔失败的 869 MB 分配是**权重**，不是激活。
⇒ **按参数量拆**就能绕开：up_blocks[0..3] 各自成段，最大的一段权重 ~300 MB。

## skip 分配（latent=64 实测）

`up[i]` 消费 skip 的**倒数第 (4-i) 组 3 条**：
`up[0]`←skip[9..11]  `up[1]`←skip[6..8]  `up[2]`←skip[3..5]  `up[3]`←skip[0..2]
"""
from __future__ import annotations
import argparse, json, os, sys
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_unet_parts import load_unet, P1, P2, OUTDIR   # noqa: E402


class UpOne(nn.Module):
    """只跑 up_blocks[idx]，输入/输出都是中间特征"""
    def __init__(self, u, idx, tail=False):
        super().__init__(); self.u = u; self.idx = idx; self.tail = tail
    def forward(self, h, emb, ehs, *skips):
        b = self.u.up_blocks[self.idx]
        s = b(h, list(skips), emb, encoder_hidden_states=ehs)
        if self.tail:
            s = self.u.conv_out(self.u.conv_act(self.u.conv_norm_out(s)))
        return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if a.dtype == "fp16" else torch.float32
    u = load_unet(dt, dev); L = a.latent
    s = torch.zeros(1, 4, L, L, dtype=dt, device=dev)
    t = torch.zeros(1, dtype=dt, device=dev); e = torch.zeros(1, 77, 1024, dtype=dt, device=dev)
    with torch.no_grad():
        o1 = P1(u, 2).eval()(s, t, e); o2 = P2(u, 2).eval()(o1[0], o1[1], e)
    sk = list(o1[2:]) + list(o2[1:])          # 12 条：64,64,64,32,32,32,16,16,16,8,8,8
    h, emb = o2[0], o1[1]
    n_up = len(u.up_blocks)
    print(f"[up] mid{tuple(h.shape)}  12 条 skip  {n_up} 个 up_block")
    man = []
    with torch.no_grad():
        for i in range(n_up):
            # up[i] 消费 skip[ (n_up-1-i)*3 : (n_up-i)*3 ]
            lo = (n_up - 1 - i) * 3
            sub = sk[lo:lo + 3]
            m = UpOne(u, i, tail=(i == n_up - 1)).eval()
            name = f"unet_up{i}"
            y = m(h, emb, e, *sub)
            print(f"[up] {name}: 消费 skip[{lo}..{lo+2}]  {tuple(h.shape)} → {tuple(y.shape)}")
            p = os.path.join(OUTDIR, name + ".onnx")
            torch.onnx.export(m, (h, emb, e) + tuple(sub), p,
                              input_names=["h", "emb", "ehs"] + [f"skip{j}" for j in range(3)],
                              output_names=["out"], opset_version=17,
                              do_constant_folding=True, dynamo=False)
            print(f"[onnx] {name}  {os.path.getsize(p)/1e6:.0f} MB")
            man.append({"name": name, "onnx": p,
                        "inputs": ["h", "emb", "ehs"] + [f"skip{j}" for j in range(3)],
                        "outputs": ["out"], "skip_slice": [lo, lo + 3]})
            h = y
    with torch.no_grad():
        ref = u(s, t, encoder_hidden_states=e).sample
    cos = float((h.ravel() @ ref.ravel()) / (h.ravel().norm() * ref.ravel().norm() + 1e-12))
    print(f"[check] P1+P2+up0..up{n_up-1} vs 完整 UNet 余弦 = {cos:.6f}")
    json.dump({"latent": L, "dtype": a.dtype, "parts": man},
              open(os.path.join(OUTDIR, "upblocks_manifest.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
