#!/usr/bin/env python3
"""export_vae_split.py — 把 SD 的 **VAE 解码器**拆成 2 段（512 档转换会 OOM）

## 结构（**没有 skip 连接**，比 UNet 好拆）

```
V1: z → post_quant_conv → decoder.conv_in → mid_block → up_blocks[0..1]  → h
V2: h → up_blocks[2..3] → conv_out                                        → img
```
`1/scaling_factor` 烘在 V1 里（与整只导出时一致）。
"""
from __future__ import annotations
import argparse, json, os, sys
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_unet_parts import OUTDIR   # noqa: E402

VAE_PT = "/home/dunaandone/rknn-work/sd/vae_sd.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--n-up", type=int, default=2)
    a = ap.parse_args()
    from diffusers import AutoencoderKL
    ck = torch.load(VAE_PT, map_location="cpu", weights_only=False)
    vae = AutoencoderKL.from_config(ck["config"]); vae.load_state_dict(ck["state_dict"]); vae.eval()
    sf = float(vae.config.scaling_factor)
    d = vae.decoder

    class V1(nn.Module):
        def __init__(self):
            super().__init__()
            self.post = vae.post_quant_conv; self.conv_in = d.conv_in
            self.mid = d.mid_block; self.ups = d.up_blocks[:a.n_up]
            self.sf = sf
        def forward(self, z):
            h = self.conv_in(self.post(z / self.sf))
            h = self.mid(h)
            for u in self.ups:
                h = u(h)
            return h

    class V2(nn.Module):
        """★★★ **必须带 `conv_norm_out` + `conv_act`**！

        diffusers 的 `Decoder.forward` 收尾是
        `up_blocks → conv_norm_out(GroupNorm) → conv_act(SiLU) → conv_out`。
        我第一版只写了 `conv_out` ⇒ 两段拼起来与整只的余弦只有 **0.465**，
        解码出来全是高频噪声（踩过）。
        """
        def __init__(self):
            super().__init__()
            self.ups = d.up_blocks[a.n_up:]
            self.norm = d.conv_norm_out
            self.act = d.conv_act
            self.out = d.conv_out
        def forward(self, h):
            for u in self.ups:
                h = u(h)
            return self.out(self.act(self.norm(h)))

    m1, m2 = V1().eval(), V2().eval()
    z = torch.zeros(1, 4, a.latent, a.latent)
    with torch.no_grad():
        h = m1(z); img = m2(h)
        ref = vae.decode(z / sf).sample
    cos = float((img.ravel() @ ref.ravel()) / (img.ravel().norm() * ref.ravel().norm() + 1e-12))
    print(f"[vae] {tuple(z.shape)} → h{tuple(h.shape)} → img{tuple(img.shape)}  两段 vs 整只余弦 {cos:.6f}")
    os.makedirs(OUTDIR, exist_ok=True)
    man = []
    for m, args, inames, onames, name in ((m1, (z,), ["z"], ["h"], "vae512_v1"),
                                          (m2, (h,), ["h"], ["img"], "vae512_v2")):
        p = os.path.join(OUTDIR, name + ".onnx")
        torch.onnx.export(m, args, p, input_names=inames, output_names=onames,
                          opset_version=17, do_constant_folding=True, dynamo=False)
        print(f"[onnx] {name}  {os.path.getsize(p)/1e6:.0f} MB")
        man.append({"name": name, "onnx": p, "inputs": inames, "outputs": onames})
    json.dump({"latent": a.latent, "parts": man}, open(os.path.join(OUTDIR, "vae_manifest.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
