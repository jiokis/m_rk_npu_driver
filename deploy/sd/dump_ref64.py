#!/usr/bin/env python3
"""dump_ref64.py — 在 GPU 上导出 **512 档每一步的中间量**，供板上逐段比对

导出：初始 z、ehs、UNet 各段输出（p1/p2/up0..up3）、最终 eps、VAE V1 输出 h。
⇒ 板上跑同样的输入，找出**第一处不一致**的环节。
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_unet_parts import load_unet, P1, P2
from export_unet_upblocks import UpOne
from export_clip import build as build_clip
from clip_tokenizer import CLIPTokenizer
from diffusers import AutoencoderKL

VAE_SCALING = 0.18215
TOKDIR = "/home/dunaandone/rknn-work/sd/clip"
SIGMA_MAX = 14.614646911621094


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--prompt", default="a photo of a cat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/ref64.npz")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32

    tok = CLIPTokenizer(os.path.join(TOKDIR, "vocab.json"), os.path.join(TOKDIR, "merges.txt"))
    ids = torch.tensor([tok(a.prompt)], dtype=torch.float32, device=dev)
    clip = build_clip().to(dev).eval()
    with torch.no_grad():
        o = clip.t(ids.long())
        ehs = next(x for x in o if x.dim() == 3) if isinstance(o, (tuple, list)) else o
        if ehs.shape[0] == 77: ehs = ehs.permute(1, 0, 2)
    print(f"[ref] ehs {tuple(ehs.shape)} 均值 {ehs.mean():.4f} std {ehs.std():.4f}")

    u = load_unet(dt, dev)
    p1, p2 = P1(u, 2).eval(), P2(u, 2).eval()
    ups = [UpOne(u, i, tail=(i == 3)).eval() for i in range(4)]
    rng = np.random.RandomState(a.seed)
    z0 = (rng.randn(1, 4, a.latent, a.latent) * SIGMA_MAX).astype(np.float32)
    z = torch.tensor(z0, device=dev)
    t = 999.0
    tb = torch.tensor([t], device=dev)
    zi = z / (SIGMA_MAX ** 2 + 1) ** 0.5            # scale_model_input（第 0 步 σ=SIGMA_MAX）
    print(f"[ref] z std {z.std():.3f}  zi std {zi.std():.3f}")
    with torch.no_grad():
        o1 = p1(zi, tb, ehs); feat, emb, sk1 = o1[0], o1[1], list(o1[2:])
        o2 = p2(feat, emb, ehs); mid, sk2 = o2[0], list(o2[1:])
        sk = sk1 + sk2
        hs = []
        h = mid
        for i in range(4):
            lo = (4 - 1 - i) * 3
            # ⚠️ PyTorch 侧的 `UpOne.forward(h, emb, ehs, *skips)` **签名固定要 ehs**
            #   （哪怕 up0 不用它；只有导出的 ONNX 才会把没用的输入剪掉）
            h = ups[i](h, emb, ehs, *sk[lo:lo + 3])
            hs.append(h)
        eps = h
    print(f"[ref] eps 均值 {eps.mean():.4f} std {eps.std():.4f}")
    ck = torch.load("/home/dunaandone/rknn-work/sd/vae_sd.pt", map_location="cpu", weights_only=False)
    vae = AutoencoderKL.from_config(ck["config"]); vae.load_state_dict(ck["state_dict"]); vae.eval().to(dev)
    d = vae.decoder
    with torch.no_grad():
        hv = d.mid_block(d.conv_in(vae.post_quant_conv(z / VAE_SCALING)))
        for i in range(2): hv = d.up_blocks[i](hv)
    print(f"[ref] vae h {tuple(hv.shape)} std {hv.std():.4f}")
    d2 = {"z0": z0, "ehs": ehs.cpu().numpy(), "zi": zi.cpu().numpy(),
          "feat": feat.cpu().numpy(), "emb": emb.cpu().numpy(),
          "mid": mid.cpu().numpy(), "eps": eps.cpu().numpy(), "vae_h": hv.cpu().numpy(),
          **{f"sk{i}": x.cpu().numpy() for i, x in enumerate(sk)},
          **{f"up{i}": x.cpu().numpy() for i, x in enumerate(hs)}}
    np.savez(a.out, **d2)
    print(f"[ref] {a.out}  {os.path.getsize(a.out)/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
