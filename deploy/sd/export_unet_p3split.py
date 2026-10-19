#!/usr/bin/env python3
"""export_unet_p3split.py — 把 UNet 的 **P3（up_blocks + conv_out）再拆成两半**

## 为什么

P3 的 ONNX 是 1029 MB，转 RKNN 峰值 **5.5 GB** ⇒ 7 GB 的 WSL 上 OOM。
拆成 `up_blocks[0..1]` 与 `up_blocks[2..3] + conv_out` 两段，每段 ~500 MB。

## skip 的分配（up 段是**倒着**消费 skip 的）

`up_blocks[i]` 消费 `down_block_res_samples` 末尾的 `len(up_blocks[i].resnets)` 条。
两段各 2 个 up_block、每块 3 条 resnet ⇒ 各消费 6 条，共 12 条 ✓
"""
from __future__ import annotations
import argparse, json, os, sys
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_unet_parts import load_unet, P1, P2, OUTDIR   # noqa: E402


class P3a(nn.Module):
    """mid + emb + ehs + 12 条 skip → 中间特征（消费 up_blocks[0..1]）"""
    def __init__(self, u, n_up=2):
        super().__init__(); self.u = u; self.n = n_up
    def forward(self, mid, emb, ehs, *skips):
        u = self.u; s = mid; pool = list(skips)
        for b in u.up_blocks[:self.n]:
            k = len(b.resnets); res = pool[-k:]; pool = pool[:-k]
            s = b(s, res, emb, encoder_hidden_states=ehs)
        return s


class P3b(nn.Module):
    """中间特征 + emb + ehs + 剩余 6 条 skip → out"""
    def __init__(self, u, lo=2):
        super().__init__(); self.u = u; self.lo = lo
    def forward(self, h, emb, ehs, *skips):
        u = self.u; s = h; pool = list(skips)
        for b in u.up_blocks[self.lo:]:
            k = len(b.resnets); res = pool[-k:]; pool = pool[:-k]
            s = b(s, res, emb, encoder_hidden_states=ehs)
        return u.conv_out(u.conv_act(u.conv_norm_out(s)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--n-up", type=int, default=2)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if a.dtype == "fp16" else torch.float32
    u = load_unet(dt, dev)
    L = a.latent
    s = torch.zeros(1, 4, L, L, dtype=dt, device=dev)
    t = torch.zeros(1, dtype=dt, device=dev)
    e = torch.zeros(1, 77, 1024, dtype=dt, device=dev)
    with torch.no_grad():
        o1 = P1(u, 2).eval()(s, t, e)
        o2 = P2(u, 2).eval()(o1[0], o1[1], e)
    sk = list(o1[2:]) + list(o2[1:])            # 12 条
    mid, emb = o2[0], o1[1]
    print(f"[p3split] mid{tuple(mid.shape)} emb{tuple(emb.shape)} skips {len(sk)}")
    pa, pb = P3a(u, a.n_up).eval(), P3b(u, a.n_up).eval()
    with torch.no_grad():
        h = pa(mid, emb, e, *sk)
        # ★★★★★ skip 的分配（实测形状，latent=64）：
        #   up[0] 消费 skip[9..11] (8x8)   up[1] 消费 skip[6..8] (16x16)
        #   up[2] 消费 skip[3..5] (32x32)  up[3] 消费 skip[0..2] (64x64)
        # ⇒ P3a(up[0..1]) 吃掉后 6 条，**P3b(up[2..3]) 必须拿前 6 条**（踩过：给了后 6 条，
        #   报 `Sizes of tensors must match ... Expected size 32 but got size 8`）
        out = pb(h, emb, e, *sk[:6])
        ref = u(s, t, encoder_hidden_states=e).sample
    cos = float((out.ravel() @ ref.ravel()) / (out.ravel().norm() * ref.ravel().norm() + 1e-12))
    print(f"[check] P1+P2+P3a+P3b vs 完整 UNet 余弦 = {cos:.6f}")
    os.makedirs(OUTDIR, exist_ok=True)
    man = []
    def exp(m, args, inames, onames, name):
        p = os.path.join(OUTDIR, name + ".onnx")
        torch.onnx.export(m, args, p, input_names=inames, output_names=onames,
                          opset_version=17, do_constant_folding=True, dynamo=False)
        print(f"[onnx] {name}  {os.path.getsize(p)/1e6:.0f} MB")
        man.append({"name": name, "onnx": p, "inputs": inames, "outputs": onames})
    exp(pa, (mid, emb, e) + tuple(sk), ["mid", "emb", "ehs"] + [f"skip{i}" for i in range(12)], ["h"], "unet_p3a")
    exp(pb, (h, emb, e) + tuple(sk[:6]), ["h", "emb", "ehs"] + [f"skip{i}" for i in range(6)], ["out"], "unet_p3b")
    json.dump({"latent": L, "dtype": a.dtype, "parts": man},
              open(os.path.join(OUTDIR, "p3split_manifest.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
