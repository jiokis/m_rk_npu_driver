#!/usr/bin/env python3
"""export_unet_parts.py — 把 SD 的 UNet **按块边界拆成 3 个子模块**分别导出

## 为什么拆

`rknn-toolkit2` 是纯 CPU 工具链；转 865 M 参数的 UNet 峰值 ~6.5 GB，
而本机 WSL 只有 7 GB ⇒ 每次 OOM。**拆成 3 段后每段 ~250–300 M ⇒ 峰值 ~2 GB。**

**⇒ 在 PyTorch 侧拆（GPU 上做），而不是对 ONNX 做手术**：
   `onnx.utils.extract_model` 会把整个 1.65 GB 模型读进内存，同样 OOM（踩过）。

## 切法与张量流

```
P1: sample,timestep,ehs → (feat, skips[0:7], emb)
P2: feat,emb,ehs        → (mid,  skips[7:13])
P3: mid,emb,ehs,skips   → out
```
* `emb`（时间步嵌入 [1,1280]）**每段都要**，所以 P1 必须把它输出出来
* **skip 连接**：down 段产生 13 条（conv_in 1 条 + 4 块 × 3 条），up 段逐块消费
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

CKPT = "/home/dunaandone/rknn-work/sd/sd_turbo.safetensors"
CFG_JSON = "/home/dunaandone/rknn-work/sd/unet_config.json"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parts")


def unet_cfg():
    import json as J
    c = J.load(open(CFG_JSON))
    keep = ("sample_size", "in_channels", "out_channels", "layers_per_block",
            "block_out_channels", "down_block_types", "up_block_types",
            "cross_attention_dim", "attention_head_dim", "norm_num_groups",
            "use_linear_projection", "mid_block_type", "upcast_attention",
            "transformer_layers_per_block", "addition_embed_type",
            "addition_time_embed_dim", "projection_class_embeddings_input_dim",
            "attention_type", "class_embed_type", "encoder_hid_dim",
            "encoder_hid_dim_type", "num_attention_heads", "num_class_embeds",
            "only_cross_attention", "resnet_time_scale_shift", "time_embed_dim")
    return {k: c[k] for k in keep if k in c}


def load_unet(dt, dev):
    from safetensors import safe_open
    from diffusers import UNet2DConditionModel
    from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
    PRE = "model.diffusion_model."
    sd = {}
    with safe_open(CKPT, framework="pt") as f:
        for k in f.keys():
            if k.startswith(PRE):
                sd[k] = f.get_tensor(k).to(dt)
    cfg = unet_cfg()
    conv = convert_ldm_unet_checkpoint(sd, cfg, path=None, extract_ema=False)
    if not conv:
        raise SystemExit("❌ 名字转换 0 项")
    del sd
    torch.cuda.empty_cache()
    conv = {k: v.to(dt) for k, v in conv.items()}
    u = UNet2DConditionModel.from_config(cfg).to(dt)
    missing, unexpected = u.load_state_dict(conv, strict=False)
    print(f"[load] missing {len(missing)} unexpected {len(unexpected)}")
    del conv
    torch.cuda.empty_cache()
    return u.to(dev).eval()


class P1(torch.nn.Module):
    def __init__(self, u, n=2):
        super().__init__(); self.u = u; self.n = n
    def forward(self, sample, timestep, ehs):
        u = self.u
        s = u.conv_in(sample)
        # ⚠️ `time_proj` 输出是 **float32**，而模型是 fp16
        #    ⇒ 不转 dtype 会报 `mat1 and mat2 must have the same dtype, but got Float and Half`（踩过）
        emb = u.time_embedding(u.time_proj(timestep).to(self.u.dtype))
        skips = [s]
        for b in u.down_blocks[:self.n]:
            s, res = b(s, emb, encoder_hidden_states=ehs)
            skips += list(res)
        return (s, emb, *skips)


class P2(torch.nn.Module):
    def __init__(self, u, lo=2):
        super().__init__(); self.u = u; self.lo = lo
    def forward(self, feat, emb, ehs):
        u = self.u
        s = feat
        skips = []
        for b in u.down_blocks[self.lo:]:
            s, res = b(s, emb, encoder_hidden_states=ehs)
            skips += list(res)
        s = u.mid_block(s, emb, encoder_hidden_states=ehs)
        return (s, *skips)


class P3(torch.nn.Module):
    def __init__(self, u):
        super().__init__(); self.u = u
    def forward(self, mid, emb, ehs, *skips):
        u = self.u
        s = mid
        pool = list(skips)
        for b in u.up_blocks:
            n = len(b.resnets)
            res = pool[-n:]
            pool = pool[:-n]
            s = b(s, res, emb, encoder_hidden_states=ehs)
        s = u.conv_out(u.conv_act(u.conv_norm_out(s)))
        return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--split-at", type=int, default=2, help="P1 含前几个 down block")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if a.dtype == "fp16" else torch.float32
    os.makedirs(OUTDIR, exist_ok=True)
    u = load_unet(dt, dev)
    L = a.latent

    s = torch.zeros(1, 4, L, L, dtype=dt, device=dev)
    t = torch.zeros(1, dtype=dt, device=dev)
    e = torch.zeros(1, 77, 1024, dtype=dt, device=dev)

    # 先跑一遍拿到各段张量的形状
    p1 = P1(u, a.split_at).eval()
    with torch.no_grad():
        o1 = p1(s, t, e)
    feat, emb, skips1 = o1[0], o1[1], o1[2:]
    print(f"[p1] feat{tuple(feat.shape)} emb{tuple(emb.shape)} skips {len(skips1)} 条 "
          f"{[tuple(x.shape) for x in skips1]}")
    p2 = P2(u, a.split_at).eval()
    with torch.no_grad():
        o2 = p2(feat, emb, e)
    mid, skips2 = o2[0], o2[1:]
    print(f"[p2] mid{tuple(mid.shape)} skips {len(skips2)} 条 {[tuple(x.shape) for x in skips2]}")
    p3 = P3(u).eval()
    with torch.no_grad():
        out = p3(mid, emb, e, *list(skips1) + list(skips2))
    print(f"[p3] out{tuple(out.shape)}")
    # 与完整 UNet 对照（同权重）
    with torch.no_grad():
        ref = u(s, t, encoder_hidden_states=e).sample
    cos = float((out.ravel() @ ref.ravel()) / (out.ravel().norm() * ref.ravel().norm() + 1e-12))
    print(f"[check] 三段拼起来 vs 完整 UNet 余弦 = {cos:.6f}")

    man = []
    def exp(mod, args, inames, onames, name):
        p = os.path.join(OUTDIR, f"{name}.onnx")
        torch.onnx.export(mod, args, p, input_names=inames, output_names=onames,
                          opset_version=17, do_constant_folding=True, dynamo=False)
        print(f"[onnx] {name}  {os.path.getsize(p)/1e6:.0f} MB")
        man.append({"name": name, "onnx": p, "inputs": inames, "outputs": onames})
    nsk1 = len(skips1)
    exp(p1, (s, t, e), ["sample", "timestep", "ehs"],
        ["feat", "emb"] + [f"skip1_{i}" for i in range(nsk1)], "unet_p1")
    exp(p2, (feat, emb, e), ["feat", "emb", "ehs"],
        ["mid"] + [f"skip2_{i}" for i in range(len(skips2))], "unet_p2")
    exp(p3, (mid, emb, e) + tuple(list(skips1) + list(skips2)),
        ["mid", "emb", "ehs"] + [f"skip{i}" for i in range(nsk1 + len(skips2))], ["out"], "unet_p3")
    json.dump({"latent": L, "dtype": a.dtype, "parts": man,
               "skip_shapes": [[list(x.shape) for x in skips1],
                               [list(x.shape) for x in skips2]]},
              open(os.path.join(OUTDIR, "manifest.json"), "w"), indent=1)
    print(f"[done] {OUTDIR}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
