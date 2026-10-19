#!/usr/bin/env python3
"""ref_gpu.py — 在 **GPU** 上用 torch 跑**与网关完全相同的算法**，作为独立参考

用途：出图不对时，先分清是「链路（RKNN）」还是「算法（调度/缩放）」的问题。
这里刻意**逐行复刻 gateway.py 的采样逻辑**（同样的 scaled_linear + DDIM + 缩放），
只是把 RKNN 换成 torch ⇒ 两边对不上就是 RKNN 侧的问题，对得上就是我算法本身不对。
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_unet_parts import load_unet, P1, P2, P3   # noqa: E402
from export_clip import build as build_clip          # noqa: E402
from clip_tokenizer import CLIPTokenizer             # noqa: E402

VAE_SCALING = 0.18215
TOKDIR = "/home/dunaandone/rknn-work/sd/clip"


def make_alphas(n=1000, b0=0.00085, b1=0.012):
    betas = np.linspace(b0 ** 0.5, b1 ** 0.5, n) ** 2
    return np.cumprod(1.0 - betas)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a photo of a cat")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="/tmp/ref_gpu.png")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = make_alphas()

    # ① 分词 + CLIP（GPU）
    tok = CLIPTokenizer(os.path.join(TOKDIR, "vocab.json"), os.path.join(TOKDIR, "merges.txt"))
    ids = torch.tensor([tok(a.prompt)], dtype=torch.float32, device=dev)
    clip = build_clip().to(dev).eval()
    # ★★★★★ CLIP 的正确前向（与网关一致；原来这里是**手写循环**，错了两个地方）：
    #   ① `open_clip` 的 `Transformer.batch_first = True` ⇒ 张量保持 **[B,77,C]，不要转置**
    #   ② 文本塔是**因果注意力** ⇒ **必须传 `attn_mask`**
    #   实测：原样+掩码 1.000000；原样无掩码 0.2535；转置+掩码 直接报错；转置无掩码 0.2659
    #   （这正是之前「GPU 参考」给出纯色图的根因）
    with torch.no_grad():
        x, am = clip.t._embeds(ids.long())
        for blk in clip.t.transformer.resblocks:
            x = blk(x, attn_mask=am)
        ehs = clip.t.ln_final(x)
    print(f"[ref] ehs {tuple(ehs.shape)}  均值 {ehs.mean():.4f} std {ehs.std():.4f}")

    # ② 采样（与网关逐行一致）
    u = load_unet(torch.float32, dev)
    p1, p2, p3 = P1(u, 2).eval(), P2(u, 2).eval(), P3(u).eval()
    L = a.size // 8
    rng = np.random.RandomState(a.seed)
    z = torch.tensor((rng.randn(1, 4, L, L) * 1.0).astype(np.float32), device=dev) * VAE_SCALING
    # ★★★★★ SD-Turbo 官方用的是 **EulerAncestralDiscreteScheduler**
    #   （`timestep_spacing="trailing"`、`guidance_scale=0`），**不是 DDIM**。
    #   我原来用 DDIM 在 t=999 起步，`1/sqrt(ᾱ)≈15` 会把 eps 的误差放大 15 倍，
    #   而 SD-Turbo 是**蒸馏**出来的 1 步模型，对噪声档位极其敏感 ⇒ 出抽象纹理。
    #   ⇒ 直接调 diffusers 的调度器，保证与官方一致。
    from diffusers import EulerDiscreteScheduler
    sched = EulerDiscreteScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", timestep_spacing="trailing",
        prediction_type="epsilon")
    sched.set_timesteps(max(a.steps, 1))
    print(f"[ref] 调度器 {type(sched).__name__}  timesteps={sched.timesteps.tolist()}  "
          f"init_noise_sigma={float(sched.init_noise_sigma)}")
    z = z * float(sched.init_noise_sigma)
    with torch.no_grad():
        for t in sched.timesteps:
            tb = torch.tensor([float(t)], device=dev)
            # ★★★★★ **必须调 `scale_model_input`**：EulerAncestral 是 `sample / sqrt(σ²+1)`。
            #   我原来把 std≈14.6 的 latent **直接**喂进 UNet ⇒ UNet 看到的输入大了 14.65 倍
            #   ⇒ 预测的 eps 全是垃圾 ⇒ 出抽象纹理（这就是真凶，踩了很久）。
            zi = sched.scale_model_input(z, t)
            o1 = p1(zi, tb, ehs)
            feat, emb, sk1 = o1[0], o1[1], list(o1[2:])
            o2 = p2(feat, emb, ehs)
            mid, sk2 = o2[0], list(o2[1:])
            eps = p3(mid, emb, ehs, *(sk1 + sk2))
            z = sched.step(eps, t, z).prev_sample
    print(f"[ref] 采样后 z 均值 {z.mean():.4f} std {z.std():.4f}")

    # ③ VAE 解码（用 torch 的 diffusers VAE，与 ONNX 里的 1/scaling 一致）
    from diffusers import AutoencoderKL
    ck = torch.load("/home/dunaandone/rknn-work/sd/vae_sd.pt", map_location="cpu", weights_only=False)
    vae = AutoencoderKL.from_config(ck["config"]); vae.load_state_dict(ck["state_dict"]); vae.eval().to(dev)
    with torch.no_grad():
        img = vae.decode(z / VAE_SCALING).sample
    a8 = np.clip((img[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255, 0, 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(a8).save(a.out)
    print(f"[ref] {a.out}  均值 {a8.mean():.1f} std {a8.std():.1f}  "
          f"相邻差 {np.abs(np.diff(a8.mean(2).astype(float),axis=0)).mean():.1f}")
    np.save("/tmp/ref_latent.npy", z.cpu().numpy())
    return 0


if __name__ == "__main__":
    sys.exit(main())
