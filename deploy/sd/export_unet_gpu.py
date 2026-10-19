#!/usr/bin/env python3
"""export_unet_gpu.py — 用 **GPU** 导出 SD 的 UNet 到 ONNX（绕开 7 GB 内存限制）

## 为什么必须用 GPU

UNet 有 **860 M 参数**：
* fp32 = 3.44 GB；加上 diffusers 模型副本再 3.44 GB ⇒ **7 GB 内存必爆**
* 显存有 8 GB ⇒ 放 GPU 上做，**用显存换内存**

## 输入设计（RKNN 静态形状）

| 输入 | 形状 |
|---|---|
| `sample` | `[1,4,L,L]` |
| `timestep` | `[1]` |
| `ehs`（encoder_hidden_states） | `[1,77,1024]` |

## 用法（必须用 sddiff 环境，CPU 版 torch 在这里没用）
  /home/dunaandone/miniconda3/envs/sddiff/bin/python deploy/sd/export_unet_gpu.py --latent 32 --dtype fp16
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

CKPT = "/home/dunaandone/rknn-work/sd/sd_turbo.safetensors"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rknn")

# ★★★★★ **必须用模型自带的 config.json**，不能套 SD1.5 的那套。
#   SD-Turbo 是 **SD2.1 系**：`attention_head_dim=[5,10,20,20]`、`use_linear_projection=True`。
#   套 SD1.5 的 `attention_head_dim=8` 会让 attention 的 `proj_in/out` 建错类型：
#     检查点是 `[320,320]`（Linear），而 SD1.5 配置建出来是 `[320,320,1,1]`（Conv2d）
#     ⇒ `size mismatch ... current model is torch.Size([320,320,1,1])`（踩过）
CFG_JSON = "/home/dunaandone/rknn-work/sd/unet_config.json"


def unet_cfg():
    import json
    c = json.load(open(CFG_JSON))
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if a.dtype == "fp16" else torch.float32
    out = a.out or os.path.join(OUTDIR, f"sdunet_l{a.latent}_{a.dtype}.onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"[env] device={dev} dtype={a.dtype} 显存 "
          f"{torch.cuda.mem_get_info()[0]/2**30:.1f} GB 可用" if dev == "cuda" else "[env] cpu")

    from safetensors import safe_open
    from diffusers import UNet2DConditionModel
    from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint

    # ① 只取 model.*（UNet），直接放 GPU，fp16
    # ★★★★★ 键名是 `model.diffusion_model.*`（CompVis/LDM 命名），而且
    #    **`convert_ldm_unet_checkpoint` 自己会去掉这个前缀**（`unet_key = LDM_UNET_KEY`）
    #    ⇒ 必须把**带前缀的原始键**交给它！
    #    （踩过两次：先按 `model.` 剥、再按 `model.diffusion_model.` 剥，两次都得到 0 项，
    #      而 `missing 686` 意味着导出的 ONNX 是**随机权重** —— 会一路错到板上。）
    PRE = "model.diffusion_model."
    sd = {}
    with safe_open(CKPT, framework="pt") as f:
        for k in f.keys():
            if k.startswith(PRE):
                sd[k] = f.get_tensor(k).to(dt)
    print(f"[load] 张量 {len(sd)}  {sum(v.numel() for v in sd.values())/1e6:.1f} M 参数  "
          f"显存 {torch.cuda.memory_allocated()/2**30:.2f} GB")

    # ② 名字转换（CompVis → diffusers）
    cfg = unet_cfg()
    print(f"[cfg] 用模型自带 config：attention_head_dim={cfg.get('attention_head_dim')} "
          f"use_linear_projection={cfg.get('use_linear_projection')}")
    conv = convert_ldm_unet_checkpoint(sd, cfg, path=None, extract_ema=False)
    if not conv:
        raise SystemExit("[load] ❌ 名字转换返回 0 项 —— 前缀或配置不对，导出会得到随机权重！")
    del sd
    torch.cuda.empty_cache()
    conv = {k: v.to(dt) for k, v in conv.items()}
    print(f"[load] 转换后 {len(conv)} 项  显存 {torch.cuda.memory_allocated()/2**30:.2f} GB")

    # ③ 建模型（fp16）并加载
    u = UNet2DConditionModel.from_config(cfg).to(dt)
    missing, unexpected = u.load_state_dict(conv, strict=False)
    print(f"[load] missing {len(missing)}  unexpected {len(unexpected)}")
    if missing[:3]: print("   missing:", missing[:3])
    if unexpected[:3]: print("   unexpected:", unexpected[:3])
    del conv
    torch.cuda.empty_cache()
    u = u.to(dev).eval()
    print(f"[model] UNet {sum(p.numel() for p in u.parameters())/1e6:.1f} M 参数  显存 "
          f"{torch.cuda.memory_allocated()/2**30:.2f} GB")

    class W(torch.nn.Module):
        def __init__(self, u):
            super().__init__()
            self.u = u
        def forward(self, sample, timestep, ehs):
            return self.u(sample, timestep, encoder_hidden_states=ehs).sample

    m = W(u).eval()
    s = torch.zeros(1, 4, a.latent, a.latent, dtype=dt, device=dev)
    t = torch.zeros(1, dtype=dt, device=dev)
    e = torch.zeros(1, 77, 1024, dtype=dt, device=dev)
    with torch.no_grad():
        y = m(s, t, e)
    print(f"[shape] {tuple(s.shape)} + {tuple(t.shape)} + {tuple(e.shape)} → {tuple(y.shape)}")
    # ⚠️ opset 14 会失败：`LayerNormalization` 是 opset 17 才有的算子，
    #    降到 14 时 onnx 的版本转换器直接断言失败
    #    （`Assertion 'false' failed: No Previous Version of LayerNormalization exists`）
    #    ⇒ 用 **opset 17**（踩过）
    # ★★★★★ 必须用**传统 TorchScript 导出器**（`dynamo=False`）。
    #   torch 2.14 默认走 dynamo 导出器，它产的图会让 RKNN 的 build 失败：
    #     `Node (node_Split_44) Op (Split) [ShapeInferenceError]
    #      Both 'split' input and 'num_outputs' attribute were given`
    #   （那个 `node_Split_44` 是 RKNN 内部重导出的名字；ONNX 本身的 Split 节点是合法的）
    #   ⇒ 换传统导出器（踩过）
    # ★ 用**外部数据格式**（use_external_data_format=True）：
    #   onnxruntime 可以对 `.onnx.data` 做 mmap，而不是把 1.65 GB 全读进内存。
    #   本机只有 7 GB，RKNN 转 865 M 参数的 UNet 时峰值 ~6.5 GB ⇒ 被 OOM kill。
    #   （dynamo 导出器默认就会外置，但它产的图会让 RKNN build 失败，所以这里
    #     用**传统导出器 + 显式外置**的组合。）
    torch.onnx.export(m, (s, t, e), out, input_names=["sample", "timestep", "ehs"],
                      output_names=["out"], opset_version=17, do_constant_folding=True,
                      dynamo=False, use_external_data_format=True)
    print(f"[onnx] {out}  {os.path.getsize(out)/1e6:.0f} MB")
    json.dump({"latent": a.latent, "dtype": a.dtype,
               "params": sum(p.numel() for p in u.parameters())}, open(out + ".json", "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
