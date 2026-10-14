#!/usr/bin/env python3
"""pressure_board.py — 逐级加压：**端到端**（潜空间 DDIM + VAE 解码）的板上实时性边界

用 `sweep/` 里**已经转好的**模型（随机权重，计时只取决于算子图）：

| 分辨率 | 潜空间 | UNet | Decoder |
|---|---|---|---|
| 64x64   | 8x8   | unet_64_b128  | dec_64_b32  |
| 128x128 | 16x16 | unet_128_b128 | dec_128_b32 |
| 256x256 | 32x32 | unet_256_b128 | dec_256_b32 |
| 512x512 | 64x64 | unet_512_b128 | dec_512_b32 |

每张图：`steps` 次 UNet + 1 次解码，逐张跑（RKNN batch 固定 1）。
"""
import json, os, sys, time
import numpy as np

STEPS = 20
LEVELS = [("64", 8), ("128", 16), ("256", 32), ("512", 64)]

def main():
    from rknnlite.api import RKNNLite
    print(f"{'分辨率':>8s} {'潜空间':>7s} {'UNet ms/步':>11s} {'解码 ms':>9s} "
          f"{'20步+解码 ms/张':>16s} {'GFLOPs/步':>10s}")
    for res, lat in LEVELS:
        um, dm = f"unet_{res}_b128.rknn", f"dec_{res}_b32.rknn"
        if not (os.path.exists(um) and os.path.exists(dm)):
            print(f"RESULT {res}: 缺模型"); continue
        ru = RKNNLite(verbose=False); ru.load_rknn(um); ru.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        rd = RKNNLite(verbose=False); rd.load_rknn(dm); rd.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        z = np.zeros((1, 4, lat, lat), np.float32)
        tb = np.zeros((1, 128, 1, 1), np.float32)
        for _ in range(3):
            ru.inference(inputs=[z.transpose(0,2,3,1), tb]); rd.inference(inputs=[z.transpose(0,2,3,1)])
        tu, td = [], []
        for _ in range(15):
            t0 = time.perf_counter(); ru.inference(inputs=[z.transpose(0,2,3,1), tb]); tu.append(time.perf_counter()-t0)
            t0 = time.perf_counter(); rd.inference(inputs=[z.transpose(0,2,3,1)]); td.append(time.perf_counter()-t0)
        ru.release(); rd.release()
        mu = sorted(tu)[len(tu)//2]*1000; md = sorted(td)[len(td)//2]*1000
        tot = mu*STEPS + md
        print(f"{res+'x'+res:>8s} {str(lat)+'x'+str(lat):>7s} {mu:11.2f} {md:9.2f} "
              f"{tot:16.1f} {'':>10s}")
        print(f"RESULT {res} {mu:.3f} {md:.3f} {tot:.1f}")

main()
