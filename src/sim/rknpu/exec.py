"""exec.py — RK3588 NPU 执行器（最小闭环）

把「从 DRAM 取权重 / 取输入 / 反量化 / 卷积 / 输出量化」串起来，
目标是能与板子实测、onnxruntime 浮点参考、官方模拟器三方对照。

已确定的三件事（都有实测交叉验证）：
  权重布局   offset(ky,kx,oc,ic) = (ky*kw+kx)*256 + oc*16 + ic      (docs/68)
  权重反量化 w = (q_signed + 128) * scale，byte 0 = 精确零          (docs/69/70)
  输入布局   原始 uint8、NHWC、缓冲区偏移 0                          (docs/71)
"""
from __future__ import annotations

import numpy as np

from .weights import load_weight_tensor, ZERO_POINT


def input_from_uint8(buf, shape_hwc, base: int = 0, mean: float = 0.0, std: float = 255.0):
    """从 DRAM 镜像取输入，返回 float32 NHWC。实测：原始 uint8、NHWC、偏移 0。"""
    buf = np.frombuffer(buf, dtype=np.uint8) if not isinstance(buf, np.ndarray) else buf
    h, w, c = shape_hwc
    n = h * w * c
    raw = buf[base:base + n].reshape(h, w, c).astype(np.float32)
    return (raw - mean) / std


def conv2d_nhwc(x, w, bias=None, pad: int = 1, stride: int = 1):
    """x: [H,W,Cin] float, w: [Cout,Cin,kh,kw] float → [Hout,Wout,Cout] float"""
    h, wd, cin = x.shape
    cout, _, kh, kw = w.shape
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    ho = (h + 2 * pad - kh) // stride + 1
    wo = (wd + 2 * pad - kw) // stride + 1
    out = np.zeros((ho, wo, cout), dtype=np.float32)
    for oy in range(ho):
        for ox in range(wo):
            patch = xp[oy * stride:oy * stride + kh, ox * stride:ox * stride + kw, :]  # [kh,kw,Cin]
            # w[cout,cin,ky,kx] × patch[ky,kx,cin]
            out[oy, ox] = np.einsum("ijk,oijk->o", patch, w)
    if bias is not None:
        out += bias.reshape(1, 1, -1)
    return out


def run_probe4(dram_buf, input_buf, scale_w: float, mean=0.0, std=255.0):
    """复现 probe4：Conv(3→16,k3,s1,p1) → Relu → GAP → Flatten → Gemm(16→2)"""
    x = input_from_uint8(input_buf, (32, 32, 3), mean=mean, std=std)
    w = load_weight_tensor(dram_buf, in_ch=3, out_ch=16, kh=3, kw=3, scale=scale_w)
    y = conv2d_nhwc(x, w, pad=1)
    y = np.maximum(y, 0.0)                       # Relu
    g = y.mean(axis=(0, 1))                      # GAP
    return y, g
