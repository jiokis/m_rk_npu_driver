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


# ─────────── 量化路径（激活量化 / CVT requant）───────────
# 参数来自 .cfg（hybrid_quantization_step1 生成，docs/67）：
#   input_int8 : scale=1/255, zp=-128        c : scale=1.0871973374310662, zp=-128
#   r          : 同 c                        gap_2conv0 : scale=1.2743302887561274, zp=-128
#   g          : scale=0.7265521479587929    output_mm  : 同 g
# 权重 scale 由实测回归得到（docs/69）：0.105814

CFG = {
    "input":  {"scale": 1.0 / 255.0, "zp": -128},
    "c":      {"scale": 1.0871973374310662, "zp": -128},
    "gap":    {"scale": 1.2743302887561274, "zp": -128},
    "g":      {"scale": 0.7265521479587929, "zp": -128},
}
SCALE_W = 0.105814


def quantize(f: np.ndarray, scale: float, zp: int = -128) -> np.ndarray:
    """float → int8：q = round(f/scale) + zp，夹到 [-128,127]"""
    q = np.round(np.asarray(f, dtype=np.float64) / scale) + zp
    return np.clip(q, -128, 127).astype(np.int32)


def dequantize(q: np.ndarray, scale: float, zp: int = -128) -> np.ndarray:
    return (np.asarray(q, dtype=np.float64) - zp) * scale


def conv2d_int8(qx: np.ndarray, qw: np.ndarray, pad: int = 1):
    """int8 卷积的整数累加：acc[oy,ox,oc] = Σ qw[oc,ic,ky,kx] × qx[iy,ix,ic]"""
    h, wd, cin = qx.shape
    cout, _, kh, kw = qw.shape
    xp = np.pad(qx, ((pad, pad), (pad, pad), (0, 0)))
    out = np.zeros((h, wd, cout), dtype=np.int64)
    for ky in range(kh):
        for kx in range(kw):
            # 取偏移后的 patch: [h, wd, cin]
            patch = xp[ky:ky + h, kx:kx + wd, :]
            # qw[:, :, ky, kx] 是 [cout, cin]
            out += np.einsum("hwc,oc->hwo", patch.astype(np.int64),
                             qw[:, :, ky, kx].astype(np.int64))
    return out


def run_probe4_quant(dram_buf, input_buf, w_scale=SCALE_W):
    """量化路径复现 probe4：Conv → Relu → GAP → Gemm（每步都过 int8 量化）"""
    raw = np.frombuffer(input_buf, dtype=np.uint8)[: 32 * 32 * 3].reshape(32, 32, 3)
    qx = quantize(raw.astype(np.float64) / 255.0, CFG["input"]["scale"], CFG["input"]["zp"])
    # int8 权重（直接从 DRAM 取有符号字节）
    from .weights import weight_offset
    buf = np.frombuffer(dram_buf, dtype=np.uint8) if not isinstance(dram_buf, np.ndarray) else dram_buf
    qw = np.zeros((16, 3, 3, 3), dtype=np.int32)
    for oc in range(16):
        for ky in range(3):
            for kx in range(3):
                for ic in range(3):
                    b = int(buf[weight_offset(ky, kx, oc, ic, 3)])
                    qw[oc, ic, ky, kx] = (b - 256) if b >= 128 else b
    acc = conv2d_int8(qx, qw, pad=1)                       # int64
    # requant: real = acc × w_scale × x_scale ; q = real/scale_c + zp_c
    m = w_scale * CFG["input"]["scale"] / CFG["c"]["scale"]
    qc = np.clip(np.round(acc * m) + CFG["c"]["zp"], -128, 127).astype(np.int32)
    qc = np.maximum(qc, -128)                              # Relu（0.0 对应 -128）
    # GAP：先反量化求均值，再量化到 gap 的 scale
    f_r = dequantize(qc, CFG["c"]["scale"], CFG["c"]["zp"])
    g_f = f_r.mean(axis=(0, 1))
    qg = quantize(g_f, CFG["gap"]["scale"], CFG["gap"]["zp"])
    g = dequantize(qg, CFG["g"]["scale"], CFG["g"]["zp"])
    return qc, g
