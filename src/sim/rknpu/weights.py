"""weights.py — RK3588 NPU 权重加载与反量化（仿真器执行器的取数路径）

布局与公式来自实测（docs/68 / docs/69），全部有交叉验证：

    布局： offset(ky, kx, oc, ic) = (ky*kw + kx) * TAP_STRIDE + oc * IC_ALIGN + ic
           ★ TAP_STRIDE = out_ch * IC_ALIGN      （**不是固定 256**）
             实测：probe4 out_ch=16 ⇒ 9*16*16 = 2304 = WEIGHT_SIZE0 ✅
                   probe5 out_ch= 2 ⇒ 9* 2*16 =  288 = WEIGHT_SIZE0 ✅
                   probe5 的 Gemm (kw=kh=1, in=2048, out=2) ⇒ 1*1*2*2048 = 4096 ✅
           IC_ALIGN   = 16  （align(in_ch, 16)）
           不存在的输入通道（ic >= in_ch）填 128
           真实零权重 → 0，且 **byte 0 是「精确零」标记，不适用反量化公式**
             （实测：oc1..15 的真值全为 0、字节全为 0；若套公式会得到
               (0+128)*scale = 13.54 ≠ 0 ⇒ 必须特判）

    反量化： w_float = (q_signed - ZERO_POINT) * scale
            q_signed = byte - 256 若 byte >= 128
            ZERO_POINT = -128（与 .cfg 里激活参数的约定一致）
            scale 每个权重张量一个（docs/69：27/27 完美拟合，per-tensor）

用法：
    import numpy as np
    from src.sim.rknpu.weights import load_weight_tensor
    w = load_weight_tensor(buf, in_ch=3, out_ch=16, kh=3, kw=3,
                           scale=0.105814)          # → [oc][ic][ky][kx] float32
"""
from __future__ import annotations

import numpy as np

TAP_STRIDE_DEFAULT = 256   # out_ch=16 时的每 tap 步长（= 16 * IC_ALIGN）
IC_ALIGN = 16         # 每个 oc 在一个 tap 内的步长 / ic 的对齐宽度
ZERO_POINT = -128     # 权重与激活共用的零点（实测 + .cfg 一致）
PAD_BYTE = 128        # 「不存在的输入通道」的填充码


def tap_stride(out_ch: int) -> int:
    """每个 kernel tap 的步长 = out_ch * align(in_ch, 16)。实测泛化（probe4/probe5）。"""
    return out_ch * IC_ALIGN


def weight_offset(ky: int, kx: int, oc: int, ic: int, kw: int, out_ch: int = 16) -> int:
    """权重张量里 (ky, kx, oc, ic) 在权重块内的字节偏移。"""
    return (ky * kw + kx) * tap_stride(out_ch) + oc * IC_ALIGN + ic


def to_signed(b: np.ndarray) -> np.ndarray:
    """无符号字节 → 有符号 int8（这一步把 docs/68 的「8 位回绕」变成连续序列）。"""
    b = b.astype(np.int16)
    return np.where(b >= 128, b - 256, b).astype(np.int32)


def dequant(q_signed: np.ndarray, scale: float, zero_point: int = ZERO_POINT) -> np.ndarray:
    """w = (q - zero_point) * scale，**但 byte 0 特判为精确 0**。

    实测依据（docs/69 的 probe4）：oc0 的真实权重 1..27 → 字节非零，公式吻合（误差 ≤ scale/2）；
    oc1..15 的真实权重全为 0 → 字节全为 0，若套公式得 13.54 ≠ 0。
    ⇒ byte 0 是「精确零」标记（与「不存在的通道」的填充 128 是两回事）。
    """
    q = np.asarray(q_signed, dtype=np.float32)
    out = (q - zero_point) * np.float32(scale)
    return np.where(q == 0, np.float32(0.0), out)


def load_weight_tensor(buf, in_ch: int, out_ch: int, kh: int, kw: int,
                       scale: float, base: int = 0, zero_point: int = ZERO_POINT):
    """从 DRAM 镜像里取出一个算子的完整权重张量，返回 [oc][ic][ky][kx] float32。

    只取 ic < in_ch 的真实槽；对齐出去的槽（填充 128）不参与计算。
    """
    buf = np.frombuffer(buf, dtype=np.uint8) if not isinstance(buf, np.ndarray) else buf
    w = np.zeros((out_ch, in_ch, kh, kw), dtype=np.float32)
    for oc in range(out_ch):
        for ky in range(kh):
            for kx in range(kw):
                for ic in range(in_ch):
                    off = base + weight_offset(ky, kx, oc, ic, kw, out_ch)
                    q = to_signed(np.asarray([buf[off]]))[0]
                    # ★ byte 0 = 精确零（实测特判）
                    w[oc, ic, ky, kx] = 0.0 if q == 0 else (q - zero_point) * scale
    return w


def expected_size(in_ch: int, out_ch: int, kh: int, kw: int) -> int:
    """该权重张量在 DRAM 里应占的字节数（= 寄存器 WEIGHT_SIZE0）。"""
    return kh * kw * tap_stride(out_ch)
