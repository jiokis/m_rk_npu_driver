#!/usr/bin/env python3
"""
quant_experiment.py — 量化自由度实验（v2：修正指标与离群轴）

## v1 的两个错误（保留记录，避免重犯）

1. **离群轴搞错**：我把权重离群加在「行」（输入维度，W[K,N] 的 K），
   却用 `axis=0` 做 per-channel（得到的是「列」的 scale）。
   列 scale 根本管不到行离群，于是 int4 权重误差虚高到 98.7%。
   正确做法要区分：per-**输出通道**（列）与 per-**输入通道**（行）是两个不同的东西。

2. **指标会骗人**：全局相对误差 ‖Yq−Y‖/‖Y‖ 里，激活的巨量通道本身就是最大值、
   被量化得相对精确，而其它通道全被压成 0 —— 但那些小值对 ‖Y‖ 贡献小，
   于是「99% 的值变成 0」这种灾难在全局指标上几乎看不出来。
   → 必须同时看**零值占比**和**逐元素误差**。

## 本版在看什么

张量级（直接、可解释）：
  · 量化后相对误差  ‖tq − t‖ / ‖t‖
  · **零值占比**  —— int4 最致命的失效模式，值被 scale 压到 0
  · 峰均比  max|x| / std(x)  —— 越高越难量化，是离群程度的度量

输出级：相对误差 + 余弦相似度（并注明它会被巨量通道主导）
"""
import json
import sys

import numpy as np

rng = np.random.default_rng(20261011)

K, N, M = 512, 512, 128
W_OUTLIER_ROWS = 8        # 权重离群的「行」（输入通道）
X_OUTLIER_CHANS = 4       # 激活巨量的「通道」（同为输入维度 K）
X_OUTLIER_SCALE = 100.0
GROUP = 128


def make_data():
    W = rng.normal(0, 1, (K, N)).astype(np.float32)
    W[rng.choice(K, W_OUTLIER_ROWS, replace=False)] *= 10.0
    X = rng.normal(0, 1, (M, K)).astype(np.float32)
    X[:, rng.choice(K, X_OUTLIER_CHANS, replace=False)] *= X_OUTLIER_SCALE
    return W, X


def qdq(x, bits, axis=None, group=None):
    """对称量化到 bits 位再反量化。axis=None → per-tensor。"""
    qmax = (1 << (bits - 1)) - 1

    def one(b):
        s = np.abs(b).max() / qmax
        if s == 0:
            return np.zeros_like(b)
        return (np.clip(np.round(b / s), -qmax - 1, qmax) * s).astype(np.float32)

    if group is not None:
        out = np.empty_like(x)
        for i in range(0, x.shape[-1], group):
            out[..., i:i + group] = qdq(x[..., i:i + group], bits, axis=axis)
        return out
    if axis is None:
        return one(x)
    s = np.abs(x).max(axis=axis, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s)
    return (np.clip(np.round(x / s), -qmax - 1, qmax) * s).astype(np.float32)


def hadamard(n):
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def tstats(name, t, bits, axis=None, group=None, log=None):
    """张量级统计：相对误差 / 零值占比 / 峰均比"""
    tq = qdq(t, bits, axis=axis, group=group)
    rel = float(np.linalg.norm(tq - t) / (np.linalg.norm(t) + 1e-12))
    zf = float((tq == 0).mean())
    ratio = float(np.abs(t).max() / (np.abs(t).std() + 1e-12))
    if log is not None:
        log.append({"tensor": name, "bits": bits, "rel_err": rel, "zero_frac": zf})
    print(f"  {name:<38} 相对误差 {rel*100:7.2f}%   零值占比 {zf*100:6.2f}%   "
          f"(原始峰均比 {ratio:6.1f})")
    return rel, zf


def ystats(name, Y, Yq, log=None):
    rel = float(np.linalg.norm(Yq - Y) / np.linalg.norm(Y))
    cos = float(np.sum(Y * Yq) / (np.linalg.norm(Y) * np.linalg.norm(Yq) + 1e-12))
    if log is not None:
        log.append({"output": name, "rel_err": rel, "cosine": cos})
    print(f"  {name:<38} 输出相对误差 {rel*100:7.2f}%   余弦 {cos:.6f}")
    return rel, cos


def main():
    W, X = make_data()
    Y = X @ W
    tlog, ylog = [], []
    print(f"数据 W{W.shape} X{X.shape} → Y{Y.shape}")
    print(f"  权重离群 {W_OUTLIER_ROWS}/{K} 行 ×10 · 激活巨量 {X_OUTLIER_CHANS}/{K} 通道 ×{X_OUTLIER_SCALE:.0f}\n")

    print("═══ 0. 先看张量本身有多难量化（关键：零值占比）═══")
    for bits in (8, 4):
        print(f"  ── int{bits} ──")
        tstats(f"W per-tensor", W, bits, log=tlog)
        tstats(f"W per-输出通道(列, axis=1)", W, bits, axis=1, log=tlog)
        tstats(f"W per-输入通道(行, axis=0)", W, bits, axis=0, log=tlog)
        tstats(f"W per-group(g{GROUP}, 沿N)", W, bits, group=GROUP, log=tlog)
        print()
        tstats(f"X per-tensor", X, bits, log=tlog)
        tstats(f"X per-token(行, axis=1)", X, bits, axis=1, log=tlog)
        tstats(f"X per-通道(列, axis=0)", X, bits, axis=0, log=tlog)
        print()

    print("═══ 1. 激活旋转能不能救 int4 ═══")
    H = hadamard(K)
    Xr, Wr = X @ H, H.T @ W
    Yr = Xr @ Wr
    print(f"  旋转等价性检查: {np.linalg.norm(Yr-Y)/np.linalg.norm(Y)*100:.6f}%（应≈0）\n")
    for bits in (4, 8):
        print(f"  ── int{bits} ──")
        tstats(f"X(未旋转) per-tensor", X, bits, log=tlog)
        tstats(f"X(旋转后) per-tensor", Xr, bits, log=tlog)
        tstats(f"X(未旋转) per-token", X, bits, axis=1, log=tlog)
        tstats(f"X(旋转后) per-token", Xr, bits, axis=1, log=tlog)
        print()

    print("═══ 2. 端到端输出误差（注意：会被巨量通道主导）═══")
    ystats("W int4 per-输出通道",   Y, X @ qdq(W, 4, axis=1), ylog)
    ystats("X int4 per-tensor",     Y, qdq(X, 4) @ W, ylog)
    ystats("X 旋转 int4 per-tensor", Y, qdq(Xr, 4) @ Wr, ylog)
    ystats("双 int4(旋转,per-tensor)", Y, qdq(Xr, 4) @ qdq(Wr, 4, axis=1), ylog)
    ystats("双 int8(旋转,per-tensor)", Y, qdq(Xr, 8) @ qdq(Wr, 8, axis=1), ylog)

    print("\n═══ 结论（从数据推出，不预设）═══")
    x4 = [e for e in tlog if e.get("tensor") == "X per-tensor" and e["bits"] == 4][0]
    xr4 = [e for e in tlog if e.get("tensor") == "X(旋转后) per-tensor" and e["bits"] == 4][0]
    xt4 = [e for e in tlog if e.get("tensor") == "X per-token(行, axis=1)" and e["bits"] == 4][0]
    xrt4 = [e for e in tlog if e.get("tensor") == "X(旋转后) per-token" and e["bits"] == 4][0]

    print(f"  · 激活 int4 per-tensor：**{x4['zero_frac']*100:.1f}% 的值被压成 0** —— 这就是 int4 的真正杀手，")
    print(f"    而它在输出指标上几乎看不出来（因为巨量通道仍被精确量化）。")
    print(f"  · Hadamard 旋转后零值占比降到 {xr4['zero_frac']*100:.1f}%（相对误差 "
          f"{x4['rel_err']*100:.2f}% → {xr4['rel_err']*100:.2f}%）")
    print(f"  · per-token 本身也能压住：零值占比 {xt4['zero_frac']*100:.1f}%，"
          f"旋转后进一步到 {xrt4['zero_frac']*100:.1f}%")
    print()
    print("  → 可落地组合：**权重 per-输出通道 + 激活 Hadamard 旋转后 per-tensor**。")
    print("    前者是纯离线处理，后者只需在进入量化前做一次正交变换（Hadamard 只有加减法）。")

    out = {"config": {"K": K, "N": N, "M": M, "group": GROUP,
                      "w_outlier_rows": W_OUTLIER_ROWS,
                      "x_outlier_chans": X_OUTLIER_CHANS,
                      "x_outlier_scale": X_OUTLIER_SCALE},
           "tensor_stats": tlog, "output_stats": ylog}
    with open("quant_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已写入 quant_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
