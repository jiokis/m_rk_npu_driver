#!/usr/bin/env python3
"""
quant_real.py — 用**真实权重**做量化实验（Qwen2.5-0.5B-Instruct F16）

与 quant_experiment.py 的区别：那个用合成数据装出离群结构，
这个直接读真实模型的权重，所以"离群在哪条轴上"是**测出来的**而不是假设的。

## 关键问题

对形状 (out, in) 的线性层权重，量化可以沿两条轴做：
  · per-输出通道 → 每行一个 scale（轴 1 求 max）
  · per-输入通道 → 每列一个 scale（轴 0 求 max）
到底哪条对，取决于离群值实际落在哪条轴上。本脚本直接量出来。

## 用法
  python3 quant_real.py [gguf路径]
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gguf_reader import GGUF, describe  # noqa: E402

DEFAULT = "/home/dunaandone/rkllm-convert/gguf-test/qwen2.5-0.5b-instruct-fp16.gguf"

# 挑有代表性的张量：注意力投影、FFN 三个、以及嵌入
WANT = [
    "token_embd.weight",
    "blk.0.attn_qkv.weight",
    "blk.0.attn_output.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight",
    "blk.12.ffn_gate.weight",
]


def qdq(x, bits, axis=None, group=None, gaxis=-1):
    qmax = (1 << (bits - 1)) - 1

    def one(b):
        s = np.abs(b).max() / qmax
        if s == 0:
            return np.zeros_like(b)
        return (np.clip(np.round(b / s), -qmax - 1, qmax) * s).astype(np.float32)

    if group is not None:
        # 沿 gaxis 分组
        xm = np.moveaxis(x, gaxis, -1)
        out = np.empty_like(xm)
        L = xm.shape[-1]
        for i in range(0, L, group):
            out[..., i:i + group] = qdq(xm[..., i:i + group], bits, axis=None if gaxis == -1 and axis is None else axis)
        return np.moveaxis(out, -1, gaxis)
    if axis is None:
        return one(x)
    s = np.abs(x).max(axis=axis, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s)
    return (np.clip(np.round(x / s), -qmax - 1, qmax) * s).astype(np.float32)


def stats(x, bits):
    """返回 dict：各方案的相对误差与零值占比"""
    n = np.linalg.norm(x)
    res = {}
    for name, kw in (
        ("per-tensor",            {}),
        ("per-输出通道(axis=1)",   {"axis": 1}),
        ("per-输入通道(axis=0)",   {"axis": 0}),
        ("per-group g128(沿 in)",  {"group": 128, "gaxis": -1}),
        ("per-group g128(沿 out)", {"group": 128, "gaxis": 0}),
    ):
        try:
            xq = qdq(x, bits, **kw)
        except Exception as e:
            res[name] = (float("nan"), float("nan"), str(e))
            continue
        rel = float(np.linalg.norm(xq - x) / (n + 1e-12))
        zf = float((xq == 0).mean())
        res[name] = (rel, zf, None)
    return res


def axis_spread(x):
    """比较两条轴的 slice-max 离散程度，判断离群在哪条轴。
    返回 (沿axis0的变异系数, 沿axis1的变异系数)，大的那条离群更严重。"""
    m0 = np.abs(x).max(axis=0)      # 每列一个 max（沿 in）
    m1 = np.abs(x).max(axis=1)      # 每行一个 max（沿 out）
    cv0 = float(m0.std() / (m0.mean() + 1e-12))
    cv1 = float(m1.std() / (m1.mean() + 1e-12))
    return cv0, cv1


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    print("═══ 真实权重解析 ═══")
    describe(path)
    g = GGUF(path)

    print("\n═══ 逐张量：离群结构 + 量化表现 ═══")
    rows = []
    for name in WANT:
        if name not in g.tensors:
            print(f"  ⚠️ 缺 {name}")
            continue
        t = g.tensors[name]
        if t["shape"] and len(t["shape"]) < 2:
            continue
        x = g.read_f16(name)
        cv0, cv1 = axis_spread(x)
        peak = float(np.abs(x).max() / (x.std() + 1e-12))

        print(f"\n  ── {name}  shape={x.shape}  类型 F16 ──")
        print(f"     峰均比 {peak:6.2f} · 沿 in 的 slice-max 变异 {cv0:.3f} · "
              f"沿 out 的 {cv1:.3f}")
        worse = "沿 in（列）" if cv0 > cv1 else "沿 out（行）"
        print(f"     → 离群更集中在: {worse}")

        for bits in (8, 4):
            s = stats(x, bits)
            print(f"     int{bits}:")
            for k, (rel, zf, err) in s.items():
                if err:
                    print(f"       {k:<24} 跳过({err})")
                else:
                    print(f"       {k:<24} 相对误差 {rel*100:6.2f}%   零值 {zf*100:6.2f}%")
        rows.append((name, x.shape, peak, cv0, cv1))

    g.close()

    print("\n═══ 汇总：真实权重的 int4 表现 ═══")
    print(f"  {'张量':<28} {'峰均比':>7} {'离群轴':>10}")
    for name, shape, peak, cv0, cv1 in rows:
        ax = "in(列)" if cv0 > cv1 else "out(行)"
        print(f"  {name:<28} {peak:7.2f} {ax:>10}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
