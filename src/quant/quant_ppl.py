#!/usr/bin/env python3
"""
quant_ppl.py — 真实困惑度实验：判断量化方案的下游影响

## 为什么必须做这个

docs/08 的最大缺口是「只测张量误差，没测下游指标」。
而实验已证明**全局 Frobenius 误差是不可靠的代理**：
`X` int4 per-tensor 有 99.35% 的值归零，输出误差却只显示 21.48%。
到底哪种量化方案能用，只能看**真实困惑度**。

## 方法

用纯 numpy 的 Qwen2 前向（qwen2_numpy.py）在真实 Qwen2.5-0.5B-Instruct 权重上
算困惑度：对每个 token 求 -log p(t_i | t_<i)，取平均后 exp。

各方案**共用同一套分词与前向**，唯一变量是权重量化方式 —— 比较是干净的。

## 注意

分词是贪心最长匹配、不是真 BPE，所以**绝对困惑度不能和公开榜单比**；
但方案之间的**相对差异**是有效的，那才是本实验要的。
"""
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from qwen2_numpy import Qwen2  # noqa: E402

TEXT = (
    "The Orange Pi 5 Plus is a single-board computer built around the Rockchip RK3588 "
    "system on a chip. It integrates a quad-core Cortex-A76 cluster running at up to "
    "2.4 GHz together with a quad-core Cortex-A55 cluster, and it provides a neural "
    "processing unit rated at six trillion operations per second. The board is commonly "
    "used for edge inference, where a small language model runs entirely on the device "
    "without sending data to a remote server. Quantization is the usual technique for "
    "fitting such a model into the available memory and memory bandwidth, because the "
    "decode phase of text generation is bound by how fast weights can be streamed from "
    "memory rather than by arithmetic throughput."
)

CTX = 256


def qdq(w, bits, scheme, group=128):
    """对称量化再反量化。scheme: tensor / channel / group"""
    qmax = (1 << (bits - 1)) - 1
    if scheme == "tensor":
        s = np.abs(w).max() / qmax
        if s == 0:
            return w.copy()
        return (np.clip(np.round(w / s), -qmax - 1, qmax) * s).astype(np.float32)
    if scheme == "channel":                       # 每行(输出通道)一个 scale
        s = np.abs(w).max(axis=1, keepdims=True) / qmax
        s = np.where(s == 0, 1.0, s)
        return (np.clip(np.round(w / s), -qmax - 1, qmax) * s).astype(np.float32)
    if scheme == "group":                         # 沿输入维分组
        L = w.shape[1]
        pad = (-L) % group
        x = np.pad(w, [(0, 0), (0, pad)]) if pad else w
        g = x.reshape(x.shape[0], x.shape[1] // group, group)
        s = np.abs(g).max(axis=-1, keepdims=True) / qmax
        s = np.where(s == 0, 1.0, s)
        q = (np.clip(np.round(g / s), -qmax - 1, qmax) * s).reshape(x.shape)
        return (q[:, :L] if pad else q).astype(np.float32)
    raise ValueError(scheme)


def make_transform(bits, scheme, group=128, skip_1d=True):
    """返回一个 weights_transform；None 表示不量化"""
    if bits is None:
        return None
    def t(name, arr):
        if skip_1d and arr.ndim < 2:
            return arr                       # norm 权重不动
        return qdq(arr, bits, scheme, group)
    return t


def perplexity(model, ids):
    logits = model.forward(ids)               # (T, vocab)
    lo = logits[:-1].astype(np.float64)
    tgt = np.asarray(ids[1:])
    m = lo.max(axis=-1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(lo - m).sum(axis=-1))
    nll = lse - lo[np.arange(len(tgt)), tgt]
    return float(np.exp(nll.mean())), float(nll.mean())


def main():
    schemes = [
        ("fp32 基线（不量化）",        None, None, None),
        ("int8 per-tensor",           8, "tensor", None),
        ("int8 per-输出通道",          8, "channel", None),
        ("int8 per-group g128",       8, "group", 128),
        ("int4 per-tensor",           4, "tensor", None),
        ("int4 per-输出通道",          4, "channel", None),
        ("int4 per-group g128",       4, "group", 128),
        ("int4 per-group g64",        4, "group", 64),
    ]

    # 先分词（一次），保证所有方案输入相同
    m0 = Qwen2()
    m0.load_vocab()
    ids = m0.encode(TEXT)[:CTX]
    m0.close()
    print(f"文本 token 数（截断到 {CTX}）: {len(ids)}")
    print(f"前 {12} 个: {ids[:12]}\n")

    print(f"{'方案':<26} {'困惑度':>10} {'平均NLL':>10} {'相对基线':>10} {'耗时':>8}")
    print("  " + "─" * 70)
    base = None
    results = []
    for name, bits, scheme, group in schemes:
        t0 = time.time()
        m = Qwen2(weights_transform=make_transform(bits, scheme, group or 128))
        ppl, nll = perplexity(m, ids)
        m.close()
        el = time.time() - t0
        if base is None:
            base = ppl
            rel = "—"
        else:
            rel = f"{ppl/base:.2f}×"
        print(f"  {name:<26} {ppl:>10.3f} {nll:>10.4f} {rel:>10} {el:>7.1f}s")
        results.append((name, ppl))

    print()
    print("═══ 结论 ═══")
    d = dict(results)
    b = d["fp32 基线（不量化）"]
    for k in ("int8 per-group g128", "int4 per-group g128", "int4 per-tensor"):
        if k in d:
            print(f"  {k:<24} 困惑度 {d[k]:7.3f}  （基线的 {d[k]/b:.2f} 倍）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
