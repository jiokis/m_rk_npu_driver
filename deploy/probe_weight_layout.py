#!/usr/bin/env python3
"""
probe_weight_layout.py — 用「已知明文」直接读出 RKNN 的权重内存布局

## 思路

权重布局是个**可证伪的比对问题**（docs/61）：ONNX 是我们自己造的，所以知道浮点权重。
但直接比对随机权重很难看出结构。**换成 one-hot 探针**就一目了然：

* 每个输出通道只放**一个**非零权重，且各通道的非零位置**互不相同**
  ⇒ 转换后的权重缓冲区里会出现若干个**孤立的非零字节**
  ⇒ 它们的位置/步长**直接暴露布局**（哪个维度步长多少、对齐到几）

再叠一层：让非零值本身编码信息（用不同的量化值），就能读出通道顺序。

## 探针模型设计

```
输入 1×3×32×32
conv1: 3→16, k=3, s=1, p=1   ← ★ one-hot 权重：oc 的非零在 (ic=oc%3, ky=oc//3%3, kx=oc//9)
conv2: 16→8,  k=1            ← 1×1，权重全为常数（用于看 1×1 的布局）
fc:    8→4
```

只保留很小的特征图（32×32），让权重缓冲区里**几乎只有权重**，便于定位。

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/probe_weight_layout.py --make          # 生成 probe_layout.onnx
  $V deploy/probe_weight_layout.py --analyze BIN   # 分析抓到的权重缓冲区
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def make_probe(out_path, size=32):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, inits = [], []

    def add_init(name, arr):
        inits.append(numpy_helper.from_array(arr.astype(np.float32), name))

    # ── conv1: 3→16, k=3, s=1, p=1，one-hot 权重 ──
    W1 = np.zeros((16, 3, 3, 3), dtype=np.float32)   # [oc][ic][ky][kx]
    pos = {}
    for oc in range(16):
        ic = oc % 3
        ky = (oc // 3) % 3
        kx = (oc // 9) % 3
        W1[oc, ic, ky, kx] = 1.0
        pos[oc] = (ic, ky, kx)
    add_init("W1", W1)
    nodes.append(helper.make_node("Conv", ["input", "W1"], ["c1"],
                                  kernel_shape=[3, 3], strides=[1, 1],
                                  pads=[1, 1, 1, 1], name="conv1"))
    nodes.append(helper.make_node("Relu", ["c1"], ["r1"], name="relu1"))

    # ── conv2: 16→8, k=1，权重为递增常数（便于看 1×1 布局）──
    W2 = np.zeros((8, 16, 1, 1), dtype=np.float32)
    for oc in range(8):
        W2[oc, :, 0, 0] = 1.0
    add_init("W2", W2)
    nodes.append(helper.make_node("Conv", ["r1", "W2"], ["c2"],
                                  kernel_shape=[1, 1], name="conv2"))
    nodes.append(helper.make_node("Relu", ["c2"], ["r2"], name="relu2"))

    # ── 收尾：全局池化 + Gemm ──
    nodes.append(helper.make_node("GlobalAveragePool", ["r2"], ["gap"], name="gap"))
    nodes.append(helper.make_node("Flatten", ["gap"], ["flat"], axis=1, name="flat"))
    Wf = np.zeros((4, 8), dtype=np.float32)
    for i in range(4):
        Wf[i, i % 8] = 1.0
    add_init("Wf", Wf)
    add_init("Bf", np.zeros(4, dtype=np.float32))
    nodes.append(helper.make_node("Gemm", ["flat", "Wf", "Bf"], ["output"],
                                  transB=1, name="fc"))

    g = helper.make_graph(nodes, "probe_layout", [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])
    ], [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    print(f"已生成 {out_path}")
    print(f"  conv1 one-hot 位置（oc → (ic,ky,kx)）: { {k: v for k, v in list(pos.items())[:8]} } …")
    print(f"  conv1 参数量 {W1.size}（真实）  通道对齐后 3→16、16→32 会放大")
    return out_path


def make_probe2(out_path, size=32, oc=16, ic=3, k=3):
    """每通道一个常数值：W[oc] = oc+1。全部非零 ⇒ 与 padding(zero_point) 天然分开。"""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    nodes, inits = [], []
    W = np.zeros((oc, ic, k, k), dtype=np.float32)
    for o in range(oc):
        W[o] = float(o + 1)          # 每个输出通道一个可区分的常数值
    inits.append(numpy_helper.from_array(W, "W"))
    nodes.append(helper.make_node("Conv", ["input", "W"], ["c"], kernel_shape=[k, k],
                                  strides=[1, 1], pads=[1, 1, 1, 1], name="conv"))
    nodes.append(helper.make_node("Relu", ["c"], ["r"], name="relu"))
    nodes.append(helper.make_node("GlobalAveragePool", ["r"], ["g"], name="gap"))
    nodes.append(helper.make_node("Flatten", ["g"], ["f"], axis=1, name="flat"))
    Wf = np.zeros((2, oc), dtype=np.float32); Wf[0, 0] = 1.0; Wf[1, 1] = 1.0
    inits.append(numpy_helper.from_array(Wf, "Wf"))
    inits.append(numpy_helper.from_array(np.zeros(2, dtype=np.float32), "Bf"))
    nodes.append(helper.make_node("Gemm", ["f", "Wf", "Bf"], ["output"], transB=1, name="fc"))
    g = helper.make_graph(nodes, "probe_layout2", [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, ic, size, size])
    ], [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
    onnx.checker.check_model(m); onnx.save(m, out_path)
    print(f"已生成 {out_path}：conv {ic}->{oc} k{k}，W[oc]={1}..{oc}（每通道一个常数）")
    return out_path


def analyze2(bin_path):
    """找出所有非 (0,128) 的取值，报它们的出现次数与位置 ⇒ 读出通道步长"""
    import collections
    d = np.frombuffer(open(bin_path, "rb").read(), dtype=np.uint8)
    print(f"文件 {bin_path}：{len(d)} 字节")
    vals, cnts = np.unique(d, return_counts=True)
    print(f"取值种类 {len(vals)}：{list(zip(vals.tolist(), cnts.tolist()))[:20]}")
    for v in vals:
        if v in (0, 128):
            continue
        pos = np.nonzero(d == v)[0]
        gaps = np.diff(pos) if len(pos) > 1 else np.array([0])
        gc = collections.Counter(gaps.tolist())
        print(f"  值 {v:>3}: 出现 {len(pos):>4} 次  首 {pos[0]:>6}  末 {pos[-1]:>6}  "
              f"间距 top3 {gc.most_common(3)}")
    return d


def analyze(bin_path, w1_oc=16, w1_ic=3, k=3, w2_oc=8, w2_ic=16):
    d = np.frombuffer(open(bin_path, "rb").read(), dtype=np.uint8)
    print(f"文件 {bin_path}：{len(d)} 字节")
    nz = np.nonzero(d)[0]
    print(f"非零字节 {len(nz)} 个（占比 {100*len(nz)/len(d):.2f}%）")
    print(f"前 40 个非零位置: {nz[:40].tolist()}")
    if len(nz) > 1:
        gaps = np.diff(nz)
        import collections
        print(f"相邻非零间距分布（前 12）: {collections.Counter(gaps.tolist()).most_common(12)}")
    # 值分布
    vals, cnts = np.unique(d, return_counts=True)
    print(f"取值种类 {len(vals)}；最常见 10 个: {list(zip(vals.tolist(), cnts.tolist()))[:10]}")
    print()
    print("★ 读法：conv1 的 16 个输出通道各有一个非零权重。")
    print("  若布局是 [oc][ic][ky][kx] 连续，非零间距应反映 oc 的步长；")
    print("  若间距里出现某个固定值（如 16/32 的倍数），那就是**通道对齐的步长**。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--make2", action="store_true")
    ap.add_argument("--analyze2", default=None)
    ap.add_argument("--analyze", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "onnx", "probe_layout.onnx"))
    a = ap.parse_args()
    if a.make:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        make_probe(a.out)
    if a.analyze:
        analyze(a.analyze)
    if a.make2:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        make_probe2(a.out)
    if a.analyze2:
        analyze2(a.analyze2)
    if not a.make and not a.analyze:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
