#!/usr/bin/env python3
"""
fix_onnx_pool.py — 把 ONNX 图里的 `ReduceMean(keepdims=1)` 改写成 `GlobalAveragePool`

## 为什么

`mnasnet0_5` 与 `shufflenet_v2_x0_5` 转换失败在同一行：

```
Invalid rank for input: /ReduceMean_output_0  Got: 4  Expected: 2
```

torch 把 `AdaptiveAvgPool2d(1)` 导成了 `ReduceMean(axes=[2,3], keepdims=1)`，
rknn-toolkit2 内部那个 onnxruntime 会话对它要求的 rank 与图上实际的不一致。
`GlobalAveragePool` 是同一个语义但**是标准池化算子**，RKNN 的原生支持更好。

## 做法

对每个 `ReduceMean` 节点：若 `axes` 覆盖空间维（2,3）且 `keepdims=1`，
换成 `GlobalAveragePool`（输入输出形状不变，只换算子类型），其余保持不动。

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/fix_onnx_pool.py --in deploy/onnx_tv --out deploy/onnx_fix [--only NAME]
"""
import argparse
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def rewrite(path, out_path):
    m = onnx.load(path)
    g = m.graph
    # 收集常量 axes
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    n_repl = 0
    new_nodes = []
    for node in g.node:
        if node.op_type == "ReduceMean":
            attrs = {a.name: a for a in node.attribute}
            axes = None
            if "axes" in attrs:
                axes = list(attrs["axes"].ints)
            elif len(node.input) > 1 and node.input[1] in consts:
                axes = list(np.atleast_1d(consts[node.input[1]]))
            keep = attrs["keepdims"].i if "keepdims" in attrs else 1
            # 空间维在 NCHW 下是 2,3；也接受负索引
            if axes is not None and set(int(a) % 4 for a in axes) == {2, 3}:
                # 语义：对空间维求均值。
                #  · keepdims=1 ⇒ 输出 rank 4 ⇒ GlobalAveragePool 直接替换
                #  · keepdims=0 ⇒ 输出 rank 2 ⇒ GAP 之后再 Flatten(axis=1) 把 rank 拉回来
                #    （实测 torch 导出的是 keepdims=0，而 RKNN 内部 ORT 会话却按 rank 4 处理它）
                mid = node.output[0] + "_gap4d"
                new_nodes.append(helper.make_node(
                    "GlobalAveragePool", [node.input[0]], [mid],
                    name=node.name + "_gap"))
                if keep == 1:
                    new_nodes.append(helper.make_node(
                        "Identity", [mid], [node.output[0]],
                        name=node.name + "_id"))
                else:
                    new_nodes.append(helper.make_node(
                        "Flatten", [mid], [node.output[0]], axis=1,
                        name=node.name + "_flat"))
                n_repl += 1
                continue
        new_nodes.append(node)
    if n_repl == 0:
        return 0, os.path.getsize(path)
    del g.node[:]
    g.node.extend(new_nodes)
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    return n_repl, os.path.getsize(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ind", required=True)
    ap.add_argument("--out", dest="outd", required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(a.outd, exist_ok=True)
    files = sorted(f for f in os.listdir(a.ind) if f.endswith(".onnx"))
    if a.only:
        files = [f for f in files if f[:-5] == a.only]
    print(f"{'模型':<24} {'改写数':>7} {'ONNX KB':>10}")
    print("─" * 46)
    for f in files:
        src = os.path.join(a.ind, f)
        dst = os.path.join(a.outd, f)
        n, sz = rewrite(src, dst)
        print(f"{f[:-5]:<24} {n:>7} {sz/1024:>10.1f}")
    print(f"\n输出 {a.outd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
