#!/usr/bin/env python3
"""fix_onnx_split.py — 修掉 torch 新版导出器给 `Split` 节点留下的非法属性

## 症状

```
onnxruntime ... Node (node_Split_44) Op (Split) [ShapeInferenceError]
Both 'split' input and 'num_outputs' attribute were given
```

ONNX 规范：`Split` 要么给 `split` **输入**，要么给 `num_outputs` **属性**，不能同时给。
torch 2.14 的 dynamo 导出器会两个都写 ⇒ **onnxruntime 的形状推断直接失败**，
RKNN 的 `build` 也随之失败。

## 处理

对每个 `Split` 节点：**若有 `split` 输入，就删掉 `num_outputs` 属性**。

## 用法
  V=/home/dunaandone/rknn-venv/bin/python deploy/sd/fix_onnx_split.py <in.onnx> <out.onnx>
"""
from __future__ import annotations

import os
import shutil
import sys

import onnx


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__); return 1
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src, load_external_data=False)
    n = 0
    for node in m.graph.node:
        if node.op_type != "Split":
            continue
        has_split_input = len(node.input) > 1
        attrs = [a for a in node.attribute if a.name == "num_outputs"]
        if has_split_input and attrs:
            for a in attrs:
                node.attribute.remove(a)
            n += 1
    print(f"[fix] 修了 {n} 个 Split 节点")
    onnx.save(m, dst)
    # 外部数据文件要一起搬
    for suf in (".data",):
        if os.path.exists(src + suf):
            shutil.copy(src + suf, dst + suf)
            print(f"[fix] 拷贝 {dst}{suf}")
    print(f"[fix] {dst}  {os.path.getsize(dst)/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
