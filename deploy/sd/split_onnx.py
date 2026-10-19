#!/usr/bin/env python3
"""split_onnx.py — 把大 ONNX **按块边界切成若干子图**，让 RKNN 能逐段转换

## 为什么需要它

`rknn-toolkit2` 是**纯 CPU 工具链**（不走 CUDA），转换 865 M 参数的 SD UNet 时
峰值内存 ~6.5 GB，而本机 WSL 只有 7 GB ⇒ **每次都被 OOM kill**
（`optimization_level=0` 也只降到 6.5 GB）。

**⇒ 把图切开，每段单独转。** 每段 ~200–300 M 参数 ⇒ 峰值 ~2 GB，稳进。

## 关键点：切点处要**带过去的张量**

UNet 的 up 段要消费 down 段的 **skip 连接**（12~13 条）。
所以每段的输出必须包含「**本段产生、但后面还要用**」的全部张量。
本脚本**自动算这个集合**（不靠手写张量名）。

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V deploy/sd/split_onnx.py --in deploy/sd/rknn/sdunet_l32_fp16.onnx \
      --outdir deploy/sd/parts --cuts down_blocks.2 down_blocks.4 up_blocks.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import onnx
from onnx.utils import extract_model


def build_index(g):
    """返回 (节点列表, {张量名: 生产它的节点下标}, {张量名: [消费它的节点下标]})"""
    nodes = list(g.node)
    producer, consumers = {}, {}
    for i, n in enumerate(nodes):
        for o in n.output:
            if o:
                producer[o] = i
        for x in n.input:
            if x:
                consumers.setdefault(x, []).append(i)
    for init in g.initializer:
        producer.setdefault(init.name, -1)          # 初始权重：-1 表示「图外」
    for inp in g.input:
        producer.setdefault(inp.name, -1)
    return nodes, producer, consumers


def boundary_names(g, marker):
    """找「名字含 marker 的**第一组**节点」的**外部**输入张量（切点张量）"""
    nodes, producer, _ = build_index(g)
    idx = [i for i, n in enumerate(nodes) if marker in n.name]
    if not idx:
        raise SystemExit(f"[split] 找不到含 {marker!r} 的节点")
    lo, hi = idx[0], idx[-1]
    produced_inside = set()
    for i in range(lo, hi + 1):
        produced_inside.update(x for x in nodes[i].output if x)
    ext = []
    for i in range(lo, hi + 1):
        for x in nodes[i].input:
            if x and x not in produced_inside and x not in ext:
                ext.append(x)
    return ext, lo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cuts", nargs="+", required=True,
                    help="切点标记（节点名子串），按图顺序给，如 down_blocks.2 mid_block")
    ap.add_argument("--prefix", default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    pref = a.prefix or os.path.splitext(os.path.basename(a.inp))[0]

    m = onnx.load(a.inp, load_external_data=False)     # ★ 不读权重，省内存
    g = m.graph
    nodes, producer, consumers = build_index(g)
    n_nodes = len(nodes)
    print(f"[split] 总节点 {n_nodes}  输入 {[i.name for i in g.input]}  输出 {[o.name for o in g.output]}")

    # 每个切点的节点下标（取该标记的第一组节点的**起始**下标）
    cut_at = []
    for c in a.cuts:
        _, lo = boundary_names(g, c)
        cut_at.append(lo)
    cut_at = sorted(set(cut_at))
    segs = []
    prev = 0
    for c in cut_at:
        segs.append((prev, c))
        prev = c
    segs.append((prev, n_nodes))
    print(f"[split] 切成 {len(segs)} 段：{segs}")

    manifest = []
    for si, (lo, hi) in enumerate(segs):
        produced_before = set()
        for i in range(0, hi):
            produced_before.update(x for x in nodes[i].output if x)
        need_out = set()
        for i in range(hi, n_nodes):
            for x in nodes[i].input:
                if x and x in produced_before:
                    need_out.add(x)
        # 本段自己的输入
        produced_here = set()
        for i in range(lo, hi):
            produced_here.update(x for x in nodes[i].output if x)
        ins = []
        for i in range(lo, hi):
            for x in nodes[i].input:
                if x and x not in produced_here and x not in ins:
                    ins.append(x)
        # 最后一段必须保留图的原始输出
        outs = list(need_out)
        if hi == n_nodes:
            outs = [o.name for o in g.output]
        name = f"{pref}_p{si}"
        dst = os.path.join(a.outdir, name + ".onnx")
        extract_model(a.inp, dst, ins, outs)
        sz = os.path.getsize(dst) / 1e6
        print(f"[split] {name}: 节点 {lo}..{hi-1}（{hi-lo} 个）  输入 {len(ins)}  输出 {len(outs)}  "
              f"→ {sz:.1f} MB")
        manifest.append({"name": name, "onnx": dst, "nodes": [lo, hi],
                         "inputs": ins, "outputs": outs})
    json.dump(manifest, open(os.path.join(a.outdir, "manifest.json"), "w"), indent=1)
    print(f"[split] 清单 → {os.path.join(a.outdir, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
