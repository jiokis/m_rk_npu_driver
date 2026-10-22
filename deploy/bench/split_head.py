"""通用修复：把产生 output0 的 Concat 拆成多个独立输出，避开 INT8 共享量化 scale"""
import sys, os, onnx
from onnx import TensorProto, helper

TASKS=["pose","seg","obb"]
for t in TASKS:
    src=f"yolo11n-{t}.onnx"; dst=f"yolo11n-{t}-split.onnx"
    m=onnx.load(src); g=m.graph
    concat=None
    for n in g.node:
        if n.op_type=="Concat" and "output0" in n.output:
            concat=n; break
    if concat is None:
        print(f"SKIP {t}: 没找到 output0 的 Concat"); continue
    outs=list(concat.input)
    other=[o.name for o in g.output if o.name not in ("output0",)]
    allouts=outs+other
    onnx.utils.extract_model(src, dst, ["images"], allouts)
    m2=onnx.load(dst)
    print(f"OK {t}: 拆成 {len(allouts)} 个输出 {[o.name for o in m2.graph.output]}")
