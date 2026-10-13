#!/usr/bin/env python3
"""ir_table.py — 从 **verbose build 的最终 IR 表** 直接读算子清单（不需要 gdb）

## 为什么这个工具很重要（docs/132）

之前我都是**用析构次数反推**算子数（`tools/gdb/probe.py --off 0x391510`）——
间接、慢、而且连「+2 是常数」这种错误都暴露不出来（docs/131）。

其实编译器**自己就把最终 IR 打出来了**（`RKNN(verbose=True)`）：

```
ID   OpType        DataType Target InputShape ... Cycles(DDR/NPU/Total)  RW(KB)  FullName
0    InputOperator INT8     CPU    ...
1    ConvRelu      INT8     NPU    (1,3,224,224),(64,3,7,7),(64)  ...  Conv:/conv1/Conv
2    MaxPool       INT8     NPU    ...
...
```

**⇒ 这是一份**直接的**算子清单：ID / 类型 / 目标端 / 形状 / 编译器自己的周期估计 / 原始名字。**
**⇒ 不用断点、不用扫描、不用反汇编。**

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V src/compiler/ir_table.py deploy/onnx_tv/resnet18.onnx
  $V src/compiler/ir_table.py --json out.json <onnx> [<onnx> ...]

## 输出
  每个模型：NPU 算子数 / 按 OpType 分类 / 按 Target 分类
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(REPO, "tools", "gdb", "build_min.py")
PYTHON = "/home/dunaandone/rknn-venv/bin/python"

# 形如： "D RKNN: [10:45:48.706] 1    ConvRelu   INT8   NPU   (...)  (...)  42447/1229312/1229312  196  Conv:/conv1/Conv"
ROW = re.compile(
    r"^\s*\S+\s*RKNN:\s*\[[^\]]+\]\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$"
)
# 表头行：用于定位「最后一张表」
HEADER = re.compile(r"\bID\s+OpType\s+DataType\s+Target\b")


def parse_ir_table(text: str) -> list[dict]:
    """取出**最后一张** IR 表（verbose 可能打印多张）"""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if HEADER.search(ln)]
    if not starts:
        return []
    s = starts[-1]
    rows: list[dict] = []
    for ln in lines[s + 1:]:
        if "----" in ln and rows:
            break
        m = ROW.match(ln)
        if not m:
            continue
        rest = m.group(5).split()
        cycles = next((x for x in rest if re.fullmatch(r"[\d/]+", x) and "/" in x), "")
        rows.append({
            "id": int(m.group(1)),
            "op_type": m.group(2),
            "dtype": m.group(3),
            "target": m.group(4),
            "cycles": cycles,
            "full_name": rest[-1] if rest else "",
        })
    return rows


def build_ir_table(onnx: str, quant: bool = True) -> list[dict]:
    env = dict(os.environ)
    env["BUILD_MIN_VERBOSE"] = "1"
    env["BUILD_MIN_ONNX"] = onnx
    if not quant:
        env["BUILD_MIN_NOQUANT"] = "1"
    p = subprocess.run([PYTHON, BUILD], capture_output=True, text=True, env=env, timeout=1800)
    return parse_ir_table(p.stdout + p.stderr)


def summarize(rows: list[dict]) -> dict:
    npu = [r for r in rows if r["target"] == "NPU"]
    conv_like = [r for r in npu if "Conv" in r["op_type"]]
    other_npu = [r for r in npu if "Conv" not in r["op_type"]]
    return {
        "rows": len(rows),
        "npu": len(npu),
        "conv_like_on_npu": len(conv_like),
        "other_npu": len(other_npu),
        "other_npu_types": dict(Counter(r["op_type"] for r in other_npu)),
        "targets": dict(Counter(r["target"] for r in rows)),
        "npu_types": dict(Counter(r["op_type"] for r in npu)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("--json", default=None, help="把完整结果写到该文件")
    ap.add_argument("--noquant", action="store_true")
    args = ap.parse_args()

    allout = {}
    for p in args.onnx:
        rows = build_ir_table(p, quant=not args.noquant)
        s = summarize(rows)
        allout[os.path.basename(p)] = {"summary": s, "rows": rows}
        print(f"=== {os.path.basename(p)} ===")
        print(f"  总行数 {s['rows']}   Target: {s['targets']}")
        print(f"  ★ NPU 上「像 Conv」的算子 = {s['conv_like_on_npu']}"
              f"   （NPU 合计 {s['npu']}，非卷积 {s['other_npu']}: {s['other_npu_types']}）")
        print(f"  NPU 上的类型分布: {s['npu_types']}")
    if args.json:
        json.dump(allout, open(args.json, "w"), ensure_ascii=False, indent=1)
        print(f"\n[ir_table] 已写 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
