#!/usr/bin/env python3
"""probe_compiler.py — RK3588 NPU 编译器内部研究工具（把 33 轮的临时命令沉淀成可复用入口）

用法（本机、离线、不上板）：
  python3 -m src.compiler.probe_compiler rules                 # 导出三张规则表
  python3 -m src.compiler.probe_compiler passes  <onnx>        # 提取编译流水线 pass 序列
  python3 -m src.compiler.probe_compiler ir      <onnx>        # 打印最终 IR 表（含 Cycles/RW/Target）
  python3 -m src.compiler.probe_compiler fire    <onnx>        # 捕获规则触发（monkeypatch）

## 关键机制（33 轮里最有用的两条）

1. **C++ 编译器会回调 Python 的规则 `proc`** ⇒ 包装 `GO.<table>[name]['proc']` 就能观测
   全部规则执行（docs/109）。
2. **`verbose=True` 的 build 会打印最终 IR 表**（含 `Format=NC1HWC2` / `Cycles(DDR/NPU/Total)`
   / `RW(KB)` / `Target` / `FullName`）与 33 个 pass 的 start/end（docs/100/103/104）。

## 已知边界（不要在这上面浪费时间）

* `layout` 属性由 `RKNNLayoutMatchPass`（第 20 步）添加，而可 hook 的规则在第 19 步及之前
  ⇒ **在这里读不到 layout**（docs/113）。要读需在流水线更后面观测。
* `hardware_rules` 的 `proc` 虽然触发，但**大多数返回空** ⇒ **不是切分机制**（docs/111）。
"""
from __future__ import annotations

import argparse
import collections
import importlib
import re
import sys

GO = importlib.import_module("rknn.api.graph_optimizer")

CYCLE_ROW = re.compile(
    r"\]\s+(\d+)\s+([A-Za-z][A-Za-z0-9_]*)\s+"
    r"(INT8|INT32|INT64|FLOAT|UINT8|BOOL)\s+(CPU|NPU)\s+.*?\s+(\d+)/(\d+)/(\d+)\s+(\d+)\s+(\S+)")
PASS_END = re.compile(r"end: rknn::([A-Za-z0-9_]+)")


def cmd_rules(_a):
    for tbl in ("fuse_rules", "convert_rules", "hardware_rules"):
        t = getattr(GO, tbl)
        print(f"##### {tbl} 共 {len(t)} 条 #####")
        for k, v in sorted(t.items()):
            print(f"{k:<42} {'→'.join(v.get('ops', []))}")
        print()


def _build(onnx, verbose=False, fire=False):
    """跑一次 build，返回 (日志行列表, 规则触发计数)"""
    import io
    import os
    import tempfile
    from rknn.api import RKNN
    fired = collections.Counter()
    if fire:
        def wrap(tbl, name, orig):
            def w(*a, **kw):
                fired[(tbl, name)] += 1
                return orig(*a, **kw)
            return w
        for tbl in ("fuse_rules", "convert_rules", "hardware_rules"):
            for name, spec in getattr(GO, tbl).items():
                if spec.get("proc") is not None:
                    spec["proc"] = wrap(tbl, name, spec["proc"])
    r = RKNN(verbose=verbose)
    r.config(target_platform="rk3588", mean_values=[[0, 0, 0]],
             std_values=[[255, 255, 255]], quantized_dtype="w8a8")
    r.load_onnx(model=onnx)
    # ⚠️ verbose 输出走的是 C 层 fd（不是 Python stdout）⇒ 必须做 **fd 级**重定向，
    #    用 contextlib.redirect_stdout 会抓到空（踩过）。
    if verbose:
        fd, path = tempfile.mkstemp(suffix=".log")
        saved = os.dup(1)
        os.dup2(fd, 1)
        try:
            r.build(do_quantization=True,
                    dataset="/home/dunaandone/rknn-work/calib/dataset.txt")
        finally:
            os.dup2(saved, 1)
            os.close(saved)
            os.close(fd)
        r.release()
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
        os.unlink(path)
        return lines, fired
    r.build(do_quantization=True,
            dataset="/home/dunaandone/rknn-work/calib/dataset.txt")
    r.release()
    return [], fired


def cmd_passes(a):
    lines, _ = _build(a.onnx, verbose=True)
    seen = []
    for ln in lines:
        m = PASS_END.search(ln)
        if m and m.group(1) not in seen:
            seen.append(m.group(1))
    print(f"共 {len(seen)} 个 pass：")
    for i, p in enumerate(seen, 1):
        print(f"{i:>3} {p}")


def cmd_ir(a):
    lines, _ = _build(a.onnx, verbose=True)
    tot = ddr = npu = rw = n = cpu = 0
    for ln in lines:
        m = CYCLE_ROW.search(ln)
        if not m:
            continue
        n += 1
        if m.group(4) == "CPU":
            cpu += 1
        ddr += int(m.group(5)); npu += int(m.group(6)); rw += int(m.group(8))
        if a.show:
            print(f"{m.group(1):>3} {m.group(2):<24} {m.group(4)} "
                  f"{m.group(5)}/{m.group(6)}/{m.group(7)}  RW={m.group(8)}KB  {m.group(9)[:40]}")
    print(f"\n算子 {n}（CPU {cpu}）  ΣDDR={ddr} ΣNPU={npu} ΣRW={rw}KB")
    print(f"⇒ 编译器估算（ΣTotal/1GHz）= {npu/1e9*1000:.3f} ms")


def cmd_fire(a):
    lines, fired = _build(a.onnx, verbose=False, fire=True)
    print(f"规则触发总数 {sum(fired.values())}，涉及 {len(fired)} 条规则")
    for (tbl, name), c in fired.most_common(a.top):
        print(f"  [{tbl}] {name:<42} {c}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rules").set_defaults(fn=cmd_rules)
    for name, fn in (("passes", cmd_passes), ("ir", cmd_ir), ("fire", cmd_fire)):
        p = sub.add_parser(name)
        p.add_argument("onnx")
        p.set_defaults(fn=fn)
        if name == "ir":
            p.add_argument("--show", action="store_true")
        if name == "fire":
            p.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
