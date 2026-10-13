#!/usr/bin/env python3
"""scan_heap.py — 在编译**跑完之后**扫描进程内存，按 **vtable 指针**找回对象

## 为什么需要它（这是本轮的关键手段）

`rknn::RKNNOperatorTilingInfo` 是 **非多态类**（没有自己的 RTTI/vtable，docs/126），
所以**不能用「提取虚函数地址 → 下断点」**那套（docs/120-125）去找它。

但它由 **`std::make_shared`** 创建 ⇒ 堆上有一个
`std::_Sp_counted_ptr_inplace<rknn::RKNNOperatorTilingInfo, ...>` 控制块，
**控制块的 vptr 就是它的 vtable**（静态可求，本轮已用反汇编确认：
该 vtable 的 `_M_get_deleter` 槽在比对 `_ZTSSt19_Sp_make_shared_tag`）。

⇒ **在内存里搜这个 vtable 指针的值**，就能找到控制块 ⇒ 对象在 `控制块 + 0x10`
   （libstdc++：`_Sp_counted_base` 16 字节，其后紧跟 `__aligned_buffer<_Tp>`）。

## 为什么要「跑完之后」再扫

tiling 发生在流水线中段；跑完之后对象已建好且（实测）**不会被析构**
（它的 `_M_dispose` 在整个 build 里 0 命中）⇒ 扫描时它一定在。

## 用法
  V=/home/dunaandone/rknn-venv/bin/python
  $V tools/gdb/scan_heap.py --vtable 0x1dc4670 --dump-off 0x58 --dump-len 0x30

## 产物
  data/gdb/scan_<tag>.log
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GDB = "/tmp/gdbbin/root/usr/bin/gdb"
GDB_LIB = "/tmp/gdbbin/root/usr/lib/x86_64-linux-gnu"
PYTHON = "/home/dunaandone/rknn-venv/bin/python"
BUILD = os.path.join(REPO, "tools", "gdb", "build_min.py")
OUTDIR = os.path.join(REPO, "data", "gdb")
TEXT_VADDR = 0x11B260

# gdb 内嵌 Python：等 build 跑完 → 扫 rw 内存找 vtable 指针 → dump 命中对象
SCAN_PY = r'''
import gdb, re, struct


def rw_regions(pid):
    out = []
    for ln in open(f"/proc/{pid}/maps"):
        p = ln.split()
        if len(p) < 2 or "w" not in p[1]:
            continue
        a, b = p[0].split("-")
        lo, hi = int(a, 16), int(b, 16)
        if hi - lo > 0x40000000:      # 跳过离谱的大映射
            continue
        out.append((lo, hi))
    return out


def scan(inf, value, regions):
    pat = struct.pack("<Q", value)
    hits = []
    CH = 0x100000
    for lo, hi in regions:
        off = lo
        while off < hi:
            n = min(CH, hi - off)
            try:
                buf = bytes(inf.read_memory(off, n))
            except Exception:
                off += n
                continue
            i = buf.find(pat)
            while i >= 0:
                hits.append(off + i)
                i = buf.find(pat, i + 1)
            off += n
    return hits


def qw(inf, addr, n):
    try:
        b = bytes(inf.read_memory(addr, 8 * n))
    except Exception:
        return None
    return [int.from_bytes(b[8 * k:8 * k + 8], "little") for k in range(n)]


def main():
    vt_off = SCAN_VTABLE
    dump_off, dump_len, deref = SCAN_DUMP_OFF, SCAN_DUMP_LEN, SCAN_DEREF
    out = gdb.execute("info sharedlibrary librknnc", to_string=True)
    m = re.search(r"(0x[0-9a-f]+)\s+0x", out)
    if not m:
        gdb.write("[scan] 解析 sharedlibrary 失败\n")
        return
    base = int(m.group(1), 16) - SCAN_TEXT_VADDR
    target = base + vt_off
    gdb.write(f"[scan] base={base:#x}  vtable={target:#x}\n")
    inf = gdb.selected_inferior()
    pid = inf.pid
    regs = rw_regions(pid)
    gdb.write(f"[scan] 可写内存区 {len(regs)} 个，共 "
              f"{sum(h - l for l, h in regs) / 2**20:.1f} MiB\n")
    hits = scan(inf, target, regs)
    gdb.write(f"[scan] 命中 vtable 指针 {len(hits)} 处\n")
    for h in hits[:SCAN_MAX]:
        gdb.write(f"\n=== 控制块 @ {h:#x}  对象 @ {h + 0x10:#x} ===\n")
        cb = qw(inf, h, 4)
        if cb:
            gdb.write("  cb: " + " ".join(f"{x:#x}" for x in cb) + "\n")
        obj = qw(inf, h + 0x10, (dump_off + dump_len) // 8 + 1)
        if not obj:
            gdb.write("  <对象不可读>\n")
            continue
        for i in range(0, dump_off // 8):
            gdb.write(f"  obj[{i:3d}] {obj[i]:#018x}\n")
        for i in range(dump_off // 8, min(len(obj), (dump_off + dump_len) // 8)):
            gdb.write(f"  obj[{i:3d}] {obj[i]:#018x}\n")
        # 把 dump 区间里像指针的槽位解引用
        for i in range(dump_off // 8, min(len(obj), (dump_off + dump_len) // 8)):
            v = obj[i]
            if v < 0x10000 or v > 0x7FFFFFFFFFFF:
                continue
            w = qw(inf, v, deref)
            if w:
                gdb.write(f"    obj[{i}] -> {v:#x}: " + " ".join(f"{x:#x}" for x in w) + "\n")


main()
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtable", required=True, help="要找的 vtable 库内 vaddr，如 0x1dc4670")
    ap.add_argument("--dump-off", default="0x50", help="从对象该偏移开始 dump")
    ap.add_argument("--dump-len", default="0x40", help="dump 长度")
    ap.add_argument("--deref", type=int, default=4, help="解引用槽位时读几个 quadword")
    ap.add_argument("--max", type=int, default=20, help="最多打印几个命中")
    ap.add_argument("--at-symbol", default="RKNNCompiler_build",
                    help="在这个导出符号处断下，finish 后扫描")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or re.sub(r"[^A-Za-z0-9_.-]", "_", args.vtable)
    os.makedirs(OUTDIR, exist_ok=True)
    log = os.path.join(OUTDIR, f"scan_{tag}.log")
    script = os.path.join(OUTDIR, f"scan_{tag}.gdb")

    body = (SCAN_PY
            .replace("SCAN_VTABLE", args.vtable)
            .replace("SCAN_TEXT_VADDR", hex(TEXT_VADDR))
            .replace("SCAN_DUMP_OFF", args.dump_off)
            .replace("SCAN_DUMP_LEN", args.dump_len)
            .replace("SCAN_DEREF", str(args.deref))
            .replace("SCAN_MAX", str(args.max)))
    # ⚠️ librknnc.so 是**运行时 dlopen** 的，脚本开头直接 `break 符号名` 会报
    #    "Function ... not defined"（踩过一次）⇒ 必须 `catch load` 等它加载后再下断。
    gdb_script = "\n".join([
        "set pagination off",
        "set confirm off",
        "set width 0",
        "set breakpoint pending on",
        "catch load librknnc.so",
        "run",
        f"break {args.at_symbol}",
        "continue",
        "finish",
        "python",
        body.strip(),
        "end",
        "quit",
    ]) + "\n"
    open(script, "w").write(gdb_script)

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = GDB_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    if args.onnx:
        env["BUILD_MIN_ONNX"] = args.onnx

    print(f"[scan] vtable={args.vtable}  at={args.at_symbol}\n[scan] 日志 {log}")
    with open(log, "w") as lf:
        p = subprocess.run([GDB, "-q", "-x", script, "--args", PYTHON, BUILD],
                           stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=3600)
    txt = open(log, errors="replace").read()
    n = re.search(r"命中 vtable 指针 (\d+) 处", txt)
    print(f"[scan] gdb exit={p.returncode}  命中 {n.group(1) if n else '?'} 处")
    for ln in txt.splitlines():
        if ln.startswith("[scan]") or ln.startswith("=== 控制块"):
            print("       " + ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
