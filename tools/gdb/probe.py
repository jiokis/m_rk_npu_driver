#!/usr/bin/env python3
"""probe.py — 用 gdb 观察 librknnc.so 内部函数（把 docs/121-125 的做法固化成工具）

## 背景（为什么需要它）

`librknnc.so` 的 `.symtab` 被剥离 ⇒ `break RKNNTilingPass::run` 找不到符号。
但 C++ RTTI + 重定位表完整 ⇒ 可以用 `src/compiler/rtti_find.py` **静态**提取虚函数地址。
再在 gdb 里 `catch load` 拿到加载基址，`break *base+off` 就能断下（docs/120-122 已验证）。

**关键点（踩过）**：ASLR 让基址每次不同（实测 `0x7fffaeC00000` / `0x7fffaeE00000`），
所以**必须先 `catch load` 再算基址**，硬编码地址一定失败。

## 用法

  V=/home/dunaandone/rknn-venv/bin/python

  # 按类名解析虚函数地址并全部下断（每个断点命中时打印 this + 6 个 quadword + bt 6）
  $V tools/gdb/probe.py --class RKNNOperatorTilingInfo

  # 按裸偏移（库内 vaddr 偏移）下断
  $V tools/gdb/probe.py --off 0x3913e0 0x391400

  # 换模型 / 调 dump 宽度 / 调命中上限
  $V tools/gdb/probe.py --class RKNNTilingPass --onnx deploy/onnx/tiny_cnn.onnx --qw 12 --cap 20

## 产物
  data/gdb/<tag>.log     gdb 原始输出（含每次命中的寄存器、内存、栈回溯）

## 注意
  * 断在**访问器**上可能被调用上千次 ⇒ 默认 `--cap 40` 后自动 `quit`。
  * 断点用 `silent` + `continue`，所以**不会停下来交互** —— 这是「普查」模式；
    要「停下来细看」，加 `--stop`（命中一次就停，随后 `bt`/`x` 由 gdb 脚本尾部执行）。
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
RTTI = os.path.join(REPO, "src", "compiler", "rtti_find.py")
OUTDIR = os.path.join(REPO, "data", "gdb")

# .text 的 ELF vaddr：base = text_load − TEXT_VADDR（docs/121 实测）
TEXT_VADDR = 0x11B260


def codes_for(cls: str) -> list[int]:
    """用 rtti_find.py 解析类名 ⇒ 虚函数地址（库内偏移）"""
    out = subprocess.run([sys.executable, RTTI, cls], capture_output=True, text=True).stdout
    print(out, end="")
    offs = [int(m, 16) for m in re.findall(r"★ CODE @vaddr 0x[0-9a-f]+ = (0x[0-9a-f]+)", out)]
    if not offs:
        raise SystemExit(f"[probe] 没从 rtti_find 拿到 {cls} 的虚函数地址")
    return offs


# gdb 内嵌 Python：一个会**自动解引用**的断点类。
#
# 为什么不用 gdb 命令语言：gdb 的 `commands` 循环能力很弱，而「dump this 的每个非零槽位
# 再解引用它」需要循环 + 异常处理。用 Python 子类 `stop()` 一次写完，比拼命令串可靠得多。
BP_CLASS = r'''
import gdb, re

class DumpBP(gdb.Breakpoint):
    """命中时打印 this 的 N 个 quadword，并把每个像指针的槽位解引用出 4 个 quadword"""
    def __init__(self, spec, off, qw, bt, stop, cap, vec_off=None, vec_stride=0):
        super().__init__(spec, internal=False)
        self.vec_off, self.vec_stride = vec_off, vec_stride
        # ⚠️ 不能叫 self.stop —— 那会把 Breakpoint.stop() **方法**覆盖成 bool（踩过一次，
        #    报错是 "attribute of type 'bool' is not callable"）
        self.off, self.qw, self.bt, self.do_stop, self.cap = off, qw, bt, stop, cap
        self.n = 0
        self.silent = True

    def stop(self):
        self.n += 1
        if self.n > self.cap:
            self.enabled = False
            return False
        inf = gdb.selected_inferior()
        rdi = int(gdb.parse_and_eval("(unsigned long)$rdi"))
        out = [f"\n=== HIT off={self.off:#x} this={rdi:#x} (hit #{self.n}) ==="]
        try:
            mem = bytes(inf.read_memory(rdi, 8 * self.qw))
        except Exception as e:
            out.append(f"  [read this failed: {e}]")
            mem = b""
        vals = []
        for i in range(len(mem) // 8):
            v = int.from_bytes(mem[8 * i:8 * i + 8], "little")
            vals.append(v)
            out.append(f"  [{i:2d}] {v:#018x}")
        for i, v in enumerate(vals):
            if v < 0x10000 or v > 0x7FFFFFFFFFFF:
                continue
            try:
                m2 = bytes(inf.read_memory(v, 32))
            except Exception:
                out.append(f"    [{i}] -> {v:#x}: <不可读>")
                continue
            w = [int.from_bytes(m2[8 * k:8 * k + 8], "little") for k in range(4)]
            out.append(f"    [{i}] -> {v:#x}: " + " ".join(f"{x:#x}" for x in w))
        # ★ 参数寄存器：System V 下 rdi=this, rsi/rdx/rcx=第 1/2/3 个参数。
        #   对 `run(Module&)` 来说 **rsi 就是 IR 图** —— 这是最有价值的观测目标。
        for rn in ("rsi", "rdx", "rcx"):
            try:
                rv = int(gdb.parse_and_eval(f"(unsigned long)${rn}"))
            except Exception:
                continue
            if rv < 0x10000 or rv > 0x7FFFFFFFFFFF:
                out.append(f"  {rn} = {rv:#x}")
                continue
            try:
                m3 = bytes(inf.read_memory(rv, 32))
            except Exception:
                out.append(f"  {rn} = {rv:#x} <不可读>")
                continue
            w3 = [int.from_bytes(m3[8 * k:8 * k + 8], "little") for k in range(4)]
            out.append(f"  {rn} = {rv:#x}: " + " ".join(f"{x:#x}" for x in w3))
        # ★ 把 `this+vec_off` 当成 std::vector 头（begin/end/cap）来解析 ——
        #   这是「数出 tile 个数」的关键：count = (end - begin) / stride
        if self.vec_off is not None and mem:
            o = self.vec_off
            if o + 24 <= len(mem):
                beg = int.from_bytes(mem[o:o + 8], "little")
                end = int.from_bytes(mem[o + 8:o + 16], "little")
                cap = int.from_bytes(mem[o + 16:o + 24], "little")
                out.append(f"  VEC@this+{o:#x}: begin={beg:#x} end={end:#x} cap={cap:#x}")
                if self.vec_stride and end >= beg and (end - beg) % self.vec_stride == 0:
                    n = (end - beg) // self.vec_stride
                    out.append(f"  ★ 元素个数 = {n}  (stride {self.vec_stride:#x})")
                    for k in range(min(n, 8)):
                        ea = beg + k * self.vec_stride
                        try:
                            eb = bytes(inf.read_memory(ea, 0x40))
                        except Exception:
                            out.append(f"    [{k}] @{ea:#x} <不可读>")
                            continue
                        hdr = [int.from_bytes(eb[8 * j:8 * j + 8], "little") for j in range(8)]
                        out.append(f"    [{k}] @{ea:#x} 头部: " + " ".join(f"{x:#x}" for x in hdr))
                        # 11 个内嵌 vector 头（间隔 0x18）
                        for j in range(11):
                            vo = ea + 0x40 + 0x18 * j
                            try:
                                vb = bytes(inf.read_memory(vo, 24))
                            except Exception:
                                continue
                            b2 = int.from_bytes(vb[0:8], "little")
                            e2 = int.from_bytes(vb[8:16], "little")
                            if b2 or e2:
                                out.append(f"        vec{j} @+{0x40 + 0x18 * j:#x}: {b2:#x}..{e2:#x} "
                                           f"(diff {e2 - b2:#x})")
                else:
                    out.append(f"  (不是 stride {self.vec_stride:#x} 的整数倍，无法算个数)")
        gdb.write("\n".join(out) + "\n")
        if self.bt:
            gdb.write(gdb.execute(f"bt {self.bt}", to_string=True))
        gdb.flush()
        return self.do_stop
'''


def make_gdb_script(offs: list[int], qw: int, bt: int, cap: int, stop: bool,
                    vec_off=None, vec_stride=0) -> str:
    """生成 gdb 脚本。基址在 gdb 里算（ASLR 每次不同），所以断点在 gdb 里设。"""
    lines = [
        "set pagination off",
        "set confirm off",
        "set width 0",
        "set print elements 0",
        "catch load librknnc.so",
        "run",
        "python",
        BP_CLASS.strip(),
        "",
        'out = gdb.execute("info sharedlibrary librknnc", to_string=True)',
        'm = re.search(r"(0x[0-9a-f]+)\\s+0x", out)',
        "if not m:",
        '    gdb.write("[probe] 解析 sharedlibrary 失败\\n")',
        "else:",
        "    text_load = int(m.group(1), 16)",
        f"    base = text_load - {TEXT_VADDR:#x}",
        '    gdb.write(f"[probe] text_load={text_load:#x}  base={base:#x}\\n")',
    ]
    for off in offs:
        lines.append(
            f"    DumpBP('*' + hex(base + {off:#x}), {off:#x}, {qw}, {bt}, "
            f"{'True' if stop else 'False'}, {cap}, {vec_off!r}, {vec_stride})"
        )
    lines += [
        "end",
        "continue",
        'printf "\\n[probe] run finished\\n"',
        "quit",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="classes", action="append", nargs="+", default=[],
                    help="C++ 类名（可多次），用 rtti_find.py 解析其虚函数地址")
    ap.add_argument("--off", dest="offs", nargs="*", default=[],
                    help="库内偏移（十六进制），如 0x3913e0")
    ap.add_argument("--onnx", default=None, help="覆盖 build_min.py 用的 ONNX")
    ap.add_argument("--noquant", action="store_true",
                    help="不做量化（非 3 通道输入的合成模型必须加：标定图片是 3 通道）")
    ap.add_argument("--qw", type=int, default=6, help="每个命中 dump 的 quadword 数")
    ap.add_argument("--bt", type=int, default=6, help="栈回溯深度（0=不回溯）")
    ap.add_argument("--cap", type=int, default=40, help="命中上限（占位，gdb 侧未强制）")
    ap.add_argument("--stop", action="store_true", help="命中即停（普查模式的反面）")
    ap.add_argument("--vec-off", default=None, help="把 this+该偏移当作 std::vector 头解析，如 0x58")
    ap.add_argument("--vec-stride", default="0", help="vector 元素大小，如 0x1a8")
    ap.add_argument("--tag", default=None, help="日志名（默认按类名/偏移自动生成）")
    args = ap.parse_args()

    vec_off = int(args.vec_off, 16) if args.vec_off else None
    vec_stride = int(args.vec_stride, 16) if args.vec_stride else 0
    offs: list[int] = []
    for c in [x for grp in args.classes for x in grp]:
        offs += codes_for(c)
    offs += [int(x, 16) for x in args.offs]
    if not offs:
        ap.error("至少要给 --class 或 --off")
    offs = sorted(set(offs))

    flat = [x for grp in args.classes for x in grp]
    tag = args.tag or ("_".join(flat) if flat else "offs")
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
    os.makedirs(OUTDIR, exist_ok=True)
    log = os.path.join(OUTDIR, f"{tag}.log")
    script = os.path.join(OUTDIR, f"{tag}.gdb")
    with open(script, "w") as f:
        f.write(make_gdb_script(offs, args.qw, args.bt, args.cap, args.stop,
                                vec_off, vec_stride))

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = GDB_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    if args.onnx:
        env["BUILD_MIN_ONNX"] = args.onnx
    if args.noquant:
        env["BUILD_MIN_NOQUANT"] = "1"

    print(f"[probe] {len(offs)} 个断点: {[hex(o) for o in offs]}")
    print(f"[probe] 脚本 {script}\n[probe] 日志 {log}")
    with open(log, "w") as lf:
        p = subprocess.run([GDB, "-q", "-x", script, "--args", PYTHON, BUILD],
                           stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=1800)
    txt = open(log, errors="replace").read()
    # ⚠️ 不能要求地址后紧跟 " ===" —— 实际格式是 "... this=0x.. (hit #N) ==="（踩过一次）
    hits = re.findall(r"=== HIT off=(0x[0-9a-f]+)\s+this=(0x[0-9a-f]+)", txt)
    print(f"[probe] gdb exit={p.returncode}  命中 {len(hits)} 次")
    for off, this in hits[:20]:
        print(f"        off={off} this={this}")
    if len(hits) > 20:
        print(f"        … 还有 {len(hits) - 20} 次，见日志")
    return 0


if __name__ == "__main__":
    sys.exit(main())
