#!/usr/bin/env python3
"""rtti_find.py — 从**被 strip 的** librknnc.so 里，用 RTTI + 重定位找出任意 C++ 类的虚函数地址

## 为什么需要它

`librknnc.so` 的 `.symtab` 被剥离（只剩 `.dynsym`，且不含 pass 符号），
所以 `gdb -ex "break RKNNTilingPass::run"` 找不到符号（docs/118）。
**但 C++ RTTI 还在 `.rodata`，vtable 在 `.data`，而重定位表 `.rela.dyn` 是完整的。**

⇒ 路径：RTTI 名 → typeinfo → vtable → **虚函数地址**（全部从文件静态解析，不需要 gdb）。

## 用法
  python3 src/compiler/rtti_find.py RKNNTilingPass
  python3 src/compiler/rtti_find.py RKNNLayoutMatchPass RKNNWeightTransposePass

## 拿到地址之后（用 tools/get-gdb.sh 装好的 gdb）
  gdb -p <python_pid>
  (gdb) info sharedlibrary            # 拿到 librknnc.so 的加载基址 BASE
  (gdb) break *BASE+0x391450          # 下断点（地址要加基址）
  (gdb) continue
"""
from __future__ import annotations

import re
import struct
import subprocess
import sys

SO = "/home/dunaandone/rknn-venv/lib/python3.12/site-packages/rknn/api/lib/linux-x86_64/librknnc.so"


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True).stdout


# 实测值（readelf -SW librknnc.so）：段名 → (file_off, vaddr, size)
# 用常量兜底，避免 readelf 输出格式差异导致解析失败（踩过一次）
KNOWN = {
    ".rodata": (0x1a84000, 0x1a84000, 0x43cfc - 0x0),
    ".text":   (0x11b260, 0x11b260, 0x1968d52),
    ".data":   (0x1e14940, 0x1e15940, 0x9e8),
    ".data.rel.ro": (0x1dc22a0, 0x1dc32a0, 0x20000),
}


def segments():
    """返回 {段名: (file_off, vaddr, size)}；解析失败则用 KNOWN 兜底"""
    out = dict(KNOWN)
    for ln in sh("readelf", "-SW", SO).splitlines():
        parts = ln.split()
        # 形如: [13] .rodata PROGBITS 0000000001a84000 01a84000 043cfc ...
        if len(parts) >= 6 and parts[0].startswith("["):
            try:
                nm = parts[1]
                va = int(parts[3], 16); off = int(parts[4], 16); sz = int(parts[5], 16)
                out[nm] = (off, va, sz)
            except (ValueError, IndexError):
                pass
    return out


def relocs():
    """返回 {被重定位的地址: 目标值}（R_X86_64_RELATIVE / R_X86_64_64）"""
    r = {}
    for ln in sh("readelf", "-rW", SO).splitlines():
        m = re.match(r"([0-9a-f]{16})\s+[0-9a-f]+\s+(R_X86_64_\w+)\s+(?:[0-9a-f]+\s+\S+\s+\+\s*)?([0-9a-f]+)?", ln)
        if not m:
            continue
        off = int(m.group(1), 16)
        val = int(m.group(3), 16) if m.group(3) else None
        if val is not None:
            r[off] = val
    return r


def find(name: str):
    d = open(SO, "rb").read()
    seg = segments()
    ro_off, ro_va, ro_sz = seg[".rodata"]
    text_va, text_sz = seg[".text"][1], seg[".text"][2]
    text_end = text_va + text_sz
    # 1) 找 RTTI 名字字符串
    #    ⚠️ .rodata 里存的是 **mangled** 名（如 N4rknn14RKNNTilingPassE），
    #    所以按**子串**找，而不是精确匹配（踩过一次）
    idx = d.find(name.encode(), ro_off, ro_off + ro_sz)
    if idx < 0:
        print(f"  找不到 RTTI 名 {name}")
        return None
    # 回退到 mangled 名的**起点**（子串命中点通常在名字中间）
    ok = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
    while idx > ro_off and d[idx - 1] in ok:
        idx -= 1
    name_va = ro_va + (idx - ro_off)
    end = d.find(b"\x00", idx)
    print(f"  mangled 名 = {d[idx:end].decode('ascii', 'replace')}")
    # 2) 在重定位表里找指向该名字的项 ⇒ 那是 typeinfo 的 name 指针
    rl = relocs()
    ti_at = [off for off, val in rl.items() if val == name_va]
    if not ti_at:
        print(f"  没有重定位指向 {hex(name_va)}（可能该 RTTI 未被使用）")
        return None
    print(f"★ {name}")
    print(f"  RTTI 名 @vaddr {hex(name_va)}")
    for t in ti_at:
        ti = t - 8                      # typeinfo: [vtable][name]
        print(f"  typeinfo @vaddr {hex(ti)}")
        # 3) 找指向该 typeinfo 的重定位 ⇒ 那是 vtable 的第 2 项
        vt2 = [off for off, val in rl.items() if val == ti]
        for v in vt2:
            print(f"  typeinfo 指针位于 @vaddr {hex(v)}")
            # ★ 向前扫描最多 16 项，筛出落在 .text 内的（= 虚函数地址）。
            #   不做「vtable 起点 = v-8」的算术 —— 多重继承/虚基类会让布局更复杂（踩过）。
            codes = []
            for k in range(0, 16):
                off = v + 8 * k
                val = rl.get(off)
                if val is None:
                    continue
                if text_va <= val < text_end:
                    codes.append((off, val))
                    print(f"      ★ CODE @vaddr {hex(off)} = {hex(val)}")
            if codes:
                print(f"  ⇒ {len(codes)} 个虚函数地址（gdb 断点用这些，需加加载基址）")
            else:
                print("      （向前 16 项内没有 .text 地址）")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for nm in sys.argv[1:]:
        find(nm)
        print()
