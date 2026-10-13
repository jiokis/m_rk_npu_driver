#!/usr/bin/env python3
"""pass_list.py — 枚举 librknnc.so 里**全部**编译器 pass（从 RTTI 名字表）

## 为什么这个工具价值很高（docs/126）

在 `.rodata` 里枚举字符串时发现：RKNN 编译器的每一个 pass 都是
**`onnc::CustomPass<rknn::XxxPass, onnc::ModulePass, void>`** 的模板实例：

```
N4onnc10CustomPassIN4rknn14RKNNTilingPassENS_10ModulePassEvEE
  = onnc::CustomPass< rknn::RKNNTilingPass, onnc::ModulePass, void >
```

**⇒ 编译器骨架是 ONNC**（Open Neural Network Compiler，开源），
   RKNN 只提供 `XxxPass` 这些叶子类。

**⇒ 于是「pass 清单」是可以**静态完整枚举**的** ——
   不需要 gdb、不需要跑编译，直接从 RTTI 名解析。

## 产出
  * stdout：按名字排序的 pass 表（来源命名空间 / pass 名 / vtable / 虚函数地址）
  * `data/analysis/rknn_passes_full.txt`：同样内容（可复现、可 diff）

## 用法
  python3 src/compiler/pass_list.py                 # 全部
  python3 src/compiler/pass_list.py --filter Tile   # 只看含 Tile 的
  python3 src/compiler/pass_list.py --vtable        # 顺带解析 vtable + 虚函数地址
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtti_find as R  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "data", "analysis", "rknn_passes_full.txt")

# onnc::CustomPass< INNER, BASE, void >
RE_CUSTOM = re.compile(r"^N4onnc10CustomPassI(.+?)E$")
# 内层：N4rknn14RKNNTilingPassE  或  NS_18CountOperatorsPassE
# ⚠️ 必须用 **Itanium 的长度前缀** 切名字，不能用 `[A-Za-z0-9_]+` 贪婪匹配 ——
#    贪婪会把后面的 `ENS_10ModulePassEv` 一起吞掉（踩过一次）。
RE_INNER = re.compile(r"^N(?:(4rknn)|S_)(\d+)")


def strings(seg_name=".rodata"):
    """产出 (vaddr, str) —— 段内所有纯 ASCII 可打印串"""
    d = open(R.SO, "rb").read()
    off, va, sz = R.segments()[seg_name]
    s = d[off:off + sz]
    i = 0
    while i < len(s):
        j = s.find(b"\x00", i)
        if j < 0:
            break
        if j - i >= 4:
            try:
                t = s[i:j].decode("ascii")
            except UnicodeDecodeError:
                t = None
            if t and re.fullmatch(r"[A-Za-z0-9_$.]+", t):
                yield va + i, t
        i = j + 1


def parse_passes() -> list[tuple[str, str, int]]:
    """返回 [(命名空间, pass 名, RTTI 名 vaddr)]，去重、按名字排序"""
    seen: dict[tuple[str, str], int] = {}
    for va, s in strings():
        m = RE_CUSTOM.match(s)
        if not m:
            continue
        inner = RE_INNER.match(m.group(1))
        if not inner:
            continue
        ns = "rknn" if inner.group(1) else "onnc"
        n = int(inner.group(2))
        name = m.group(1)[inner.end():inner.end() + n]
        if len(name) != n:
            continue
        seen.setdefault((ns, name), va)
    return sorted(((ns, nm, va) for (ns, nm), va in seen.items()), key=lambda x: (x[0], x[1]))


def vtables(names: list[str]) -> dict[str, list[int]]:
    """用 rtti_find 的静态解析拿每个 pass 的虚函数地址。

    ⚠️ 传**裸类名**即可：`rtti_find.find` 按子串找，第一次命中的就是
       `N4onnc10CustomPassIN4rknn14<名>ENS_10ModulePassEvEE` 里的那段 ——
       **正是我们要的 CustomPass 特化**（不是裸类的 RTTI）。
    """
    rl = R.relocs()
    seg = R.segments()
    text_va, text_sz = seg[".text"][1], seg[".text"][2]
    text_end = text_va + text_sz
    out: dict[str, list[int]] = {}
    for nm in names:
        va = None
        for v, s in strings():
            if nm in s and "CustomPass" in s:
                va = v
                break
        if va is None:
            out[nm] = []
            continue
        # 回退到 mangled 名起点
        d = open(R.SO, "rb").read()
        ro_off, ro_va, _ = seg[".rodata"]
        idx = ro_off + (va - ro_va)
        ok = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
        while idx > ro_off and d[idx - 1] in ok:
            idx -= 1
        name_va = ro_va + (idx - ro_off)
        codes: list[int] = []
        for ti in [k for k, v in rl.items() if v == name_va]:
            for vt2 in [k for k, v in rl.items() if v == ti - 8]:
                for k in range(16):
                    v = rl.get(vt2 + 8 * k)
                    if v is not None and text_va <= v < text_end:
                        codes.append(v)
                break
        out[nm] = codes
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default=None, help="只显示名字含该子串的 pass")
    ap.add_argument("--vtable", action="store_true", help="顺带解析虚函数地址（慢）")
    ap.add_argument("--write", action="store_true", help="写入 data/analysis/rknn_passes_full.txt")
    args = ap.parse_args()

    passes = parse_passes()
    if args.filter:
        passes = [p for p in passes if args.filter.lower() in p[1].lower()]

    vts = vtables([p[1] for p in passes]) if args.vtable else {}

    lines = [f"# librknnc.so 编译器 pass 全表（从 RTTI 名静态枚举）  共 {len(passes)} 个",
             "# 全部是 onnc::CustomPass<NS::Name, onnc::ModulePass, void> 的实例",
             ""]
    rk = [p for p in passes if p[0] == "rknn"]
    on = [p for p in passes if p[0] == "onnc"]
    lines.append(f"## rknn 自己的 pass（{len(rk)}）")
    for ns, nm, va in rk:
        extra = ""
        if vts.get(nm):
            extra = "  vfn=" + ",".join(hex(x) for x in vts[nm][:6])
        lines.append(f"  {nm:34s} RTTI@{va:#x}{extra}")
    lines.append("")
    lines.append(f"## ONNC 框架的 pass（{len(on)}）")
    for ns, nm, va in on:
        lines.append(f"  {nm:34s} RTTI@{va:#x}")
    txt = "\n".join(lines) + "\n"
    print(txt)
    if args.write:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        open(OUT, "w").write(txt)
        print(f"[pass_list] 已写 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
