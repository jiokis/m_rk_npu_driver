#!/usr/bin/env python3
"""
reg_coverage.py — 寄存器覆盖率：我们是「发现了上游不知道的东西」，还是「上游都知道」

## 为什么需要这个脚本

docs/36 曾声称「我们观测到 39 个上游未定义的寄存器」。**那是错的** —— 它的
「来源 A」用的是 Mesa 的 `registers.xml`（154 条，是内核头文件的子集），
而不是内核头文件 `driver/accel/rocket/rocket_registers.h`（199 条）。

这个脚本把覆盖率算成**可复现**的一件事：解析内核头文件里全部 `#define REG_*`，
与语料里实际出现过的偏移取交集/差集。同时把「上游有我们没见过」的那批也列出来
—— 那批同样有信息量（它们标记了我们**从未探索过的硬件路径**）。

## 用法
  python3 src/analysis/reg_coverage.py                       # 默认 align1 语料
  python3 src/analysis/reg_coverage.py --dumps DIR --out F   # 写文件
"""
import argparse
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from tiling import load_segments                                    # noqa: E402

DEF_HDR = os.path.expanduser("~/rocket-upstream/kernel/rocket_registers.h")
DEF_DUMPS = os.path.expanduser("~/align1/align1")

# Mesa 以「裸偏移」写入、因此头文件没给名字的两个
MESA_RAW = {
    0x3030: "mesa/rkt_regcmd.c: emit_raw(regs, CORE | 0x1, 0x3030, 0)",
    0x40c4: "mesa/rkt_regcmd.c: emit_raw(regs, DPU | 0x1, 0x40c4, 0)"
            " —— DPU_SURFACE_ADD(0x40c0) 的高 32 位（低字 mask 0xfffffff0）",
}


def parse_header(path):
    up = {}
    rx = re.compile(r"#define\s+REG_([A-Z0-9_]+?)_([A-Z0-9_]+)\s+(0x[0-9a-fA-F]+)\s*$")
    for line in open(path, encoding="utf-8", errors="replace"):
        m = rx.match(line)
        if m and not m.group(2).startswith("RESERVED"):
            up.setdefault(int(m.group(3), 16), (m.group(1), m.group(2)))
    return up


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdr", default=DEF_HDR)
    ap.add_argument("--dumps", default=DEF_DUMPS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    up = parse_header(a.hdr)
    segs = load_segments(a.dumps)
    seen = collections.Counter()
    n_seg = 0
    for s, d in segs:
        n_seg += 1
        for i in s.insns:
            seen[i.reg] += 1
        # 段尾的 PC 链 / 开火位图不在 insns 里，单独挂在 Segment 上；
        # 漏掉它们会把 0x0010/0x0014/0x0008 误报成「从未观测」。
        for extra in (s.chain, s.chain_len, s.enable):
            if extra is not None:
                seen[extra.reg] += 1

    both = sorted(set(up) & set(seen))
    only_up = sorted(set(up) - set(seen))
    only_us = sorted(set(seen) - set(up))

    L = []
    L.append("RK3588 NPU 寄存器覆盖率（可复现，由 src/analysis/reg_coverage.py 生成）")
    L.append("=" * 96)
    L.append(f"来源 A（上游权威命名）：{a.hdr}")
    L.append(f"来源 B（我们的观测）：  {a.dumps}  （{n_seg} 个 segment/tile）")
    L.append("")
    L.append(f"  上游头文件 #define REG_* 条数 : {len(up)}")
    L.append(f"  我们观测到的不同寄存器偏移数 : {len(seen)}")
    L.append(f"  ├─ 交集（上游有名 + 我们观测到）: {len(both)}")
    L.append(f"  ├─ 我们观测到、上游头文件没名字 : {len(only_us)}")
    L.append(f"  └─ 上游有、我们从未观测到       : {len(only_up)}")
    L.append("")
    L.append(f"  ⇒ 我们观测到的寄存器里，{100.0*len(both)/max(len(seen),1):.1f}% 上游有名字；"
             f"剩下 {len(only_us)} 个上游也**在写**（见下）。")
    L.append(f"  ⇒ 上游覆盖率 = {100.0*(len(both)+len(only_us))/max(len(seen),1):.1f}%"
             f"（按「上游是否知道这个偏移」计）")
    L.append("")

    L.append("── 我们观测到、但上游头文件未命名的 ──")
    for r in only_us:
        note = MESA_RAW.get(r, "（未在上游 Mesa/内核里找到写入点 —— 真正的空白）")
        L.append(f"  0x{r:06x}  观测 {seen[r]:>5} 次   {note}")
    if not only_us:
        L.append("  （无）")
    L.append("")

    L.append("── 上游有、我们从未观测到的（= 未探索的硬件路径）──")
    groups = collections.defaultdict(list)
    for r in only_up:
        groups[up[r][0]].append((r, up[r][1]))
    for blk in sorted(groups, key=lambda b: min(r for r, _ in groups[b])):
        rs = groups[blk]
        L.append(f"  [{blk}]  {len(rs)} 个")
        L.append("      " + ", ".join(f"{n}@0x{r:06x}" for r, n in sorted(rs)))
    L.append("")
    L.append("── 交集明细（偏移 | 上游名 | 观测次数）──")
    for r in both:
        blk, name = up[r]
        L.append(f"  0x{r:06x}  {blk:<9} {name:<34} {seen[r]:>6}")

    txt = "\n".join(L) + "\n"
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)
        print(f"已写入 {a.out}（{len(L)} 行）")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
