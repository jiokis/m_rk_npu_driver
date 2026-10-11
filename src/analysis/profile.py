#!/usr/bin/env python3
"""
profile.py — 算子画像：把每个算子的「流水组成 + 工作量 + 数据通路」列出来

## 为什么这是"最大程度利用 NPU"的基础

利用率优化不能靠猜。要先能回答：
  · 这个算子激活了哪些硬件块？（流水组成 ⇒ 用到了哪几级）
  · 每级的数据量多大？（工作量 ⇒ 理论耗时）
  · 数据走哪块内存、是否需要回到 DRAM？（带宽墙在哪）
  · 多核/多任务有没有并行空间？

本工具把这些从命令流里**直接量出来**，产出可分析的画像表。

## 三个可量化的结构信息

1. **`OP_ENABLE` 位图**（block 0x0081 / reg 0x0008）—— 实测只有 3 种取值：
      0x1D = CNA|CORE|DPU|DPU_RDMA   （卷积类）
      0x18 = DPU|DPU_RDMA            （纯逐元素/激活）
      0x60 = PPU|PPU_RDMA            （池化类）
   ⇒ 这一位图直接给出「算子用了哪几级流水」

2. **各流水级的 (尺寸−1, 通道−1)**（见 docs/17）
   ⇒ 给出每级的数据量，进而估算搬运字节数与 MAC 数

3. **地址落在哪个缓冲区**（见 dataflow.py）
   ⇒ 给出数据通路：是否在片上复用、还是来回 DRAM

## 用法
  python3 profile.py list            # 全部算子的画像表
  python3 profile.py summary         # 按流水组成归类统计
  python3 profile.py op <起始task>    # 单个算子的详细画像
"""
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))

from rknpu import regcmd as rc                      # noqa: E402
from rknpu import regmap as rm                      # noqa: E402
from rknpu import loader                            # noqa: E402
from rknpu.synth import operator_segments           # noqa: E402

sys.path.insert(0, HERE)
from dataflow import load_buffers, which_buffer     # noqa: E402

# 各流水级的 (尺寸寄存器, 通道寄存器)，来自 docs/17 的自证结果
STAGE_SHAPE = {
    "CNA":      None,                      # CNA 的尺寸在 0x100c/0x1010 一族，暂不纳入
    "CORE":     (0x0801, 0x3014, 0x0801, 0x3018),
    "DPU":      (0x1001, 0x4030, 0x1001, 0x4058),
    "DPU_RDMA": (0x2001, 0x500c, 0x2001, 0x5014),
    "PPU":      (None, None, 0x4001, 0x6014),
    "PPU_RDMA": (None, None, 0x8001, 0x7014),
}

ENABLE_MEANING = {
    0x18: "DPU|DPU_RDMA",
    0x1D: "CNA|CORE|DPU|DPU_RDMA",
    0x60: "PPU|PPU_RDMA",
}


def op_profile(mem, tasks, by_addr, start):
    """算出一个算子的画像"""
    segs = operator_segments(tasks, by_addr, start)
    if not segs:
        return None
    regs = defaultdict(list)
    blocks = Counter()
    for i, t in segs:
        for j in rc.decode(mem.read(t.addr, t.amount * 8), limit=t.amount):
            regs[(j.block, j.reg)].append(j.value)
            blocks[j.block_name] += 1

    def first(b, r):
        v = regs.get((b, r))
        return v[0] if v else None

    en = first(0x0081, 0x0008)
    shape = {}
    for name, spec in STAGE_SHAPE.items():
        if not spec or spec[0] is None:
            continue
        sv = first(*spec[:2])
        cv = first(*spec[2:])
        if sv is not None:
            shape[name] = ((sv & 0xFFFF) + 1, (cv + 1) if cv is not None else None)
    # 张量地址
    addrs = {}
    for i, t in segs:
        for j in rc.decode(mem.read(t.addr, t.amount * 8), limit=t.amount):
            if rm.is_tensor_addr(j.block, j.reg) and j.value:
                addrs[(j.block_name, j.reg)] = j.value

    return {
        "start": start, "op": segs[0][1].op_idx, "segs": len(segs),
        "insns": sum(t.amount for _, t in segs),
        "enable": en, "pipeline": ENABLE_MEANING.get(en, f"?0x{en:x}" if en else "—"),
        "shape": shape, "blocks": blocks, "addrs": addrs,
    }


def all_starts(tasks):
    out, prev = [], None
    for seq, idx, t in tasks:
        if seq != 0:
            continue
        if prev is None or t.op_idx != prev:
            out.append(idx)
        prev = t.op_idx
    return out


def cmd_list():
    mem, tasks, by_addr = loader.load_corpus()
    starts = all_starts(tasks)
    print(f"═══ 算子画像（{len(starts)} 个）═══\n")
    print(f"  {'#':>3} {'task':>5} {'op':>4} {'段':>4} {'指令':>6} "
          f"{'使能':>5} {'流水组成':<22} {'CORE 输出(W×C)':<16}")
    for n, st in enumerate(starts):
        p = op_profile(mem, tasks, by_addr, st)
        if not p:
            continue
        core = p["shape"].get("CORE")
        cs = f"{core[0]}×{core[1]}" if core else "—"
        print(f"  {n:>3} {st:>5} {p['op']:>4} {p['segs']:>4} {p['insns']:>6} "
              f"{p['enable'] if p['enable'] is not None else '—':>5} "
              f"{p['pipeline']:<22} {cs:<16}")
    return 0


def cmd_summary():
    mem, tasks, by_addr = loader.load_corpus()
    starts = all_starts(tasks)
    profs = [p for p in (op_profile(mem, tasks, by_addr, s) for s in starts) if p]

    print(f"═══ 按流水组成归类（{len(profs)} 个算子）═══\n")
    by = defaultdict(list)
    for p in profs:
        by[p["pipeline"]].append(p)
    print(f"  {'流水组成':<24} {'算子数':>6} {'指令数':>8} {'平均段数':>8}  说明")
    desc = {"CNA|CORE|DPU|DPU_RDMA": "含卷积/矩阵乘，主力计算",
            "DPU|DPU_RDMA": "纯逐元素/激活，无卷积",
            "PPU|PPU_RDMA": "池化/后处理",
            "—": "未使能任何块（控制类）"}
    for k, ps in sorted(by.items(), key=lambda x: -len(x[1])):
        ins = sum(p["insns"] for p in ps)
        seg = sum(p["segs"] for p in ps) / len(ps)
        print(f"  {k:<24} {len(ps):>6} {ins:>8} {seg:>8.1f}  {desc.get(k, '')}")

    print(f"\n═══ 各流水级的输出规模分布（来自自证字段）═══\n")
    for stage in ("CORE", "DPU", "DPU_RDMA"):
        sizes = Counter()
        for p in profs:
            s = p["shape"].get(stage)
            if s:
                sizes[f"{s[0]}×{s[1]}"] += 1
        top = " ".join(f"{k}({v})" for k, v in sizes.most_common(8))
        print(f"  {stage:<10} {top}")

    print(f"\n═══ 数据通路：各算子用了几个张量地址 ═══\n")
    n_addr = Counter(len(p["addrs"]) for p in profs)
    for k in sorted(n_addr):
        print(f"    用 {k} 个地址: {n_addr[k]} 个算子")

    print(f"\n═══ 缓冲区使用（哪块内存被哪些级引用）═══\n")
    bufs = load_buffers()
    use = defaultdict(Counter)
    for p in profs:
        for (bn, reg), v in p["addrs"].items():
            b, _ = which_buffer(bufs, v)
            if b:
                use[f"{b['dma']:#x} ({b['size']//1024}K)"][bn] += 1
    for buf, c in sorted(use.items(), key=lambda x: -sum(x[1].values())):
        print(f"    {buf:<22} {dict(c)}")
    return 0


def cmd_op(start):
    mem, tasks, by_addr = loader.load_corpus()
    bufs = load_buffers()
    p = op_profile(mem, tasks, by_addr, start)
    if not p:
        print("找不到算子"); return 1
    print(f"═══ 算子画像：起始 task{start} ═══\n")
    print(f"  op{op['op']} · {p['segs']} 段 · {p['insns']} 条指令" if False else
          f"  op{p['op']} · {p['segs']} 段 · {p['insns']} 条指令")
    print(f"  使能位图 0x{p['enable']:x} → 流水组成: {p['pipeline']}")
    print(f"\n  ── block 指令分布 ──")
    for k, v in p["blocks"].most_common():
        print(f"     {k:<10} {v:>5}")
    print(f"\n  ── 各流水级输出规模（真值 = 字段值 + 1）──")
    for k, (w, c) in p["shape"].items():
        print(f"     {k:<10} {w}×{c}" if c else f"     {k:<10} {w}")
    print(f"\n  ── 张量地址 ──")
    for (bn, reg), v in sorted(p["addrs"].items()):
        b, off = which_buffer(bufs, v)
        bs = f"{b['dma']:#x}+{off:#x} ({b['size']//1024}K)" if b else "不在已知缓冲区"
        print(f"     {bn:<10} 0x{reg:04x} = {v:>#12x}  → {bs}")
    return 0


def main():
    c = sys.argv[1] if len(sys.argv) > 1 else "list"
    if c == "list":    return cmd_list()
    if c == "summary": return cmd_summary()
    if c == "op":      return cmd_op(int(sys.argv[2]))
    print(__doc__); return 1


if __name__ == "__main__":
    sys.exit(main())
