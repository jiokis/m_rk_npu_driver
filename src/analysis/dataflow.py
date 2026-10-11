#!/usr/bin/env python3
"""
dataflow.py — 从张量地址还原 NPU 的数据流拓扑与缓冲复用

## 为什么这是"最大程度利用 NPU"的第一步

算力利用率的瓶颈几乎从不是"算得慢"，而是：
  · 数据在片上/片外之间来回搬（带宽墙）
  · 流水级之间没能重叠（串行等待）
  · 多核调度不当（有的核空转）

要优化这三件事，必须先知道**硬件的数据通路长什么样**：
哪个块读哪块内存、写到哪里、谁消费谁。

我们有一个别处没有的抓手：**每个流水级都有自己的张量地址寄存器**
（已确证 10 个，见 docs/15）。把它们指向的缓冲区连起来，
就能把真实的数据流图和缓冲复用方案还原出来。

## 判据

- 地址落在哪个 MEM_CREATE 缓冲区（含偏移）
- 同一缓冲区的不同偏移 → 可能是同一大张量的切片，也可能复用
- 某块写的地址 = 另一块读的地址 → **生产者→消费者**关系

## 用法
  python3 dataflow.py buffers         # 列出所有设备缓冲区
  python3 dataflow.py op <起始task>    # 单个算子的数据流
  python3 dataflow.py topology        # 全量统计：块之间的数据依赖
  python3 dataflow.py reuse           # 缓冲区复用情况
"""
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = os.environ.get("RKNPU_DUMPS") or os.path.join(HERE, "..", "..", "data", "dumps2")
sys.path.insert(0, os.path.join(HERE, "..", "sim"))

from rknpu import regcmd as rc                      # noqa: E402
from rknpu import regmap as rm                      # noqa: E402
from rknpu import loader                            # noqa: E402
from rknpu.synth import operator_segments           # noqa: E402


def load_buffers():
    """从 capture.log 解析设备缓冲区：dma_addr → 大小"""
    out = []
    log = os.path.join(DUMPS, "capture.log")
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.search(r"\[MEM_CREATE\] handle=(\d+) flags=(0x[0-9a-f]+) size=(\d+) "
                      r"obj_addr=(0x[0-9a-f]+) dma_addr=(0x[0-9a-f]+)", line)
        if m:
            out.append({"handle": int(m.group(1)), "flags": int(m.group(2), 16),
                        "size": int(m.group(3)), "dma": int(m.group(5), 16)})
    out.sort(key=lambda b: b["dma"])
    return out


def which_buffer(bufs, addr):
    for b in bufs:
        if b["dma"] <= addr < b["dma"] + b["size"]:
            return b, addr - b["dma"]
    return None, None


def op_addrs(mem, tasks, by_addr, start):
    """某算子里所有「已确证地址寄存器」的写入：(段序, block名, reg, 地址, 缓冲区, 偏移)"""
    rows = []
    for k, (i, t) in enumerate(operator_segments(tasks, by_addr, start)):
        ins = rc.decode(mem.read(t.addr, t.amount * 8), limit=t.amount)
        for j in ins:
            if rm.is_tensor_addr(j.block, j.reg):
                rows.append((k, i, j.block_name, j.reg, j.value))
    return rows


def cmd_buffers():
    bufs = load_buffers()
    print(f"═══ 设备缓冲区（{len(bufs)} 个）═══\n")
    print(f"  {'dma_addr':>12} {'大小':>10} {'flags':>8}  用途推测")
    for b in bufs:
        use = ""
        if b["size"] >= 4 << 20:
            use = "大块：特征图/权重"
        elif b["size"] >= 1 << 20:
            use = "中等：特征图"
        elif b["size"] >= 100 << 10:
            use = "小块"
        else:
            use = "很小：控制/参数"
        print(f"  {b['dma']:>#12x} {b['size']:>10} {b['flags']:>#8x}  {use}")
    tot = sum(b["size"] for b in bufs)
    print(f"\n  合计 {tot/1048576:.1f} MB")
    return 0


def cmd_op(start):
    mem, tasks, by_addr = loader.load_corpus()
    bufs = load_buffers()
    segs = operator_segments(tasks, by_addr, start)
    if not segs:
        print("找不到算子"); return 1
    print(f"═══ 算子（起始 task{start}，op{segs[0][1].op_idx}，{len(segs)} 段）的数据流 ═══\n")
    print(f"  {'段':>3} {'task':>5} {'block':<9} {'reg':>7} {'地址':>12} "
          f"{'缓冲区':>12} {'偏移':>10}")
    rows = op_addrs(mem, tasks, by_addr, start)
    for k, i, bn, reg, v in rows:
        b, off = which_buffer(bufs, v)
        bs = f"{b['dma']:#x}" if b else "—"
        os_ = f"{off:#x}" if off is not None else "—"
        print(f"  {k:>3} {i:>5} {bn:<9} 0x{reg:04x} {v:>#12x} {bs:>12} {os_:>10}")
    print(f"\n  涉及 {len(set((bn, reg) for _, _, bn, reg, _ in rows))} 个地址寄存器，"
          f"{len(set(v for *_, v in rows))} 个不同地址")
    return 0


def cmd_topology():
    mem, tasks, by_addr = loader.load_corpus()
    bufs = load_buffers()
    starts, prev = [], None
    for seq, idx, t in tasks:
        if seq != 0: continue
        if prev is None or t.op_idx != prev: starts.append(idx)
        prev = t.op_idx

    print(f"═══ 全量：每个地址寄存器指向哪个缓冲区（{len(starts)} 个算子）═══\n")
    # (block,reg) → 缓冲区分布
    dist = defaultdict(Counter)
    for st in starts:
        for k, i, bn, reg, v in op_addrs(mem, tasks, by_addr, st):
            b, _ = which_buffer(bufs, v)
            dist[(bn, reg)][f"{b['dma']:#x}" if b else "未落在缓冲区"] += 1

    print(f"  {'block':<10} {'reg':>7}  {'总次数':>6}  主要缓冲区分布")
    for (bn, reg), c in sorted(dist.items()):
        tot = sum(c.values())
        top = " ".join(f"{k}×{v}" for k, v in c.most_common(3))
        print(f"  {bn:<10} 0x{reg:04x}  {tot:>6}  {top}")

    # 数据流边：同一算子内，若某块的地址 == 另一块读的地址，则存在依赖
    print(f"\n═══ 块间地址重合（潜在的生产者→消费者边）═══\n")
    edges = Counter()
    for st in starts:
        rows = op_addrs(mem, tasks, by_addr, st)
        seen = defaultdict(set)
        for k, i, bn, reg, v in rows:
            seen[v].add((bn, reg))
        for v, who in seen.items():
            if len(who) > 1:
                names = sorted(f"{b}@0x{r:04x}" for b, r in who)
                edges[tuple(names)] += 1
    if not edges:
        print("  没有发现同一算子内不同块共用同一地址的情况")
        print("  （说明各流水级之间通过**不同偏移**交接，需按偏移邻近性进一步分析）")
    for combo, n in edges.most_common(10):
        print(f"  ×{n:<4} {'  ↔  '.join(combo)}")
    return 0


def cmd_reuse():
    mem, tasks, by_addr = loader.load_corpus()
    bufs = load_buffers()
    starts, prev = [], None
    for seq, idx, t in tasks:
        if seq != 0: continue
        if prev is None or t.op_idx != prev: starts.append(idx)
        prev = t.op_idx

    print("═══ 缓冲区复用：哪些块共用同一块内存 ═══\n")
    use = defaultdict(lambda: defaultdict(int))
    for st in starts:
        for k, i, bn, reg, v in op_addrs(mem, tasks, by_addr, st):
            b, off = which_buffer(bufs, v)
            if b:
                use[f"{b['dma']:#x} ({b['size']//1024}K)"][bn] += 1
    for buf, blocks in sorted(use.items(), key=lambda x: -sum(x[1].values())):
        tot = sum(blocks.values())
        dist = " ".join(f"{k}×{v}" for k, v in sorted(blocks.items(), key=lambda x: -x[1]))
        print(f"  {buf:<22} 被引用 {tot:>4} 次：{dist}")
    return 0


def main():
    c = sys.argv[1] if len(sys.argv) > 1 else "buffers"
    if c == "buffers":  return cmd_buffers()
    if c == "op":       return cmd_op(int(sys.argv[2]))
    if c == "topology": return cmd_topology()
    if c == "reuse":    return cmd_reuse()
    print(__doc__); return 1


if __name__ == "__main__":
    sys.exit(main())
