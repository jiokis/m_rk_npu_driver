#!/usr/bin/env python3
"""
rknpu.cli — RKNPU 模拟开发环境命令行

  python3 -m rknpu.cli verify      # 把全部真实样本跑一遍，检查是否自洽
  python3 -m rknpu.cli chains      # 列出重建出的算子（task 链）
  python3 -m rknpu.cli trace <task># 跟踪一个算子的完整执行
  python3 -m rknpu.cli state       # 跑完后转储寄存器状态
  python3 -m rknpu.cli regs        # 打印已知寄存器表
  python3 -m rknpu.cli rebuild     # 重新生成全部算子并逐字节比对（最强验证）
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

from . import loader, regcmd as rc, regmap as rm
from .machine import Machine


def _num(s):
    return int(s, 0) if s.startswith(("0x", "0X")) else int(s)


def cmd_verify(dumps=None):
    mem, tasks, by_addr = loader.load_corpus(dumps) if dumps else loader.load_corpus()
    print("═══ 用模拟器跑全部真实样本 ═══\n")
    print(f"  内存块 {len(mem.spans)} · 覆盖 {mem.total/1024:.0f} KB")
    print(f"  task {len(tasks)} 个\n")

    m = Machine(mem)
    segs = m.run_all(tasks)

    print(m.state_dump())
    print()
    ends = Counter(s.end for s in segs)
    print(f"  ── 段结束原因 ──")
    for k, v in ends.most_common():
        print(f"    {k:<12} {v}")

    links = Machine.chain_links(segs, by_addr)
    print(f"\n  ── PC 链指向（预期：指向下一个 task）──")
    for k, v in links.items():
        print(f"    {k:<14} {v}")

    # 算子分组：连续相同 op_idx
    runs, cur = [], []
    for seq, idx, t in tasks:
        if cur and cur[-1][2].op_idx != t.op_idx:
            runs.append(cur); cur = []
        cur.append((seq, idx, t))
    if cur: runs.append(cur)

    print(f"\n  ── 算子划分（连续相同 op_idx）──")
    print(f"    task {len(tasks)} → 算子 {len(runs)} 个")
    sizes = Counter(len(r) for r in runs)
    print(f"    每个算子的 task 数分布: {dict(sorted(sizes.items()))}")
    tot = [sum(t.amount + 1 for _, _, t in r) for r in runs]
    print(f"    指令数: 最少 {min(tot)} · 平均 {sum(tot)//len(tot)} · 最多 {max(tot)}")

    print(f"\n  执行指令总计 {m.stats['insns']} 条（每个段一次，O(n)）")
    return 0


def _build(dumps=None):
    mem, tasks, by_addr = loader.load_corpus(dumps) if dumps else loader.load_corpus()
    m = Machine(mem)
    return m, tasks, by_addr


def cmd_chains(dumps=None):
    m, tasks, by_addr = _build(dumps)
    seen, chains = set(), []
    for seq, idx, t in tasks:
        if t.key in seen:
            continue
        # 找到链首：不在任何链的中途
        ch = m.run_chain(t, by_addr)
        for s in ch.segments:
            seen.add(s.task.key)
        chains.append(ch)
    chains.sort(key=lambda c: -c.total_insns)
    print(f"═══ 重建出的算子（{len(chains)} 个）═══\n")
    print(f"  {'指令数':>7} {'task':>5} {'op':>4} {'完整':>5}  主要 block")
    for ch in chains[:25]:
        print(f"  {ch.total_insns:>7} {len(ch.segments):>5} {ch.op_idx:>4} "
              f"{'✅' if ch.complete else '—':>5}  "
              f"{' '.join(f'{k}×{v}' for k,v in list(ch.block_histogram().items())[:4])}")
    if len(chains) > 25:
        print(f"  … 其余 {len(chains)-25} 个省略")
    return 0


def cmd_trace(task_idx, seq=0, dumps=None):
    m, tasks, by_addr = _build(dumps)
    hit = next((t for s, i, t in tasks if i == task_idx and s == seq), None)
    if not hit:
        print(f"没有 seq{seq} task{task_idx}"); return 1
    ch = m.run_chain(hit, by_addr)
    print(f"═══ {ch.summary()} ═══\n")
    for si, seg in enumerate(ch.segments):
        print(f"  ── 段 {si}: task{seg.task.index} @ {seg.task.addr:#x}"
              f"（{len(seg.insns)} 条，结束={seg.end}）──")
        for i in seg.insns[:10]:
            note = ""
            role = rm.describe(i.block, i.reg)
            if role != "未观测":
                note = f"   {role}"
            print(f"    {i}{note}")
        if len(seg.insns) > 10:
            print(f"    … 省略 {len(seg.insns)-10} 条")
    print(f"\n  ── 张量地址（语义已确证）──")
    for bn, r, v in ch.tensor_addrs():
        print(f"    {bn:<8} 0x{r:04x} = {v:#x}")
    return 0


def cmd_state(dumps=None):
    m, tasks, by_addr = _build(dumps)
    for _, _, t in tasks:
        m.run_chain(t, by_addr)
    print("═══ 寄存器状态（跑完全部样本后）═══\n")
    print(m.state_dump())
    print(f"\n  ── 按 block 统计写入的寄存器数 ──")
    cnt = Counter()
    for (b, r) in m.regs.items():
        cnt[rc.block_name(b)] += 1
    for k, v in cnt.most_common():
        print(f"    {k:<8} {v:>3} 个寄存器")
    print(f"\n  ── 状态快照（只列语义已知的）──")
    for line in m.regs.dump()[:20]:
        print(line)
    return 0


def cmd_rebuild(dumps=None):
    """对**全部**算子做「重新生成并逐字节比对」"""
    from . import synth
    mem, tasks, by_addr = (loader.load_corpus(dumps) if dumps
                           else loader.load_corpus())
    # 算子起点 = 每个连续同 op_idx 段的首个 task
    starts, prev = [], None
    for seq, idx, t in tasks:
        if seq != 0:
            continue
        if prev is None or t.op_idx != prev:
            starts.append(idx)
        prev = t.op_idx

    print(f"═══ 全部 {len(starts)} 个算子的重新生成验证 ═══\n")
    print("  链部分**纯用内核公式算**，不看原字节；体部分逐字节比对\n")

    good = 0; body_ok = chain_ok = body_all = chain_all = 0
    bad_ops = []
    for st in starts:
        ok, info = synth.reconstruct_operator(mem, tasks, by_addr, st, verbose=False)
        if not isinstance(info, tuple):
            continue
        b, c, n = info
        body_ok += b; chain_ok += c; body_all += n; chain_all += n
        if ok:
            good += 1
        else:
            bad_ops.append((st, b, c, n))
    print(f"  ✅ 完全重建成功 : {good} / {len(starts)} 个算子")
    print(f"     体  逐字节一致: {body_ok} / {body_all} 段")
    print(f"     链  纯公式重建: {chain_ok} / {chain_all} 段")
    if bad_ops:
        print(f"\n  ── 未完全成功的算子（前 8）──")
        for st, b, c, n in bad_ops[:8]:
            print(f"    task{st}: 体 {b}/{n}  链 {c}/{n}")
    return 0


def cmd_regs():
    print(f"═══ 已知寄存器表（{len(rm.REGISTERS)} 个）═══\n")
    by = defaultdict(list)
    for (b, r), v in rm.REGISTERS.items():
        by[b].append((r, v))
    for b in sorted(by):
        rs = sorted(by[b])
        kinds = Counter(v.kind for _, v in rs)
        print(f"  ── {rc.block_name(b):<8} (block 0x{b:04x})  {len(rs)} 个"
              f"  {dict(kinds)} ──")
        for r, v in rs[:8]:
            mark = " ★地址" if rm.is_tensor_addr(b, r) else ""
            print(f"     0x{r:04x}  {v.kind:<11} 取值 {v.distinct_values} 种{mark}")
        if len(rs) > 8:
            print(f"     … 其余 {len(rs)-8} 个")
    return 0


def main(argv):
    c = argv[1] if len(argv) > 1 else "verify"
    if c == "verify": return cmd_verify()
    if c == "chains": return cmd_chains()
    if c == "trace":  return cmd_trace(_num(argv[2]), _num(argv[3]) if len(argv) > 3 else 0)
    if c == "state":  return cmd_state()
    if c == "regs":   return cmd_regs()
    if c == "rebuild":return cmd_rebuild()
    print(__doc__); return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
