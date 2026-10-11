#!/usr/bin/env python3
"""
chain_walk.py — 顺 PC 链把多段命令流接起来

## 为什么需要

已确证 `regcfg_amount` 是**本段**的指令条数，段的最后一条是 PC 链指令
（block 0x0101 / reg 0x0010 = 下一段地址，reg 0x0014 = 段长度）。
所以一条完整算子配置是**多段链式拼接**的 —— 我们此前每条样本只看到第一段。

## 方法（纯离线，零硬件风险）

1. 用所有 dump 建「设备地址 → 字节」映射（每个 dump 的起始地址来自 capture.log）
2. 从某 task 的 regcmd_addr 出发，解出本段
3. 在段内找 PC 链指令 → 取下一段地址 → 回到第 2 步
4. 直到链的值为 0 / 找不到 / 形成环

## 用法
  python3 chain_walk.py stats      # 统计能接多长
  python3 chain_walk.py trace <task>
  python3 chain_walk.py coverage   # 链上地址有多少落在我们的 dump 覆盖范围内
"""
import os, re, struct, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = os.environ.get("RKNPU_DUMPS") or os.path.join(HERE, "..", "..", "data", "dumps2")

BLOCK = {0x0000:"NONE",0x0041:"OP40",0x0081:"EN",0x0101:"PC",0x0201:"CNA",
         0x0801:"CORE",0x1001:"DPU",0x2001:"DPU_RD",0x4001:"PPU",0x8001:"PPU_RD"}
PC_ADDR, PC_LEN = 0x0010, 0x0014


def decode(buf):
    out=[]
    for i in range(0, len(buf)//8*8, 8):
        w0,w1 = struct.unpack_from('<II', buf, i)
        out.append((w1>>16, w0 & 0xFFFF, ((w1 & 0xFFFF)<<16)|(w0>>16)))
    return out


def load_meta():
    """(seq,task) → {op_idx, amount, addr}"""
    log = os.path.join(DUMPS, "capture.log")
    out, seq = {}, -1
    for line in open(log, encoding='utf-8', errors='replace'):
        if line.startswith('[SUBMIT]'):
            seq += 1; continue
        m = re.search(r'task\[(\d+)\] flags=.*?op_idx=(\d+).*?regcfg_amount=(\d+)'
                      r'.*?regcmd_addr=(0x[0-9a-f]+)', line)
        if m and seq >= 0:
            out[(seq,int(m.group(1)))] = {"op_idx":int(m.group(2)),
                                          "amount":int(m.group(3)),
                                          "addr":int(m.group(4),16)}
    return out


def load_segments():
    """建「地址 → 字节」映射：每个 dump 从它 task 的 regcmd_addr 开始"""
    meta = load_meta()
    segs = []                       # (start, end, bytes, key)
    for f in sorted(os.listdir(DUMPS)):
        m = re.match(r'regcmd_(\d+)_task(\d+)\.bin$', f)
        if not m: continue
        k = (int(m.group(1)), int(m.group(2)))
        mm = meta.get(k)
        if not mm: continue
        b = open(os.path.join(DUMPS, f), 'rb').read()
        segs.append((mm["addr"], mm["addr"]+len(b), b, k, mm["amount"]))
    return segs, meta


def find_at(segs, addr):
    """返回 (字节, 该段内的偏移, 对应task)，要求 addr 落在某个 dump 覆盖范围内"""
    for s, e, b, k, amt in segs:
        if s <= addr < e:
            return b[addr-s:], addr-s, k
    return None, None, None


def walk(segs, meta, key, max_seg=16):
    """从某个 task 出发顺链走，返回 [(段起始地址, 条数, 段内指令)] """
    mm = meta.get(key)
    if not mm: return []
    cur, chain, seen = mm["addr"], [], set()
    for _ in range(max_seg):
        if cur in seen:            # 环
            chain.append(("CYCLE", 0, [])); break
        seen.add(cur)
        buf, off, owner = find_at(segs, cur)
        if buf is None:
            chain.append((cur, 0, None)); break      # 没抓到这段
        ins = decode(buf)
        chain.append((cur, len(ins), ins))
        # 找本段的 PC 链指令
        nxt = 0
        for b_, r_, v_ in ins:
            if b_ == 0x0101 and r_ == PC_ADDR:
                nxt = v_
        if nxt == 0:
            break
        cur = nxt
    return chain


def cmd_stats():
    segs, meta = load_segments()
    print(f"═══ PC 链跟踪统计 ═══\n")
    print(f"  dump 段 {len(segs)} 个 · 覆盖地址区间 "
          f"{min(s for s,_,_,_,_ in segs):#x} .. {max(e for _,e,_,_,_ in segs):#x}")
    covered = sum(e-s for s,e,_,_,_ in segs)
    print(f"  覆盖字节 {covered} ({covered/1048576:.1f} MB)\n")

    lens = defaultdict(int)
    reach = defaultdict(int)
    sample = []
    for k in sorted(meta):
        ch = walk(segs, meta, k)
        # 去掉最后那个"未捕获/环/结束"标记
        real = [c for c in ch if c[2]]
        total = sum(c[1] for c in real)
        lens[len(real)] += 1
        reach["总条数_" + str(total//50*50)] += 1
        if k[0]==0 and len(sample) < 8:
            sample.append((k, meta[k]["op_idx"], meta[k]["amount"], ch))

    print("  ── 每条链能接几段 ──")
    for n in sorted(lens):
        print(f"    {n} 段: {lens[n]} 条样本")
    print()
    print("  ── 示例（前 8 条）──")
    for k, op, amt, ch in sample:
        desc = []
        for addr, n, ins in ch:
            if ins is None: desc.append(f"{addr:#x}(未捕获)")
            elif addr == "CYCLE": desc.append("环")
            else: desc.append(f"{addr:#x}×{n}")
        total = sum(n for _,n,ins in ch if ins)
        print(f"    task{k[1]:>4}(op{op:>3}) 本段声明 {amt:>4} 条 → "
              f"链: {' → '.join(desc)}  共 {total} 条")
    return 0


def cmd_trace(task, seq=0):
    segs, meta = load_segments()
    ch = walk(segs, meta, (seq, task))
    mm = meta.get((seq,task), {})
    print(f"seq{seq} task{task} · op{mm.get('op_idx')} · 本段声明 {mm.get('amount')} 条\n")
    for si, (addr, n, ins) in enumerate(ch):
        if ins is None:
            print(f"  ── 段 {si}: {addr:#x} —— **未被 dump 覆盖**，链到此断掉 ──\n")
            continue
        if addr == "CYCLE":
            print(f"  ── 段 {si}: 检测到环 ──"); continue
        print(f"  ── 段 {si}: @ {addr:#x}，{n} 条指令 ──")
        for i,(b_,r_,v_) in enumerate(ins[:14]):
            note = ""
            if b_==0x0101 and r_==PC_ADDR: note = "  ← 下一段地址"
            elif b_==0x0101 and r_==PC_LEN: note = "  ← 段长度"
            elif b_==0x0081: note = "  ← 总使能"
            print(f"     [{i:>3}] {BLOCK.get(b_,hex(b_)):<7} 0x{r_:04x} 0x{v_:08x}{note}")
        if n > 14: print(f"     … 省略 {n-14} 条")
        print()
    return 0


def main():
    c = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if c == "stats":  return cmd_stats()
    if c == "trace":  return cmd_trace(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv)>3 else 0)
    print(__doc__); return 1


if __name__ == "__main__":
    sys.exit(main())
