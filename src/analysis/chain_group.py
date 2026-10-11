#!/usr/bin/env python3
"""
chain_group.py — 把 task 按 PC 链分组，重建**完整算子配置**

## 已确证的结构（见 docs/13）

- 每个 task 占一个槽位，槽内有 regcfg_amount 条指令（每条 8 字节）
- 槽内**最后**是 PC 链指令（block 0x0101 / reg 0x0010）
  · 值为 0        → 链结束
  · 值为下一地址  → 该 task 与下一个 task 属于**同一个算子**
- 所以「一个算子配置」= 一条 task 链，而不是单个 task

本工具按链分组并拼接，得到真正完整的算子命令流。
"""
import os, re, struct, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = os.environ.get("RKNPU_DUMPS") or os.path.join(HERE, "..", "..", "data", "dumps2")
BLOCK = {0x0000:"NONE",0x0041:"OP40",0x0081:"EN",0x0101:"PC",0x0201:"CNA",
         0x0801:"CORE",0x1001:"DPU",0x2001:"DPU_RD",0x4001:"PPU",0x8001:"PPU_RD"}
PC_ADDR = 0x0010


def decode(buf):
    return [(struct.unpack_from('<I',buf,i+4)[0]>>16,
             struct.unpack_from('<I',buf,i)[0]&0xFFFF,
             ((struct.unpack_from('<I',buf,i+4)[0]&0xFFFF)<<16)|(struct.unpack_from('<I',buf,i)[0]>>16))
            for i in range(0, len(buf)//8*8, 8)]


def load_meta():
    out, seq = {}, -1
    for line in open(os.path.join(DUMPS,'capture.log'), encoding='utf-8', errors='replace'):
        if line.startswith('[SUBMIT]'): seq += 1; continue
        m = re.search(r'task\[(\d+)\] flags=.*?op_idx=(\d+).*?regcfg_amount=(\d+).*?regcmd_addr=(0x[0-9a-f]+)', line)
        if m and seq >= 0:
            out[(seq,int(m.group(1)))] = {"op_idx":int(m.group(2)),"amount":int(m.group(3)),"addr":int(m.group(4),16)}
    return out


def load_segs():
    meta = load_meta(); segs = []
    for f in sorted(os.listdir(DUMPS)):
        m = re.match(r'regcmd_(\d+)_task(\d+)\.bin$', f)
        if not m: continue
        k = (int(m.group(1)), int(m.group(2))); mm = meta.get(k)
        if not mm: continue
        segs.append((mm["addr"], mm["addr"]+os.path.getsize(os.path.join(DUMPS,f)),
                     open(os.path.join(DUMPS,f),'rb').read(), k, mm["amount"]))
    return segs, meta


def addr2seg(segs, addr):
    for s,e,b,k,amt in segs:
        if s <= addr < e: return b[addr-s:], k
    return None, None


def groups(segs, meta):
    """按 addr 连续性把 task 分组成链；返回 {首task: [(task, 指令表)]}"""
    by_addr = {mm["addr"]: k for k,mm in meta.items()}
    used = set(); out = []
    for seq in range(4):
        keys = sorted(k for k in meta if k[0]==seq)
        for k in keys:
            if k in used: continue
            chain = []
            cur_k = k
            for _ in range(64):
                if cur_k is None or cur_k in used: break
                mm = meta.get(cur_k)
                if not mm: break
                buf, owner = addr2seg(segs, mm["addr"])
                if buf is None: break
                ins = decode(buf)
                nxt = 0
                for b_,r_,v_ in ins:
                    if b_==0x0101 and r_==PC_ADDR: nxt = v_
                # 只保留本段声明的条数（槽位可能有富余）
                ins = ins[:mm["amount"]]
                chain.append((cur_k, mm, ins))
                used.add(cur_k)
                cur_k = by_addr.get(nxt) if nxt else None
            if chain: out.append(chain)
    return out


def cmd_stats():
    segs, meta = load_segs()
    gs = groups(segs, meta)
    print(f"═══ 按 PC 链分组后的完整算子配置 ═══\n")
    print(f"  task 总数 {len(meta)} → 分成 {len(gs)} 组（= 算子的真实数量）\n")
    sz = defaultdict(int)
    for ch in gs: sz[len(ch)] += 1
    print("  ── 每组含几个 task ──")
    for n in sorted(sz): print(f"    {n} 个 task: {sz[n]} 组")
    tot = [sum(len(i) for _,_,i in ch) for ch in gs]
    print(f"\n  ── 完整算子的指令数分布 ──")
    print(f"    最少 {min(tot)} 条 · 平均 {sum(tot)//len(tot)} 条 · 最多 {max(tot)} 条")
    big = sorted(zip(tot,gs), key=lambda x:-x[0])[:5]
    print(f"\n  ── 最大的 5 个算子 ──")
    for n, ch in big:
        tasks = [str(k[1]) for k,_,_ in ch]
        print(f"    {n:>5} 条指令 · {len(ch)} 个 task · op{ch[0][1]['op_idx']} · task {','.join(tasks[:8])}")
    return 0


def cmd_show(op=None):
    segs, meta = load_segs()
    gs = groups(segs, meta)
    if op is None:
        gs.sort(key=lambda ch: -sum(len(i) for _,_,i in ch))
        ch = gs[0]
    else:
        ch = next((c for c in gs if c[0][1]['op_idx']==op), None)
        if not ch: print(f"没有 op {op}"); return 1
    n = sum(len(i) for _,_,i in ch)
    print(f"═══ 完整算子配置：op{ch[0][1]['op_idx']} · {len(ch)} 个 task · {n} 条指令 ═══\n")
    idx = 0
    for k, mm, ins in ch:
        print(f"  ── task{k[1]} @ {mm['addr']:#x}（{len(ins)} 条）──")
        for b_,r_,v_ in ins:
            note = ""
            if b_==0x0101 and r_==0x0010: note = "  ← PC 链"
            elif b_==0x0101 and r_==0x0014: note = "  ← 段长度"
            elif b_==0x0081: note = "  ← 总使能"
            if idx < 100:
                print(f"    [{idx:>3}] {BLOCK.get(b_,hex(b_)):<7} 0x{r_:04x} 0x{v_:08x}{note}")
            idx += 1
    if n > 100: print(f"    … 其余 {n-100} 条省略")
    print(f"\n  ── 寄存器写入次数 top 12 ──")
    cnt = defaultdict(int)
    for _,_,ins in ch:
        for b_,r_,v_ in ins: cnt[(b_,r_)] += 1
    for (b_,r_),c in sorted(cnt.items(), key=lambda x:-x[1])[:12]:
        print(f"    {BLOCK.get(b_,hex(b_)):<7} 0x{r_:04x}  写了 {c} 次")
    return 0


def main():
    c = sys.argv[1] if len(sys.argv)>1 else "stats"
    if c=="stats": return cmd_stats()
    if c=="show":  return cmd_show(int(sys.argv[2]) if len(sys.argv)>2 else None)
    print(__doc__); return 1


if __name__ == "__main__":
    sys.exit(main())
