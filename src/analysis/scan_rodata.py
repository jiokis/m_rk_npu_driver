#!/usr/bin/env python3
"""在 .rodata 里找 block_id 的成簇出现 = 发射器的字段描述符表"""
import struct, sys
from elftools.elf.elffile import ELFFile

SO = "/home/dunaandone/rkllm-convert/libs/rkllm_runtime-1.3.0.so"
BLK = {0x0000:'NONE',0x0041:'OP40',0x0081:'EN',0x0101:'PC',0x0201:'CNA',
       0x0801:'CORE',0x1001:'DPU',0x2001:'DPU_RD',0x4001:'PPU',0x8001:'PPU_RD'}

def scan(secname, width):
    """width=2 按 u16 扫，width=4 按 u32 扫"""
    with open(SO,'rb') as f:
        elf = ELFFile(f)
        s = elf.get_section_by_name(secname)
        if not s: print(f"{secname} 不存在"); return
        d = s.data(); base = s['sh_addr']
    fmt = '<H' if width==2 else '<I'
    vals = [struct.unpack_from(fmt, d, i)[0] for i in range(0, len(d)-width+1, width)]
    # 找连续 window 里 block 值密集的位置
    n = len(vals)
    W = 24
    best = []
    for i in range(n - W):
        cnt = sum(1 for v in vals[i:i+W] if v in BLK and v != 0)
        if cnt >= 8:
            best.append((cnt, i, vals[i:i+W]))
    best.sort(reverse=True)
    print(f"══ {secname} 按 {width} 字节扫，窗口 {W} 个值 ══")
    print(f"   命中窗口 {len(best)} 处，取前 5：")
    seen=set()
    for cnt, i, win in best[:5]:
        va = base + i*width
        if any(abs(va-x)<64 for x in seen): continue
        seen.add(va)
        tr = []
        for v in win[:20]:
            if v in BLK: tr.append(f"{BLK[v]}@0x{v:04x}")
            else: tr.append(f"{v:#06x}")
        print(f"\n   @ {va:#08x}  密度 {cnt}/{W}")
        print(f"     {' '.join(tr)}")

if __name__ == '__main__':
    for sec in ('.rodata','.data.rel.ro'):
        for w in (2,4):
            scan(sec, w)
            print()
