#!/usr/bin/env python3
"""
xref_string.py — 找引用了指定字符串的代码

## 为什么用这个方法
小常量（0x41/0x201…）在二进制里假阳性极多（ASCII、枚举、尺寸都会撞）。
而**指向某个特定字符串的地址**唯一，假阳性极低。

## 实现要点（踩过的坑）
编译器会把 `add` 排在 `adrp` **之后好几条指令**（指令调度），
所以不能要求 adrp/add 相邻。必须**按寄存器跟踪** adrp 算出的页地址：
  · adrp Rd, #page      → pending[Rd] = page
  · add  Rd, Rn, #imm   → 若 Rn 在 pending 里，目标 = pending[Rn] + imm
  · 其它指令写 Rd       → 清除 pending[Rd]（避免误配）
  · blr/br              → 清空（跨函数不保留）
"""
import sys
from capstone import Cs, CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN
from elftools.elf.elffile import ELFFile

SO = "/home/dunaandone/rkllm-convert/libs/rkllm_runtime-1.3.0.so"


def str_vas(elf, blob, needles):
    out = {}
    for nd in needles:
        off = blob.find(nd.encode())
        if off < 0:
            print(f"  ⚠️ 未找到: {nd}"); continue
        for s in elf.iter_sections():
            if s['sh_offset'] <= off < s['sh_offset'] + s['sh_size']:
                out[s['sh_addr'] + (off - s['sh_offset'])] = nd
                break
    return out


def main(needles):
    with open(SO, 'rb') as f:
        elf = ELFFile(f)
        blob = f.read()
        want = str_vas(elf, blob, needles)
        if not want:
            return 1
        print("── 目标字符串 ──")
        for va, nd in want.items():
            print(f"  {va:#010x}  {nd[:60]!r}")

        text = elf.get_section_by_name('.text')
        code = text.data(); base = text['sh_addr']
        md = Cs(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN)
        md.detail = True
        print(f"\n  反汇编 {len(code)//4} 条指令，按寄存器跟踪 adrp…\n")

        hits = {v: [] for v in want}
        pending = {}
        n = 0
        for ins in md.disasm(code, base):
            n += 1
            try:
                if ins.mnemonic == 'adrp':
                    pending[ins.operands[0].reg] = ins.operands[1].imm
                    continue
                if ins.mnemonic == 'add' and len(ins.operands) == 3:
                    rd, rn, imm = ins.operands
                    if rn.type == 1 and rd.type == 1 and imm.type == 2 and rn.reg in pending:
                        tgt = pending[rn.reg] + imm.imm
                        if tgt in hits:
                            hits[tgt].append(ins.address)
                    if rd.type == 1:
                        pending.pop(rd.reg, None)
                    continue
                if ins.mnemonic in ('blr', 'br', 'ret'):
                    pending.clear(); continue
                # 其它指令若写了某个寄存器，清掉它的 pending
                if ins.operands and ins.operands[0].type == 1 and \
                   ins.mnemonic not in ('str', 'stp', 'cmp', 'tst', 'cbz', 'cbnz'):
                    pending.pop(ins.operands[0].reg, None)
            except Exception:
                pass

        for va, nd in want.items():
            lst = hits[va]
            print(f"── {nd[:56]!r}  → {len(lst)} 处引用 ──")
            for a in lst[:12]:
                print(f"    {a:#010x}")
            if not lst:
                print("    （无引用）")
            print()
    return 0


if __name__ == '__main__':
    main(sys.argv[1:] or ["failed to submit!, op id:",
                          "RKNPU F2", "RKNPU V2", "RKNPU W2",
                          "/dev/dri/%s", "Meet unknown rknpu target"])
