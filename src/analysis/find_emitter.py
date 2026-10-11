#!/usr/bin/env python3
"""
find_emitter.py — 静态反汇编：在闭源运行时里定位「构造 regcmd 的代码」

## 思路

regcmd 的每条指令是 8 字节 [reg u16][value u32][block_id u16]。
构造它的机器码里必然出现 **block_id 立即数**（0x0201=CNA、0x1001=DPU、0x0081=ENABLE…）。

所以：反汇编 .text，搜这些立即数 → 定位到候选函数 → 看上下文（寄存器/内存写模式）
→ 反推字段语义。这是**不依赖硬件的正规静态逆向**。

## 已知的 RTTI 线索（让搜索有的放矢）

    rknn::runtime::RKNPUEmitter      ← 基类
    rknn::runtime::RKNPUEmitterF2 / T4 / V2 / Lite
    rknn::runtime::RKNPUEmitterF2 等

有 RTTI 说明二进制保留了类型信息，vtable 可定位，进而能反汇编各 emitter 的方法。

## 用法
  python3 find_emitter.py blocks     # 搜 block_id 立即数
  python3 find_emitter.py rtti       # 找 emitter 类的 typeinfo / vtable
  python3 find_emitter.py dis <地址>  # 反汇编指定地址附近
"""
import struct
import sys

from capstone import Cs, CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN
from elftools.elf.elffile import ELFFile

SO = "/home/dunaandone/rkllm-convert/libs/rkllm_runtime-1.3.0.so"

BLOCKS = {
    0x0000: "OP_NONE", 0x0041: "OP_40", 0x0081: "OP_ENABLE",
    0x0101: "OP_REG_PC", 0x0201: "OP_REG_CNA", 0x0801: "OP_REG_CORE",
    0x1001: "OP_REG_DPU", 0x2001: "OP_REG_DPU_RDMA",
    0x4001: "OP_REG_PPU", 0x8001: "OP_REG_PPU_RDMA",
}


def load(path=SO):
    f = open(path, "rb")
    elf = ELFFile(f)
    text = elf.get_section_by_name(".text")
    return f, elf, text


def va2off(elf, va):
    for s in elf.iter_sections():
        a = s["sh_addr"]
        if a and a <= va < a + s["sh_size"]:
            return s["sh_offset"] + (va - a)
    return None


def cmd_blocks(path=SO):
    f, elf, text = load(path)
    code = text.data()
    base = text["sh_addr"]
    md = Cs(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN)
    md.detail = True

    print(f".text: {len(code)} 字节 @ {base:#x}")
    print("反汇编中…（几万条指令，稍等）\n")

    hits = {v: [] for v in BLOCKS}
    n = 0
    for ins in md.disasm(code, base):
        n += 1
        # 看立即数操作数
        for op in ins.operands:
            if op.type == 2:                       # ARM64_OP_IMM
                v = op.imm & 0xFFFFFFFF
                for bv in BLOCKS:
                    if bv and v == bv:
                        hits[bv].append((ins.address, ins.mnemonic, ins.op_str))
    print(f"共反汇编 {n} 条指令\n")

    for bv, name in BLOCKS.items():
        lst = hits[bv]
        if not lst:
            continue
        print(f"── {name} (0x{bv:04x}): {len(lst)} 处引用 ──")
        for a, m, o in lst[:8]:
            print(f"   {a:#010x}  {m:<10} {o}")
        print()
    f.close()
    return 0


def cmd_rtti(path=SO):
    """找 RKNPUEmitter 类名的字符串地址，以及引用它的 typeinfo"""
    f, elf, text = load(path)
    data = f.read() if False else None
    f.seek(0)
    blob = f.read()

    names = [b"N4rknn7runtime14RKNPUEmitterF2E",
             b"N4rknn7runtime14RKNPUEmitterT4E",
             b"N4rknn7runtime14RKNPUEmitterV2E",
             b"N4rknn7runtime16RKNPUEmitterLiteE",
             b"N4rknn7runtime12RKNPUEmitterE"]
    print("── 类名字符串的文件偏移 / 虚拟地址 ──")
    found = {}
    for nm in names:
        off = blob.find(nm)
        if off < 0:
            print(f"   {nm.decode()} 未找到"); continue
        # 找它属于哪个段，换算 VA
        va = None
        for s in elf.iter_sections():
            if s["sh_offset"] <= off < s["sh_offset"] + s["sh_size"]:
                va = s["sh_addr"] + (off - s["sh_offset"])
                sec = s.name
                break
        found[nm.decode()] = (off, va)
        print(f"   {nm.decode():<36} off={off:#x} va={va:#x} ({sec})")

    print("\n── 在数据段里找指向这些类名的指针（= typeinfo 的 name 字段）──")
    target = {va for _, va in found.values() if va}
    hits = []
    for s in elf.iter_sections():
        if s["sh_type"] not in ("SHT_PROGBITS", "SHT_INIT_ARRAY", "SHT_REL", "SHT_RELA"):
            continue
        if not (s["sh_flags"] & 0x2):     # SHF_ALLOC
            continue
        d = s.data()
        for i in range(0, len(d) - 8, 8):
            v = struct.unpack_from("<Q", d, i)[0]
            if v in target:
                va = s["sh_addr"] + i
                nm = [k for k, (_, a) in found.items() if a == v][0]
                hits.append((va, nm, s.name))
    for va, nm, sec in hits[:20]:
        print(f"   {va:#010x} → {nm}   ({sec})")
    if not hits:
        print("   没找到直接指针 —— 很可能用了 R_AARCH64_RELATIVE 重定位（需应用重定位再看）")
    f.close()
    return 0


def cmd_dis(addr, n=60, path=SO):
    f, elf, text = load(path)
    base = text["sh_addr"]
    off = addr - base
    md = Cs(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN)
    code = text.data()[max(0, off - 16): off + n * 4]
    print(f"── 反汇编 {addr:#x} 起 {n} 条 ──")
    for ins in md.disasm(code, base + max(0, off - 16)):
        mark = "  ←" if ins.address == addr else ""
        print(f"   {ins.address:#010x}  {ins.bytes.hex():<8} {ins.mnemonic:<10} {ins.op_str}{mark}")
    f.close()
    return 0


def main():
    c = sys.argv[1] if len(sys.argv) > 1 else "blocks"
    if c == "blocks":
        return cmd_blocks()
    if c == "rtti":
        return cmd_rtti()
    if c == "vtable":
        return cmd_vtable(sys.argv[2] if len(sys.argv) > 2 else None)
    if c == "dis":
        return cmd_dis(int(sys.argv[2], 0), int(sys.argv[3]) if len(sys.argv) > 3 else 60)
    print(__doc__)
    return 1

# ─────────── vtable 定位（处理 R_AARCH64_RELATIVE 重定位）───────────
TYPEINFO = {                       # 类名 → .data.rel.ro 里 name 指针的地址
    "RKNPUEmitter":      0x750928,
    "RKNPUEmitterF2":    0x750948,
    "RKNPUEmitterLite":  0x750960,
    "RKNPUEmitterT4":    0x750978,
    "RKNPUEmitterV2":    0x750990,
}


def build_relocs(elf):
    """{被重定位的地址: 加数}，只取 R_AARCH64_RELATIVE(=1027)"""
    rel = {}
    for s in elf.iter_sections():
        if s["sh_type"] != "SHT_RELA":
            continue
        for r in s.iter_relocations():
            if r["r_info_type"] == 1027:          # R_AARCH64_RELATIVE
                rel[r["r_offset"]] = r["r_addend"]
    return rel


def read_ptr(elf, blob, rel, va):
    off = va2off(elf, va)
    if off is None:
        return None
    if va in rel:
        return rel[va]
    return struct.unpack_from("<Q", blob, off)[0]


def cmd_vtable(which=None):
    f, elf, text = load(SO)
    f.seek(0); blob = f.read()
    rel = build_relocs(elf)
    print(f"R_AARCH64_RELATIVE 重定位 {len(rel)} 条\n")

    targets = TYPEINFO if not which else {k: v for k, v in TYPEINFO.items() if which in k}
    for name, name_ptr_va in targets.items():
        ti = name_ptr_va - 8                       # typeinfo = {vptr, name}
        # 在 .data.rel.ro 里找指向 typeinfo 的槽位 → 那是 vtable[-1]
        s = elf.get_section_by_name(".data.rel.ro")
        d = s.data(); base = s["sh_addr"]
        vt = None
        for i in range(0, len(d) - 8, 8):
            v = read_ptr(elf, blob, rel, base + i)
            if v == ti:
                vt = base + i + 8
                break
        print(f"── {name} ──")
        print(f"   typeinfo @ {ti:#x}   vtable @ {vt:#x}" if vt else
              f"   typeinfo @ {ti:#x}   ⚠️ 未找到 vtable")
        if not vt:
            continue
        # 读 vtable 条目（只取落在 .text 里的）
        tstart, tend = text["sh_addr"], text["sh_addr"] + text["sh_size"]
        for k in range(12):
            fn = read_ptr(elf, blob, rel, vt + k * 8)
            if fn is None or not (tstart <= fn < tend):
                break
            print(f"     [{k}] {fn:#010x}")
        print()
    f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
