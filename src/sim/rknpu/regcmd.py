"""
rknpu.regcmd — 指令编解码与 block 定义

## 指令编码（已由实测 + 外部资料双路确认）

每条 **8 字节**，小端：
    word0 = [u16 reg_offset][u16 value_lo]
    word1 = [u16 value_hi ][u16 block_id ]
    ↓
    block_id = word1 >> 16
    value    = (value_hi << 16) | value_lo

## 为什么确信是 8 字节
解出的 reg_offset 全部落在规整的寄存器 bank 里（0x10xx CNA、0x40xx DPU…），
且与 GPL 头文件的 RKNPU_OFFSET_* 偏移表吻合。试过 6 字节步长，
解出来的是 block_id 值，明显错位。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, List

INSN_SIZE = 8

# ── block 定义（615 条真实样本逐条核对通过）──
@dataclass(frozen=True)
class Block:
    id: int
    name: str
    window: int | None      # 寄存器窗口高半字节（0x1 → 0x1xxx），None = 无窗口
    note: str = ""


BLOCKS = {
    0x0000: Block(0x0000, "NONE",       None, "空操作"),
    0x0041: Block(0x0041, "OP40",       None, "op_en 前置"),
    0x0081: Block(0x0081, "ENABLE",     None, "总使能位图（reg 0x0008）"),
    0x0101: Block(0x0101, "PC",         0x0,  "程序控制 / 链"),
    0x0201: Block(0x0201, "CNA",        0x1,  "卷积主算"),
    0x0801: Block(0x0801, "CORE",       0x3,  "核心配置"),
    0x1001: Block(0x1001, "DPU",        0x4,  "数据通路"),
    0x2001: Block(0x2001, "DPU_RDMA",   0x5,  "DPU 读 DMA"),
    0x4001: Block(0x4001, "PPU",        0x6,  "像素/后处理"),
    0x8001: Block(0x8001, "PPU_RDMA",   0x7,  "PPU 读 DMA"),
}


def block_name(bid: int) -> str:
    b = BLOCKS.get(bid)
    return b.name if b else f"0x{bid:04x}"


@dataclass(frozen=True)
class Insn:
    """一条 regcmd 指令"""
    block: int
    reg: int
    value: int
    index: int = 0

    @property
    def block_name(self) -> str:
        return block_name(self.block)

    def __str__(self) -> str:
        return (f"[{self.index:>4}] {self.block_name:<8} "
                f"0x{self.reg:04x} 0x{self.value:08x}")

    def pack(self) -> bytes:
        """打包回 8 字节（供汇编器使用）"""
        w0 = (self.reg & 0xFFFF) | ((self.value & 0xFFFF) << 16)
        w1 = ((self.value >> 16) & 0xFFFF) | ((self.block & 0xFFFF) << 16)
        return struct.pack("<II", w0, w1)


def decode(buf: bytes, limit: int | None = None) -> List[Insn]:
    """字节流 → 指令列表。limit 指最多解码多少条。"""
    out = []
    n = len(buf) // INSN_SIZE
    if limit is not None:
        n = min(n, limit)
    for i in range(n):
        w0, w1 = struct.unpack_from("<II", buf, i * INSN_SIZE)
        out.append(Insn(block=w1 >> 16,
                        reg=w0 & 0xFFFF,
                        value=((w1 & 0xFFFF) << 16) | (w0 >> 16),
                        index=i))
    return out


def encode(insns: Iterable[Insn]) -> bytes:
    """指令列表 → 字节流（汇编器的出口）"""
    return b"".join(x.pack() for x in insns)


class RegcmdBuilder:
    """
    命令流构造器 —— 生成自定义 regcmd 的起点。

    ⚠️ 目前只做**编码**，不做语义校验。真正的"造一个能跑通的算子配置"
    还需要字段语义（等 NDA 对照）与 PC 链构造（已有模板）。
    """

    def __init__(self):
        self._ins: List[Insn] = []

    def emit(self, block: int, reg: int, value: int) -> "RegcmdBuilder":
        self._ins.append(Insn(block, reg, value, index=len(self._ins)))
        return self

    def emit_none(self, n: int = 1) -> "RegcmdBuilder":
        for _ in range(n):
            self.emit(0x0000, 0x0000, 0)
        return self

    def emit_enable(self, bitmap: int) -> "RegcmdBuilder":
        """总使能。实测取值：0x18(24) / 0x1D(29) / 0x60(96)"""
        return self.emit(0x0081, 0x0008, bitmap)

    def emit_op40(self) -> "RegcmdBuilder":
        """op_en 前置（每个真实样本里 op_en 之前都有它）"""
        return self.emit(0x0041, 0x0000, 0)

    def emit_pc_chain(self, next_addr: int, length: int) -> "RegcmdBuilder":
        """PC 链：指向下一段。next_addr=0 表示链结束。"""
        self.emit(0x0101, 0x0010, next_addr)
        self.emit(0x0101, 0x0014, length)
        return self

    def build(self) -> bytes:
        return encode(self._ins)

    def __len__(self) -> int:
        return len(self._ins)

    def __iter__(self):
        return iter(self._ins)
