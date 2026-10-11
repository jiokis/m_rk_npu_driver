"""
rknpu.machine — RKNPU 结构级模拟器

## 定位（重要）

**结构级**模拟：把硬件建模成
  · 一个寄存器空间（153 个已实测寄存器）
  · 一个稀疏设备内存
  · 一条 PC 链驱动的执行流程

它**不**计算卷积结果 —— 因为 DMA/算子的字段语义还需 NVDLA 对照才能确定。
但它已经能做很多实事：

  · 把真实 dump 跑一遍，验证我们对结构的理解自不自洽
  · 当反汇编/汇编的宿主（trace、状态快照、往返测试）
  · 回归：改了假设之后，615 条样本还能不能全部"跑通"
  · 为后续**功能级**建模留好接口（hooks）

## 执行模型（已由实测确证）

    task 槽位内 = regcfg_amount 条指令（每条 8 字节）
    槽位最后一条通常是 PC 链指令：
        block 0x0101 / reg 0x0010 = 下一段地址（0 = 链结束）
        block 0x0101 / reg 0x0014 = 段长度
    ⚠️ 实测修正：PC 链是**「下一个 task」的指针**（610/611 的链目标 == 下一个 task 地址），
    链值从不為 0。硬件执行多少个 task 由 PC_TASK_CONTROL 的 task_number 决定，
    不是靠链值为 0 终止。

    所以**算子边界不是链，而是「连续相同 op_idx 的 task 段」**（615 task → 229 算子）。
    链只是说明 task 在内存里怎么串起来的。见 docs/14。

见 docs/12（regcfg_amount 语义）与 docs/13（算子 = task 链）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import regcmd as rc
from . import regmap as rm

Hook = Callable[["Machine", rc.Insn], None]


# ─────────────────────────── 设备内存 ───────────────────────────
class Memory:
    """稀疏设备内存。地址按 dump 分块载入，未覆盖的地址读为 None。"""

    def __init__(self):
        self._blocks: List[Tuple[int, int, bytes]] = []
        self._index: Dict[int, int] = {}          # start -> block 下标

    def load(self, addr: int, data: bytes) -> None:
        self._index[addr] = len(self._blocks)
        self._blocks.append((addr, addr + len(data), data))
        self._blocks.sort()
        self._index = {s: i for i, (s, _, _) in enumerate(self._blocks)}

    def find(self, addr: int) -> Optional[Tuple[int, bytes]]:
        """
        返回 (该块内偏移, 剩余字节)，未覆盖返回 None。

        ⚠️ 曾经的 bug：直接返回**第一个**包含该地址的块。
        但 dump 之间会重叠 —— 抓取时每个槽位 dump 了 `amount*8 + 64` 字节，
        而槽位间距只有 ~192 字节，于是 dump 会伸进下一个 task 的起始处十几字节。
        若返回那个"先开始的"块，剩下的字节数不够，read 就返回 None，
        表现为"379 个段读不出来"。

        正解：在覆盖该地址的所有块里，选**起始地址最大**的那个（最具体的）。
        """
        best = None
        for s, e, b in self._blocks:
            if s <= addr < e and (best is None or s > best[0]):
                best = (s, e, b)
        if best is None:
            return None
        s = best[0]
        return addr - s, best[2][addr - s:]

    def read(self, addr: int, n: int) -> Optional[bytes]:
        hit = self.find(addr)
        if hit is None:
            return None
        off, rest = hit
        return rest[:n] if len(rest) >= n else None

    def covers(self, addr: int) -> bool:
        return self.find(addr) is not None

    @property
    def spans(self):
        return [(s, e) for s, e, _ in self._blocks]

    @property
    def total(self) -> int:
        return sum(e - s for s, e, _ in self._blocks)


# ─────────────────────────── 寄存器空间 ───────────────────────────
class RegisterFile:
    """{(block, offset): (value, write_count)}"""

    def __init__(self):
        self._r: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.history: List[Tuple[int, int, int]] = []     # (block, reg, value) 按序

    def write(self, block: int, reg: int, value: int) -> None:
        prev = self._r.get((block, reg))
        self._r[(block, reg)] = (value, (prev[1] if prev else 0) + 1)
        self.history.append((block, reg, value))

    def read(self, block: int, reg: int) -> Optional[int]:
        v = self._r.get((block, reg))
        return v[0] if v else None

    def writes(self, block: int, reg: int) -> int:
        v = self._r.get((block, reg))
        return v[1] if v else 0

    def items(self):
        return {k: v[0] for k, v in self._r.items()}

    def dump(self, only_known: bool = True) -> List[str]:
        out = []
        for (b, r) in sorted(self._r):
            v, n = self._r[(b, r)]
            role = rm.describe(b, r)
            if only_known and role == "未观测":
                continue
            out.append(f"  {rc.block_name(b):<8} 0x{r:04x} = 0x{v:08x}"
                       f"   ×{n}   {role}")
        return out

    def __len__(self):
        return len(self._r)


# ─────────────────────────── 任务与链 ───────────────────────────
@dataclass
class Task:
    seq: int
    index: int
    op_idx: int
    addr: int
    amount: int                     # 本段指令条数

    @property
    def key(self):
        return (self.seq, self.index)


@dataclass
class Segment:
    task: Task
    insns: List[rc.Insn]
    next_addr: int = 0
    chain: Optional[rc.Insn] = None      # 段尾 PC 链：下一段地址（0x0010）
    chain_len: Optional[rc.Insn] = None  # 段尾 PC 链：下一段 PC_DATA_AMOUNT（0x0014）
    end: str = ""                   # "chain"/"end"/"nochain"/"uncaptured"/"cycle"
    truncated: bool = False         # dump 未覆盖完整段


@dataclass
class Chain:
    """一条 task 链 = 从某个 task 顺着 next 指针走出来的序列

    注意：这**不是**算子边界。算子边界见 docs/14（连续同 op_idx）。
    """
    segments: List[Segment] = field(default_factory=list)

    @property
    def op_idx(self) -> Optional[int]:
        return self.segments[0].task.op_idx if self.segments else None

    @property
    def total_insns(self) -> int:
        return sum(len(s.insns) for s in self.segments)

    @property
    def complete(self) -> bool:
        return bool(self.segments) and self.segments[-1].end == "end"

    def block_histogram(self) -> Dict[str, int]:
        h: Dict[str, int] = {}
        for s in self.segments:
            for i in s.insns:
                h[i.block_name] = h.get(i.block_name, 0) + 1
        return dict(sorted(h.items(), key=lambda x: -x[1]))

    def tensor_addrs(self) -> List[Tuple[str, int, int]]:
        """[(block名, 寄存器, 值)]，只列语义已确证的地址寄存器"""
        out = []
        for s in self.segments:
            for i in s.insns:
                if rm.is_tensor_addr(i.block, i.reg):
                    out.append((i.block_name, i.reg, i.value))
        return out

    def summary(self) -> str:
        h = self.block_histogram()
        top = " ".join(f"{k}×{v}" for k, v in list(h.items())[:5])
        return (f"op{self.op_idx} · {len(self.segments)} task · "
                f"{self.total_insns} 条指令 · "
                f"{'完整' if self.complete else '不完整'} · {top}")


# ─────────────────────────── 模拟器 ───────────────────────────
class Machine:
    def __init__(self, memory: Optional[Memory] = None,
                 hooks: Optional[List[Hook]] = None):
        self.mem = memory or Memory()
        self.regs = RegisterFile()
        self.hooks = hooks or []
        self.stats = {"insns": 0, "segments": 0, "chains": 0,
                      "unknown_regs": 0, "uncaptured": 0}
        self.unknown_regs: Dict[Tuple[int, int], int] = {}

    # ── 单段执行 ──
    def run_segment(self, task: Task) -> Segment:
        # 需要 amount 条「体」+ 2 条 PC 链（地址 + 长度）
        need = (task.amount + 2) * rc.INSN_SIZE
        buf = self.mem.read(task.addr, need)
        truncated = buf is None
        if truncated:                       # 覆盖不足 → 尽量读能读到的部分
            hit = self.mem.find(task.addr)
            buf = (hit[1] if hit else b"")[:need]
            self.stats["uncaptured"] += 1
        # ★ 段 = amount 条「体」+ **2 条** PC 链（地址 + 长度）。
        #   实测：0x0010 精确在 index == amount（611/611），0x0014 在其后一条。
        all_ins = rc.decode(buf, limit=task.amount + 2)
        body = all_ins[: task.amount]
        chain = all_ins[task.amount] if len(all_ins) > task.amount else None
        chain_len = (all_ins[task.amount + 1]
                     if len(all_ins) > task.amount + 1
                     and (all_ins[task.amount + 1].block,
                          all_ins[task.amount + 1].reg) == rm.PC_CHAIN_LEN
                     else None)

        seg = Segment(task=task, insns=body, truncated=truncated)
        for i in body:
            self._exec(i)
        if chain is not None:
            seg.chain = chain
            self._exec(chain)
            if (chain.block, chain.reg) == rm.PC_CHAIN_ADDR:
                seg.next_addr = chain.value
        if chain_len is not None:
            seg.chain_len = chain_len
            self._exec(chain_len)
        if truncated:
            seg.end = "uncaptured"
        elif chain is None:
            seg.end = "nochain"
        elif seg.next_addr == 0:
            seg.end = "end"          # 链值为 0 = 显式结束（实测罕见）
        else:
            seg.end = "chain"
        self.stats["segments"] += 1
        return seg

    def _exec(self, i: rc.Insn) -> None:
        self.regs.write(i.block, i.reg, i.value)
        self.stats["insns"] += 1
        if rm.lookup(i.block, i.reg) is None and (i.block, i.reg) != rm.ENABLE_REG:
            self.unknown_regs[(i.block, i.reg)] = \
                self.unknown_regs.get((i.block, i.reg), 0) + 1
        for h in self.hooks:
            h(self, i)

    # ── 顺链执行 ──
    def run_chain(self, start: Task, by_addr: Dict[int, Task],
                  max_seg: int = 64) -> Chain:
        ch = Chain()
        cur: Optional[Task] = start
        seen = set()
        while cur is not None and len(ch.segments) < max_seg:
            if cur.key in seen:             # 环
                ch.segments.append(Segment(task=cur, insns=[], end="cycle"))
                break
            seen.add(cur.key)
            seg = self.run_segment(cur)
            ch.segments.append(seg)
            cur = by_addr.get(seg.next_addr) if seg.next_addr else None
        self.stats["chains"] += 1
        return ch


    # ── 遍历整个 task 列表（每个段只执行一次，O(n)）──
    def run_all(self, tasks) -> List[Segment]:
        """
        按顺序执行每个 task 的段，各执行一次。

        为什么不逐个 run_chain：链是指向下一个 task 的链表，
        从每个 task 出发走链会变成 O(n²)（615 个 task 会跑出 350 万条指令）。
        验证"结构自洽"只需要每个段执行一次。
        """
        segs = []
        for _, _, t in tasks:
            seg = self.run_segment(t)
            segs.append(seg)
        return segs

    @staticmethod
    def chain_links(segs: List[Segment], by_addr) -> dict:
        """统计链的指向是否符合预期（应指向下一个 task）"""
        ok = bad = zero = unknown = 0
        for seg in segs:
            if seg.chain is None:
                bad += 1; continue
            if seg.next_addr == 0:
                zero += 1
            elif seg.next_addr in by_addr:
                ok += 1
            else:
                unknown += 1
        return {"指向已知 task": ok, "指向未知地址": unknown,
                "链值为 0": zero, "无链指令": bad}

    # ── PC 链长度字段的语义（已全量验证 610/610）──
    PC_DATA_EXTRA = 4     # RKNPU_PC_DATA_EXTRA_AMOUNT
    PC_DATA_SCALE = 2     # pc_data_amount_scale（RK3588）

    @classmethod
    def pc_chain_amount(cls, target_regcfg_amount: int) -> int:
        """
        算出链指令 0x0014 的低 16 位 —— 就是内核写 PC_DATA_AMOUNT 的公式：
            (regcfg_amount + EXTRA + scale − 1) / scale − 1
        高 16 位是一个相位标志（实测只在某段 task 区间置位），本函数不含。
        """
        a, e, s = target_regcfg_amount, cls.PC_DATA_EXTRA, cls.PC_DATA_SCALE
        return (a + e + s - 1) // s - 1

    # ── 状态快照 ──
    def state_dump(self) -> str:
        lines = [f"指令执行 {self.stats['insns']} 条 · "
                 f"段 {self.stats['segments']} · 链 {self.stats['chains']}",
                 f"寄存器被写 {len(self.regs)} 个不同地址"
                 f"（历史 {len(self.regs.history)} 次写）"]
        if self.unknown_regs:
            lines.append(f"⚠️ 未观测过的寄存器 {len(self.unknown_regs)} 个:")
            for (b, r), n in sorted(self.unknown_regs.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"    {rc.block_name(b):<8} 0x{r:04x}  ×{n}")
        return "\n".join(lines)
