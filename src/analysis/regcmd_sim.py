#!/usr/bin/env python3
"""
regcmd_sim.py — regcmd 软件模拟器（不依赖硬件）

## 为什么要有它

逆向不能只有一种手段。抓取（动态）告诉我们"硬件收到什么"，
反汇编（静态）告诉我们"代码想干什么"，而**模拟器**能回答
"我的假设自不自洽" —— 而且**零硬件风险、跑得比等板子快得多**。

具体到当前卡住的问题：`regcfg_amount` 的单位。
op1 声明 126 字节，但 126÷8 = 15.75 不是整数，而确凿的地址寄存器
`0x1070` 出现在指令[26]（第 208 字节）。到底哪个是真实边界？

模拟器把命令流展开成**寄存器写时序**，就能从结构上看出边界：
  · 流在哪里重新开始（首条指令重复）
  · PC 链指令（block 0x0101 / reg 0x0010）出现在哪
  · 这些位置与 regcfg_amount 是什么关系

## 模型

把 NPU 寄存器空间建成 { (block, reg): [写入历史] }。
执行一条命令流 = 按顺序记录每次写。硬件细节（计算）不建模 ——
我们需要的是**控制流的形状**，不是数值结果。

## 用法
  python3 regcmd_sim.py probe        # 用真实样本检验边界假设
  python3 regcmd_sim.py trace <task> # 打印一条流的寄存器写时序
  python3 regcmd_sim.py replay <file># 执行任意 regcmd 文件
"""
import glob
import os
import re
import struct
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = os.environ.get("RKNPU_DUMPS") or os.path.join(HERE, "..", "..", "data", "dumps")

BLOCK = {0x0000: "NONE", 0x0041: "OP40", 0x0081: "EN", 0x0101: "PC",
         0x0201: "CNA", 0x0801: "CORE", 0x1001: "DPU",
         0x2001: "DPU_RD", 0x4001: "PPU", 0x8001: "PPU_RD"}


class NPUSim:
    """NPU 寄存器空间的软件模型"""

    def __init__(self):
        self.regs = defaultdict(list)   # (block, reg) → [值序列]
        self.trace = []                 # [(idx, block, reg, value)]
        self.dma_base = 0
        self.pc_segments = []           # PC 链记录的段边界

    def write(self, block, reg, value):
        i = len(self.trace)
        self.trace.append((i, block, reg, value))
        self.regs[(block, reg)].append(value)
        # PC 链：block PC / reg 0x0010 = 下一段命令流地址
        if block == 0x0101 and reg == 0x0010:
            self.pc_segments.append((i, value))
        if block == 0x0101 and reg == 0x0014:
            pass

    def run(self, buf):
        for i in range(0, len(buf) // 8 * 8, 8):
            w0, w1 = struct.unpack_from("<II", buf, i)
            self.write(w1 >> 16, w0 & 0xFFFF,
                       ((w1 & 0xFFFF) << 16) | (w0 >> 16))
        return self

    # ── 分析 ──
    def first_repeat(self):
        """首条指令在流中第二次出现的位置（= 新一段开始的强信号）"""
        if not self.trace:
            return None
        first = self.trace[0][1:]
        for i, b, r, v in self.trace[1:]:
            if (b, r, v) == first:
                return i
        return None

    def writes_per_reg(self):
        d = defaultdict(int)
        for _, b, r, _ in self.trace:
            d[(b, r)] += 1
        return d


def load_meta():
    """(seq,task) → {op_idx, amount, addr}"""
    log = os.path.join(DUMPS, "capture.log")
    out, seq = {}, -1
    for line in open(log, encoding="utf-8", errors="replace"):
        if line.startswith("[SUBMIT]"):
            seq += 1; continue
        m = re.search(r"task\[(\d+)\] flags=.*?op_idx=(\d+).*?regcfg_amount=(\d+)"
                      r".*?regcmd_addr=(0x[0-9a-f]+)", line)
        if m and seq >= 0:
            out[(seq, int(m.group(1)))] = {"op_idx": int(m.group(2)),
                                           "amount": int(m.group(3))}
    return out


def load_samples():
    out = {}
    for f in sorted(glob.glob(os.path.join(DUMPS, "regcmd_*.bin"))):
        m = re.search(r"regcmd_(\d+)_task(\d+)\.bin$", f)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = open(f, "rb").read()
    return out


def cmd_probe():
    """用真实样本检验：真实流边界到底在哪？"""
    meta, samp = load_meta(), load_samples()
    print("═══ 边界假设检验（seq 0 的样本）═══\n")
    print(f"  {'task':>5} {'op':>4} {'声明amt':>8} {'amt/8':>7} {'文件长':>7} "
          f"{'首条重复@':>10} {'PC链@':>16} {'重复×8':>7}")
    print("  " + "─" * 78)

    rows = []
    for k in sorted(samp):
        if k[0] != 0:
            continue
        m = meta.get(k)
        if not m:
            continue
        sim = NPUSim().run(samp[k])
        fr = sim.first_repeat()
        pcs = [i for i, _ in sim.pc_segments]
        amt = m["amount"]
        rows.append((k[1], m["op_idx"], amt, len(sim.trace), fr, pcs))
        if len(rows) <= 25:
            print(f"  {k[1]:>5} {m['op_idx']:>4} {amt:>8} {amt/8:>7.2f} "
                  f"{len(sim.trace):>7} {str(fr):>10} {str(pcs[:3]):>16} "
                  f"{(fr*8 if fr else '-'):>7}")

    print(f"\n  共 {len(rows)} 条样本参与分析")
    # 统计：首条重复的位置分布
    frs = [r[4] for r in rows if r[4] is not None]
    if frs:
        from collections import Counter
        c = Counter(frs)
        print(f"\n  ── 首条重复位置分布（前 8）──")
        for pos, n in c.most_common(8):
            print(f"    第 {pos:>3} 条 ({pos*8:>4} 字节)  {n:>4} 次")
    # regcfg_amount 与指令数的关系
    print(f"\n  ── regcfg_amount 与「到首条重复为止的字节数」的关系 ──")
    pairs = [(r[2], r[4] * 8) for r in rows if r[4]]
    if pairs:
        ratios = sorted(set(round(a / b, 3) for a, b in pairs if b))
        print(f"    regcfg_amount / (重复位置×8) 的取值: {ratios[:12]}")
    return 0


def cmd_trace(task, seq=0):
    meta, samp = load_meta(), load_samples()
    k = (seq, task)
    if k not in samp:
        print(f"无样本 {k}"); return 1
    sim = NPUSim().run(samp[k])
    m = meta.get(k, {})
    print(f"seq{seq} task{task} · op{m.get('op_idx')} · 声明 amt={m.get('amount')} "
          f"· 实际解码 {len(sim.trace)} 条\n")
    print(f"  {'#':>4} {'block':<8} {'reg':>6} {'value':>12}   备注")
    for i, b, r, v in sim.trace:
        note = ""
        if b == 0x0101 and r == 0x0010:
            note = "← PC 链：下一段地址"
        elif b == 0x0101 and r == 0x0014:
            note = "← PC 链：段长度"
        elif b == 0x0081:
            note = "← 总使能位图"
        dup = sim.regs[(b, r)]
        if len(dup) > 1 and dup.index(v) < dup.count(v) - 1:
            pass
        print(f"  {i:>4} {BLOCK.get(b,hex(b)):<8} 0x{r:04x} 0x{v:08x}   {note}")
    print(f"\n  ── 被写多次的寄存器（= 分阶段配置的特征）──")
    for (b, r), vs in sim.writes_per_reg().items():
        if vs > 1:
            print(f"    {BLOCK.get(b,hex(b)):<8} 0x{r:04x}  写了 {vs} 次")
    return 0


def main():
    c = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if c == "probe":
        return cmd_probe()
    if c == "trace":
        return cmd_trace(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
