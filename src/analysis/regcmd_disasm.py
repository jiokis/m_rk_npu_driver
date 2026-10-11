#!/usr/bin/env python3
"""
regcmd_disasm.py — regcmd 命令流反汇编器（纯静态分析，零硬件风险）

## 定位

615 条真实命令流已经抓在手里（data/dumps/）。本工具把它们从字节流
变成**可读的、带语义标注的反汇编**，并回答逆向的核心问题：

  · 一个"算子配置"由哪些寄存器写组成？
  · 哪些指令在同种算子的多次出现中**恒定**（= 模板/规格）
  · 哪些是**变量**（= 张量地址 / 形状 / 步长 / 量化参数）
  · 变量与真实缓冲区的 dma_addr 能否对上（能对上就坐实了"地址字段"）

这是"能自己生成算子配置"的前置条件 —— 也就是提升部署自由度的关键。

## 指令编码（已由实测 + 外部资料双路确认）

每条 8 字节，小端：
    word0 = [u16 reg_offset][u16 value_lo]
    word1 = [u16 value_hi ][u16 block_id ]
  即 block_id = word1 >> 16, value = (value_hi << 16) | value_lo

## 用法
  python3 regcmd_disasm.py list                  # 列出样本与算子
  python3 regcmd_disasm.py show <task号>         # 反汇编单条
  python3 regcmd_disasm.py analyze               # 逐算子做模板/变量分析
  python3 regcmd_disasm.py xref                  # 变量值与真实地址交叉比对
  python3 regcmd_disasm.py regmap                # 寄存器语义表
"""
import glob
import os
import re
import struct
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = os.environ.get("RKNPU_DUMPS") or os.path.join(HERE, "..", "..", "data", "dumps")

# block_id → 名称（来自外部资料，已用我们的 615 条样本逐条核对通过）
BLOCK = {
    0x0000: "OP_NONE",      0x0041: "OP_40",
    0x0081: "OP_ENABLE",    0x0101: "OP_REG_PC",
    0x0201: "OP_REG_CNA",   0x0801: "OP_REG_CORE",
    0x1001: "OP_REG_DPU",   0x2001: "OP_REG_DPU_RDMA",
    0x4001: "OP_REG_PPU",   0x8001: "OP_REG_PPU_RDMA",
}
# 寄存器窗口（block 与寄存器地址空间是两套编号）
WINDOW = {0x0101: 0x0, 0x0201: 0x1, 0x0801: 0x3, 0x1001: 0x4,
          0x2001: 0x5, 0x4001: 0x6, 0x8001: 0x7}

# PC block (0x0101) 里含义已知的偏移
PC_KNOWN = {0x10: "下一个 regcmd 地址（PC 链）", 0x14: "该段长度"}


def decode(buf):
    """字节流 → [(reg_off, value, block_id)]"""
    out = []
    for i in range(0, len(buf) // 8 * 8, 8):
        w0, w1 = struct.unpack_from("<II", buf, i)
        out.append((w0 & 0xFFFF, ((w1 & 0xFFFF) << 16) | (w0 >> 16), w1 >> 16))
    return out


def load_tasks():
    """
    解析 capture.log，**按 [SUBMIT] 分块**，返回 {(seq, task): 元信息}。

    ⚠️ 曾经的 bug：只用 task 号做键。但 4 次 SUBMIT 的 task 序号会重叠
    （seq0 用 0..587，seq1/2/3 用 196..212），于是不同提交的样本被混在一起，
    表现为"同种算子的寄存器集合不一致、缺项一大堆"。
    正确做法是把提交序号也纳入键。
    """
    log = os.path.join(DUMPS, "capture.log")
    out = {}
    if not os.path.exists(log):
        return out
    seq = -1
    for line in open(log, encoding="utf-8", errors="replace"):
        if line.startswith("[SUBMIT]"):
            seq += 1
            continue
        m = re.search(r"task\[(\d+)\] flags=.*?op_idx=(\d+).*?regcfg_amount=(\d+)"
                      r".*?regcmd_addr=(0x[0-9a-f]+)", line)
        if m and seq >= 0:
            out[(seq, int(m.group(1)))] = {
                "op_idx": int(m.group(2)),
                "amount": int(m.group(3)),
                "addr": int(m.group(4), 16),
            }
    return out


def load_dma():
    """capture.log 里的 MEM_CREATE → 已知的 dma_addr → 缓冲区大小"""
    log = os.path.join(DUMPS, "capture.log")
    regs = []
    if not os.path.exists(log):
        return regs
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.search(r"\[MEM_CREATE\] handle=(\d+) flags=(0x[0-9a-f]+) size=(\d+) "
                      r"obj_addr=(0x[0-9a-f]+) dma_addr=(0x[0-9a-f]+)", line)
        if m:
            regs.append({"handle": int(m.group(1)), "flags": int(m.group(2), 16),
                         "size": int(m.group(3)),
                         "dma": int(m.group(5), 16)})
    return regs


def samples():
    """{(seq, task): 字节内容}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(DUMPS, "regcmd_*.bin"))):
        m = re.search(r"regcmd_(\d+)_task(\d+)\.bin$", f)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = open(f, "rb").read()
    return out


# ─────────────── 子命令 ───────────────
def cmd_list():
    s, t = samples(), load_tasks()
    byop = defaultdict(list)
    for k in s:
        byop[t.get(k, {}).get("op_idx")].append(k)
    seqs = sorted({q for q, _ in s})
    print(f"样本 {len(s)} 条 · 覆盖 {len(byop)} 种算子 · 提交序号 {seqs}\n")
    print("  ⚠️ 只有 seq 0 是完整图（588 task）；其余是增量提交，task 号与 seq0 重叠，"
          "模板分析只应使用单次提交。\n")
    print(f"  {'op_idx':>7} {'数量':>6} {'指令数':>7} {'字节数':>7}  主要 block 组成")
    for op in sorted(k for k in byop if k is not None):
        ks = byop[op]
        ins = decode(s[ks[0]])
        blk = Counter(BLOCK.get(b, hex(b)) for _, _, b in ins)
        comp = " ".join(f"{n}×{c}" for n, c in blk.most_common(4))
        print(f"  {op:>7} {len(ks):>6} {len(ins):>7} {len(s[ks[0]]):>7}  {comp}")


def cmd_show(task, seq=0):
    s = samples()
    key = (seq, task)
    if key not in s:
        print(f"没有 seq{seq} task{task} 的样本"); return 1
    ins = decode(s[key])
    t = load_tasks().get(key, {})
    print(f"seq{seq} task{task} · op_idx={t.get('op_idx')} · {len(s[key])} 字节 · "
          f"{len(ins)} 条指令\n")
    print(f"  {'#':>3} {'block':<16} {'reg':>6} {'value':>12}")
    for i, (ro, v, b) in enumerate(ins):
        note = ""
        if b == 0x0101:
            note = "  ← " + PC_KNOWN.get(ro, "")
        print(f"  {i:>3} {BLOCK.get(b, hex(b)):<16} 0x{ro:04x} 0x{v:08x}{note}")
    return 0


def cmd_analyze():
    """
    逐算子做「模板 vs 变量」分析。

    ⚠️ 两个曾经的错误：
      1. 用整个 dump 文件比较长度 —— 那是 regcfg_amount+slack，
         不同样本撞到的缓冲区边界不同，长度自然不同。
         必须只取**真实命令流**（前 regcfg_amount 字节）。
      2. 按**位置**对齐比较 —— 同种算子的指令顺序会变，位置对齐无效
         （表现为"每个位置都不同，但取值集合只有 1 个元素"）。
         正确做法是按 (block, 寄存器) 归并，比较取值集合。
    """
    s, t = samples(), load_tasks()
    use = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    byop = defaultdict(list)
    for k in s:
        if k[0] != use:          # ← 只取单次提交，避免 task 号跨提交重叠
            continue
        op = t.get(k, {}).get("op_idx")
        if op is not None:
            byop[op].append(k)

    print(f"逐算子模板分析（仅 seq {use}，样本数 ≥4 的算子）\n")
    print(f"  {'op':>4} {'样本':>5} {'指令':>5} {'恒定':>5} {'变量':>5}  " 
          f"{'缺项':>5}  变量寄存器（前 6 个）")
    print("  " + "─" * 100)

    detail = []
    for op in sorted(byop):
        ks = byop[op]
        if len(ks) < 4:
            continue
        per = []
        for k in ks:
            amt = t[k]["amount"]
            d = defaultdict(list)
            for ro, v, b in decode(s[k][:amt]):
                d[(b, ro)].append(v)
            per.append(d)
        keys = sorted(set().union(*per))
        const, var, miss = [], [], []
        for key in keys:
            got = [d.get(key) for d in per]
            if any(g is None for g in got):
                miss.append(key); continue
            flat = [v for lst in got for v in lst]
            if len(set(flat)) == 1:
                const.append((key, flat[0]))
            else:
                var.append((key, sorted(set(flat))))
        nins = len(decode(s[ks[0]][:t[ks[0]]["amount"]]))
        names = " ".join(f"{BLOCK.get(b,hex(b)).replace('OP_REG_','')}@0x{ro:04x}"
                         for (b, ro), _ in var[:6])
        print(f"  {op:>4} {len(ks):>5} {nins:>5} {len(const):>5} {len(var):>5}  "
              f"{len(miss):>5}  {names}")
        detail.append((op, const, var, miss))

    # 挑一个样本多的算子展开细节
    if detail:
        op, const, var, miss = max(detail, key=lambda x: len(x[1]) + len(x[2]))
        print(f"\n{'='*100}\n展开 op {op} 的细节：\n")
        print(f"  ── 恒定项 {len(const)} 条（= 算子规格）──")
        for (b, ro), v in const:
            print(f"     {BLOCK.get(b,hex(b)):<16} 0x{ro:04x} = 0x{v:08x}")
        print(f"\n  ── 变量项 {len(var)} 条（= 张量地址 / 形状 / 步长…）──")
        for (b, ro), vals in var[:20]:
            rng = f"{min(vals):#x}..{max(vals):#x}"
            print(f"     {BLOCK.get(b,hex(b)):<16} 0x{ro:04x}  {len(vals):>3} 种  "
                  f"{rng}")
        if miss:
            print(f"\n  ── 只在部分样本中出现 {len(miss)} 条（条件分支）──")
            for b, ro in miss[:10]:
                print(f"     {BLOCK.get(b,hex(b)):<16} 0x{ro:04x}")
    return 0


def cmd_xref():
    """
    把命令流里的"变量值"与 capture.log 里已知的 dma_addr 交叉比对。
    能对上 → 坐实那个字段就是张量地址。
    """
    s, t = samples(), load_tasks()
    regs = load_dma()
    dmas = {r["dma"]: r for r in regs}
    print(f"已知缓冲区 {len(regs)} 个：")
    for r in regs:
        print(f"  dma={r['dma']:#010x}  size={r['size']:>9}  flags={r['flags']:#x}")
    print()

    # 收集所有出现在指令里的"像地址"的值（>0x100000，且 16 字节对齐）
    hits = defaultdict(list)
    for k, buf in s.items():
        for i, (ro, v, b) in enumerate(decode(buf)):
            if v < 0x100000:
                continue
            for d in dmas:
                if d <= v < d + dmas[d]["size"]:
                    hits[d].append((k, t.get(k, {}).get("op_idx"), i,
                                    BLOCK.get(b, hex(b)), ro, v))
                    break
    print("命令流里落在已知缓冲区内的值（= 张量地址字段的强证据）：")
    for d, lst in sorted(hits.items()):
        offs = sorted({v - d for _, _, _, _, _, v in lst})
        print(f"\n  缓冲区 dma={d:#010x}（size {dmas[d]['size']}）被引用 {len(lst)} 次")
        print(f"    出现过的偏移: {[hex(o) for o in offs[:8]]}"
              f"{' …' if len(offs) > 8 else ''}")
        for k, op, i, blk, ro, v in lst[:4]:
            print(f"      seq{k[0]} task{k[1]}(op{op}) 指令[{i}] {blk} "
                  f"reg 0x{ro:04x} = {v:#x}")
    if not hits:
        print("  ⚠️ 没有值的直接落在已知缓冲区内 —— 说明地址可能带偏移或经过重定位")
    return 0



def cmd_regmap():
    """
    产出**寄存器语义表**：对每个 (block, 寄存器)，统计它跨全部样本的行为。

    判读规则：
      · 取值唯一 → 固定常量（如使能位图、模式字）
      · 取值少（≤4 种） → 模式/档位字段
      · 取值多且是地址形态（>0x100000 且落在已知缓冲区）→ 张量地址字段
      · 取值多但数值小 → 形状/步长/长度字段
    """
    s, t = samples(), load_tasks()
    regs = load_dma()
    buf = [(r["dma"], r["dma"] + r["size"]) for r in regs]

    def sign_ext_small(v):
        """是不是「符号扩展的小负数」（如 0xffffff80 = -128）—— 这种值不是地址。

        ⚠️ 曾经的假阳性：CNA 0x1184 的取值 0xffffff80 恰好落在某缓冲区区间内，
        被误判成"张量地址"。子代理比对 Mesa/内核定义指出它其实是 PAD_VALUE（填充值）。
        教训：**"值落在缓冲区区间内"是必要不充分条件**，还要排除符号扩展小负数。
        """
        return v >= 0xFF000000 and (0x100000000 - v) < 0x400000

    def kind(vals):
        u = set(vals)
        if len(u) == 1:
            return "常量", ""
        mx = max(u)
        if mx > 0x100000:
            n_in = sum(1 for v in u if any(a <= v < b for a, b in buf)
                       and not sign_ext_small(v))
            n_addr = sum(1 for v in u if not sign_ext_small(v))
            if n_in >= max(1, n_addr) * 0.8:
                return "**张量地址**", f"{n_in}/{n_addr} 落在已知缓冲区"
            return "疑似地址", f"仅 {n_in}/{n_addr} 落在已知缓冲区"
        if len(u) <= 4:
            return "模式/档位", f"{sorted(u)[:4]}"
        return "数值字段", f"范围 {min(u):#x}..{mx:#x}"

    agg = defaultdict(list)
    used = defaultdict(set)
    for (sq, tk), b in s.items():
        meta = t.get((sq, tk))
        if not meta:
            continue
        op = meta["op_idx"]
        # ⚠️ 用**完整 dump**而不是 regcfg_amount 截断：
        # 实测发现确凿的张量地址寄存器（CNA 0x1070 / 0x1110、DPU 0x4020）
        # 出现在指令 [26]/[35]，**超出声明的 regcfg_amount**。
        # 而"值精确落在真实缓冲区里"是比算术推导更强的证据，
        # 所以以完整 dump 为准。regcfg_amount 的单位仍是未决问题（见 docs/10）。
        for ro, v, blk in decode(b):
            agg[(blk, ro)].append(v)
            used[(blk, ro)].add(op)

    print(f"寄存器语义表（按 block 分组，共 {len(agg)} 个不同寄存器）\n")
    cur = None
    for (blk, ro) in sorted(agg):
        if blk != cur:
            cur = blk
            print(f"\n  ══ {BLOCK.get(blk, hex(blk))} (block 0x{blk:04x}, "
                  f"寄存器窗口 0x{WINDOW.get(blk,0):x}xxx) ══")
            print(f"     {'寄存器':>8} {'样本数':>7} {'取值数':>7} {'算子数':>7}  "
                  f"{'判读':<12} 备注")
        vals = agg[(blk, ro)]
        k, note = kind(vals)
        print(f"     0x{ro:04x} {len(vals):>8} {len(set(vals)):>8} "
              f"{len(used[(blk,ro)]):>8}  {k:<12} {note}")
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    c = sys.argv[1]
    if c == "list":
        return cmd_list()
    if c == "show":
        return cmd_show(int(sys.argv[2]))
    if c == "analyze":
        return cmd_analyze()
    if c == "xref":
        return cmd_xref()
    if c == "regmap":
        return cmd_regmap()
    print(__doc__); return 1


if __name__ == "__main__":
    sys.exit(main())
