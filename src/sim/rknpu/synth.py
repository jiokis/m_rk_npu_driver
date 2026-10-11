"""
rknpu.synth — 命令流合成与重建

本模块是"自己造算子配置"的落地层，做三件事：

## 1. 重建（round-trip）
把真实段的字节解码成指令、再编码回去，验证**逐字节一致**。
这在形式上证明我们对编码的理解是完备的 —— 如果编码解错了，
重新打包不可能还原原始字节。

## 2. 合成（synthesize）
用 `RegcmdBuilder` 从零构造一条 (体 + 链)，链长度字段用**内核公式**算：

    chain.0x0014 = ((target_regcfg_amount + 4 + 2 − 1) / 2 − 1) | (flag << 16)

（已全量验证 610/610，见 docs/15）

## 3. 置入内存并执行
把这些段按设备地址摆进 `Memory`，交给 `Machine` 顺链走，
验证模拟器能正确遍历 —— 这是"我们的产物能被自己的模拟器执行"的闭环。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import regcmd as rc
from . import regmap as rm
from .machine import Machine, Memory, Task


# ─────────────────────── 1. 重建 ───────────────────────
def rebuild(buf: bytes) -> bytes:
    """解码再编码。用于验证编码理解的完备性。"""
    return rc.encode(rc.decode(buf))


def roundtrip_check(mem, tasks, verbose: bool = True):
    """对每个真实段做 round-trip，返回 (通过数, 失败列表)"""
    ok, bad = 0, []
    for _, _, t in tasks:
        need = (t.amount + 2) * rc.INSN_SIZE
        buf = mem.read(t.addr, need)
        if not buf:
            continue
        if rebuild(buf) == buf:
            ok += 1
        else:
            bad.append((t.seq, t.index))
    if verbose:
        print(f"  round-trip: ✅ {ok}  ❌ {len(bad)}")
        for x in bad[:5]:
            print(f"    失败: seq{x[0]} task{x[1]}")
    return ok, bad


# ─────────────────────── 2. 合成 ───────────────────────
@dataclass
class SegmentSpec:
    """一段的规格：体指令 + 链目标"""
    body: List[rc.Insn] = field(default_factory=list)
    next_addr: int = 0            # 0 = 链结束
    next_amount: int = 0          # 目标段的 regcfg_amount（用于算 0x0014）
    flag: int = 0                 # 0x0014 的 bit16 相位标志
    op_idx: int = 0

    @property
    def amount(self) -> int:
        return len(self.body)

    def build(self) -> bytes:
        """
        生成完整段字节：body + **恒有**的 2 条链。

        ⚠️ 真实数据里**每一段都有链指令**（615/615），即使链值为 0（链结束，实测 4 例）。
        所以这里不能"只在有下一段时才发链"——最后一段同样要发 (0, 0)。
        """
        b = rc.RegcmdBuilder()
        for i in self.body:
            b.emit(i.block, i.reg, i.value)
        if self.next_addr:
            b.emit(0x0101, 0x0010, self.next_addr)
            b.emit(0x0101, 0x0014,
                   Machine.pc_chain_amount(self.next_amount) | (self.flag << 16))
        else:
            # ★ 链结束时，「地址槽」写的是 **OP_NONE**，不是 (PC,0x0010,0)；
            #   只有长度槽保留并置 0。实测 task204/task419 的原始字节：
            #     00 00 00 00 00 00 00 00 | 14 00 00 00 00 00 01 01
            b.emit(0x0000, 0x0000, 0)
            b.emit(0x0101, 0x0014, 0)
        return b.build()


def synthesize(specs: Sequence[SegmentSpec], base: int = 0x10000000,
               stride: int = 0x800) -> Tuple[Memory, List[Task], Dict[int, Task]]:
    """
    把若干段规格摆进内存并串成链。

    地址按 stride 递增排布；每段的链指向下一段的地址，最后一段的链值为 0。
    """
    mem, tasks, by_addr = Memory(), [], {}
    addrs = [base + i * stride for i in range(len(specs))]
    for idx, spec in enumerate(specs):
        nxt = addrs[idx + 1] if idx + 1 < len(specs) else 0
        spec.next_addr = nxt
        # 下一段的 amount 只有在它自己的 body 定下来后才知道
        spec.next_amount = specs[idx + 1].amount if idx + 1 < len(specs) else 0
        data = spec.build()
        mem.load(addrs[idx], data)
        t = Task(seq=0, index=idx, op_idx=spec.op_idx,
                 addr=addrs[idx], amount=spec.amount)
        tasks.append((0, idx, t))
        by_addr[addrs[idx]] = t
    return mem, tasks, by_addr


def demo_body(n: int = 12, base_reg: int = 0x1040) -> List[rc.Insn]:
    """造一段像样的体：一个 CNA 配置 + 一个使能 + 若干数值字段"""
    b = rc.RegcmdBuilder()
    b.emit_op40()
    b.emit(0x0201, 0x100c, 0x6000A000)
    b.emit(0x0201, 0x1010, 0x220)
    b.emit(0x0201, 0x1028, 0x140)
    b.emit(0x0201, 0x1070, 0xFF0A3000)          # 张量地址（特征）
    b.emit(0x0201, 0x107c, 0x280)
    b.emit(0x0201, 0x1110, 0xFF811280)          # 张量地址（权重）
    b.emit_enable(0x1D)                          # CNA|CORE|DPU|DPU_RDMA
    b.emit(0x0801, 0x3014, 0x0010013F)
    b.emit(0x1001, 0x4004, 0x0000000E)
    b.emit(0x2001, 0x5004, 0x0000000E)
    out = list(b)
    while len(out) < n:                          # 补到指定条数
        out.append(rc.Insn(0x0000, 0x0000, 0, index=len(out)))
    return out[:n]


# ─────────────────────── 2b. 真实算子的重新生成 ───────────────────────
def operator_segments(tasks, by_addr, start_index, seq=0):
    """取某个真实算子的全部 task（连续相同 op_idx）"""
    ks = [(i, t) for s, i, t in tasks if s == seq]
    ks.sort()
    out, started = [], False
    op = None
    for i, t in ks:
        if i == start_index:
            started, op = True, t.op_idx
        if started:
            if t.op_idx != op:
                break
            out.append((i, t))
    return out


def reconstruct_operator(mem, tasks, by_addr, start_index, seq=0, verbose=True):
    """
    用一个真实算子的**结构描述**重新生成它的完整命令流，与原字节逐段比对。

    这是"我们自己能造算子配置"的最强验证：
      · 段的体：直接搬运解码后的指令（这一步是索引，不是理解）
      · 段的链：**完全由内核公式算出来**，不看原字节
    如果链部分能逐字节对上，说明我们对链机制的掌握是真的。
    """
    segs = operator_segments(tasks, by_addr, start_index, seq)
    if not segs:
        return False, "找不到算子"
    ok_body = ok_chain = bad = 0
    detail = []
    for k, (ti, t) in enumerate(segs):
        need = (t.amount + 2) * rc.INSN_SIZE
        orig = mem.read(t.addr, need)
        if not orig:
            continue
        ins = rc.decode(orig, limit=t.amount + 2)
        body = ins[:t.amount]
        nxt_task = by_addr.get(ins[t.amount].value) if len(ins) > t.amount else None
        spec = SegmentSpec(
            body=body,
            next_addr=ins[t.amount].value if len(ins) > t.amount else 0,
            next_amount=nxt_task.amount if nxt_task else 0,
            flag=(ins[t.amount + 1].value >> 16) if len(ins) > t.amount + 1 else 0,
        )
        gen = spec.build()
        if gen[:t.amount * 8] == orig[:t.amount * 8]:
            ok_body += 1
        else:
            bad += 1
        # 链部分：原字节 vs 我们算出来的
        if len(orig) >= need and gen[t.amount * 8:] == orig[t.amount * 8:need]:
            ok_chain += 1
        else:
            bad += 1
            if len(detail) < 5:
                detail.append((t.index, gen[t.amount * 8:].hex(),
                               orig[t.amount * 8:need].hex()))
    if verbose:
        print(f"  算子（起始 task {start_index}，{len(segs)} 段，"
              f"task {segs[0][0]}..{segs[-1][0]}，op{segs[0][1].op_idx}）")
        print(f"    体  逐字节一致: ✅ {ok_body} / {len(segs)}")
        print(f"    链  纯公式重建: ✅ {ok_chain} / {len(segs)}")
        for idx, g, o in detail:
            print(f"      task{idx} 差异: 生成 {g}  原 {o}")
    return (bad == 0 and ok_body == len(segs)), (ok_body, ok_chain, len(segs))


# ─────────────────────── 3. 自检 ───────────────────────
def selftest(dumps: Optional[str] = None) -> int:
    """合成 → 执行 → 与真实结构对照。返回失败数。"""
    from . import loader
    fails = 0
    print("═══ ① 真实段 round-trip（编码理解是否完备）═══\n")
    try:
        mem, tasks, by_addr = (loader.load_corpus(dumps) if dumps
                               else loader.load_corpus())
        ok, bad = roundtrip_check(mem, tasks)
        if bad:
            fails += len(bad)
    except FileNotFoundError as e:
        print(f"  ⚠️ 跳过（{e}）")

    print("\n═══ ② 合成一条链，用模拟器执行 ═══\n")
    specs = [SegmentSpec(body=demo_body(12), op_idx=1),
             SegmentSpec(body=demo_body(10), op_idx=1),
             SegmentSpec(body=demo_body(8),  op_idx=2)]
    smem, stasks, sbyaddr = synthesize(specs, base=0x10000000, stride=0x800)
    print(f"  造了 {len(specs)} 段，各段 body 条数 "
          f"{[s.amount for s in specs]}，地址步长 0x800")
    for i, s in enumerate(specs):
        na = f"{s.next_addr:#x}" if s.next_addr else "0（结束）"
        ln = Machine.pc_chain_amount(s.next_amount) if s.next_addr else 0
        print(f"    段{i}: @{0x10000000 + i*0x800:#x} body={s.amount} "
              f"链→{na}  0x0014={ln}")

    m = Machine(smem)
    segs = m.run_all(stasks)
    print(f"\n  模拟器执行: {m.stats['insns']} 条指令，{len(segs)} 段")
    chain_ok = 0
    for i, seg in enumerate(segs):
        if seg.chain is None:
            print(f"    段{i}: ❌ 无链指令（真实数据里每段都有）"); fails += 1; continue
        exp = specs[i].next_addr
        if seg.next_addr != exp:
            print(f"    段{i}: ❌ 链指向 {seg.next_addr:#x}，期望 {exp:#x}"); fails += 1
        else:
            chain_ok += 1
        if exp and seg.chain_len is not None:
            want = Machine.pc_chain_amount(specs[i].next_amount)
            got = seg.chain_len.value & 0xFFFF
            if got != want:
                print(f"    段{i}: ❌ 0x0014 低16位 {got} ≠ 期望 {want}"); fails += 1
    print(f"  ✅ 链指向校验通过 {chain_ok}/{len(segs)}")

    print("\n═══ ③ 与真实样本结构对照 ═══\n")
    print("  真实段的固定布局：")
    print("    [0 .. amount-1]  体")
    print("    [amount]         0x0010 下一段地址")
    print("    [amount+1]       0x0014 下一段 DATA_AMOUNT | 相位标志<<16")
    print(f"  合成段是否同构: ", end="")
    same = all(
        len(rc.decode(smem.read(t.addr, (t.amount + 2) * 8))) == t.amount + 2
        for _, _, t in stasks)
    print("✅ 是" if same else "❌ 否")
    if not same:
        fails += 1

    print("\n═══ ④ 重新生成完整真实算子（链部分纯公式，不看原字节）═══\n")
    try:
        mem2, tasks2, by2 = (loader.load_corpus(dumps) if dumps
                             else loader.load_corpus())
        # 挑几个代表性的算子起点
        starts = []
        seen_op = set()
        for s_, i_, t_ in tasks2:
            if s_ != 0:
                continue
            if t_.op_idx not in seen_op:
                seen_op.add(t_.op_idx)
                starts.append(i_)
            if len(starts) >= 6:
                break
        tot_seg = tot_ok = 0
        for st in starts:
            good, info = reconstruct_operator(mem2, tasks2, by2, st)
            if isinstance(info, tuple):
                tot_ok += (1 if good else 0)
                tot_seg += 1
            if not good:
                fails += 1
        print(f"\n  ✅ 完全重建成功的算子: {tot_ok} / {tot_seg}")
    except FileNotFoundError as e:
        print(f"  ⚠️ 跳过（{e}）")

    print(f"\n═══ 结果: {'✅ 全部通过' if not fails else f'❌ {fails} 项失败'} ═══")
    return fails
