#!/usr/bin/env python3
"""
dram_model.py — 从命令流直接量出 NPU 的 DRAM 流量（不依赖任何硬件计数器）

## 为什么需要它

RK3588 的驱动**没有接线**带宽计数器（`rknpu_config.amount_top = NULL`，
`RKNPU_GET_*_RW_AMOUNT` 直接返回 0），硬件寄存器也不能盲扫
（实测盲扫会把内核打挂 —— 见 docs/29）。所以换一条路：

**每个流水级都有自己的张量地址寄存器（10 个，docs/15 已确证）。**
同一个算子里相邻 tile 的地址差值 = 那个 tile 的真实字节数 —— 这是硬证据，
不需要知道张量形状、不需要猜数据格式、不需要读计数器。

## ★ 必须先按缓冲区聚类（第一版就栽在这里）

第一版我直接对同一寄存器的所有地址取 max−min 当跨度，结果单个算子算出 36 MB 输出
—— 而整个设备只有 7 个缓冲区、最大 8.28 MB。荒谬值的原因是：
**只有 2 个 tile 时，那两个地址可能一个在 A 缓冲区、一个在 B 缓冲区**，
差值跨了半个地址空间，被我当成了 tile 步长。

所以现在：先用 MEM_CREATE 记录把地址归属到具体缓冲区，**只在同一缓冲区内算跨度**；
同缓冲区内若步长不等距，也保守取最小步长并标记为不可靠。

## 为什么结果是"下限"

地址跨度覆盖不到的部分：
  · 权重被每个 tile 重复读取
  · 层与层之间经由 DRAM 的往返（片上没有保留）
所以真实流量 ≥ 本工具算出的值；反过来说，**算出来的有效带宽是保守上界**。

## 用法
  python3 src/analysis/dram_model.py summary
  python3 src/analysis/dram_model.py ops [--top 25]
  python3 src/analysis/dram_model.py tensor
"""
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from sim.rknpu import loader, regcmd as rc, regmap as rm          # noqa: E402
from sim.rknpu.machine import Machine                             # noqa: E402

# 实测的推理耗时（docs/28：yolov5s 640×640，3 核，30 次中位，8 轮交错）
MEASURED_MS = {"ddr2400": 28.91, "ddr534": 46.00}

# 地址寄存器的角色（全部出自 regmap.TENSOR_ADDR_REGS，均为"确证"级）
ROLE = {
    0x1070: ("CNA", "读 输入特征"),
    0x1110: ("CNA", "读 权重"),
    0x4020: ("DPU", "写 输出"),
    0x5020: ("DPU_RDMA", "写/读"),
    0x5018: ("DPU_RDMA", "写/读"),
    0x5038: ("DPU_RDMA", "写/读"),
    0x6070: ("PPU", "写 输出"),
    0x701c: ("PPU_RDMA", "写/读"),
}
IS_WRITE = {0x4020, 0x5020, 0x6070, 0x701c}


def load_buffers(dumps):
    """从 capture.log 解析 MEM_CREATE：[(dma_addr, size, flags)]，按地址排序"""
    out = []
    log = os.path.join(dumps, "capture.log")
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.search(r"\[MEM_CREATE\] handle=(\d+) flags=(0x[0-9a-f]+) size=(\d+) "
                      r"obj_addr=(0x[0-9a-f]+) dma_addr=(0x[0-9a-f]+)", line)
        if m:
            out.append((int(m.group(5), 16), int(m.group(3)), int(m.group(2), 16)))
    out.sort()
    return out


def which_buffer(bufs, addr):
    for i, (dma, size, _f) in enumerate(bufs):
        if dma <= addr < dma + size:
            return i
    return None


def load_ops(dumps):
    mem, tasks, by_addr = loader.load_corpus(dumps)
    segs = Machine(mem).run_all(tasks)
    ops, cur = [], []
    for s in segs:
        if cur and s.task.op_idx != cur[-1].task.op_idx:
            ops.append(cur)
            cur = []
        cur.append(s)
    if cur:
        ops.append(cur)
    return ops


def addr_values(op):
    """{(block,reg): [每段的最后写入值]}"""
    per_seg = []
    for s in op:
        d = {}
        for i in s.insns:
            if rm.is_tensor_addr(i.block, i.reg):
                d[(i.block_name, i.reg)] = i.value
        per_seg.append(d)
    keys = sorted({k for d in per_seg for k in d})
    return {k: [d.get(k) for d in per_seg] for k in keys}


def span_in_buffer(vals):
    """
    同一寄存器、同一缓冲区内的地址列表 → (字节数, 说明, 可靠性)
    注意：调用方保证 vals 已过滤到同一缓冲区。
    """
    u = sorted(set(vals))
    if len(u) == 1:
        return 0, "单地址（定不出尺寸）", False
    deltas = [u[i + 1] - u[i] for i in range(len(u) - 1)]
    dmin, dmax = min(deltas), max(deltas)
    if dmin == dmax:
        return len(u) * dmin, f"{len(u)}×{dmin}B 等距", True
    # 步长不等距：可能是 1) tile 尺寸不一 2) 夹杂了别的缓冲区的地址
    # 保守取最小步长 × tile 数，并标记不可靠让调用方单独列出
    return len(u) * dmin, f"{len(u)}×{dmin}B（步长 {dmin}..{dmax} 不等，保守）", False


def analyse(ops, bufs):
    rows = []
    for k, op in enumerate(ops):
        rd = wr = 0
        detail, unreliable = [], []
        for (blk, reg), vals in addr_values(op).items():
            if reg not in ROLE:
                continue
            # ★ 关键：按缓冲区聚类
            buckets = defaultdict(list)
            outside = 0
            for v in vals:
                if not v:            # 地址 0 = 空槽
                    continue
                bi = which_buffer(bufs, v)
                if bi is None:
                    outside += 1
                    continue
                buckets[bi].append(v)
            for bi, bvals in buckets.items():
                n_bytes, why, ok = span_in_buffer(bvals)
                if n_bytes == 0:
                    continue
                who = "写" if reg in IS_WRITE else "读"
                if who == "写":
                    wr += n_bytes
                else:
                    rd += n_bytes
                rec = (blk, reg, bi, n_bytes, why, who)
                (detail if ok else unreliable).append(rec)
            if outside:
                unreliable.append((blk, reg, -1, 0, f"{outside} 个地址不在任何缓冲区", "?"))
        rows.append({"k": k, "op": op[0].task.op_idx, "task": op[0].task.index,
                     "tiles": len(op), "read": rd, "write": wr,
                     "detail": detail, "bad": unreliable})
    return rows


def bw(total_bytes, ms):
    return total_bytes / (ms / 1000.0) / 1e9


def cmd_summary(ops, bufs):
    rows = analyse(ops, bufs)
    R = sum(r["read"] for r in rows)
    W = sum(r["write"] for r in rows)
    nb = sum(1 for r in rows for _ in r["bad"])
    print("═══ yolov5s 单次推理的 DRAM 流量（由命令流地址实测）═══\n")
    print(f"  缓冲区            {len(bufs)} 个，合计 {sum(s for _, s, _ in bufs)/1048576:.1f} MB")
    print(f"  算子数            {len(rows)}")
    print(f"  总段数            {sum(r['tiles'] for r in rows)}")
    print(f"  读（输入+权重）   {R:>14,} B = {R/1048576:8.2f} MB")
    print(f"  写（输出）        {W:>14,} B = {W/1048576:8.2f} MB")
    print(f"  合计（下限）      {R+W:>14,} B = {(R+W)/1048576:8.2f} MB")
    print(f"  不可靠的测量项    {nb} 项（步长不等距 / 地址在缓冲区外，已按保守值计入）")
    print()
    print("── 用实测耗时反推有效带宽 ──")
    for tag, ms in MEASURED_MS.items():
        print(f"  {tag:>8} ({ms:>5.2f} ms): {bw(R+W, ms):6.2f} GB/s")
    print()
    print("  这是一个**保守上界**（真实流量只会更大，所以真实带宽需求更高）。")
    print("  对照：RK3588 LPDDR4x/5 @2400MHz 理论峰值约 30~38 GB/s，实测可达通常 50~70%。")


def cmd_ops(ops, bufs, top=25):
    rows = sorted(analyse(ops, bufs), key=lambda r: -(r["read"] + r["write"]))
    tot = sum(r["read"] + r["write"] for r in rows) or 1
    print(f"═══ 按 DRAM 流量排序的算子（前 {top}）═══\n")
    print(f"  {'#':>3} {'op':>4} {'task':>5} {'段':>4} {'读MB':>7} {'写MB':>7} "
          f"{'合计MB':>7} {'占比':>6}  明细（流水级+寄存器+缓冲区）")
    c = 0
    for r in rows[:top]:
        t = r["read"] + r["write"]
        c += t
        det = "  ".join(f"{b}+0x{reg:04x}#buf{bi}:{nb/1024:.0f}K({why})"
                        for b, reg, bi, nb, why, w in r["detail"][:2])
        print(f"  {r['k']:>3} {r['op']:>4} {r['task']:>5} {r['tiles']:>4} "
              f"{r['read']/1048576:>7.2f} {r['write']/1048576:>7.2f} "
              f"{t/1048576:>7.2f} {t/tot*100:>5.1f}%  {det}")
    print(f"\n  前 {top} 个算子占全部流量的 {c/tot*100:.1f}%")
    print("  这些就是融合优化的目标：把它们的中间结果留在片上，省掉「写回 DRAM + 再读回」")


def cmd_tensor(ops, bufs):
    use = defaultdict(lambda: {"r": 0, "w": 0, "ops": set(), "n": 0})
    for k, op in enumerate(ops):
        for (blk, reg), vals in addr_values(op).items():
            if reg not in ROLE:
                continue
            buckets = defaultdict(list)
            for v in vals:
                if not v:
                    continue
                bi = which_buffer(bufs, v)
                if bi is not None:
                    buckets[bi].append(v)
            for bi, bvals in buckets.items():
                n_bytes, why, ok = span_in_buffer(bvals)
                if n_bytes == 0:
                    continue
                key = (ROLE[reg][0], reg, bi)
                use[key]["w" if reg in IS_WRITE else "r"] += n_bytes
                use[key]["ops"].add(k)
                use[key]["n"] += len(set(bvals))
    print("═══ 各流水级 / 缓冲区的总流量 ═══\n")
    print(f"  {'流水级':>10} {'寄存器':>8} {'缓冲':>4} {'大小MB':>8} {'读MB':>9} {'写MB':>9} {'段数':>6} {'算子':>5}")
    for (blk, reg, bi), v in sorted(use.items(), key=lambda kv: -(kv[1]["r"] + kv[1]["w"])):
        dma, size, _f = bufs[bi]
        print(f"  {blk:>10} 0x{reg:04x} {bi:>4} {size/1048576:>8.2f} "
              f"{v['r']/1048576:>9.2f} {v['w']/1048576:>9.2f} {v['n']:>6} {len(v['ops']):>5}")


def main():
    dumps = os.environ.get("RKNPU_DUMPS", "data/dumps2")
    if not os.path.exists(os.path.join(dumps, "capture.log")):
        dumps = os.path.join(HERE, "..", "..", "data", "dumps2")
    ops = load_ops(dumps)
    bufs = load_buffers(dumps)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "summary":
        cmd_summary(ops, bufs)
    elif cmd == "ops":
        top = 25
        if "--top" in sys.argv:
            top = int(sys.argv[sys.argv.index("--top") + 1])
        cmd_ops(ops, bufs, top)
    elif cmd == "tensor":
        cmd_tensor(ops, bufs)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
