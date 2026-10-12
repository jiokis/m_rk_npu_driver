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
# 明确的读/写角色（用于 producer→consumer 配对）。
# 依据：0x5018 在实测中"读 63.53MB / 写 0"，是 DPU_RDMA 把 DPU 的结果读回来喂 PPU；
#       0x5038 同理。0x5020 不是逐 tile 的数据地址（每算子一个固定值=参数表基址）。
READ_REGS = {0x1070, 0x1110, 0x5018, 0x5038}
WRITE_REGS = {0x4020, 0x5020, 0x6070, 0x701c}


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


def span_in_buffer(vals, cap=None):
    """
    同一寄存器、同一缓冲区内的地址列表 → (字节数, 说明, 可靠性)
    调用方保证 vals 已过滤到同一缓冲区。

    ★ 第二版加的硬校验（第一版在这里严重虚增过）：
      实测 0x5018 只有 39 个不同地址，却算出"读了 63.53 MB" —— 平均每 tile 1.63 MB，
      而整个缓冲区才 7.78 MB，39 个 1.63MB 的 tile 根本装不下。
      根因：**地址稀疏时"中位数步长 × tile 数"会把地址之间的空洞也算成数据。**
      所以必须用 cap（缓冲区容量）做物理校验：n_tiles × tile > cap 是不可能的，
      这种结果直接判为不可信并剔出总量，绝不静默累加。
    """
    u = sorted(set(vals))
    if len(u) == 1:
        return 0, "单地址（定不出尺寸）", False
    deltas = [u[i + 1] - u[i] for i in range(len(u) - 1)]
    dmin, dmax = min(deltas), max(deltas)
    tile = dmin
    if dmin == dmax:
        total, why = len(u) * tile, f"{len(u)}×{tile}B 等距"
    else:
        total, why = len(u) * tile, f"{len(u)}×{tile}B（步长 {tile}..{dmax} 不等，保守）"
    if cap is not None and total > cap:
        return 0, f"剔除：{len(u)}×{tile}B={total}B 超过缓冲容量 {cap}B（地址稀疏，跨度法失效）", False
    return total, why, (dmin == dmax)


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
                n_bytes, why, ok = span_in_buffer(bvals, bufs[bi][1])
                if n_bytes == 0:
                    if not ok:
                        unreliable.append((blk, reg, bi, 0, why, "剔除"))
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


def collect_occurrences(ops, bufs):
    """
    展开成 (算子序 k, 寄存器, 地址, 该地址的 tile 字节数) 的四元组流。
    tile 字节数来自"同缓冲区内相邻地址步长"，与 summary 用的是同一套算法。
    """
    occ = []
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
                n_bytes, why, ok = span_in_buffer(bvals, bufs[bi][1])
                if n_bytes == 0:
                    continue
                # span_in_buffer 返回"总数"；单 tile 大小 = 总数 / 不同地址数
                uniq = len(set(bvals))
                tile = n_bytes // uniq if uniq else 0
                if tile <= 0:
                    continue
                for v in set(bvals):
                    occ.append((k, reg, v, tile))
    return occ


def cmd_shapes(ops, bufs):
    """
    只用「确证」级字段算张量字节数 —— 用来交叉验证跨度法。

    依据：CORE 0x3014 低16 = 输出宽−1（确证）、CORE 0x3018 = 输出通道−1（确证）。
          yolov5s 的 H 与 W 相等（特征图都是方的），所以 bytes = W×W×C。
    一个算子的输出被切成多个 tile，但总和仍是 W×H×C，不用乘 tile 数。
    """
    rows = []
    for k, op in enumerate(ops):
        r = {}
        for sl in op:
            for i in sl.insns:
                r[i.reg] = i.value
        w = (r.get(0x3014, -1) & 0xFFFF) + 1 if 0x3014 in r else 0
        c = r.get(0x3018, -1) + 1 if 0x3018 in r else 0
        rows.append({"k": k, "op": op[0].task.op_idx, "tiles": len(op),
                     "w": w, "c": c, "out": w * w * c})
    # 链式：本算子的输入 = 上一个算子的输出；首个算子是 640×640×3
    prev = 640 * 640 * 3
    tot_in = tot_out = 0
    for x in rows:
        x["in"] = prev
        tot_in += prev
        tot_out += x["out"]
        if x["out"]:
            prev = x["out"]
    INPUT = 640 * 640 * 3
    print("═══ 用「确证」字段算张量字节（交叉验证）═══\n")
    print(f"  {'#':>3} {'op':>4} {'tile':>5} {'W':>5} {'C':>5} {'输出 MB':>9} {'输入 MB':>9}")
    for x in rows:
        if x["out"] == 0 and x["k"] > 6:
            continue
        print(f"  {x['k']:>3} {x['op']:>4} {x['tiles']:>5} {x['w']:>5} {x['c']:>5} "
              f"{x['out']/1048576:>9.3f} {x['in']/1048576:>9.3f}")
    print(f"\n  输入图像        {INPUT:>14,} B = {INPUT/1048576:7.2f} MB")
    print(f"  中间+最终输出   {tot_out:>14,} B = {tot_out/1048576:7.2f} MB")
    print(f"  各算子输入合计  {tot_in:>14,} B = {tot_in/1048576:7.2f} MB")
    print(f"  特征总量(读+写) {(tot_in+tot_out):>14,} B = {(tot_in+tot_out)/1048576:7.2f} MB")
    print()
    print("  这是**结构性下限**：每个中间张量被「写一次 + 读一次」计入。")
    print("  对照 yolov5s 公开的中间张量（320²×32=3.28MB, 160²×64=1.64MB, 80²×128=0.82MB,")
    print("  40²×256=0.41MB, 20²×512=0.20MB）—— 表里应能看到同样的序列。")


def cmd_fusion(ops, bufs):
    """
    把"写出去又被读回来"的往返流量逐地址算出来 —— 这是融合优化的**可省上限**。

    判据很硬：同一个地址，先被某个写寄存器写、后被某个读寄存器读。
    省掉这一趟 = 省掉一次写 + 一次读 = 该地址字节数的 2 倍。
    """
    occ = collect_occurrences(ops, bufs)
    reads = defaultdict(list)
    for (k, reg, addr, tile) in occ:
        if reg in READ_REGS:
            reads[addr].append((k, tile))

    sav_write = 0
    pairs = defaultdict(lambda: [0, 0])   # (wreg,rreg) -> [次数, 字节]
    matched_addr = set()
    for (k, reg, addr, tile) in occ:
        if reg not in WRITE_REGS:
            continue
        later = [(rk, rt) for (rk, rt) in reads.get(addr, []) if rk > k]
        if not later:
            continue
        rreg = None
        for (rk, rt) in later:
            for (k2, r2, a2, t2) in occ:
                if k2 == rk and a2 == addr and r2 in READ_REGS:
                    rreg = r2
                    break
            if rreg:
                break
        use = min([tile] + [t for _, t in later])
        sav_write += use
        matched_addr.add(addr)
        pairs[(reg, rreg or 0)][0] += 1
        pairs[(reg, rreg or 0)][1] += use

    total_occ = sum(t for (_, reg, _, t) in occ if reg in WRITE_REGS)
    grand = 143_864_320          # summary 算出的总流量（下限），仅用于算占比
    print("═══ 融合可省流量：逐地址配对「写出去 → 又被读回来」 ═══\n")
    print(f"  参与配对的写地址数      {len(matched_addr)}")
    print(f"  被读过一次的写字节数    {sav_write:>14,} B = {sav_write/1048576:7.2f} MB")
    print(f"  对应往返流量（写+读）   {2*sav_write:>14,} B = {2*sav_write/1048576:7.2f} MB")
    print()
    print(f"  占总流量（{grand/1048576:.2f} MB）的  {2*sav_write/grand*100:.1f}%")
    print("  ↑ 这就是「把这些中间结果留在片上」能省掉的 DRAM 流量上限")
    print()
    print("── 按流水级对的明细 ──")
    print(f"  {'写寄存器':>10} {'读寄存器':>10} {'配对数':>8} {'字节 MB':>10}")
    for (w, r), (n, b) in sorted(pairs.items(), key=lambda kv: -kv[1][1]):
        print(f"  0x{w:04x}     0x{r:04x} {n:>8} {b/1048576:>10.2f}")


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
                n_bytes, why, ok = span_in_buffer(bvals, bufs[bi][1])
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
    elif cmd == "shapes":
        cmd_shapes(ops, bufs)
    elif cmd == "fusion":
        cmd_fusion(ops, bufs)
    elif cmd == "tensor":
        cmd_tensor(ops, bufs)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
