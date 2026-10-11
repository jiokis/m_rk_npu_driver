#!/usr/bin/env python3
"""
pipeline.py — 层间串行度与重叠空间分析

## 要回答的问题

"最大程度利用 NPU" 的核心是：**同一时刻能不能让更多硬件级同时干活**。
硬件是四级流水（CNA / CORE / DPU / PPU + 各自的 RDMA），
如果相邻两段占用的级不冲突，原则上就能重叠。

本工具从命令流**静态算出重叠空间的上限**：
  · 每段激活了哪几级（来自 ENABLE 位图，已确证）
  · 相邻段之间是否资源冲突（冲突 = 不能重叠）
  · 相邻段之间是否有数据依赖（读的是上一段写的地址吗）
  · 配置开销占比（每段有多少条指令是"重复写同样的值"）

## 为什么这是上限而不是实测

静态分析给出的是"**允许**重叠的最大范围"。硬件**是否真的**重叠，
需要实测（计时/性能计数器），那需要硬件在环 —— 见 docs/19 的实验计划。
但先算出上限很有价值：**如果连上限都很低，就不用费劲优化调度了。**

## 用法
  python3 pipeline.py stages       # 每段用哪几级（时序视图）
  python3 pipeline.py overlap      # 相邻段的重叠空间统计
  python3 pipeline.py redundant    # 配置冗余度（重复写同值）
  python3 pipeline.py deps         # 数据依赖（地址层面的生产者→消费者）
"""
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))

from rknpu import regcmd as rc                      # noqa: E402
from rknpu import regmap as rm                      # noqa: E402
from rknpu import loader                            # noqa: E402
from rknpu.machine import Machine                   # noqa: E402
from rknpu.synth import operator_segments           # noqa: E402

# 位图 → 占用的级（Mesa/内核位定义 + 我们 615 段验证，见 docs/18）
BIT_STAGE = {0: "CNA", 2: "CORE", 3: "DPU", 4: "DPU_RDMA", 5: "PPU", 6: "PPU_RDMA"}


def stages_of(bitmap):
    return {name for bit, name in BIT_STAGE.items() if bitmap >> bit & 1}


def all_segments():
    """按执行顺序返回全部段（含使能位图、地址、指令）"""
    mem, tasks, by_addr = loader.load_corpus()
    m = Machine(mem)
    out = []
    for seq, idx, t in tasks:
        if seq != 0:
            continue
        hit = mem.find(t.addr)
        if not hit:
            continue
        avail = len(hit[1])
        ins = rc.decode(mem.read(t.addr, avail), limit=avail // 8)
        seg = m.run_segment(t)
        en = seg.enable.value if seg.enable else None
        addrs = {}
        for j in ins[:t.amount]:
            if rm.is_tensor_addr(j.block, j.reg):
                addrs[(j.block_name, j.reg)] = j.value
        out.append({"task": idx, "op": t.op_idx, "amount": t.amount,
                    "ins": ins[:t.amount], "enable": en, "addrs": addrs,
                    "stages": stages_of(en) if en is not None else set()})
    out.sort(key=lambda s: s["task"])
    return out


def cmd_stages():
    segs = all_segments()
    print(f"═══ 逐段时序视图（{len(segs)} 段）═══\n")
    print(f"  {'task':>5} {'op':>4} {'amt':>5} {'位图':>6}  占用的流水级")
    for s in segs[:40]:
        bm = f"{s['enable']:#04x}" if s["enable"] is not None else "—"
        st = " + ".join(sorted(s["stages"])) if s["stages"] else "（未使能）"
        print(f"  {s['task']:>5} {s['op']:>4} {s['amount']:>5} {bm:>6}  {st}")
    if len(segs) > 40:
        print(f"  … 其余 {len(segs)-40} 段省略")
    print(f"\n  ── 各流水级被多少段占用 ──")
    c = Counter()
    for s in segs:
        for st in s["stages"]:
            c[st] += 1
    for k, v in c.most_common():
        print(f"    {k:<10} {v:>4} 段 ({v/len(segs)*100:.0f}%)")
    return 0


def cmd_overlap():
    segs = all_segments()
    print(f"═══ 相邻段的重叠空间（{len(segs)-1} 对）═══\n")
    ok = conflict = nodata = 0
    pairs = Counter()
    for a, b in zip(segs, segs[1:]):
        if not a["stages"] or not b["stages"]:
            nodata += 1
            continue
        shared = a["stages"] & b["stages"]
        if shared:
            conflict += 1
            pairs["冲突: " + "+".join(sorted(shared))] += 1
        else:
            ok += 1
            pairs["可重叠: " + "+".join(sorted(a["stages"])) + " | "
                  + "+".join(sorted(b["stages"]))] += 1
    tot = ok + conflict
    print(f"  ✅ 资源不冲突（理论可重叠）: {ok} / {tot} = {ok/tot*100:.1f}%")
    print(f"  ❌ 有资源冲突（必须串行）  : {conflict} / {tot}")
    print(f"  无使能数据                : {nodata}")
    print(f"\n  ── 冲突/可重叠的类型分布（前 12）──")
    for k, v in pairs.most_common(12):
        print(f"    {v:>4}×  {k}")

    # 更长距离：隔一段看是否有重叠机会（乱序执行的潜力）
    print(f"\n═══ 隔一段（i 与 i+2）的重叠空间 ═══\n")
    ok2 = conf2 = 0
    for a, b in zip(segs, segs[2:]):
        if not a["stages"] or not b["stages"]:
            continue
        if a["stages"] & b["stages"]:
            conf2 += 1
        else:
            ok2 += 1
    t2 = ok2 + conf2
    if t2:
        print(f"  ✅ 不冲突: {ok2} / {t2} = {ok2/t2*100:.1f}%")
    return 0


def cmd_redundant():
    segs = all_segments()
    print("═══ 配置冗余度：段里有多少条指令是「重复写同样的值」═══\n")
    print("  （冗余高 = 发射器把每段都完整重写一遍，配置开销大；")
    print("    这在硬件上可能被优化，也可能就是真实的配置带宽成本）\n")
    tot_ins = tot_rep = 0
    for s in segs[:60]:
        seen = {}
        rep = 0
        for j in s["ins"]:
            k = (j.block, j.reg)
            if k in seen and seen[k] == j.value:
                rep += 1
            seen[k] = j.value
        tot_ins += len(s["ins"]); tot_rep += rep
        if s["task"] < 8 or rep > len(s["ins"]) * 0.5:
            print(f"    task{s['task']:<5} amt={s['amount']:<5} "
                  f"重复 {rep:>4} / {len(s['ins']):<5} ({rep/max(1,len(s['ins']))*100:.0f}%)")
    print(f"\n  前 60 段合计: {tot_rep} / {tot_ins} 条重复 "
          f"({tot_rep/max(1,tot_ins)*100:.1f}%)")

    # 跨段重复：相邻段有多少寄存器写了相同的值
    print(f"\n═══ 相邻段之间的重复（跨段冗余）═══\n")
    same = diff = 0
    for a, b in zip(segs, segs[1:]):
        da = {(j.block, j.reg): j.value for j in a["ins"]}
        db = {(j.block, j.reg): j.value for j in b["ins"]}
        for k in set(da) & set(db):
            if da[k] == db[k]:
                same += 1
            else:
                diff += 1
    print(f"  相邻段共有寄存器中：值相同 {same} · 值不同 {diff} "
          f"（相同占 {same/max(1,same+diff)*100:.0f}%）")
    print(f"  → 相同比例高 = 相邻段配置高度相似，硬件若支持差分配置可省大量带宽")
    return 0


def cmd_deps():
    segs = all_segments()
    print("═══ 数据依赖：相邻段是否读上一段写的地址 ═══\n")
    print("  （地址层面的 RAW 依赖；有依赖则不能重排，但**可以重叠**）\n")
    raw = waw = none = 0
    for a, b in zip(segs, segs[1:]):
        va = set(a["addrs"].values())
        vb = set(b["addrs"].values())
        shared = va & vb
        # 简化判定：地址集合有交集 → 认为可能存在依赖
        if shared:
            raw += 1
            # 简化：把 DPU 视作写方、CNA 视作读方
        else:
            none += 1
    tot = raw + none
    print(f"  相邻段地址有交集（可能有依赖）: {raw} / {tot} = {raw/tot*100:.1f}%")
    print(f"  地址完全不相交（独立）        : {none} / {tot} = {none/tot*100:.1f}%")
    print(f"\n  ⚠️ 这是**地址层面的粗判**：同一大缓冲区里的不同偏移会被误判为无依赖，")
    print(f"     反之同一偏移被复用也会误判为有依赖。精确判定需要按偏移区间做。")
    return 0


def main():
    c = sys.argv[1] if len(sys.argv) > 1 else "stages"
    if c == "stages":    return cmd_stages()
    if c == "overlap":   return cmd_overlap()
    if c == "redundant": return cmd_redundant()
    if c == "deps":      return cmd_deps()
    print(__doc__); return 1


if __name__ == "__main__":
    sys.exit(main())
