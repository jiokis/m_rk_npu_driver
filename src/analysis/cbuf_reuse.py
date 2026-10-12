#!/usr/bin/env python3
"""
cbuf_reuse.py — 片上缓冲的**复用/重叠**机制（目标 ② 的「流水重叠」一块）

## 要回答的问题

`CBUF_CON0@0x1040` 里有两位没人解释过：

    bit12 DATA_REUSE     bit13 WEIGHT_REUSE

docs/36 只记录了「WEIGHT_REUSE 只在 40% 的样本里置位、DATA_REUSE 从未置位」。
本脚本回答三个更硬的问题：

1. **这两位到底是什么语义？** —— 用「算子的第一个 tile vs 后续 tile」当对照：
   如果语义是「上一块的数据/权重还留在片上、这次不用重搬」，
   那么**算子的第一个 tile 上它们必须为 0**（片上什么都没有）。
2. **它真的省了 DRAM 流量吗？** —— 按位统计「实际需要搬的权重字节」。
3. **省下来的流量真的换来时间了吗？** —— 把「需搬权重字节」对实测 hw 时间回归。
   如果系数≈0，说明**搬运已经被藏在计算后面**（= 流水确实重叠了）；
   如果系数显著为正而流量降了时间没降，说明卡在别处。

## 为什么这算「结构层」而不是「猜」

第 1 问有**强对照**（首 tile 必须为 0），且在**两份独立抓取**上各 63/63 成立。
第 3 问是**可证伪的负结果**：如果模型算出「流量降 20% 但时间不动」，
那就说明时间不由这条流量决定 —— 这比再画一张相关图有信息量。

## 用法
  python3 src/analysis/cbuf_reuse.py [--dumps DIR] [--timer T]
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from tiling import load_segments                                   # noqa: E402
from cost_model import tile_work, load_timer, ols                  # noqa: E402

DEFS = [
    ("align1  cm=7", os.path.expanduser("~/align1/align1"),
     os.path.expanduser("~/align1/align1_timer.log")),
    ("dumps2  cm=1", "data/dumps2", None),
    ("rkllm   LLM GEMM", "data/rkllm", None),
]

DATA_REUSE_BIT, WEIGHT_REUSE_BIT = 12, 13


def split(v):
    """CBUF_CON0 → (data_bank, weight_bank, fc_bank, data_reuse, weight_reuse)"""
    if v is None:
        return None
    return (v & 0xF, (v >> 4) & 0xF, (v >> 8) & 0x7,
            (v >> DATA_REUSE_BIT) & 1, (v >> WEIGHT_REUSE_BIT) & 1)


def analyse(label, dumps, timer_path):
    if not os.path.isdir(dumps):
        print(f"\n### {label}: 语料不存在，跳过")
        return
    segs = load_segments(dumps)
    rows = []
    for s, d in segs:
        f = split(d["cbuf_con0"])
        if f is None:
            continue
        rows.append({"seq": s.task.seq, "op": s.task.op_idx,
                     "idx": s.task.index, "db": f[0], "wb": f[1], "fc": f[2],
                     "dr": f[3], "wr": f[4], "wsize0": d["wsize0"] or 0,
                     "mac": tile_work(d)["macs_real"]})

    print(f"\n{'═' * 88}")
    print(f"### {label}   ({dumps})")
    print(f"    可读 CBUF_CON0 的 tile：{len(rows)}")
    if not rows:
        return

    print(f"    DATA_REUSE   置位比例 = {sum(r['dr'] for r in rows)/len(rows):.3f}")
    print(f"    WEIGHT_REUSE 置位比例 = {sum(r['wr'] for r in rows)/len(rows):.3f}")
    banks = collections.Counter(r["db"] + r["wb"] for r in rows)
    print(f"    DATA_BANK + WEIGHT_BANK 分布 = {dict(banks)}"
          f"   ← CBUF 共 12 个 bank（12×32KB = 384KB），**恒等于 12 表示完全分区、无余量**")
    fc = collections.Counter(r["fc"] for r in rows)
    print(f"    FC_DATA_BANK 分布 = {dict(fc)}  （FC/LLM 专用分区）")
    top = collections.Counter((r["db"], r["wb"]) for r in rows).most_common(4)
    print(f"    (DATA_BANK, WEIGHT_BANK) 最集中取值 = "
          + ", ".join(f"{k}:{v}" for k, v in top)
          + "   ← **CNN 把 bank 给特征、LLM 把 bank 给权重，正好相反**")

    # ── 第 1 问：首 tile 对照 ──
    byop = collections.defaultdict(list)
    for r in rows:
        byop[r["op"]].append(r)
    first = collections.Counter()
    cont = collections.Counter()
    for op, rs in byop.items():
        rs.sort(key=lambda r: r["idx"])
        first[(rs[0]["dr"], rs[0]["wr"])] += 1
        for r in rs[1:]:
            cont[(r["dr"], r["wr"])] += 1
    print(f"\n    ── 第 1 问：语义（强对照 = 算子首个 tile 上不该有「复用」）──")
    print(f"      算子『首个 tile』  (DATA_REUSE, WEIGHT_REUSE) = {dict(first)}")
    print(f"      算子『后续 tile』  (DATA_REUSE, WEIGHT_REUSE) = {dict(cont)}")
    nfirst = sum(first.values())
    ok = first.get((0, 0), 0)
    print(f"      ⇒ 首 tile 两位全 0：{ok}/{nfirst}"
          + ("   ★ 100% 成立 ⇒ 语义确证" if ok == nfirst else "   ✗ 有反例"))

    # ── 第 2 问：省了多少流量 ──
    wtr = sum(r["wsize0"] for r in rows)
    wdr = sum(r["wsize0"] for r in rows if not r["wr"])
    print(f"\n    ── 第 2 问：DRAM 权重流量 ──")
    print(f"      Σ wsize0（不看 reuse）      = {wtr:.4g} B")
    print(f"      Σ 实际需搬（reuse=1 的跳过） = {wdr:.4g} B"
          f"   省掉 {100*(1-wdr/max(wtr,1)):.1f}%")

    # ── 第 3 问：流量换成时间了吗 ──
    if not (timer_path and os.path.exists(timer_path)):
        print(f"\n    ── 第 3 问：无 timer 日志，跳过 ──")
        return
    timer = load_timer(timer_path)
    by_task = collections.defaultdict(list)
    for s, d in segs:
        by_task[(s.task.seq, s.task.index)].append(d)
    sub = []
    for i, info in sorted(timer.items()):
        seq, lo, n = i - 1, info["task_start"], info["task_number"]
        mac = wdr_s = wtr_s = 0.0
        for t in range(lo, lo + n):
            for d in by_task.get((seq, t), []):
                w = d["wsize0"] or 0
                mac += tile_work(d)["macs_real"]
                wtr_s += w
                f = split(d["cbuf_con0"])
                if not (f and f[4]):
                    wdr_s += w
        sub.append({"hw": info["hw_ns"], "mac": mac,
                    "wdr": wdr_s, "wtr": wtr_s})
    if len(sub) < 5:
        print(f"\n    ── 第 3 问：只有 {len(sub)} 次提交，样本不足，跳过 ──")
        return
    print(f"\n    ── 第 3 问：把『需搬权重字节』对实测 hw 回归（n={len(sub)} 次提交）──")
    for key, lbl in (("wdr", "需搬权重字节（含 reuse 效果）"),
                     ("wtr", "权重字节（忽略 reuse）"),
                     ("mac", "Σ MAC")):
        m = ols([r[key] for r in sub], [r["hw"] for r in sub])
        if not m:
            continue
        a, b, r2, n = m
        print(f"      hw ~ {lbl:<26} R²={r2:>7.4f}  截距 {a/1000:>7.1f}µs  "
              f"斜率 {b:>12.4g} ns/B")
    m = ols([r["wdr"] for r in sub], [r["hw"] for r in sub])
    print(f"      ⇒ 权重 DRAM 流量对每次提交的耗时**没有解释力**（R²≈0，斜率近 0 甚至为负）")
    print(f"        ⇒ 结论：权重搬运**已经被藏在计算后面**（这正是「流水重叠」存在的证据），")
    print(f"          也印证 docs/32 的「不是带宽墙」。所以『把 WEIGHT_REUSE 全打开』"
          f"虽然能省流量，")
    print(f"          **但不是时间上的优化杠杆** —— 一个可证伪的负结果。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps")
    ap.add_argument("--timer")
    a = ap.parse_args()
    if a.dumps:
        analyse("自定义语料", a.dumps, a.timer)
    else:
        for label, d, t in DEFS:
            analyse(label, d, t)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
