#!/usr/bin/env python3
"""driver_trace.py — 解析板端 ftrace 抓到的 rknpu 驱动调用序列

## 输入怎么来

板端（**注意顺序，踩过坑**）：

    T=/sys/kernel/tracing
    echo 0 > $T/tracing_on
    echo nop > $T/current_tracer
    echo > $T/trace
    echo > $T/set_graph_function
    echo > $T/set_ftrace_filter
    echo 65536 > $T/buffer_size_kb
    echo "rknpu_*" > $T/set_ftrace_filter
    echo 1 > $T/options/func_stack_trace      # 每次调用附带调用栈（体积 ×3，但能看调用者）
    echo function > $T/current_tracer
    echo 1 > $T/tracing_on
    ... 触发一次推理 ...
    echo 0 > $T/tracing_on
    cp $T/trace /path/out.txt                 # ★★ 先落盘！
    echo nop > $T/current_tracer              # ★★ 后关 tracer
                                              #    先关会把环形缓冲区清空 ⇒ 抓到 0 条（踩过）

## 这条路的边界（实测）

- `function` tracer 在这个内核上**正常**；`function_graph` + `set_graph_function='rknpu_*'`
  也能跑，但深度不限时一次推理产出 **1.5 GB**，不可用。
- 结论：**用扁平 function + 过滤器 + func_stack_trace**，体积 8 MB 级，信息最全。

## 用法
  python3 src/analysis/driver_trace.py trace.txt            # 结构概览
  python3 src/analysis/driver_trace.py trace.txt --gaps     # 空闲间隙分解
  python3 src/analysis/driver_trace.py trace.txt --cores    # 每 submit 的核数模式
  python3 src/analysis/driver_trace.py trace.txt --near 8.7 --near 13.1   # 某窗口内逐事件
"""
from __future__ import annotations

import argparse
import bisect
import collections
import re
import statistics as st

LINE = re.compile(r'^\s*(\S+)-(\d+)\s+\[(\d+)\]\s+\S+\s+([\d.]+):\s+(\S+)(?: <-(\S+))?')


def load(path):
    ev = []
    with open(path, errors="replace") as f:
        for ln in f:
            if ln.startswith("#") or "=>" in ln:
                continue
            m = LINE.match(ln)
            if not m:
                continue
            task, pid, cpu, ts, fn, caller = m.groups()
            ev.append((float(ts), fn, caller, task, cpu))
    ev.sort()
    return ev


def submits(ev):
    return [t for t, f, c, *_ in ev if f == "rknpu_submit_ioctl" and c == "__rknpu_submit_ioctl"]


def cmd_overview(ev):
    t0 = ev[0][0]
    sub = submits(ev)
    span = sub[-1] - t0
    cnt = collections.Counter(f for _, f, *_ in ev if f != "<stack")
    print(f"事件 {len(ev)} 条 / 提交 {len(sub)} 次 / 跨度 {span:.2f} s / "
          f"平均 {len(sub)/span:.1f} 提交·秒⁻¹\n")
    print(f"  {'调用次数':>8}  驱动函数")
    for f, c in cnt.most_common():
        print(f"  {c:>8}  {f}")

    gaps = [(a, b) for a, b in zip(sub, sub[1:]) if b - a > 0.2]
    idle = sum(b - a for a, b in gaps)
    print(f"\n  NPU 无提交的间隙: {len(gaps)} 段 / 合计 {idle:.2f} s / "
          f"占全程 {idle/span*100:.1f}%")
    for a, b in sorted(gaps, key=lambda x: x[0] - x[1])[:5]:
        print(f"    {a-t0:>7.2f}s → {b-t0:>7.2f}s   空 {b-a:>6.3f}s")

    c0 = [t for t, f, *_ in ev if f == "rknpu_core0_irq_handler"]
    c1 = [t for t, f, *_ in ev if f == "rknpu_core1_irq_handler"]
    c2 = [t for t, f, *_ in ev if f == "rknpu_core2_irq_handler"]
    print(f"\n  IRQ: core0={len(c0)} core1={len(c1)} core2={len(c2)}   submit={len(sub)}")
    print(f"  ⇒ core0 : submit = {len(c0)/len(sub):.3f}（1.0 = 每个提交恰好一次完成中断）")

    # 每 submit 的核数模式
    dist = collections.Counter()
    for i, s in enumerate(sub):
        e = sub[i + 1] if i + 1 < len(sub) else s + 1e9
        n = tuple(bisect.bisect_left(x, e) - bisect.bisect_left(x, s) for x in (c0, c1, c2))
        dist[n] += 1
    print("\n  每个 submit 之后的各核 IRQ 次数：")
    for k, v in dist.most_common(5):
        print(f"    core0={k[0]} core1={k[1]} core2={k[2]}  →  {v:>4} 次 "
              f"({v/len(sub)*100:.1f}%)")


def cmd_gaps(ev):
    t0 = ev[0][0]
    sub = submits(ev)
    print("空闲间隙内部发生了什么（每个间隙里出现的驱动事件按函数计数）：")
    for a, b in zip(sub, sub[1:]):
        if b - a <= 0.2:
            continue
        inside = [(t - t0, f) for t, f, *_ in ev if a < t < b and f != "<stack"]
        c = collections.Counter(f for _, f in inside)
        print(f"\n  ── {a-t0:.2f}s → {b-t0:.2f}s  空 {b-a:.3f}s，窗口内 {len(inside)} 个驱动事件")
        if not inside:
            print("     （窗口内没有任何驱动调用 ⇒ 纯 CPU 侧工作）")
            continue
        for f, n in c.most_common(6):
            print(f"       {n:>3}  {f}")
        # 窗口内最长的静默段
        ts = [t for t, _ in inside]
        best = (0, 0, 0)
        for x, y in zip(ts, ts[1:]):
            if y - x > best[0]:
                best = (y - x, x, y)
        if best[0] > 0.05:
            print(f"     ★ 窗口内最长静默 {best[0]:.3f}s：{best[1]:.2f}s → {best[2]:.2f}s"
                  f"（无任何驱动调用 = 纯 CPU/IO）")


def cmd_cores(ev):
    t0 = ev[0][0]
    sub = submits(ev)
    c1 = [t for t, f, *_ in ev if f == "rknpu_core1_irq_handler"]
    c2 = [t for t, f, *_ in ev if f == "rknpu_core2_irq_handler"]
    seq = []
    for i, s in enumerate(sub):
        e = sub[i + 1] if i + 1 < len(sub) else s + 1e9
        three = (bisect.bisect_left(c1, e) > bisect.bisect_left(c1, s) and
                 bisect.bisect_left(c2, e) > bisect.bisect_left(c2, s))
        seq.append((s - t0, 3 if three else 1))
    print("时间 × 核数（每 0.5 s 窗口）：")
    b = 0.0
    end = seq[-1][0]
    while b < end:
        one = sum(1 for t, m in seq if b <= t < b + 0.5 and m == 1)
        thr = sum(1 for t, m in seq if b <= t < b + 0.5 and m == 3)
        if one or thr:
            print(f"  {b:>6.1f}s  单核 {one:>4}  三核 {thr:>4}  "
                  f"三核占比 {thr/(one+thr)*100:>5.1f}%")
        b += 0.5

    # 周期性：用大间隙切周期
    tail = [t for t, m in seq]
    cyc = [[tail[0]]]
    for a, b2 in zip(tail, tail[1:]):
        if b2 - a > 0.3:
            cyc.append([b2])
        else:
            cyc[-1].append(b2)
    lens = [len(g) for g in cyc]
    print(f"\n  按 >300 ms 切周期：{len(cyc)} 个周期，每周期提交数众数 "
          f"{collections.Counter(lens).most_common(1)[0]}")


def cmd_near(ev, lo, hi):
    t0 = ev[0][0]
    for t, f, c, task, cpu in ev:
        x = t - t0
        if lo <= x <= hi and f != "<stack":
            print(f"  {x:>8.3f}s  {f:<40} <- {c}   [{task}/{cpu}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--gaps", action="store_true")
    ap.add_argument("--cores", action="store_true")
    ap.add_argument("--near", nargs=2, type=float, metavar=("LO", "HI"))
    a = ap.parse_args()
    ev = load(a.path)
    if not ev:
        raise SystemExit("没解析到任何事件——先确认抓包时是【先 cp 后 nop】")
    if a.near:
        cmd_near(ev, *a.near)
        return
    cmd_overview(ev)
    if a.gaps:
        cmd_gaps(ev)
    if a.cores:
        cmd_cores(ev)


if __name__ == "__main__":
    main()
