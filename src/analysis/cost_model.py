#!/usr/bin/env python3
"""
cost_model.py — 指令流水的**结构级耗时模型**：从算子表预测耗时

## 目标（目标 ③）

输入：一个模型的 regcmd 命令流（= 算子表）。
输出：**预测耗时**，以及每条预测的置信区间。

这条路能走通，靠的是上一步 `tiling.py` 撞出来的结论：
上游 Mesa 的分块模型对，所以每个 tile 的
**输出宽/高/通道、输入通道、卷积核尺寸** 都能从寄存器直接读出
（15 项字段级预测全中，见 tiling.py verify）。

⇒ 每个 tile 的 **MAC 数** 是可算的。剩下只要问两件事：

1. **硬件吞吐是多少**（MAC/s）？—— 回归拟合
2. **每次提交的固定地板是多少**？以及模型形式是
   `t = a + b·W`（每提交一个固定开销）
   还是 `t = max(floor, b·W)`（固定地板，超过后线性）？

第 2 问是 docs/39→docs/40 留下的悬案。本脚本用**同一份语料里同时变化
的 submits 数与工作量**去判：把 hw 时间对「该 submit 的总工作量」回归，
看残差结构 —— 若在 W→0 时截距为正常数则为 `a + bW`；若小 W 处时间被
钳在一个常数上、且该常数随 submit 数不变（只随 submit 个数出现），
则为 `max(f, bW)`。

## 工作量候选特征（谁是真凶，让数据说）

  macs_real    = Σ out_w·out_h·真实输入通道·kh·kw·真实输出通道（真实算法量）
  macs_align   = Σ out_w·out_h·对齐输入通道·kh·kw·对齐输出通道（硬件实际算的量）
  grain_macs   = Σ grains·out_w·对齐输入通道·kh·kw      （若 FEATURE_GRAINS 是行粒度）
  out_bytes    = Σ out_w·out_h·对齐输出通道
  in_bytes     = Σ in_w·in_h·对齐输入通道
  weight_bytes = Σ WEIGHT_SIZE0

## 用法
  python3 src/analysis/cost_model.py fit   [--dumps D] [--timer T]
  python3 src/analysis/cost_model.py table [--dumps D] [--timer T]
  python3 src/analysis/cost_model.py resid [--dumps D] [--timer T]
"""
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from tiling import load_segments                                   # noqa: E402

DEF_DUMPS = os.path.expanduser("~/align1/align1")
DEF_TIMER = os.path.expanduser("~/align1/align1_timer.log")


# ───────────────────── 每 tile 的工作量 ─────────────────────
def tile_work(d):
    """返回该 tile 的各候选工作量特征"""
    W, H = d["out_w"] or 0, d["out_h"] or 0
    kw, kh = d["kw"] or 0, d["kh"] or 0
    cra, cal = d["ch_real"] or 0, d["ch_align"] or 0
    krn = d["kernels"] or 0
    out_real = d["cube_orig"] or 0          # 真实输出通道
    out_al = d["core_c"] or 0               # 对齐输出通道
    g = d["grains"] or 0
    px = W * H
    # ★ 深度卷积：WEIGHT_KERNELS==1 时每个输出通道只吃自己那一个输入通道，
    #   不能再乘 out_ch（否则单 tile 会算出 5.6e8 的假 MAC，把回归彻底带偏）。
    depthwise = (krn == 1 and out_real > 1)
    if depthwise:
        macs_real = px * kh * kw * out_real
        macs_align = px * kh * kw * cal
    else:
        macs_real = px * cra * kh * kw * out_real
        macs_align = px * cal * kh * kw * krn
    return {
        "macs_real": macs_real,
        "macs_align": macs_align,
        "depthwise": 1 if depthwise else 0,
        "grain_macs": g * W * cal * max(kh, 1) * max(kw, 1),
        "depthwise_tiles": 1 if depthwise else 0,
        "out_bytes": px * out_al,
        "in_bytes": (d["in_w"] or 0) * (d["in_h"] or 0) * cal,
        "weight_bytes": d["wsize0"] or 0,
        "atomics": d["atomics"] or 0,
        "tiles": 1,
    }


FEATS = ["macs_real", "macs_align", "grain_macs", "out_bytes", "in_bytes",
         "weight_bytes", "atomics", "tiles"]
# 这些不进回归（只是标记）
MARK = ["depthwise", "depthwise_tiles"]


# ───────────────────── 读 timer 日志 ─────────────────────
def load_timer(path):
    """→ {序号: {task_start, task_number, core_mask, wall_ns, hw_ns}}"""
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 7 or not p[0].isdigit():
            continue
        i = int(p[0])
        out[i] = {
            "task_start": int(p[1]), "task_number": int(p[2]),
            "core_mask": int(p[3], 16),
            "wall_ns": int(p[4]), "hw_ns": int(p[5]),
        }
    return out


def build(dumps, timer_path):
    """→ [(submit_idx, work_sum_dict, hw_ns, n_tasks, task_start)]"""
    segs = load_segments(dumps)
    by_task = defaultdict(list)
    for s, d in segs:
        by_task[(s.task.seq, s.task.index)].append(d)
    timer = load_timer(timer_path)

    # 计数器：本语料里 seq 是 0-based 提交序
    seqs = sorted({k[0] for k in by_task})
    rows = []
    for i, info in sorted(timer.items()):
        seq = i - 1
        lo, hi = info["task_start"], info["task_start"] + info["task_number"]
        w = {f: 0 for f in FEATS + MARK}
        ntile = 0
        ntask_seen = 0
        for tidx in range(lo, hi):
            ds = by_task.get((seq, tidx))
            if not ds:
                continue
            ntask_seen += 1
            for d in ds:
                tw = tile_work(d)
                for f in FEATS + MARK:
                    w[f] += tw[f]
                ntile += 1
        rows.append({
            "submit": i, "seq": seq, "task_start": lo,
            "n_tasks": info["task_number"], "n_tasks_dumped": ntask_seen,
            "n_tiles": ntile, "core_mask": info["core_mask"],
            "hw_ns": info["hw_ns"], "wall_ns": info["wall_ns"], "work": w,
        })
    return rows


# ───────────────────── 回归 ─────────────────────
def ols(xs, ys):
    """返回 (a, b, r2, n) —— 最小二乘 y = a + b·x"""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0
    return a, b, r2, n


def cmd_fit(dumps, timer_path, full_only=True):
    rows = build(dumps, timer_path)
    ok = [r for r in rows if r["n_tasks_dumped"] == r["n_tasks"] and r["n_tiles"]]
    print(f"提交总数 {len(rows)}，dump 完整且非空 {len(ok)}")
    if not full_only:
        ok = [r for r in rows if r["n_tiles"]]
    print(f"参与回归 {len(ok)} 条提交\n")

    # 单变量
    print(f"{'特征':<14} {'截距 a(µs)':>12} {'斜率 b':>16} {'R²':>8}  "
          f"{'隐含吞吐':>22}")
    print("─" * 88)
    best = None
    for f in FEATS:
        xs = [r["work"][f] for r in ok]
        ys = [r["hw_ns"] for r in ok]
        m = ols(xs, ys)
        if not m:
            continue
        a, b, r2, n = m
        thr = ""
        if f.startswith("macs"):
            thr = f"{1e9/b/1e12:.2f} T-MAC/s" if b > 0 else "-"
        elif f in ("out_bytes", "in_bytes", "weight_bytes"):
            thr = f"{1e9/b/1e9:.1f} GB/s" if b > 0 else "-"
        elif f == "atomics":
            thr = f"{1e9/b/1e9:.2f} G-atomic/s" if b > 0 else "-"
        elif f == "tiles":
            thr = f"{b/1000:.1f} µs/tile"
        print(f"{f:<14} {a/1000:>12.1f} {b:>16.4g} {r2:>8.4f}  {thr:>22}")
        if best is None or r2 > best[2]:
            best = (f, a, b, r2)
    print()

    # 模型形式 A vs B：用「小工作量」子集看截距是否随工作量下降
    f = best[0]
    xs = [r["work"][f] for r in ok]
    ys = [r["hw_ns"] for r in ok]
    order = sorted(range(len(ok)), key=lambda i: xs[i])
    print(f"最优特征 = {f}；按工作量五等分，看『每提交时间 − b·W』是否恒定")
    print(f"{'分位':<6} {'W 范围':>26} {'n':>4} {'均值 hw(µs)':>12} "
          f"{'b·W(µs)':>10} {'残差(µs)':>10} {'残差/W':>10}")
    print("─" * 88)
    a, b, r2 = best[1], best[2], best[3]
    k = len(order) // 5
    for q in range(5):
        idx = order[q * k:(q + 1) * k] if q < 4 else order[4 * k:]
        if not idx:
            continue
        wm = sum(xs[i] for i in idx) / len(idx)
        hm = sum(ys[i] for i in idx) / len(idx)
        print(f"Q{q+1:<5} {xs[idx[0]]:>12.3g} … {xs[idx[-1]]:>12.3g} "
              f"{len(idx):>4} {hm/1000:>12.1f} {b*wm/1000:>10.1f} "
              f"{(hm-b*wm)/1000:>10.1f} {((hm-b*wm)/max(wm,1)):>10.3g}")
    print()
    print(f"  模型 A  t = {a/1000:.1f}µs + {b:.4g}·W")
    print(f"  模型 B  t = max(f, {b:.4g}·W)   —— 若 Q1 残差显著小于 Q5 残差则 B 更真")
    print()


def cmd_resid(dumps, timer_path):
    rows = build(dumps, timer_path)
    ok = [r for r in rows if r["n_tasks_dumped"] == r["n_tasks"] and r["n_tiles"]]
    for f in ("macs_align", "atomics", "out_bytes"):
        xs = [r["work"][f] for r in ok]
        ys = [r["hw_ns"] for r in ok]
        m = ols(xs, ys)
        if not m:
            continue
        a, b, r2, n = m
        print(f"\n=== {f}:  t = {a/1000:.1f}µs + {b:.4g}·W , R²={r2:.4f} ===")
        print(f"{'submit':>7} {'ntask':>6} {'W':>14} {'hw(µs)':>10} "
              f"{'pred(µs)':>10} {'残差(µs)':>10}")
        for r in sorted(ok, key=lambda r: r["work"][f])[:12]:
            w = r["work"][f]
            pred = a + b * w
            print(f"{r['submit']:>7} {r['n_tasks']:>6} {w:>14.4g} "
                  f"{r['hw_ns']/1000:>10.1f} {pred/1000:>10.1f} "
                  f"{(r['hw_ns']-pred)/1000:>10.1f}")


def cmd_table(dumps, timer_path):
    """每个 op 的工作量/耗时表 —— 结构级画像"""
    segs = load_segments(dumps)
    rows = build(dumps, timer_path)
    timer = load_timer(timer_path)
    byop = defaultdict(lambda: {f: 0 for f in FEATS} | {"tiles": 0})
    for s, d in segs:
        w = tile_work(d)
        tgt = byop[s.task.op_idx]
        for f in FEATS:
            tgt[f] += w[f]
    print(f"{'op':>4} {'tiles':>6} {'MAC(真实)':>14} {'MAC(对齐)':>14} "
          f"{'输出字节':>13} {'权重字节':>12}")
    print("─" * 78)
    tot = {f: 0 for f in FEATS}
    for op in sorted(byop):
        b = byop[op]
        for f in FEATS:
            tot[f] += b[f]
        print(f"{op:>4} {b['tiles']:>6} {b['macs_real']:>14.4g} "
              f"{b['macs_align']:>14.4g} {b['out_bytes']:>13.4g} "
              f"{b['weight_bytes']:>12.4g}")
    print("─" * 78)
    print(f"{'Σ':>4} {tot['tiles']:>6} {tot['macs_real']:>14.4g} "
          f"{tot['macs_align']:>14.4g} {tot['out_bytes']:>13.4g} "
          f"{tot['weight_bytes']:>12.4g}")
    tot_hw = sum(r["hw_ns"] for r in rows)
    print(f"\nΣ hw = {tot_hw/1e6:.2f} ms  ({(tot_hw/1000)/max(tot['tiles'],1):.1f} µs/tile)")
    print(f"真实 MAC 总量 = {tot['macs_real']/1e9:.3f} G-MAC")
    print(f"⇒ 端到端等效吞吐 = {tot['macs_real']/ (tot_hw/1e9) / 1e12:.3f} T-MAC/s")
    print(f"   若按对齐算     = {tot['macs_align']/ (tot_hw/1e9) / 1e12:.3f} T-MAC/s")


def main(argv):
    dumps, timer = DEF_DUMPS, DEF_TIMER
    if "--dumps" in argv:
        dumps = argv[argv.index("--dumps") + 1]
    if "--timer" in argv:
        timer = argv[argv.index("--timer") + 1]
    cmd = argv[1] if len(argv) > 1 else "fit"
    if cmd == "fit":
        cmd_fit(dumps, timer)
    elif cmd == "resid":
        cmd_resid(dumps, timer)
    elif cmd == "table":
        cmd_table(dumps, timer)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
