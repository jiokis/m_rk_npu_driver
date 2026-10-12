"""optimize.py — 命令流优化分析器（把离线实验沉淀成可复用工具）

用法：
    python3 -m src.sim.rknpu.optimize <dump_dir> [--floor-us 67.4] [--cores 3]

输入：一次 capture（regcmd 的 .bin + capture.log + timer 日志）
输出：这份命令流的优化分析报告 ——
  · 提交数 / tile 数 / 轮数
  · **纯搬运轮**数（可安全删除 ⇒ 省地板）      docs/83
  · **可删的重复写搬运**数（幂等冗余）           docs/86
  · **合并提交的参数化收益曲线**                 docs/87

设计原则（来自这几轮踩过的坑）：
  * 地板必须**逐模型给**（用全局 67.4 会让 shufflenet 算出 102%，docs/84）
  * 时间收益只按**轮**算，不按 tile 算（tile×地板 会算出比总时间还大的数，docs/86）
  * 每个数字都标注它依赖的假设（docs/82/87）
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import sys

FLOOR_DEFAULT = 67.4


def analyze(dump_dir, floor_us=FLOOR_DEFAULT, cores=3, timer_log=None):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from analysis.tiling import load_segments          # noqa
    from analysis import cost_model as C               # noqa

    segs = load_segments(dump_dir)
    log = timer_log or os.path.join(os.path.dirname(dump_dir.rstrip("/")),
                                    os.path.basename(dump_dir.rstrip("/")) + "_timer.log")
    items = sorted(C.load_timer(log).items())

    en = {(s.task.seq, s.task.index): (s.enable.value if s.enable else None) for s, d in segs}
    dst60, other_write = [], set()
    for s, d in segs:
        reg = {i.reg: i.value for i in s.insns}
        e = s.enable.value if s.enable else None
        if e == 0x60:
            if reg.get(0x6070):
                dst60.append(reg[0x6070])
        else:
            for r in (0x4020, 0x6070):
                if reg.get(r):
                    other_write.add(reg[r])

    # 逐提交：tile 数、纯搬运轮
    n_tiles = 0
    rounds = only60 = 0
    for i, v in items:
        seq, lo, n = i - 1, v["task_start"], v["task_number"]
        n_tiles += n
        nr = (n + cores - 1) // cores
        rounds += nr
        for r in range(nr):
            ts = range(lo + r * cores, min(lo + (r + 1) * cores, lo + n))
            es = [en.get((seq, t)) for t in ts]
            if es and all(e == 0x60 for e in es):
                only60 += 1

    cnt = collections.Counter(dst60)
    deletable = sum(c - 1 for a, c in cnt.items() if a not in other_write and c > 1)

    n = len(items)
    return {
        "n_submit": n, "n_tiles": n_tiles, "n_rounds": rounds,
        "tiles_per_submit": n_tiles / n if n else 0,
        "pure_copy_rounds": only60,
        "save_copy_rounds_ms": only60 * floor_us / 1000.0,      # ★ 只按轮算（docs/86）
        "deletable_copies": deletable,
        "merge_curve_ms": {p: n * p * floor_us / 1000.0 for p in (0.25, 0.50, 0.75)},
        "floor_us": floor_us, "cores": cores,
    }


def report(dump_dir, floor_us=FLOOR_DEFAULT, cores=3):
    r = analyze(dump_dir, floor_us, cores)
    name = os.path.basename(dump_dir.rstrip("/"))
    print(f"══ {name} ══")
    print(f"  提交 {r['n_submit']}   tile {r['n_tiles']}   轮 {r['n_rounds']}   "
          f"tile/提交 {r['tiles_per_submit']:.1f}")
    print(f"  地板 {r['floor_us']:.1f} µs × {r['cores']} 核")
    print(f"  ── 去冗余拷贝（docs/83/86）")
    print(f"     纯搬运轮 {r['pure_copy_rounds']} ⇒ 可省 {r['save_copy_rounds_ms']:.3f} ms"
          f"   [只按轮算：tile×地板 会算错，docs/86]")
    print(f"     可删的重复写搬运 {r['deletable_copies']} 个   [前置条件已验证，docs/86]")
    print(f"  ── 合并提交（docs/87，参数化）")
    for p, v in r["merge_curve_ms"].items():
        print(f"     提交数减 {int(p*100)}% ⇒ 省 {v:.2f} ms")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--floor-us", type=float, default=FLOOR_DEFAULT)
    ap.add_argument("--cores", type=int, default=3)
    ap.add_argument("--timer-log", default=None)
    a = ap.parse_args()
    report(a.dump_dir, a.floor_us, a.cores)
    return 0


if __name__ == "__main__":
    sys.exit(main())
