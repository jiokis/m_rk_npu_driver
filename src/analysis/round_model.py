#!/usr/bin/env python3
"""
round_model.py — 多核为什么只有 1.76×：**每轮同步开销**

## 起因

docs/43 给出三核算术效率 75.6%、单核 95.8%，但 2 核只有 51.8% —— **非单调**，
当时解释不了。本脚本用「提交里的 tile 怎么分到核上」把这件事拆开。

## 关键结构事实（本脚本验证）

运行时的 `task_number` **永远是核数的整数倍**：

```
cm=7（3 核）: task_number ∈ {3, 6, 9, 12, 15, 21}   ← 3 的倍数
cm=3（2 核）: task_number ∈ {2, 4, 6, 8, 10, 14, 20} ← 2 的倍数
cm=1（1 核）: task_number = 588 的巨型提交
```

⇒ **一轮 = 每个核一个 tile**。一个提交 = 若干轮。
⇒ 提交的耗时由**最慢的那个核**决定 ⇒ 一轮的耗时 = 该轮内 **max(tile 工作量)**。

## 模型

```
t_总 = Σ_轮 [ max_work(该轮) / 单核峰值 ]  +  轮数 × SYNC
```

`SYNC` 是「多核每轮都要付一次、单核不用付」的同步/派发开销。
本脚本把它从三份语料里**分别反解**出来，看它们是否一致 —— 一致才叫模型。

## 用法
  python3 src/analysis/round_model.py
"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from tiling import load_segments                                   # noqa: E402
from cost_model import tile_work, load_timer                       # noqa: E402

PEAK = 1.024e12               # 单核阵列峰值（docs/44：32×32×1.0GHz）
BASE = os.path.expanduser("~/align2")

# (cm, 核数, dump 语料, 与 dump 对齐的 timer, 只挂 timer 的干净 hw 日志)
CFGS = [
    (7, 3, f"{BASE}/steady7", f"{BASE}/steady7_timer.log", f"{BASE}/hw7.log"),
    (3, 2, f"{BASE}/align2/cm3", f"{BASE}/align2/cm3_timer.log", f"{BASE}/hw3.log"),
    (1, 1, f"{BASE}/align2/cm1", f"{BASE}/align2/cm1_timer.log", f"{BASE}/hw1.log"),
]


def split_inferences(timer):
    """按 task_start 回绕把提交切成「趟」"""
    infs, cur = [], []
    for i, v in sorted(timer.items()):
        if cur and v["task_start"] < cur[-1][1]["task_start"]:
            infs.append(cur)
            cur = []
        cur.append((i, v))
    if cur:
        infs.append(cur)
    return infs


def main():
    print(f"{'═' * 96}")
    print("### 每轮同步开销模型：t = Σ_轮 max_work/峰值 + 轮数 × SYNC")
    print(f"    单核阵列峰值 = {PEAK/1e12:.3f} T-MAC/s（docs/44）")
    print()
    print(f"    {'cm':>3} {'核':>3} {'提交':>5} {'轮数':>6} "
          f"{'task_number 取值':>26} {'Σ每轮max':>12} {'算术(ms)':>9} "
          f"{'干净hw(ms)':>11} {'残差(ms)':>9} {'SYNC(µs/轮)':>12}")

    res = {}
    for cm, cores, dumps, tpath, hwpath in CFGS:
        if not (os.path.isdir(dumps) and os.path.exists(tpath)):
            print(f"    cm={cm}: 语料缺失，跳过")
            continue
        segs = load_segments(dumps)
        bt = collections.defaultdict(list)
        for s, d in segs:
            bt[(s.task.seq, s.task.index)].append(d)
        infs = split_inferences(load_timer(tpath))
        inf = infs[0] if cm != 7 else infs[0]

        nsub = len(inf)
        per_round, tn = [], collections.Counter()
        for i, v in inf:
            seq, lo, n = i - 1, v["task_start"], v["task_number"]
            tn[n] += 1
            tiles = [sum(tile_work(d)["macs_real"] for d in bt.get((seq, tt), []))
                     for tt in range(lo, lo + n)]
            for r in range(0, len(tiles), cores):
                g = tiles[r:r + cores]
                per_round.append(max(g) if g else 0)

        # 干净 hw（低开销探针的最后一趟）
        clean = None
        if os.path.exists(hwpath):
            ci = split_inferences(load_timer(hwpath))
            clean = sum(v["hw_ns"] for _, v in ci[-1]) / 1e6
        if clean is None:
            clean = sum(v["hw_ns"] for _, v in inf) / 1e6

        ar = sum(per_round) / PEAK * 1000.0
        nr = len(per_round)
        sync = (clean - ar) / nr * 1000.0
        res[cm] = (nsub, nr, sum(per_round), ar, clean, sync)
        tns = ", ".join(f"{k}×{v}" for k, v in sorted(tn.items()))
        print(f"    {cm:>3} {cores:>3} {nsub:>5} {nr:>6} {tns:>26} "
              f"{sum(per_round):>12.5g} {ar:>9.3f} {clean:>11.3f} "
              f"{clean-ar:>9.3f} {sync:>12.1f}")

    print()
    print("### 读法")
    print("    · 单核（cm=1）：SYNC ≈ 1.5 µs/轮 —— **基本没有每轮开销**，"
          "所以算术效率能到 95.8%")
    print("    · 多核（cm=3/7）：SYNC ≈ 22~37 µs/轮 —— **每一轮都要付一次同步/派发代价**，"
          "单核不用付")
    print("    · 2 核的轮数是 3 核的 1.43 倍（同一批 tile，核少就轮多）")
    print("      ⇒ 2 核付的同步次数更多，且单次更高 ⇒ 效率反而低于 3 核（非单调的来源）")
    print()
    print("### 这个模型解释了什么、没解释什么")
    print("    ✅ 解释了：为什么单核能到 95.8%、3 核只有 75.6% —— 差的就是每轮同步")
    print("    ✅ 解释了：为什么核数从 3 降到 2 反而更差 —— 轮数×1.43 且单次同步更高")
    print("    ⛔ 没解释：SYNC 为什么随核数变化（22.3 vs 36.7 µs/轮）。"
          "按「每轮一次同步」应该是常数")
    print("    ⛔ 没解释：把「每提交地板」也放进模型后方程组无解（会出现负的每提交地板）")
    print("      ⇒ 说明地板与轮开销**不可分离**，现有三个配置的自由度不足以定两者")
    return 0


if __name__ == "__main__":
    sys.exit(main())
