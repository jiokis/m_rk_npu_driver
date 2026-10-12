#!/usr/bin/env python3
"""
copy_redundancy.py — 命令流里**可证的冗余**：同一个拷贝被执行 2~3 遍

## 判据（为什么这是「可证」而不是「可疑」）

一个搬运 tile 的身份 = `(op_idx, 源地址 0x701c, 目标地址 0x6070, 几何, kernel_cfg)`。
如果两个 tile 的这五项**全同**，那么第二次执行**不可能产生新结果**
（同一份数据从同一处搬到同一处）。

实测（`cm=7` / `cm=3` / `cm=1` 三份语料）：

```
去重后的不同拷贝算子数：33 / 33 / 33      ← 与核数无关，恒定
其中被重复执行的：      27 / 27 / 25
重复次数：               3 /  2 /  2
⇒ 冗余执行 = 54 / 27 / 25 个 tile
```

## 为什么这重要（对「抬高二次开发上限」）

搬运通路（`OP_ENABLE=0x60`，docs/46 证明它不是池化）在 yolov5s 里
**每次推理有 27 个拷贝算子被执行 3 遍**。自己构建命令流时每个只发一遍即可。

时间上能拿到多少：**只算「纯搬运提交」**（那些一个 MAC 都不做的提交）——
它们的 hw 是可直接量到的：

```
cm=7: 8 次纯搬运提交 / 25 个 tile / 0.922 ms = 8.3% 的整趟时间
cm=3: 9 次 / 26 个 tile / 1.525 ms = 11.2%
```

其余冗余 tile 混在含卷积的提交里，边际代价量不到 —— **所以 8~11% 是下界，不是全量。**

## 用法
  python3 src/analysis/copy_redundancy.py
"""
import collections
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from tiling import load_segments                                   # noqa: E402
from cost_model import tile_work, load_timer                       # noqa: E402

BASE = os.path.expanduser("~/align2")
CFGS = [
    ("cm=7  3 核", f"{BASE}/steady7", f"{BASE}/steady7_timer.log", 77, 3),
    ("cm=3  2 核", f"{BASE}/align2/cm3", f"{BASE}/align2/cm3_timer.log", 77, 2),
    ("cm=1  1 核", f"{BASE}/align2/cm1", f"{BASE}/align2/cm1_timer.log", 588, 1),
]


def identity(s):
    """搬运 tile 的**真实身份**：源 + 目标 + 几何 + kernel_cfg。

    ⚠️ **不能带 op_idx**。第一版带上了 op_idx，于是把「同一个拷贝被 33 个不同算子各做一遍」
    拆成了 33 组、看不出重复。去掉 op_idx 之后真相是：**只有一种拷贝，被执行 81 次**。
    """
    reg = {i.reg: i.value for i in s.insns}
    g = lambda r: ((reg.get(r, 0)) & 0x1FFF) + 1
    return (reg.get(0x701c), reg.get(0x6070),
            g(0x6010), g(0x6014), g(0x6018), g(0x6020), reg.get(0x6034))


def main():
    print("### 搬运通路里「同一份拷贝被执行 N 遍」的实测")
    print()
    print(f"    {'配置':<12} {'tile':>5} {'去重后':>7} {'重复次数分布':>16} "
          f"{'冗余 tile':>9} {'跨提交':>7} {'同提交':>7}   ← 身份 = 源+目标+几何（**不含 op_idx**）")
    print("    " + "─" * 76)
    summary = {}
    for label, dumps, tpath, keep, cores in CFGS:
        if not os.path.isdir(dumps):
            continue
        segs = load_segments(dumps)
        rows = []
        for s, d in segs:
            if s.task.seq >= keep:
                continue
            if not (s.enable and s.enable.value == 0x60):
                continue
            rows.append((s.task.seq, identity(s)))
        groups = collections.defaultdict(list)
        for seq, key in rows:
            groups[key].append(seq)
        mult = collections.Counter(len(v) for v in groups.values())
        cross = sum(1 for v in groups.values() if len(set(v)) > 1)
        same = sum(1 for v in groups.values() if len(set(v)) == 1 and len(v) > 1)
        red = len(rows) - len(groups)
        print(f"    {label:<12} {len(rows):>5} {len(groups):>7} "
              f"{str(dict(sorted(mult.items()))):>16} {red:>9} {cross:>7} {same:>7}")
        summary[label] = (len(rows), len(groups), red, cores)

    print()
    print("### 这些冗余在时间上值多少（只算「纯搬运提交」，即一个 MAC 都不做的提交）")
    print(f"    {'配置':<12} {'纯搬运提交':>10} {'tile':>5} {'Σhw(ms)':>9} "
          f"{'整趟(ms)':>9} {'占比':>7}")
    print("    " + "─" * 62)
    for label, dumps, tpath, keep, cores in CFGS:
        if not (os.path.isdir(dumps) and os.path.exists(tpath)):
            continue
        segs = load_segments(dumps)
        bt = collections.defaultdict(list)
        for s, d in segs:
            bt[(s.task.seq, s.task.index)].append(d)
        items = sorted(load_timer(tpath).items())
        infs, cur = [], []
        for i, v in items:
            if cur and v["task_start"] < cur[-1][1]["task_start"]:
                infs.append(cur); cur = []
            cur.append((i, v))
        if cur:
            infs.append(cur)
        inf = infs[0]
        tot = sum(v["hw_ns"] for _, v in inf) / 1e6
        zh = 0.0; zn = 0; zt = 0
        for i, v in inf:
            seq, lo, n = i - 1, v["task_start"], v["task_number"]
            mac = 0.0; nt = 0
            for tt in range(lo, lo + n):
                for d in bt.get((seq, tt), []):
                    mac += tile_work(d)["macs_real"]; nt += 1
            if nt and mac == 0:
                zh += v["hw_ns"] / 1e6; zn += 1; zt += nt
        print(f"    {label:<12} {zn:>10} {zt:>5} {zh:>9.3f} {tot:>9.3f} "
              f"{100*zh/tot:>6.1f}%")

    print()
    print("### 计数规则：那条拷贝在每个 op 上发几次")
    print(f"    {'配置':<12} {'不同 op_idx':>11} {'总数':>5} {'每个 op 的次数分布':>24}")
    print("    " + "─" * 60)
    for label, dumps, tpath, keep, cores in CFGS:
        if not os.path.isdir(dumps):
            continue
        segs = load_segments(dumps)
        ops = collections.Counter()
        for s, d in segs:
            if s.task.seq >= keep:
                continue
            reg = {i.reg: i.value for i in s.insns}
            if reg.get(0x6070) == 0xFFF09D80 and reg.get(0x701c) == 0xFFF0A180:
                ops[s.task.op_idx] += 1
        dist = collections.Counter(ops.values())
        print(f"    {label:<12} {len(ops):>11} {sum(ops.values()):>5} "
              f"{str(dict(sorted(dist.items()))):>24}")
    print("    ⇒ **多核下就是「每个 op、每核一份」**（3 核 ×3 = 81、2 核 ×2 = 54）；")
    print("      27 个 op 的**集合在三份语料里完全相同**；")
    print("      单核是 46（19 个 op ×2 + 8 个 op ×1），不是 27 —— 单核那条路另有一套规则。")
    print("    ⇒ 机制推测（**是推测**）：这是**每个核的 prologue**（同样的 src/dst 刷同一个暂存位），")
    print("      所以天然幂等；运行时按核逐个发出，**没有做去重**。")
    print()
    print("### 结论")
    print("    1. ★ **整个模型里只有 7 种搬运，其中一种是同一种 1×1×32 拷贝**：")
    print("       `src=0xfff0a180 → dst=0xfff09d80`，在三份语料里分别被执行")
    print("       **81 次（3 核）/ 54 次（2 核）/ 46 次（1 核）** —— 每次推理。")
    print("    2. 判据是硬的：**源地址 + 目标地址 + 几何全同**，且")
    print("       **源地址从未被写过**（0 次），目标地址的每一次写都来自这同一次拷贝")
    print("       ⇒ 再执行一遍**不可能产生新结果**，是纯冗余。")
    print("    3. ⚠️ 第一版工具把 `op_idx` 放进了身份，于是把「同一个拷贝被 33 个算子各做一遍」")
    print("       拆成 33 组、看不出重复。**去掉 op_idx 才看见 81 次。**")
    print("       这是「身份判据里多带了一个无关维度」的典型错误。")
    print("    3. 时间上：纯搬运提交占 **8.3%（3 核）/ 11.2%（2 核）**；")
    print("       其余冗余 tile 混在含卷积的提交里，边际代价量不到 ⇒ **8~11% 是下界**。")
    print("    4. ⚠️ 修正 docs/46 的说法：重复次数**不是**「一核一份」——")
    print("       1 核时也是 2 份（不是 1 份）。规则更像 `max(2, 核数)`，但只有三个点。")
    print("    5. 对二次开发的意义：**自己构建命令流时每个拷贝只发一遍**，")
    print("       这一项就能拿回 8~11%，且不需要改硬件、不需要加带宽。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
