#!/usr/bin/env python3
"""taskmap.py — 把运行侧的每个 task 映射回**算子**（docs/134）

## 这一步解决什么

docs/133 只看清了「提交」这一层（核掩码、任务数）。本轮往下走一层：
**把 74 个 task 逐个映射到算子**，靠的是 regcmd 里的 CNA 寄存器。

## 关键寄存器（实测语义，docs/134 §2）

| 寄存器 | 语义 | 证据 |
|---|---|---|
| CNA(513) **4136** | **空间尺寸**（特征图边长） | resnet18 的取值序列 = `112, 56×4, 28×5, 14×5, 7×5, 1` —— 与它的特征图**逐一对应** |
| CNA(513) **4368** | **输入张量地址** | 随 task 单调递增；等于上一个 task 的 RDMA 输出地址附近（差 512） |
| DPU(4097) **16416** | 疑似权重/描述符地址 | 三次重复中**取值不同** ⇒ 见 §3 |

## ★★★ 本轮最重要的发现（docs/134 §3）

**task 序列里「算子序列」重复了 3 次**，而且：

```
task 79 （第 1 遍）: 权重 = 4282006528   输入 = 4283531264
task 105（第 2 遍）: 权重 = 4282074624   输入 = 4283531264   ← 权重不同，输入**相同**
task 129（第 3 遍）: 权重 = 4282140928   输入 = 4283531264   ← 权重又不同，输入仍相同
```

**⇒ 与 `RKNNDuplicateWeightPass`（权重复制）+ `RKNNMultiSurfacePass`（多 surface）一致：
   **权重复制 3 份（每核一份），三个核读同一份输入、各处理一片数据。****

⚠️ **标记为「强指示」而非「已证明」**：DPU 16416 的语义尚未独立确认。

## 用法
  PYTHONPATH=src/sim python3 -m rknpu.taskmap ~/align2/tvcap/resnet18
  PYTHONPATH=src/sim python3 -m rknpu.taskmap --repetitions ~/align2/tvcap
"""
from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass

from . import regcmd

#: 实测语义（docs/134 §2）
REG_SPATIAL = (513, 4136)     # CNA：空间尺寸（特征图边长）
REG_INPUT_ADDR = (513, 4368)  # CNA：输入张量地址
REG_WEIGHT = (4097, 16416)    # DPU：疑似权重/描述符地址


@dataclass
class TaskInfo:
    index: int
    spatial: int | None
    in_addr: int | None
    weight: int | None
    n_insn: int

    def __str__(self) -> str:
        return (f"task {self.index:4d}  空间={self.spatial!s:>5s}  "
                f"输入={self.in_addr}  权重={self.weight}  指令={self.n_insn}")


def load_task(path: str, index: int) -> TaskInfo:
    ins = regcmd.decode(open(path, "rb").read())
    regs = {(i.block, i.reg): i.value for i in ins}
    return TaskInfo(index=index, spatial=regs.get(REG_SPATIAL),
                    in_addr=regs.get(REG_INPUT_ADDR), weight=regs.get(REG_WEIGHT),
                    n_insn=len(ins))


def load_dir(model_dir: str) -> list[TaskInfo]:
    """按 **task 全局序号**排序载入（文件名的 `_task<N>` 就是序号）"""
    fs = glob.glob(os.path.join(model_dir, "regcmd_*.bin"))
    fs.sort(key=lambda p: int(p.split("_task")[1][:-4]))
    return [load_task(f, int(f.split("_task")[1][:-4])) for f in fs]


def spatial_runs(tasks: list[TaskInfo]) -> list[tuple[int | None, list[int]]]:
    """把 task 序列按「空间尺寸」切成连续段 —— 段的形状直接反映图的形状"""
    runs: list[tuple[int | None, list[int]]] = []
    for t in tasks:
        if runs and runs[-1][0] == t.spatial:
            runs[-1][1].append(t.index)
        else:
            runs.append((t.spatial, [t.index]))
    return runs


def find_repetitions(tasks: list[TaskInfo]) -> list[list[int]]:
    """按「空间尺寸序列回到起点」找重复（用于发现多核/多次执行）

    启发式：从每个 `spatial` 取到最大值的 task 起，切一段。
    """
    if not tasks:
        return []
    mx = max((t.spatial or 0) for t in tasks)
    starts = [i for i, t in enumerate(tasks) if t.spatial == mx]
    out: list[list[int]] = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(tasks)
        out.append([t.index for t in tasks[s:e]])
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    if sys.argv[1] == "--repetitions":
        root = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/align2/tvcap")
        for d in sorted(glob.glob(os.path.join(root, "*/"))):
            name = os.path.basename(d.rstrip("/"))
            if not glob.glob(os.path.join(d, "regcmd_*.bin")):
                continue
            ts = load_dir(d)
            reps = find_repetitions(ts)
            same_in = len({t.in_addr for t in ts if t.in_addr})
            same_w = len({t.weight for t in ts if t.weight})
            print(f"{name:20s} {len(ts):5d} tasks  {len(reps)} 段重复  "
                  f"distinct 输入={same_in:4d} 权重={same_w:4d}  "
                  f"各段 task 数={[len(r) for r in reps]}")
        return 0

    ts = load_dir(sys.argv[1])
    print(f"{len(ts)} 个 task")
    for t in ts:
        print(" ", t)
    print("\n按空间尺寸的连续段:")
    for sp, idx in spatial_runs(ts):
        print(f"  空间={sp!s:>5s}  ×{len(idx):3d}  tasks {idx[0]}..{idx[-1]}")
    reps = find_repetitions(ts)
    print(f"\n发现 {len(reps)} 段重复，各段 task 数 = {[len(r) for r in reps]}")
    if len(reps) >= 2:
        print("各段第一个 task 的（输入地址, 权重地址）:")
        for r in reps:
            t = next(t for t in ts if t.index == r[0])
            print(f"  task {t.index:4d}  输入={t.in_addr}  权重={t.weight}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
