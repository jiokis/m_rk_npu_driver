#!/usr/bin/env python3
"""taskdesc.py — 解析 40 字节的 `rknpu_task` 描述符（docs/135/136）

## 数据从哪来

板子上抓取时会写 `tasks_<submit>.bin`（40 字节 × `task_number`），
但**之前的抓取只把 `regcmd_*.bin` 拷回来了**，`tasks_*.bin` 留在板子的 `/tmp`（tmpfs，已清空）。
重抓：

  LD_PRELOAD=~/rknpu_capture.so RKNPU_DUMP_DIR=/tmp/x RKNPU_DUMP_BUFS_MAX=0 \\
    python3 -c "RKNNLite: load; init_runtime(core_mask=...); inference 一次"

## 字段（实测，docs/135 §2）

| 偏移 | 含义 | 取值 |
|---|---|---|
| +0x00 | flags | 恒 0 |
| +0x04 | **算子序号 `op_idx`** | resnet18 取 1..23，**每遍从 1 重启** |
| +0x08 | f08 | 29 / 96 |
| +0x0c | f0c | 768 / 3072 |
| +0x10 | f10 | 恒 0x1ffff |
| +0x14 | f14 | 0 / 512 / 1024 / 2048（随 core_mask 变） |
| +0x18 | f18 | 122 / 26 / 126 |
| +0x1c | f1c | 恒 0 |
| +0x20 | f20 | 递增地址（相邻差 0x400） |
| +0x24 | f24 | 恒 0 |

## ★ 两条**实测结论**（别搞混）

1. **`op_idx` 从 1 重启 ⇒ task 列表里有 3 遍完整算子序列。**
   但 [docs/136] 证明**这 3 遍与核数无关**（`core_mask=1` 时依然 3 遍）
   ⇒ 是编译期烘进模型的 **3 个 surface**，**不是 3 个核**。
   （docs/134/135 曾误判为「3 个核」，已被 1 核重抓推翻。）
2. **`f14` 随 `core_mask` 变**（1 核 {0,512}；2 核 {0,1024}；3 核 {0,1024,2048}）
   ⇒ 与分片/缓冲布局有关，**不是核序号**。

## 用法
  PYTHONPATH=src/sim python3 -m rknpu.taskdesc data/captures/resnet18_tasks
  PYTHONPATH=src/sim python3 -m rknpu.taskdesc --compare <目录1> <目录2> ...
"""
from __future__ import annotations

import glob
import os
import struct
import sys
from dataclasses import dataclass

TASK_SIZE = 40

#: 字段名 → 偏移（docs/135 §2）
FIELDS = {
    "flags": 0x00, "op_idx": 0x04, "f08": 0x08, "f0c": 0x0C, "f10": 0x10,
    "f14": 0x14, "f18": 0x18, "f1c": 0x1C, "f20": 0x20, "f24": 0x24,
}


@dataclass
class TaskDesc:
    submit: int
    index: int
    op_idx: int
    f08: int
    f0c: int
    f10: int
    f14: int
    f18: int
    f20: int

    def __str__(self) -> str:
        return (f"submit {self.submit} task {self.index:3d}  op={self.op_idx:3d}  "
                f"f08={self.f08:3d} f0c={self.f0c:5d} f14={self.f14:5d} f18={self.f18:3d}")


def parse_file(path: str, submit: int) -> list[TaskDesc]:
    b = open(path, "rb").read()
    assert len(b) % TASK_SIZE == 0, f"{path} 长度 {len(b)} 不是 40 的倍数"
    out = []
    for i in range(len(b) // TASK_SIZE):
        v = struct.unpack_from("<10I", b, TASK_SIZE * i)
        out.append(TaskDesc(submit=submit, index=i, **{
            k: v[off // 4] for k, off in FIELDS.items() if k not in ("flags", "f1c", "f24")
        }))
    return out


def load_dir(d: str) -> list[TaskDesc]:
    """按文件名里的 submit 序号排序载入"""
    fs = glob.glob(os.path.join(d, "tasks_*.bin"))
    fs.sort(key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
    out: list[TaskDesc] = []
    for f in fs:
        out += parse_file(f, len(out) and out[-1].submit + 1 or 0)
    return out


def op_runs(tasks: list[TaskDesc]) -> list[tuple[int, int, int]]:
    """把 op_idx 序列切成「遍」：返回 [(起始下标, 结束下标, 遍序号), ...]

    判据：`op_idx` 回到 1（且上一个不是 1）就是新的一遍开始。
    """
    runs = []
    start = 0
    rep = 0
    for i in range(1, len(tasks)):
        if tasks[i].op_idx == 1 and tasks[i - 1].op_idx != 1:
            runs.append((start, i, rep))
            start, rep = i, rep + 1
    runs.append((start, len(tasks), rep))
    return runs


def summarize(tasks: list[TaskDesc]) -> str:
    lines = [f"{len(tasks)} 个 task，{len({t.submit for t in tasks})} 个 submit"]
    subs: dict[int, list[TaskDesc]] = {}
    for t in tasks:
        subs.setdefault(t.submit, []).append(t)
    for s, ts in sorted(subs.items()):
        ops = [t.op_idx for t in ts]
        f14 = sorted({t.f14 for t in ts})
        lines.append(f"  submit {s}: {len(ts):3d} tasks  op {ops[0]}..{ops[-1]}  "
                     f"f14∈{f14}")
    runs = op_runs(tasks)
    lines.append(f"  ★ op_idx 从 1 重启 {len(runs) - 1} 次 ⇒ {len(runs)} 遍"
                 f"（{len(runs)} 个 surface，与核数无关，见 docs/136）")
    lines.append("  各遍 task 数: " + str([b - a for a, b, _ in runs]))
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    if sys.argv[1] == "--compare":
        for d in sys.argv[2:]:
            print(f"=== {d} ===")
            print(summarize(load_dir(d)))
            print()
        return 0
    print(summarize(load_dir(sys.argv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
