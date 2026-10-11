"""
rknpu.loader — 从抓取语料载入任务与内存
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

from .machine import Memory, Task

DEFAULT_DUMPS = os.environ.get("RKNPU_DUMPS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "dumps2")

TASK_RE = re.compile(
    r"task\[(\d+)\] flags=.*?op_idx=(\d+).*?regcfg_amount=(\d+)"
    r".*?regcmd_addr=(0x[0-9a-f]+)")


def load_corpus(dumps: str = DEFAULT_DUMPS):
    """
    返回 (Memory, tasks, by_addr)
      tasks   : [(seq, task_index, Task)] 按提交与序号排序
      by_addr : {设备地址: Task}，用于顺链查找
    """
    log = os.path.join(dumps, "capture.log")
    if not os.path.exists(log):
        raise FileNotFoundError(f"找不到 {log}")

    meta: Dict[Tuple[int, int], dict] = {}
    seq = -1
    for line in open(log, encoding="utf-8", errors="replace"):
        if line.startswith("[SUBMIT]"):
            seq += 1
            continue
        m = TASK_RE.search(line)
        if m and seq >= 0:
            meta[(seq, int(m.group(1)))] = {
                "op_idx": int(m.group(2)),
                "amount": int(m.group(3)),
                "addr": int(m.group(4), 16),
            }

    mem = Memory()
    tasks: List[Tuple[int, int, Task]] = []
    by_addr: Dict[int, Task] = {}
    for f in sorted(os.listdir(dumps)):
        m = re.match(r"regcmd_(\d+)_task(\d+)\.bin$", f)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        mm = meta.get(key)
        if not mm:
            continue
        data = open(os.path.join(dumps, f), "rb").read()
        mem.load(mm["addr"], data)
        t = Task(seq=key[0], index=key[1], op_idx=mm["op_idx"],
                 addr=mm["addr"], amount=mm["amount"])
        tasks.append((key[0], key[1], t))
        by_addr.setdefault(mm["addr"], t)

    tasks.sort(key=lambda x: (x[0], x[1]))
    return mem, tasks, by_addr
