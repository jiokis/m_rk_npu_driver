#!/usr/bin/env python3
"""addr_chain.py — 检验「tensor 地址链能否标记段边界」（**负结果**，docs/138）

## 结论（先说）

**不能。** 9 个模型里只有 2 个「段内差值集中」，跨界的差值分布与段内无可区分特征。
详见 docs/138。

## 为什么还留着这个脚本

1. 它是**可复现的否证**：将来若有人提出「段边界 = 缓冲复用边界」，跑一下就知道。
2. 它顺带产出一个**全量差值分布** ——
   docs/134 曾据「前 8 对差值 512」外推成「地址链」，全量看 512 只占 18%（docs/138 §3）。

## 用法
  PYTHONPATH=src/sim python3 src/analysis/addr_chain.py <regcmd 目录>
  PYTHONPATH=src/sim python3 src/analysis/addr_chain.py --all ~/align2   # 扫 rc_* 目录
"""
from __future__ import annotations

import glob
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))
from rknpu import regcmd  # noqa: E402

REG_IN = (513, 4368)         # CNA：输入张量地址（实测语义，docs/134）
REG_OUT = (8193, 20512)      # DPU_RDMA：输出地址


def _key(p: str) -> tuple[int, int]:
    m = re.search(r"regcmd_(\d+)_task(\d+)", os.path.basename(p))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def load_seq(d: str) -> list[tuple[int, int, int, int]]:
    """返回 [(submit, task, 输入地址, 输出地址), ...]，按 (submit, task) 排序"""
    seq = []
    for f in sorted(glob.glob(os.path.join(d, "regcmd_*.bin")), key=_key):
        sub, tk = _key(f)
        r = {}
        for i in regcmd.decode(open(f, "rb").read()):
            r[(i.block, i.reg)] = i.value
        seq.append((sub, tk, r.get(REG_IN, 0), r.get(REG_OUT, 0)))
    return seq


def analyze(d: str) -> dict:
    seq = load_seq(d)
    inside, cross = Counter(), Counter()
    for a, b in zip(seq, seq[1:]):
        if not a[3] or not b[2]:
            continue
        dv = b[2] - a[3]
        (cross if a[0] != b[0] else inside)[dv] += 1
    ni, nc = sum(inside.values()), sum(cross.values())
    return {
        "name": os.path.basename(d.rstrip("/")).replace("rc_", ""),
        "inside": inside.most_common(3),
        "cross": cross.most_common(3),
        "inside_top_share": (inside.most_common(1)[0][1] / ni) if ni else 0.0,
        "n_cross": nc,
        # ⚠️ 不能写 Counter({**inside, **cross}) —— 字典合并会让**同名键互相覆盖**而不是相加
        "full": (inside + cross).most_common(6),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    if sys.argv[1] == "--all":
        root = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/align2")
        dirs = [d for d in glob.glob(os.path.join(root, "rc_*")) if glob.glob(os.path.join(d, "regcmd_*.bin"))]
    else:
        dirs = [sys.argv[1]]
    print(f"{'model':20s} {'段内差值(众数)':>24s} {'跨界差值(众数)':>24s} {'段内首位占比':>10s} {'全量差值(前6)':>34s}")
    for d in sorted(dirs):
        r = analyze(d)
        print(f"{r['name']:20s} {str(r['inside']):>24s} {str(r['cross']):>24s} "
              f"{r['inside_top_share']:>9.0%} {str(r['full']):>34s}")
    print("\n★ 判定：只有「段内首位占比」明显高（>50%）的模型才有信号；docs/138 实测只有 2/9。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
