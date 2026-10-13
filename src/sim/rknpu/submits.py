#!/usr/bin/env python3
"""submits.py — 运行侧**提交结构**的解析与两条已验证定律（docs/133）

## 数据从哪来

板子上用 `rknpu_submit_timer.so` 抓的计时日志：`~/align2/tvcap/<model>_timer.log`

```
# 序号 task_start task_number core_mask wall_us hw_us overhead_us
1 79 6 0x7 506912 489995 16917
2 85 1 0x1 256956 247622 9334
...
# TOTAL submits=5 wall_us=1919 hw_us=1866 hw_ratio=97.3%
```

以及每个 task 的 regcmd：`~/align2/tvcap/<model>/regcmd_<submit>_task<N>.bin`

## 两条**全模型验证过**的定律（docs/133 §1）

1. `Σtask_number` == regcmd 文件个数（9/9 模型）
2. **`core_mask=0x7`（3 核）的提交，其 `task_number` 一定是 3 的倍数**
   —— 329 次提交、9 个模型，**无一例外**

定律 2 的含义：一次 3 核提交总是携带**整数个「轮」**（每轮给 3 个核各一个 task）。

## 边界（别把这里当提交数预测模型）

* 提交数**怎么从模型算出来**仍未解决 —— 本模块只做**解析与校验**，不做预测。
* 「串行提交数 = BatchNormalization 数」这条**已被反例打掉**
  （densenet121 是 62=62，但 efficientnet_b0 的 BN=0 而串行提交=66）。**不要复活它。**

## 用法
  PYTHONPATH=src/sim python3 -m rknpu.submits ~/align2/tvcap
  PYTHONPATH=src/sim python3 -m rknpu.submits --self-check
"""
from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, field

#: 3 核提交的 core_mask（docs/133 §2：实测只有 0x7 与 0x1 两种）
MASK_ALL_CORES = 0x7
MASK_ONE_CORE = 0x1

#: 定律 2 的模数：3 核提交的任务数必须是它的倍数
ROUND_MODULUS = 3


@dataclass
class Submit:
    index: int
    task_start: int
    task_number: int
    core_mask: int
    wall_us: int = 0

    @property
    def cores(self) -> int:
        """该提交用了几个核（按 mask 的置位数）"""
        return bin(self.core_mask).count("1")

    @property
    def rounds(self) -> int | None:
        """轮数 = 任务数 / 核数（整除才有意义）"""
        c = self.cores
        return self.task_number // c if c and self.task_number % c == 0 else None


@dataclass
class Capture:
    model: str
    submits: list[Submit] = field(default_factory=list)

    # ---- 定律 1 ----
    def regcmd_files(self, root: str) -> list[str]:
        return sorted(glob.glob(os.path.join(root, self.model, "regcmd_*.bin")))

    def law1_ok(self, root: str) -> tuple[bool, int, int]:
        """Σtask_number == regcmd 文件数"""
        tot = sum(s.task_number for s in self.submits)
        n = len(self.regcmd_files(root))
        return tot == n, tot, n

    # ---- 定律 2 ----
    def law2_violations(self) -> list[Submit]:
        """返回违反「3 核提交任务数必为 3 的倍数」的提交（应为空）"""
        return [s for s in self.submits
                if s.core_mask == MASK_ALL_CORES
                and s.task_number % ROUND_MODULUS]

    # ---- 分解 ----
    def parallel_rounds(self) -> int:
        return sum(s.rounds or 0 for s in self.submits if s.core_mask == MASK_ALL_CORES)

    def serial_tasks(self) -> int:
        return sum(s.task_number for s in self.submits if s.core_mask != MASK_ALL_CORES)

    def total_tasks(self) -> int:
        return sum(s.task_number for s in self.submits)

    def masks(self) -> set[int]:
        return {s.core_mask for s in self.submits}


def parse_timer_log(path: str) -> Capture:
    model = os.path.basename(path).replace("_timer.log", "")
    cap = Capture(model=model)
    for ln in open(path, errors="replace"):
        if ln.startswith("#") or len(ln.split()) < 4:
            continue
        p = ln.split()
        try:
            cap.submits.append(Submit(
                index=int(p[0]), task_start=int(p[1]), task_number=int(p[2]),
                core_mask=int(p[3], 16), wall_us=int(p[4]) if len(p) > 4 else 0))
        except ValueError:
            continue
    return cap


def load_all(root: str) -> list[Capture]:
    return [parse_timer_log(p) for p in sorted(glob.glob(os.path.join(root, "*_timer.log")))]


# ------------------------------------------------------------------ 自检

def self_check(root: str | None = None) -> int:
    """把两条定律在**全部抓取**上重跑一遍。返回失败数。

    root 缺省用 ~/align2/tvcap（板子抓取的位置）。
    """
    root = root or os.path.expanduser("~/align2/tvcap")
    caps = load_all(root)
    if not caps:
        print(f"[submits] 在 {root} 下没找到 *_timer.log")
        return 1

    bad = 0
    nsub = 0
    print(f"{'model':20s} {'submits':>7s} {'m0x7':>5s} {'定律1':>6s} {'定律2':>6s} "
          f"{'轮数':>6s} {'串行task':>8s} {'Σtask':>6s}")
    for c in caps:
        ok1, tot, nf = c.law1_ok(root)
        v2 = c.law2_violations()
        nsub += len(c.submits)
        n7 = sum(1 for s in c.submits if s.core_mask == MASK_ALL_CORES)
        print(f"{c.model:20s} {len(c.submits):7d} {n7:5d} "
              f"{'✓' if ok1 else f'✗({tot}≠{nf})':>6s} "
              f"{'✓' if not v2 else f'✗({len(v2)})':>6s} "
              f"{c.parallel_rounds():6d} {c.serial_tasks():8d} {c.total_tasks():6d}")
        bad += (0 if ok1 else 1) + (0 if not v2 else 1)

    # ★★★★ 恒等式 Σtasks = 3P + S（docs/134 §3）
    print(f"\n{'model':20s} {'P':>6s} {'S':>6s} {'3P+S':>7s} {'Σtask':>6s} {'恒等式':>6s} {'第1段-S':>8s}")
    for c in caps:
        ok, est, tot = identity_ok(c)
        seg = REPETITION_SEGMENTS.get(c.model)
        extra = (seg[0] - seg[1]) if seg else None
        tag = "" if extra is None else ("✓" if extra == c.serial_tasks() else f"✗({extra})")
        print(f"{c.model:20s} {c.parallel_rounds():6d} {c.serial_tasks():6d} "
              f"{est:7d} {tot:6d} {'✓' if ok else '✗':>6s} {tag:>8s}")
        bad += 0 if ok else 1

    print(f"\n合计 {len(caps)} 个模型 / {nsub} 次提交；失败 {bad} 项")
    print()
    print(falsified_report())
    return bad


# ------------------------------------------------- 恒等式（docs/134 §3）

#: ★★★★ `Σtasks = CORES_PARALLEL × P + S`，9/9 真实模型**精确**成立（docs/134 §3）
#:   P = 3 核提交的轮数之和；S = 1 核提交的任务数之和
#:   含义：并行部分的每个算子 × 3 个核各一个 task；串行部分只在 1 个核上跑
CORES_PARALLEL = 3


def identity_ok(cap: "Capture") -> tuple[bool, int, int]:
    """返回 (是否成立, 3P+S, Σtasks)"""
    est = CORES_PARALLEL * cap.parallel_rounds() + cap.serial_tasks()
    tot = cap.total_tasks()
    return est == tot, est, tot


#: ★ 实测的「各段 task 数」（docs/134 §2）：模型被复制 3 份，后两段**完全等长**
#:   第 1 段多出来的部分 == S（串行槽），这条也在下面的自检里核对
REPETITION_SEGMENTS: dict[str, list[int]] = {
    "resnet18":           [26, 24, 24],
    "mnasnet0_5":         [54, 53, 53],
    "mobilenet_v2":       [54, 53, 53],
    "squeezenet1_0":      [65, 60, 60],
    "shufflenet_v2_x0_5": [187, 156, 156],
    "densenet121":        [902, 835, 835],
    "regnet_x_400mf":     [1302, 921, 921],
}


# ------------------------------------------------------- 被证伪的假设（别复活）

#: ★ 已被**反例**打掉的假设（docs/133 §3）——
#: 「串行提交数 == BatchNormalization 算子数」看起来很像：
#:     densenet121 串行提交 62，IR 里 BatchNormalizationRelu 也是 62 —— **完全相等**
#: 但：
#:     efficientnet_b0 的 IR 里 BN = 0，串行提交 = 66  ⇒ **反例**
#: ⇒ 所以 densenet121 的 62=62 是**数值巧合**。把它记在这里，防止有人（包括我）复活它。
FALSIFIED: dict[str, dict] = {
    "serial_submits_equals_batch_norm": {
        "claim": "串行提交数 == BatchNormalization 算子数",
        "looked_good": "densenet121: 串行提交 62 == BatchNormalizationRelu 62",
        "counterexample": "efficientnet_b0: BatchNormalization 0 而串行提交 66",
    },
    "every_task_regcmd_is_130_insns": {
        "claim": "每个 task 的 regcmd 都是 130 条指令（1040 字节）",
        "looked_good": "resnet18 的 regcmd_000_task79.bin 是 1040 字节",
        "counterexample": "全量看：还有 34 / 77 / 134 条，且 efficientnet_b0 有 65 个 1105 条的",
    },
}


def falsified_report() -> str:
    out = ["已被反例打掉的假设（不要复活）："]
    for k, v in FALSIFIED.items():
        out.append(f"  · {v['claim']}")
        out.append(f"      看起来像：{v['looked_good']}")
        out.append(f"      反例    ：{v['counterexample']}")
    return "\n".join(out)


if __name__ == "__main__":
    if "--self-check" in sys.argv or len(sys.argv) == 1:
        raise SystemExit(1 if self_check() else 0)
    root = sys.argv[1]
    raise SystemExit(1 if self_check(root) else 0)
