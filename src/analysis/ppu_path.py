#!/usr/bin/env python3
"""
ppu_path.py — `OP_ENABLE=0x60` 到底是什么：**不是池化，是搬运引擎**

## 为什么这算一个发现

`OP_ENABLE` 位图（`0x0081/0x0008`）我们一直这样读：

```
0x1d = CNA|CORE|DPU|DPU_RDMA   → 卷积
0x18 = DPU|DPU_RDMA            → 逐元素/激活
0x60 = PPU|PPU_RDMA            → 「池化」   ← 这个标签是错的
```

yolov5s 里 **87 个 `0x60` tile 中有 81 个是退化的 1×1×32 拷贝**
（`in_height=1, out_width=1, in_channel=out_channel=32, POOLING_KERNEL_CFG=0`），
只有 6 个真的做了 2×2×256 的核。

⇒ **PPU 在这个模型里是运行时的「张量搬运/切片引擎」**（通道切片、concat 前的拷贝），
不是池化单元。「池化」这个标签是从位图名字推出来的，从来没有验证过。

## 顺带得到的两件事

1. **最纯的提交地板**：这些提交只搬 64 字节，却要 52.8~89.0 µs。
   ⇒ 地板不是被数据量撑起来的。
2. **拷贝 tile 数随核数放大**：27 个拷贝算子 ⇒ 3 核下 81 个 tile、2 核下 54 个。
   **同一个琐碎算子被每个核各发一遍** —— 多核在这条路上是「重复」而不是「并行」。

## 用法
  python3 src/analysis/ppu_path.py
"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from tiling import load_segments                                   # noqa: E402
from cost_model import load_timer                                  # noqa: E402

BASE = os.path.expanduser("~/align2")
CORPORA = [
    ("cm=7  3 核", f"{BASE}/steady7"),
    ("cm=3  2 核", f"{BASE}/align2/cm3"),
    ("cm=1  1 核", f"{BASE}/align2/cm1"),
]

# PPU 寄存器（上游命名，掩码均 13 位宽）
PPU_FIELDS = {
    0x6010: "DATA_CUBE_IN_HEIGHT",
    0x6014: "DATA_CUBE_IN_CHANNEL",
    0x6018: "DATA_CUBE_OUT_WIDTH",
    0x6020: "DATA_CUBE_OUT_CHANNEL",
    0x6024: "OPERATION_MODE_CFG",
    0x6034: "POOLING_KERNEL_CFG",
    0x6070: "DST_BASE_ADDR",
    0x7010: "RDMA_CUBE_IN_HEIGHT",
    0x7014: "RDMA_CUBE_IN_CHANNEL",
}


def main():
    print("### `OP_ENABLE = 0x60`（PPU|PPU_RDMA）到底是什么")
    print()
    for label, dumps in CORPORA:
        if not os.path.isdir(dumps):
            print(f"（跳过 {dumps}）")
            continue
        segs = load_segments(dumps)
        rows = []
        for s, d in segs:
            if s.task.seq >= 77:
                continue
            if not (s.enable and s.enable.value == 0x60):
                continue
            reg = {i.reg: i.value for i in s.insns}
            g = lambda r: ((reg.get(r, 0)) & 0x1FFF) + 1        # 减一编码
            rows.append((s.task.op_idx, g(0x6010), g(0x6014), g(0x6018),
                         g(0x6020), reg.get(0x6034), reg.get(0x6024)))
        if not rows:
            continue
        geo = collections.Counter(r[1:6] for r in rows)
        byop = collections.Counter(r[0] for r in rows)
        print(f"  ═══ {label}：`0x60` tile {len(rows)} 个，涉及 {len(byop)} 个算子")
        print(f"      (in_h, in_ch, out_w, out_ch, kernel_cfg) 分布：")
        for k, v in geo.most_common():
            kind = ("**退化的 1×1×32 拷贝**（不是池化）" if k[:4] == (1, 32, 1, 32)
                    else "真做了核的算子" if k[4] else "?")
            print(f"        {k} × {v:<4} {kind}")
        print(f"      每个算子的 tile 数分布：{dict(collections.Counter(byop.values()))}"
              f"   ← **随核数放大：同一个琐碎算子每核发一遍**")
        print()

    print("### 这些提交是「最纯的地板」：搬 64 字节要 50~140 µs")
    print(f"    {'配置':<12} {'零算术量提交':>12} {'中位 hw':>10} {'每次搬运字节':>14}")
    print("    " + "─" * 54)
    # 1 个 1×1×32 拷贝 = 32 B 读 + 32 B 写 = 64 B
    for label, cores, dumps, tpath in (
            ("cm=7  3 核", 3, f"{BASE}/steady7", f"{BASE}/hw7.log"),
            ("cm=3  2 核", 2, f"{BASE}/align2/cm3", f"{BASE}/hw3.log")):
        if not (os.path.isdir(dumps) and os.path.exists(tpath)):
            continue
        segs = load_segments(dumps)
        bt = collections.defaultdict(list)
        for s, d in segs:
            bt[(s.task.seq, s.task.index)].append(d)
        items = sorted(load_timer(f"{BASE}/steady7_timer.log" if cores == 3
                                  else f"{BASE}/align2/cm3_timer.log").items())
        infs, cur = [], []
        for i, v in items:
            if cur and v["task_start"] < cur[-1][1]["task_start"]:
                infs.append(cur); cur = []
            cur.append((i, v))
        if cur:
            infs.append(cur)
        sig = set()
        for k, (i, v) in enumerate(infs[0]):
            seq, lo, n = i - 1, v["task_start"], v["task_number"]
            if v["core_mask"] != (2 ** cores - 1):
                continue
            from cost_model import tile_work
            mac = sum(tile_work(d)["macs_real"] for tt in range(lo, lo + n)
                      for d in bt.get((seq, tt), []))
            if mac == 0:
                sig.add((k, v["task_start"], v["task_number"]))
        ci, cur = [], []
        for i, v in sorted(load_timer(tpath).items()):
            if cur and v["task_start"] < cur[-1][1]["task_start"]:
                ci.append(cur); cur = []
            cur.append((i, v))
        if cur:
            ci.append(cur)
        import statistics
        hws = []
        for inf in ci:
            for k, (i, v) in enumerate(inf):
                if (k, v["task_start"], v["task_number"]) in sig:
                    hws.append(v["hw_ns"] / 1000.0)
        if hws:
            print(f"    {label:<12} {len(hws):>12} {statistics.median(hws):>9.1f} µs "
                  f"{'64 B':>14}")

    print()
    print("### 结论")
    print("    1. **`0x60` 不是池化**：yolov5s 里 93% 的 `0x60` tile 是 1×1×32 的通道拷贝。")
    print("       位图的名字是从上游表里读来的，**语义从来没有被验证过** —— 这次验证了。")
    print("    2. 这些提交搬 64 字节却要 50~140 µs ⇒ **地板不是数据量撑起来的**。")
    print("    3. 拷贝 tile 数随核数放大（27 算子 → 3 核 81 tile / 2 核 54 tile）")
    print("       ⇒ **多核在这条路上是重复而不是并行**。")
    print("    4. ⛔ 仍未解释：同一进程交错 A/B 下，2 核的零算术量提交（138 µs）")
    print("       比 3 核（89 µs）**更慢**，而 2 核的 tile 数还更少（2 vs 3）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
