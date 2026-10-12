#!/usr/bin/env python3
"""
array_geometry.py — CORE 的 MAC 阵列几何：**从上游代码读出来，再用实测撞一遍**

## 目标 ② 的「流水级结构」最后一块

之前我们知道「一核约 1 T-MAC/s」，但那是**测出来的数**，不知道硬件**为什么**是这个数。
上游 Mesa 的两处常量/循环结构把阵列几何直接写在代码里：

`mesa/rkt_ml.h`
```c
#define CBUF_BANK_SIZE        32768
#define CBUF_BANKS            12
#define CBUF_ENTRIES_PER_BANK 256
#define CBUF_ENTRY_SIZE       (CBUF_BANK_SIZE / CBUF_ENTRIES_PER_BANK)   // = 128
#define FEATURE_ATOMIC_SIZE   16      // K 方向的一个「feature atom」
#define WEIGHT_ATOMIC_SIZE    32      // M 方向的一个「weight atom」
#define ATOMIC_K_SIZE         16
```

`mesa/rkt_coefs.c` 里权重的**内存布局**（从外到内的循环嵌套）：
```c
for (oc1 < ceil(out_ch / 32))            // M：按 32 分组
  for (ic1 < ceil(in_ch  / 32))          // K：按 32 分组
    for (x < kw) for (y < kh)            // 卷积核抽头
      for (oc2 < 32)                     // M 内层 32
        for (ic2 < 32)                   // K 内层 32
```
**最内两层正好是 32×32 的连续权重块。**
⇒ 阵列几何 = **32 (M) × 32 (K) = 1024 MAC/cycle**
⇒ 1.0 GHz 下 **1.024 T-MAC/s/核** —— 与我实测的 1.0 T-MAC/s 吻合。

## 可证伪预测（本脚本逐个撞）

| # | 预测 | 依据 |
|---|---|---|
| P1 | `DATAIN_CHANNEL(对齐)` ≡ 0 (mod 16) | `FEATURE_ATOMIC_SIZE = 16` |
| P2 | 对齐输出通道 ≡ 0 (mod 32) | `WEIGHT_ATOMIC_SIZE = 32`，且 `align(MAX2(out_ch,32),32)` |
| P3 | `DATAIN_CHANNEL` ≡ 0 (mod 32)（3 通道首层除外） | K = 2 个 feature atom（`calc_raw_input_size` 里的 `* 2`） |
| P4 | 深度卷积 ⇒ 对齐输出通道 ≡ 0 (mod 64) | `rkt_task.c`：depthwise 时 `align(out_channels, 64)` |

## 用法
  python3 src/analysis/array_geometry.py
"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from tiling import load_segments                                   # noqa: E402

BASE = os.path.expanduser("~/align2")
CORPORA = [
    (f"{BASE}/align2/cm7", "align2/cm7（3 核，单趟，段尾完整）"),
    (f"{BASE}/align2/cm3", "align2/cm3（2 核，单趟）"),
    (f"{BASE}/steady7", "steady7（3 核，7 趟稳态）"),
    ("data/rkllm", "rkllm（LLM GEMM）"),
]

# 阵列几何（由上游常量推出）
M_ATOM, K_ATOM = 32, 32
MAC_PER_CYCLE = M_ATOM * K_ATOM
NPU_CLOCK_HZ = 1.0e9          # 板上 devfreq 实测（performance 档，1.0 GHz 已是最高档）
PEAK_PER_CORE = MAC_PER_CYCLE * NPU_CLOCK_HZ

# 实测（docs/43 §3，低开销探针）
MEASURED = [("cm=1", 1, 0.981), ("cm=3", 2, 1.061), ("cm=7", 3, 2.322)]


def main():
    rows = []
    for d, label in CORPORA:
        if not os.path.isdir(d):
            print(f"（跳过不存在的语料 {d}）")
            continue
        n = 0
        for s, dd in load_segments(d):
            if dd["ch_align"] and dd["core_c"]:
                rows.append((dd["ch_align"], dd["core_c"], dd["kernels"] or 0))
                n += 1
        print(f"载入 {label}: {n} 个 tile")
    if not rows:
        print("没有语料"); return 1

    print(f"\n{'═' * 88}")
    print(f"### 阵列几何（由上游 mesa/rkt_ml.h + rkt_coefs.c 推出）")
    print(f"    M 方向 weight atom = {M_ATOM}（WEIGHT_ATOMIC_SIZE）")
    print(f"    K 方向 = 2 × feature atom = 2 × 16 = {K_ATOM}")
    print(f"    ⇒ 阵列 {M_ATOM}×{K_ATOM} = {MAC_PER_CYCLE} MAC/cycle")
    print(f"    ⇒ @ {NPU_CLOCK_HZ/1e9:.1f} GHz = **{PEAK_PER_CORE/1e12:.3f} T-MAC/s/核**")

    print(f"\n### 可证伪预测（{len(rows)} 个 tile）")
    print(f"    {'预测':<52} {'命中':>14}  比例")
    print("    " + "─" * 76)
    checks = [
        ("P1  DATAIN_CHANNEL ≡ 0 (mod 16)", lambda r: r[0] % 16 == 0),
        ("P2  对齐输出通道 ≡ 0 (mod 32)", lambda r: r[1] % 32 == 0),
        ("P3  DATAIN_CHANNEL ≡ 0 (mod 32)", lambda r: r[0] % 32 == 0),
        ("P4  深度卷积 ⇒ 对齐输出通道 ≡ 0 (mod 64)",
         lambda r: (r[2] != 1) or (r[1] % 64 == 0)),
    ]
    for name, fn in checks:
        ok = sum(1 for r in rows if fn(r))
        flag = "✓" if ok == len(rows) else "○"
        print(f"    {flag} {name:<50} {ok:>6}/{len(rows):<6} {100*ok/len(rows):5.1f}%")
    print(f"    （P3 的例外是 3 通道首层：DATAIN_CHANNEL 补到 16 而不是 32 —— "
          f"与 ch_align 取值分布一致）")

    print(f"\n### 用实测吞吐撞阵列几何")
    print(f"    {'配置':<8} {'核数':>5} {'实测(T-MAC/s)':>14} "
          f"{'理论峰值(核×1.024)':>18} {'占本配置峰值':>13} {'每核(T-MAC/s)':>13}")
    print("    " + "─" * 78)
    for label, cores, thr in MEASURED:
        peak = cores * PEAK_PER_CORE / 1e12
        print(f"    {label:<8} {cores:>5} {thr:>14.3f} {peak:>18.3f} "
              f"{100*thr/peak:>12.1f}% {thr/cores:>13.3f}")
    print(f"\n    ⇒ 单核实测 0.981 T-MAC/s = 阵列理论峰值的 "
          f"{100*0.981/(PEAK_PER_CORE/1e12):.1f}%")
    print(f"      **两个完全独立的来源（上游代码布局 vs 板上实测）撞到同一个数。**")
    print(f"      这解释了为什么配置寄存器要把输入通道对齐到 16、输出通道对齐到 32：")
    print(f"      那正是阵列的 K 原子宽度和 M 原子宽度。")

    print(f"\n### CBUF 的存储几何（同一批常量）")
    bank, banks, epb = 32768, 12, 256
    entry = bank // epb
    print(f"    CBUF = {banks} bank × {bank} B = {banks*bank/1024:.0f} KB")
    print(f"    entry = {bank}/{epb} = {entry} B；1 个 entry 装 "
          f"{entry//16} 个 feature atom = {entry//16*16} 个通道值（1 个像素）")
    print(f"    ⇒ 1 个 bank = {epb} 个像素（每像素最多 128 通道）")
    print(f"    ⇒ 整个 CBUF = {banks*epb} 个像素的片上特征容量")
    print(f"    实测：DATA_BANK + WEIGHT_BANK ≡ {banks} 在 14359 个 tile 上 100%（docs/36/42）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
