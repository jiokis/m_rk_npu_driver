#!/usr/bin/env python3
"""
tiling.py — 用上游 Mesa `rkt_task.c` 的分块模型去撞我们自己的 dump

## 为什么这一步是「结构级模型」的地基

上游 Mesa rocket 驱动里有一个完整的 **分块（split task）模型**
(`mesa/rkt_task.c: rkt_split_tasks()`)，它把每个算子切成若干 tile，每个 tile
一行命令流。如果这个模型对，那么：

  · 每个 tile 的寄存器值可以被**预测**（而不是只能事后解码）
  · 每个 tile 的 **MAC 数**可以被算出来
  · ⇒ 给定一个模型的算子表，就能算出「总工作量」，再乘以实测吞吐 = **预测耗时**

这就是目标 ③ 要的东西。但前提是：先把上游模型的预测逐字段撞一遍。

## 上游给出的可证伪预测（`rkt_regcmd.c` + `rkt_task.c`）

| 寄存器 | 上游语义 | 预测 |
|---|---|---|
| CNA 0x1020[0..10]   | `DATAIN_HEIGHT`   | = **本 tile 的输入高度**（不是整层高度） |
| CNA 0x1020[16..26]  | `DATAIN_WIDTH`    | = 整层输入宽度 |
| CNA 0x1024[0..15]   | `DATAIN_CHANNEL`  | = 对齐后输入通道 |
| CNA 0x1024[16..29]  | `DATAIN_CHANNEL_REAL` | = 真实输入通道 − 1 |
| CNA 0x1028[0..10]   | `DATAOUT_WIDTH`   | = 本 tile 输出宽度 |
| CNA 0x102c[0..21]   | `DATAOUT_ATOMICS` | = **out_w × out_h**（本 tile 的输出像素数）★ |
| CNA 0x1030          | `WEIGHT_SIZE0`    | = kw·kh·in_ch·kernels |
| CNA 0x1034          | `WEIGHT_SIZE1`    | = kw·kh·in_ch |
| CNA 0x1038[0..13]   | `WEIGHT_KERNELS`  | = align(out_ch, 2) |
| CNA 0x1038[16..20]  | `WEIGHT_HEIGHT`   | = kh |
| CNA 0x1038[24..28]  | `WEIGHT_WIDTH`    | = kw |
| CNA 0x1084[0..15]   | `FC_DATA_SIZE0.DMA_WIDTH`  | = 整层输入宽度 |
| CNA 0x1084[16..31]  | `FC_DATA_SIZE0.DMA_HEIGHT` | = **本 tile 输入高度** ★★ |
| CNA 0x1010[4..13]   | `CONV_CON2.FEATURE_GRAINS` | 上游**自己也不知道**（写死魔数 50+stride+1） |
| CORE 0x3014[0..15]  | `DATAOUT_SIZE_0.DATAOUT_WIDTH`  | = out_w − 1 |
| CORE 0x3014[16..31] | `DATAOUT_SIZE_0.DATAOUT_HEIGHT` | = out_h − 1 |
| CORE 0x3018[0..15]  | `DATAOUT_SIZE_1.DATAOUT_CHANNEL`| = out_ch(对齐) − 1 |

★★ 是关键：0x1084 与 0x1020 **同一次**写入里，宽度用整层、高度用 tile
    —— 上游代码本身就把这个不对称写死了。这独立证明「高度字段是 tile 级」。

## 三条独立自洽检查（不依赖上游）

对每个 tile：
  (A) `0x102c & 0x3fffff` 必须能被 `0x1028 & 0x7ff` 整除 ⇒ out_h
  (B) 由 (A) 得到的 out_h 必须 == `CORE 0x3014[16..31] + 1`      （两条寄存器路径）
  (C) 一个算子所有 tile 的 out_h 之和 == 该算子整层输出高度
      （整层高度可从 0x1020 的 in_h 序列与最后一层推断，见 --op）

## 用法
  python3 src/analysis/tiling.py verify [--dumps DIR]
  python3 src/analysis/tiling.py op <op_idx> [--dumps DIR]
  python3 src/analysis/tiling.py work [--dumps DIR]        # 每算子 MAC 表
"""
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from sim.rknpu import loader                                        # noqa: E402
from sim.rknpu.machine import Machine                               # noqa: E402

DEF_DUMPS = os.path.expanduser("~/align1/align1")

# ── 字段掩码（全部出自上游 kernel/rocket_registers.h）──
F_IN_W      = (0x1020, 16, 0x7ff)
F_IN_H      = (0x1020,  0, 0x7ff)
F_CH_REAL   = (0x1024, 16, 0x3fff)      # 存的是 ch_real − 1
F_CH_ALIGN  = (0x1024,  0, 0xffff)
F_OUT_W     = (0x1028,  0, 0x7ff)
F_ATOMICS   = (0x102c,  0, 0x3fffff)
F_WSIZE0    = (0x1030,  0, 0xffffffff)
F_WSIZE1    = (0x1034,  0, 0xffffffff)
F_KERNELS   = (0x1038,  0, 0x3fff)
F_KH        = (0x1038, 16, 0x1f)
F_KW        = (0x1038, 24, 0x1f)
F_GRAINS    = (0x1010,  4, 0x3ff)
F_KGROUP    = (0x1010, 16, 0xff)
F_FC_W      = (0x1084, 16, 0x3fff)   # DMA_WIDTH  在高半字（上游头文件如此）
F_FC_H      = (0x1084,  0, 0x7ff)    # DMA_HEIGHT 在低半字
F_FC_C      = (0x1088,  0, 0xffff)
F_CORE_W    = (0x3014,  0, 0xffff)
F_CORE_H    = (0x3014, 16, 0xffff)
F_CORE_C    = (0x3018,  0, 0xffff)
F_CBUF_CON0 = (0x1040,  0, 0xffff)
F_CBUF_ENT  = (0x1044,  0, 0xffff)
F_PAD0      = (0x1068,  0, 0xff)
F_CUBE_W    = (0x4030,  0, 0x1fff)
F_CUBE_H    = (0x4034,  0, 0x1fff)
F_CUBE_C    = (0x403c,  0, 0x1fff)      # CHANNEL(对齐)
F_CUBE_ORIG = (0x403c, 16, 0x1fff)      # ORIG_CHANNEL(真实−1)
F_WDMA_C    = (0x4058,  0, 0x1fff)
F_WDMA_W    = (0x405c,  0, 0x1fff)
F_WDMA_H    = (0x405c, 16, 0x1fff)


def block_of(reg):
    return {0x1: 0x0201, 0x3: 0x0801, 0x4: 0x1001,
            0x5: 0x2001, 0x6: 0x4001, 0x7: 0x8001}.get(reg >> 12)


def seg_fields(seg):
    """一个 segment 内每个 (reg) 的最后写入值 → 字段字典"""
    last = {}
    for i in seg.insns:
        last[i.reg] = i.value

    def g(spec):
        reg, sh, m = spec
        v = last.get(reg)
        if v is None:
            return None
        return (v >> sh) & m

    d = {
        "in_w": g(F_IN_W), "in_h": g(F_IN_H),
        "ch_real": (g(F_CH_REAL) + 1) if g(F_CH_REAL) is not None else None,
        "ch_align": g(F_CH_ALIGN),
        "out_w": g(F_OUT_W), "atomics": g(F_ATOMICS),
        "wsize0": g(F_WSIZE0), "wsize1": g(F_WSIZE1),
        "kernels": g(F_KERNELS), "kh": g(F_KH), "kw": g(F_KW),
        "grains": g(F_GRAINS), "kgroup": g(F_KGROUP),
        "fc_w": g(F_FC_W), "fc_h": g(F_FC_H), "fc_c": g(F_FC_C),
        "core_w": (g(F_CORE_W) + 1) if g(F_CORE_W) is not None else None,
        "core_h": (g(F_CORE_H) + 1) if g(F_CORE_H) is not None else None,
        "core_c": (g(F_CORE_C) + 1) if g(F_CORE_C) is not None else None,
        "cbuf_con0": g(F_CBUF_CON0), "cbuf_ent": g(F_CBUF_ENT),
        "pad0": g(F_PAD0),
        "cube_w": (g(F_CUBE_W) + 1) if g(F_CUBE_W) is not None else None,
        "cube_h": (g(F_CUBE_H) + 1) if g(F_CUBE_H) is not None else None,
        "cube_c": (g(F_CUBE_C) + 1) if g(F_CUBE_C) is not None else None,
        "cube_orig": (g(F_CUBE_ORIG) + 1) if g(F_CUBE_ORIG) is not None else None,
        "wdma_c": (g(F_WDMA_C) + 1) if g(F_WDMA_C) is not None else None,
        "wdma_w": (g(F_WDMA_W) + 1) if g(F_WDMA_W) is not None else None,
        "wdma_h": (g(F_WDMA_H) + 1) if g(F_WDMA_H) is not None else None,
    }
    d["out_h"] = (d["atomics"] // d["out_w"]
                  if d["atomics"] and d["out_w"] else None)
    d["atomics_exact"] = bool(d["atomics"] and d["out_w"]
                              and d["atomics"] % d["out_w"] == 0)
    return d


def load_segments(dumps=DEF_DUMPS):
    mem, tasks, _ = loader.load_corpus(dumps)
    segs = Machine(mem).run_all(tasks)
    out = []
    for s in segs:
        out.append((s, seg_fields(s)))
    return out


# ─────────────────────────── verify ───────────────────────────
def cmd_verify(dumps):
    rows = load_segments(dumps)
    print(f"语料: {dumps}")
    print(f"segment/tile 总数: {len(rows)}\n")

    checks = defaultdict(lambda: [0, 0, []])   # name -> [hit, tot, samples]

    def chk(name, ok, sample=None):
        c = checks[name]
        c[1] += 1
        if ok:
            c[0] += 1
        elif sample is not None and len(c[2]) < 5:
            c[2].append(sample)

    for s, d in rows:
        W = d["out_w"]
        # (A) atomics 整除 out_w
        if d["atomics"] is not None and W:
            chk("A. DATA_SIZE3 能被 DATA_SIZE2 整除 ⇒ out_h",
                d["atomics_exact"], (s.task.index, d["atomics"], W))
        # (B) out_h == CORE_DATAOUT_SIZE_0 高 16 位 + 1
        if d["out_h"] is not None and d["core_h"] is not None:
            chk("B. out_h(来自 atomics) == CORE_DATAOUT_SIZE_0[16..31]+1",
                d["out_h"] == d["core_h"],
                (s.task.index, d["out_h"], d["core_h"]))
        # (C) out_w 两条路径一致
        if W is not None and d["core_w"] is not None:
            chk("C. out_w(DATA_SIZE2) == CORE_DATAOUT_SIZE_0[0..15]+1",
                W == d["core_w"], (s.task.index, W, d["core_w"]))
        # (D) ★ DATAIN_HEIGHT(tile) == FEATURE_GRAINS
        if d["in_h"] is not None and d["grains"] is not None:
            chk("D. ★ DATA_SIZE0.in_h == CONV_CON2.FEATURE_GRAINS",
                d["in_h"] == d["grains"],
                (s.task.index, d["in_h"], d["grains"]))
        # (E) ★ 0x1084 高度字段也是 tile 高度（上游写死的不对称）
        if d["in_h"] is not None and d["fc_h"] is not None:
            chk("E. ★ FC_DATA_SIZE0.DMA_HEIGHT == DATA_SIZE0.in_h (tile 级)",
                d["in_h"] == d["fc_h"], (s.task.index, d["in_h"], d["fc_h"]))
        # (F) 0x1084 宽度字段 = 整层宽度（≥ tile 宽度，且通常远大于）
        if d["in_w"] is not None and d["fc_w"] is not None:
            chk("F. FC_DATA_SIZE0.DMA_WIDTH == DATA_SIZE0.in_w (整层宽)",
                d["in_w"] == d["fc_w"], (s.task.index, d["in_w"], d["fc_w"]))
        # (G) WEIGHT_SIZE1 == kw·kh·in_ch(对齐)
        if None not in (d["wsize1"], d["kw"], d["kh"], d["ch_align"]):
            chk("G. WEIGHT_SIZE1 == kw·kh·DATAIN_CHANNEL(对齐)",
                d["wsize1"] == d["kw"] * d["kh"] * d["ch_align"],
                (s.task.index, d["wsize1"], d["kw"] * d["kh"] * d["ch_align"]))
        # (H) WEIGHT_SIZE0 == WEIGHT_SIZE1 × WEIGHT_KERNELS
        if None not in (d["wsize0"], d["wsize1"], d["kernels"]):
            chk("H. WEIGHT_SIZE0 == WEIGHT_SIZE1 × WEIGHT_KERNELS",
                d["wsize0"] == d["wsize1"] * d["kernels"],
                (s.task.index, d["wsize0"], d["wsize1"] * d["kernels"]))
        # (I) DATAIN_CHANNEL_REAL == ch_real − 1（即我们存的已 +1，恒真性检查：
        #     真实通道必须 ≤ 对齐通道，且对齐到 16）
        if None not in (d["ch_real"], d["ch_align"]):
            chk("I. ch_real ≤ ch_align 且 ch_align % 16 == 0",
                d["ch_real"] <= d["ch_align"] and d["ch_align"] % 16 == 0,
                (s.task.index, d["ch_real"], d["ch_align"]))
        # (J) DPU 侧立方体尺寸与 CNA 侧一致
        if None not in (d["cube_w"], d["cube_h"], d["out_w"], d["out_h"]):
            chk("J. DPU DATA_CUBE_WIDTH/HEIGHT == CNA out_w/out_h",
                d["cube_w"] == d["out_w"] and d["cube_h"] == d["out_h"],
                (s.task.index, (d["cube_w"], d["cube_h"]), (d["out_w"], d["out_h"])))
        # (K) 输出通道：CORE_DATAOUT_SIZE_1 == DPU CUBE CHANNEL
        if None not in (d["core_c"], d["cube_c"]):
            chk("K. CORE_DATAOUT_SIZE_1 == DPU DATA_CUBE_CHANNEL(对齐)",
                d["core_c"] == d["cube_c"], (s.task.index, d["core_c"], d["cube_c"]))
        # (L) WEIGHT_KERNELS == align(out_ch_real, 2)
        if None not in (d["kernels"], d["cube_orig"]):
            exp = d["cube_orig"] + (d["cube_orig"] % 2)
            chk("L. WEIGHT_KERNELS == align(真实输出通道, 2)",
                d["kernels"] == exp, (s.task.index, d["kernels"], exp))
        # (M) 硬件必须在 out_w ≤ in_w 的假设下（卷积不放大）——弱检查
        # (M2) WDMA_SIZE_0/1 == out_ch/out_h/out_w
        if None not in (d["wdma_c"], d["core_c"]):
            chk("M1. DPU WDMA_SIZE_0 == 对齐输出通道",
                d["wdma_c"] == d["core_c"], (s.task.index, d["wdma_c"], d["core_c"]))
        if None not in (d["wdma_w"], d["wdma_h"], d["out_w"], d["out_h"]):
            chk("M2. DPU WDMA_SIZE_1 == out_h|out_w",
                d["wdma_w"] == d["out_w"] and d["wdma_h"] == d["out_h"],
                (s.task.index, (d["wdma_w"], d["wdma_h"]), (d["out_w"], d["out_h"])))
        # (N) CBUF_DATA_BANK + WEIGHT_BANK 关系
        if d["cbuf_con0"] is not None:
            db, wb = d["cbuf_con0"] & 0xf, (d["cbuf_con0"] >> 4) & 0xf
            chk("N. CBUF DATA_BANK + WEIGHT_BANK ≤ 12",
                db + wb <= 12 and db > 0 and wb > 0,
                (s.task.index, db, wb))

    order = ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J.", "K.",
             "L.", "M1", "M2", "N."]
    print(f"{'检查项':<52} {'命中':>10}  比例")
    print("─" * 90)
    for key in sorted(checks, key=lambda k: order.index(k[:2]) if k[:2] in order else 99):
        hit, tot, bad = checks[key]
        flag = "✓" if hit == tot else "✗"
        print(f"{flag} {key:<50} {hit:>5}/{tot:<5} {100.0*hit/max(tot,1):6.1f}%")
        for b in bad:
            print(f"      反例 task={b[0]}  {b[1]} vs {b[2]}")
    print()


# ─────────────────────────── op 视图 ───────────────────────────
def cmd_op(dumps, want):
    rows = load_segments(dumps)
    byop = defaultdict(list)
    for s, d in rows:
        byop[s.task.op_idx].append((s, d))
    for op in (sorted(byop) if want is None else [want]):
        tiles = byop[op]
        print(f"\n═══ op_idx {op} — {len(tiles)} 个 tile ═══")
        print(f"{'task':>6} {'in_w':>5} {'in_h':>5} {'ch_r':>5} {'ch_a':>5} "
              f"{'out_w':>6} {'out_h':>6} {'kern':>5} {'kw':>3} {'kh':>3} "
              f"{'grains':>6} {'fc_h':>5} {'core_h':>6}")
        tot_h = 0
        for s, d in tiles:
            tot_h += d["out_h"] or 0
            print(f"{s.task.index:>6} {d['in_w']:>5} {d['in_h']:>5} "
                  f"{d['ch_real']:>5} {d['ch_align']:>5} {d['out_w']:>6} "
                  f"{d['out_h']:>6} {d['kernels']:>5} {d['kw']:>3} {d['kh']:>3} "
                  f"{d['grains']:>6} {d['fc_h']:>5} {d['core_h']:>6}")
        print(f"  Σ out_h = {tot_h}")


def main(argv):
    dumps = DEF_DUMPS
    if "--dumps" in argv:
        dumps = argv[argv.index("--dumps") + 1]
    cmd = argv[1] if len(argv) > 1 else "verify"
    if cmd == "verify":
        cmd_verify(dumps)
    elif cmd == "op":
        want = int(argv[2]) if len(argv) > 2 and argv[2].isdigit() else None
        cmd_op(dumps, want)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
