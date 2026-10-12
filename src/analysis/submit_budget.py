#!/usr/bin/env python3
"""
submit_budget.py — 「提交预算表」：给定命令流，给出耗时预测与它的上界

## 目标 ③ 的交付物

结构级模型（docs/41）：

    t_total ≈ n_submit · FLOOR  +  MAC_total / THROUGHPUT

两项都不是猜的：
  · `FLOOR`       = **直接在零算术量的提交上量出来的**（PPU-only 提交 74.4/75.0/88.7 µs）
  · `MAC_total`   = 从寄存器逐 tile 算出来的（tiling.py 已验证 13/15 项字段预测）
  · `THROUGHPUT`  = 上式的**反解**，并用「1 核 588-tile 单提交」独立校验

## 为什么它是**下界**而不是点估计

我们手上的 dump 有几个 tile 的寄存器不全（抓取时 slack=0 截断），
所以 `MAC_total` 是**低估**，于是反解出的 THROUGHPUT 也是**低估**。
脚本会显式打印覆盖率，绝不把不完整语料当成完整语料用。

## 用法
  python3 src/analysis/submit_budget.py                       # 默认三份语料
  python3 src/analysis/submit_budget.py --dumps DIR [--timer T]
"""
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from tiling import load_segments                                   # noqa: E402
from cost_model import tile_work, load_timer                       # noqa: E402

# 实测的提交地板（docs/40 + docs/41）：单位 µs
FLOOR_LO, FLOOR_MID, FLOOR_HI = 74.0, 80.0, 115.0
# RK3588 NPU：板上实测 NPU devfreq = 1.0 GHz（performance，1.0 GHz 已是最高档）
#   6 TOPS / 3 核 = 2 TOPS/核 = 1.024 T-MAC/s/核（1024 MAC/cycle @1GHz）
#   ⇒ 3 核峰值 3.072 T-MAC/s
PEAK_1CORE = 1.024e12
PEAK_3CORE = 3.072e12

# 每个提交的「是否做算术」由 OP_ENABLE 位图判定（见 docs/41 §5.3）：
#   0x1d = CNA|CORE|DPU|DPU_RDMA  （CNN 卷积通路，**必须**有 MAC）
#   0x0d = CNA|CORE|DPU          （**LLM GEMM 通路**：少了 DPU_RDMA —— docs/41 新观测）
#   0x18 = DPU|DPU_RDMA          （逐元素/激活，天然无 MAC）
#   0x60 = PPU|PPU_RDMA          （池化，天然无 MAC）
# ⇒ 「做算术」的判据 = **CNA 位（bit0）**，上面两种都算
CNA_BIT = 0x01

# (标签, 语料, timer 日志, 显式 hw 毫秒覆盖, 核数, 只看这些 seq)
CASES = [
    ("dumps2  cm=1  ★588-tile 单提交（锚点：地板可忽略）", "data/dumps2", None,
     14.107474, 1, {0}),
    ("align1  cm=7  (同模型 2 趟)", os.path.expanduser("~/align1/align1"),
     os.path.expanduser("~/align1/align1_timer.log"), None, 3, None),
    ("rkllm   decode（地板 >> 算术）", "data/rkllm", "/tmp/llm_timer.log", None, 1, None),
]


def summarise(dumps, timer_path=None, seqs=None):
    segs = load_segments(dumps)
    if seqs is not None:
        segs = [(s, d) for s, d in segs if s.task.seq in seqs]
    mac = 0
    mac_align = 0
    wt = 0
    n_all = n_ok = 0
    # OP_ENABLE 分类：conv 通路里「算不出 MAC 的」是真缺口；其余天然无算术
    n_conv = n_conv_ok = 0
    n_nat0 = 0
    n_trunc = 0
    by_submit_mac = defaultdict(float)
    by_submit_tiles = defaultdict(int)
    for s, d in segs:
        w = tile_work(d)
        n_all += 1
        en = s.enable.value if s.enable is not None else None
        if w["macs_real"] > 0:
            n_ok += 1
        if en is None:
            # 语料段尾被截断（slack=0 抓的）⇒ 读不到 OP_ENABLE，无法分类
            n_trunc += 1
            if w["macs_real"] > 0:
                n_conv += 1
                n_conv_ok += 1
        elif en & CNA_BIT:
            n_conv += 1
            if w["macs_real"] > 0:
                n_conv_ok += 1
        else:
            n_nat0 += 1
        mac += w["macs_real"]
        mac_align += w["macs_align"]
        wt += w["weight_bytes"]
        by_submit_mac[(s.task.seq, s.task.index)] = w["macs_real"]
        by_submit_tiles[s.task.seq] += 1

    measured_ns = None
    n_submit = len(by_submit_tiles)
    n_task_decl = None
    if timer_path and os.path.exists(timer_path):
        t = load_timer(timer_path)
        measured_ns = sum(v["hw_ns"] for v in t.values())
        n_submit = len(t)
        n_task_decl = sum(v["task_number"] for v in t.values())
    return {
        "tiles": n_all, "tiles_with_mac": n_ok,
        "cover": (n_ok / n_all) if n_all else 0.0,
        "n_conv": n_conv, "n_conv_ok": n_conv_ok, "n_nat0": n_nat0,
        "n_trunc": n_trunc,
        "mac": mac, "mac_align": mac_align, "weight_bytes": wt,
        "n_submit": n_submit, "n_task_decl": n_task_decl,
        "hw_ns": measured_ns,
    }


def report(label, dumps, timer_path, hw_ms_override=None, cores=3, seqs=None):
    if not os.path.isdir(dumps):
        print(f"\n### {label}: 语料不存在 ({dumps})，跳过")
        return
    r = summarise(dumps, timer_path, seqs)
    if hw_ms_override is not None:
        r["hw_ns"] = hw_ms_override * 1e6
    peak = PEAK_1CORE * cores
    print(f"\n{'═' * 84}")
    print(f"### {label}")
    print(f"    语料 {dumps}")
    print(f"    tile 总数 {r['tiles']}")
    if r["n_trunc"]:
        print(f"      ⚠ 段尾被截断（slack=0 抓的）的 tile：{r['n_trunc']}"
              f" —— **读不到 OP_ENABLE，无法判定是否该有算术**")
    if r["n_conv"]:
        gap = r["n_conv"] - r["n_conv_ok"]
        print(f"      · 走算术的 tile（OP_ENABLE 含 CNA 位 0x0d/0x1d）：{r['n_conv']}，"
              f"其中几何可读 {r['n_conv_ok']}"
              + (f"，**几何缺口 {gap}（{100*gap/r['n_conv']:.1f}%）**" if gap
                 else "（几何无缺口）"))
    if r["n_nat0"]:
        print(f"      · 天然无算术 tile（0x18 逐元素 / 0x60 池化）：{r['n_nat0']} —— 不是缺口")
    print(f"    Σ MAC(真实)  = {r['mac']:.4g}   （几何缺口部分算不出，这是**下界**）")
    print(f"    Σ MAC(对齐)  = {r['mac_align']:.4g}   （对齐/真实 = "
          f"{r['mac_align']/max(r['mac'],1):.4f}）")
    print(f"    Σ 权重读取字节 = {r['weight_bytes']:.4g}")
    print(f"    提交数 {r['n_submit']}"
          + (f"，声明任务数 {r['n_task_decl']}" if r['n_task_decl'] else ""))
    if r["hw_ns"] is None:
        print("    无配套 timer 日志 ⇒ 只能出预测，无法现场校验")
        return
    hw_ms = r["hw_ns"] / 1e6
    print(f"    实测 Σ hw      = {hw_ms:.3f} ms   ({r['n_submit']} 次提交，"
          f"均 {r['hw_ns']/max(r['n_submit'],1)/1000:.1f} µs/提交)")
    phys_cap = peak * (r["hw_ns"] / 1e9)      # 物理上不可能超过的 MAC 数
    thr_all = r["mac"] / (r["hw_ns"] / 1e9)
    print(f"    端到端等效吞吐 = {thr_all/1e12:.3f} T-MAC/s"
          f"   （含全部提交地板；因 MAC 是下界，此值也是下界）")
    print(f"    ★ 物理上限    = {peak/1e12:.3f} T-MAC/s × {hw_ms:.3f} ms"
          f" = {phys_cap:.4g} MAC")
    frac = r["mac"] / phys_cap
    if frac > 1.0:
        print(f"      ⚠ 可读 tile 已占物理上限的 **{100*frac:.1f}%**（>100%）—— 说明")
        print(f"        MAC 模型或峰值常数略有偏差（{100*(frac-1):.1f}%），"
              f"结论只能取「≈满速」而不是一个精确百分比。")
    else:
        print(f"      ⇒ 可读 tile 占物理上限的 {100*frac:.1f}%；"
              f"**未读到的部分最多只能承载剩下的 {100*(1-frac):.1f}%**")
        print(f"        （这是被物理夹出来的界，不是假设）")
        if frac >= 0.95:
            print(f"      ⇒ ★ **单核算术效率 ≥ {100*frac:.1f}%**："
                  f"硬件算术单元几乎满速，瓶颈不在乘法器")
    print()
    print(f"    ── 结构级模型反解（t = n_submit × FLOOR + MAC/THROUGHPUT）──")
    print(f"    {'地板假设':>10} {'地板合计(ms)':>13} {'算术时间(ms)':>13} "
          f"{'反解吞吐(T-MAC/s)':>18} {'占峰值':>10}")
    for floor in (FLOOR_LO, FLOOR_MID, FLOOR_HI):
        fl = r["n_submit"] * floor / 1000.0
        ar = hw_ms - fl
        if ar <= 0:
            print(f"    {floor:>8.0f}µs {fl:>13.3f} {ar:>13.3f} "
                  f"{'nan':>18} {'--':>10}")
            continue
        thr = r["mac"] / (ar / 1000.0) / 1e12
        print(f"    {floor:>8.0f}µs {fl:>13.3f} {ar:>13.3f} {thr:>18.3f} "
              f"{100*thr/(peak/1e12):>9.1f}%")
    print(f"    ⇒ 地板取大值会让反解吞吐**超过物理峰值**，那些取值直接排除；")
    print(f"      剩下的可行区间就是「算术效率」的真区间。")


def main(argv):
    if "--dumps" in argv:
        dumps = argv[argv.index("--dumps") + 1]
        timer = argv[argv.index("--timer") + 1] if "--timer" in argv else None
        report("自定义语料", dumps, timer, None, 3, None)
        return 0
    print(__doc__)
    for label, dumps, timer, hw, cores, seqs in CASES:
        report(label, dumps, timer, hw, cores, seqs)
    print(f"\n{'═' * 84}")
    print("结论（可复现）：")
    print("  1. 提交地板是**实测**的：零算术量提交 74.4 / 75.0 / 88.7 µs。")
    print("  2. ★ 单核 588-tile 单提交：可读几何已占**物理上限的 98%** ⇒ 单核算术效率 ≥98%，")
    print("     未被读到的 tile 最多只能承载剩下的 2%（被物理夹住，不是假设）。")
    print("  3. 一次 LLM decode 提交只做 3.44 M-MAC（形状无歧义：12px×896×320），")
    print("     单核标称峰值下只要 3.4 µs ⇒ **≥94% 的提交时间与算术量无关**。")
    print("  4. 3 核级模型：MAC 下界 / 实测 hw ⇒ 等效吞吐下界，")
    print("     扣掉地板后算术单元的效率落在 [50%, 100%] 区间（见上表）。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
