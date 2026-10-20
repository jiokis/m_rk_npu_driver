"""timing.py — RK3588 NPU 结构级时序模型（仿真器阶段 ⑤）

不依赖权重布局：只需要**命令流结构**（提交数、每提交的 tile 数、tile 的 MAC 量）。
这正是「让开发不再依赖硬件试错」的直接工具 ——
去掉冗余拷贝、合并提交、改 tiling 这些实验只需要结构，不需要数值。

## 模型（每一项都有实测依据）

    t_total ≈ n_submit × FLOOR  +  MAC_total / PEAK

  · PEAK  = 1.024 T-MAC/s / 核（阵列 32×32 = 1024 MAC/cycle @1GHz）   docs/44
  · FLOOR = 每次提交的固定代价。逐模型实测 49~71 µs（docs/59）；
            全局 67.4 µs 对「细碎提交」的模型成立（docs/59）
  · n_rounds = 每核每轮一个 tile（多核时 task_number 是核数的倍数）  docs/45

## 加带宽项后的形式（docs/80 实测）

    t_total ≈ max( n_submit × FLOOR ,  traffic / BW )

  · BW = 23.84 GB/s（实测 DDR 上限，docs/55）
  · traffic = Σ(权重字节 + 输入特征 + 输出特征)
  · 这一项把**带宽受限**的模型救回来了（mnasnet −72.7%→−33.1%，mobilenet −77.6%→−15.7%）

## 适用边界（docs/55/80 已实测）

  * **同一核数内**误差 ≤7%
  * **跨核数不适用**（2 核的地板比 3 核高，docs/55 的 −31.7% 就是这个原因）
  * 对**提交地板占主导**的模型预测最好：regnet −0.2% / shufflenet +9.5% / densenet −13.4%
  * 加带宽项后**带宽受限**的模型也大幅改善（mnasnet/mobilenet）
  * ⚠️ **但仍有 4 个模型低估 35~56%**（resnet18/squeezenet/googlenet/efficientnet）
    ⇒ 还有**第三个分量未建模**。这个模型目前只在「地板主导」区间可靠。
"""
from __future__ import annotations

from dataclasses import dataclass, field

PEAK_PER_CORE = 1.024e12      # MAC/s
FLOOR_US_DEFAULT = 67.4       # µs/提交


@dataclass
class Prediction:
    n_submit: int = 0
    n_tiles: int = 0
    n_rounds: int = 0
    mac_total: float = 0.0
    t_arith_ms: float = 0.0
    t_floor_ms: float = 0.0
    t_total_ms: float = 0.0
    cores: int = 3
    floor_us: float = FLOOR_US_DEFAULT
    notes: list = field(default_factory=list)

    @property
    def floor_share(self) -> float:
        return self.t_floor_ms / self.t_total_ms if self.t_total_ms else 0.0


def predict(submits, tile_macs, cores: int = 3, floor_us: float = FLOOR_US_DEFAULT) -> Prediction:
    """submits: [(task_start, task_number), ...]；tile_macs: {tile_index: macs}
    返回结构级耗时预测。"""
    p = Prediction(cores=cores, floor_us=floor_us)
    p.n_submit = len(submits)
    peak = PEAK_PER_CORE * cores
    for start, n in submits:
        p.n_tiles += n
        # 每核每轮一个 tile ⇒ 轮数 = ceil(n / cores)
        p.n_rounds += (n + cores - 1) // cores
        for t in range(start, start + n):
            p.mac_total += tile_macs.get(t, 0.0)
    p.t_arith_ms = p.mac_total / peak * 1000.0
    p.t_floor_ms = p.n_submit * floor_us / 1000.0
    p.t_total_ms = p.t_arith_ms + p.t_floor_ms
    return p


def from_segments(segs, submits, cores: int = 3, floor_us: float = FLOOR_US_DEFAULT):
    """从 load_segments() 的结果直接预测。submits 由 timer 日志给出。"""
    macs = {}
    for s, d in segs:
        # 用 cost_model 的 tile_work 口径
        try:
            from analysis.cost_model import tile_work
        except Exception:
            from cost_model import tile_work
        macs.setdefault((s.task.seq, s.task.index), 0.0)
        macs[(s.task.seq, s.task.index)] += tile_work(d)["macs_real"]
    # 按 seq 分组
    out = []
    by_seq = {}
    for (seq, idx), m in macs.items():
        by_seq.setdefault(seq, {})[idx] = m
    for seq, lo, n in submits:
        d = by_seq.get(seq, {})
        out.append(predict([(lo, n)], d, cores=cores, floor_us=floor_us))
    return out


# =====================================================================================
# ★★★★★ 算子类边际成本模型（Phase A2/A3/C1a 实测，2026-10）
#
# 这一节是对上面「MAC 周期 + 提交地板」模型的**替代**，因为它对 SD-Turbo 这种
# 「注意力 + 高分辨率」负载误差达 1.6~19 倍。
#
# ## 实测方法（关键：不能用单算子模型）
#
# 每次 `inference()` 有 **约 50 ms 固定开销**，任何比它小的算子差异都会被淹没。
# 曾因此得出「3×3 Conv 只比纯 Mul 贵 1.4~1.7 倍」的错误结论（那其实是假象）。
# **必须用同构链的差分**：造 N=1 与 N=16 的同种算子链，`边际 = (t16 − t1) / 15`。
#
# ⚠️ 另一个坑：纯线性链会被编译器折叠（`x*1.5*0.8*…` 的 64 层链实测 NPU 算子数 = **1**）。
#    必须插入非线性算子（relu / softmax）或用**独立权重**，并**核对 IR 算子数 == N**。
#
# ## 实测参数（640ch × 64×64 = 2.62 M 元素下标定）
#
# | 算子类                    | 边际成本   | ns/元素 | 相对逐元素 |
# |---------------------------|-----------|---------|-----------|
# | `exSDPAttention`          | 490.9 ms  | 187.26  | 550×      |
# | `Conv` / `ConvAdd`        |  32.9 ms  |  12.55  |  37×      |
# | `exNorm`（GroupNorm）     |  17.3 ms  |   6.60  |  19×      |
# | 逐元素（Mul/Add/Reshape…）|   0.88 ms |   0.34  |   1×      |
#
# ## 已验证
#
# `unet_up3`（172 算子）：预测 ≈2355 ms vs 实测 2475 ms ⇒ **误差 −4.8%**
#
# ## 已知边界（不要当成万能）
#
# * 类成本只在 **640ch×64×64** 一档标定过；跨尺寸的线性性**未验证**（A1 的单算子曲线
#   在此尺寸范围内是线性的，但类成本是否同样线性尚不确定）。
# * `exSDPAttention` 的成本强烈依赖 **head 维度**：实测用 8 heads（d=80）得 187 ns/元素；
#   SD 真实配置是 `attention_head_dim=[5,10,20,20]`（up3 那层 ⇒ 32 heads）。
#   **32-head 版本在本机（7 GB 内存）导出失败，未能标定** ⇒ 用此模型预测 SD 的注意力段会偏高。
# =====================================================================================

#: 每次 `inference()` 的固定开销（A2 实测：1 个算子的模型就要 48.9 ms）
A_FIXED_MS = 50.0

#: 各类算子的边际成本（ns/元素）。键为 IR 表里的 OpType。
# ⚠️ **单位**：这里的 ns 是按 **(in_el + out_el)** 计的。
#   标定时一个算子张量是 640×64×64 = 2.62 M 元素，读写合计 5.24 M
#   ⇒ 例：Conv 边际 33.03 ms / 5.24 M = **6.30 ns/(in+out)**。
#   （一度按单张量算成 12.55，导致整段预测偏高 1.86 倍 —— 踩过。）
COST_NS_PER_ELEM: dict[str, float] = {
    # 注意力：最贵，比 Conv 贵约一个量级
    "exSDPAttention": 93.6,
    # 卷积族（Linear 在 RKNN 里也会变成 1×1 Conv）
    "Conv": 6.30, "ConvAdd": 6.30, "ConvTranspose": 6.30, "Conv2d": 6.30,
    # 归一化
    "exNorm": 3.30, "BatchNormalization": 3.30, "InstanceNorm": 3.30,
    # 逐元素 / 形状类（最便宜）
    "Mul": 0.168, "Add": 0.168, "MulRelu": 0.168, "exSwish": 0.168, "MulAdd": 0.168,
    "Reshape": 0.168, "Split": 0.168, "Concat": 0.168, "Transpose": 0.168,
    "MaxPool": 0.168, "AveragePool": 0.168, "Pad": 0.168, "Slice": 0.168,
    "Resize": 0.168, "Softmax": 0.168, "Relu": 0.168, "Clip": 0.168,
}

#: 未列出的算子类按逐元素处理
DEFAULT_NS_PER_ELEM = 0.168


def predict_part_ops(ir_rows) -> dict:
    """按算子类边际成本模型预测**单个模型**（`ir_rows` 为 `ir_table.parse_ir_table` 的结果）

    返回 dict：总预测、各算子类的耗时占比、逐类明细。
    只统计 `target == "NPU"` 的行（CPU 行不占 NPU 时间）。

    ⚠️ `ir_rows` 里的 `in_el` / `out_el` 需要由调用方从 InputShape / OutputShape 算出
       （见 `src/analysis/…` 或 `deploy/sd/…` 里的解析脚本）。
    """
    from collections import defaultdict

    by_class: dict[str, float] = defaultdict(float)
    by_class_ops: dict[str, int] = defaultdict(int)
    total_ns = 0.0
    for r in ir_rows:
        if r.get("target") != "NPU":
            continue
        op = r.get("op_type", "")
        el = int(r.get("in_el", 0)) + int(r.get("out_el", 0))
        ns = COST_NS_PER_ELEM.get(op, DEFAULT_NS_PER_ELEM) * el
        by_class[op] += ns
        by_class_ops[op] += 1
        total_ns += ns

    total_ms = A_FIXED_MS + total_ns / 1e6
    return {
        "t_total_ms": total_ms,
        "t_fixed_ms": A_FIXED_MS,
        "t_ops_ms": total_ns / 1e6,
        "share_by_class": {k: v / total_ns for k, v in
                           sorted(by_class.items(), key=lambda kv: -kv[1])} if total_ns else {},
        "ms_by_class": {k: v / 1e6 for k, v in
                        sorted(by_class.items(), key=lambda kv: -kv[1])},
        "ops_by_class": dict(by_class_ops),
        "n_npu_ops": sum(by_class_ops.values()),
    }


if __name__ == "__main__":
    # 自检：用 `unet_up3` 的真实算子构成回测（数据来自 data/analysis/npu_cost_model.txt）
    up3 = {"Conv": 37, "ConvAdd": 21, "exNorm": 18, "Reshape": 50, "Mul": 12}
    # 每个算子的元素数按 640×64×64 = 2.62 M 量级估（in+out 各一份 ⇒ ×2）
    el_per_op = int(729.9e6 / 172)   # up3 实测平均每算子元素数（in+out）
    rows = []
    for op, n in up3.items():
        rows += [{"op_type": op, "target": "NPU", "in_el": el_per_op // 2,
                  "out_el": el_per_op // 2}] * n
    r = predict_part_ops(rows)
    print(f"[timing] unet_up3 预测 {r['t_total_ms']:.0f} ms"
          f"（固定 {r['t_fixed_ms']:.0f} + 算子 {r['t_ops_ms']:.0f}），实测 2475 ms")
    print(f"[timing] 各类占比: "
          f"{ {k: f'{v*100:.0f}%' for k, v in list(r['share_by_class'].items())[:4]} }")
