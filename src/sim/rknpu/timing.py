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
