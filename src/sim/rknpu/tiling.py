#!/usr/bin/env python3
"""tiling.py — RK3588 NPU 的 **算子拆分** 与 **tile 数** 模型（从编译器内部实测回灌）

## 这个模块的边界（先读这里）

**实测到的（可信）**：

1. `RKNNOperatorTilingInfo` **一个 Conv 算子一个**（受控实验，docs/128 §1）
2. **group conv 被按组拆成 G 个 Conv** —— `group=1/2/4/8` 的单 conv 模型
   分别产生 `1/2/4/8` 个 tiling-info 对象，**1:1**（docs/128 §1）
3. tile 数随 `cout` 与 `H×W` **单调递增**（docs/128 §2 的 3×3 表）

**没测到的（不要当成模型用）**：

* tile 头部的字段语义（docs/128 §3 明确留白）
* tile 数**不是**「MAC 数除以常数」（MAC 相同的两个模型 tile 数是 9 vs 17）
* 提交数与 tile 数**没有**找到稳定关系（`data/analysis/tile_vs_submit.txt`）

⇒ 所以这里给的是**算子计数模型** + **tile 数的实测查表**，
   **不是**一个能外推的解析公式。**外推是禁止的**：表外的尺寸请实测。

## 怎么实测（复现）

  V=/home/dunaandone/rknn-venv/bin/python
  $V tools/gdb/probe.py --off 0x391510 --qw 16 --bt 0 \
      --vec-off 0x58 --vec-stride 0x1a8 --onnx <model.onnx> [--noquant]

  日志里 `★ 元素个数 = N` 就是该算子的 tile 数（对象 = 算子）。

## 为什么「按组拆开」这件事值得单独写下来

模型里有 71 个 Conv，编译器里会变成 400+ 个算子（regnet_x_400mf，docs/128 §1）。
**算子数不是模型说了算的，是编译器拆出来的** —— 这直接影响
「一个模型要发多少次提交」「每核要跑多少轮」这类估算。
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- 算子拆分

#: 一个 group conv 会被拆成 `group` 个 Conv（受控实验 1:1，docs/128 §1）
GROUP_SPLIT_IS_PER_GROUP = True

#: ⚠️ **没有常数项**（docs/131 §2 的受控反例：两个 conv 的模型 = 2 个对象）
#: docs/128~130 里那个「+2」是**过拟合** —— 它只是 6 个真实模型恰好各有
#: 1 个 GlobalAveragePool + 1 个 Gemm 凑出来的。保留这个名字只为兼容旧调用，
#: **新代码请用 `tiling_info_count()` 的逐算子分解。**
TILING_INFO_OVERHEAD = 0

#: 各算子对 tiling-info 对象数的**实测贡献**（docs/131 §1 受控实验）
#: 做法：单 conv（基线 1 个对象）上追加一个算子，看对象数变化
OP_CONTRIBUTION: dict[str, int] = {
    "Conv": 1,
    "GlobalAveragePool": 3,   # ⚠️ 56x56 输入时是 3；28x28 及更小是 2（docs/131 §1）
    "Gemm": 1,                # 经 ReplaceGemmByConv 变成 Conv
    "Relu": 0, "Add": 0, "Mul": 0, "Concat": 0,
    "MaxPool": 0, "Reshape": 0, "Transpose": 0,
}

#: GlobalAveragePool 的贡献**随输入尺寸变**（受控实测 4 个尺寸，docs/131 §1）
GAP_CONTRIBUTION_BY_SIZE: dict[int, int] = {56: 3, 28: 2, 14: 2, 7: 2}

#: ★ 但**真实模型反推出来的是 1**（resnet18 / mnasnet / mobilenet_v2 / regnet 都是
#:  `Σ拆分 + GAP + Gemm` 才等于实测）。⇒ 受控实验与真实模型**对不上**，
#:  所以这里只能给**区间** [1, 3]，不能给点值。**这个矛盾未解决（docs/131 §3）。**
GAP_CONTRIBUTION_RANGE: tuple[int, int] = (1, 3)


def gap_contribution(h: int) -> int | None:
    """GlobalAveragePool 在 H×H 输入下的贡献；表外返回 None（不外推）"""
    return GAP_CONTRIBUTION_BY_SIZE.get(int(h))


def split_factor(group: int, cin: int | None) -> int:
    """一个 Conv 会被拆成几个算子

    **规则（docs/130，受控实验定的边界）**：

        拆分倍数 = group    若 group > 1 且 cin/group >= 8
                   1        否则

    证据：
      * cin/group = 16 / 8 ⇒ 拆成 group 个（s128_g8→8, s128_g16→16, s64_g8→8）
      * cin/group = 4 / 2 / 1 ⇒ 不拆（s128_g32→1, s64_g16→1, s128_g128→1）
      * 判别实验：`cout/group = 8` 而 `cin/group = 1` ⇒ **不拆** ⇒ 条件在 cin 侧
      * 1×1 核照样拆 ⇒ 与 kernel 无关
      * **depthwise（cin/group = 1）自动落在「不拆」，不需要特例**

    ⚠️ cin 未知时无法判 `cin/group`，退化成「group > 1 就拆」——
       **这在 depthwise 上会错得离谱**（efficientnet 的 Σgroup = 9025，实测算子只有 111）。
    """
    g = int(group or 1)
    if g <= 1:
        return 1
    if cin is None:
        return g
    return g if (int(cin) // g) >= SPLIT_MIN_CIN_PER_GROUP else 1


#: 分组卷积被拆开所需的最小 `cin/group`（实测分界，docs/130 §2）
SPLIT_MIN_CIN_PER_GROUP = 8


def split_conv_count(convs) -> int:
    """把模型里的 Conv 列表换算成**编译器里**的 Conv 算子数

    convs: 可迭代项，每项可以是
        * int                 —— 只给 group
        * dict(group=, cin=)  —— 推荐（能区分 depthwise）
    """
    n = 0
    for c in convs:
        if isinstance(c, dict):
            n += split_factor(c.get("group", 1), c.get("cin"))
        else:
            n += split_factor(c, None)
    return n


def tiling_info_count(convs, gaps: int = 0, gemms: int = 0,
                      gap_h: int | None = None) -> int:
    """预计的 `RKNNOperatorTilingInfo` 对象数（**逐算子分解**，docs/131 §3）

    convs: Conv 列表（见 `split_conv_count`）
    gaps : `GlobalAveragePool` 的个数
    gemms: `Gemm` 的个数（会被 `ReplaceGemmByConv` 变成 Conv）
    gap_h: GAP 的输入边长；给了就用实测表，没给就按 3 估（56x56 的值）

    ⚠️ **这是区间模型，不是点模型**：受控实验说 GAP 贡献 2~3，
       真实模型反推出来却是 1 —— **两者矛盾且未解决**（docs/131 §3）。
       所以本函数返回 **(下界, 上界)**，调用方要按区间用。
    """
    base = split_conv_count(convs) + gemms * OP_CONTRIBUTION["Gemm"]
    if not gaps:
        return (base, base)
    if gap_h is not None:
        per = gap_contribution(gap_h)
        if per is not None:
            return (base + gaps * per, base + gaps * per)
    lo, hi = GAP_CONTRIBUTION_RANGE
    return (base + gaps * lo, base + gaps * hi)


#: ★ 受控实验：分组卷积拆分边界（docs/130 §2/§3）—— (cin, group, 实测对象数)
SPLIT_BOUNDARY: list[tuple[int, int, int]] = [
    (128,   8,   8),   # cin/group = 16  ⇒ 拆
    (128,  16,  16),   # cin/group =  8  ⇒ 拆
    (128,  32,   1),   # cin/group =  4  ⇒ 不拆
    (128,  64,   1),   # cin/group =  2  ⇒ 不拆
    (128, 128,   1),   # cin/group =  1  ⇒ 不拆（depthwise）
    (64,    8,   8),   # cin/group =  8  ⇒ 拆
    (64,   16,   1),   # cin/group =  4  ⇒ 不拆
    (64,   32,   1),   # cin/group =  2  ⇒ 不拆
    (128,   8,   8),   # 1x1 核（k=1）同样拆 ⇒ 与 kernel 无关
]

#: ★ 真实模型对账（docs/129 §1 + docs/131 §3）
#: (模型, Σ拆分, GAP 数, Gemm 数, 实测对象数)
#: 判据：实测必须落在 `tiling_info_count()` 返回的**区间**内
REAL_MODEL_CHECK: list[tuple[str, int, int, int, int]] = [
    ("regnet_x_400mf",    429,  1, 1, 431),   # 含 group ∈ {1,2,4,10,25}
    ("resnet18",           20,  1, 1,  22),
    ("densenet121",       120,  1, 1, 122),
    ("squeezenet1_0",      26,  1, 0,  28),
    ("mnasnet0_5",         52,  1, 1,  54),
    ("mobilenet_v2",       52,  1, 1,  54),
    ("googlenet",          57,  1, 1,  61),
    ("efficientnet_b0",    81, 17, 1, 113),
    # ★ 唯一落在区间外的：区间 [58,60]，实测 77 ⇒ 多 17~19 个，**来源未知**
    ("shufflenet_v2_x0_5", 56,  1, 1,  77),
]
#: 落在区间外的模型（真·未解）——与「已知残差」区分开
OUT_OF_RANGE_MODELS = {"shufflenet_v2_x0_5"}


# ---------------------------------------------------------------- tile 数

#: 实测表（单 conv，cin=64，k=3，s=1，p=1，**量化后**）：(cout, h) -> tile 数
#: 来源 docs/128 §2。**只在这 9 个点上有数据。**
MEASURED_TILES: dict[tuple[int, int], int] = {
    (16, 56): 7,   (16, 112): 17,   (16, 224): 71,
    (64, 56): 9,   (64, 112): 30,   (64, 224): 114,
    (256, 56): 21, (256, 112): 114, (256, 224): 892,
}


def tile_count(cout: int, h: int, w: int | None = None) -> int | None:
    """查实测表；**表外返回 None（不外推）**

    取方形输入（w 缺省 = h）。表外的尺寸请用 tools/gdb/probe.py 实测。
    """
    if w is not None and w != h:
        return None
    return MEASURED_TILES.get((int(cout), int(h)))


def tile_count_known() -> list[tuple[int, int, int]]:
    """返回实测点 [(cout, h, tiles), ...]，便于外部做插值/画图"""
    return [(c, h, t) for (c, h), t in sorted(MEASURED_TILES.items())]


# ---------------------------------------------------------------- 自检

@dataclass
class _Check:
    name: str
    ok: bool
    detail: str


def self_check() -> list[_Check]:
    """把「实测值」和「模型算出来的」对一遍 —— 任一不符就说明这个模块被改坏了"""
    out: list[_Check] = []

    # ① group 拆分 1:1（docs/128 §1 的四个受控点）
    for g in (1, 2, 4, 8):
        got = split_conv_count([g])
        out.append(_Check(f"group={g} 拆分后算子数", got == g, f"期望 {g}，得 {got}"))

    # ② ★ 分组卷积拆分边界（docs/130）—— 逐点复算受控实验
    #    规则：group > 1 且 cin/group >= 8 ⇒ 拆成 group 个，否则 1 个
    for cin, g, measured in SPLIT_BOUNDARY:
        # 注意：最后一项是 1x1 核的重复点（同一个 (cin,g)），用集合去重意义不大
        got = split_factor(g, cin)
        out.append(_Check(f"cin={cin} group={g} ⇒ {measured}",
                          got == measured, f"得 {got}，实测 {measured}"))
    # ★ 判别实验：条件在 cin 侧而不是 cout 侧
    out.append(_Check("cin/group=8 但 cout/group=1 ⇒ 仍拆",
                      split_factor(16, 128) == 16, f"得 {split_factor(16, 128)}"))
    out.append(_Check("cin/group=1 但 cout/group=8 ⇒ 不拆",
                      split_factor(16, 16) == 1, f"得 {split_factor(16, 16)}"))
    out.append(_Check("group=1 ⇒ 1", split_factor(1, 64) == 1, f"得 {split_factor(1, 64)}"))

    # ③ ★ 真实模型对账（区间判据，docs/131 §3）
    #    区间内的必须一直在区间内；区间外的（shufflenet）必须一直在区间外 ——
    #    这样将来有人真把它解释清楚了，这条测试会提醒「该更新文档了」
    for name, sigma, gaps, gemms, objs in REAL_MODEL_CHECK:
        lo, hi = tiling_info_count([{"group": 1, "cin": 64}] * sigma,
                                   gaps=gaps, gemms=gemms)
        inside = lo <= objs <= hi
        if name in OUT_OF_RANGE_MODELS:
            out.append(_Check(f"{name} 仍在区间外（未解）", not inside,
                              f"区间 [{lo},{hi}] vs 实测 {objs}"))
        else:
            out.append(_Check(f"{name} 落在区间内", inside,
                              f"区间 [{lo},{hi}] vs 实测 {objs}"))

    # ③ tile 表单调性（两个方向）
    for cout in (16, 64, 256):
        seq = [MEASURED_TILES[(cout, h)] for h in (56, 112, 224)]
        out.append(_Check(f"cout={cout} 随 H 单调", seq == sorted(seq), f"{seq}"))
    for h in (56, 112, 224):
        seq = [MEASURED_TILES[(c, h)] for c in (16, 64, 256)]
        out.append(_Check(f"H={h} 随 cout 单调", seq == sorted(seq), f"{seq}"))

    # ④ ★★ 逐算子贡献（docs/131 §1 受控实验）
    for op, contrib in [("Relu", 0), ("Add", 0), ("Mul", 0), ("Concat", 0),
                        ("MaxPool", 0), ("Reshape", 0), ("Transpose", 0),
                        ("GlobalAveragePool", 3), ("Gemm", 1)]:
        out.append(_Check(f"{op} 的贡献 = {contrib}",
                          OP_CONTRIBUTION[op] == contrib,
                          f"表里是 {OP_CONTRIBUTION[op]}"))

    # ⑤ ★★★ 反例守卫：「+2 常数」不存在（docs/131 §2）
    #    两个 conv 的模型实测 = 2 个对象 ⇒ 任何把 +2 加回来的改动都会让这条失败
    two = tiling_info_count([{"group": 1, "cin": 64}] * 2)
    out.append(_Check("两个 conv ⇒ 2（没有 +2 常数）", two == (2, 2), f"得 {two}"))
    out.append(_Check("TILING_INFO_OVERHEAD 必须为 0",
                      TILING_INFO_OVERHEAD == 0, f"得 {TILING_INFO_OVERHEAD}"))

    # ⑥ ★ GAP 贡献随尺寸变（实测 4 点）
    for h, c in GAP_CONTRIBUTION_BY_SIZE.items():
        out.append(_Check(f"GAP@{h}x{h} 贡献 = {c}",
                          gap_contribution(h) == c, f"得 {gap_contribution(h)}"))

    # ⑦ ★ 反例守卫：MAC 相同的两个点 tile 数**不同** —— 如果谁「优化」成按 MAC 算，这条会失败
    mac1 = 64 * 64 * 9 * 56 * 56
    mac2 = 64 * 16 * 9 * 112 * 112
    t1, t2 = MEASURED_TILES[(64, 56)], MEASURED_TILES[(16, 112)]
    out.append(_Check("MAC 相同但 tile 数不同（守卫）",
                      mac1 == mac2 and t1 != t2,
                      f"MAC {mac1}=={mac2}, tiles {t1} vs {t2}"))

    return out


if __name__ == "__main__":
    bad = 0
    for c in self_check():
        print(f"{'ok ' if c.ok else 'FAIL'}  {c.name:34s} {c.detail}")
        bad += 0 if c.ok else 1
    print(f"\n{len(self_check()) - bad}/{len(self_check())} 通过")
    raise SystemExit(1 if bad else 0)
