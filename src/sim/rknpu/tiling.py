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

#: 实测：非 group conv 的模型，tiling-info 对象数 ≈ Conv 数 + 2
#: （那 2 个是输入/输出相关的 conv；见 docs/128 §1 的表）
TILING_INFO_OVERHEAD = 2


def split_conv_count(convs) -> int:
    """把模型里的 Conv 列表换算成**编译器里**的 Conv 算子数

    convs: 可迭代的 (group,) 或 dict 含 'group'（缺省 1）

    ⚠️ 这是**计数模型**：实测比值在普通 conv 为主的模型上是 1.02~1.10，
       在 depthwise/grouped 为主的模型上是 1.38~5.6+。偏差来源就是这里。
    """
    n = 0
    for c in convs:
        g = c["group"] if isinstance(c, dict) else c
        g = int(g or 1)
        n += g if GROUP_SPLIT_IS_PER_GROUP else 1
    return n


def tiling_info_count(convs) -> int:
    """预计的 `RKNNOperatorTilingInfo` 对象数（= 编译器里的 Conv 算子数 + 2）"""
    return split_conv_count(convs) + TILING_INFO_OVERHEAD


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

    # ② 真实模型：普通 conv 为主时，对象数 ≈ Conv + 2
    for name, convs, objs in [
        ("resnet18", [1] * 20, 22),
        ("squeezenet1_0", [1] * 26, 28),
        ("densenet121", [1] * 120, 122),
    ]:
        got = tiling_info_count(convs)
        out.append(_Check(f"{name} 对象数", got == objs, f"期望 {objs}，得 {got}"))

    # ③ tile 表单调性（两个方向）
    for cout in (16, 64, 256):
        seq = [MEASURED_TILES[(cout, h)] for h in (56, 112, 224)]
        out.append(_Check(f"cout={cout} 随 H 单调", seq == sorted(seq), f"{seq}"))
    for h in (56, 112, 224):
        seq = [MEASURED_TILES[(c, h)] for c in (16, 64, 256)]
        out.append(_Check(f"H={h} 随 cout 单调", seq == sorted(seq), f"{seq}"))

    # ④ ★ 反例守卫：MAC 相同的两个点 tile 数**不同** —— 如果谁「优化」成按 MAC 算，这条会失败
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
