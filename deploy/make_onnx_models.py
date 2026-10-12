#!/usr/bin/env python3
"""
make_onnx_models.py — 程序化构造一组不同架构的 ONNX 模型（不依赖 torch / 不下载权重）

## 为什么自己造而不是下现成的

本机网络受限（huggingface / github API 不通），但 `download.pytorch.org` 可达。
不过更关键的是：**要「针对特定模型优化」，就需要能控制架构**。
自己造可以精确地一次只改一个变量（深度/宽度/算子组合），
从而把「命令流的差异」归因到「架构的差异」。

## 覆盖的算子组合（每一条都对应一类真实的 RKNN 编译行为）

| 模型 | 关键算子 | 想看的现象 |
|---|---|---|
| `tiny_cnn` | Conv/Relu/MaxPool/Gemm | 基线：纯卷积 + 全连接 |
| `dwsep` | **DepthwiseConv** + 1×1 Conv | 深度可分离（`WEIGHT_KERNELS=1` 那条分支，docs/41 的 4 个反例） |
| `residual` | Conv + **Add**（残差） | 逐元素通路 `OP_ENABLE=0x18` |
| `multiscale` | 三尺度输出 + Concat | 检测头结构；搬运通路（`0x60`）的重复拷贝会不会变多 |
| `bottleneck` | 1×1→3×3→1×1（C3/Bottleneck） | 通道数剧烈变化时的分块 |
| `wide1x1` | 全 1×1 Conv | 算术强度低、权重主导（对比 LLM GEMM 的 `0x0d`） |

## 用法
  /home/dunaandone/rknn-venv/bin/python deploy/make_onnx_models.py [--outdir DIR] [--size N]
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 13


class Builder:
    """极简 ONNX 图构造器：只做本文件需要的那些算子"""

    def __init__(self, name, size=224, in_ch=3):
        self.name = name
        self.size = size
        self.in_ch = in_ch
        self.nodes = []
        self.inits = []
        self.n = 0
        self.rng = np.random.default_rng(20261012)

    # ── 工具 ──
    def _name(self, p):
        self.n += 1
        return f"{p}{self.n}"

    def _init(self, shape, name=None, kind="weight", std=None):
        name = name or self._name("w")
        if kind == "weight":
            # ★ He 初始化：std = sqrt(2/fan_in)。
            #   固定 std=0.1 时，多层 depthwise 会把信号衰减到浮点噪声（相对依赖 4e-6），
            #   工具链会判定「输出是常数」。用 He 让各层激活方差大致守恒。
            a = self.rng.normal(0, std if std else 0.1, size=shape).astype(np.float32)
        elif kind == "scale":
            # ⚠️ 不能用 0！Mul(x,0) 会把整条信号链清零，
            #    工具链会（正确地）判定「输出是常数、模型无效」。踩过一次。
            a = (1.0 + self.rng.normal(0, 0.05, size=shape)).astype(np.float32)
        else:   # "const" = 偏置，接近 0 但非零
            a = self.rng.normal(0, 0.02, size=shape).astype(np.float32)
        self.inits.append(numpy_helper.from_array(a, name))
        return name

    # ── 算子 ──
    def conv(self, x, out_ch, k=3, s=1, p=None, g=1, in_ch=None):
        p = k // 2 if p is None else p
        if in_ch is None:
            raise ValueError("需要 in_ch")
        fan_in = (k * k * in_ch) // g
        w = self._init([out_ch, in_ch // g, k, k], std=(2.0 / max(fan_in, 1)) ** 0.5)
        y = self._name("conv")
        self.nodes.append(helper.make_node(
            "Conv", [x, w], [y], kernel_shape=[k, k], strides=[s, s],
            pads=[p, p, p, p], group=g, name=y))
        return y

    def bn_relu(self, x, ch):
        """用 Mul+Add 模拟 BN（避免 BN 在量化时被折叠掉，保留逐元素通路）"""
        # ⚠️ ONNX 广播从右对齐：NCHW 输入必须用 [ch,1,1]，用 [ch] 会撞到 W 维
        sc = self._init([ch, 1, 1], kind="scale")   # ≈1，不能是 0
        bi = self._init([ch, 1, 1], kind="const")   # ≈0
        a = self._name("mul")
        self.nodes.append(helper.make_node("Mul", [x, sc], [a], name=a))
        b = self._name("add")
        self.nodes.append(helper.make_node("Add", [a, bi], [b], name=b))
        r = self._name("relu")
        self.nodes.append(helper.make_node("Relu", [b], [r], name=r))
        return r

    def maxpool(self, x, k=2, s=2):
        y = self._name("pool")
        self.nodes.append(helper.make_node(
            "MaxPool", [x], [y], kernel_shape=[k, k], strides=[s, s], name=y))
        return y

    def avgpool(self, x, k=2, s=2):
        y = self._name("avgpool")
        self.nodes.append(helper.make_node(
            "AveragePool", [x], [y], kernel_shape=[k, k], strides=[s, s], name=y))
        return y

    def add(self, a, b):
        y = self._name("res")
        self.nodes.append(helper.make_node("Add", [a, b], [y], name=y))
        return y

    def concat(self, xs, axis=1):
        y = self._name("cat")
        self.nodes.append(helper.make_node("Concat", xs, [y], axis=axis, name=y))
        return y

    def gemm(self, x, out_f, in_f):
        w = self._init([out_f, in_f], std=(2.0 / max(in_f, 1)) ** 0.5)
        b = self._init([out_f], kind="const")
        y = self._name("fc")
        self.nodes.append(helper.make_node("Gemm", [x, w, b], [y],
                                           transB=1, name=y))
        return y

    def flatten(self, x):
        y = self._name("flat")
        self.nodes.append(helper.make_node("Flatten", [x], [y], axis=1, name=y))
        return y

    def global_avgpool(self, x):
        y = self._name("gap")
        self.nodes.append(helper.make_node("GlobalAveragePool", [x], [y], name=y))
        return y

    def softmax(self, x, axis=-1):
        y = self._name("sm")
        self.nodes.append(helper.make_node("Softmax", [x], [y], axis=axis, name=y))
        return y

    def matmul(self, a, b):
        y = self._name("mm")
        self.nodes.append(helper.make_node("MatMul", [a, b], [y], name=y))
        return y

    def transpose(self, x, perm):
        y = self._name("tp")
        self.nodes.append(helper.make_node("Transpose", [x], [y], perm=perm, name=y))
        return y

    def reshape(self, x, shape):
        sh = numpy_helper.from_array(np.array(shape, dtype=np.int64), self._name("shp"))
        self.inits.append(sh)
        y = self._name("rs")
        self.nodes.append(helper.make_node("Reshape", [x, sh], [y], name=y))
        return y

    def mul_broadcast(self, a, b):
        y = self._name("se_mul")
        self.nodes.append(helper.make_node("Mul", [a, b], [y], name=y))
        return y

    def sigmoid(self, x):
        y = self._name("sg")
        self.nodes.append(helper.make_node("Sigmoid", [x], [y], name=y))
        return y

    # ── 收尾 ──
    def finish(self, x, out_shape, out_name="output"):
        info = helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, out_shape)
        # 图输出必须是某个节点的输出 —— 用一个 Identity 收口
        self.nodes.append(helper.make_node("Identity", [x], [out_name],
                                           name=self._name("ident")))
        g = helper.make_graph(self.nodes, self.name, [
            helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                          [1, self.in_ch, self.size, self.size])
        ], [info], self.inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
        m.ir_version = 8
        onnx.checker.check_model(m)
        return m


# ───────────────────── 各架构 ─────────────────────
def build_tiny_cnn(size):
    b = Builder("tiny_cnn", size)
    x = b.conv("input", 16, 3, 2, in_ch=3)
    x = b.bn_relu(x, 16)
    x = b.maxpool(x)
    x = b.conv(x, 32, 3, 1, in_ch=16)
    x = b.bn_relu(x, 32)
    x = b.maxpool(x)
    x = b.conv(x, 64, 3, 1, in_ch=32)
    x = b.bn_relu(x, 64)
    x = b.global_avgpool(x)
    x = b.flatten(x)
    x = b.gemm(x, 10, 64)
    return b.finish(x, [1, 10])


def build_dwsep(size):
    """深度可分离：大量 WEIGHT_KERNELS==1 的 depthwise（docs/41 的 4 个反例就是这一类）"""
    b = Builder("dwsep", size)
    x = b.conv("input", 16, 3, 2, in_ch=3)
    x = b.bn_relu(x, 16)
    ch = 16
    for out_ch in (32, 64, 64, 128, 128):
        x = b.conv(x, ch, 3, 1, g=ch, in_ch=ch)     # depthwise
        x = b.bn_relu(x, ch)
        x = b.conv(x, out_ch, 1, 1, in_ch=ch)       # pointwise
        x = b.bn_relu(x, out_ch)
        ch = out_ch
        if out_ch in (32, 64, 128):
            x = b.maxpool(x)
    x = b.global_avgpool(x)
    x = b.flatten(x)
    x = b.gemm(x, 10, ch)
    return b.finish(x, [1, 10])


def build_residual(size):
    """残差：引入 Add（逐元素通路 OP_ENABLE=0x18）"""
    b = Builder("residual", size)
    x = b.conv("input", 32, 3, 2, in_ch=3)
    x = b.bn_relu(x, 32)
    for _ in range(4):
        y = b.conv(x, 32, 3, 1, in_ch=32)
        y = b.bn_relu(y, 32)
        y = b.conv(y, 32, 3, 1, in_ch=32)
        y = b.bn_relu(y, 32)
        x = b.add(x, y)
    x = b.maxpool(x)
    x = b.conv(x, 64, 3, 1, in_ch=32)
    x = b.bn_relu(x, 64)
    x = b.global_avgpool(x)
    x = b.flatten(x)
    x = b.gemm(x, 10, 64)
    return b.finish(x, [1, 10])


def build_multiscale(size):
    """三尺度输出 + Concat：检测头结构，看搬运通路（0x60）的重复拷贝是否变多

    三个分支必须落到**同一空间尺度**才能 Concat：
      p3(112) --两次 stride2--> 28
      p4( 56) --一次 stride2--> 28
      p5( 28) ------------------> 28
    """
    b = Builder("multiscale", size)
    x = b.conv("input", 16, 3, 2, in_ch=3)          # 112
    x = b.bn_relu(x, 16)
    p3 = b.conv(x, 32, 3, 1, in_ch=16)              # 112
    p3 = b.bn_relu(p3, 32)
    d = b.maxpool(p3)                               # 56
    p4 = b.conv(d, 64, 3, 1, in_ch=32)
    p4 = b.bn_relu(p4, 64)                          # 56
    d = b.maxpool(p4)                               # 28
    p5 = b.conv(d, 128, 3, 1, in_ch=64)
    p5 = b.bn_relu(p5, 128)                         # 28
    # 三个尺度的分类分支
    o3 = b.conv(p3, 18, 1, 1, in_ch=32)             # 112
    o3 = b.conv(o3, 18, 3, 2, in_ch=18)             # 56
    o3 = b.conv(o3, 18, 3, 2, in_ch=18)             # 28  ✔
    o4 = b.conv(p4, 18, 1, 1, in_ch=64)             # 56
    o4 = b.conv(o4, 18, 3, 2, in_ch=18)             # 28  ✔
    o5 = b.conv(p5, 18, 1, 1, in_ch=128)            # 28  ✔
    cat = b.concat([o3, o4, o5], axis=1)            # 28, 54ch
    cat = b.bn_relu(cat, 54)
    x = b.global_avgpool(cat)
    x = b.flatten(x)
    x = b.gemm(x, 10, 54)
    return b.finish(x, [1, 10])


def build_bottleneck(size):
    """C3/Bottleneck 风格：1×1 → 3×3 → 1×1，通道剧烈变化

    ⚠️ 残差的 Add 要求**两侧通道数相同** —— 第一块把 32 抬到 64 之后，
       后续块必须按 64 进 64 出，否则 shape inference 直接报错。踩过一次。
    """
    b = Builder("bottleneck", size)
    x = b.conv("input", 32, 3, 2, in_ch=3)          # 112
    x = b.bn_relu(x, 32)
    # 第一块：32 -> 64（没有残差，因为通道变了）
    y = b.conv(x, 32, 1, 1, in_ch=32)
    y = b.bn_relu(y, 32)
    y = b.conv(y, 32, 3, 1, in_ch=32)
    y = b.bn_relu(y, 32)
    y = b.conv(y, 64, 1, 1, in_ch=32)
    y = b.bn_relu(y, 64)
    x = b.maxpool(y)                                 # 56, 64ch
    # 后续两块：64 -> 64，带残差
    for c_mid in (64, 128):
        y = b.conv(x, c_mid, 1, 1, in_ch=64)
        y = b.bn_relu(y, c_mid)
        y = b.conv(y, c_mid, 3, 1, in_ch=c_mid)
        y = b.bn_relu(y, c_mid)
        y = b.conv(y, 64, 1, 1, in_ch=c_mid)
        y = b.bn_relu(y, 64)
        x = b.add(x, y)                              # 两侧都是 64ch ✔
    x = b.global_avgpool(x)
    x = b.flatten(x)
    x = b.gemm(x, 10, 64)
    return b.finish(x, [1, 10])


def build_wide1x1(size):
    """全 1×1：算术强度低、权重主导（对比 LLM GEMM 的 0x0d 通路）"""
    b = Builder("wide1x1", size)
    # 用 stride=4 的 1×1 先把空间降到 56×56：
    # 否则 GlobalAveragePool 在 224×224 上会把输入差异平均掉（相对依赖只剩 ~1e-3）。
    x = b.conv("input", 64, 1, 4, in_ch=3)
    x = b.bn_relu(x, 64)
    prev = 64
    for ch in (128, 256, 256, 512):
        x = b.conv(x, ch, 1, 1, in_ch=prev)     # 显式跟踪上一层的通道数
        x = b.bn_relu(x, ch)
        prev = ch
    x = b.global_avgpool(x)
    x = b.flatten(x)
    x = b.gemm(x, 10, 512)
    return b.finish(x, [1, 10])



# ─────────────── 受控 A/B：SE 块（全局池化 + 广播乘）是不是提交切分的驱动 ───────────────
def _ab_common(b, size, se):
    """同一骨干，唯一变量是 se=True/False"""
    x = b.conv("input", 32, 3, 2, in_ch=3)
    x = b.bn_relu(x, 32)
    ch = 32
    for i, out_ch in enumerate((64, 128, 256, 256, 256, 256)):
        x = b.conv(x, out_ch, 3, 1, in_ch=ch)
        x = b.bn_relu(x, out_ch)
        ch = out_ch
        if se:
            # SE：全局平均池化 -> Sigmoid -> 与特征图逐通道相乘
            g = b.global_avgpool(x)              # [1,ch,1,1]
            g = b.conv(g, ch, 1, 1, in_ch=ch)    # 用 1x1 conv 代替 FC（同语义，便于导出）
            g = b.sigmoid(g)
            x = b.mul_broadcast(x, g)            # Mul
        if i in (1, 3, 5):
            x = b.maxpool(x)
    x = b.global_avgpool(x)
    x = b.flatten(x)
    x = b.gemm(x, 10, ch)
    return b.finish(x, [1, 10])


def build_ab_se(size):
    return _ab_common(Builder("ab_se", size), size, True)


def build_ab_nose(size):
    return _ab_common(Builder("ab_nose", size), size, False)


# ─────────── 受控 A/B 第二轮：分组卷积 / 残差 Add 是不是提交切分的驱动 ───────────
def _ab2_common(b, size, group=1, add=False):
    """同一骨干，变量：group（分组卷积）与 add（残差 Add）。

    通道序列取 (64,64,64,128,128,128)，保证 add 的成对通道一致。
    """
    x = b.conv("input", 32, 3, 2, in_ch=3)          # 首层 group 固定为 1（输入只有 3 通道）
    x = b.bn_relu(x, 32)
    ch = 32
    for i, out_ch in enumerate((64, 64, 64, 128, 128, 128)):
        y = b.conv(x, out_ch, 3, 1, in_ch=ch, g=group)
        y = b.bn_relu(y, out_ch)
        x = b.add(x, y) if (add and out_ch == ch) else y
        ch = out_ch
        if i in (1, 3, 5):
            x = b.maxpool(x)
    x = b.global_avgpool(x)
    x = b.flatten(x)
    x = b.gemm(x, 10, ch)
    return b.finish(x, [1, 10])


def build_ab_group(size):
    return _ab2_common(Builder("ab_group", size), size, group=8, add=False)


def build_ab_nogroup(size):
    return _ab2_common(Builder("ab_nogroup", size), size, group=1, add=False)


def build_ab_add(size):
    return _ab2_common(Builder("ab_add", size), size, group=1, add=True)


# ─────────── 受控 A/B 第三轮：Concat 是不是提交切分的驱动 ───────────
def _ab3_common(b, size, use_concat):
    """同一骨干，唯一变量：两分支合并处用 Concat 还是 Add。

    两种情形合并后都接一个 1×1 conv 把通道收回 c，
    所以「合并算子」是唯一的结构差异。
    """
    x = b.conv("input", 32, 3, 2, in_ch=3)
    x = b.bn_relu(x, 32)
    ch = 32
    for i, c in enumerate((64, 64, 128, 128, 128, 128)):
        a = b.conv(x, c, 3, 1, in_ch=ch)
        a = b.bn_relu(a, c)
        bb = b.conv(x, c, 3, 1, in_ch=ch)
        bb = b.bn_relu(bb, c)
        if use_concat:
            m = b.concat([a, bb], axis=1)        # 2c
            m = b.conv(m, c, 1, 1, in_ch=2 * c)  # 收回 c
        else:
            m = b.add(a, bb)                     # c
            m = b.conv(m, c, 1, 1, in_ch=c)      # 收回 c（保持与上面同构）
        x = b.bn_relu(m, c)
        ch = c
        if i in (1, 3, 5):
            x = b.maxpool(x)
    x = b.global_avgpool(x)
    x = b.flatten(x)
    x = b.gemm(x, 10, ch)
    return b.finish(x, [1, 10])


def build_ab_cat(size):
    return _ab3_common(Builder("ab_cat", size), size, True)


def build_ab_nocat(size):
    return _ab3_common(Builder("ab_nocat", size), size, False)

BUILDERS = {
    "tiny_cnn": build_tiny_cnn,
    "dwsep": build_dwsep,
    "residual": build_residual,
    "multiscale": build_multiscale,
    "bottleneck": build_bottleneck,
    "wide1x1": build_wide1x1,
    "ab_se": build_ab_se,
    "ab_nose": build_ab_nose,
    "ab_group": build_ab_group,
    "ab_nogroup": build_ab_nogroup,
    "ab_add": build_ab_add,
    "ab_cat": build_ab_cat,
    "ab_nocat": build_ab_nocat,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "onnx"))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for name, fn in BUILDERS.items():
        if a.only and name != a.only:
            continue
        m = fn(a.size)
        p = os.path.join(a.outdir, f"{name}.onnx")
        onnx.save(m, p)
        sz = os.path.getsize(p)
        nnode = len(m.graph.node)
        # ★ 自检：输出必须真的随输入变化，否则工具链会（正确地）拒绝
        import onnxruntime as ort
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        rng = np.random.default_rng(1)
        o1 = sess.run(None, {iname: rng.random((1, 3, a.size, a.size), dtype=np.float32)})[0]
        o2 = sess.run(None, {iname: rng.random((1, 3, a.size, a.size), dtype=np.float32)})[0]
        dep = float(np.abs(o1 - o2).max())
        rel = dep / max(float(np.abs(o1).max()), 1e-30)
        # ★ 用**相对**依赖判据：|Δ| 很小但输出本身也很小时，绝对判据会误判
        if dep == 0.0:
            flag = "❌结构性常数（工具链必拒）"
        elif rel < 1e-5:
            flag = "⚠️依赖极弱（可能被判常数）"
        else:
            flag = "✅依赖输入"
        print(f"  {name:<12} {nnode:>4} 节点 {sz/1024:>7.1f}KB "
              f"|Δ|={dep:>9.2e} 相对={rel:>9.2e} {flag}")


if __name__ == "__main__":
    main()
