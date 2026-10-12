# deploy/ —— RKNN 部署链路（ONNX → .rknn → 板上跑通）

> 本目录是「**让更多模型真正跑在 NPU 上**」这条线的工具链。
> 逆向（`../src/`、`../docs/`）是手段；这里是目的。

## 一、环境（一次性）

```bash
# 1) 建 venv（隔离：rknn-toolkit2 要求 numpy<=1.26.4，系统里是 2.x）
python3 -m venv /home/dunaandone/rknn-venv

# 2) 装 rknn-toolkit2（cp312 x86_64 wheel；pip 走清华源）
pip3 --python /home/dunaandone/rknn-venv/bin/python install \
  /path/to/rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# 3) 依赖（**onnx 必须 <=1.17**：1.22 删掉了 rknn-toolkit2 还在用的 onnx.mapping）
pip3 --python /home/dunaandone/rknn-venv/bin/python install \
  "numpy==1.26.4" "onnx==1.17.0" "onnxruntime>=1.17.0" \
  "protobuf>=4.21.6,<=4.25.4" psutil "ruamel.yaml" scipy tqdm \
  opencv-python-headless fast-histogram

# 4) torch：**只转 ONNX 也必须装**（`import RKNN` 会 import torch）。
#    用 CPU-only 源，避免几个 GB 的 nvidia 依赖：
pip3 --python /home/dunaandone/rknn-venv/bin/python install \
  --index-url https://download.pytorch.org/whl/cpu "torch==2.4.0"
```

**网络实况（实测）**：`pypi.org/simple` / `github.com` / `api.github.com` / `huggingface.co` 均**不通**；
pip 走**清华源**可用；`download.pytorch.org` 可用（所以能拿预训练权重）。

## 二、用法

```bash
V=/home/dunaandone/rknn-venv/bin/python

# 1) 造 ONNX（程序化构造，不下载权重）
$V deploy/make_onnx_models.py                  # 全部 6 个
$V deploy/make_onnx_models.py --only dwsep

# 2) ONNX → RKNN（rk3588, w8a8, 合成标定集）
$V deploy/convert_rknn.py                      # 全部
$V deploy/convert_rknn.py --only tiny_cnn
$V deploy/convert_rknn.py --no-quant           # 冒烟

# 3) 上板跑
scp deploy/rknn/*.rknn orangepi@192.168.31.128:~/rknn-models/
ssh orangepi@192.168.31.128 'python3 rknn-bench.py ~/rknn-models/*.rknn'
```

## 三、踩过的四个坑（都写进代码注释了）

| # | 现象 | 原因 | 修法 |
|---|---|---|---|
| 1 | `AttributeError: module 'onnx' has no attribute 'mapping'` | onnx **1.22** 删了 `onnx.mapping`，而 rknn-toolkit2 2.3.2 还在用 | 钉 `onnx==1.17.0` |
| 2 | `Unsupport file /home/dunaandone/orange!` | **仓库路径含空格**（`orange pi`），工具链按空白切分 `dataset.txt` 的行 | 标定集放无空格路径 `/home/dunaandone/rknn-work/calib` |
| 3 | `All outputs of model are constants, The model is invalid!` | 我用 `Mul(x, 0)` 当 BN 的 scale，把整条信号链清零 —— **工具链判得没错** | scale 用 ≈1 的随机值；并加「输出是否依赖输入」自检 |
| 4 | 多层 depthwise 后输出与输入**相对**依赖只剩 `4e-6` | 固定 `std=0.1` 初始化让信号逐层衰减到浮点噪声 | 改 **He 初始化** `std=sqrt(2/fan_in)` |

**坑 3/4 的教训**：工具链报「模型是常数」时，**先信它**。
我第一反应是怀疑工具链，实际两次都是我的模型有问题 —— 所以加了自检
（`|Δ输出| / |输出| > 1e-5`）把它变成**生成阶段就拦下来**，而不是等转换时才炸。

## 四、已跑通的模型（板上实测，core_mask=7，224×224 输入，20 次）

| 模型 | 关键算子 | median | min | RKNN |
|---|---|---|---|---|
| `tiny_cnn` | Conv/Relu/MaxPool/Gemm | **0.748 ms** | 0.692 | v6 |
| `dwsep` | **DepthwiseConv** + 1×1 | **0.913 ms** | 0.784 | v6 |
| `multiscale` | 三尺度 + Concat | **1.061 ms** | 0.782 | v6 |
| `wide1x1` | 全 1×1（低算术强度） | **1.073 ms** | 0.902 | v6 |
| `bottleneck` | 1×1→3×3→1×1 + Add | **1.176 ms** | 1.040 | v6 |
| `residual` | Conv + **Add** 残差 | **1.259 ms** | 1.061 | v6 |

**6/6 全部 init + 推理成功。**

## 五、顺带结掉的旧悬案：mobilenet_ssd 为什么永远跑不起来

```
/opt/rknn-models/mobilenet_ssd.rknn   头 = RKNN v1     ← 运行时只接受 v6
/opt/rknn-models/yolov5s-640-640.rknn 头 = RKNN v6     ← 能跑
本轮新转的 6 个                        头 = RKNN v6     ← 能跑
```

v1 是 RKNPU1 / RK3568 时代（RKNN-Toolkit1）的格式。**它不是配置问题，是格式代差 ——
只能重新转换，不能修。** 这条以前只记着「永久损坏」，现在有了确切原因。
