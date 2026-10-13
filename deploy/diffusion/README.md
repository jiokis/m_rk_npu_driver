# 在 RK3588 NPU 上跑扩散模型生图

## 这套东西做了什么

一个**能在 RK3588 NPU 上采样出图**的 DDPM：UNet 走 NPU（w8a8 量化），
采样调度器走 CPU（纯 numpy），板子上只要 `rknnlite` + `numpy`。

## 文件

| 文件 | 作用 |
|---|---|
| `unet.py` | **为 RKNN 设计**的极小 UNet（静态形状、时间步当张量喂、可切 norm） |
| `train_ddpm.py` | CPU 上训练 DDPM（MNIST pad 28→32） |
| `to_rknn.py` | ckpt → ONNX → **.rknn**（含**真实分布**的多输入标定集） |
| `sample.py` | 采样循环，**UNet 可切 torch(CPU) / rknn(NPU)** |
| `check_numeric.py` | **先用同一输入对齐单次前向**，再谈采样 |

## 为什么架构要这样设计（每条都对应一次实测）

| 设计 | 依据 |
|---|---|
| **静态形状** `x[1,1,32,32]` + `temb[1,D,1,1]` | RKNN 只吃静态形状 |
| **时间步嵌入由 CPU 算好当张量喂** | 省掉 MLP/MatMul（RKNN 上又慢又不一定支持）；CPU 侧算 128 维正弦嵌入几乎免费 |
| **`norm=batch`**（默认） | 导出时**折进 Conv**：实测 `.rknn` 与 `norm=none` **大小完全相同**（5,687,822 B）⇒ 推理零成本 |
| **不用 `norm=group`** | IR 表实测会多出 **54 个 `Reshape`**（GroupNorm → `exNorm` + 2 个 Reshape/个）⇒ 慢 |
| **不做 attention** | Softmax/MatMul 在 NPU 上又慢又不一定支持 |
| **激活用 SiLU** | IR 表里是 `exSwish`，且能被融进 Conv（`ConvAdd`/`exSwish` 同层） |
| **预测 ε** | 输出范围有界，量化友好 |

## 接口约定（踩过的坑，都在这里）

1. **ONNX 要 NCHW + 静态形状**（工具链按 NCHW 读；NHWC 会被读成 H）
2. **标定集路径不能含空格** —— 工具链按空白切分 `dataset.txt` 每一行。
   本仓库路径是 `orange pi`（含空格）⇒ 标定集强制写到 `/home/dunaandone/rknn-work/diffcalib`
3. **多输入模型的标定集**：`dataset.txt` **每行要给每个输入一个文件**（空格分隔）。
   只给一个会报 `The input num: 1 ... not match input num: 2 in model!`
4. **非 3 通道模型不能用图片标定** ⇒ 全部用 `.npy`
5. **标定分布要匹配真实输入**：扩散模型的输入分布随 t 剧烈变化
   ⇒ 用**真数据 + 随机 t** 算出真实的 `x_t` 与 `temb`（不是随机噪声）
6. **RKNN 模型 batch 固定为 1** ⇒ 采样时**逐张喂**（喂 16 张只会返回第一张的 1024 个元素）

## 采样为什么用 DDIM(η=0) 而不是 DDPM 祖先采样

* **跳步采样时祖先采样的方差公式要用「下一个实际采样时刻」的 `abar`**，
  写错（用 `abar[i-1]`）会直接产出 **NaN**（踩过）
* 祖先采样每步注入随机噪声 ⇒ **torch 与 NPU 无法逐点对比**
* DDIM(η=0) 确定性 ⇒ 同种子两侧应逐点一致，**差异只剩量化误差**

## 复现

```bash
V=/home/dunaandone/rknn-venv/bin/python
# 1) 训练（CPU，~2.3 h）
$V deploy/diffusion/train_ddpm.py --steps 20000 --base 16 --norm batch \
     --out deploy/diffusion/ckpt/ddpm_mnist.pt
# 2) 导出 + 转 rknn
$V deploy/diffusion/to_rknn.py --ckpt deploy/diffusion/ckpt/ddpm_mnist.pt \
     --out deploy/diffusion/rknn/ddpm_mnist.rknn --calib-n 32
# 3) 单次前向对齐（笔记本 ↔ 板子）
$V deploy/diffusion/check_numeric.py --make-probe /tmp/probe.npz
$V deploy/diffusion/check_numeric.py --backend torch --ckpt <ckpt> --in /tmp/probe.npz --out /tmp/a.npy
#    板子：python3 check_numeric.py --backend rknn --model x.rknn --in /tmp/probe.npz --out /tmp/b.npy
$V deploy/diffusion/check_numeric.py --compare /tmp/a.npy /tmp/b.npy
# 4) 板子上采样出图
#    python3 sample.py --backend rknn --model ddpm_mnist.rknn --meta ddpm_mnist.json \
#        --n 64 --steps 100 --out /tmp/npu_grid.png
```
