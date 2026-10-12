# 44 · CORE 的 MAC 阵列几何：32×32 = 1024 MAC/cycle（上游代码读出，实测撞实）

> 目标 ②「搞清 CORE 数据通路、流水级结构与各级缓冲」的最后一块硬骨头。
> 之前我们知道「一核约 1 T-MAC/s」，但那是**测出来的数**，不知道硬件**为什么**是这个数 ——
> 也就无法判断「还能不能更快」。
>
> 本轮从上游代码里把阵列几何**读了出来**，再用实测撞了一遍。
> 工具：`src/analysis/array_geometry.py`（产物 `data/analysis/array_geometry.txt`）

---

## 0. 结论

```
M 方向：WEIGHT_ATOMIC_SIZE = 32
K 方向：2 × FEATURE_ATOMIC_SIZE = 2 × 16 = 32
⇒ MAC 阵列 = 32 × 32 = **1024 MAC/cycle**
⇒ @ 1.0 GHz（板上实测最高档）= **1.024 T-MAC/s / 核**

实测（docs/43 §3，低开销探针）：单核 **0.981 T-MAC/s = 95.8%** 的阵列理论峰值
```

**两个完全独立的来源撞到同一个数**：一个是上游 Mesa 的权重内存布局循环，
一个是板子上 15 万次推理的实测吞吐。这条链闭合了。

---

## 1. 几何是从哪读出来的

### 1.1 常量（`mesa/rkt_ml.h`）

```c
// http://nvdla.org/hw/v1/ias/unit_description.html#convolution-buffer
#define CBUF_BANK_SIZE        32768
#define CBUF_BANKS            12
#define CBUF_ENTRIES_PER_BANK 256
#define CBUF_ENTRY_SIZE       (CBUF_BANK_SIZE / CBUF_ENTRIES_PER_BANK)   // = 128
#define FEATURE_ATOMIC_SIZE   16      // K 方向的一个「feature atom」
#define WEIGHT_ATOMIC_SIZE    32      // M 方向的一个「weight atom」
#define ATOMIC_K_SIZE         16
```

注释里那行 NVDLA 链接很关键 —— 这套原子/bank 术语是从 **NVDLA** 继承的，
与 docs/15 的「NVDLA 对照」一致。

### 1.2 权重的内存布局（`mesa/rkt_coefs.c`）

```c
for (oc1 < DIV_ROUND_UP(output_channels, WEIGHT_ATOMIC_SIZE))   // M：按 32 分组
  for (ic1 < input_channels_1)                                  // K：按 32 分组
    for (x < weights_width) for (y < weights_height)            // 卷积核抽头
      for (oc2 < MIN2(output_channels, 32))                     // M 内层 32
        for (ic2 < input_channels_2)                            // K 内层 32
```

**最内两层正好是 32×32 的连续权重块** —— 硬件一次消费 32 个输出通道 × 32 个输入通道。
`weights_size = kw · kh · out_ch · align(in_ch, 32) · 2` 里的 `align(...,32)` 与 `×2` 也都指向 32。

### 1.3 配置寄存器侧的独立呼应（`mesa/rkt_task.c`）

```c
task->input_channels  = align(MAX2(in_ch, FEATURE_ATOMIC_SIZE), FEATURE_ATOMIC_SIZE);  // 16
task->output_channels = align(MAX2(out_ch, 32), 32);                                    // 32
if (operation->depthwise) task->output_channels = align(task->output_channels, 64);     // 64
```

**对齐宽度就是阵列的原子宽度**：K 对齐到 16（一个 feature atom），M 对齐到 32（一个 weight atom），
深度卷积再对齐到 64。这就解释了为什么写配置寄存器**必须填对齐值**（docs/38 的结论）。

---

## 2. 四项可证伪预测，在 15709 个 tile 上撞

`python3 src/analysis/array_geometry.py`（四份语料：align2/cm7、align2/cm3、steady7、rkllm）：

| # | 预测 | 依据 | 命中 |
|---|---|---|---|
| P1 | `DATAIN_CHANNEL`（对齐输入通道）≡ 0 (mod 16) | `FEATURE_ATOMIC_SIZE = 16` | **15709/15709 = 100%** |
| P2 | 对齐输出通道 ≡ 0 (mod 32) | `WEIGHT_ATOMIC_SIZE = 32` | **15709/15709 = 100%** |
| P3 | `DATAIN_CHANNEL` ≡ 0 (mod 32) | K = 2 个 feature atom | 15467/15709 = 98.5% |
| P4 | 深度卷积 ⇒ 对齐输出通道 ≡ 0 (mod 64) | `rkt_task.c` 的 depthwise 分支 | **15709/15709 = 100%** |

**P3 的 242 个例外全是同一个原因**：3 通道首层补到 16 而不是 32。
`ch_align` 取值分布也印证了这点（`{896: 13383, 128: 588, 256: 575, 64: 420, 512: 277,
16: 242, 32: 172, 1024: 52}` —— 只有 16 和 32 这两个小于 64 的值）。

**四项里三项 100%、第四项的例外被完全解释** ⇒ 阵列几何不是猜的。

---

## 3. 实测撞阵列几何

| 配置 | 核数 | 实测 (T-MAC/s) | 理论峰值（核×1.024） | 占本配置峰值 | 每核 |
|---|---|---|---|---|---|
| `cm=1` | 1 | 0.981 | 1.024 | **95.8%** | 0.981 |
| `cm=3` | 2 | 1.061 | 2.048 | 51.8% | 0.530 |
| `cm=7` | 3 | 2.322 | 3.072 | 75.6% | 0.774 |

**单核 95.8%** —— 这是「阵列几何 = 32×32」最直接的实测支持：
如果阵列是 16×32 = 512 MAC/cycle，实测就会是理论值的 192%，物理上不可能。
**反过来说，这个 95.8% 也证明了 MAC 账是准的**（否则不会恰好贴到理论峰值上）。

---

## 4. CBUF 的存储几何（同一批常量推出来）

```
CBUF = 12 bank × 32768 B = 384 KB
entry = 32768 / 256 = 128 B
1 个 entry = 128 / 16 = 8 个 feature atom = 128 个通道值 = **1 个像素**（最多 128 通道）
⇒ 1 个 bank = 256 个像素
⇒ 整个 CBUF = 3072 个像素的片上特征容量
```

这解释了 `calc_input_banks()` 的算法：

```c
entries_per_slice = f(input_channels, input_width)     // 一行输入占多少个 entry
banks = DIV_ROUND_UP(entries_per_slice * input_height, CBUF_ENTRIES_PER_BANK)
```

**「一个 entry = 一个像素的 128 个通道」是 CBUF 的存储单位** —— 之前只知道
「12 个 bank、每个 32 KB」，现在知道它**按像素组织**。这也是 docs/36/42 里
`DATA_BANK + WEIGHT_BANK ≡ 12` 这条约束的物理含义：**12 个 bank 分给「像素」和「权重」两类数据。**

---

## 5. 一个上游代码里的疑点（如实记录）

`rkt_coefs.c` 分配权重缓冲时：

```c
weights_size = weights_width * weights_height * output_channels *
               align(input_channels, WEIGHT_ATOMIC_SIZE) * 2;      // ← 这个 ×2
```

但紧接着的填充循环总共只写 `kw · kh · out_ch · in_ch(对齐)` 个字节 ——
**`×2` 的那一半从来没被写过。** 可能是：
* 硬件要求权重缓冲按 2 倍对齐/双缓冲（下一 tile 的权重预取），
* 或者是从 blob 抄过来的历史遗留。

我**不能**从现有证据判断是哪种，如实记为疑点。

### 顺带：权重总量与真实缓冲的对照（**不紧，仅供参考**）

| 口径 | 字节 | 相对权重 arena (8,282,112 B) |
|---|---|---|
| Σ 每算子 `max(wsize0)`（寄存器口径） | 4,726,016 | 0.571 |
| Σ `kw·kh·out_ch·ch_align`（几何口径） | 5,704,192 | 0.689 |
| Σ 上游布局公式（含 `align(...,32)·2`） | 11,421,696 | 1.379 |
| 权重 arena（`MEM_CREATE handle=2`） | **8,282,112** | 1.000 |
| 模型文件 `yolov5s-640-640.rknn` | 8,477,315 | 1.023 |

**几何口径 5.70 MB < arena 8.28 MB < 上游布局公式 11.42 MB。**
⇒ 运行时的权重布局**不是**上游那套（没有 `align(...,32)·2` 的膨胀），
arena 里还有 bias/LUT/对齐余量。
**这一条只能算「不矛盾」，不能算「验证」** —— 我不想把它写成验证。

---

## 6. 目标 ② 现在的完成度

| 要素 | 状态 | 依据 |
|---|---|---|
| 数据通路 | ✅ | CNA→CORE→DPU→PPU；DPU 内部 `BS_*`→`BN_*`→`EW_*`→`LUT_ACCESS_*`（docs/36）；`OP_ENABLE` 四种流水组成 `0x1d/0x0d/0x18/0x60`（docs/41/43） |
| **流水级结构** | ✅ **本轮补齐** | 阵列 **32×32 = 1024 MAC/cycle**；K 原子 16、M 原子 32、depthwise 64 |
| 各级缓冲 | ✅ | CBUF 384 KB / 12 bank / entry=128 B = **1 像素×128 通道** / 共 3072 像素；bank 划分 CNN(11+1) 与 LLM(1+11) 相反；`FC_DATA_BANK` 未用（docs/36/42） |
| 流水重叠机制 | ✅（定性） | `OP_ENABLE` 一次开火多级并行；`WEIGHT_REUSE` 跨 tile 驻留（强对照 63/63）；权重流量对时间 R²=0.0004 ⇒ 搬运藏在计算后面（docs/42） |
| **stage 级重叠度**（CNA/CORE/DPU 各占多少时间） | ⛔ **缺测量手段** | 需要硬件计数器；RK3588 驱动没接线、寄存器不能盲扫（docs/29 事故）。**这是本目标唯一没做到的部分**，而且原因是**工具不可得**，不是没做 |

---

## 7. 教训（本轮新增）

1. **★ 「实测到一个数」和「知道这个数为什么是这个数」是两件事。**
   之前 1 T-MAC/s 只是个观测；现在知道它 = 32×32×1 GHz。
   差别在于：**前者不能判断还有多少空间，后者可以**（95.8% ⇒ 单核没空间了）。
2. **★ 找「常数里的物理量」。** `WEIGHT_ATOMIC_SIZE` 这种名字看起来像实现细节，
   其实是**阵列宽度**。上游代码里凡是出现「对齐到 N」的地方，N 往往就是硬件原子宽度。
3. **★ 循环嵌套顺序会泄露硬件结构。** `rkt_coefs.c` 的最内两层 `oc2 × ic2` = 32×32，
   这不是随手写的，是**按硬件一次消费的形状**排的。读代码要读**循环的次序**，
   不只是读它算了什么。
4. **不要为了让表格好看而把「不矛盾」写成「验证」。** §5 的权重总量对照
   三个口径差 2 倍，我明确写了「只能算不矛盾」。**上一轮我已经因为类似的事栽过一次
   （拿错参照物得出「39 个未定义寄存器」）。**
5. **可证伪预测要设计成「猜错就不可能出现」。** P1/P2/P4 是这种（对齐约束违反一次就露馅）；
   P3 是弱一点的（98.5%），但**把 242 个例外全部解释掉**之后，
   它反而比一个 100% 的弱预测更有说服力。
