# rknpu — RK3588 NPU 的 CPU 侧模拟开发环境

一套**纯 Python、无第三方依赖**的工具链，用于在不碰硬件的前提下做
RK3588 NPU 的逆向、验证与回归。

## 为什么要有它

板子被实验搞挂过两次（都需要物理断电）。**CPU 模拟没有这个代价**，
而且跑得比等硬件快得多。事实证明：`regcfg_amount` 的语义、
PC 链的真实拓扑、算子边界的定义 —— 这三个关键修正**全是靠离线分析/模拟做出来的**。

## 定位：结构级

模拟器把硬件建模成「寄存器空间 + 稀疏设备内存 + 链式执行流程」，
**不计算卷积结果** —— DMA 与算子字段的语义还需 NVDLA 对照才能确定。

它已经能做的事：
- 把 615 条真实 dump 全跑一遍，验证结构理解是否自洽
- 当反汇编/汇编的宿主（trace、状态快照、编解码往返）
- 回归：改了假设后，全部样本还能不能跑通
- 未知寄存器告警（当前 0 个 → 我们的寄存器表是全的）
- hook 机制，为后续功能级建模留接口

## 用法

```bash
export PYTHONPATH=src/sim

python3 -m rknpu.cli verify      # 全量验证（615 样本，72044 条指令）
python3 -m rknpu.cli chains      # 列出重建出的算子
python3 -m rknpu.cli trace 0     # 跟踪 task 0 的完整执行
python3 -m rknpu.cli state       # 跑完后转储寄存器状态
python3 -m rknpu.cli regs        # 打印已知寄存器表
```

## 当前验证结果

```
内存块 615 · 覆盖 596 KB · task 615 个
指令执行 72044 条 · 段 615
段结束原因: chain 611 · end 4
PC 链指向 : 指向已知 task 610 · 未知 1 · 值为 0 4 · 无链指令 0
算子划分  : 615 task → 229 算子
寄存器    : 150 个不同地址被写 · 0 个未观测过的寄存器
```

## 组件

| 模块 | 作用 |
|---|---|
| `regcmd.py` | 指令编解码（8 字节三字段）+ `RegcmdBuilder` 命令流构造器 |
| `regmap.py` | 寄存器空间模型（153 个实测寄存器 + 语义标注，自动生成） |
| `machine.py` | 结构级模拟器：`Memory` / `RegisterFile` / `Task` / `Segment` / `Machine` |
| `loader.py` | 从抓取语料（`data/dumps2/`）载入 task 与内存 |
| `cli.py` | 命令行入口 |

## 关键语义（都有实证支撑）

```
指令       8 字节：word0=[reg u16][val_lo u16], word1=[val_hi u16][block u16]
段          = amount 条「体」+ 1 条 PC 链（链在 index == amount，611/611 中）
PC 链       block 0x0101 / reg 0x0010 = 下一个 task 的地址
PC 链长度   block 0x0101 / reg 0x0014 = 段长度
算子边界    连续若干个 op_idx 相同的 task（615 task → 229 算子）
使能位图    block 0x0081 / reg 0x0008 ∈ {0x18, 0x1D, 0x60}
张量地址    每流水级一个，共 10 个（见 regmap.TENSOR_ADDR_REGS）
```

## 换语料目录

```bash
RKNPU_DUMPS=/path/to/dumps python3 -m rknpu.cli verify
```
