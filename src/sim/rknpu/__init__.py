"""
rknpu — RK3588 NPU 的 CPU 侧模拟开发环境

一套纯软件的工具链，用于在不碰硬件的前提下做逆向、验证与回归：

    regcmd    指令编解码（8 字节三字段）+ 命令流构造器
    regmap    寄存器空间模型（153 个实测寄存器 + 语义标注）
    machine   结构级模拟器（寄存器空间 / PC 链 / 任务链）
    loader    从抓取语料载入任务与内存
    cli       命令行入口

用法：
    PYTHONPATH=src/sim python3 -m rknpu.cli verify
"""
__all__ = ["regcmd", "regmap", "machine", "loader"]
