"""
rknpu — RK3588 NPU 的 CPU 侧模拟开发环境

一套纯软件的工具链，用于在不碰硬件的前提下做逆向、验证与回归：

    regcmd    指令编解码（8 字节三字段）+ 命令流构造器
    regmap    寄存器空间模型（153 个实测寄存器 + 语义标注）
    machine   结构级模拟器（寄存器空间 / PC 链 / 任务链）
    loader    从抓取语料载入任务与内存
    weights   权重张量布局与反量化
    exec      卷积/量化的参考实现（与硬件逐字节对照）
    timing    提交/轮次的耗时模型
    optimize  提交合并与冗余搬运的分析
    tiling    **算子拆分与 tile 数模型**（从编译器内部实测回灌，见 docs/128）
    synth     合成命令流（造测试向量）
    cli       命令行入口

用法：
    PYTHONPATH=src/sim python3 -m rknpu.cli verify
"""
__all__ = ["regcmd", "regmap", "machine", "loader", "weights", "exec",
           "timing", "optimize", "tiling", "synth"]
