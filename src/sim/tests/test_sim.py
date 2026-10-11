#!/usr/bin/env python3
"""rknpu 模拟环境冒烟测试：python3 src/sim/tests/test_sim.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rknpu import regcmd as rc, regmap as rm, loader
from rknpu.machine import Machine, Memory

fails = []
def check(name, cond, extra=""):
    print(f"  {'✅' if cond else '❌'} {name}{('  ' + extra) if extra else ''}")
    if not cond: fails.append(name)

print("═══ rknpu 模拟环境冒烟测试 ═══\n")

# 1. 编解码往返
b = (rc.RegcmdBuilder().emit_op40().emit(0x0201, 0x1040, 0x1b)
     .emit_enable(0x1d).emit_pc_chain(0x1234, 0x10).build())
check("编解码往返一致", rc.encode(rc.decode(b)) == b, f"{len(b)} 字节")

# 2. 字段拆分正确
i = rc.decode(b)[1]
check("字段拆分", (i.block, i.reg, i.value) == (0x0201, 0x1040, 0x1b),
      f"{i.block_name} 0x{i.reg:04x}=0x{i.value:x}")

# 3. 寄存器模型
check("寄存器表非空", len(rm.REGISTERS) == 153, f"{len(rm.REGISTERS)} 个")
check("张量地址寄存器 10 个", len(rm.TENSOR_ADDR_REGS) == 10)
check("地址判定", rm.is_tensor_addr(0x0201, 0x1070) and not rm.is_tensor_addr(0x0201, 0x1040))

# 4. 内存重叠查找（曾经的 bug）
m = Memory()
m.load(0x1000, b"\x11" * 16)
m.load(0x1008, b"\x22" * 16)          # 与上一块重叠
check("重叠块取最具体的", m.read(0x1008, 4) == b"\x22" * 4)
check("未覆盖返回 None", m.read(0x9999, 4) is None)

# 5. 真实语料（存在才跑）
if os.path.exists(os.path.join(loader.DEFAULT_DUMPS, "capture.log")):
    mem, tasks, by_addr = loader.load_corpus()
    check("语料载入", len(tasks) == 615, f"{len(tasks)} task")
    mach = Machine(mem)
    segs = mach.run_all(tasks)
    check("全部段可执行", all(s.insns for s in segs), f"{mach.stats['insns']} 条指令")
    chan = sum(1 for s in segs if s.chain is not None)
    check("链指令可提取", chan >= 610, f"{chan}/615")
    check("无未知寄存器", len(mach.unknown_regs) == 0,
          f"{len(mach.unknown_regs)} 个")
else:
    print("  ⚠️ 语料不存在，跳过真实数据测试")

print()
print(f"  结果: {'✅ 全部通过' if not fails else f'❌ {len(fails)} 项失败: {fails}'}")
sys.exit(1 if fails else 0)
