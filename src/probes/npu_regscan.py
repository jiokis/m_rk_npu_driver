#!/usr/bin/env python3
"""
NPU 寄存器扫描器（逆向用）—— 通过 /dev/mem 直接读 NPU 寄存器，找性能计数器

背景：厂商驱动在 RK3588 上把带宽计数器"没接线"（config 里 amount_top = NULL），
      所以 `RKNPU_GET_*_RW_AMOUNT` 这套 ioctl 在 RK3588 上直接返回 0。
      但厂商在 rk3576 上用的是 0x2210/0x2234/0x2238/0x223c（top）与
      0x2410/0x2434/0x2438/0x243c（core）—— 都落在 0x2000 块，
      而 0x2000 块在 RK3588 的已知布局里是空白的。
      → 怀疑寄存器存在、只是没接上。本工具用"推理前后 diff"把它实证出来。

寄存器窗口（设备树实测）：
      core0: 0xfdab0000  core1: 0xfdac0000  core2: 0xfdad0000   各 0x10000 (64KB)

⚠ 安全须知：读未映射的设备地址会触发 external abort 直接把内核打挂。
   所以**只在设备树声明的 64KB 窗口内读**，绝不越界。

用法（需要 root）：
    sudo python3 npu_regscan.py                 # 扫默认范围，对比推理前后
    sudo python3 npu_regscan.py --iters 50
    sudo python3 npu_regscan.py --full          # 扫整个 64KB 窗口
    sudo python3 npu_regscan.py --watch 0x2234  # 持续观察某几个地址
"""
import argparse, mmap, os, struct, sys, time

CORES = {0: 0xFDAB0000, 1: 0xFDAC0000, 2: 0xFDAD0000}
WIN = 0x10000  # 64KB，设备树里声明的窗口大小

# 默认扫描范围：已知块 + 厂商在别的 SoC 上放计数器的位置
DEFAULT_RANGES = [
    (0x0000, 0x0100),   # GLOBAL / PC
    (0x1000, 0x1200),   # CNA
    (0x2000, 0x2600),   # ★ 未知块，rk3576 的 amount 在这里
    (0x3000, 0x3200),   # CORE
    (0x4000, 0x4200),   # DPU
    (0x5000, 0x5200),   # DPU_RDMA
    (0x6000, 0x6200),   # PPU
    (0x7000, 0x7200),   # PPU_RDMA
    (0x8000, 0x8100),   # ★ 未知块，老 SoC 的 amount 在这里
]


def open_win(base):
    f = os.open("/dev/mem", os.O_RDONLY | os.O_SYNC)
    m = mmap.mmap(f, WIN, mmap.MAP_SHARED, mmap.PROT_READ, offset=base)
    return f, m


def rd(m, off):
    return struct.unpack_from("<I", m, off)[0]


def snapshot(m, ranges):
    """返回 {offset: value}"""
    d = {}
    for a, b in ranges:
        for off in range(a, b, 4):
            d[off] = rd(m, off)
    return d


def make_runner(iters):
    sys.path.insert(0, "/usr/lib/python3/dist-packages")
    from rknnlite.api import RKNNLite
    import numpy as np
    MODEL = "/opt/rknn-models/yolov5s-640-640.rknn"
    r = RKNNLite()
    if r.load_rknn(MODEL) != 0:
        raise RuntimeError("load_rknn 失败")
    if r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        raise RuntimeError("init_runtime 失败")
    x = np.zeros((1, 640, 640, 3), dtype="uint8")

    def run():
        for _ in range(iters):
            r.inference(inputs=[x])
    return r, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30, help="两次快照之间跑多少次推理")
    ap.add_argument("--core", type=int, default=0, help="看哪个核心的窗口（0/1/2）")
    ap.add_argument("--full", action="store_true", help="扫整个 64KB 窗口")
    ap.add_argument("--watch", type=lambda s: int(s, 0), nargs="+",
                    help="只持续观察这些偏移")
    ap.add_argument("--top", type=int, default=40, help="最多打印多少个变化项")
    args = ap.parse_args()

    if os.geteuid() != 0:
        print("需要 root（sudo）才能读 /dev/mem", file=sys.stderr)
        return 1

    base = CORES[args.core]
    print(f"═══ NPU 寄存器扫描 · core{args.core} @ {base:#x} ═══")

    # ── 观察模式 ──
    if args.watch:
        f, m = open_win(base)
        try:
            r, run = make_runner(args.iters)
            print(f"  观察地址: {[hex(o) for o in args.watch]}")
            print(f"  基线（推理前）: {[hex(rd(m, o)) for o in args.watch]}")
            t0 = time.time(); run(); dt = time.time() - t0
            print(f"  跑完 {args.iters} 次推理（{dt:.2f}s）后: {[hex(rd(m, o)) for o in args.watch]}")
            for o in args.watch:
                print(f"    +{o:#06x}: {rd(m, o)}  ({(rd(m, o))/max(args.iters,1):.0f}/次)")
        finally:
            m.close(); os.close(f)
        return 0

    ranges = [(0, WIN)] if args.full else DEFAULT_RANGES
    total = sum(b - a for a, b in ranges) // 4
    print(f"  扫描 {len(ranges)} 段 / {total} 个寄存器")

    f, m = open_win(base)
    try:
        r, run = make_runner(args.iters)
        # 预热，避免把首次初始化的变化混进来
        run()
        before = snapshot(m, ranges)
        print(f"  基线已取（预热 {args.iters} 次）")
        t0 = time.time(); run(); dt = time.time() - t0
        after = snapshot(m, ranges)
        print(f"  再跑 {args.iters} 次推理，耗时 {dt:.2f}s")
    finally:
        m.close(); os.close(f)

    changed = [(o, before[o], after[o]) for o in before if before[o] != after[o]]
    print(f"\n  变化的寄存器: {len(changed)} 个")
    if not changed:
        print("   （没有任何寄存器变化 —— 说明这个窗口里没有计数器，或计数器需要先使能）")
        return 0

    # 按"增量"排序，增量大的更像计数器
    def delta(c):
        o, a, b = c
        return (b - a) & 0xFFFFFFFF
    changed.sort(key=delta, reverse=True)
    print(f"\n  {'偏移':>8} {'前':>12} {'后':>12} {'增量':>12}  每次推理")
    for o, a, b in changed[:args.top]:
        d = delta((o, a, b))
        print(f"  0x{o:04x} {a:>12} {b:>12} {d:>12}  {d/args.iters:>10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
