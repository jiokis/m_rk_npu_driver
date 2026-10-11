#!/usr/bin/env python3
"""
capture-run.py — 在板子上跑一次 RKNN 推理并抓取命令流

用法（板子上）：
    python3 capture-run.py <model.rknn> <core_mask> <输出目录> [标签]

core_mask: 0=自动, 1=core0, 2=core1, 4=core2, 7=三核
前置：~/rknpu_capture.so 已编译（见 docs/04）
"""
import os
import sys


def main():
    if len(sys.argv) < 4:
        print(__doc__); return 1
    model, cm, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    tag = sys.argv[4] if len(sys.argv) > 4 else f"cm{cm}"

    os.makedirs(outdir, exist_ok=True)
    # ⚠️ LD_PRELOAD 只对**新 exec 的进程**生效，在已启动的 Python 里设 os.environ 没用！
    # 必须由 shell 在启动本脚本前设好，例如：
    #   LD_PRELOAD=~/rknpu_capture.so RKNPU_DUMP_DIR=/tmp/x python3 capture-run.py ...
    so = os.path.expanduser("~/rknpu_capture.so")
    if not os.path.exists(so):
        print(f"❌ 找不到 {so}，请先编译 rknpu_capture.c"); return 1
    if so not in os.environ.get("LD_PRELOAD", ""):
        print("❌ LD_PRELOAD 未生效 —— 请在 shell 里这样启动：")
        print(f"   LD_PRELOAD={so} RKNPU_DUMP_DIR={outdir} python3 {sys.argv[0]} "
              f"{model} {cm} {outdir} {tag}")
        return 1
    os.environ["RKNPU_DUMP_DIR"] = outdir
    os.environ.setdefault("RKNPU_DUMP_SLACK", "64")

    from rknnlite.api import RKNNLite
    import numpy as np

    CMAP = {0: RKNNLite.NPU_CORE_AUTO, 1: RKNNLite.NPU_CORE_0,
            2: RKNNLite.NPU_CORE_1, 4: RKNNLite.NPU_CORE_2,
            3: RKNNLite.NPU_CORE_0_1, 7: RKNNLite.NPU_CORE_0_1_2}

    r = RKNNLite(verbose=False)
    ret = r.load_rknn(model)
    if ret != 0:
        print(f"❌ load_rknn 失败 {ret}"); return 1
    ret = r.init_runtime(core_mask=CMAP.get(cm, RKNNLite.NPU_CORE_AUTO))
    if ret != 0:
        print(f"❌ init_runtime 失败 {ret}"); return 1

    # 输入尺寸从模型元信息拿不到就按常见值；两种模型都覆盖
    shape = (1, 640, 640, 3) if "yolo" in model else (1, 300, 300, 3)
    print(f"▶ 模型 {os.path.basename(model)} · core_mask={cm} · 输入 {shape} · 标签 {tag}")
    r.inference(inputs=[np.zeros(shape, dtype=np.uint8)])
    r.release()
    print(f"✅ 完成，命令流写入 {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
