#!/usr/bin/env python3
"""backtest_cost_model.py — 用 `timing.predict_part_ops()` 严格回测 SD 各段

与之前的「近似自检」不同：这里**从真实 IR 表逐算子取形状**算元素数，不做均摊估计。

## 用法
  PYTHONPATH=src /home/dunaandone/rknn-venv/bin/python src/analysis/backtest_cost_model.py
  （首次会重建 IR 日志，约 1 分钟/段，缓存在 /home/dunaandone/rknn-work/ir/）
"""
from __future__ import annotations
import os, re, sys, json
import numpy as np

CACHE = "/home/dunaandone/rknn-work/ir"
PARTS = {
    "unet_p1": 1504.5, "unet_p2": 354.0, "unet_up0": 76.6, "unet_up1": 539.1,
    "unet_up2": 771.9, "unet_up3": 2474.7, "vae512_v1": 1686.0, "vae512_v2": 7143.0,
}
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROW = re.compile(r"\]\s+(\d+)\s+(\w+)\s+(\w+)\s+(NPU|CPU)\s+(.*?)\s+(\S+)\s+(\d+)/(\d+)/(\d+)\s+(\d+)\s+(.*)$")


def dump_ir(name: str) -> str:
    """带 verbose 编译一次，把日志落到缓存（已存在则复用）"""
    os.makedirs(CACHE, exist_ok=True)
    p = f"{CACHE}/{name}.log"
    if os.path.exists(p) and os.path.getsize(p) > 10000:
        return p
    from rknn.api import RKNN
    onnx = f"{REPO}/deploy/sd/parts/{name}.onnx"
    if not os.path.exists(onnx):
        raise FileNotFoundError(onnx)
    saved = os.dup(1); f = open(p, "w")
    os.dup2(f.fileno(), 1)
    try:
        r = RKNN(verbose=True)
        r.config(target_platform="rk3588", optimization_level=3)
        r.load_onnx(model=onnx); r.build(do_quantization=False); r.release()
    finally:
        os.dup2(saved, 1); f.close()
    return p


def rows_of(path: str):
    out = []
    for line in open(path, errors="replace"):
        if "RKNN:" not in line:
            continue
        m = ROW.search(line)
        if not m:
            continue
        idx, op, dt, tgt, ins, outs, ddr, npuc, tot, rw, full = m.groups()
        if tgt != "NPU":
            continue
        el = lambda s: int(np.prod([int(x) for x in s.split(",") if x.strip()])) if s.strip() else 0
        # ★★★★★ **只算第 1 个输入（激活）与输出**，不算权重/bias！
        #   IR 表的 InputShape 里**包含权重和 bias**：
        #     `Conv (1,640,64,64),(640,640,3,3),(640) → (1,640,64,64)`
        #   一个 3×3 Conv 的权重是 3.7 M 元素，比激活(2.62 M)还大
        #   ⇒ 全算进去会把元素数高估约 1.7 倍，整段预测系统性偏高（这就是 C3 失败的主因）
        in_shapes = [mm.group(1) for mm in re.finditer(r"\(([\d,\s]+)\)", ins)]
        out_shapes = [mm.group(1) for mm in re.finditer(r"\(([\d,\s]+)\)", outs)]
        in_el = el(in_shapes[0]) if in_shapes else 0
        out_el = sum(el(x) for x in out_shapes)
        out.append({"op_type": op, "target": "NPU", "in_el": in_el, "out_el": out_el})
    return out


def main() -> int:
    sys.path.insert(0, f"{REPO}/src")
    from sim.rknpu.timing import predict_part_ops
    print(f"{'part':12s} {'算子':>5s} {'预测ms':>9s} {'实测ms':>9s} {'误差':>8s}  主要耗时类")
    errs = []
    for name, meas in PARTS.items():
        try:
            rows = rows_of(dump_ir(name))
        except Exception as e:
            print(f"{name:12s} 跳过: {type(e).__name__}: {str(e)[:50]}")
            continue
        r = predict_part_ops(rows)
        pred = r["t_total_ms"]
        err = (pred - meas) / meas * 100
        errs.append(abs(err))
        top = ", ".join(f"{k}:{v*100:.0f}%" for k, v in list(r["share_by_class"].items())[:2])
        print(f"{name:12s} {r['n_npu_ops']:5d} {pred:9.0f} {meas:9.1f} {err:+7.1f}%  {top}")
    if errs:
        ok = sum(1 for e in errs if e <= 30)
        print(f"\n★ 误差 ≤30% 的段: {ok}/{len(errs)}  中位绝对误差 {np.median(errs):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
