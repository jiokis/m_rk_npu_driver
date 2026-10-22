#!/usr/bin/env python3
"""tiled_detect.py — 1280 输入的 YOLO11 加速推理架构

## 为什么是这个架构（不是拍脑袋）

基于 docs/147 + 本轮实测的三条硬证据：

1. **原生 1280 模型超线性**：像素 4×，耗时 5.4×（27.3 → 147 ms）。
   而编译器 IR 模型的 cycle 数正好是 4.00×、DDR/NPU 比也几乎不变（0.341 vs 0.326）
   ⇒ 多出来的 1.35× 不在"算得多"和"搬得多"上。

2. **原生 1280 撞了寄存器字段上限**：转换时报
   `offset 0x4038 / 0x5048, limit 0x1fff(8191), value 0x7fc0(32704)`
   （对应 docs/146 的 stage3 DPU / stage4 DPU_RDMA 窗口）。640 没有这个报错。

3. **提交次数和核利用率与分辨率无关**：640 约 99 次提交、单核 56.6%；
   1280 约 94 次、单核 56.4%。变的是每次提交的耗时（0.28 → 1.56 ms）。

⇒ 结论：**把 1280 拆成 4 个 640 的 tile 来跑，比让模型自己吃 1280 更快，
   而且完全避开字段溢出这条暗雷。**

## 三种模式

  native   直接用 1280 模型（基线）
  tile     2×2 等分，每块走 640 模型（**主推**）
  overlap  2×2 带 15% 重叠（边界目标更稳，但每块要缩放，反而更慢）

## 用法
  python3 tiled_detect.py --mode tile --image x.jpg --size 1280
"""
import argparse
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite

CONF = 0.25
IOU = 0.45
SIZE = 640


def letterbox(img, new=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), color, dtype=np.uint8)
    top, left = (new - nh) // 2, (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def decode(out, conf=CONF):
    """YOLO11 导出模型的输出已解码为像素坐标 xywh，只需阈值过滤。

    out: (1, 84, N) —— 前 4 是 xywh，后面 80 个是类别分数
    """
    p = out[0] if isinstance(out, (list, tuple)) else out
    p = np.squeeze(p)                      # (84, N)
    p = p.T                                # (N, 84)
    scores = p[:, 4:]
    cls = scores.argmax(1)
    sc = scores.max(1)
    m = sc > conf
    if not m.any():
        return np.zeros((0, 6), dtype=np.float32)
    box = p[m, :4]
    sc = sc[m]
    cls = cls[m]
    xyxy = np.concatenate([box[:, :2] - box[:, 2:] / 2,
                           box[:, :2] + box[:, 2:] / 2], axis=1)
    return np.concatenate([xyxy, sc[:, None], cls[:, None].astype(np.float32)], axis=1)


def nms(boxes, iou=IOU):
    if len(boxes) == 0:
        return boxes
    out = []
    for c in np.unique(boxes[:, 5]):
        b = boxes[boxes[:, 5] == c]
        xywh = np.concatenate([b[:, :2], b[:, 2:4] - b[:, :2]], axis=1)
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), b[:, 4].tolist(), CONF, iou)
        if len(idx):
            out.append(b[np.array(idx).ravel()])
    return np.concatenate(out, 0) if out else boxes[:0]


class TiledDetector:
    """一个 640 模型 + 分块调度；native 模式另加载 1280 模型。"""

    def __init__(self, model640, model1280=None):
        self.r640 = RKNNLite(verbose=False)
        assert self.r640.load_rknn(model640) == 0
        assert self.r640.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0
        self.r1280 = None
        if model1280:
            self.r1280 = RKNNLite(verbose=False)
            assert self.r1280.load_rknn(model1280) == 0
            assert self.r1280.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0

    def native1280(self, img):
        lb, r, l, t = letterbox(img, 1280)
        out = self.r1280.inference(inputs=[lb[None]])
        b = decode(out)
        if len(b):
            b[:, [0, 2]] = (b[:, [0, 2]] - l) / r
            b[:, [1, 3]] = (b[:, [1, 3]] - t) / r
        return nms(b)

    def tile(self, img, n=2, overlap=0.0):
        h, w = img.shape[:2]
        th, tw = h // n, w // n
        pad_h = int(th * overlap)
        pad_w = int(tw * overlap)
        allb = []
        import time as _t
        self.t_lb = getattr(self, "t_lb", 0.0); self.t_inf = getattr(self, "t_inf", 0.0)
        self.t_dec = getattr(self, "t_dec", 0.0); self.t_nms = getattr(self, "t_nms", 0.0)
        for i in range(n):
            for j in range(n):
                y0 = max(0, i * th - pad_h)
                y1 = min(h, (i + 1) * th + pad_h)
                x0 = max(0, j * tw - pad_w)
                x1 = min(w, (j + 1) * tw + pad_w)
                _a = _t.perf_counter()
                sub = img[y0:y1, x0:x1]
                lb, r, l, t = letterbox(sub, SIZE)
                _b = _t.perf_counter(); self.t_lb += _b - _a
                out = self.r640.inference(inputs=[lb[None]])
                _c = _t.perf_counter(); self.t_inf += _c - _b
                b = decode(out)
                _d = _t.perf_counter(); self.t_dec += _d - _c
                if len(b):
                    # 回到子图坐标，再加 tile 偏移
                    b[:, [0, 2]] = (b[:, [0, 2]] - l) / r + x0
                    b[:, [1, 3]] = (b[:, [1, 3]] - t) / r + y0
                    allb.append(b)
        boxes = np.concatenate(allb, 0) if allb else np.zeros((0, 6), np.float32)
        _e = _t.perf_counter()
        r_ = nms(boxes)
        self.t_nms += _t.perf_counter() - _e
        self.n_calls = getattr(self, "n_calls", 0) + 1
        return r_

    def release(self):
        self.r640.release()
        if self.r1280:
            self.r1280.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["native", "tile", "overlap"], default="tile")
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--m640", default="/home/orangepi/y11/yolo11n_640.rknn")
    ap.add_argument("--m1280", default="/home/orangepi/y11/yolo11n_1280.rknn")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"读不到图片 {a.image}")
    img = cv2.resize(img, (a.size, a.size), interpolation=cv2.INTER_LINEAR)

    det = TiledDetector(a.m640, a.m1280)
    fn = {"native": det.native1280,
          "tile": lambda im: det.tile(im, 2, 0.0),
          "overlap": lambda im: det.tile(im, 2, 0.15)}[a.mode]
    for _ in range(3):
        b = fn(img)
    ts = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        b = fn(img)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    print(f"mode={a.mode:<8} size={a.size}  中位 {ts[len(ts)//2]:7.2f} ms  "
          f"min {ts[0]:6.2f}  p90 {ts[int(len(ts)*0.9)]:6.2f}  "
          f"FPS {1000/ts[len(ts)//2]:5.1f}   检出 {len(b)} 个")
    det.release()


if __name__ == "__main__":
    main()
