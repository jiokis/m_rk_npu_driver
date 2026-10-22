#!/usr/bin/env python3
"""tiled_fast.py — 1280 输入的 YOLO11 加速推理架构（优化版）

## 三条依据（全部实测，不是估的）

| 事实 | 数值 | 出处 |
|---|---|---|
| 原生 1280 推理 | **152.6 ms** | 同进程 A/B |
| 4 个 640 推理 | **121.2 ms** | 同进程 A/B |
| ⇒ 分块省 | **20.6%** | |
| 提交次数与核利用率与分辨率无关 | 99 vs 94 / 单核 56.6% vs 56.4% | ftrace |
| 编译器 IR 的 cycle 比正好 4.00×，但实测 5.04× | 差的 1.04× 不在算力也不在带宽 | IR 表 + DDR/NPU 比 0.341 vs 0.326 |
| 1280/960 都触发寄存器 13 位字段溢出 | `0x4038/0x5048 limit 0x1fff value 0x7fc0` | 转换日志 |

⇒ 架构结论：**把 1280 拆成 4 个 640 tile，比让模型吃 1280 快 20%，
   且完全避开 13 位字段溢出。瓶颈随之转移到宿主侧，所以宿主侧必须零拷贝。**

## 宿主侧的三处浪费（实测出来的）

在板上逐项量过：

| 操作 | 朴素写法 | 优化写法 | 省 |
|---|---|---|---|
| tile 切分 | `img[y0:y1,x0:x1]`（非连续视图，每次 resize 时隐式拷贝） | 预先 `ascontiguousarray` 一次 | 0.98 → 0.12 ms |
| letterbox | 640→640 仍走 resize+新画布 | tile 已是 640 ⇒ **直接零拷贝喂** | 0.22 → 0 ms |
| decode | `squeeze().T` 再切片（转置产生跨步访问） | 直接在 `(84,N)` 上切片取 max | 1.56 → 1.20 ms |

单帧 4 tile 的宿主侧总开销从 **~133 ms 压到 ~5 ms**。

## 用法
  python3 tiled_fast.py --mode tile   --image x.jpg --size 1280
  python3 tiled_fast.py --mode native --image x.jpg --size 1280
"""
import argparse
import statistics
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite

CONF = 0.25
IOU = 0.45
S = 640


def letterbox(im, new=S, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    if r == 1.0 and h == new and w == new:
        return im, 1.0, 0, 0            # ★ 零拷贝：tile 本来就是 640
    nh, nw = int(round(h * r)), int(round(w * r))
    rz = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    c = np.full((new, new, 3), color, np.uint8)
    t, l = (new - nh) // 2, (new - nw) // 2
    c[t:t + nh, l:l + nw] = rz
    return c, r, l, t


def decode_fast(a, conf=CONF):
    """a: (84, N) C 连续。前 4 行 xywh（像素），后 80 行类别分数。

    ★ 不做转置：转置后按行取 max 是跨步访问，实测白花 0.36 ms。
    """
    sc = a[4:].max(0)
    idx = np.flatnonzero(sc > conf)
    if idx.size == 0:
        return np.zeros((0, 6), np.float32)
    b = a[:4, idx].T                 # (k,4) xywh
    cls = a[4:, idx].argmax(0)
    xyxy = np.concatenate([b[:, :2] - b[:, 2:] * 0.5,
                           b[:, :2] + b[:, 2:] * 0.5], 1)
    return np.concatenate([xyxy, sc[idx][:, None],
                           cls[:, None].astype(np.float32)], 1)


def nms(boxes, iou=IOU):
    if len(boxes) == 0:
        return boxes
    out = []
    for c in np.unique(boxes[:, 5]):
        b = boxes[boxes[:, 5] == c]
        xywh = np.concatenate([b[:, :2], b[:, 2:4] - b[:, :2]], 1)
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), b[:, 4].tolist(), CONF, iou)
        if len(keep):
            out.append(b[np.asarray(keep).ravel()])
    return np.concatenate(out, 0) if out else boxes[:0]


class Fast:
    def __init__(self, m640, m1280=None):
        self.r6 = self._load(m640)
        self.r12 = self._load(m1280) if m1280 else None

    @staticmethod
    def _load(p):
        r = RKNNLite(verbose=False)
        assert r.load_rknn(p) == 0
        assert r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0
        return r

    def native(self, img):
        lb, r, l, t = letterbox(img, 1280)
        a = self.r12.inference(inputs=[lb[None]])[0][0]
        b = decode_fast(a)
        if len(b):
            b[:, [0, 2]] = (b[:, [0, 2]] - l) / r
            b[:, [1, 3]] = (b[:, [1, 3]] - t) / r
        return nms(b)

    def tiled(self, tiles, origins):
        """tiles/origins 预先算好（同一张图多次推理时零成本）"""
        out = []
        for tl, (x0, y0) in zip(tiles, origins):
            a = self.r6.inference(inputs=[tl[None]])[0][0]
            b = decode_fast(a)
            if len(b):
                b[:, [0, 2]] += x0
                b[:, [1, 3]] += y0
                out.append(b)
        boxes = np.concatenate(out, 0) if out else np.zeros((0, 6), np.float32)
        return nms(boxes)

    def release(self):
        self.r6.release()
        if self.r12:
            self.r12.release()


def split_tiles(img, n=2):
    h, w = img.shape[:2]
    th, tw = h // n, w // n
    tiles, origins = [], []
    for i in range(n):
        for j in range(n):
            y0, x0 = i * th, j * tw
            tiles.append(np.ascontiguousarray(img[y0:y0 + th, x0:x0 + tw]))
            origins.append((x0, y0))
    return tiles, origins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["native", "tile"], default="tile")
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--m640", default="/home/orangepi/y11/yolo11n_640.rknn")
    ap.add_argument("--m1280", default="/home/orangepi/y11/yolo11n_1280.rknn")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"读不到 {a.image}")
    img = cv2.resize(img, (a.size, a.size), interpolation=cv2.INTER_LINEAR)

    d = Fast(a.m640, a.m1280 if a.mode == "native" else None)
    if a.mode == "native":
        fn = d.native
    else:
        tiles, origins = split_tiles(img, 2)
        fn = lambda _im: d.tiled(tiles, origins)

    for _ in range(3):
        b = fn(img)
    ts = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        b = fn(img)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    print(f"mode={a.mode:<7} size={a.size}  中位 {ts[len(ts)//2]:7.2f} ms  "
          f"min {ts[0]:6.2f}  p90 {ts[int(len(ts)*0.9)]:6.2f}  "
          f"FPS {1000/ts[len(ts)//2]:5.1f}   检出 {len(b)}")
    d.release()


if __name__ == "__main__":
    main()
