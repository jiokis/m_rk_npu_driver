#!/usr/bin/env python3
"""pipeline1280.py — 1280 输入的完整可用链路

    1280 图 → 4 块 640 → 6 worker（3 ctx 各绑一核）→ 640 双输出 INT8 模型
           → 每 tile 算 DFL（只对候选锚点）→ 坐标回映 → 跨 tile NMS

修复了 docs/152 那个「INT8 把分类分数打成 0」的问题（用双输出模型）。
"""
import argparse
import glob
import queue
import statistics
import threading
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite

MASK = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
AR16 = np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)
CONF, IOU = 0.25, 0.45


def letterbox(im, new=640):
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    rz = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if nh == new and nw == new:
        return rz, 1.0, 0, 0
    c = np.full((new, new, 3), 114, np.uint8)
    t, l = (new - nh) // 2, (new - nw) // 2
    c[t:t + nh, l:l + nw] = rz
    return c, r, l, t


def dfl_candidates(logits, cls, anchor, stride, conf=CONF):
    """只对 score>conf 的锚点算 DFL"""
    sc = cls[0].max(0)
    idx = np.flatnonzero(sc > conf)
    if idx.size == 0:
        return np.zeros((0, 6), np.float32)
    L = logits[0][:, idx]                      # (64, k)
    k = idx.size
    d = np.empty((4, k), np.float32)
    for g in range(4):
        s = L[g * 16:(g + 1) * 16]
        m = s.max(0)
        e = np.exp(s - m); e /= e.sum(0)
        d[g] = AR16.ravel() @ e
    a = anchor[:, :, idx]; st = stride[:, idx]
    x1 = a - d[None, 0:2]; x2 = a + d[None, 2:4]
    box = np.concatenate([(x1 + x2) * 0.5, x2 - x1], 1) * st
    cl = cls[0][:, idx].max(0)[None]
    cls_id = cls[0][:, idx].argmax(0)[None].astype(np.float32)
    return np.concatenate([box[0].T, cl.T, cls_id.T], 1)   # (k,6)


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


class Pipeline:
    def __init__(self, model640, anchor, stride, nworkers=6):
        self.anchor, self.stride = anchor, stride
        self.nworkers = nworkers
        # ★ 每个 worker 一个独立上下文：rknnlite 不是线程安全的，共用会互相踩输出
        self.ctx = []
        for i in range(nworkers):
            r = RKNNLite(verbose=False)
            assert r.load_rknn(model640) == 0
            assert r.init_runtime(core_mask=MASK[i % 3]) == 0
            self.ctx.append(r)
        self.q = queue.Queue()
        self.res = {}
        self.pending = {}
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.ths = [threading.Thread(target=self._work, args=(i,), daemon=True) for i in range(nworkers)]
        for t in self.ths:
            t.start()

    def _work(self, wid):
        r = self.ctx[wid]
        while True:
            job = self.q.get()
            if job is None:
                self.q.task_done(); return
            fid, tid, tile = job
            rgb = tile[:, :, ::-1]           # ★ 模型按 RGB 标定，cv2 读进来是 BGR
            a = r.inference(inputs=[np.ascontiguousarray(rgb)[None]])
            lg = cl = None
            for t in a:
                t = np.asarray(t)
                if t.ndim == 3 and t.shape[1] == 64: lg = t
                elif t.ndim == 3 and t.shape[1] == 80: cl = t
            b = dfl_candidates(lg, cl, self.anchor, self.stride)
            with self.cv:
                if len(b): self.res.setdefault(fid, []).append((tid, b))
                self.pending[fid] = self.pending.get(fid, 1) - 1
                if self.pending[fid] <= 0: self.cv.notify_all()
            self.q.task_done()

    def submit(self, fid, tiles, origins):
        with self.lock:
            self.pending[fid] = len(tiles)
        for tid, (tl, _) in enumerate(zip(tiles, origins)):
            self.q.put((fid, tid, tl))

    def collect(self, fid, origins):
        with self.cv:
            while self.pending.get(fid, 0) > 0:
                self.cv.wait(timeout=30)
            parts = self.res.pop(fid, [])
            self.pending.pop(fid, None)
        out = []
        for tid, b in parts:
            x0, y0 = origins[tid]
            bb = b.copy(); bb[:, [0, 2]] += x0; bb[:, [1, 3]] += y0
            out.append(bb)
        allb = np.concatenate(out, 0) if out else np.zeros((0, 6), np.float32)
        return nms(allb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/home/orangepi/y11/real640/*.png")
    ap.add_argument("--model", default="/home/orangepi/y11/yolo11n_640_head_real.rknn")
    ap.add_argument("--nworkers", type=int, default=6)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--size", type=int, default=1280)
    a = ap.parse_args()

    anchor = np.load("/home/orangepi/y11/anchor640.npy")
    stride = np.load("/home/orangepi/y11/stride640.npy")
    files = sorted(glob.glob(a.images))
    if not files:
        raise SystemExit(f"没有图片: {a.images}")
    print(f"图片 {len(files)} 张，worker {a.nworkers}，输入 {a.size}")

    p = Pipeline(a.model, anchor, stride, a.nworkers)

    # 预切块（模拟真实流：每帧一次）
    frames = []
    for pth in files:
        im = cv2.imread(pth)
        if im is None: continue
        im = cv2.resize(im, (a.size, a.size), interpolation=cv2.INTER_LINEAR)
        n = 2; th = a.size // n
        tiles, origins = [], []
        for i in range(n):
            for j in range(n):
                y0, x0 = i * th, j * th
                tiles.append(np.ascontiguousarray(im[y0:y0 + th, x0:x0 + th]))
                origins.append((x0, y0))
        frames.append((tiles, origins))
    print(f"已切块 {len(frames)} 帧")

    # 预热
    for i in range(min(3, len(frames))):
        p.submit(i, *frames[i]); p.collect(i, frames[i][1])

    lat, dets = [], 0
    N = min(a.frames, len(frames))
    t_all = time.perf_counter()
    for i in range(N):
        t0 = time.perf_counter()
        p.submit(i + 100, *frames[i % len(frames)])
        b = p.collect(i + 100, frames[i % len(frames)][1])
        lat.append((time.perf_counter() - t0) * 1000)
        dets += len(b)
    wall = time.perf_counter() - t_all
    lat.sort()
    print(f"  {N} 帧 / {wall:.2f}s = {N/wall:.2f} FPS")
    print(f"  单帧延迟: 中位 {statistics.median(lat):.2f} ms  p90 {lat[int(N*0.9)]:.2f} ms")
    print(f"  检出合计 {dets}（平均 {dets/N:.1f}/帧）")


if __name__ == "__main__":
    main()
