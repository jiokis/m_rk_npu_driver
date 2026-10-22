#!/usr/bin/env python3
"""nms_unit_test.py — 验证 NMS 实现等价性（torchvision / OpenCV / 贪心）"""
import numpy as np, cv2, torch
from torchvision.ops import nms as tv_nms

rng = np.random.default_rng(0)
boxes = []
for _ in range(30):
    x, y = rng.uniform(0, 500), rng.uniform(0, 500)
    w, h = rng.uniform(20, 120), rng.uniform(20, 120)
    for _ in range(4):                      # 每个目标 4 个近似框
        boxes.append([x + rng.normal(0, 4), y + rng.normal(0, 4),
                      w + rng.normal(0, 4), h + rng.normal(0, 4)])
boxes = np.array(boxes, np.float32)
scores = np.sort(rng.uniform(0.25, 0.95, len(boxes)))[::-1].astype(np.float32)
xyxy = np.concatenate([boxes[:, :2], boxes[:, :2] + boxes[:, 2:4]], 1)

keep_tv = tv_nms(torch.from_numpy(xyxy), torch.from_numpy(scores), 0.45).numpy()
k = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.25, 0.45)
keep_cv = np.asarray(k).ravel() if len(k) else np.array([], int)

def greedy(xyxy, sc, thr=0.45):
    order = np.argsort(-sc); out = []; sup = np.zeros(len(sc), bool)
    for a in order:
        if sup[a]: continue
        out.append(a)
        for b in order:
            if sup[b] or b == a: continue
            iw = max(0, min(xyxy[a,2], xyxy[b,2]) - max(xyxy[a,0], xyxy[b,0]))
            ih = max(0, min(xyxy[a,3], xyxy[b,3]) - max(xyxy[a,1], xyxy[b,1]))
            inter = iw * ih
            aa = (xyxy[a,2]-xyxy[a,0])*(xyxy[a,3]-xyxy[a,1])
            bb = (xyxy[b,2]-xyxy[b,0])*(xyxy[b,3]-xyxy[b,1])
            if inter/(aa+bb-inter+1e-9) > thr: sup[b] = True
    return np.array(out)

keep_gr = greedy(xyxy, scores)
print(f"输入 {len(boxes)}   torchvision {len(keep_tv)}  OpenCV {len(keep_cv)}  贪心 {len(keep_gr)}")
print(f"tv∩greedy {len(set(keep_tv)&set(keep_gr))}   cv∩tv {len(set(keep_cv)&set(keep_tv))}")
