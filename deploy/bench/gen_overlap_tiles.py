#!/usr/bin/env python3
"""gen_overlap_tiles.py — 生成重叠分块的 raw tile 数据（768 裁剪 → 缩到 640）"""
import glob
import sys

import cv2
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/dunaandone/rknn-work/calib/real/*.jpg"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tiles_ovl27.raw"
TILEPX, STRIDE, S = 768, 512, 640

fs = sorted(glob.glob(SRC))
buf = bytearray()
n = 0
for p in fs:
    im = cv2.imread(p)
    if im is None:
        continue
    h, w = im.shape[:2]
    r = min(1280 / h, 1280 / w)
    rz = cv2.resize(im, (int(round(w * r)), int(round(h * r))))
    c = np.full((1280, 1280, 3), 114, np.uint8)
    t, l = (1280 - rz.shape[0]) // 2, (1280 - rz.shape[1]) // 2
    c[t:t + rz.shape[0], l:l + rz.shape[1]] = rz
    for y0 in (0, STRIDE):
        for x0 in (0, STRIDE):
            tl = c[y0:y0 + TILEPX, x0:x0 + TILEPX]
            sm = cv2.resize(tl, (S, S), interpolation=cv2.INTER_LINEAR)
            buf += np.ascontiguousarray(sm[:, :, ::-1]).tobytes()   # RGB
    n += 1
open(OUT, "wb").write(bytes(buf))
print(f"{n} 帧 × 4 tile({TILEPX}→{S}) = {len(buf)/1048576:.0f} MB → {OUT}")
