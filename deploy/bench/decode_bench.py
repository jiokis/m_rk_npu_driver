#!/usr/bin/env python3
"""大图端到端里 CPU 那部分：JPEG 解码 + letterbox"""
import statistics, time
import numpy as np, cv2

for src in (640, 1280, 1920, 3840):
    img = np.random.randint(0, 255, (src, src, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    data = buf.tobytes()
    dec, pre = [], []
    for _ in range(20):
        t0 = time.perf_counter(); im = cv2.imdecode(buf, cv2.IMREAD_COLOR); dec.append((time.perf_counter()-t0)*1000)
        t0 = time.perf_counter()
        r = 640.0 / max(im.shape[:2]); nw, nh = int(im.shape[1]*r), int(im.shape[0]*r)
        s = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        c = np.zeros((640, 640, 3), dtype=np.uint8); c[:nh, :nw] = s
        pre.append((time.perf_counter()-t0)*1000)
    print("  源 %4d2  JPEG %5.1f KB  解码 %6.2f ms  letterbox %5.2f ms  合计 %6.2f ms"
          % (src, len(data)/1024, statistics.median(dec), statistics.median(pre),
             statistics.median(dec)+statistics.median(pre)))
