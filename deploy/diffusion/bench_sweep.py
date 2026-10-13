#!/usr/bin/env python3
"""bench_sweep.py — 在板子上给 sweep/ 下所有 .rknn 计时（板上运行）"""
import glob, json, os, sys, time
import numpy as np

def main():
    from rknnlite.api import RKNNLite
    tbl = json.load(open("table.json"))
    out = {}
    for name, meta in tbl.items():
        rk = f"{name}.rknn"
        if not os.path.exists(rk):
            continue
        lat = meta["lat"]
        r = RKNNLite(verbose=False)
        if r.load_rknn(rk) != 0 or r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
            print(f"SWEEP {name} LOAD_FAIL"); continue
        z = np.zeros((1, 4, lat, lat), dtype=np.float32)
        ins = [z, np.zeros((1, 128, 1, 1), dtype=np.float32)] if meta["kind"] == "unet" else [z]
        for _ in range(3):
            r.inference(inputs=ins)
        ts = []
        for _ in range(15):
            t0 = time.perf_counter(); r.inference(inputs=ins); ts.append(time.perf_counter() - t0)
        r.release()
        ms = sorted(ts)[len(ts) // 2] * 1000
        gfl = meta["gflops"]
        out[name] = {**meta, "ms": ms, "tops": gfl / 1e9 / (ms / 1000) if ms else 0}
        print(f"SWEEP {name:20s} {ms:8.2f} ms  {gfl:7.3f} GFLOPs  "
              f"⇒ {gfl/1e9/(ms/1000):5.2f} TFLOP/s")
    json.dump(out, open("bench.json", "w"), indent=1)

main()
