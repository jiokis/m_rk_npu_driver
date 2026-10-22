import os
from rknn.api import RKNN
CAL="/home/dunaandone/rknn-work/calib/dataset640.txt"
for t in ["pose","seg","obb"]:
    try:
        r=RKNN(verbose=False)
        r.config(target_platform="rk3588", mean_values=[[0,0,0]], std_values=[[255,255,255]],
                 quantized_dtype="w8a8", optimization_level=3)
        r.load_onnx(model=f"yolo11n-{t}-split.onnx")
        r.build(do_quantization=True, dataset=CAL)
        out=f"yolo11n-{t}-split-int8.rknn"
        r.export_rknn(out); r.release()
        print(f"OK {t}: {os.path.getsize(out)/1048576:.2f} MB")
    except Exception as e:
        print(f"FAIL {t}: {type(e).__name__}: {str(e)[:120]}")
