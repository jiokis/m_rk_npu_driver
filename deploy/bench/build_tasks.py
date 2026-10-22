import os
from rknn.api import RKNN
CAL="/home/dunaandone/rknn-work/calib/dataset640.txt"
for m in ["pose","seg","obb","cls"]:
    try:
        r=RKNN(verbose=False)
        r.config(target_platform="rk3588", mean_values=[[0,0,0]], std_values=[[255,255,255]],
                 quantized_dtype="w8a8", optimization_level=3)
        r.load_onnx(model=f"yolo11n-{m}.onnx")
        r.build(do_quantization=True, dataset=CAL)
        out=f"yolo11n-{m}-int8.rknn"
        r.export_rknn(out); r.release()
        print(f"OK {m}: {os.path.getsize(out)/1048576:.2f} MB")
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__}: {str(e)[:120]}")
