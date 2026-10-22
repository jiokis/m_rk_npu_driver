from ultralytics import YOLO
import sys
for m in ["pose","seg","obb","cls"]:
    try:
        mod = YOLO(f"yolo11n-{m}.pt")
        p = mod.export(format="onnx", imgsz=640, opset=12, simplify=False, dynamic=False)
        print(f"OK {m} -> {p}")
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__}: {str(e)[:100]}")
