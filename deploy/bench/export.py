from ultralytics import YOLO
import sys
sz = int(sys.argv[1]) if len(sys.argv) > 1 else 1280
m = YOLO("yolo11n.pt")
p = m.export(format="onnx", imgsz=sz, opset=12, simplify=False, dynamic=False)
print("EXPORTED:", p)
