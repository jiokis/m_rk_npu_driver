"""把 ONNX 里的 Conv 权重按 4:2 结构化稀疏剪枝（每组 4 个输入通道保留幅值最大的 2 个）"""
import sys
import numpy as np, onnx
from onnx import numpy_helper
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
conv_w = set()
for n in g.node:
    if n.op_type == "Conv" and len(n.input) >= 2:
        conv_w.add(n.input[1])
pruned = 0
for i, init in enumerate(g.initializer):
    if init.name not in conv_w: continue
    w = numpy_helper.to_array(init).astype(np.float32)
    if w.ndim != 4: continue
    co, ci, kh, kw = w.shape
    if ci < 4: continue
    g4 = ci // 4
    wv = w.reshape(co, g4, 4, kh, kw)
    # 每组 4 个通道里保留幅值最大的 2 个
    idx = np.argsort(-np.abs(wv), axis=2)[:, :, :2]          # (co,g4,2,kh,kw)
    mask = np.zeros_like(wv, dtype=bool)
    for k in range(2):
        np.put_along_axis(mask, idx[:, :, k:k+1], True, axis=2)
    wv = np.where(mask, wv, 0.0)
    new = numpy_helper.from_array(wv.reshape(co, ci, kh, kw), init.name)
    g.initializer.remove(init); g.initializer.insert(i, new)
    pruned += 1
print(f"剪枝 {pruned} 个 Conv 权重")
onnx.save(m, dst)
onnx.checker.check_model(onnx.load(dst))
print("保存", dst)
