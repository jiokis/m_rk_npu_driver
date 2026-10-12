# deploy/quant/ —— 编译器自己的量化参数与权重块（仿真器的「标准答案」）

## 怎么拿到的（一条被忽略很久的捷径）

```python
from rknn.api import RKNN
r = RKNN(); r.config(target_platform='rk3588', ...)
r.load_onnx(model='probe4.onnx')
r.hybrid_quantization_step1(dataset='dataset.txt')
# ⇒ 生成三个文件：
#    <name>.quantization.cfg   ← 逐层量化参数（scale / zero_point / qtype）
#    <name>.model              ← RKNN IR（含量化后的权重块）
#    <name>.data               ← 量化数据
```

**这是 `rknn-toolkit2` 的公开 API**，一直在方法列表里（`hybrid_quantization_step1/2`），
而我在权重布局上盲探了 6 轮（docs/62~67）才想到用它。

## 拿到了什么

### 1. 逐层量化参数（`.cfg`）

```yaml
quantize_parameters:
    input:   {qtype: asym, dtype: float32, min:[0.0], max:[1.0]}
    c:       {qtype: asym, qmethod: layer, dtype: int8,
              min:[0.0], max:[277.235...], scale:[1.0871973374310662], zero_point:[-128]}
    r:       {同 c}
    gap_2conv0: {qtype: asym, scale:[1.2743302887561274], zero_point:[-128], ...}
```

**⇒ `zero_point = -128`、`qtype = asym`、`qmethod = layer` 都是权威值**，
反量化公式由此确定：`w_float = (q_int8 − zero_point) × scale`。
（这也**证实了 docs/62 的推断**：字节 `128` 就是 `-128`。）

### 2. 量化后的权重块（`.model`）—— 与板上 DRAM **吻合 93.75%**

```
probe4.model 中偏移 3616 起的 2304 字节 vs 板上 buf_000_dmaffff9000.bin 前 2304 字节
  ⇒ 逐字节相同 2160 / 2304 = 93.75%
```

**⇒ 编译器自己的权重块与 NPU 实际读的那块是同一套布局。**
⇒ 权重布局**不必再猜**：`.model` 里的就是标准答案（差异的 144 字节待查，疑为填充槽的 0/128 差异）。

## 这对仿真器的意义

| 阶段 | 之前 | 现在 |
|---|---|---|
| ③ 权重布局 + 量化参数 | 🟡 盲探 6 轮未定 | ✅ **有标准答案**（`.cfg` 参数 + `.model` 权重块 + 与 DRAM 吻合 93.75%） |
| ④ 算子执行器 | 无法验证 | 可以用 `.cfg` 反量化、按 `.model` 的布局取权重，再与板子/官方模拟器对齐 |
