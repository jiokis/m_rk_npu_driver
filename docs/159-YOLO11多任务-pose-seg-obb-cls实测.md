# docs/159 — YOLO11 的其它任务：pose / seg / OBB / cls 在 RK3588 上的实测

> 之前只做了 detection。这份把 YOLO11n 的另外四个任务也导出、转换、跑通，
> 并验证了 `docs/152` 那个「INT8 把分类分数打成 0」的坑是**系统性**的。

---

## 0. 结果

| 任务 | 输出形状 | 中位延迟 | FPS | 模型 |
|---|---|---|---|---|
| **det**（基准） | `(1,56,8400)` 拆前 / 双输出 | **28.27 ms** | 35.4 | 4.06 MB |
| **pose** | `(1,4)+(1,1)+(1,51)` | 33.77 ms | 29.6 | 5.43 MB |
| **seg** | `(1,4)+(1,80)+(1,32)+(1,32,160,160)` | 35.91 ms | 27.8 | 5.02 MB |
| **obb** | `(1,4)+(1,15)+(1,1)` | **26.84 ms** | 37.3 | 4.69 MB |
| **cls** | `(1,1000)` | **18.53 ms** | 54.0 | 3.59 MB |

单实例、绑 3 核、640 输入、零张量输入（纯推理）。检测类任务若套用 1280 分块+并发，
可再乘 `docs/151` 的 4.9× 系数。

---

## 1. 五个任务全部导出成功

```python
from ultralytics import YOLO
for m in ["", "-pose", "-seg", "-obb", "-cls"]:
    YOLO(f"yolo11n{m}.pt").export(format="onnx", imgsz=640, opset=12,
                                  simplify=False, dynamic=False)
```

权重从 `gh-proxy` 镜像取（GitHub 直连不通）：

```
https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-{pose,seg,obb,cls}.pt
```

---

## 2. ★ 三个检测类任务**全部**踩了同一个坑

直接整模型转 INT8 后，检查输出：

| 任务 | 行0-3（框） | **行4（分数）** | 判定 |
|---|---|---|---|
| pose | max 640.3 | **0.0000** | ❌ 分数全零 |
| seg | max 642.9 | **0.0000** | ❌ 分数全零 |
| obb | max 641.6 | **0.0000**（行4+ 非零 = 0） | ❌ 分数全零 |
| cls | — | max 0.264 | ✅ 正常（没有那个 Concat） |

**只有 cls 幸免** —— 因为它的输出没有把「框坐标（0~640）」和「分数（0~1）」拼进同一个张量。

⇒ `docs/152` 的结论是**系统性的**：任何 Ultralytics 检测头（det/pose/seg/obb）
整模型 INT8 量化，分数都会被共享量化 scale 压成 0。
症状一样：不报错、形状对、框合理、**只有分数是 0**，NMS 后检出为 0。

---

## 3. 通用修复：把最终 Concat 拆成多个输出

`deploy/bench/split_head.py` 自动定位产生 `output0` 的 `Concat` 节点，
用 `onnx.utils.extract_model` 把它的每个输入提升为独立图输出：

```python
concat = next(n for n in graph.node if n.op_type == "Concat" and "output0" in n.output)
onnx.utils.extract_model(src, dst, ["images"], list(concat.input) + 其它输出)
```

拆分结果：

| 任务 | 拆出的输出 |
|---|---|
| pose | `(1,4,8400)` 框 + `(1,1,8400)` 置信度 + `(1,51,8400)` 关键点 |
| seg | `(1,4)` 框 + `(1,80)` 类别 + `(1,32)` mask 系数 + `(1,32,160,160)` 原型 |
| obb | `(1,4)` 框 + `(1,15)` 类别 + `(1,1)` **角度** |

拆分后重新 INT8 量化，分数立刻恢复：

| 任务 | ONNX FP32 分数 max | INT8 拆分后 max | 说明 |
|---|---|---|---|
| pose | 0.0972 | 0.0350 | 该图无清晰人体，两者都低 |
| seg | **0.8386** | **0.7470** | 强目标，INT8 只低 11% |
| obb | 0.1696 | 0.0840 | OBB 是 DOTA 航拍训练的，COCO 图本就低 |

✅ 全部非零，**量化链路恢复正常**。

> 注意 OBB 的输出布局：`(1,4)` 框 + `(1,15)` 类别 + `(1,1)` **角度**（范围约 ±1.6）。
> 一开始我把角度当类别分数比，闹了乌龙（ONNX "0.1696" vs "1.6246" 就是这么来的）。

---

## 4. 各任务的速度差异说明

| 任务 | ms | 为什么 |
|---|---|---|
| cls | **18.5** | 没有检测头，直接全局池化 + FC |
| obb | **26.8** | 头最简（4+15 通道）；比 det 还快 |
| det | 28.3 | 基线（4+80） |
| pose | 33.8 | 多 51 个关键点通道 + 更重的 head 卷积 |
| seg | 35.9 | 多 32 个 mask 系数 + 160×160 原型分支 |

---

## 5. 使用建议

| 场景 | 选型 |
|---|---|
| 通用目标检测 | det（`docs/157` 的完整链路：1280 分块 + 并发，32 FPS） |
| 人体姿态 | pose（640 约 30 FPS；1280 需同样做分块） |
| 实例分割 | seg（最贵，640 约 28 FPS；mask 后处理还要额外 CPU） |
| 遥感/航拍旋转框 | obb（**必须用 DOTA 类数据**，普通 COCO 图上分数天然极低） |
| 纯分类 | cls（54 FPS，最划算） |

---

## 6. 边界（同前）

| 项 | 状态 |
|---|---|
| 精度 | **只验证了"分数非零 + 与 FP32 同量级"，没有 mAP** |
| 1280 适配 | 只有 det 做了完整的分块+并发链路；其余四个只测了 640 单实例 |
| pose/seg/obb 的后处理 | 只做了输出拆分与数值检查，**没写 CPU 后处理**（关键点解析、mask 组装、旋转框 NMS） |
| 热稳定 | 未针对这些任务复测 |

---

## 7. 产物

```
~/rknn-work/yolo11/
  yolo11n-{pose,seg,obb,cls}.pt         官方权重
  yolo11n-{pose,seg,obb,cls}.onnx       FP32 ONNX
  yolo11n-{pose,seg,obb,cls}-int8.rknn  整模型 INT8（分数全零，留作反例）
  yolo11n-{pose,seg,obb}-split.onnx     拆开输出后的 ONNX
  yolo11n-{pose,seg,obb}-split-int8.rknn  ★ 可用的 INT8 模型
```

脚本：`deploy/bench/{export_tasks.py, split_head.py, build_tasks.py, bench_tasks.py}`
