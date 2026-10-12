# 109 · ★★★ 规则触发**可观测**：monkeypatch 成功，tiling 规则确实触发

> 接 [docs/108](108-编译器三张规则表已导出-355融合206转换12tiling.md)（三张规则表在 Python 侧）。
> 本轮的关键推理：**如果 C++ pass 回调这些 Python `proc`，我就能 monkeypatch 它们。**

---

## 1. ★★★ monkeypatch **成功**

把三张表的每个 `proc` 包一层计数/打印，然后跑真实 build：

```
已包装规则数: 573（355 + 206 + 12）
BUILD OK，触发规则数: 270（probe6）
```

**⇒ C++ 编译器**确实回调 Python 的 `proc`**，所以整条规则执行链**可以被观测**。**

### 观测到的规则序列（前 12）

```
[fuse_rules]    remove_parallel_1x1_conv
[fuse_rules]    absorb_gather_into_attention_conv
[fuse_rules]    fuse_align_conv_backward
[fuse_rules]    fuse_align_conv_forward
[fuse_rules]    remove_invalid_relu
[fuse_rules]    remove_parallel_relu
[convert_rules] seperate_large_kernel_conv
[convert_rules] convert_3dconv_to_2dconv
[convert_rules] convert_group_conv2d_to_conv2d
[convert_rules] convert_large_dilation_conv_to_normal_conv
[convert_rules] convert_large_stride_conv_to_two_conv
[convert_rules] minimize_conv_stride
```

**⇒ 编译器「做了什么」现在是**可观测的过程**，不再是黑盒。**

---

## 2. ★★ 最重要的：**tiling 规则确实触发**

```
hardware_rules 触发（各模型）：
  convert_conv_to_pad_conv      每个模型都触发
  tiling_depthwise_conv         每个模型都触发
  tiling_conv_channelin         每个模型都触发
  tiling_maxpool                2 个模型
  tiling_group_conv_act         1 个模型

每模型 tiling 触发总数：
  efficientnet_b0 1113 / densenet121 1095 / regnet_x_400mf 654 /
  googlenet 567 / mobilenet_v2 483 / resnet18 195 / squeezenet1_0 87
```

**⇒ docs/99 的卡点解开了**：那 12 条 `proc` 我之前手工调用返回空，
**但在真实 build 里它们正常工作**（因为此时 IR 已有形状与量化信息）。

**⇒ 我现在可以 hook 它们、捕获**入参与返回值** ⇒ **直接观察 tiling 做了什么。**

---

## 3. ⚠️ 但「规则触发次数」也不预测提交数

```
模型              触发/conv   提交/conv
mobilenet_v2        90.8       0.06
resnet18           119.0       0.25
squeezenet1_0       54.5       0.85
googlenet          106.3       0.61
regnet_x_400mf     110.0       1.59
efficientnet_b0    125.1       1.09
densenet121        123.7       1.52

原始  r(总触发, 提交)          = +0.919    ← 看着很强
归一化 r(触发/conv, 提交/conv) = +0.269    ← ★ 几乎消失
```

**⇒ 那个 +0.919 是**模型大小的混杂**（大模型触发多、提交也多）。
归一化后 r=0.269（R²=0.07）⇒ **规则触发次数也不决定提交数。**

**⇒ 这是第四个被否掉的粒度**（IR 算子数 / 搬运算子数 / 按类型 / **规则触发次数**）。

---

## 4. 但本轮**拿到了观测通道**，这才是关键

| 之前 | 现在 |
|---|---|
| tiling proc 手工调用返回空（docs/99 卡点） | **真实 build 里正常工作** ✅ |
| 只能看最终 IR 表 | **能看到 270 条规则按序触发** ✅ |
| 提交数只能用「结果」去相关（全失败） | **可以捕获 tiling proc 的入参/返回值** ⬜ **下一步** |

**⇒ 下一步不是「再做一次相关性」，而是**捕获 tiling 的入参与返回**，
直接看它**把什么切成了什么**。**

---

## 5. 下一步（明确）

| # | 动作 | 预期 |
|---|---|---|
| 1 | **包装 tiling proc 时同时捕获 `nodes` 入参与返回值** | **直接看到「哪个算子被切成哪些 tile」** |
| 2 | 对同一个算子，看**不同模型的 tiling 结果差异** | 为什么 regnet 的 `tile/conv=44.3` 而 mnasnet 只有 3.2 |
| 3 | 检验「tile 总数」与提交数的关系 | **tile 数是切分的直接产物** |

**#1 是决定性的**：**入参（算子）+ 返回值（tiles）= 切分规则本身。**

---

## 6. 教训（本轮新增）

1. **★★★ 「C++ 回调 Python」这个结构假设一旦成立，整条链就打开了。**
   我看到规则表是 Python 的 `cyfunction` 时，第一反应是「那是给 Python 调用的」；
   **真正的用法是「C++ 在 pass 里回调它们」** —— 所以 monkeypatch 能拦住全部规则执行。
   **看到「回调型」接口时，要问「谁在调它」，而不是「我怎么调它」。**
2. **★★ 混杂变量要**主动归一化**才看得出来。** 原始 r=+0.919 看起来极强，
   按 conv 数归一化后掉到 +0.269 —— **两个都随模型大小增长的量，必然强相关。**
   **做相关性前先问「有没有一个共同的大小因子」。**
3. **★ 否证四次之后，「拿到观测通道」比「再试第五个特征」有价值得多。**
   第四个粒度（规则触发）又被否掉，但本轮真正的收获是**能捕获 tiling 的入参/返回**。
   **当相关法连续失败时，应该换方法（从观测结果 → 观测过程），而不是换特征。**
