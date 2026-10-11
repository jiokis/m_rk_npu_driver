#!/usr/bin/env python3
"""
gguf_reader.py — 极简 GGUF 读取器（只做我们需要的：读元数据 + 取 F16 张量）

为什么自己写：装 onnx/gguf 包要么网络受限、要么依赖太重。
GGUF 格式本身很简单，读 F16 张量只需几十行。

格式（v2/v3）：
  magic "GGUF" | version u32 | tensor_count u64 | kv_count u64
  kv[]    : key(string) type(u32) value
  info[]  : name(string) n_dims(u32) dims(u64×n) type(u32) offset(u64)
  [padding 到 alignment]
  data    : 各张量按 offset 排布

注意：GGUF 的 dims 是 GGML 顺序（ne[0] 最快变化），numpy 里要反过来。
"""
import struct

import numpy as np

GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q4_1 = 0, 1, 2, 3
GGML_Q8_0 = 8
GGML_Q4_K, GGML_Q6_K, GGML_Q8_K = 12, 13, 14

TYPE_NAME = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0",
             12: "Q4_K", 13: "Q6_K", 14: "Q8_K"}

# 每个元素的字节数（仅用于跳过不支持的类型）
TYPE_SIZE = {"F32": 4, "F16": 2}

_VT = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
       5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8),
       12: ("d", 8)}


class GGUF:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        magic = self.f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"不是 GGUF 文件: magic={magic!r}")
        self.version = self._u32()
        self.tensor_count = self._u64()
        kv_count = self._u64()

        self.meta = {}
        for _ in range(kv_count):
            k = self._str()
            self.meta[k] = self._value()

        self.alignment = self.meta.get("general.alignment", 32)

        self.tensors = {}
        for _ in range(self.tensor_count):
            name = self._str()
            n = self._u32()
            dims = [self._u64() for _ in range(n)]     # GGML 顺序
            ttype = self._u32()
            off = self._u64()
            self.tensors[name] = {"dims": dims, "type": ttype, "offset": off,
                                  "shape": tuple(reversed(dims))}  # numpy 顺序

        pos = self.f.tell()
        self.data_start = (pos + self.alignment - 1) // self.alignment * self.alignment

    # ── 基础读取 ──
    def _u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def _str(self):
        n = self._u64()
        return self.f.read(n).decode("utf-8", "replace")

    def _value(self):
        t = self._u32()
        if t == 8:                       # STRING
            return self._str()
        if t == 9:                       # ARRAY
            et = self._u32()
            n = self._u64()
            if et == 8:
                return [self._str() for _ in range(n)]
            fmt, sz = _VT[et]
            return list(struct.unpack(f"<{n}{fmt}", self.f.read(n * sz)))
        fmt, sz = _VT[t]
        return struct.unpack(f"<{fmt}", self.f.read(sz))[0]

    # ── 取张量 ──
    def read_f16(self, name):
        """读 F16/F32 张量，返回 float32 的 numpy 数组（形状为 numpy 顺序）"""
        t = self.tensors[name]
        tn = TYPE_NAME.get(t["type"], f"type{t['type']}")
        if tn not in TYPE_SIZE:
            raise ValueError(f"{name}: 类型 {tn} 暂不支持（本读取器只处理 F16/F32）")
        n = int(np.prod(t["shape"]))
        self.f.seek(self.data_start + t["offset"])
        raw = self.f.read(n * TYPE_SIZE[tn])
        arr = np.frombuffer(raw, dtype="<f2" if tn == "F16" else "<f4")
        return arr.astype(np.float32).reshape(t["shape"])

    def close(self):
        self.f.close()


def describe(path):
    g = GGUF(path)
    arch = g.meta.get("general.architecture", "?")
    print(f"文件: {path}")
    print(f"  GGUF v{g.version} · 架构 {arch} · 张量 {g.tensor_count} 个 · "
          f"对齐 {g.alignment}")
    for k in ("general.name", f"{arch}.block_count", f"{arch}.embedding_length",
              f"{arch}.attention.head_count", f"{arch}.feed_forward_length"):
        if k in g.meta:
            print(f"  {k} = {g.meta[k]}")
    types = {}
    for t in g.tensors.values():
        types[TYPE_NAME.get(t["type"], t["type"])] = \
            types.get(TYPE_NAME.get(t["type"], t["type"]), 0) + 1
    print(f"  张量类型分布: {types}")
    g.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        describe(sys.argv[1])
    else:
        print(__doc__)
