#!/usr/bin/env python3
"""
qwen2_numpy.py — 纯 numpy 的 Qwen2 前向传播（用于量化实验的下游指标）

## 为什么要自己实现

docs/08 里最大的缺口是「只测了张量误差，没测下游任务指标」。
而全局 Frobenius 误差被证明是**不可靠的代理**（99.35% 的值归零，
输出误差却只显示 21.48%）。要判断量化方案到底行不行，必须算**真实困惑度**。

本机没有 llama.cpp/transformers（且板子断电中），所以直接用 numpy 实现。
模型是 Qwen2.5-0.5B-Instruct：24 层 / hidden 896 / 14 Q头 / 2 KV头(GQA) /
head_dim 64 / FFN 4864 / 有 QKV bias / 独立 lm_head。

## 关于分词

用 GGUF 里的词表做**贪心最长匹配**，不是真正的 BPE。
对本实验无害 —— 关键在**所有量化方案用完全相同的分词**，比较是公平的。
（局限：算出的困惑度不能和公开榜单数字直接比。）

## 用法
  python3 qwen2_numpy.py             # 自检：验证前向传播是否合理
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gguf_reader import GGUF  # noqa: E402

DEFAULT = "/home/dunaandone/rkllm-convert/gguf-test/qwen2.5-0.5b-instruct-fp16.gguf"


# ─────────────── 基础算子 ───────────────
def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def rope(x, pos, theta):
    """旋转位置编码。x: (B,H,T,D) → 同形状。

    ⚠️ 自检抓到的第二个真 bug：RoPE 有两种配对约定，必须用对：
      · interleaved（GPT-NeoX/GPT-J）：把相邻的 (0,1),(2,3)… 当一对
      · **half-split（llama / Qwen2 / HF）**：把 (i, i+D/2) 当一对

    推导（对齐 HF 的 rotate_half）：
      rotate_half(x) = concat(-x2, x1)，其中 x1=x[:half], x2=x[half:]
      out = x*cos + rotate_half(x)*sin
          = concat(x1*cos - x2*sin,  x2*cos + x1*sin)

    我最初写成了 interleaved，模型输出完全退化（top-1 是个孤立空格）。
    """
    B, H, T, D = x.shape
    half = D // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))   # (D/2,)
    ang = np.outer(pos, inv).astype(np.float32)                            # (T, D/2)
    cos, sin = np.cos(ang)[None, None], np.sin(ang)[None, None]            # (1,1,T,D/2)
    x1, x2 = x[..., :half], x[..., half:]
    o = np.empty_like(x)
    o[..., :half] = x1 * cos - x2 * sin
    o[..., half:] = x1 * sin + x2 * cos
    return o


# ─────────────── 模型 ───────────────
class Qwen2:
    def __init__(self, path=DEFAULT, weights_transform=None):
        """
        weights_transform(name, arr) -> arr
        用于在加载时对权重做量化/变换（量化实验的挂钩点）。
        """
        self.g = GGUF(path)
        m = self.g.meta
        self.n_layer = m["qwen2.block_count"]
        self.d = m["qwen2.embedding_length"]
        self.n_head = m["qwen2.attention.head_count"]
        self.n_kv = m["qwen2.attention.head_count_kv"]
        self.hd = self.d // self.n_head
        self.ffn = m["qwen2.feed_forward_length"]
        self.eps = m["qwen2.attention.layer_norm_rms_epsilon"]
        self.theta = m["qwen2.rope.freq_base"]
        self.n_rep = self.n_head // self.n_kv
        self.transform = weights_transform

    def w(self, name):
        a = self.g.read_f16(name)
        return self.transform(name, a) if self.transform else a

    def forward(self, ids):
        """ids: (T,) int → logits (T, vocab)"""
        g, d, hd = self.g, self.d, self.hd
        T = len(ids)
        pos = np.arange(T, dtype=np.float32)

        tok = self.g.read_f16("token_embd.weight")
        x = tok[ids].astype(np.float32)                      # (T, d)
        del tok

        for L in range(self.n_layer):
            p = f"blk.{L}."
            # ── 注意力 ──
            h = rms_norm(x, self.w(p + "attn_norm.weight"), self.eps)
            q = h @ self.w(p + "attn_q.weight").T + self.w(p + "attn_q.bias")
            k = h @ self.w(p + "attn_k.weight").T + self.w(p + "attn_k.bias")
            v = h @ self.w(p + "attn_v.weight").T + self.w(p + "attn_v.bias")
            q = q.reshape(T, self.n_head, hd).transpose(1, 0, 2)[None]     # (1,H,T,hd)
            k = k.reshape(T, self.n_kv, hd).transpose(1, 0, 2)[None]
            v = v.reshape(T, self.n_kv, hd).transpose(1, 0, 2)[None]
            q = rope(q, pos, self.theta)
            k = rope(k, pos, self.theta)
            k = np.repeat(k, self.n_rep, axis=1)                            # GQA 展开
            v = np.repeat(v, self.n_rep, axis=1)

            sc = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)                # (1,H,T,T)
            mask = np.triu(np.full((T, T), -1e30, dtype=np.float32), 1)
            sc = sc + mask
            a = softmax(sc, axis=-1)
            o = (a @ v).transpose(0, 2, 1, 3).reshape(T, d)                 # (T,d)
            x = x + o @ self.w(p + "attn_output.weight").T

            # ── FFN (SwiGLU) ──
            h = rms_norm(x, self.w(p + "ffn_norm.weight"), self.eps)
            gate = h @ self.w(p + "ffn_gate.weight").T
            up = h @ self.w(p + "ffn_up.weight").T
            x = x + (silu(gate) * up) @ self.w(p + "ffn_down.weight").T

        x = rms_norm(x, self.w("output_norm.weight"), self.eps)
        return x @ self.w("output.weight").T                                # (T, vocab)

    # ── 分词：byte-level 映射 + 贪心最长匹配 ──
    #
    # ⚠️ 自检抓到的真 bug：GPT-2 / llama-bpe 系词表是 **byte-level BPE**，
    # 空格在词表里存的是 'Ġ'(U+0120) 而不是字面空格。
    # 不做字节→unicode 映射的话，每个空格都匹配失败、退化成 id 0，
    # 于是 'The capital of France is' 被切成 [785, 0, 65063, 0, ...] —— 输入全毁了。
    # 所以必须先把文本按 GPT-2 的字节映射转成"词表空间"的字符串。
    #
    # 另外这仍是贪心最长匹配、不是真 BPE，属近似。
    # 但对本实验无害：所有量化方案用完全相同的分词，比较公平。
    @staticmethod
    def _byte_encoder():
        bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
              + list(range(ord("®"), ord("ÿ") + 1)))
        cs, n = bs[:], 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}

    def load_vocab(self):
        toks = self.g.meta["tokenizer.ggml.tokens"]
        self.vocab = toks
        self.bmap = self._byte_encoder()
        self.vmap = {}
        for i, t in enumerate(toks):
            if t and len(t) <= 32:
                self.vmap.setdefault(t, i)
        self.maxlen = max((len(k) for k in self.vmap), default=1)
        # 兜底 token：优先 <unk>，否则用 id 0
        self.unk = self.vmap.get("<unk>", 0)

    def encode(self, text):
        """文本 → token id 列表（byte-level 映射后贪心最长匹配）"""
        s = "".join(self.bmap[b] for b in text.encode("utf-8"))
        ids, i = [], 0
        while i < len(s):
            hit = None
            for L in range(min(self.maxlen, len(s) - i), 0, -1):
                sub = s[i:i + L]
                if sub in self.vmap:
                    hit = (self.vmap[sub], L)
                    break
            if hit:
                ids.append(hit[0]); i += hit[1]
            else:
                ids.append(self.unk); i += 1
        return ids

    def close(self):
        self.g.close()


def selftest():
    print("═══ 自检：前向传播是否合理 ═══")
    m = Qwen2()
    print(f"  层数 {m.n_layer} · hidden {m.d} · Q头 {m.n_head} · KV头 {m.n_kv} · "
          f"head_dim {m.hd} · FFN {m.ffn}")
    m.load_vocab()
    print(f"  词表 {len(m.vocab)} · 贪心匹配最大词长 {m.maxlen}")

    for text, expect in (("The capital of France is", "Paris"),
                         ("1 + 1 =", "2")):
        ids = m.encode(text)
        print(f"\n  输入 {text!r} → {len(ids)} 个 token: {ids[:12]}")
        logits = m.forward(ids)
        nxt = logits[-1]
        top = np.argsort(-nxt)[:8]
        print(f"  下一个 token 的 top-8 预测:")
        for t in top:
            s = m.vocab[t].replace("\n", "\\n")
            print(f"    {s!r:<16} logit {nxt[t]:8.3f}")
        ok = any(expect.lower() in m.vocab[t].lower() for t in top[:3])
        print(f"  期望含 {expect!r}: {'✅ 命中' if ok else '❌ 未命中'}")
    m.close()


if __name__ == "__main__":
    selftest()
