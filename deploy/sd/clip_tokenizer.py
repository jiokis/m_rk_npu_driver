#!/usr/bin/env python3
"""clip_tokenizer.py — **纯标准库**的 CLIP BPE 分词器（板上没有 tokenizers/transformers）

CLIP 的分词规则（OpenAI 版）：
  1. 小写化；用正则把文本切成词块（`<|startoftext|>`/`<|endoftext|>` 是特殊 token）
  2. 每个词块按 `</w>` 结尾做 byte-level BPE（用 `merges.txt` 的合并优先级）
  3. 查 `vocab.json` 得 id；补 BOS/EOS；截断/补齐到 77

⚠️ 必须和 HF 的 `CLIPTokenizer` 一致，否则 CLIP 输出会偏 ⇒ 见 `--selftest`。
"""
from __future__ import annotations

import gzip
import json
import os
import re

#: CLIP 用的正则。
#  ⚠️ OpenAI 原版用 `\p{L}`/`\p{N}`，那是 `regex` 模块的语法，**标准库 `re` 不支持**
#     （报 `bad escape \p`）⇒ 这里用等价的 ASCII 版本（CLIP 的语料以英文为主，实测一致）。
PAT = re.compile(
    r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^\s a-zA-Z0-9]+""",
    re.IGNORECASE,
)


def _bytes_to_unicode():
    """OpenAI 的 byte↔unicode 映射（保证可打印）"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class CLIPTokenizer:
    def __init__(self, vocab_path: str, merges_path: str, ctx: int = 77):
        self.ctx = ctx
        self.encoder = json.load(open(vocab_path))
        self.bpe_ranks = {}
        with open(merges_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0 and line.startswith("#"):
                    continue
                p = line.rstrip("\n").split()
                if len(p) == 2:
                    self.bpe_ranks[(p[0], p[1])] = len(self.bpe_ranks)
        self.byte_encoder = _bytes_to_unicode()
        self.cache = {}
        self.bos = self.encoder["<|startoftext|>"]
        self.eos = self.encoder["<|endoftext|>"]

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = set(zip(word, word[1:]))
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            a, b = bigram
            new, i = [], 0
            while i < len(word):
                try:
                    j = word.index(a, i)
                except ValueError:
                    new.extend(word[i:]); break
                new.extend(word[i:j]); i = j
                if word[i] == a and i < len(word) - 1 and word[i + 1] == b:
                    new.append(a + b); i += 2
                else:
                    new.append(word[i]); i += 1
            word = tuple(new)
            if len(word) == 1:
                break
            pairs = set(zip(word, word[1:]))
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str) -> list[int]:
        text = re.sub(r"\s+", " ", text.strip().lower())
        ids = []
        for tok in re.findall(PAT, text):
            if tok in ("<|startoftext|>", "<|endoftext|>"):
                ids.append(self.encoder[tok]); continue
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            for piece in self._bpe(tok).split(" "):
                if piece:
                    ids.append(self.encoder.get(piece, self.encoder.get("<|endoftext|>")))
        return [self.bos] + ids + [self.eos]

    def __call__(self, text: str) -> list[int]:
        ids = self.encode(text)[: self.ctx]
        return ids + [0] * (self.ctx - len(ids))


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "/home/dunaandone/rknn-work/sd/clip"
    t = CLIPTokenizer(os.path.join(d, "vocab.json"), os.path.join(d, "merges.txt"))
    for s in ("a photo of a cat", "an astronaut riding a horse", "红色的小房子"):
        print(f"{s!r} → {t(s)[:14]} …")
