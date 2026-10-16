"""BM25 检索，中文按字符 bigram 切词（再加单字兜底），英文数字按词。

知识库（lore）和长期记忆共用这一个检索器；需要换成向量检索时只替换这里。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RUN_RE = re.compile(r"[㐀-鿿]+")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    toks = _WORD_RE.findall(text)
    for run in _CJK_RUN_RE.findall(text):
        toks.extend(run[i:i + 2] for i in range(len(run) - 1))
        if len(run) == 1:
            toks.append(run)
    return toks


def bigram_jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def coverage(part: str, whole: str) -> float:
    """part 的切词有多少比例出现在 whole 里——用来判断一句话/一条事实有没有原文依据。"""
    sp = set(tokenize(part))
    if not sp:
        return 0.0
    return len(sp & set(tokenize(whole))) / len(sp)


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(c.values()) for c in self.docs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for c, dl in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                tf = c.get(t)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out.append(s)
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        ranked = sorted(enumerate(self.scores(query)), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]
