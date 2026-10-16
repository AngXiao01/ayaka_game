"""世界观知识库（外部只读记忆）：content/lore/*.md 按「## 标题」切块，BM25 检索。

NPC 不靠参数记忆去编稻妻设定，而是需要时调用 lookup_lore 查这里——
权威来源可控、可更新，也能在 trace 里看到它到底查到了什么。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from models.retrieval import BM25


@dataclass
class LoreChunk:
    title: str
    text: str
    source: str


class LoreBase:
    def __init__(self, chunks: list[LoreChunk]):
        self.chunks = chunks
        self.index = BM25([f"{c.title}\n{c.title}\n{c.text}" for c in chunks])   # 标题加权

    @classmethod
    def load(cls, lore_dir: Path) -> "LoreBase":
        chunks: list[LoreChunk] = []
        for path in sorted(lore_dir.glob("*.md")) if lore_dir.is_dir() else []:
            title, buf = None, []
            for line in path.read_text(encoding="utf-8").splitlines() + ["## "]:
                if line.startswith("## "):
                    if title and "".join(buf).strip():
                        chunks.append(LoreChunk(title, "\n".join(buf).strip(), path.name))
                    title, buf = line[3:].strip(), []
                elif title is not None:
                    buf.append(line)
        return cls(chunks)

    def search(self, query: str, k: int = 2) -> list[LoreChunk]:
        return [self.chunks[i] for i, _ in self.index.top(query, k)]
