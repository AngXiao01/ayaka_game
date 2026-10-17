"""长期记忆（情节记忆）：JSONL 存储，BM25 + 重要度 + 时近性混合打分。

写入时的正确性保障：
- 事实要有原文依据：facts 必须在本次对话里找得到支撑（coverage 阈值），否则丢弃并记 trace
- 去重合并：与已有同类记忆高度相似时合并（重要度取大、刷新时间、累计次数），不无限堆叠
- 来源可追溯：每条记忆带 provenance（第几次见面、哪一幕、结束原因）
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from models.retrieval import BM25, bigram_jaccard

DEDUP_THRESHOLD = 0.6


@dataclass
class MemoryItem:
    text: str
    kind: str = "dialogue_summary"      # dialogue_summary | fact
    importance: int = 3                 # 1~5
    world_time: int = 0
    topics: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    hits: int = 1                       # 被合并的次数
    accessed: int = 0                   # 最近一次被取用（检索到）的世界回合；时近性按它和创建时间里较晚的算
    id: str = ""                        # 由内容生成（同一局重放得到同样的编号，反思的证据才对得上）
    ts: float = field(default_factory=lambda: round(time.time(), 3))     # 墙钟，只供人看；先后顺序以库里的位置为准

    def __post_init__(self) -> None:
        if not self.id:
            self.id = hashlib.sha1(f"{self.kind}|{self.world_time}|{self.text}".encode()).hexdigest()[:12]


class MemoryStore:
    def __init__(self, path: Path):
        self.path = path
        self.items: list[MemoryItem] = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        self.items.append(MemoryItem(**json.loads(line)))
                    except (TypeError, ValueError):
                        continue    # 坏行跳过，不让一行脏数据毁掉整个存档
        self._index: Optional[BM25] = None
        self._dirty = False             # 有条目的“最近取用时间”变了，文件里的还是旧的

    def flush(self) -> None:
        """把检索时刷新的“最近取用”写回文件。每次检索都重写整个文件太浪费，所以攒到会话结束 / 存档时一起写。"""
        if self._dirty:
            self._rewrite()

    def _rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(asdict(m), ensure_ascii=False) + "\n" for m in self.items), encoding="utf-8")
        tmp.replace(self.path)       # 原子替换，写一半崩溃也不会留下半个文件
        self._dirty = False

    def add(self, item: MemoryItem) -> tuple[MemoryItem, bool]:
        """返回 (落库的条目, 是否为合并)。"""
        self._index = None
        dup = max((m for m in self.items if m.kind == item.kind),
                  key=lambda m: bigram_jaccard(m.text, item.text), default=None)
        if dup is not None and bigram_jaccard(dup.text, item.text) >= DEDUP_THRESHOLD:
            dup.importance = max(dup.importance, item.importance)
            dup.world_time = max(dup.world_time, item.world_time)
            dup.hits += 1
            dup.topics = list(dict.fromkeys(dup.topics + item.topics))[:8]
            self._rewrite()
            return dup, True
        self.items.append(item)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        return item, False

    def retrieve(self, query: str, k: int = 5, now: Optional[int] = None, exclude_kinds: tuple = (), touch: bool = True) -> list[MemoryItem]:
        """取最相关的 k 条（按时间顺序给模型看）。取到的条目会刷新“最近被用到”的时间（touch=False 则只看不动），
        经常被想起的事就不会因为发生得早而沉下去——生成式智能体里 recency 也是按最近取用算的。"""
        if not self.items or k <= 0:
            return []
        if self._index is None:
            self._index = BM25([m.text + " " + " ".join(m.topics) for m in self.items])
        now = now if now is not None else max(m.world_time for m in self.items)
        rel = self._index.scores(query)
        top = max(rel) or 1.0

        def score(i: int) -> float:
            m = self.items[i]
            recency = 1.0 / (1.0 + max(0, now - max(m.world_time, m.accessed)) / 20.0)
            return 0.6 * rel[i] / top + 0.25 * m.importance / 5 + 0.15 * recency

        ranked = [i for i in sorted(range(len(self.items)), key=score, reverse=True) if self.items[i].kind not in exclude_kinds][:k]
        got = [self.items[i] for i in ranked]
        if touch:
            for m in got:
                if now > m.accessed:
                    m.accessed, self._dirty = now, True
        return sorted(got, key=lambda m: m.world_time)
