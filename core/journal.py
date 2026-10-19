"""行动日志：把一次行动里“模型答了什么、玩家选了什么”按内容记下来，崩了之后重启时不用重新问。

和 core/uow.py 的分工（这是“谨慎”的关键）：
  · uow 保证**状态**：没提交的行动，碰过的文件一律还原到行动开始之前——恢复永远先回滚，从不“跳过”有副作用的节点；
  · journal 只保证**不白花钱、不换说法**：回滚之后，同一段代码带着同样的世界重新跑一遍，
    遇到“以前问过、问法一字不差”的模型调用 / 判定 / 玩家选择，直接用当时的答案，其余照常现问。
    有副作用的节点因此还是照常重新执行、照常走事务，只是它们拿到的输入和上次一样。

为什么不存“节点跑到哪、局部状态是什么”：世界状态在内存里，只在提交时才落盘；中途存检查点就得序列化一整个世界，
还要保证它和旁边的记忆文件互相对得上——那是在重新发明事务。按内容记答案不需要这些：

  · 记的是 `(种类, 请求内容的哈希) → 答案`，用的时候**请求内容必须完全一样**才命中，所以日志永远不会让代码拿到别的问题的答案；
    世界、代码、随机数哪里不一样了，请求就变，自然未命中，退回现问——最坏情况就是没有日志；
  · 玩家的选择按顺序记，每一次连同“当时的选项”一起校验，对不上就从那里起全部现问（后面的选择已经对不上了）；
  · 日志绑定事务号：事务提交了（存档里的号 ≥ 日志的号）日志就是过期的，直接丢弃。

文件是只追加的 JSONL，每条 fsync；最后一行被撕掉（写到一半崩了）就忽略它。这个文件不依赖游戏的任何东西。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from core import crashpoints


def content_key(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:20]


class Journal:
    def __init__(self, path: Path, uow: Any):
        self.path, self.uow = Path(path), uow
        self._lock = threading.Lock()
        self.meta: Optional[dict] = None
        self.started = False
        self._entries: list[dict] = []            # 上次没提交的行动留下的记录（只读）
        self._used: list[bool] = []
        self._cursor = 0                          # 玩家选择按顺序取：下一条没用过的 input 记录
        self._inputs_blocked = False
        self.served: Counter = Counter()          # 这次行动里用日志答了多少次（按种类）
        self.live: Counter = Counter()            # 日志里还有记录时，现问了多少次
        self._load()
        uow.on_commit.append(self._after_commit)

    # ------------------------------------------------------------ 读

    def _load(self) -> None:
        if not self.path.exists():
            return
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                break                              # 最后一行写到一半：之前的都有效，这行忽略
        head = rows[0] if rows else None
        if not head or head.get("kind") != "begin" or head.get("tx") != self.uow.tx:
            self.path.unlink(missing_ok=True)      # 属于已经提交（或者更早）的事务：过期了
            return
        self.started, self.meta = True, head.get("meta")
        self._entries = rows[1:]
        self._used = [False] * len(self._entries)

    @property
    def resuming(self) -> bool:
        """上次没提交的行动留下了日志，而且还有没用上的记录。"""
        return any(not u for u in self._used)

    # ------------------------------------------------------------ 写

    def _append(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def begin(self, meta: Optional[dict] = None) -> bool:
        """一次行动开始。返回 True = 新的一次；False = 接着上次没提交的那次（meta 在 self.meta 里）。"""
        with self._lock:
            if self.started:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")
            self._append({"kind": "begin", "tx": self.uow.tx, "meta": meta or {}})
            self.started, self.meta = True, meta or {}
            return True

    def record(self, kind: str, key: str, payload: Any) -> None:
        with self._lock:
            if not self.started:
                return
            self._append({"kind": kind, "key": key, "payload": payload})
            crashpoints.hit("journal.after_append")
            if self.resuming:
                self.live[kind] += 1

    def _after_commit(self) -> None:
        self.path.unlink(missing_ok=True)
        self.started, self.meta, self._entries, self._used, self._cursor = False, None, [], [], 0
        self._inputs_blocked = False

    # ------------------------------------------------------------ 用

    def serve(self, kind: str, key: str) -> tuple[bool, Any]:
        """模型调用 / 判定：有请求内容一字不差、还没用过的记录就用它。返回 (命中, 答案)。"""
        with self._lock:
            for i, e in enumerate(self._entries):
                if not self._used[i] and e["kind"] == kind and e["key"] == key:
                    self._used[i] = True
                    self.served[kind] += 1
                    return True, e["payload"]
        return False, None

    def serve_input(self, key: str) -> tuple[bool, Any]:
        """玩家的选择：按顺序取下一条，而且当时的选项必须一样；对不上就从这里起不再用日志里的选择。"""
        with self._lock:
            if self._inputs_blocked:
                return False, None
            while self._cursor < len(self._entries) and (self._used[self._cursor] or self._entries[self._cursor]["kind"] != "input"):
                self._cursor += 1
            if self._cursor >= len(self._entries):
                return False, None
            e = self._entries[self._cursor]
            if e["key"] != key:
                self._inputs_blocked = True
                return False, None
            self._used[self._cursor] = True
            self.served["input"] += 1
            return True, e["payload"]
