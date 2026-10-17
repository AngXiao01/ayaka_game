"""录制：把玩家的输入、状态差异、便宜判断的结果记进 trace，供 tools/replay_game.py 重放。"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from core import eventlog
from core.player_agent import PlayerAction


def _jsonable(meta: dict) -> dict:
    try:
        return json.loads(json.dumps(meta, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return {}


class RecordingPlayer:
    """包住真正的玩家：每次选择之前先记“上次输入以来世界变了什么”，选完再记这次输入。"""

    def __init__(self, inner: Any, engine: Any):
        self.inner, self.engine = inner, engine
        self.n = 0
        self.last: dict | None = None

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)

    def _delta(self) -> None:
        snap = self.engine.snapshot()
        ops = eventlog.diff(self.last, snap) if self.last is not None else []
        if self.last is None:
            self.engine.tracer.log("state_init", state=snap, settings=self.engine.replay_settings())
        elif ops:
            self.engine.tracer.log("state_delta", before_input=self.n + 1, ops=ops)
        self.last = snap

    def choose(self, options, ctx):
        self._delta()
        action: PlayerAction = self.inner.choose(options, ctx)
        self.n += 1
        self.engine.tracer.log("input", n=self.n, at="dialogue" if ctx is not None else "location", kind=action.kind,
                               text=action.text, intent=action.intent, meta=_jsonable(action.meta))
        return action

    def finish(self) -> None:
        self._delta()


class RecordingDecider:
    """包住便宜判断（Jev 等）：每次调用的结果——包括命中缓存的、返回 None 的——都按顺序记下。"""

    def __init__(self, inner: Any, tracer: Any):
        self.inner, self.tracer = inner, tracer
        self.name = inner.name

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)

    @staticmethod
    def key(tag: str, state: dict, questions: dict) -> str:
        return hashlib.sha1(json.dumps([tag, state, questions], ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def decide(self, tag: str, state: dict, questions: dict):
        out = self.inner.decide(tag, state, questions)
        self.tracer.log("decide_io", tag=tag, key=self.key(tag, state, questions),
                        result=None if out is None else {k: {"answer": v.answer, "p": v.p, "probs": v.probs} for k, v in out.items()})
        return out
