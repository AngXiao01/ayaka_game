"""把行动日志（core/journal.py）接到游戏的三个入口上：模型、便宜判定、玩家。

三个都是“透明的包装”：没有日志可用时，行为和原来一模一样，只多一次追加写。
包装的位置在录制（trace）的**里面**——命中日志的调用照常出现在 trace 里，重放工具看到的和现问的没有区别。
"""
from __future__ import annotations

import dataclasses
from typing import Any

from core.journal import Journal, content_key
from core.player_agent import PlayerAction
from engine.recording import RecordingDecider, _jsonable
from llm.base import LLMRequest, LLMResponse
from llm.decider import Decision


class JournaledBackend:
    def __init__(self, inner: Any, journal: Journal):
        self.inner, self.journal = inner, journal

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)

    def complete(self, req: LLMRequest) -> LLMResponse:
        key = content_key(dataclasses.asdict(req))
        hit, payload = self.journal.serve("llm", key)
        if hit:
            return LLMResponse(**payload)
        resp = self.inner.complete(req)
        self.journal.record("llm", key, dataclasses.asdict(resp))      # 失败（抛异常）的不记：下次照样现问
        return resp


class JournaledDecider:
    def __init__(self, inner: Any, journal: Journal):
        self.inner, self.journal = inner, journal
        self.name = inner.name

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)

    def decide(self, tag: str, state: dict, questions: dict):
        key = RecordingDecider.key(tag, state, questions)
        hit, payload = self.journal.serve("decide", key)
        if hit:
            return None if payload is None else {k: Decision(v["answer"], v["p"], v.get("probs") or {}) for k, v in payload.items()}
        out = self.inner.decide(tag, state, questions)
        self.journal.record("decide", key, None if out is None else
                            {k: {"answer": v.answer, "p": v.p, "probs": v.probs} for k, v in out.items()})
        return out


class JournaledPlayer:
    def __init__(self, inner: Any, journal: Journal):
        self.inner, self.journal = inner, journal

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)

    def choose(self, options, ctx):
        key = content_key("dialogue" if ctx is not None else "location", [(o.kind, o.text, o.intent) for o in options])
        hit, p = self.journal.serve_input(key)
        if hit:
            return PlayerAction(p["kind"], p["text"], p["intent"], dict(p["meta"]))
        action = self.inner.choose(options, ctx)
        self.journal.record("input", key, {"kind": action.kind, "text": action.text, "intent": action.intent,
                                           "meta": _jsonable(action.meta)})
        return action


def wrap_router(router: Any, journal: Journal) -> None:
    """路由里每个后端都包一层（原地改，已经包过的不再包）。"""
    if not isinstance(router.default, JournaledBackend):
        router.default = JournaledBackend(router.default, journal)
    for task, b in list(router.backends.items()):
        if not isinstance(b, JournaledBackend):
            router.backends[task] = JournaledBackend(b, journal)
