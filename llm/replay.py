"""重放：用 trace 里记下的玩家输入、模型回答、便宜判断，把一局原样再跑一遍。

引擎本身不知道它在被重放——它照常调后端、判断器、玩家；这三个都被换成从 trace 里按顺序取答案的替身。
任何一处对不上（请求的任务、判断的标签、输入发生的位置不一样）都会立刻报 ReplayMismatch，说明代码行为变了。
"""
from __future__ import annotations

from typing import Optional

from llm.base import LLMError, LLMRequest, LLMResponse
from llm.decider import Decider, Decision


class ReplayMismatch(RuntimeError):
    pass


class ReplayBackend:
    name = "replay"

    def __init__(self, events: list[dict]):
        self.calls = [e for e in events if e.get("event") == "llm"]
        self.i = 0

    def complete(self, req: LLMRequest) -> LLMResponse:
        if self.i >= len(self.calls):
            raise ReplayMismatch(f"记录里的模型回答用完了，但引擎又发了一次 {req.task} 请求")
        rec = self.calls[self.i]
        self.i += 1
        if rec["task"] != req.task:
            raise ReplayMismatch(f"第 {self.i} 次模型调用：记录里是 {rec['task']}，现在发的是 {req.task}")
        if rec.get("output") is None:
            raise LLMError(rec.get("error") or "记录里这次调用失败了")
        return LLMResponse(text=rec["output"], finish_reason=rec.get("finish_reason") or "stop", usage=rec.get("usage") or {},
                           latency_s=rec.get("latency_s") or 0.0)

    def list_models(self):
        return []


class ReplayDecider(Decider):
    name = "replay"

    def __init__(self, events: list[dict]):
        from engine.recording import RecordingDecider
        self.calls = [e for e in events if e.get("event") == "decide_io"]
        self.i = 0
        self._key = RecordingDecider.key

    def decide(self, tag: str, state: dict, questions: dict) -> Optional[dict[str, Decision]]:
        if self.i >= len(self.calls):
            raise ReplayMismatch(f"记录里的判断用完了，但引擎又判了一次 {tag}")
        rec = self.calls[self.i]
        self.i += 1
        if rec["tag"] != tag or rec["key"] != self._key(tag, state, questions):
            raise ReplayMismatch(f"第 {self.i} 次判断：记录里是 {rec['tag']}，现在是 {tag}（或者问的内容变了）")
        if rec["result"] is None:
            return None
        return {k: Decision(v["answer"], v["p"], v.get("probs") or {}) for k, v in rec["result"].items()}


class ReplayPlayer:
    stops_at_ending = True

    def __init__(self, events: list[dict]):
        self.inputs = [e for e in events if e.get("event") == "input"]
        self.i = 0

    def choose(self, options, ctx):
        from core.player_agent import PlayerAction
        if self.i >= len(self.inputs):
            return PlayerAction("quit")
        rec = self.inputs[self.i]
        self.i += 1
        at = "dialogue" if ctx is not None else "location"
        if rec["at"] != at:
            raise ReplayMismatch(f"第 {self.i} 次输入：记录里发生在{rec['at']}，现在是{at}")
        return PlayerAction(rec["kind"], rec["text"], rec["intent"], dict(rec["meta"]))
