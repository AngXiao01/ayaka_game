"""重放：用 trace 里记下的玩家输入、模型回答、便宜判断，把一局原样再跑一遍。

引擎本身不知道它在被重放——它照常调后端、判断器、玩家；这三个都被换成从 trace 里按顺序取答案的替身。
任何一处对不上（请求的任务、判断的标签、输入发生的位置不一样）都会立刻报 ReplayMismatch，说明代码行为变了。
并行发出的几次调用在记录里是按“谁先回来”排的：取记录时先试下一条，对不上再在没用过的里找第一条对得上的（并记一次乱序）；
串行录的局不该有乱序（金标准局的测试要求是 0）。
"""
from __future__ import annotations

from typing import Optional

from llm.base import LLMError, LLMRequest, LLMResponse
from llm.decider import Decider, Decision


class ReplayMismatch(RuntimeError):
    pass


def _take(records: list[dict], used: list[bool], matches) -> tuple[Optional[int], bool]:
    """取下一条没用过的记录。正常就是按顺序的下一条；并行录下来的局里，几次同时发出的调用是按“谁先回来”记的，
    顺序可能和重放时不同——这时在没用过的里找第一条对得上的，并算一次“乱序”。找不到才是真的对不上。"""
    nxt = next((i for i, u in enumerate(used) if not u), None)
    if nxt is None:
        return None, False
    if matches(records[nxt]):
        return nxt, False
    for i in range(nxt + 1, len(records)):
        if not used[i] and matches(records[i]):
            return i, True
    return None, False


class ReplayBackend:
    name = "replay"

    def __init__(self, events: list[dict]):
        self.calls = [e for e in events if e.get("event") == "llm"]
        self.used = [False] * len(self.calls)
        self.reordered = 0

    @property
    def i(self) -> int:
        return sum(self.used)

    def complete(self, req: LLMRequest) -> LLMResponse:
        if all(self.used):
            raise ReplayMismatch(f"记录里的模型回答用完了，但引擎又发了一次 {req.task} 请求")
        k, out_of_order = _take(self.calls, self.used, lambda r: r["task"] == req.task)
        if k is None:
            nxt = next(i for i, u in enumerate(self.used) if not u)
            raise ReplayMismatch(f"第 {self.i + 1} 次模型调用：记录里是 {self.calls[nxt]['task']}，现在发的是 {req.task}")
        self.used[k] = True
        self.reordered += out_of_order
        rec = self.calls[k]
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
        self.used = [False] * len(self.calls)
        self.reordered = 0
        self._key = RecordingDecider.key

    @property
    def i(self) -> int:
        return sum(self.used)

    def decide(self, tag: str, state: dict, questions: dict) -> Optional[dict[str, Decision]]:
        if all(self.used):
            raise ReplayMismatch(f"记录里的判断用完了，但引擎又判了一次 {tag}")
        key = self._key(tag, state, questions)
        k, out_of_order = _take(self.calls, self.used, lambda r: r["tag"] == tag and r["key"] == key)
        if k is None:
            nxt = next(i for i, u in enumerate(self.used) if not u)
            raise ReplayMismatch(f"第 {self.i + 1} 次判断：记录里是 {self.calls[nxt]['tag']}，现在是 {tag}（或者问的内容变了）")
        self.used[k] = True
        self.reordered += out_of_order
        rec = self.calls[k]
        if rec["result"] is None:
            return None
        return {k2: Decision(v["answer"], v["p"], v.get("probs") or {}) for k2, v in rec["result"].items()}


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
