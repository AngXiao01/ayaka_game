"""玩家选项：LLM 生成台词（按语气区分）→ 去重、去掉和玩家说过的话重复的 → 失败时用人物卡里的兜底台词。

动作类选项（出示线索、汇报、离开）由 DialogueSession 按世界状态确定性地追加，不交给模型。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.context import TurnContext
from llm.base import LLMError
from llm.player_option_llm_adapter import PlayerOptionLLMAdapter
from models.retrieval import bigram_jaccard

REPEAT_THRESHOLD = 0.6


@dataclass
class PlayerOption:
    text: str
    intent: str
    kind: str = "say"           # say | show_menu | clue | report | propose | confront_menu | testimony | evidence | leave | cancel
                                # | talk | event | examine | move | wait | view | quit
    meta: dict = field(default_factory=dict)
    tone: str = ""


class PlayerOptionGenerator:
    def __init__(self, adapter: Optional[PlayerOptionLLMAdapter], mode: str = "llm", n: int = 3,
                 tracer: Any = None):
        self.adapter = adapter
        self.mode = mode
        self.n = n
        self.tracer = tracer

    def generate(self, ctx: TurnContext, npc_name: str, fallback: list[dict],
                 force_static: bool = False) -> tuple[list[PlayerOption], bool]:
        """返回 (台词选项, 是否用了兜底)。force_static：预算降级时不调 LLM。"""
        raw: list[dict] = []
        degraded = False
        if self.mode == "llm" and self.adapter is not None and not force_static:
            try:
                raw = self.adapter.generate(ctx, npc_name, self.n)
            except LLMError as e:
                degraded = True
                if self.tracer:
                    self.tracer.log("fallback", where="player_options", error=str(e))
        said = [h["text"] for h in ctx.history if h["speaker"] == "player"]
        out = self._polish(raw, said)
        if len(out) < 2:
            out = self._polish(raw + list(fallback), said)
        return [PlayerOption(o["text"], o["intent"], tone=o.get("tone", "")) for o in out], degraded

    def _polish(self, raw: list[dict], said: list[str]) -> list[dict]:
        out: list[dict] = []
        for o in raw:
            if any(bigram_jaccard(o["text"], x) >= REPEAT_THRESHOLD for x in said + [y["text"] for y in out]):
                continue
            out.append(o)
        return out[: self.n]
