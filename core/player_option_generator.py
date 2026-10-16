"""玩家选项：LLM 生成台词（按语气区分）→ 去重、去掉和玩家说过的话重复的 → 失败时用人物卡里的兜底台词。

动作类选项（出示线索、汇报、离开）由 DialogueSession 按世界状态确定性地追加，不交给模型。
台词选项是替玩家说话，所以还要过两道代码关（模型只提议）：
  剧透  出现了玩家还没听说过的人名 / 案情关键词（ctx.player_view 的 spoiler_terms 不在 known_text 和对话原文里）
  出戏  游戏、玩家、剧情、选项这类元话语
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from core.context import TurnContext
from llm.base import LLMError
from llm.player_option_llm_adapter import PlayerOptionLLMAdapter
from models.retrieval import bigram_jaccard

REPEAT_THRESHOLD = 0.6
OOC_RE = re.compile(r"游戏|玩家|剧情|选项|推进任务|系统|NPC|AI|人工智能|模型|派蒙", re.I)


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
        known = "\n".join([ctx.player_view.get("known_text", ""), ctx.history_summary]
                          + [h["text"] + (h.get("reply") or {}).get("action", "") for h in ctx.history])
        dropped: list[dict] = []
        out = self._polish(raw, said, known, ctx.player_view.get("spoiler_terms", []), dropped)
        if len(out) < 2:
            out = self._polish(raw + list(fallback), said, known, ctx.player_view.get("spoiler_terms", []), dropped)
        if dropped and self.tracer:
            self.tracer.log("options_filter", dropped=dropped)
        return [PlayerOption(o["text"], o["intent"], tone=o.get("tone", "")) for o in out], degraded

    def _polish(self, raw: list[dict], said: list[str], known: str = "", terms: Optional[list[str]] = None,
                dropped: Optional[list[dict]] = None) -> list[dict]:
        out: list[dict] = []
        for o in raw:
            if any(bigram_jaccard(o["text"], x) >= REPEAT_THRESHOLD for x in said + [y["text"] for y in out]):
                continue
            why = self._reject(o["text"], known, terms or [])
            if why:
                if dropped is not None and not any(d["text"] == o["text"] for d in dropped):
                    dropped.append({"text": o["text"], "reason": why})
                continue
            out.append(o)
        return out[: self.n]

    @staticmethod
    def _reject(text: str, known: str, terms: list[str]) -> str:
        m = OOC_RE.search(text)
        if m:
            return f"出戏（{m.group(0)}）"
        leaked = [t for t in terms if t and t in text and t not in known]
        return f"玩家还不知道：{'、'.join(leaked)}" if leaked else ""
