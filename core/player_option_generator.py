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
from llm.schemas import FREE_INTENT
from models.retrieval import bigram_jaccard

ACT_INTENT = {"leave": "告别", "invite": "邀请"}     # 台词本身就在做这件事时，给对方看的意图标签；其余台词和自己打字一样算自由发言
ECHO_THRESHOLD = 0.45          # 和对方最近说的话这么像，就是把对方的台词塞给了玩家
REPEAT_THRESHOLD = 0.6
OOC_RE = re.compile(r"游戏|玩家|剧情|选项|推进任务|系统|NPC|AI|人工智能|模型|派蒙", re.I)


@dataclass
class PlayerOption:
    text: str
    intent: str
    kind: str = "say"           # say | show_menu | clue | report | propose | confront_menu | testimony | evidence | leave | cancel
                                # | talk | event | examine | move | wait | view | quit
    meta: dict = field(default_factory=dict)


class PlayerOptionGenerator:
    def __init__(self, adapter: Optional[PlayerOptionLLMAdapter], mode: str = "llm", n: int = 3,
                 tracer: Any = None):
        self.adapter = adapter
        self.mode = mode
        self.n = n
        self.tracer = tracer
        self.verify = None          # (足迹, [台词]) → 要丢掉的下标（模型判：替玩家编了没有的经历）；引擎装配

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
        heard = [h["text"] for h in ctx.history if h["speaker"] == "npc"][-3:]
        known = "\n".join([ctx.player_view.get("known_text", ""), ctx.history_summary]
                          + [h["text"] + (h.get("reply") or {}).get("action", "") for h in ctx.history])
        dropped: list[dict] = []
        terms = ctx.player_view.get("spoiler_terms", [])
        if raw or (self.mode == "llm" and self.adapter is not None and not force_static and not degraded):
            out = self._polish(raw, said, known, terms, dropped, heard)  # 模型说给几句就几句，0 句也行
        else:                                                          # 模型挂了 / 降级 / 静态模式：用兜底台词
            out = self._polish(list(fallback), said, known, terms, dropped)
        if self.verify and out and ctx.player_view.get("footprint"):
            bad = self.verify(ctx.player_view["footprint"], [o["text"] for o in out]) or set()
            dropped += [{"text": o["text"], "reason": "编了没有的经历"} for i, o in enumerate(out) if i in bad]
            out = [o for i, o in enumerate(out) if i not in bad]
        if dropped and self.tracer:
            self.tracer.log("options_filter", dropped=dropped)
        acts = {a["id"]: a for a in ctx.player_view.get("actions", [])}
        res, used = [], set()
        for o in out:
            act = o.get("act") or ""
            if not act and o.get("intent") == "告别" and "leave" in acts:     # 只有作者写的兜底台词带意图
                act = "leave"                          # 告别的话就是告辞：说了就走，不再另点按钮
            a = acts.get(act)
            if a and act not in used:                  # 只认此刻真能做的事，每件事一句
                used.add(act)
                res.append(PlayerOption(o["text"], ACT_INTENT.get(act, ""), kind=a["kind"],
                                        meta={**a.get("meta", {}), "line": True, "act": act, "act_label": a["tag"]}))
            else:
                res.append(PlayerOption(o["text"], o.get("intent") or FREE_INTENT))
        return res, degraded

    def _polish(self, raw: list[dict], said: list[str], known: str = "", terms: Optional[list[str]] = None,
                dropped: Optional[list[dict]] = None, heard: Optional[list[str]] = None) -> list[dict]:
        out: list[dict] = []
        for o in raw:
            if any(bigram_jaccard(o["text"], x) >= ECHO_THRESHOLD or (len(o["text"]) >= 8 and o["text"][:8] in x)
                   for x in heard or []):
                if dropped is not None:                # 把对方刚说的话塞进玩家嘴里：串了视角
                    dropped.append({"text": o["text"], "reason": "复述了对方的话"})
                continue
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
