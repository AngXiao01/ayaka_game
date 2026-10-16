"""分支 agent：tree 层里的模型部分。

剧情分支本身写在 quests/*.json 里（幕、信号及依赖、提议、结局），由 GodEnv 按规则推进。
原来关键抉择只能从菜单里点；现在玩家也可以直接打字说出自己的处置方案，由这里判断它对应哪个预设方案。

约束：
1. 只能在「此刻可以提出的预设方案」里选，或者判为 none——造不出新分支
2. 证据要落地：给出的 evidence 必须能在玩家原话里找到，找不到就不算
3. 注入轮不判：输入防护命中的这一轮直接跳过
4. 判定之后照常走菜单那条路（GodEnv.propose），NPC 表态、审核、结算都不变
"""
from __future__ import annotations

from typing import Any, Optional

from core.god_env import GodEnv
from llm.base import LLMError
from models.retrieval import coverage

EVIDENCE_COVERAGE = 0.8


def rules_classify(text: str, proposals: list[dict]) -> dict:
    hits = [(p, next(k for k in p.get("keywords", []) if k and k in text))
            for p in proposals if any(k and k in text for k in p.get("keywords", []))]
    if len(hits) != 1:          # 没提，或者两个方向都沾边（拿不准）
        return {"proposal": "none", "evidence": "", "reason": "没有明确提出" if not hits else "同时沾到多个方案"}
    p, k = hits[0]
    return {"proposal": p["flag"], "evidence": k, "reason": f"关键词「{k}」"}


class BranchAgent:
    def __init__(self, god: GodEnv, adapter: Any = None, mode: str = "llm", tracer: Any = None):
        self.god = god
        self.adapter = adapter
        self.mode = mode
        self.tracer = tracer

    def classify(self, npc_id: str, text: str, suspicious: bool = False, degraded: bool = False) -> Optional[dict]:
        """返回被采纳的预设方案（quest.proposals 里的一项），或 None。"""
        props = self.god.proposals_for(npc_id)
        if self.mode == "off" or not props or not text.strip():
            return None
        if suspicious:
            self._log("skip", text, None, False, "输入防护命中，这一轮不判")
            return None
        used, out = ("rules" if degraded else self.mode), None
        if used == "llm" and self.adapter is not None:
            try:
                out = self.adapter.classify(text, props)
            except LLMError as e:
                used = "rules"
                if self.tracer:
                    self.tracer.log("fallback", where="branch", error=str(e))
        if out is None:
            used, out = "rules", rules_classify(text, props)
        prop = next((p for p in props if p["flag"] == out["proposal"]), None)
        ok, why = False, out.get("reason", "")
        if prop is None:
            why = why or "没有提出任何方案"
        elif not out["evidence"] or not (out["evidence"] in text or coverage(out["evidence"], text) >= EVIDENCE_COVERAGE):
            why = f"证据「{out['evidence']}」在玩家原话里找不到"
        else:
            ok = True
        self._log(used, text, out, ok, why)
        return prop if ok else None

    def _log(self, mode: str, text: str, out: Optional[dict], ok: bool, reason: str) -> None:
        if self.tracer:
            self.tracer.log("branch", mode=mode, text=text, result=out, applied=ok, reason=reason)
