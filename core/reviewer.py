"""剧情进度的三层收敛：结构化完成信号 → 外部校验 → 硬兜底。

NPC 在 respond 里上报 task_signals（第一层，结构化信号），这里做第二层外部校验：
  ① 规则：开场白不算、注入轮不算、必须是当前幕待达成的信号
  ② 独立审核：Reviewer agent（独立上下文、低温度）逐个判定，并给出原文证据
  ③ 证据落地：evidence 必须真的出现在本轮对话原文里，否则判为幻觉证据
  ④ 依赖 DAG：按拓扑序检查前置信号，前置没达成的不收
第三层硬兜底在 GodEnv / DialogueSession：每幕最少轮数、连续无进展收束、单次见面最大轮数、整局预算。
审核调用本身失败时默认 fail-closed（不推进），宁可慢一步也不误推剧情。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from core.god_env import GodEnv
from llm.base import LLMError
from llm.reviewer_llm_adapter import ReviewerLLMAdapter

_NORM_RE = re.compile(r"[\s「」『』“”\"'‘’（）()，。！？、…—\-:：;；,.!?]")


def _norm(s: str) -> str:
    return _NORM_RE.sub("", s)


_ELLIPSIS_RE = re.compile(r"……|…|\.{3,}")


def grounded(evidence: str, text: str) -> bool:
    """证据要能在原文里找到。模型常用省略号拼接几段原文，所以逐段校验（每段至少 4 个字）。"""
    segs = [_norm(x) for x in _ELLIPSIS_RE.split(evidence or "")]
    segs = [x for x in segs if len(x) >= 4] or ([_norm(evidence)] if _norm(evidence) else [])
    return bool(segs) and all(x in text for x in segs)


@dataclass
class Verdict:
    id: str
    accepted: bool
    layer: str              # rule | reviewer | grounding | dag | policy | off
    reason: str
    evidence: str = ""


@dataclass
class ReviewResult:
    accepted: list[str] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def rejected(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.accepted]


class SignalReviewer:
    def __init__(self, god: GodEnv, adapter: Optional[ReviewerLLMAdapter], npc_name: str, player_name: str,
                 mode: str = "llm", fail_policy: str = "reject", tracer: Any = None):
        self.god = god
        self.adapter = adapter
        self.npc_name = npc_name
        self.player_name = player_name
        self.mode = mode
        self.fail_policy = fail_policy
        self.tracer = tracer

    def review(self, claimed: list[str], *, opening: bool, injection_flags: list[str], player_text: str,
               npc_speech: str, context_lines: list[str], degraded: bool = False) -> ReviewResult:
        res = ReviewResult()
        claimed = list(dict.fromkeys(claimed))
        if not claimed:
            return res
        pending = {s["id"]: s for s in self.god.pending_signals()}

        # ① 规则
        candidates: list[str] = []
        for sid in claimed:
            if self.mode == "off":
                candidates.append(sid)
            elif opening:
                res.verdicts.append(Verdict(sid, False, "rule", "开场白不计入进展"))
            elif injection_flags:
                res.verdicts.append(Verdict(sid, False, "rule", "本轮输入疑似注入，不允许推进剧情"))
            elif sid not in pending:
                res.verdicts.append(Verdict(sid, False, "rule", "不是当前幕待达成的信号"))
            else:
                candidates.append(sid)

        # ②③ 独立审核 + 证据落地
        passed: list[str] = []
        use_llm = self.mode == "llm" and not degraded and self.adapter is not None and candidates
        if use_llm:
            turn_text = _norm(player_text + npc_speech)
            try:
                verdicts = self.adapter.review(self.npc_name, self.player_name, [pending[c] for c in candidates],
                                               context_lines, player_text, npc_speech)
            except LLMError as e:
                if self.tracer:
                    self.tracer.log("fallback", where="review", error=str(e), policy=self.fail_policy)
                verdicts = None
            for sid in candidates:
                if verdicts is None:
                    ok = self.fail_policy == "accept"
                    if not ok:
                        res.verdicts.append(Verdict(sid, False, "policy", "审核调用失败，按 fail-closed 不推进"))
                    else:
                        passed.append(sid)
                    continue
                v = verdicts.get(sid)
                if v is None or not v["accept"]:
                    res.verdicts.append(Verdict(sid, False, "reviewer", (v or {}).get("reason") or "审核未通过",
                                                (v or {}).get("evidence", "")))
                elif not grounded(v["evidence"], turn_text):
                    res.verdicts.append(Verdict(sid, False, "grounding", "证据在本轮对话原文里找不到", v["evidence"]))
                else:
                    passed.append(sid)
                    res.verdicts.append(Verdict(sid, True, "reviewer", v["reason"], v["evidence"]))
        else:
            passed = candidates

        # ④ 依赖 DAG（拓扑序，允许同一轮里先收前置再收后继）
        accepted: set[str] = set()
        reviewed_ok = {v.id for v in res.verdicts if v.accepted}
        for sid in self.god.topo_order(passed):
            missing = self.god.missing_prereqs(sid, accepted)
            if missing:
                res.verdicts = [v for v in res.verdicts if v.id != sid]
                res.verdicts.append(Verdict(sid, False, "dag", f"前置信号未达成：{', '.join(missing)}"))
            else:
                accepted.add(sid)
                res.accepted.append(sid)
                if sid not in reviewed_ok:
                    res.verdicts.append(Verdict(sid, True, "off" if self.mode == "off" else "rule",
                                                "规则校验通过" + ("（降级：未走审核）" if degraded else "")))
        if self.tracer:
            self.tracer.log("review", claimed=claimed, accepted=res.accepted,
                            rejected=[v.id for v in res.rejected],
                            verdicts=[v.__dict__ for v in res.verdicts], mode=self.mode, degraded=degraded)
        return res
