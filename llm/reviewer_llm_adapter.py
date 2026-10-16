"""进度审核员：独立上下文、低温度，只判断「NPC 声称达成的信号」在这一轮里是否真的发生。

它看不到角色卡和剧情秘密，只看信号定义和对话原文——上下文隔离，避免被 NPC 的叙事带偏。
"""
from __future__ import annotations

from typing import Any

from config.settings import TaskParams
from llm.base import LLMBackend, LLMRequest
from llm.schemas import review_schema, validate_review
from llm.structured import call_json

SYSTEM = """你是叙事游戏的「剧情进度审核员」。NPC 声称这一轮对话达成了某些进度信号，你要逐个核实。

判定标准：
1. 只有当信号描述的行为在「本轮对话」里被明确的言行支撑时才判 accept=true；暗示、可能、即将发生都不算。
2. 主语要对：信号说「对方（{player}）做了什么」，就必须是{player}的话里体现；说「{npc}说明了什么」，就必须是{npc}的话里体现。
3. evidence 必须逐字摘抄本轮对话原文中的一小段（不超过 30 字）作为依据；找不到原文依据就判 false，evidence 写空字符串。
4. 拿不准时判 false。

只输出一个 JSON 对象，不要任何其他文字，不要代码块：
{{"verdicts": [{{"id": "信号id", "accept": true, "evidence": "原文摘抄", "reason": "一句话理由"}}]}}"""


class ReviewerLLMAdapter:
    def __init__(self, backend: LLMBackend, params: TaskParams, tracer: Any = None, repair_attempts: int = 1):
        self.backend = backend
        self.params = params
        self.tracer = tracer
        self.repair_attempts = repair_attempts

    def review(self, npc_name: str, player_name: str, signals: list[dict], context_lines: list[str],
               player_line: str, npc_line: str) -> dict[str, dict]:
        ids = [s["id"] for s in signals]
        defs = "\n".join(f"- {s['id']}：{s['desc']}" for s in signals)
        ctx = "\n".join(context_lines[-4:]) or "（无）"
        user = (f"# 待核实的信号\n{defs}\n\n# 之前的对话（仅供理解）\n{ctx}\n\n"
                f"# 本轮对话\n{player_name}：「{player_line}」\n{npc_name}：「{npc_line}」\n\n请逐个给出结论。")
        req = LLMRequest(
            task="review",
            messages=[{"role": "system", "content": SYSTEM.format(player=player_name, npc=npc_name)},
                      {"role": "user", "content": user}],
            temperature=self.params.temperature, max_tokens=self.params.max_tokens,
            json_schema=review_schema(ids), schema_name="signal_review",
            meta={"signals": ids, "player_line": player_line, "npc_line": npc_line},
        )
        return call_json(self.backend, req, lambda o: validate_review(o, ids), self.tracer, self.repair_attempts)
