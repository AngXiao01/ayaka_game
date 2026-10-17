"""反思：把一次谈话里零碎的事，合成“我对对方的看法 / 我的疑问”，写进长期记忆（记忆流的第三步）。

只写 NPC 自己心里琢磨出来的东西，不是复述发生了什么（那是摘要的活），也不新增事实：看法要有谈话里的依据。
下一次见面，这份看法会作为“你自己上次琢磨出来的”放进上下文——人和人的关系是靠这种累积的印象走的。
"""
from __future__ import annotations

from typing import Any

from config.settings import TaskParams
from llm.base import LLMBackend, LLMError, LLMRequest
from llm.schemas import ValidationError, _str, _str_list
from llm.structured import call_json

SYSTEM = """你是{npc}的内心。刚和{player}谈完一次话，请你以第一人称，琢磨出两样东西：

- impression：你现在怎么看{player}——是个什么样的人，值不值得信任，哪里让你放心或不放心。一句话，不超过 40 字。要有谈话里的依据，不要凭空下结论。
- doubts：你心里还没解开的疑问（最多 3 条，每条一句话）。比如对方为什么这么做、某句话对不上、想再问的事。没有就留空。
- next：下次见面你打算怎么做（一句话，可留空）。

不要复述谈话的经过。只输出一个 JSON 对象：{{"impression": "……", "doubts": ["……"], "next": "……"}}，不要任何其他文字。"""

SCHEMA = {"type": "object", "properties": {"impression": {"type": "string"}, "doubts": {"type": "array", "items": {"type": "string"}},
                                           "next": {"type": "string"}}, "required": ["impression", "doubts", "next"]}


def validate_reflection(obj: Any) -> dict:
    if not isinstance(obj, dict):
        raise ValidationError("顶层必须是 JSON 对象")
    impression = _str(obj.get("impression"))[:80]
    if not impression:
        raise ValidationError("impression 为空")
    return {"impression": impression, "doubts": [x[:60] for x in _str_list(obj.get("doubts"))[:3]], "next": _str(obj.get("next"))[:60]}


class ReflectionLLMAdapter:
    def __init__(self, backend: LLMBackend, params: TaskParams, tracer: Any = None, repair_attempts: int = 1):
        self.backend, self.params, self.tracer, self.repair_attempts = backend, params, tracer, repair_attempts

    def reflect(self, npc: str, player: str, transcript: str, previous: list[str]) -> dict:
        prev = "\n".join(f"- {x}" for x in previous) or "（这是第一次）"
        req = LLMRequest(task="reflect", messages=[
            {"role": "system", "content": SYSTEM.format(npc=npc, player=player)},
            {"role": "user", "content": f"# 你之前琢磨出的看法\n{prev}\n\n# 这次谈话\n{transcript[-2500:]}"}],
            temperature=self.params.temperature, max_tokens=min(400, self.params.max_tokens),
            json_schema=SCHEMA, schema_name="reflection", meta={"npc": npc, "player_name": player, "transcript": transcript[:300]})
        try:
            return call_json(self.backend, req, validate_reflection, self.tracer, self.repair_attempts)
        except LLMError as e:
            if self.tracer:
                self.tracer.log("fallback", where="reflect", error=str(e))
            return {}
