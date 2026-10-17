"""反思：把一次谈话里零碎的事，合成“我对对方的看法 / 我的疑问”，写进长期记忆（记忆流的第三步）。

分两步（生成式智能体的做法）：先看最近的几条记忆，挑出 2~3 个“此刻最想弄明白的问题”（focal_questions）；
代码拿每个问题去整个记忆库里检索相关记忆；再让模型对着“问题 + 各自的相关记忆”下结论（reflect）。
这样结论不只是对刚才那几句话的反应——会把很久以前的事也翻出来对一对。

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
- evidence：你的看法和疑问是根据下面「最近的记忆」里的哪几条得出的（写它们的编号，至少 1 条）。

不要复述谈话的经过。只输出一个 JSON 对象：{{"impression": "……", "doubts": ["……"], "next": "……", "evidence": [0, 2]}}，不要任何其他文字。"""

FOCUS_SYSTEM = """你是{npc}的内心。下面是你最近的几条记忆。请想一想：凭这些，你此刻最想弄明白的 2~3 个问题是什么？
问题要具体、有指向——关于{player}这个人，或者眼下这件事里对不上、想验证的地方（例如“他为什么……”“她说的……是不是真的”）。
不要问已经有答案的事，不要复述记忆。只输出一个 JSON 对象：{{"questions": ["……", "……"]}}，不要任何其他文字。"""

FOCUS_SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "string"}}}, "required": ["questions"]}

GROUPED = """

下面「想弄明白的问题」每个都附了你翻出来的相关记忆。请先对着每个问题的那几条记忆想一想，再合成你对{player}的整体看法；evidence 写你最终依据的记忆编号（编号是全局的，不是每个问题内部的序号）。"""

SCHEMA = {"type": "object", "properties": {"impression": {"type": "string"}, "doubts": {"type": "array", "items": {"type": "string"}},
                                           "next": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "integer"}}},
          "required": ["impression", "doubts", "next", "evidence"]}


def validate_reflection(obj: Any, n_memories: int = 0) -> dict:
    if not isinstance(obj, dict):
        raise ValidationError("顶层必须是 JSON 对象")
    impression = _str(obj.get("impression"))[:80]
    if not impression:
        raise ValidationError("impression 为空")
    ev = []
    for x in obj.get("evidence") or []:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n_memories and i not in ev:
            ev.append(i)
    if n_memories and not ev:
        raise ValidationError("evidence 必须写出依据的记忆编号（0~%d）" % (n_memories - 1))
    return {"impression": impression, "doubts": [x[:60] for x in _str_list(obj.get("doubts"))[:3]],
            "next": _str(obj.get("next"))[:60], "evidence": ev}


def validate_focus(obj: Any) -> list[str]:
    if not isinstance(obj, dict):
        raise ValidationError("顶层必须是 JSON 对象")
    out: list[str] = []
    for q in _str_list(obj.get("questions")):
        q = q.strip()[:40]
        if len(q) >= 4 and q not in out:
            out.append(q)
    if not out:
        raise ValidationError("questions 为空：至少写出一个具体的问题")
    return out[:3]


class ReflectionLLMAdapter:
    def __init__(self, backend: LLMBackend, params: TaskParams, tracer: Any = None, repair_attempts: int = 1):
        self.backend, self.params, self.tracer, self.repair_attempts = backend, params, tracer, repair_attempts

    def focal_questions(self, npc: str, player: str, memories: list[str], transcript: str = "") -> list[str]:
        """从最近的记忆里挑出此刻最想弄明白的几个问题。失败返回 []（调用方退回“只看最近几条”）。"""
        if not memories:
            return []
        listing = "\n".join(f"- {m}" for m in memories)
        req = LLMRequest(task="reflect_focus", messages=[
            {"role": "system", "content": FOCUS_SYSTEM.format(npc=npc, player=player)},
            {"role": "user", "content": f"# 你最近的记忆\n{listing}\n\n# 刚才的谈话（节选）\n{transcript[-800:]}"}],
            temperature=self.params.temperature, max_tokens=min(200, self.params.max_tokens),
            json_schema=FOCUS_SCHEMA, schema_name="reflection_focus", meta={"npc": npc, "player_name": player})
        try:
            return call_json(self.backend, req, validate_focus, self.tracer, self.repair_attempts)
        except LLMError as e:
            if self.tracer:
                self.tracer.log("fallback", where="reflect_focus", error=str(e))
            return []

    def reflect(self, npc: str, player: str, transcript: str, previous: list[str], memories: list[str] | None = None,
                groups: list[tuple[str, list[int]]] | None = None) -> dict:
        """groups = [(问题, [它检索到的记忆在 memories 里的下标])]；没有就只按 memories 列表反思。"""
        prev = "\n".join(f"- {x}" for x in previous) or "（这是第一次）"
        mems = memories or []
        listing = "\n".join(f"{i}. {m}" for i, m in enumerate(mems)) or "（无）"
        if groups:
            asked = "\n".join(f"问题{n}：{q}\n" + "\n".join(f"  [{i}] {mems[i]}" for i in idx) for n, (q, idx) in enumerate(groups, 1))
            body = f"# 你之前琢磨出的看法\n{prev}\n\n# 想弄明白的问题，和各自相关的记忆（方括号里是编号）\n{asked}\n\n# 这次谈话\n{transcript[-1500:]}"
        else:
            body = f"# 你之前琢磨出的看法\n{prev}\n\n# 最近的记忆（编号）\n{listing}\n\n# 这次谈话\n{transcript[-2500:]}"
        req = LLMRequest(task="reflect", messages=[
            {"role": "system", "content": SYSTEM.format(npc=npc, player=player) + (GROUPED.format(player=player) if groups else "")},
            {"role": "user", "content": body}],
            temperature=self.params.temperature, max_tokens=min(400, self.params.max_tokens),
            json_schema=SCHEMA, schema_name="reflection", meta={"npc": npc, "player_name": player, "transcript": transcript[:300]})
        try:
            return call_json(self.backend, req, lambda o: validate_reflection(o, len(mems)), self.tracer, self.repair_attempts)
        except LLMError as e:
            if self.tracer:
                self.tracer.log("fallback", where="reflect", error=str(e))
            return {}
