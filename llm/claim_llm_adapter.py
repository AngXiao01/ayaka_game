"""声明抽取 agent：玩家这句话里，有没有说自己做过什么（去过哪、查过什么、和谁谈过）。

只做抽取，不判真假、不判对方知不知道——那是 core/claims.py 的事（代码对着玩家真实的足迹核对）。
"""
from __future__ import annotations

from typing import Any

from config.settings import TaskParams
from llm.base import LLMBackend, LLMError, LLMRequest
from llm.structured import call_json

SYSTEM = """你从{player}的一句话里，找出他**声称自己已经做过**的事。只抽取，不判断真假。

三类：
- visited：说自己去过 / 到过某个地方
- examined：说自己查看、调查、检查过某个地方或某样东西
- talked：说自己和某个人谈过话、问过话

规则：
1. target 只能从下面给出的编号里选（地点、物件、人物），对不上就不要记。
2. 只记「已经做过」的：计划（我打算去）、疑问（你去过吗）、假设（如果我去了）、别人做的事，都不算。
3. 一句话最多记 3 条。没有就 {{"claims": []}}。
4. text 写这一条声称的原话片段，不超过 30 字。

只输出一个 JSON 对象：{{"claims": [{{"kind": "visited", "target": "编号", "text": "……"}}]}}，不要任何其他文字。"""

SCHEMA = {"type": "object", "properties": {"claims": {"type": "array", "items": {
    "type": "object", "properties": {"kind": {"type": "string", "enum": ["visited", "examined", "talked"]},
                                      "target": {"type": "string"}, "text": {"type": "string"}},
    "required": ["kind", "target", "text"]}}}, "required": ["claims"]}


def validate_claims(obj: Any, targets: set[str]) -> list[dict]:
    items = obj.get("claims") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if isinstance(it, dict) and it.get("kind") in ("visited", "examined", "talked") and str(it.get("target")) in targets:
            out.append({"kind": it["kind"], "target": str(it["target"]), "text": str(it.get("text") or "")[:40]})
    return out[:3]


class ClaimLLMAdapter:
    def __init__(self, backend: LLMBackend, params: TaskParams, tracer: Any = None, repair_attempts: int = 1):
        self.backend, self.params, self.tracer, self.repair_attempts = backend, params, tracer, repair_attempts

    def extract(self, player: str, text: str, places: dict[str, str], objects: dict[str, str],
                people: dict[str, str]) -> list[dict]:
        listing = ("地点：" + "；".join(f"{i}={n}" for i, n in places.items()) + "\n物件：" +
                   "；".join(f"{i}={n}" for i, n in objects.items()) + "\n人物：" + "；".join(f"{i}={n}" for i, n in people.items()))
        req = LLMRequest(
            task="claims",
            messages=[{"role": "system", "content": SYSTEM.format(player=player)},
                      {"role": "user", "content": f"# 可选的编号\n{listing}\n\n# {player}说\n{text}"}],
            temperature=0.0, max_tokens=min(300, self.params.max_tokens), json_schema=SCHEMA, schema_name="claims",
            meta={"text": text, "places": places, "objects": objects, "people": people})
        ids = set(places) | set(objects) | set(people)
        try:
            return call_json(self.backend, req, lambda o: validate_claims(o, ids), self.tracer, self.repair_attempts)
        except LLMError:
            return []
