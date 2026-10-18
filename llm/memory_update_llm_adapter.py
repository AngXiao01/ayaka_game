"""新事实 vs 已有事实：新增、更新、作废，还是不用存（Mem0 的 ADD / UPDATE / DELETE / NONE）。

以前只按字面相似度去重：≥0.6 就当重复合并。可“旅行者喜欢喝茶”和“旅行者不喜欢喝茶”字面相似度是 0.62——
矛盾被当成重复吞掉，旧的那句还保留着；相似度低一点的矛盾则新旧并存、互相打架。
这里只在“有相近旧事实”时才问一次小模型（没有就不调用，一局里多半一次都用不上）；
模型只能在给定的几条旧事实里选，它的意见能不能采纳由 NPCAgent 里的代码再检查。
"""
from __future__ import annotations

from typing import Any

from config.settings import TaskParams
from llm.base import LLMBackend, LLMError, LLMRequest
from llm.schemas import ValidationError, _str
from llm.structured import call_json

EVENTS = ("ADD", "UPDATE", "DELETE", "NONE")

SYSTEM = """你是{npc}的记忆管理员。下面是这次谈话新得到的几条事实，每条后面列着记忆里已有的、看起来相近的事实（带编号）。
对每条新事实，只看这些文字本身（不要添加它们没写的内容），决定：

- ADD：是新的，和那些旧事实不是一回事 → 新增。
- UPDATE：和某条旧事实说的是同一件事，只是补充了细节或说得更准 → id 写被更新的那条，text 写合并后的一句话。
- DELETE：和某条旧事实直接矛盾，旧的已经不对了 → id 写被作废的那条（新事实会另外存下，text 留空）。
- NONE：旧事实已经说了同样的意思 → 不用存。

只输出一个 JSON 对象：{{"decisions": [{{"i": 新事实的序号, "event": "ADD / UPDATE / DELETE / NONE", "id": "旧事实编号或空", "text": "UPDATE 时的合并句，其余留空"}}]}}。
每条新事实一个决定，不要任何其他文字。"""

SCHEMA = {"type": "object", "properties": {"decisions": {"type": "array", "items": {
    "type": "object", "properties": {"i": {"type": "integer"}, "event": {"type": "string"},
                                     "id": {"type": "string"}, "text": {"type": "string"}},
    "required": ["i", "event"]}}}, "required": ["decisions"]}


def validate_decisions(obj: Any, n_facts: int, neighbor_ids: list[set[str]]) -> dict[int, dict]:
    """返回 {新事实序号: {event, id, text}}。看不懂的、编号不在候选里的，一律当 ADD（宁可多存一条，不乱改旧记忆）。"""
    if not isinstance(obj, dict) or not isinstance(obj.get("decisions"), list):
        raise ValidationError("顶层要有 decisions 数组")
    out: dict[int, dict] = {}
    for it in obj["decisions"]:
        if not isinstance(it, dict):
            continue
        try:
            i = int(it.get("i"))
        except (TypeError, ValueError):
            continue
        if not 0 <= i < n_facts or i in out:
            continue
        event, tid, text = _str(it.get("event")).upper(), _str(it.get("id")), _str(it.get("text"))[:80]
        if event not in EVENTS:
            event = "ADD"
        if event in ("UPDATE", "DELETE") and tid not in neighbor_ids[i]:
            event, tid = "ADD", ""
        if event == "UPDATE" and not text:
            event = "ADD"
        out[i] = {"event": event, "id": tid, "text": text}
    return out


class MemoryUpdateLLMAdapter:
    def __init__(self, backend: LLMBackend, params: TaskParams, tracer: Any = None, repair_attempts: int = 1):
        self.backend, self.params, self.tracer, self.repair_attempts = backend, params, tracer, repair_attempts

    def decide(self, npc: str, facts: list[str], neighbors: list[list[tuple[str, str]]]) -> dict[int, dict]:
        """facts[i] 的相近旧事实是 neighbors[i] = [(编号, 文字)]。失败返回 {}（调用方退回原来的字面去重）。"""
        listing = "\n\n".join(f"新事实 {i}：{f}\n已有的相近事实：\n" + "\n".join(f"  [{tid}] {t}" for tid, t in neighbors[i])
                              for i, f in enumerate(facts))
        req = LLMRequest(task="memory_update", messages=[
            {"role": "system", "content": SYSTEM.format(npc=npc)}, {"role": "user", "content": listing}],
            temperature=0.0, max_tokens=min(400, self.params.max_tokens), json_schema=SCHEMA, schema_name="memory_update",
            meta={"npc": npc, "facts": facts, "neighbors": neighbors})
        ids = [{tid for tid, _ in nb} for nb in neighbors]
        try:
            return call_json(self.backend, req, lambda o: validate_decisions(o, len(facts), ids), self.tracer, self.repair_attempts)
        except LLMError as e:
            if self.tracer:
                self.tracer.log("fallback", where="memory_update", error=str(e))
            return {}
