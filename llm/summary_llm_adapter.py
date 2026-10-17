"""对话结束后，以 NPC 第一人称把这次见面总结成一条长期记忆。"""
from __future__ import annotations

from typing import Any

from config.settings import TaskParams
from llm.base import LLMBackend, LLMRequest
from llm.base import LLMError
from llm.schemas import COMPRESS_SCHEMA, SUMMARY_SCHEMA, validate_compress, validate_commitments, validate_summary
from llm.structured import call_json

SYSTEM = """你是{npc}的记忆整理者。请以{npc}的第一人称，把下面这次与{player}的见面整理成一条记忆。只写「发生了什么、你的感受」，具体的事实、没聊完的话题和约定由另一位整理者单独记录，这里不用管。

字段说明：
- summary：2~3 句，像{npc}回想起这次见面时会记住的样子，包含发生了什么、对方给你的印象
- topics：谈到的话题关键词
- player_intent：对方这次来主要想做什么
- npc_feeling：你此刻对对方的感受（一个短语）
- importance：这次见面的重要程度，1~5 的整数

只输出一个 JSON 对象，不要任何其他文字，不要代码块。"""

COMMITMENTS_SYSTEM = """你是{npc}的备忘录整理者。只从下面这次与{player}的见面记录里抽出三类可以核对的内容，不写感受、不做概括：

- facts：这次见面里以后值得记住的具体事实——对方透露的，以及{npc}自己说出口的新细节（往事、喜好、身边人的事）。每条一句，必须能在对话里找到原话依据（没有就 []）
- open_threads：这次没聊完、下次见面应该接上的话题或疑问（每条一句，没有就 []）
- promises：双方明确说好的事、约定或答应对方的事（每条一句；对话里没说过的不要写，没有就 []）

只输出一个 JSON 对象，不要任何其他文字，不要代码块。"""

COMMITMENTS_SCHEMA = {"type": "object", "properties": {
    "facts": {"type": "array", "items": {"type": "string"}},
    "open_threads": {"type": "array", "items": {"type": "string"}},
    "promises": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts", "open_threads", "promises"]}


class SummaryLLMAdapter:
    def __init__(self, backend: LLMBackend, params: TaskParams, tracer: Any = None, repair_attempts: int = 2):
        self.backend = backend
        self.params = params
        self.tracer = tracer
        self.repair_attempts = repair_attempts

    def summarize(self, npc_name: str, player_name: str, transcript: str, end_reason: str, turns: int) -> dict:
        """两次调用：叙述（摘要、感受、重要度）和备忘（事实、没聊完的、约定）各由一次专门的调用整理。"""
        meta = {"player_name": player_name, "turns": turns, "transcript": transcript[:400]}
        user = f"# 对话记录\n{transcript}\n\n# 结束方式\n{end_reason}"
        req = LLMRequest(
            task="summary",
            messages=[{"role": "system", "content": SYSTEM.format(npc=npc_name, player=player_name)},
                      {"role": "user", "content": user}],
            temperature=self.params.temperature, max_tokens=self.params.max_tokens,
            json_schema=SUMMARY_SCHEMA, schema_name="memory_summary", meta=meta)
        out = call_json(self.backend, req, validate_summary, self.tracer, self.repair_attempts)
        memo = LLMRequest(
            task="commitments",
            messages=[{"role": "system", "content": COMMITMENTS_SYSTEM.format(npc=npc_name, player=player_name)},
                      {"role": "user", "content": user}],
            temperature=0.0, max_tokens=self.params.max_tokens,
            json_schema=COMMITMENTS_SCHEMA, schema_name="commitments", meta=meta)
        try:
            got = call_json(self.backend, memo, validate_commitments, self.tracer, min(1, self.repair_attempts))
            out.update(got)
        except LLMError as e:
            if self.tracer:
                self.tracer.log("fallback", where="commitments", error=str(e))
            out.update({"facts": [], "open_threads": [], "promises": []})
        return out


COMPRESS_SYSTEM = """把一段较早的对话压缩成「前情提要」，供角色在后续对话里参考。
要求：第三人称；保留双方说出的具体事实（包括角色自己随口讲的往事、喜好和身边人的小事——这些说过就算数，后面要前后一致）、双方的承诺与约定、尚未解决的问题、情绪上的转折；
丢掉寒暄与重复；与已有提要合并成一段，不超过 150 字。
只输出一个 JSON 对象：{"summary": "……"}"""


class CompressLLMAdapter:
    def __init__(self, backend: LLMBackend, params: TaskParams, tracer: Any = None, repair_attempts: int = 1):
        self.backend = backend
        self.params = params
        self.tracer = tracer
        self.repair_attempts = repair_attempts

    def compress(self, previous: str, lines: list[str]) -> str:
        req = LLMRequest(
            task="compress",
            messages=[{"role": "system", "content": COMPRESS_SYSTEM},
                      {"role": "user", "content": f"# 已有提要\n{previous or '（无）'}\n\n# 需要压缩的对话\n" + "\n".join(lines)}],
            temperature=self.params.temperature, max_tokens=self.params.max_tokens,
            json_schema=COMPRESS_SCHEMA, schema_name="compress", meta={"lines": len(lines)},
        )
        return call_json(self.backend, req, validate_compress, self.tracer, self.repair_attempts)
