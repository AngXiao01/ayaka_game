"""对话结束后，以 NPC 第一人称把这次见面总结成一条长期记忆。"""
from __future__ import annotations

from typing import Any

from config.settings import TaskParams
from llm.base import LLMBackend, LLMRequest
from llm.schemas import COMPRESS_SCHEMA, SUMMARY_SCHEMA, validate_compress, validate_summary
from llm.structured import call_json

SYSTEM = """你是{npc}的记忆整理者。请以{npc}的第一人称，把下面这次与{player}的见面整理成一条记忆。

字段说明：
- summary：2~3 句，像{npc}回想起这次见面时会记住的样子，包含发生了什么、对方给你的印象
- topics：谈到的话题关键词
- player_intent：对方这次来主要想做什么
- npc_feeling：你此刻对对方的感受（一个短语）
- facts：这次见面里以后值得记住的具体事实——对方透露的，以及你自己说出口的新细节（往事、喜好、身边人的事）（没有就 []）
- importance：这次见面的重要程度，1~5 的整数
- open_threads：这次没聊完、下次见面应该接上的话题或疑问（每条一句，没有就 []）
- promises：双方说好的事、约定或答应对方的事（每条一句，没有就 []）

只输出一个 JSON 对象，不要任何其他文字，不要代码块。"""


class SummaryLLMAdapter:
    def __init__(self, backend: LLMBackend, params: TaskParams, tracer: Any = None, repair_attempts: int = 2):
        self.backend = backend
        self.params = params
        self.tracer = tracer
        self.repair_attempts = repair_attempts

    def summarize(self, npc_name: str, player_name: str, transcript: str, end_reason: str, turns: int) -> dict:
        req = LLMRequest(
            task="summary",
            messages=[{"role": "system", "content": SYSTEM.format(npc=npc_name, player=player_name)},
                      {"role": "user", "content": f"# 对话记录\n{transcript}\n\n# 结束方式\n{end_reason}"}],
            temperature=self.params.temperature, max_tokens=self.params.max_tokens,
            json_schema=SUMMARY_SCHEMA, schema_name="memory_summary",
            meta={"player_name": player_name, "turns": turns, "transcript": transcript[:400]},
        )
        return call_json(self.backend, req, validate_summary, self.tracer, self.repair_attempts)


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
